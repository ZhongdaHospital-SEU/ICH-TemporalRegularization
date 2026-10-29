from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.train_evaluate import (
    Dataset,
    SEQUENCE_DEFAULTS,
    XGBOOST_DEFAULTS,
    nested_stacking_predictions,
    predict_stacking_meta_model,
    stacking_meta_features,
    train_full_and_external,
)


def make_stacking_dataset() -> Dataset:
    n_rows = 30
    labels = np.tile(np.arange(3, dtype=int), n_rows // 3)
    stay_id = np.arange(1000, 1000 + n_rows)
    subject_key = [f"patient-{index // 2}" for index in range(n_rows)]
    return Dataset(
        name="unit",
        cohort=pd.DataFrame(
            {
                "stay_id": stay_id,
                "subject_key": subject_key,
            }
        ),
        sequences=np.zeros((n_rows, 2, 2), dtype=np.float32),
        masks=np.ones((n_rows, 2, 2), dtype=np.float32),
        labels=labels,
        static_all=pd.DataFrame(
            {"x": np.arange(n_rows, dtype=float)}
        ),
        static_deep=pd.DataFrame(
            {"x": np.arange(n_rows, dtype=float)}
        ),
    )


class StackingFeatureTests(unittest.TestCase):
    def test_meta_features_follow_fixed_member_and_class_order(self) -> None:
        first = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
            ]
        )
        second = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.4, 0.3],
            ]
        )
        features = stacking_meta_features(
            {"second": second, "first": first},
            members=("first", "second"),
        )
        np.testing.assert_allclose(features[:, :3], first)
        np.testing.assert_allclose(features[:, 3:], second)

    def test_meta_features_reject_misaligned_members(self) -> None:
        with self.assertRaisesRegex(ValueError, "different row counts"):
            stacking_meta_features(
                {
                    "first": np.full((2, 3), 1.0 / 3.0),
                    "second": np.full((3, 3), 1.0 / 3.0),
                },
                members=("first", "second"),
            )


class NestedStackingTests(unittest.TestCase):
    def test_inner_oof_is_patient_grouped_and_meta_fits_before_test_refit(
        self,
    ) -> None:
        dataset = make_stacking_dataset()
        train_index = np.arange(24, dtype=int)
        test_index = np.arange(24, 30, dtype=int)
        timeline: list[tuple[str, str, int, int]] = []
        member_calls: list[tuple[str, np.ndarray, np.ndarray]] = []
        meta_fits: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
        inner_cache_values: list[object] = []
        outer_cache_values: list[object] = []
        shared_cache: dict[str, dict] = {}

        def fake_member(
            current_dataset: Dataset,
            member_train: np.ndarray,
            member_test: np.ndarray,
            *,
            name: str,
            hyperparameter_cache: dict[str, dict] | None = None,
            **_: object,
        ) -> tuple[np.ndarray, list[int], list[float]]:
            member_train = np.asarray(member_train, dtype=int)
            member_test = np.asarray(member_test, dtype=int)
            self.assertFalse(
                set(member_train).intersection(member_test),
                "Member train and validation indices overlap",
            )
            timeline.append(
                ("member", name, len(member_train), len(member_test))
            )
            member_calls.append(
                (name, member_train.copy(), member_test.copy())
            )
            if len(member_train) == len(train_index):
                outer_cache_values.append(hyperparameter_cache)
            else:
                inner_cache_values.append(hyperparameter_cache)
            counts = np.bincount(
                current_dataset.labels[member_train],
                minlength=3,
            ).astype(float)
            probabilities = np.tile((counts + 1.0) / (counts.sum() + 3.0), (
                len(member_test),
                1,
            ))
            return probabilities, [4], [0.1]

        original_fit = te.fit_stacking_meta_model

        def wrapped_fit(
            oof: dict[str, np.ndarray],
            labels: np.ndarray,
            **kwargs: object,
        ):
            members = tuple(kwargs["members"])
            features = stacking_meta_features(oof, members=members)
            timeline.append(
                ("meta", "logistic", len(labels), features.shape[1])
            )
            meta_fits.append(
                (tuple(oof), features.copy(), np.asarray(labels).copy())
            )
            return original_fit(oof, labels, **kwargs)

        with (
            mock.patch.object(
                te,
                "_fit_member_predictions",
                side_effect=fake_member,
            ),
            mock.patch.object(
                te,
                "fit_stacking_meta_model",
                side_effect=wrapped_fit,
            ),
        ):
            (
                stacked,
                oof,
                meta_model,
                inner_splits,
                member_test_probabilities,
                member_epochs,
                member_lambdas,
            ) = nested_stacking_predictions(
                dataset,
                train_index,
                test_index,
                seed=71,
                fold=2,
                epochs=5,
                lambda_monotone=0.05,
                lambda_smooth=0.02,
                model_seeds=(0, 1),
                members=("xgb_summary", "gru", "dggru_tr"),
                inner_folds=5,
                stacking_c=0.1,
                hyperparameter_cache=shared_cache,
            )

        self.assertEqual(len(inner_splits), 5)
        self.assertEqual(len(inner_cache_values), 15)
        self.assertTrue(
            all(value is None for value in inner_cache_values),
            "Inner OOF folds must independently select hyperparameters",
        )
        self.assertEqual(len(outer_cache_values), 3)
        self.assertTrue(
            all(value is shared_cache for value in outer_cache_values),
            "Final outer members should share only their model-specific cache",
        )
        self.assertEqual(len(meta_fits), 1)
        self.assertEqual(meta_fits[0][1].shape, (24, 9))
        np.testing.assert_array_equal(meta_fits[0][2], dataset.labels[train_index])
        self.assertEqual(set(meta_fits[0][0]), {
            "xgb_summary",
            "gru",
            "dggru_tr",
        })
        meta_position = next(
            index for index, event in enumerate(timeline) if event[0] == "meta"
        )
        final_member_positions = [
            index
            for index, event in enumerate(timeline)
            if event[0] == "member"
            and event[2] == len(train_index)
            and event[3] == len(test_index)
        ]
        self.assertEqual(len(final_member_positions), 3)
        self.assertLess(meta_position, min(final_member_positions))

        inner_calls = [
            event
            for event in timeline
            if event[0] == "member"
            and not (
                event[2] == len(train_index)
                and event[3] == len(test_index)
            )
        ]
        self.assertEqual(len(inner_calls), 15)
        outer_train_set = set(train_index.tolist())
        outer_test_set = set(test_index.tolist())
        for _, member_train, member_test in member_calls:
            if (
                len(member_train) == len(train_index)
                and len(member_test) == len(test_index)
            ):
                np.testing.assert_array_equal(member_train, train_index)
                np.testing.assert_array_equal(member_test, test_index)
                continue
            self.assertTrue(set(member_train.tolist()) <= outer_train_set)
            self.assertTrue(set(member_test.tolist()) <= outer_train_set)
            self.assertFalse(set(member_train.tolist()).intersection(outer_test_set))
            self.assertFalse(set(member_test.tolist()).intersection(outer_test_set))
        for member in ("xgb_summary", "gru", "dggru_tr"):
            self.assertEqual(
                sum(event[1] == member for event in inner_calls),
                5,
            )
            values = oof[member]
            self.assertTrue(np.isfinite(values).all())
            np.testing.assert_allclose(values.sum(axis=1), 1.0)
        for current_train, current_test in inner_splits:
            train_groups = set(
                dataset.cohort["subject_key"].iloc[
                    train_index[current_train]
                ]
            )
            test_groups = set(
                dataset.cohort["subject_key"].iloc[
                    train_index[current_test]
                ]
            )
            self.assertFalse(train_groups.intersection(test_groups))

        expected = predict_stacking_meta_model(
            meta_model,
            member_test_probabilities,
            members=("xgb_summary", "gru", "dggru_tr"),
        )
        np.testing.assert_allclose(stacked, expected)
        np.testing.assert_allclose(stacked.sum(axis=1), 1.0)
        self.assertEqual(
            set(member_test_probabilities),
            {"xgb_summary", "gru", "dggru_tr"},
        )
        self.assertEqual(set(member_epochs), set(member_test_probabilities))
        self.assertEqual(set(member_lambdas), set(member_test_probabilities))

    def test_train_fold_reuses_member_predictions_from_nested_run(self) -> None:
        dataset = make_stacking_dataset()
        train_index = np.arange(24, dtype=int)
        test_index = np.arange(24, 30, dtype=int)
        members = ("xgb_summary", "gru", "dggru_tr")
        member_probabilities = {
            member: np.full((6, 3), (index + 1) / 6.0)
            for index, member in enumerate(members)
        }
        member_probabilities = {
            member: values / values.sum(axis=1, keepdims=True)
            for member, values in member_probabilities.items()
        }
        stacked = np.full((6, 3), 1.0 / 3.0)

        with (
            mock.patch.object(
                te,
                "_fit_member_predictions",
                side_effect=AssertionError("base members must not be fit twice"),
            ),
            mock.patch.object(
                te,
                "nested_stacking_predictions",
                return_value=(
                    stacked,
                    {},
                    None,
                    [],
                    member_probabilities,
                    {member: [4] for member in members},
                    {member: [0.1] for member in members},
                ),
            ),
        ):
            _, predictions, epochs_used, lambdas_used, _ = te._train_fold(
                dataset,
                fold=0,
                train_index=train_index,
                test_index=test_index,
                epochs=5,
                seed=71,
                models=("xgb_summary", "gru", "dggru_tr", "nested_stack"),
                lambda_monotone=0.05,
                lambda_smooth=0.02,
                lambda_grid=(),
                lambda_smooth_ratio=0.4,
                model_seeds=(0,),
            )

        np.testing.assert_allclose(
            predictions["nested_stack"][test_index],
            stacked,
        )
        for member in members:
            np.testing.assert_allclose(
                predictions[member][test_index],
                member_probabilities[member],
            )
            self.assertEqual(epochs_used[member], [4])
            self.assertEqual(lambdas_used[member], [0.1])

    def test_fixed_transport_settings_reach_oof_and_final_members(self) -> None:
        dataset = make_stacking_dataset()
        train_index = np.arange(24, dtype=int)
        test_index = np.arange(24, 30, dtype=int)
        members = ("xgb_summary", "gru", "dggru_tr")
        calls: list[tuple[str, bool, dict | None, float | None, float | None, int | None]] = []
        hyperparameter_overrides = {
            member: {"member": member} for member in members
        }
        lambda_overrides = {member: 0.03 for member in members}
        smooth_overrides = {member: 0.012 for member in members}
        epoch_overrides = {member: 6 for member in members}

        def fake_member(
            current_dataset: Dataset,
            member_train: np.ndarray,
            member_test: np.ndarray,
            *,
            name: str,
            hyperparameter_override: dict | None = None,
            lambda_override: float | None = None,
            smooth_override: float | None = None,
            epoch_override: int | None = None,
            **_: object,
        ) -> tuple[np.ndarray, list[int], list[float]]:
            del current_dataset
            is_final = (
                len(member_train) == len(train_index)
                and len(member_test) == len(test_index)
            )
            calls.append(
                (
                    name,
                    is_final,
                    hyperparameter_override,
                    lambda_override,
                    smooth_override,
                    epoch_override,
                )
            )
            probabilities = np.full((len(member_test), 3), 1.0 / 3.0)
            return probabilities, [epoch_override or 4], [
                lambda_override or 0.1
            ]

        with mock.patch.object(
            te,
            "_fit_member_predictions",
            side_effect=fake_member,
        ):
            nested_stacking_predictions(
                dataset,
                train_index,
                test_index,
                seed=71,
                fold=2,
                epochs=5,
                lambda_monotone=0.05,
                lambda_smooth=0.02,
                model_seeds=(0,),
                members=members,
                inner_folds=2,
                stacking_c=0.1,
                hyperparameter_overrides=hyperparameter_overrides,
                lambda_overrides=lambda_overrides,
                smooth_overrides=smooth_overrides,
                epoch_overrides=epoch_overrides,
            )

        self.assertEqual(len(calls), len(members) * 3)
        self.assertEqual(
            sum(is_final for _, is_final, *_ in calls),
            len(members),
        )
        for member, _, hyperparameters, selected_lambda, selected_smooth, epochs in calls:
            self.assertEqual(hyperparameters, hyperparameter_overrides[member])
            self.assertEqual(selected_lambda, lambda_overrides[member])
            self.assertEqual(selected_smooth, smooth_overrides[member])
            self.assertEqual(epochs, epoch_overrides[member])

    def test_negative_controls_inherit_nested_primary_lambda(self) -> None:
        dataset = make_stacking_dataset()
        train_index = np.arange(24, dtype=int)
        test_index = np.arange(24, 30, dtype=int)
        members = ("xgb_summary", "gru", "dggru_tr")
        member_probabilities = {
            member: np.full((len(test_index), 3), 1.0 / 3.0)
            for member in members
        }
        observed_lambdas: dict[str, float] = {}

        def fake_member(
            current_dataset: Dataset,
            member_train: np.ndarray,
            member_test: np.ndarray,
            *,
            name: str,
            lambda_monotone: float,
            **_: object,
        ) -> tuple[np.ndarray, list[int], list[float]]:
            del current_dataset, member_train
            observed_lambdas[name] = float(lambda_monotone)
            values = np.full((len(member_test), 3), 1.0 / 3.0)
            return values, [5], [float(lambda_monotone)]

        with (
            mock.patch.object(
                te,
                "nested_stacking_predictions",
                return_value=(
                    np.full((len(test_index), 3), 1.0 / 3.0),
                    {},
                    None,
                    [],
                    member_probabilities,
                    {member: [5] for member in members},
                    {
                        "xgb_summary": [0.02],
                        "gru": [0.02],
                        "dggru_tr": [0.3],
                    },
                ),
            ),
            mock.patch.object(
                te,
                "_fit_member_predictions",
                side_effect=fake_member,
            ),
        ):
            te._train_fold(
                dataset,
                fold=0,
                train_index=train_index,
                test_index=test_index,
                epochs=5,
                seed=71,
                models=(
                    "dggru_tr",
                    "nested_stack",
                    "dggru_tr_treverse",
                    "dggru_tr_tpermute",
                    "dggru_laplace",
                ),
                lambda_monotone=0.05,
                lambda_smooth=0.02,
                lambda_grid=(0.05, 0.3),
                lambda_smooth_ratio=0.4,
                model_seeds=(0,),
            )

        self.assertEqual(
            observed_lambdas,
            {
                "dggru_tr_treverse": 0.3,
                "dggru_tr_tpermute": 0.3,
                "dggru_laplace": 0.3,
            },
        )

    def test_external_meta_features_reuse_transport_member_settings(
        self,
    ) -> None:
        development = make_stacking_dataset()
        external = make_stacking_dataset()
        external.name = "external"
        external.cohort["stay_id"] = external.cohort["stay_id"] + 10_000
        external.cohort["subject_key"] = (
            external.cohort["subject_key"] + "-external"
        )
        members = ("xgb_summary", "gru", "dggru_tr")
        observed_oof_settings: dict[str, object] = {}
        xgb_parameters = {
            **XGBOOST_DEFAULTS,
            "n_estimators": 111,
        }
        gru_parameters = {
            **SEQUENCE_DEFAULTS,
            "hidden_size": 48,
        }
        dggru_parameters = {
            **SEQUENCE_DEFAULTS,
            "hidden_size": 96,
        }
        selected_hyperparameters = {
            "xgb_summary": [
                {"family": "xgboost", "parameters": xgb_parameters}
            ],
            "gru": [
                {"family": "sequence_models", "parameters": gru_parameters}
            ],
            "dggru_tr": [
                {
                    "family": "sequence_models",
                    "parameters": dggru_parameters,
                }
            ],
        }

        def fake_oof(
            current_dataset: Dataset,
            train_index: np.ndarray,
            **kwargs: object,
        ) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
            del current_dataset, train_index
            observed_oof_settings.update(kwargs)
            probabilities = {
                member: np.full((len(development.labels), 3), 1.0 / 3.0)
                for member in members
            }
            return probabilities, []

        def fake_transport(*_: object, **__: object) -> np.ndarray:
            return np.full((len(external.labels), 3), 1.0 / 3.0)

        with (
            mock.patch.object(
                te,
                "_fit_transport_member",
                side_effect=fake_transport,
            ),
            mock.patch.object(
                te,
                "_stacking_oof_probabilities",
                side_effect=fake_oof,
            ),
            mock.patch.object(
                te,
                "fit_stacking_meta_model",
                return_value=object(),
            ),
            mock.patch.object(
                te,
                "predict_stacking_meta_model",
                return_value=np.full((len(external.labels), 3), 1.0 / 3.0),
            ),
            mock.patch.object(
                te,
                "evaluate_predictions",
                return_value={},
            ),
        ):
            train_full_and_external(
                development,
                {"external": external},
                models=("nested_stack",),
                epochs=8,
                seed=100,
                lambda_monotone=0.05,
                lambda_smooth=0.02,
                cv_best_epochs={
                    "gru": [4, 6],
                    "dggru_tr": [5, 7],
                },
                cv_selected_lambdas={
                    "xgb_summary": [],
                    "gru": [0.02, 0.02],
                    "dggru_tr": [0.03, 0.05],
                },
                lambda_smooth_ratio=0.4,
                model_seeds=(0,),
                metric_workers=1,
                replicates=0,
                stacking_members=members,
                cv_selected_hyperparameters=selected_hyperparameters,
            )

        overrides = observed_oof_settings["hyperparameter_overrides"]
        self.assertEqual(overrides["xgb_summary"], xgb_parameters)
        self.assertEqual(overrides["gru"], gru_parameters)
        self.assertEqual(overrides["dggru_tr"], dggru_parameters)
        self.assertAlmostEqual(
            observed_oof_settings["lambda_overrides"]["dggru_tr"],
            0.04,
        )
        self.assertAlmostEqual(
            observed_oof_settings["smooth_overrides"]["dggru_tr"],
            0.016,
        )
        self.assertEqual(
            observed_oof_settings["epoch_overrides"]["gru"],
            5,
        )
        self.assertEqual(
            observed_oof_settings["epoch_overrides"]["dggru_tr"],
            6,
        )


if __name__ == "__main__":
    unittest.main()
