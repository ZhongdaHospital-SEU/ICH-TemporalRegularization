"""Contracts for prespecified tuning and matched temporal ablations."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import load_config, validate_study_config
from src.ich_dynamic.models import SequencePreprocessor
from src.ich_dynamic.train_evaluate import (
    MODEL_META,
    SEQUENCE_DEFAULTS,
    XGBOOST_DEFAULTS,
    Dataset,
    _seed_epoch_override,
    _temporal_hyperparameter_cache_key,
    _selection_seed_values,
    configured_candidates,
    model_feature_contract,
)


def _contract_dataset() -> Dataset:
    dynamic_columns = {
        f"{channel}_{statistic}"
        for channel in (
            "gcs_eye",
            "gcs_verbal",
            "gcs_motor",
            "gcs_total",
            "hr",
            "rr",
            "spo2",
            "sbp",
            "dbp",
            "mbp",
            "temp",
            "urine_output",
            "fio2",
            "rass",
        )
        for statistic in ("n", "mean", "min", "max", "sd", "slope")
    }
    dynamic_columns.add("monitoring_intensity_24h")
    static_columns = [
        f"static_{index}" for index in range(210 - len(dynamic_columns))
    ]
    columns = sorted(dynamic_columns) + static_columns
    static = pd.DataFrame(
        np.arange(3 * len(columns), dtype=np.float32).reshape(3, -1),
        columns=columns,
    )
    return Dataset(
        name="contract",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "subject_key": ["a", "b", "c"],
            }
        ),
        sequences=np.ones((3, 4, 14), dtype=np.float32),
        masks=np.ones((3, 4, 14), dtype=np.float32),
        labels=np.array([0, 1, 2]),
        static_all=static,
        static_deep=static.iloc[:, :23],
    )


class HyperparameterContractTests(unittest.TestCase):
    def test_config_has_multiple_explicit_candidates(self) -> None:
        config = load_config()
        xgboost = configured_candidates(
            config,
            "xgboost",
            set(XGBOOST_DEFAULTS),
        )
        sequence = configured_candidates(
            config,
            "sequence_models",
            set(SEQUENCE_DEFAULTS),
        )
        self.assertGreaterEqual(len(xgboost), 2)
        self.assertGreaterEqual(len(sequence), 2)
        self.assertEqual(
            config["hyperparameters"]["xgboost"]["search_strategy"],
            "explicit_prespecified_grid",
        )
        self.assertEqual(
            config["hyperparameters"]["sequence_models"]["search_strategy"],
            "explicit_prespecified_grid",
        )

    def test_duplicate_configuration_sources_must_agree(self) -> None:
        config = load_config()
        copied = copy.deepcopy(config)
        copied["lambda_grid"] = [0.0, 0.01]
        with self.assertRaisesRegex(ValueError, "lambda_grid"):
            validate_study_config(copied)

        copied = copy.deepcopy(config)
        copied["hyperparameters"]["stacking"]["inner_folds"] = [4]
        with self.assertRaisesRegex(ValueError, "nested_inner_folds"):
            validate_study_config(copied)

    def test_matched_ablation_feature_counts_are_explicit(self) -> None:
        contract = model_feature_contract(_contract_dataset())
        self.assertEqual(
            contract["dggru_tr"]["static_feature_count"],
            23,
        )
        self.assertEqual(
            contract["dggru_tr_noseq"]["static_feature_count"],
            23,
        )
        self.assertFalse(contract["dggru_tr_noseq"]["sequence_enabled"])
        self.assertEqual(
            contract["dggru_tr_static"]["static_feature_count"],
            125,
        )
        self.assertTrue(contract["dggru_tr_static"]["sequence_enabled"])
        self.assertEqual(
            contract["dggru_tr_traj"]["static_feature_count"],
            1,
        )
        self.assertTrue(contract["dggru_tr_traj"]["sequence_enabled"])


class TrajectoryMaskTests(unittest.TestCase):
    def test_sequences_are_causally_forward_filled_without_backfill(self) -> None:
        sequences = np.full((1, 4, 2), np.nan, dtype=np.float32)
        masks = np.zeros((1, 4, 2), dtype=np.float32)
        sequences[0, 1, 0] = 5.0
        sequences[0, 3, 0] = 9.0
        masks[0, 1, 0] = 1.0
        masks[0, 3, 0] = 1.0
        sequences[0, :, 1] = 1.0
        masks[0, :, 1] = 1.0
        preprocessor = SequencePreprocessor.fit(
            sequences,
            masks,
            np.zeros((1, 1), dtype=np.float32),
        )
        transformed, _ = preprocessor.transform_sequence(sequences, masks)

        # Channel 0 is standardized before filling: values [NaN, -1, NaN, 1]
        # become [0, -1, -1, 1]. Hour 0 is not backfilled from hour 1.
        np.testing.assert_allclose(
            transformed[0, :, 0],
            np.array([0.0, -1.0, -1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(transformed[0, :, 2], masks[0, :, 0])
        np.testing.assert_array_equal(transformed[0, :, 3], masks[0, :, 1])

    def test_gcs_and_none_mask_modes_are_distinct(self) -> None:
        sequences = np.zeros((1, 3, 14), dtype=np.float32)
        masks = np.zeros((1, 3, 14), dtype=np.float32)
        masks[0, 0, 3] = 1.0
        masks[0, 2, 4] = 1.0
        preprocessor = SequencePreprocessor.fit(
            sequences,
            masks,
            np.zeros((1, 1), dtype=np.float32),
        )
        _, gcs_mask = preprocessor.transform_sequence(
            sequences,
            masks,
            mask_mode="gcs",
        )
        _, no_mask = preprocessor.transform_sequence(
            sequences,
            masks,
            mask_mode="none",
        )
        np.testing.assert_array_equal(gcs_mask, [[1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(no_mask, np.zeros_like(no_mask))


class HyperparameterIndependenceTests(unittest.TestCase):
    def test_selection_helper_retains_final_ensemble_seed_list(self) -> None:
        self.assertEqual(
            _selection_seed_values(7, (7, 1007, 2007)),
            (7, 1007, 2007),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            _selection_seed_values(7, (7, 7))

    def test_xgboost_selection_uses_one_fixed_probe_seed(self) -> None:
        n_rows = 90
        generator = np.random.default_rng(23)
        train_x = generator.normal(size=(n_rows, 4)).astype(np.float32)
        labels = np.tile(np.arange(3, dtype=int), n_rows // 3)
        groups = np.array(
            [f"patient-{index // 2}" for index in range(n_rows)],
            dtype=object,
        )
        candidates = tuple(
            {
                **copy.deepcopy(XGBOOST_DEFAULTS),
                "n_estimators": n_estimators,
            }
            for n_estimators in (100, 200, 300)
        )
        observed_seeds: list[int] = []

        def witness(
            candidate_train_x,
            candidate_train_y,
            candidate_valid_x,
            selection_seed,
            sample_weight,
            *,
            parameters,
        ):
            del candidate_train_x, candidate_train_y, sample_weight, parameters
            observed_seeds.append(int(selection_seed))
            return np.tile(
                np.array([[0.6, 0.3, 0.1]], dtype=float),
                (len(candidate_valid_x), 1),
            )

        with mock.patch.object(te, "fit_xgboost", side_effect=witness):
            te.select_xgboost_parameters(
                train_x,
                labels,
                groups,
                np.ones(n_rows, dtype=np.float64),
                seed=7,
                candidates=candidates,
                selection_seeds=(7, 1007, 2007),
            )

        self.assertEqual(observed_seeds, [7, 7, 7])

    def test_member_fitting_passes_every_selection_seed(self) -> None:
        dataset = _contract_dataset()
        probabilities = np.tile(
            np.array([[0.8, 0.15, 0.05]], dtype=float),
            (1, 1),
        )
        calls: list[dict] = []

        def witness(*args, **kwargs):
            calls.append(dict(kwargs))
            return probabilities, 4, 0.1

        with mock.patch.object(te, "fit_temporal", side_effect=witness):
            fitted, epochs, lambdas = te._fit_member_predictions(
                dataset,
                np.array([0, 1]),
                np.array([2]),
                name="dggru_tr",
                fold=0,
                seed=0,
                epochs=25,
                lambda_monotone=0.1,
                lambda_smooth=0.04,
                lambda_grid=(),
                lambda_smooth_ratio=0.4,
                model_seeds=(0, 1, 2),
            )

        self.assertEqual(len(calls), 3)
        for offset, call in enumerate(calls):
            self.assertEqual(call["selection_seeds"], (100, 1100, 2100))
            self.assertEqual(call["selection_offset"], offset)
        self.assertEqual(fitted.shape, (1, 3))
        self.assertEqual(epochs, [4, 4, 4])
        self.assertEqual(lambdas, [0.1, 0.1, 0.1])

    def test_matched_arm_epoch_schedule_is_seed_aligned(self) -> None:
        schedule = (7, 11, 9, 13, 8)
        self.assertEqual(
            tuple(
                _seed_epoch_override(schedule, offset, len(schedule))
                for offset in range(len(schedule))
            ),
            schedule,
        )
        self.assertEqual(_seed_epoch_override(12, 3, 5), 12)
        with self.assertRaises(ValueError):
            _seed_epoch_override((7,), 0, 5)

    def test_temporal_cache_shares_matched_arms_but_separates_input_contracts(
        self,
    ) -> None:
        matched_names = (
            "dggru",
            "dggru_tr",
            "dggru_tr_treverse",
            "dggru_tr_tpermute",
            "dggru_laplace",
        )
        matched_keys = {
            _temporal_hyperparameter_cache_key(
                MODEL_META[name],
                model_name=name,
                mask_mode="any_channel",
                static_count=23,
                channel_count=14,
            )
            for name in matched_names
        }
        self.assertEqual(len(matched_keys), 1)

        distinct: dict[str, str] = {}
        for name in (
            "gru",
            "dggru_tr_gcsmask",
            "dggru_tr_noseq",
            "dggru_tr_static",
            "dggru_tr_traj",
        ):
            metadata = MODEL_META[name]
            distinct[name] = _temporal_hyperparameter_cache_key(
                metadata,
                model_name=name,
                mask_mode=str(metadata.get("mask_mode", "any_channel")),
                static_count=(
                    1
                    if name == "dggru_tr_traj"
                    else 125
                    if name == "dggru_tr_static"
                    else 23
                ),
                channel_count=14,
            )
        self.assertEqual(len(set(distinct.values())), len(distinct))
        self.assertNotIn(matched_keys.pop(), set(distinct.values()))

    def test_xgboost_selection_imputes_within_inner_training_only(
        self,
    ) -> None:
        n_rows = 90
        generator = np.random.default_rng(17)
        frame = pd.DataFrame(
            generator.normal(size=(n_rows, 4)),
            columns=[f"x{index}" for index in range(4)],
        )
        frame.iloc[::11, 0] = np.nan
        labels = np.tile(np.arange(3, dtype=int), n_rows // 3)
        groups = np.array(
            [f"patient-{index // 2}" for index in range(n_rows)],
            dtype=object,
        )
        observed_shapes: list[tuple[int, int]] = []
        original_impute = te.impute_tabular

        def witness(
            train_frame: pd.DataFrame,
            validation_frame: pd.DataFrame,
        ) -> tuple[np.ndarray, np.ndarray]:
            observed_shapes.append((len(train_frame), len(validation_frame)))
            return original_impute(train_frame, validation_frame)

        with mock.patch.object(te, "impute_tabular", side_effect=witness):
            parameters, score, candidate_index = te.select_xgboost_parameters(
                np.zeros((n_rows, 4), dtype=np.float32),
                labels,
                groups,
                np.ones(n_rows, dtype=np.float64),
                seed=7,
                candidates=(copy.deepcopy(XGBOOST_DEFAULTS),),
                raw_train_frame=frame,
            )

        self.assertTrue(observed_shapes)
        self.assertTrue(
            all(train_n < n_rows for train_n, _ in observed_shapes),
            "Hyperparameter selection used the complete outer training fold",
        )
        self.assertEqual(candidate_index, 0)
        self.assertEqual(parameters, XGBOOST_DEFAULTS)
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
