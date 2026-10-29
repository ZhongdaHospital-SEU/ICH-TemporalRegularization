"""Contracts for IPW-weighted fitting, evaluation, and model comparisons."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ich_dynamic.models import SequenceDataset, weighted_nll_loss
from src.ich_dynamic.train_evaluate import (
    Dataset,
    MODEL_META,
    aggregate_point_metrics,
    binary_expected_calibration_error,
    calibration_slope_intercept,
    categorical_nri,
    comparison_metrics,
    comparison_metrics_repeated,
    continuous_nri,
    cumulative_auc,
    decision_curve_net_benefit,
    evaluate_predictions,
    evaluate_repeated_cv,
    expected_calibration_error,
    fit_slope_logistic,
    fit_xgboost,
    group_bootstrap,
    group_bootstrap_repeated,
    integrated_discrimination_improvement,
    multiclass_auc,
    paired_auc_difference_repeated,
    point_metrics,
    tabular_frame,
)


class WeightedLossTests(unittest.TestCase):
    def test_zero_weight_patient_does_not_contribute_to_nll(self) -> None:
        probabilities = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 2])
        weights = torch.tensor([1.0, 0.0], dtype=torch.float64)
        loss = weighted_nll_loss(probabilities, labels, weights)
        expected = -torch.log(torch.tensor(0.8, dtype=torch.float64))
        self.assertTrue(torch.allclose(loss, expected))

    def test_sequence_dataset_returns_patient_aligned_weights(self) -> None:
        dataset = SequenceDataset(
            np.zeros((3, 2, 1), dtype=np.float32),
            np.ones((3, 2, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.array([0, 1, 2]),
            sample_weight=np.array([0.5, 1.0, 2.0]),
        )
        self.assertEqual(dataset[2][-1].item(), 2.0)


class BaselineContractTests(unittest.TestCase):
    def test_static_xgboost_uses_all_static_covariates(self) -> None:
        n = 4
        static = pd.DataFrame(
            {
                "age": np.linspace(40, 80, n),
                "mechanical_ventilation_24h": [0, 0, 1, 1],
                "lab_sodium_first": [138, 139, 140, 141],
                "gcs_total_mean": [12, 13, 14, 15],
                "gcs_total_slope": [0, 0, 1, -1],
                "monitoring_intensity_24h": [0.2, 0.3, 0.4, 0.5],
            }
        )
        dataset = Dataset(
            name="static_contract",
            cohort=pd.DataFrame(
                {
                    "stay_id": np.arange(1, n + 1),
                    "subject_key": [f"p{index}" for index in range(n)],
                }
            ),
            sequences=np.zeros((n, 2, 2), dtype=np.float32),
            masks=np.ones((n, 2, 2), dtype=np.float32),
            labels=np.array([0, 1, 1, 2]),
            static_all=static,
            static_deep=static[["age"]],
            analysis_weight=np.ones(n),
        )
        columns = set(tabular_frame(dataset, "static").columns)
        self.assertIn("age", columns)
        self.assertIn("mechanical_ventilation_24h", columns)
        self.assertIn("lab_sodium_first", columns)
        self.assertNotIn("gcs_total_mean", columns)
        self.assertNotIn("gcs_total_slope", columns)
        self.assertNotIn("monitoring_intensity_24h", columns)

    def test_gcs_slope_is_explicit_in_model_registry(self) -> None:
        self.assertEqual(MODEL_META["gcs_slope"]["kind"], "slope_logistic")
        self.assertEqual(MODEL_META["gcs_slope"]["features"], "gcs_slope")

    def test_gcs_slope_logistic_uses_one_feature_and_all_classes(self) -> None:
        train_x = np.array([[-1.0], [-0.2], [0.4], [0.8], [1.2], [1.8]])
        train_y = np.array([0, 0, 1, 1, 2, 2])
        test_x = np.array([[-0.5], [0.5], [1.5]])
        probabilities = fit_slope_logistic(
            train_x,
            train_y,
            test_x,
            seed=7,
            sample_weight=np.ones(len(train_y)),
        )
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_xgboost_accepts_patient_weights(self) -> None:
        generator = np.random.default_rng(11)
        train_x = generator.normal(size=(30, 4))
        train_y = np.tile(np.arange(3), 10)
        test_x = generator.normal(size=(6, 4))
        probabilities = fit_xgboost(
            train_x,
            train_y,
            test_x,
            seed=3,
            sample_weight=np.linspace(0.5, 2.0, len(train_y)),
        )
        self.assertEqual(probabilities.shape, (6, 3))
        self.assertTrue(np.isfinite(probabilities).all())


def _assert_metric_tree_equal(
    test_case: unittest.TestCase,
    actual: object,
    expected: object,
) -> None:
    if isinstance(expected, dict):
        test_case.assertIsInstance(actual, dict)
        test_case.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_metric_tree_equal(test_case, actual[key], expected[key])
        return
    if isinstance(expected, list):
        np.testing.assert_allclose(actual, expected)
        return
    if isinstance(expected, str):
        test_case.assertEqual(actual, expected)
        return
    if isinstance(expected, float) and np.isnan(expected):
        test_case.assertTrue(np.isnan(actual))
        return
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _metric_fixture() -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.82, 0.14, 0.04],
            [0.61, 0.29, 0.10],
            [0.18, 0.70, 0.12],
            [0.11, 0.63, 0.26],
            [0.04, 0.18, 0.78],
            [0.12, 0.30, 0.58],
        ],
        dtype=float,
    )
    return labels, probabilities


def _weighted_dataset(
    weights: np.ndarray | None,
    *,
    include_zero_patient: bool = False,
) -> Dataset:
    labels, probabilities = _metric_fixture()
    if include_zero_patient:
        labels = np.append(labels, 2)
        weights = np.append(
            np.ones(len(labels) - 1) if weights is None else weights,
            0.0,
        )
    n = len(labels)
    return Dataset(
        name="weighted_metrics",
        cohort=pd.DataFrame(
            {
                "stay_id": np.arange(1, n + 1),
                "subject_key": [f"p{index}" for index in range(n)],
            }
        ),
        sequences=np.zeros((n, 2, 2), dtype=np.float32),
        masks=np.ones((n, 2, 2), dtype=np.float32),
        labels=labels,
        static_all=pd.DataFrame({"x": np.arange(n)}),
        static_deep=pd.DataFrame({"x": np.arange(n)}),
        analysis_weight=weights,
    )


class WeightedEvaluationTests(unittest.TestCase):
    def test_weighted_comparison_does_not_report_unweighted_delong_p(self) -> None:
        labels, probabilities = _metric_fixture()
        comparator = probabilities * 0.8 + 1.0 / 3.0 * 0.2
        weights = np.array([0.5, 2.0, 0.8, 1.5, 0.7, 1.2])
        result = comparison_metrics(
            labels,
            probabilities,
            comparator,
            sample_weight=weights,
        )
        self.assertTrue(np.isnan(result["delong_any_p"]))
        self.assertTrue(np.isnan(result["delong_severe_p"]))
        self.assertEqual(
            result["delong_inference_status"],
            "not_applicable_for_weighted_analysis_use_group_bootstrap",
        )
        self.assertTrue(np.isfinite(result["delta_macro_auc"]))

    def test_zero_weight_patients_do_not_change_metrics_or_bootstrap_ci(self) -> None:
        labels, probabilities = _metric_fixture()
        groups = np.array([f"p{index}" for index in range(len(labels))])
        weights = np.ones(len(labels))
        extended_labels = np.append(labels, 2)
        extended_probabilities = np.vstack(
            [probabilities, np.array([[0.99, 0.005, 0.005]])]
        )
        extended_groups = np.append(groups, "zero_weight_patient")
        extended_weights = np.append(weights, 0.0)

        unweighted = point_metrics(labels, probabilities)
        with_zero = point_metrics(
            extended_labels,
            extended_probabilities,
            sample_weight=extended_weights,
        )
        _assert_metric_tree_equal(self, with_zero, unweighted)

        unweighted_ci = group_bootstrap(
            labels, probabilities, groups, replicates=40, seed=17
        )
        with_zero_ci = group_bootstrap(
            extended_labels,
            extended_probabilities,
            extended_groups,
            replicates=40,
            seed=17,
            sample_weight=extended_weights,
        )
        _assert_metric_tree_equal(self, with_zero_ci, unweighted_ci)

    def test_all_one_weights_match_every_legacy_metric_path(self) -> None:
        labels, probabilities = _metric_fixture()
        groups = np.array([f"p{index}" for index in range(len(labels))])
        weights = np.ones(len(labels))
        severe_labels = (labels >= 2).astype(int)
        severe_probability = probabilities[:, 2]
        old_probability = np.clip(severe_probability - 0.03, 0.0, 1.0)

        self.assertEqual(
            multiclass_auc(labels, probabilities),
            multiclass_auc(labels, probabilities, weights),
        )
        self.assertEqual(
            cumulative_auc(labels, probabilities, 1),
            cumulative_auc(labels, probabilities, 1, weights),
        )
        self.assertEqual(
            calibration_slope_intercept(
                labels, probabilities, threshold=2
            ),
            calibration_slope_intercept(
                labels, probabilities, threshold=2, sample_weight=weights
            ),
        )
        self.assertEqual(
            expected_calibration_error(labels, probabilities),
            expected_calibration_error(
                labels, probabilities, sample_weight=weights
            ),
        )
        self.assertEqual(
            binary_expected_calibration_error(
                severe_labels, severe_probability
            ),
            binary_expected_calibration_error(
                severe_labels,
                severe_probability,
                sample_weight=weights,
            ),
        )
        self.assertEqual(
            decision_curve_net_benefit(
                severe_labels, severe_probability
            ),
            decision_curve_net_benefit(
                severe_labels,
                severe_probability,
                sample_weight=weights,
            ),
        )
        self.assertEqual(
            continuous_nri(
                severe_labels, severe_probability, old_probability
            ),
            continuous_nri(
                severe_labels,
                severe_probability,
                old_probability,
                weights,
            ),
        )
        self.assertEqual(
            categorical_nri(
                severe_labels, severe_probability, old_probability
            ),
            categorical_nri(
                severe_labels,
                severe_probability,
                old_probability,
                sample_weight=weights,
            ),
        )
        self.assertEqual(
            integrated_discrimination_improvement(
                severe_labels, severe_probability, old_probability
            ),
            integrated_discrimination_improvement(
                severe_labels,
                severe_probability,
                old_probability,
                weights,
            ),
        )
        _assert_metric_tree_equal(
            self,
            point_metrics(labels, probabilities, sample_weight=weights),
            point_metrics(labels, probabilities),
        )
        _assert_metric_tree_equal(
            self,
            comparison_metrics(
                labels,
                probabilities,
                probabilities * 0.8 + 1.0 / 3.0 * 0.2,
                sample_weight=weights,
            ),
            comparison_metrics(
                labels,
                probabilities,
                probabilities * 0.8 + 1.0 / 3.0 * 0.2,
            ),
        )
        _assert_metric_tree_equal(
            self,
            group_bootstrap(
                labels,
                probabilities,
                groups,
                replicates=25,
                seed=4,
                sample_weight=weights,
            ),
            group_bootstrap(
                labels,
                probabilities,
                groups,
                replicates=25,
                seed=4,
            ),
        )

    def test_sensitivity_evaluation_path_uses_dataset_analysis_weight(self) -> None:
        weights = np.array([9.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dataset = _weighted_dataset(weights)
        labels, probabilities = _metric_fixture()
        result = evaluate_predictions(
            dataset,
            {"model": probabilities},
            replicates=20,
            seed=11,
        )
        expected = point_metrics(
            labels,
            probabilities,
            sample_weight=weights,
        )
        _assert_metric_tree_equal(self, result["model"]["point"], expected)
        expected_ci = group_bootstrap(
            labels,
            probabilities,
            dataset.cohort["subject_key"].to_numpy(),
            replicates=20,
            seed=11,
            sample_weight=weights,
        )
        _assert_metric_tree_equal(self, result["model"]["ci"], expected_ci)
        self.assertNotAlmostEqual(
            result["model"]["point"]["multiclass_brier"],
            point_metrics(labels, probabilities)["multiclass_brier"],
        )

    def test_repeated_cv_uses_dataset_weight_for_point_and_ci(self) -> None:
        weights = np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dataset = _weighted_dataset(weights)
        labels, probabilities = _metric_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeat.npz"
            np.savez_compressed(
                path,
                labels=labels,
                stay_id=dataset.cohort["stay_id"].to_numpy(),
                subject_key=dataset.cohort["subject_key"].to_numpy(),
                repeat0_model=probabilities,
                repeat1_model=probabilities,
            )
            result = evaluate_repeated_cv(
                dataset,
                path,
                ("model",),
                repeats=2,
                replicates=12,
                seed=5,
            )
        expected = aggregate_point_metrics(
            [
                point_metrics(labels, probabilities, sample_weight=weights),
                point_metrics(labels, probabilities, sample_weight=weights),
            ]
        )
        _assert_metric_tree_equal(self, result["model"]["point"], expected)
        expected_ci = group_bootstrap_repeated(
            labels,
            [probabilities, probabilities],
            dataset.cohort["subject_key"].to_numpy(),
            replicates=12,
            seed=5,
            sample_weight=weights,
        )
        _assert_metric_tree_equal(self, result["model"]["ci"], expected_ci)
        self.assertNotAlmostEqual(
            result["model"]["point"]["multiclass_brier"],
            point_metrics(labels, probabilities)["multiclass_brier"],
        )
        self.assertIn("macro_auc", result["model"]["ci"])

    def test_repeated_comparisons_use_repeat_specific_weights(self) -> None:
        labels, probabilities = _metric_fixture()
        comparator = probabilities * 0.8 + 1.0 / 3.0 * 0.2
        weights_a = np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 8.0])
        result = comparison_metrics_repeated(
            labels,
            [probabilities, probabilities],
            [comparator, comparator],
            sample_weight=[weights_a, weights_b],
        )
        expected = np.mean(
            [
                comparison_metrics(
                    labels,
                    probabilities,
                    comparator,
                    sample_weight=weights_a,
                )["delta_macro_auc"],
                comparison_metrics(
                    labels,
                    probabilities,
                    comparator,
                    sample_weight=weights_b,
                )["delta_macro_auc"],
            ]
        )
        self.assertAlmostEqual(result["delta_macro_auc"], expected)
        self.assertEqual(
            result["delong_inference_status"],
            "not_applicable_for_weighted_analysis_use_group_bootstrap",
        )

        validity: dict = {}
        paired_auc_difference_repeated(
            labels,
            [probabilities, probabilities],
            [comparator, comparator],
            np.array([f"p{index}" for index in range(len(labels))]),
            replicates=20,
            seed=9,
            sample_weight=[weights_a, weights_b],
            validity_out=validity,
        )
        self.assertEqual(validity["requested_replicates"], 20)


if __name__ == "__main__":
    unittest.main()
