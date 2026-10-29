"""Tests for metric behavior under incomplete bootstrap samples."""
from __future__ import annotations

import unittest

import numpy as np

from src.ich_dynamic.train_evaluate import (
    add_holm_comparisons,
    calibration_slope_intercept_grade,
    group_bootstrap,
    multiclass_auc,
    paired_auc_difference,
    point_metrics,
)


class MulticlassAUCTests(unittest.TestCase):
    def test_macro_auc_requires_every_grade(self) -> None:
        labels = np.array([0, 0, 2, 2])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.1, 0.8],
            ]
        )
        self.assertTrue(np.isnan(multiclass_auc(labels, probabilities)))

    def test_macro_auc_averages_all_three_grades(self) -> None:
        labels = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ]
        )
        self.assertAlmostEqual(multiclass_auc(labels, probabilities), 1.0)

    def test_point_metrics_reports_each_grade_calibration(self) -> None:
        labels = np.repeat(np.array([0, 1, 2]), 20)
        probabilities = np.concatenate(
            [
                np.tile([0.8, 0.1, 0.1], (20, 1)),
                np.tile([0.1, 0.8, 0.1], (20, 1)),
                np.tile([0.1, 0.1, 0.8], (20, 1)),
            ]
        )
        metrics = point_metrics(labels, probabilities)
        for grade in range(3):
            self.assertIn(f"calibration_slope_grade_{grade}", metrics)
            self.assertIn(f"calibration_intercept_grade_{grade}", metrics)

        slope, intercept = calibration_slope_intercept_grade(
            labels,
            probabilities,
            grade=1,
        )
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(intercept))


class InferenceContractTests(unittest.TestCase):
    def test_bootstrap_reports_valid_replicate_counts(self) -> None:
        labels = np.repeat(np.array([0, 1, 2]), 20)
        probabilities = np.concatenate(
            [
                np.tile([0.8, 0.15, 0.05], (20, 1)),
                np.tile([0.1, 0.8, 0.1], (20, 1)),
                np.tile([0.05, 0.15, 0.8], (20, 1)),
            ]
        )
        groups = np.array([f"p{index}" for index in range(len(labels))])
        validity: dict = {}
        group_bootstrap(
            labels,
            probabilities,
            groups,
            replicates=24,
            seed=13,
            validity_out=validity,
        )
        self.assertEqual(validity["requested_replicates"], 24)
        self.assertEqual(validity["clusters"], len(labels))
        self.assertIn("macro_auc", validity["metric_valid_replicates"])
        self.assertEqual(
            validity["metric_valid_replicates"]["macro_auc"],
            24,
        )

        paired_validity: dict = {}
        paired_auc_difference(
            labels,
            probabilities,
            probabilities,
            groups,
            replicates=24,
            seed=13,
            validity_out=paired_validity,
        )
        self.assertEqual(paired_validity["requested_replicates"], 24)
        self.assertEqual(paired_validity["valid_replicates"], 24)

    def test_holm_family_does_not_shrink_with_model_subset(self) -> None:
        output = {
            "comparisons_vs_primary": {
                "xgb_static": {
                    "delong_any_p": 0.01,
                    "delong_severe_p": 0.5,
                },
                "xgb_summary": {
                    "delong_any_p": 0.04,
                    "delong_severe_p": 0.5,
                },
            }
        }
        add_holm_comparisons(output)
        comparisons = output["comparisons_vs_primary"]
        self.assertAlmostEqual(
            comparisons["xgb_static"]["delong_any_p_holm"],
            0.10,
        )
        self.assertAlmostEqual(
            comparisons["xgb_summary"]["delong_any_p_holm"],
            0.36,
        )
        self.assertEqual(
            comparisons["xgb_static"]["delong_holm_family"],
            "fixed_five_outline_baselines_x_any_severe",
        )
        self.assertEqual(
            comparisons["xgb_static"]["inference_role"],
            "primary_holm_family",
        )


if __name__ == "__main__":
    unittest.main()
