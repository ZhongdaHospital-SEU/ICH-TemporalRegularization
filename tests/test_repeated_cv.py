from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.train_evaluate import (
    Dataset,
    aggregate_point_metrics,
    build_repeated_splits,
    evaluate_repeated_cv,
    point_metrics,
)


def make_dataset() -> Dataset:
    return Dataset(
        name="unit",
        cohort=pd.DataFrame(
            {
                "stay_id": [11, 12, 13, 14, 15, 16],
                "subject_key": ["p1", "p2", "p3", "p4", "p5", "p6"],
            }
        ),
        sequences=np.zeros((6, 2, 2), dtype=np.float32),
        masks=np.ones((6, 2, 2), dtype=np.float32),
        labels=np.array([0, 0, 1, 1, 2, 2], dtype=int),
        static_all=pd.DataFrame({"x": np.arange(6)}),
        static_deep=pd.DataFrame({"x": np.arange(6)}),
    )


class AggregationTests(unittest.TestCase):
    def test_scalar_metrics_are_unweighted_repeat_means(self) -> None:
        rows = [
            {"n": 6.0, "macro_auc": 0.6, "confusion_matrix": [[2, 0, 0]]},
            {"n": 6.0, "macro_auc": 0.8, "confusion_matrix": [[1, 1, 0]]},
        ]
        result = aggregate_point_metrics(rows)
        self.assertAlmostEqual(result["macro_auc"], 0.7)
        np.testing.assert_allclose(result["confusion_matrix"], [[1.5, 0.5, 0.0]])

    def test_rows_with_different_schemas_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_point_metrics(
                [{"n": 6.0, "macro_auc": 0.6}, {"n": 6.0}]
            )

    def test_repeated_splits_are_reproducible_and_patient_disjoint(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        groups = np.array([f"p{index}" for index in range(9)])
        first, meta = build_repeated_splits(
            labels,
            groups,
            folds=3,
            repeats=2,
            seed=17,
        )
        second, _ = build_repeated_splits(
            labels,
            groups,
            folds=3,
            repeats=2,
            seed=17,
        )
        self.assertEqual(meta, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
            self.assertFalse(set(groups[train_a]).intersection(groups[test_a]))


class RepeatedEvaluationTests(unittest.TestCase):
    def test_repeated_cv_matches_manual_metric_mean_and_checks_ids(self) -> None:
        dataset = make_dataset()
        repeat0 = {
            "model": np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.7, 0.2, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.1, 0.8],
                ]
            )
        }
        repeat1 = {
            "model": np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.1, 0.8],
                    [0.2, 0.1, 0.7],
                ]
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeat.npz"
            np.savez_compressed(
                path,
                labels=dataset.labels,
                stay_id=dataset.cohort["stay_id"].to_numpy(),
                subject_key=dataset.cohort["subject_key"].to_numpy(),
                repeat0_model=repeat0["model"],
                repeat1_model=repeat1["model"],
            )
            result = evaluate_repeated_cv(
                dataset,
                path,
                ("model",),
                repeats=2,
                replicates=0,
                seed=7,
            )
        expected = aggregate_point_metrics(
            [
                point_metrics(dataset.labels, repeat0["model"]),
                point_metrics(dataset.labels, repeat1["model"]),
            ]
        )
        for key, value in expected.items():
            actual = result["model"]["point"][key]
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    np.testing.assert_allclose(actual[subkey], subvalue)
            elif isinstance(value, list):
                np.testing.assert_allclose(actual, value)
            elif np.isfinite(value):
                self.assertAlmostEqual(actual, value)
            else:
                self.assertTrue(np.isnan(actual))

    def test_repeated_cv_rejects_misaligned_stay_ids(self) -> None:
        dataset = make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeat.npz"
            np.savez_compressed(
                path,
                labels=dataset.labels,
                stay_id=np.arange(6),
                subject_key=dataset.cohort["subject_key"].to_numpy(),
                repeat0_model=np.full((6, 3), 1.0 / 3.0),
            )
            with self.assertRaisesRegex(ValueError, "stay_id"):
                evaluate_repeated_cv(
                    dataset,
                    path,
                    ("model",),
                    repeats=1,
                    replicates=0,
                    seed=7,
                )


if __name__ == "__main__":
    unittest.main()
