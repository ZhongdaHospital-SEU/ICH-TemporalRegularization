"""Contract tests for the outline-aligned temporal-structure exporter."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic import export_trajectories


EXPECTED_MODELS = (
    "dggru",
    "dggru_tr",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)


def synthetic_trajectories() -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray
]:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    masks = np.ones((len(labels), 8, 3), dtype=np.float32)
    hours = np.arange(8, dtype=np.float32)
    trajectories = {}
    for model_index, model in enumerate(EXPECTED_MODELS):
        matrix = np.empty((len(labels), len(hours)), dtype=np.float32)
        for row, grade in enumerate(labels):
            start = 0.08 + 0.22 * int(grade) + 0.005 * model_index
            risk = start + 0.018 * hours
            matrix[row] = np.clip(risk, 0.01, 0.99)
        trajectories[model] = matrix
    return trajectories, masks, labels


class TrajectoryModelContractTests(unittest.TestCase):
    def test_current_control_set_and_metadata_semantics(self) -> None:
        self.assertEqual(export_trajectories.TRAJECTORY_MODELS, EXPECTED_MODELS)
        self.assertEqual(tuple(export_trajectories.MODELS), EXPECTED_MODELS)
        source = Path(export_trajectories.__file__).read_text(encoding="utf-8")
        self.assertNotIn("dggru_smooth", source)
        self.assertNotIn("dggru_tr_anti", source)

        for model in EXPECTED_MODELS:
            options = export_trajectories.temporal_training_options(
                export_trajectories.MODELS[model],
                lambda_monotone=0.05,
                lambda_smooth=0.02,
            )
            self.assertTrue(options["use_graph"])
            self.assertEqual(options["architecture"], "dggru")

        reversal = export_trajectories.temporal_training_options(
            export_trajectories.MODELS["dggru_tr_treverse"],
            lambda_monotone=0.05,
            lambda_smooth=0.02,
        )
        self.assertEqual(reversal["time_mode"], "reverse")
        self.assertTrue(reversal["use_regularization"])

        permutation = export_trajectories.temporal_training_options(
            export_trajectories.MODELS["dggru_tr_tpermute"],
            lambda_monotone=0.05,
            lambda_smooth=0.02,
        )
        self.assertEqual(permutation["time_mode"], "permute")
        self.assertTrue(permutation["use_regularization"])

        laplace = export_trajectories.temporal_training_options(
            export_trajectories.MODELS["dggru_laplace"],
            lambda_monotone=0.05,
            lambda_smooth=0.02,
        )
        self.assertTrue(laplace["laplace_only"])
        self.assertEqual(laplace["laplace_input_scale"], 0.01)
        self.assertEqual(laplace["lambda_monotone"], 0.0)
        self.assertEqual(laplace["lambda_smooth"], 0.02)
        self.assertEqual(laplace["time_mode"], "forward")

    def test_trajectory_statistics_preserve_outline_metrics(self) -> None:
        trajectories, masks, labels = synthetic_trajectories()
        statistics = export_trajectories.trajectory_statistics(
            trajectories["dggru_tr"],
            masks,
            labels,
        )
        required = {
            "outcome",
            "last_hour",
            "final_risk",
            "mean_risk",
            "negative_step_fraction",
            "monotone_violation",
            "roughness",
            "total_variation",
        }
        self.assertTrue(required.issubset(statistics.columns))
        self.assertEqual(len(statistics), len(labels))
        self.assertTrue(statistics["last_hour"].eq(8).all())

    def test_figure5_name_does_not_occupy_decision_curve_figure(self) -> None:
        trajectories, masks, labels = synthetic_trajectories()
        landmark_metrics = pd.DataFrame(
            [
                {
                    "landmark_hours": hours,
                    "model": model,
                    "macro_auc": 0.65 + 0.01 * index,
                    "ci_low": 0.60 + 0.01 * index,
                    "ci_high": 0.70 + 0.01 * index,
                }
                for index, (hours, model) in enumerate(
                    (
                        (hours, model)
                        for hours in (6, 24, 48)
                        for model in EXPECTED_MODELS
                    )
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            figures = Path(temporary)
            sentinel = figures / "figure5_decision_curve.png"
            sentinel.write_bytes(b"decision-curve-owned-by-dca")
            with patch.object(export_trajectories, "FIGURES", figures):
                summary = export_trajectories.create_figure(
                    trajectories,
                    masks,
                    labels,
                    landmark_metrics,
                )

            self.assertEqual(tuple(summary["model"]), EXPECTED_MODELS)
            self.assertEqual(
                export_trajectories.FIGURE5_STEM,
                "figure5_temporal_structure",
            )
            self.assertIn(
                "generic Laplace input-noise",
                export_trajectories.MODEL_LABELS["dggru_laplace"],
            )
            self.assertNotIn(
                "Smoothness",
                export_trajectories.MODEL_LABELS["dggru_laplace"],
            )
            self.assertTrue(
                (figures / "figure5_temporal_structure.png").exists()
            )
            self.assertTrue(
                (figures / "figure5_temporal_structure.pdf").exists()
            )
            self.assertFalse(
                (figures / "figure7_risk_trajectories.png").exists()
            )
            self.assertFalse(
                (figures / "figure7_risk_trajectories.pdf").exists()
            )
            self.assertEqual(
                sentinel.read_bytes(),
                b"decision-curve-owned-by-dca",
            )

    def test_landmark_loader_uses_the_registered_landmark_model_set(self) -> None:
        block = {
            model: {
                "point": {"macro_auc": 0.65},
                "ci": {"macro_auc": [0.60, 0.70]},
            }
            for model in EXPECTED_MODELS
        }
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            landmark_dir = results / "sensitivity" / "mimic4"
            landmark_dir.mkdir(parents=True)
            for hours in (6, 24, 48):
                (landmark_dir / f"s6_landmark_{hours}h.json").write_text(
                    json.dumps({"development": block}),
                    encoding="utf-8",
                )
            with patch.object(export_trajectories, "RESULTS", results):
                frame = export_trajectories.landmark_metrics_frame()
        self.assertEqual(len(frame), 15)
        self.assertEqual(set(frame["model"]), set(EXPECTED_MODELS))

    def test_manifest_cell_values_are_repeat_fold_seed_aligned(self) -> None:
        manifest = {
            "folds": 2,
            "cv_repeats": 2,
            "cv_best_epochs": {
                "dggru_tr": {
                    "values": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            },
        }
        self.assertEqual(
            export_trajectories._cell_values(
                manifest,
                "cv_best_epochs",
                "dggru_tr",
                1,
                2,
            ),
            [3, 4],
        )
        with self.assertRaises(ValueError):
            export_trajectories._cell_values(
                {
                    "folds": 2,
                    "cv_repeats": 2,
                    "cv_best_epochs": {
                        "dggru_tr": {"values": [1, 2, 3]},
                    },
                },
                "cv_best_epochs",
                "dggru_tr",
                0,
                2,
            )

    def test_trajectory_export_uses_the_manifest_seed_contract(self) -> None:
        source = Path(export_trajectories.__file__).read_text(encoding="utf-8")
        self.assertIn("model_seeds = tuple(", source)
        self.assertIn("best_epochs=selected_epochs", source)
        self.assertNotIn("for offset in range(3)", source)
        self.assertNotIn("folds[\"repeat\"].eq(0)", source)


if __name__ == "__main__":
    unittest.main()
