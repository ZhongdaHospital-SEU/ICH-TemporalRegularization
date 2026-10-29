from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.ich_dynamic import explain


RUNTIME_SOURCES = (
    "explain.py",
    "common.py",
    "models.py",
    "train_evaluate.py",
)
XGB_PARAMETERS = {
    "n_estimators": 321,
    "max_depth": 4,
    "learning_rate": 0.025,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 4.0,
}
TEMPORAL_PARAMETERS = {
    "hidden_size": 96,
    "dropout": 0.3,
    "learning_rate": 0.0007,
    "weight_decay": 0.0005,
    "batch_size": 128,
    "max_epochs": 25,
    "patience": 7,
}


def make_manifest(source_root: Path) -> dict:
    source_hashes = {}
    for filename in RUNTIME_SOURCES:
        path = source_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {filename}\n", encoding="utf-8")
        source_hashes[filename] = explain._sha256_file(path)
    return {
        "features": ["age", "baseline_gcs_max"],
        "config": {
            "channels": ["gcs_total", "hr"],
            "sequence_hours": 24,
        },
        "seed": 20261024,
        "lambda_smooth_ratio": 0.4,
        "source_sha256": source_hashes,
        "cv_selected_hyperparameters": {
            "xgb_summary": [
                {
                    "family": "xgboost",
                    "parameters": dict(XGB_PARAMETERS),
                }
            ],
            "dggru_tr": [
                {
                    "family": "sequence_models",
                    "parameters": dict(TEMPORAL_PARAMETERS),
                }
            ],
        },
        "development_modal_hyperparameters": {
            "xgb_summary": dict(XGB_PARAMETERS),
            "dggru_tr": dict(TEMPORAL_PARAMETERS),
        },
        "cv_best_epochs": {
            "dggru_tr": {
                "values": [3, 4, 5],
                "external_transport_epochs": 4,
            }
        },
        "cv_selected_lambdas": {
            "dggru_tr": {
                "values": [0.2, 0.3, 0.4],
                "median": 0.3,
            }
        },
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ExplanationManifestTests(unittest.TestCase):
    def test_missing_formal_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                explain.ExplanationManifestError,
                "manifest is missing",
            ):
                explain.load_explanation_manifest(
                    root / "run_manifest.json",
                    source_root=root / "src",
                    runtime_sources=RUNTIME_SOURCES,
                )

    def test_stale_runtime_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "src"
            manifest = make_manifest(source_root)
            manifest_path = root / "run_manifest.json"
            write_manifest(manifest_path, manifest)
            (source_root / "models.py").write_text(
                "# changed after formal fitting\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                explain.ExplanationManifestError,
                "runtime source hashes are stale",
            ):
                explain.load_explanation_manifest(
                    manifest_path,
                    source_root=source_root,
                    runtime_sources=RUNTIME_SOURCES,
                )

    def test_missing_selected_parameters_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "src"
            manifest = make_manifest(source_root)
            del manifest["development_modal_hyperparameters"]["xgb_summary"]
            manifest_path = root / "run_manifest.json"
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(
                explain.ExplanationManifestError,
                "development_modal_hyperparameters.xgb_summary",
            ):
                explain.load_explanation_manifest(
                    manifest_path,
                    source_root=source_root,
                    runtime_sources=RUNTIME_SOURCES,
                )

    def test_manifest_selected_values_are_used_and_bound_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "src"
            manifest = make_manifest(source_root)
            manifest_path = root / "run_manifest.json"
            write_manifest(manifest_path, manifest)

            selection = explain.load_explanation_manifest(
                manifest_path,
                source_root=source_root,
                runtime_sources=RUNTIME_SOURCES,
            )

            self.assertEqual(selection.xgb_parameters, XGB_PARAMETERS)
            self.assertEqual(selection.temporal_parameters, TEMPORAL_PARAMETERS)
            self.assertEqual(selection.temporal_epochs, 4)
            self.assertAlmostEqual(selection.lambda_monotone, 0.3)
            self.assertAlmostEqual(selection.lambda_smooth, 0.12)
            self.assertEqual(
                selection.manifest_sha256,
                explain._sha256_file(manifest_path),
            )
            output = root / "table.csv"
            output.write_text("feature,value\nage,1\n", encoding="utf-8")
            binding = explain._output_binding(selection, [output])
            self.assertEqual(
                binding["manifest_sha256"],
                selection.manifest_sha256,
            )
            self.assertEqual(
                binding["outputs"][str(output)],
                explain._sha256_file(output),
            )

    def test_xgb_refit_uses_manifest_selected_parameters(self) -> None:
        fitted = mock.Mock()
        with mock.patch.object(
            explain.xgb,
            "XGBClassifier",
            return_value=fitted,
        ) as model_class:
            result = explain._fit_tree_model(
                np.zeros((4, 2), dtype=float),
                np.array([0, 1, 2, 0]),
                17,
                XGB_PARAMETERS,
            )
        self.assertIs(result, fitted)
        kwargs = model_class.call_args.kwargs
        for key, value in XGB_PARAMETERS.items():
            self.assertEqual(kwargs[key], value)
        self.assertEqual(kwargs["random_state"], 17)


if __name__ == "__main__":
    unittest.main()
