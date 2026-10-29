"""Fail-closed contracts for formal-run validation and promotion."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from work import audit_source_drift, promote_formal_run, validate_formal_run
from src.ich_dynamic import train_evaluate


class PromotionGateTests(unittest.TestCase):
    def test_failed_validation_report_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            report = results / "audits" / "formal_run_validation_round6.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps({"status": "failed", "tag": "round6"}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(promote_formal_run, "RESULTS", results),
                mock.patch.object(promote_formal_run, "validate") as validate,
            ):
                with self.assertRaises(ValueError):
                    promote_formal_run.promote("round6")
                validate.assert_not_called()

    def test_tag_mismatch_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            report = results / "audits" / "formal_run_validation_round6.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps({"status": "passed", "tag": "round5"}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(promote_formal_run, "RESULTS", results),
                mock.patch.object(promote_formal_run, "validate") as validate,
            ):
                with self.assertRaises(ValueError):
                    promote_formal_run.promote("round6")
                validate.assert_not_called()

    def test_passed_report_preflights_all_artifacts_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            report = results / "audits" / "formal_run_validation_round6.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps({"status": "passed", "tag": "round6"}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(promote_formal_run, "RESULTS", results),
                mock.patch.object(promote_formal_run, "validate", return_value={}),
                mock.patch.object(promote_formal_run, "_copy_atomic") as copy,
            ):
                with self.assertRaises(FileNotFoundError):
                    promote_formal_run.promote("round6")
                copy.assert_not_called()


class ValidatorScopeTests(unittest.TestCase):
    def test_runtime_source_scope_is_derived_from_imports(self) -> None:
        self.assertEqual(
            validate_formal_run.RUNTIME_SOURCE_FILES,
            set(audit_source_drift.training_runtime_sources()),
        )
        self.assertIn("models.py", validate_formal_run.RUNTIME_SOURCE_FILES)
        self.assertNotIn(
            "sensitivity_analysis.py",
            validate_formal_run.RUNTIME_SOURCE_FILES,
        )

    def test_training_snapshot_covers_exactly_the_import_closure(self) -> None:
        self.assertEqual(
            set(train_evaluate.training_source_files()),
            set(audit_source_drift.training_runtime_sources()),
        )
        self.assertNotIn(
            "make_manuscript.py",
            train_evaluate.training_source_files(),
        )

    def test_repeated_predictions_reject_negative_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "repeat.npz"
            labels = np.array([0, 1, 2])
            probabilities = np.full((3, 3), 1.0 / 3.0)
            probabilities[0, 0] = -0.01
            np.savez_compressed(
                path,
                labels=labels,
                stay_id=np.array([1, 2, 3]),
                repeat0_model=probabilities,
            )
            with self.assertRaises(validate_formal_run.ValidationFailure):
                validate_formal_run._repeated_probability_checks(
                    path,
                    ("model",),
                    model_repeats={"model": 1},
                    expected_ids=np.array([1, 2, 3]),
                    expected_labels=labels,
                )

    def test_sparse_post_landmark_gcs_is_never_allowed(self) -> None:
        with self.assertRaises(validate_formal_run.ValidationFailure):
            validate_formal_run._require_post_landmark_gcs_eligibility(
                "synthetic",
                minimum_post=np.array([0, 1]),
                death_in_icu=np.array([True, True]),
            )
        with self.assertRaises(validate_formal_run.ValidationFailure):
            validate_formal_run._require_post_landmark_gcs_eligibility(
                "synthetic",
                minimum_post=np.array([0, 1, 2, 3]),
                death_in_icu=np.array([True, True, False, False]),
            )
        validate_formal_run._require_post_landmark_gcs_eligibility(
            "synthetic",
            minimum_post=np.array([2, 3, 4]),
            death_in_icu=np.array([True, False, False]),
        )

    def test_formal_design_hard_codes_five_by_five_and_2000(self) -> None:
        validate_formal_run._require_formal_design(
            folds=5,
            repeats=5,
            replicates=2000,
        )
        for kwargs in (
            {"folds": 4, "repeats": 5, "replicates": 2000},
            {"folds": 5, "repeats": 4, "replicates": 2000},
            {"folds": 5, "repeats": 5, "replicates": 1999},
        ):
            with self.assertRaises(validate_formal_run.ValidationFailure):
                validate_formal_run._require_formal_design(**kwargs)

    def test_primary_bootstrap_validity_is_fail_closed(self) -> None:
        def block(models) -> dict:
            return {
                **{
                    model: {
                        "ci_validity": {
                            "requested_replicates": 2000,
                            "metric_valid_replicates": {
                                "macro_auc": 2000,
                                "auc_any_deterioration": 2000,
                                "auc_severe_deterioration": 2000,
                            },
                        }
                    }
                    for model in models
                },
                "paired_differences_bootstrap_validity_vs_dggru_tr": {
                    "xgb_static": {
                        "requested_replicates": 2000,
                        "valid_replicates": 2000,
                    }
                },
            }

        metrics = {
            "development": block(validate_formal_run.REQUIRED_MODELS),
            "external": {
                "mimic3": block(validate_formal_run.OUTLINE_MODEL_ORDER),
                "eicu": block(validate_formal_run.OUTLINE_MODEL_ORDER),
            },
        }
        report = validate_formal_run._primary_bootstrap_validity_checks(
            metrics,
            2000,
        )
        self.assertEqual(report["development"]["models"], 16)

        metrics["external"]["eicu"]["dggru_tr"]["ci_validity"][
            "metric_valid_replicates"
        ]["macro_auc"] = 1999
        with self.assertRaises(validate_formal_run.ValidationFailure):
            validate_formal_run._primary_bootstrap_validity_checks(
                metrics,
                2000,
            )


if __name__ == "__main__":
    unittest.main()
