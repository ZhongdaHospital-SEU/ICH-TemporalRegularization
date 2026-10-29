"""Regression tests for manuscript wording and generated-table inputs."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic import make_manuscript


def metric_point(offset: float = 0.0) -> dict[str, float | int]:
    return {
        "n": 100,
        "macro_auc": 0.70 + offset,
        "auc_any_deterioration": 0.72 + offset,
        "auc_severe_deterioration": 0.74 + offset,
        "multiclass_brier": 0.50 - offset,
        "calibration_slope_any": 0.90 + offset,
        "expected_calibration_error": 0.05,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class LambdaGridTests(unittest.TestCase):
    def test_lambda_grid_discovers_files_and_reads_manifest_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {
                "lambda_m0p00_s0p00": (0.0, 0.0, 0.01),
                "lambda_m0p05_s0p02": (0.05, 0.02, 0.02),
                "lambda_m0p10_s0p02": (0.10, 0.02, 0.03),
            }
            for tag, (monotone, smooth, offset) in settings.items():
                write_json(
                    root / f"metrics_{tag}.json",
                    {
                        "development": {
                            "dggru_tr": {
                                "point": metric_point(offset),
                            }
                        }
                    },
                )
                write_json(
                    root / f"run_manifest_{tag}.json",
                    {
                        "models": ["dggru_tr"],
                        "lambda_monotone": monotone,
                        "lambda_smooth": smooth,
                    },
                )
            with patch.object(make_manuscript, "RESULTS", root):
                frame = make_manuscript.lambda_frame()

        self.assertEqual(len(frame), 3)
        self.assertEqual(
            list(zip(frame["lambda monotone"], frame["lambda smooth"])),
            [(0.0, 0.0), (0.05, 0.02), (0.10, 0.02)],
        )

    def test_lambda_grid_does_not_silently_drop_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "metrics_lambda_unknown.json",
                {
                    "development": {
                        "dggru_tr": {"point": metric_point()},
                    }
                },
            )
            with patch.object(make_manuscript, "RESULTS", root):
                with self.assertRaises(FileNotFoundError):
                    make_manuscript.lambda_frame()


class SensitivityTableContractTests(unittest.TestCase):
    def test_all_outline_variants_use_the_correct_dataset_reference(self) -> None:
        model_points = {
            "xgb_summary": (0.58, 0.60),
            "gru": (0.59, 0.61),
            "dggru_tr": (0.60, 0.62),
        }

        def scenario_metrics(macro_auc: float) -> dict:
            return {
                "development": {
                    model: {
                        "point": {
                            "macro_auc": (
                                macro_auc if model == "dggru_tr" else point[0]
                            ),
                            "auc_any_deterioration": point[1],
                        }
                    }
                    for model, point in model_points.items()
                }
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audits = {"mimic4": [], "eicu": []}
            for analysis in make_manuscript.OUTLINE_SENSITIVITY_ANALYSES:
                dataset = str(analysis["dataset"])
                for scenario, label in analysis["variants"]:
                    supported = scenario in {
                        "s1_baseline_max",
                        "s1_baseline_last",
                        "s8_eicu_gcs_unadjusted",
                    }
                    audits[dataset].append(
                        {
                            "scenario": {
                                "scenario_id": scenario,
                                "variant_title": label,
                            },
                            "status": {
                                "support": "supported" if supported else "unsupported",
                                "model_run_allowed": supported,
                                "reason": "fixture",
                            },
                            "outcomes": {"0": 2, "1": 1, "2": 1},
                        }
                    )
                    if not supported:
                        continue
                    directory = root / "sensitivity" / dataset
                    directory.mkdir(parents=True, exist_ok=True)
                    macro_auc = 0.65 if scenario == "s1_baseline_last" else 0.60
                    write_json(
                        directory / f"{scenario}.json",
                        scenario_metrics(macro_auc),
                    )
                    labels = (
                        np.array([0, 1, 2, 2])
                        if scenario == "s1_baseline_last"
                        else np.array([0, 0, 1, 2])
                    )
                    np.savez_compressed(
                        directory / f"{scenario}_predictions.npz",
                        labels=labels,
                    )
            for dataset, records in audits.items():
                directory = root / "sensitivity" / dataset
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "scenario_audit.json", records)

            with patch.object(make_manuscript, "RESULTS", root):
                frame = make_manuscript.sensitivity_frame()

        self.assertEqual(len(frame), 17)
        last = frame[
            frame["Variant"].eq("Final pre-landmark GCS")
        ].iloc[0]
        self.assertAlmostEqual(
            float(last["Change in any event rate vs primary (pp)"]),
            25.0,
        )
        self.assertAlmostEqual(
            float(last["Change in DG-GRU + TR macro AUC vs primary"]),
            0.05,
        )
        self.assertTrue(
            frame["Change in DG-GRU + TR macro AUC vs primary"]
            .notna()
            .any()
        )

    def test_lambda_grid_requires_at_least_one_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(make_manuscript, "RESULTS", root):
                with self.assertRaises(FileNotFoundError):
                    make_manuscript.lambda_frame()

    def test_sensitivity_requires_every_prespecified_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(make_manuscript, "RESULTS", root):
                with self.assertRaises(FileNotFoundError):
                    make_manuscript.sensitivity_frame()


class DynamicModelTests(unittest.TestCase):
    def test_manifest_model_list_supports_eight_models(self) -> None:
        names = tuple(f"model_{index}" for index in range(8))
        block = {
            name: {"point": metric_point(index / 100)}
            for index, name in enumerate(names)
        }
        block["paired_differences_vs_dggru_tr"] = {}
        ordered = make_manuscript.ordered_model_names(
            block,
            {"models": list(names)},
        )
        self.assertEqual(ordered, names)

    def test_table2_loader_rejects_stale_model_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "tables"
            tables.mkdir()
            rows = [
                {"Cohort": cohort, "Model": model, "n": 100}
                for cohort in ("development", "mimic3", "eicu")
                for model in ("model_0", "model_1")
            ]
            pd.DataFrame(rows).to_csv(
                tables / "table3_model_performance.csv",
                index=False,
            )
            with patch.object(make_manuscript, "RESULTS", root):
                with self.assertRaises(ValueError):
                    make_manuscript.table2_output_frame(
                        {},
                        ("model_0", "model_1", "model_2"),
                    )


class ManuscriptWordingTests(unittest.TestCase):
    def test_title_emphasis_follows_the_paired_development_gain(self) -> None:
        block = {
            model: {"point": metric_point(index / 1000)}
            for index, model in enumerate(make_manuscript.OUTLINE_MODEL_ORDER)
        }
        block["paired_differences_vs_dggru_tr"] = {
            model: [-0.01, 0.01]
            for model in make_manuscript.OUTLINE_MODEL_ORDER
            if model != "dggru_tr"
        }
        no_gain = {
            "development": block,
            "external": {"mimic3": block, "eicu": block},
        }
        context = make_manuscript.title_context(no_gain)
        self.assertEqual(context["title"], make_manuscript.TITLE_WITHOUT_GAIN)
        self.assertFalse(context["temporal_regularization_primary"])
        self.assertNotIn("temporal regularization", context["keywords"].lower())

        block["paired_differences_vs_dggru_tr"]["dggru"] = [0.001, 0.010]
        with_gain = {
            "development": block,
            "external": {"mimic3": block, "eicu": block},
        }
        context = make_manuscript.title_context(with_gain)
        self.assertEqual(context["title"], make_manuscript.TITLE)
        self.assertTrue(context["temporal_regularization_primary"])
        self.assertIn("temporal regularization", context["keywords"].lower())

    def test_static_feature_wording_follows_the_manifest_count(self) -> None:
        self.assertIn(
            "23 pre-specified core first-24-hour static predictors",
            make_manuscript.shared_static_sentence(23),
        )
        self.assertIn(
            "same 23-variable core static matrix",
            make_manuscript.shared_static_discussion_sentence(23),
        )

    def test_feature_count_uses_the_primary_static_subset(self) -> None:
        manifest = {
            "features": [f"variable_{index}" for index in range(210)],
            "temporal_static_feature_count": 23,
        }
        self.assertEqual(make_manuscript.feature_count(manifest), 23)
        self.assertEqual(
            make_manuscript.eligible_feature_count(manifest), 210
        )
        self.assertEqual(
            make_manuscript.eligible_feature_count(manifest)
            - make_manuscript.dynamic_summary_count(),
            125,
        )

    def test_prohibited_claims_and_hard_coded_counts_are_absent(self) -> None:
        source = Path(make_manuscript.__file__).read_text(encoding="utf-8")
        for text in (
            "earlier-generation",
            "predates it",
            "prespecified",
            "strongest internal",
            "These intervals describe transport variation",
            "internal sampling and fold-partition uncertainty",
            "supports reproducible temporal validation",
            "five runs with lambda_smooth",
            "fixed-smoothness series with lambda_smooth",
            "using the primary weight of 0.05",
            '"Primary monotone weight", "0.05"',
            '"Primary smoothness weight", "0.02"',
            "all seven models",
            "{len(models)} models were evaluated",
        ):
            if text == "{len(models)} models were evaluated":
                self.assertIn(text, source)
            else:
                self.assertNotIn(text, source)

    def test_required_qualifiers_and_single_table2_source_are_present(self) -> None:
        source = Path(make_manuscript.__file__).read_text(encoding="utf-8")
        self.assertIn("highest out-of-fold point estimate", source)
        self.assertIn(
            "conditional patient-cluster bootstrap interval",
            source,
        )
        self.assertIn(
            "did not re-estimate models, generate new cross-validation "
            "partitions",
            source,
        )
        self.assertIn("combined CV-partition", source)
        self.assertIn("training stochasticity", source)
        self.assertIn(
            "missing RASS was not imputed as no deep sedation",
            source,
        )
        self.assertIn(
            "database-generation and source shift",
            source,
        )
        self.assertIn(
            "availability and variance screening",
            source,
        )
        self.assertIn(
            "minor transductive step",
            source,
        )
        self.assertIn(
            "Every included patient, including a patient who died in the ICU "
            "by hour 72, was required to have at least two distinct ICU hours",
            source,
        )
        self.assertIn(
            "Sparse-GCS deaths were therefore excluded rather than promoted "
            "to Grade 2",
            source,
        )
        self.assertNotIn(
            "Death was treated as Grade 2 when it occurred in the ICU by "
            "hour 72, irrespective of the number of post-landmark GCS "
            "measurements",
            source,
        )
        self.assertIn(
            "Cross-family comparisons therefore describe complete pipelines",
            source,
        )
        self.assertIn(
            "pre-specified core first-24-hour static predictors",
            source,
        )
        self.assertIn(
            'f"{n_static}-variable core static matrix',
            source,
        )
        self.assertNotIn("n_static == 120", source)
        self.assertIn(
            "supports standardized landmark-based external evaluation",
            source,
        )
        self.assertNotIn("MIMIC-III temporal external", source)
        self.assertIn(
            '"table2": table2_output_frame(metrics, models)',
            source,
        )
        self.assertEqual(
            source.count("{markdown_table(tables['table2'])}"),
            2,
        )
        self.assertNotIn("\nMODELS = (", source)

    def test_public_package_excludes_patient_level_predictions(self) -> None:
        source = Path(make_manuscript.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "The public analysis package is limited to aggregate tables, "
            "model code, manifests, and figure-level aggregated data",
            source,
        )
        self.assertIn(
            "contains no patient-level predictions, stay_id values, "
            "subject_key values, hospital identifiers, or row-level source data",
            source,
        )
        self.assertIn(
            "Patient-level prediction files are used only in the local audit "
            "workspace and are not redistributed",
            source,
        )
        self.assertIn("Exact numerical identity is not guaranteed", source)
        self.assertIn(
            "No claim of bitwise deterministic reproduction is made",
            source,
        )
        self.assertNotIn(
            "Derived aggregate tables, predictions, model manifests",
            source,
        )
        self.assertNotIn(
            "aggregate figure-source data only",
            source,
        )
        self.assertNotIn(
            "The statistical outputs were reproduced",
            source,
        )


if __name__ == "__main__":
    unittest.main()
