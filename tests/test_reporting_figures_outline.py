"""Contract tests for outline-aligned tables and figures."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic import make_tables_figures as reporting


def _audit_record(
    scenario_id: str,
    *,
    support: str,
    model_run_allowed: bool,
    variant_title: str,
    n: int | None = None,
) -> dict:
    record = {
        "scenario": {
            "analysis_id": scenario_id[1],
            "analysis_title": f"Outline analysis {scenario_id[1]}",
            "scenario_id": scenario_id,
            "variant_title": variant_title,
        },
        "status": {
            "support": support,
            "model_run_allowed": model_run_allowed,
            "reason": f"{support} fixture",
        },
        "n": n,
        "outcomes": {"0": 2, "1": 1, "2": 1} if n is not None else None,
        "artifact": None,
    }
    return record


def _metrics_fixture() -> dict:
    return {
        model: {
            "point": {
                "macro_auc": 0.6,
                "auc_any_deterioration": 0.65,
                "multiclass_brier": 0.2,
            }
        }
        for model in reporting.SENSITIVITY_MODELS
    }


class SensitivityReportingTests(unittest.TestCase):
    def test_current_dataset_audits_preserve_support_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "tables"
            mimic4 = root / "sensitivity" / "mimic4"
            mimic3 = root / "sensitivity" / "mimic3"
            eicu = root / "sensitivity" / "eicu"
            mimic4.mkdir(parents=True)
            mimic3.mkdir(parents=True)
            eicu.mkdir(parents=True)

            mimic4_records = [
                _audit_record(
                    "s1_baseline_last",
                    support="supported",
                    model_run_allowed=True,
                    variant_title="Final pre-landmark GCS",
                    n=4,
                ),
                _audit_record(
                    "s6_landmark_6h",
                    support="outcome_only",
                    model_run_allowed=False,
                    variant_title="6 h landmark",
                ),
                _audit_record(
                    "s3_unsedated_only",
                    support="unsupported",
                    model_run_allowed=False,
                    variant_title="Observed unsedated subgroup",
                ),
            ]
            eicu_records = [
                _audit_record(
                    "s8_eicu_gcs_unadjusted",
                    support="supported",
                    model_run_allowed=True,
                    variant_title="Recorded GCS",
                    n=4,
                ),
                _audit_record(
                    "s8_eicu_gcs_rass_adjusted",
                    support="unsupported",
                    model_run_allowed=False,
                    variant_title="RASS-adjusted GCS",
                ),
            ]
            (mimic4 / "scenario_audit.json").write_text(
                json.dumps(mimic4_records),
                encoding="utf-8",
            )
            (mimic3 / "scenario_audit.json").write_text(
                json.dumps(
                    [
                        _audit_record(
                            "s3_unsedated_only",
                            support="unsupported",
                            model_run_allowed=False,
                            variant_title="Observed unsedated subgroup",
                        )
                    ]
                ),
                encoding="utf-8",
            )
            (eicu / "scenario_audit.json").write_text(
                json.dumps(eicu_records),
                encoding="utf-8",
            )

            for dataset in (mimic4, eicu):
                scenario_id = (
                    "s1_baseline_last"
                    if dataset == mimic4
                    else "s8_eicu_gcs_unadjusted"
                )
                (dataset / f"{scenario_id}.json").write_text(
                    json.dumps({"development": _metrics_fixture()}),
                    encoding="utf-8",
                )
                np.savez_compressed(
                    dataset / f"{scenario_id}_predictions.npz",
                    labels=np.array([0, 0, 1, 2], dtype=int),
                )

            reporting.create_sensitivity_table(root, tables)
            table = pd.read_csv(
                tables / "table_s3_sensitivity.csv",
                keep_default_na=False,
            )
            self.assertEqual(len(table), 6)
            self.assertEqual(
                set(table["Support"]),
                {"supported", "outcome_only", "unsupported"},
            )
            supported = table[table["Support"] == "supported"]
            self.assertTrue(supported["Model run allowed"].all())
            self.assertEqual(supported["n"].astype(int).tolist(), [4, 4])
            self.assertEqual(supported["Grade 0/1/2"].tolist(), ["2/1/1", "2/1/1"])
            excluded = table[table["Support"] != "supported"]
            self.assertFalse(excluded["Model run allowed"].astype(bool).any())
            self.assertEqual(
                excluded["n"].astype(str).tolist(),
                ["NA", "NA", "NA", "NA"],
            )
            self.assertTrue(
                (tables / "table_s3_sensitivity.md").exists()
            )


class CalibrationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.array([0, 1, 2]), 10)
        probabilities = np.tile(np.array([0.8, 0.1, 0.1]), (30, 1))
        self.probabilities = probabilities

    def test_each_grade_and_cumulative_target_is_explicit(self) -> None:
        expected_predicted = {
            "grade_0": 0.8,
            "grade_1": 0.1,
            "grade_2": 0.1,
            "cumulative_any": 0.2,
            "cumulative_severe": 0.1,
        }
        expected_observed = {
            "grade_0": 1 / 3,
            "grade_1": 1 / 3,
            "grade_2": 1 / 3,
            "cumulative_any": 2 / 3,
            "cumulative_severe": 1 / 3,
        }
        for target_name, observed_value in expected_observed.items():
            with self.subTest(target=target_name):
                expected, observed = reporting.calibration_points(
                    self.labels,
                    self.probabilities,
                    bins=1,
                    target=target_name,
                )
                self.assertEqual(len(expected), 1)
                np.testing.assert_allclose(
                    expected,
                    [expected_predicted[target_name]],
                )
                np.testing.assert_allclose(observed, [observed_value])

    def test_repeated_calibration_uses_the_same_explicit_target(self) -> None:
        expected, observed = reporting.calibration_points_repeated(
            self.labels,
            [self.probabilities, self.probabilities],
            bins=1,
            target="cumulative_any",
        )
        np.testing.assert_allclose(expected, [0.2])
        np.testing.assert_allclose(observed, [2 / 3])

    def test_unknown_calibration_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reporting.calibration_points(
                self.labels,
                self.probabilities,
                target="unlabeled_any",
            )


class FigureNumberingAndArchitectureTests(unittest.TestCase):
    def test_table1_uses_the_same_three_cohort_columns_without_stitching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            tables = root / "tables"
            tables.mkdir()
            for cohort in reporting.COHORTS:
                directory = processed / cohort
                directory.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "stay_id": [1, 2, 3],
                        "age": [50.0, 60.0, 70.0],
                        "sex_male": [0, 1, 1],
                        "outcome": [0, 1, 2],
                    }
                ).to_csv(directory / "cohort.csv", index=False)

            with (
                patch.object(reporting, "PROCESSED", processed),
                patch.object(reporting, "TABLES", tables),
            ):
                reporting.create_table1()

            frame = pd.read_csv(
                tables / "table1a_cohorts.csv",
                encoding="utf-8-sig",
            )
            expected = [
                "Characteristic",
                *(
                    reporting.COHORT_LABELS[cohort]
                    for cohort in reporting.COHORTS
                ),
            ]
            self.assertEqual(frame.columns.tolist(), expected)
            self.assertEqual(len(frame.columns), 4)
            self.assertFalse(frame["Characteristic"].duplicated().any())

    def test_dca_uses_only_the_prespecified_ten_to_forty_percent_range(self) -> None:
        self.assertEqual(reporting.DCA_THRESHOLD_MIN, 0.10)
        self.assertEqual(reporting.DCA_THRESHOLD_MAX, 0.40)
        self.assertEqual(len(reporting.DCA_THRESHOLDS), 61)
        self.assertEqual(reporting.DCA_THRESHOLDS[0], 0.10)
        self.assertEqual(reporting.DCA_THRESHOLDS[-1], 0.40)
        self.assertTrue(
            np.all(
                (reporting.DCA_THRESHOLDS >= 0.10)
                & (reporting.DCA_THRESHOLDS <= 0.40)
            )
        )

    def test_figure_stems_follow_the_outline(self) -> None:
        source = Path(reporting.__file__).read_text(encoding="utf-8")
        self.assertEqual(
            reporting.FIGURE3_STEM,
            "figure3_model_performance",
        )
        self.assertEqual(
            reporting.FIGURE4_DCA_STEM,
            "figure4_decision_curve",
        )
        self.assertEqual(
            reporting.FIGURE4_CALIBRATION_STEM,
            "figure4_calibration",
        )
        self.assertEqual(
            reporting.FIGURE6_SUBGROUP_STEM,
            "figure6_subgroups",
        )
        self.assertNotIn("figure5_decision_curve", source)
        self.assertNotIn("figure7_risk_trajectories", source)

    def test_figure3_combines_performance_calibration_and_transport(self) -> None:
        source = Path(reporting.__file__).read_text(encoding="utf-8")
        start = source.index("def create_model_performance_figure")
        end = source.index("def create_model_architecture_figure")
        function_source = source[start:end]
        self.assertIn("Development discrimination", function_source)
        self.assertIn("Development calibration", function_source)
        self.assertIn("External transport discrimination", function_source)
        self.assertIn(
            'target="cumulative_any"',
            function_source,
        )
        self.assertIn("gcs_slope", reporting.FIGURE3_CALIBRATION_MODELS)

    def test_architecture_uses_fourteen_channels_and_two_cumulative_branches(
        self,
    ) -> None:
        self.assertIn("14 channels", reporting.ARCHITECTURE_SEQUENCE_TEXT)
        self.assertIn(
            "two cumulative-logit branches",
            reporting.ARCHITECTURE_ORDINAL_TEXT,
        )
        source = Path(reporting.__file__).read_text(encoding="utf-8")
        self.assertNotIn("8 channels", source)
        self.assertNotIn("global cut points", source)

    def test_gcs_slope_is_a_main_comparator_with_display_metadata(self) -> None:
        self.assertIn("gcs_slope", reporting.MODELS)
        self.assertIn("gcs_slope", reporting.MODEL_LABELS)
        self.assertIn("gcs_slope", reporting.COLORS)
        self.assertNotIn("gcs_slope", reporting.SENSITIVITY_MODELS)
        self.assertEqual(len(set(reporting.COLORS.values())), len(reporting.COLORS))

    def test_laplace_control_labels_match_the_implemented_input_noise(self) -> None:
        label = reporting.MODEL_LABELS["dggru_laplace"].lower()
        self.assertIn("generic laplace input noise", label)
        self.assertIn("0.01", label)
        self.assertNotIn("smoothness", label)
        self.assertNotIn("total-variation", label)
        self.assertEqual(
            reporting.LANDMARK_SENSITIVITY_MODELS["dggru_laplace"],
            "Laplace input noise",
        )


if __name__ == "__main__":
    unittest.main()
