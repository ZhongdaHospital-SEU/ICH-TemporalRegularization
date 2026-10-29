"""Disjoint tests for the prespecified outline section 2.8 analyses."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic.common import CHANNELS
from src.ich_dynamic.sensitivity_analysis import (
    SCENARIOS,
    UnsupportedScenarioError,
    assess_scenario_support,
    continuous_sedation_windows,
    effective_include_intubation,
    estimate_censoring_weights,
    matched_landmark_predictor_columns,
    prepare_scenario,
    scenario_frame,
    scenario_registry_manifest,
)
from work import run_sensitivity_frozen as frozen


def cohort(
    stay_id: int = 1,
    *,
    outcome: float = np.nan,
    icu_out_hour: float = 80.0,
    death: bool = False,
    death_hour: float = 48.0,
    intubation: float = 0.0,
) -> dict:
    return {
        "stay_id": stay_id,
        "subject_key": f"subject-{stay_id}",
        "outcome": outcome,
        "icu_out_hour": icu_out_hour,
        "death_within_72h": death,
        "death_hour_from_icu": death_hour if death else np.nan,
        "recorded_intubation_24_72h": intubation,
        "verified_unplanned_intubation_24_72h": np.nan,
        "age": 66.0,
        "sex_male": 1.0,
        "race_white": 1.0,
        "emergency_admission": 1.0,
        "min_rass_24h": 0.0,
        "any_deep_sedation_24h": 0.0,
    }


def gcs_rows(stay_id: int, values: list[tuple[float, float]]) -> list[dict]:
    return [
        {"stay_id": stay_id, "time_hour": hour, "gcs": value, "ett_verbal": 0.0}
        for hour, value in values
    ]


def _write_support_plan_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    project = root / "project"
    processed = project / "data" / "processed"
    config = project / "configs" / "study.json"
    results = project / "results"
    for relative in frozen.RUNTIME_SOURCE_PATHS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}\n", encoding="utf-8")
    for dataset in frozen.DATASET_NAMES:
        dataset_root = processed / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([cohort(1), cohort(2)]).to_csv(
            dataset_root / "cohort_pre_filter.csv",
            index=False,
        )
        pd.DataFrame(
            gcs_rows(
                1,
                [(2, 15), (25, 12), (26, 12), (65, 12)],
            )
            + gcs_rows(
                2,
                [(2, 15), (25, 15), (26, 15), (65, 15)],
            )
        ).to_csv(dataset_root / "gcs_events.csv", index=False)
        pd.DataFrame(
            {
                "stay_id": [1, 1],
                "time_hour": [25.0, 26.0],
                "rass": [-4.0, -4.0],
            }
        ).to_csv(dataset_root / "rass_events.csv", index=False)
    return project, processed, config, results


class RegistryTests(unittest.TestCase):
    def test_registry_has_exactly_eight_outline_analyses(self) -> None:
        manifest = scenario_registry_manifest()
        self.assertEqual(manifest["outline_section"], "2.8")
        self.assertEqual(manifest["analysis_count"], 8)
        self.assertEqual(
            [analysis["analysis_id"] for analysis in manifest["analyses"]],
            [str(number) for number in range(1, 9)],
        )

    def test_required_variants_are_present(self) -> None:
        ids = {scenario.scenario_id for scenario in SCENARIOS}
        self.assertTrue(
            {
                "s1_baseline_max",
                "s1_baseline_last",
                "s2_persistence_1h",
                "s2_persistence_2h",
                "s3_exclude_continuous_sedation",
                "s3_unsedated_only",
                "s4_censor_exclude",
                "s4_censor_grade0",
                "s4_censor_ipw",
                "s5_with_intubation",
                "s5_without_intubation",
                "s6_landmark_6h",
                "s6_landmark_24h",
                "s6_landmark_48h",
                "s7_exclude_death72h",
                "s8_eicu_gcs_unadjusted",
                "s8_eicu_gcs_rass_adjusted",
            }.issubset(ids)
        )


class EndpointReconstructionTests(unittest.TestCase):
    def test_pre_landmark_death_is_never_reconstructed_as_grade_two(self) -> None:
        prefilter = pd.DataFrame(
            [
                cohort(1, death=True, death_hour=12.0, icu_out_hour=12.0),
                cohort(2, outcome=0.0, icu_out_hour=80.0),
            ]
        )
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15)])
            + gcs_rows(2, [(2, 15), (25, 15), (26, 15), (65, 15)])
        )
        frame = scenario_frame("s1_baseline_max", prefilter, gcs)
        self.assertEqual(frame["stay_id"].tolist(), [2])
        self.assertTrue(frame["landmark_alive"].astype(bool).all())

    def test_pre_landmark_icu_transfer_is_never_reconstructed_as_grade_two(
        self,
    ) -> None:
        prefilter = pd.DataFrame(
            [
                cohort(
                    1,
                    death=True,
                    death_hour=40.0,
                    icu_out_hour=20.0,
                ),
                cohort(2, outcome=0.0, icu_out_hour=80.0),
            ]
        )
        gcs = pd.DataFrame(
            gcs_rows(2, [(2, 15), (25, 15), (26, 15), (65, 15)])
        )
        frame = scenario_frame("s1_baseline_max", prefilter, gcs)
        self.assertEqual(frame["stay_id"].tolist(), [2])
        self.assertTrue(frame["in_icu_at_landmark"].astype(bool).all())

    def test_max_and_last_baseline_are_distinct(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(
            gcs_rows(1, [(1, 15), (23, 12), (25, 12), (26, 12), (65, 12)])
        )
        maximum = scenario_frame("s1_baseline_max", prefilter, gcs)
        last = scenario_frame("s1_baseline_last", prefilter, gcs)
        self.assertEqual(int(maximum.loc[0, "outcome"]), 1)
        self.assertEqual(int(last.loc[0, "outcome"]), 0)
        self.assertEqual(float(maximum.loc[0, "baseline_gcs_max"]), 15.0)
        self.assertEqual(float(last.loc[0, "baseline_gcs_last"]), 12.0)

    def test_one_hour_and_two_hour_persistence_differ(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 12), (65, 15)]))
        one_hour = scenario_frame("s2_persistence_1h", prefilter, gcs)
        two_hour = scenario_frame("s2_persistence_2h", prefilter, gcs)
        self.assertEqual(int(one_hour.loc[0, "outcome"]), 1)
        self.assertEqual(int(two_hour.loc[0, "outcome"]), 0)

    def test_two_hour_persistence_requires_adjacent_hourly_bins(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 12), (26, 12), (65, 15)])
        )
        frame = scenario_frame("s2_persistence_2h", prefilter, gcs)
        self.assertEqual(int(frame.loc[0, "outcome"]), 1)
        separated = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 12), (27, 12), (65, 15)])
        )
        frame = scenario_frame("s2_persistence_2h", prefilter, separated)
        self.assertEqual(int(frame.loc[0, "outcome"]), 0)

    def test_primary_sensitivity_endpoint_does_not_require_gcs_after_hour_60(
        self,
    ) -> None:
        prefilter = pd.DataFrame([cohort(icu_out_hour=80.0)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 12), (26, 12)]))
        frame = scenario_frame("s1_baseline_max", prefilter, gcs)
        self.assertEqual(int(frame.loc[0, "outcome"]), 1)

    def test_six_hour_landmark_ascertains_deterioration_from_hour_six(self) -> None:
        prefilter = pd.DataFrame([cohort(icu_out_hour=80.0)])
        gcs = pd.DataFrame(
            gcs_rows(1, [(1, 15), (7, 12), (8, 12)])
        )
        six_hour = scenario_frame("s6_landmark_6h", prefilter, gcs)
        twenty_four_hour = scenario_frame("s6_landmark_24h", prefilter, gcs)
        self.assertEqual(int(six_hour.loc[0, "outcome"]), 1)
        self.assertTrue(twenty_four_hour.empty)

    def test_recorded_intubation_is_not_substituted_for_unplanned(self) -> None:
        prefilter = pd.DataFrame([cohort(intubation=1.0)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        with_intubation = assess_scenario_support(
            "s5_with_intubation",
            "mimic4",
            prefilter=prefilter,
            gcs=gcs,
        )
        without_intubation = scenario_frame("s5_without_intubation", prefilter, gcs)
        self.assertEqual(with_intubation.support, "unsupported")
        self.assertFalse(with_intubation.model_run_allowed)
        self.assertIn("planned versus unplanned", with_intubation.reason)
        self.assertEqual(int(without_intubation.loc[0, "outcome"]), 0)

    def test_duplicate_values_within_an_hour_use_the_hourly_median(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(
            gcs_rows(
                1,
                [(2, 15), (24.1, 12), (24.4, 15), (25.1, 15), (65, 15)],
            )
        )
        frame = scenario_frame(
            "s5_without_intubation",
            prefilter,
            gcs,
            dataset="mimic4",
        )
        self.assertEqual(float(frame.loc[0, "outcome"]), 0.0)

    def test_death_is_excluded_only_in_death_sensitivity(self) -> None:
        prefilter = pd.DataFrame([cohort(death=True, icu_out_hour=50.0)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 14), (40, 13)]))
        primary = scenario_frame("s1_baseline_max", prefilter, gcs)
        excluded = scenario_frame("s7_exclude_death72h", prefilter, gcs)
        self.assertEqual(len(primary), 1)
        self.assertEqual(int(primary.loc[0, "outcome"]), 2)
        self.assertEqual(len(excluded), 0)

    def test_csv_string_false_is_not_treated_as_death(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        prefilter["death_within_72h"] = "False"
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        result = scenario_frame("s1_baseline_max", prefilter, gcs)
        self.assertEqual(int(result.loc[0, "outcome"]), 0)


class CensoringTests(unittest.TestCase):
    def test_grade_zero_retains_early_transfer_and_known_outcome(self) -> None:
        prefilter = pd.DataFrame(
            [
                cohort(1, outcome=1.0, icu_out_hour=40.0),
                cohort(2, outcome=np.nan, icu_out_hour=40.0),
            ]
        )
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 12), (26, 12)])
            + gcs_rows(2, [(2, 15), (25, 15), (26, 15)])
        )
        result = scenario_frame("s4_censor_grade0", prefilter, gcs)
        labels = dict(zip(result["stay_id"], result["outcome"]))
        self.assertEqual(labels[1], 0)
        self.assertEqual(labels[2], 0)
        self.assertTrue(result["analysis_weight"].eq(1.0).all())

    def test_death_after_live_icu_discharge_is_grade_two_in_primary(self) -> None:
        prefilter = pd.DataFrame(
            [cohort(1, outcome=2.0, icu_out_hour=30.0, death=True, death_hour=40.0)]
        )
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 12), (26, 12)])
        )
        primary = scenario_frame("s1_baseline_max", prefilter, gcs)
        grade_zero = scenario_frame("s4_censor_grade0", prefilter, gcs)
        # A death within 72 h of ICU admission is grade 2 even when it occurs
        # after a live ICU discharge; the in-ICU flag remains false because it
        # describes completion of the ICU observation window, not the endpoint.
        self.assertEqual(len(primary), 1)
        self.assertEqual(int(primary.loc[0, "outcome"]), 2)
        self.assertFalse(bool(primary.loc[0, "death_in_icu_within_72h"]))
        self.assertTrue(bool(primary.loc[0, "early_icu_transfer"]))
        self.assertEqual(int(grade_zero.loc[0, "outcome"]), 0)
        self.assertTrue(bool(grade_zero.loc[0, "early_icu_transfer"]))
        self.assertFalse(bool(grade_zero.loc[0, "death_in_icu_within_72h"]))

    def test_ipw_is_runnable_after_weighted_trainer_contract_is_added(self) -> None:
        prefilter = pd.DataFrame(
            [
                cohort(
                    index + 1,
                    outcome=float(index % 2) if index % 2 == 0 else np.nan,
                    icu_out_hour=80.0 if index % 2 == 0 else 40.0,
                )
                for index in range(20)
            ]
        )
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 15), (65, 15)])
        )
        status = assess_scenario_support(
            "s4_censor_ipw",
            "mimic4",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(status.support, "supported")
        self.assertTrue(status.model_run_allowed)

    def test_ipw_without_censoring_is_not_reported_as_supported(self) -> None:
        prefilter = pd.DataFrame(
            [
                cohort(index + 1, outcome=0.0, icu_out_hour=80.0)
                for index in range(10)
            ]
        )
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (25, 15), (65, 15)])
        )
        status = assess_scenario_support(
            "s4_censor_ipw",
            "mimic4",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(status.support, "unsupported")
        self.assertFalse(status.model_run_allowed)

    def test_ipw_is_patient_grouped_cross_fitted_and_drops_all_missing_covariates(
        self,
    ) -> None:
        rows = []
        for index in range(20):
            observed = index % 2 == 0
            rows.append(
                {
                    "stay_id": index + 1,
                    "subject_key": f"patient-{index + 1}",
                    "followup_complete": observed,
                    "outcome": float(index % 3) if observed else np.nan,
                    "age": 45.0 + index,
                    "sex_male": float(index % 2),
                    "race_white": float((index // 2) % 2),
                    "emergency_admission": np.nan,
                    "baseline_gcs_max": 8.0 + (index % 7),
                    "min_rass_24h": float(-(index % 4)),
                    "any_deep_sedation_24h": float(index % 3 == 0),
                }
            )
        weighted = estimate_censoring_weights(pd.DataFrame(rows))
        self.assertEqual(len(weighted), 10)
        self.assertTrue(np.isfinite(weighted["analysis_weight"]).all())
        self.assertTrue((weighted["analysis_weight"] > 0).all())
        self.assertTrue(weighted["ipw_crossfit_fold"].between(0, 4).all())
        audit = weighted.attrs["ipw_audit"]
        self.assertEqual(
            audit["excluded_all_missing_covariates"],
            ["emergency_admission"],
        )
        self.assertNotIn("emergency_admission", audit["covariates_used"])
        self.assertEqual(audit["crossfit_splits"], 5)
        self.assertGreater(audit["effective_sample_size"], 0)


class SupportAndMetadataTests(unittest.TestCase):
    def test_support_plan_binds_runtime_config_and_all_data_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, processed, config, results = _write_support_plan_fixture(
                root
            )
            plan, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            self.assertTrue(path.is_file())
            self.assertFalse(list(path.parent.glob("*.tmp")))
            self.assertEqual(
                set(plan["binding"]["runtime_sources"]),
                set(frozen.RUNTIME_SOURCE_PATHS),
            )
            self.assertEqual(
                set(plan["binding"]["data"]),
                {
                    f"{dataset}/{filename}"
                    for dataset in frozen.DATASET_NAMES
                    for filename in frozen.DATA_LINEAGE_FILES
                },
            )
            self.assertTrue(plan["generated_at_utc"].endswith("Z"))
            loaded = frozen.load_support_plan(
                path=path,
                project_root=project,
                processed_root=processed,
                config_path=config,
            )
            self.assertEqual(loaded, plan)

    def test_support_plan_rejects_runtime_source_and_data_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, processed, config, results = _write_support_plan_fixture(
                root
            )
            _, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            runtime = project / "src" / "ich_dynamic" / "models.py"
            runtime.write_text("# changed runtime\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                frozen.load_support_plan(
                    path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

            runtime.write_text(
                "# src/ich_dynamic/models.py\n", encoding="utf-8"
            )
            data_path = processed / "mimic4" / "gcs_events.csv"
            with data_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                frozen.load_support_plan(
                    path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

    def test_require_supported_never_uses_a_stale_plan_result(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, processed, config, results = _write_support_plan_fixture(
                root
            )
            _, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            status = frozen.require_supported(
                "mimic4",
                "s1_baseline_max",
                plan_path=path,
                project_root=project,
                processed_root=processed,
                config_path=config,
            )
            self.assertTrue(status.model_run_allowed)

            config.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                frozen.require_supported(
                    "mimic4",
                    "s1_baseline_max",
                    plan_path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

    def test_support_plan_rejects_old_or_incomplete_binding(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, processed, config, results = _write_support_plan_fixture(
                root
            )
            _, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            plan = json.loads(path.read_text(encoding="utf-8"))
            plan["generated_at_utc"] = "2000-01-01T00:00:00Z"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "older"):
                frozen.load_support_plan(
                    path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

            _, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            plan = json.loads(path.read_text(encoding="utf-8"))
            del plan["binding"]["runtime_sources"][
                "src/ich_dynamic/common.py"
            ]
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime source"):
                frozen.load_support_plan(
                    path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

            _, path = frozen.write_support_plan(
                project_root=project,
                processed_root=processed,
                config_path=config,
                results_root=results,
            )
            plan = json.loads(path.read_text(encoding="utf-8"))
            del plan["binding"]["data"]["eicu/rass_events.csv"]
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data"):
                frozen.load_support_plan(
                    path=path,
                    project_root=project,
                    processed_root=processed,
                    config_path=config,
                )

    def test_landmark_curves_use_the_same_six_hour_predictor_set(
        self,
    ) -> None:
        cohort = pd.read_csv(
            Path("data/processed/landmark_inputs/mimic4/landmark_24h/cohort.csv")
        )
        columns_24 = matched_landmark_predictor_columns(
            "mimic4",
            24,
            cohort.columns,
        )
        columns_48 = matched_landmark_predictor_columns(
            "mimic4",
            48,
            pd.read_csv(
                Path(
                    "data/processed/landmark_inputs/mimic4/"
                    "landmark_48h/cohort.csv"
                )
            ).columns,
        )
        self.assertEqual(len(columns_24), 92)
        self.assertEqual(len(columns_48), 92)
        self.assertIn("monitoring_intensity_24h", columns_24)
        self.assertIn("monitoring_intensity_48h", columns_48)
        self.assertFalse(any(column.startswith("lab_") for column in columns_24))
        self.assertFalse(any(column.startswith("lab_") for column in columns_48))

    def test_sedation_requires_event_level_rass(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        status = assess_scenario_support(
            "s3_exclude_continuous_sedation",
            "mimic4",
            prefilter=prefilter,
            gcs=gcs,
            rass_events=None,
        )
        self.assertEqual(status.support, "unsupported")
        self.assertIn("RASS", status.reason)
        with self.assertRaises(UnsupportedScenarioError):
            scenario_frame(
                "s3_exclude_continuous_sedation",
                prefilter,
                gcs,
                rass_events=None,
            )

    def test_continuous_sedation_window_is_explicit(self) -> None:
        rass = pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "time_hour": [25.0, 26.0, 30.0],
                "rass": [-3.0, -4.0, -2.0],
            }
        )
        windows = continuous_sedation_windows(rass)
        self.assertEqual(len(windows), 1)
        self.assertEqual(int(windows.loc[0, "stay_id"]), 1)
        self.assertEqual(float(windows.loc[0, "sedation_start_hour"]), 25.0)
        self.assertEqual(float(windows.loc[0, "sedation_end_hour"]), 26.0)

    def test_eicu_rass_adjustment_is_marked_unsupported(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        status = assess_scenario_support(
            "s8_eicu_gcs_rass_adjusted",
            "eicu",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(status.support, "unsupported")
        self.assertFalse(status.model_run_allowed)
        self.assertIn("not a valid substitute", status.reason)
        with self.assertRaises(UnsupportedScenarioError):
            scenario_frame(
                "s8_eicu_gcs_rass_adjusted",
                prefilter,
                gcs,
                dataset="eicu",
            )

    def test_explicit_adjusted_gcs_is_used_when_complete(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(
            gcs_rows(1, [(2, 15), (23, 12), (25, 12), (65, 12)])
        )
        gcs["gcs_rass_adjusted"] = [15.0, 13.0, 13.0, 13.0]
        prepared = prepare_scenario(
            "s8_eicu_gcs_rass_adjusted",
            prefilter,
            gcs,
            dataset="eicu",
        )
        self.assertEqual(prepared.status.support, "supported")
        self.assertTrue(prepared.rass_adjustment_applied)
        self.assertEqual(float(prepared.frame.loc[0, "baseline_gcs_max"]), 15.0)

    def test_non_24h_landmarks_are_not_silently_reused(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (10, 12), (25, 12), (65, 12)]))
        with TemporaryDirectory() as temporary, patch(
            "src.ich_dynamic.sensitivity_analysis.LANDMARK_INPUT_ROOT",
            Path(temporary),
        ):
            status = assess_scenario_support(
                "s6_landmark_6h",
                "mimic4",
                prefilter=prefilter,
                gcs=gcs,
            )
        self.assertEqual(status.support, "outcome_only")
        self.assertFalse(status.model_run_allowed)
        self.assertIn("build_landmark_inputs", status.reason)

    def test_landmark_support_requires_all_rebuilt_artifacts(self) -> None:
        prefilter = pd.DataFrame([cohort()])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "src.ich_dynamic.sensitivity_analysis.LANDMARK_INPUT_ROOT",
                root,
            ):
                missing = assess_scenario_support(
                    "s6_landmark_48h",
                    "mimic4",
                    prefilter=prefilter,
                    gcs=gcs,
                )
                self.assertEqual(missing.support, "outcome_only")
                landmark = root / "mimic4" / "landmark_48h"
                landmark.mkdir(parents=True)
                for filename in ("cohort.csv", "sequences.npz", "audit.json"):
                    (landmark / filename).write_text("", encoding="utf-8")
                invalid = assess_scenario_support(
                    "s6_landmark_48h",
                    "mimic4",
                    prefilter=prefilter,
                    gcs=gcs,
                )
                self.assertEqual(invalid.support, "outcome_only")
                self.assertFalse(invalid.model_run_allowed)

                def sha256(path: Path) -> str:
                    return hashlib.sha256(path.read_bytes()).hexdigest()

                for landmark_hours in (6, 48):
                    directory = (
                        root
                        / "mimic4"
                        / f"landmark_{landmark_hours}h"
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    cohort_path = directory / "cohort.csv"
                    sequence_path = directory / "sequences.npz"
                    pd.DataFrame({"stay_id": [1], "outcome": [0]}).to_csv(
                        cohort_path, index=False
                    )
                    np.savez_compressed(
                        sequence_path,
                        values=np.zeros(
                            (1, landmark_hours, len(CHANNELS)),
                            dtype=np.float32,
                        ),
                        masks=np.zeros(
                            (1, landmark_hours, len(CHANNELS)),
                            dtype=np.float32,
                        ),
                        stay_id=np.asarray([1], dtype=np.int64),
                        outcome=np.asarray([0], dtype=np.int64),
                        channels=np.asarray(CHANNELS),
                        landmark_hours=np.asarray(
                            landmark_hours, dtype=np.int64
                        ),
                    )
                    audit = {
                        "schema_version": 1,
                        "dataset": "mimic4",
                        "landmark_hours": landmark_hours,
                        "eligibility": {"analysis_n": 1},
                        "sequence": {
                            "shape": [1, landmark_hours, len(CHANNELS)],
                            "channels": list(CHANNELS),
                        },
                        "artifact_hashes": {
                            "cohort.csv": sha256(cohort_path),
                            "sequences.npz": sha256(sequence_path),
                            "hash_algorithm": "sha256",
                        },
                    }
                    (directory / "audit.json").write_text(
                        json.dumps(audit), encoding="utf-8"
                    )
                available = assess_scenario_support(
                    "s6_landmark_48h",
                    "mimic4",
                    prefilter=prefilter,
                    gcs=gcs,
                )
        self.assertEqual(available.support, "supported")
        self.assertTrue(available.model_run_allowed)

    def test_eicu_composite_with_missing_intubation_is_unsupported(self) -> None:
        prefilter = pd.DataFrame([cohort(intubation=np.nan)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        status = assess_scenario_support(
            "s5_with_intubation",
            "eicu",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(status.support, "unsupported")
        self.assertIn("intubation", status.reason)

    def test_mimic3_composite_is_unsupported_despite_a_procedure_flag(self) -> None:
        prefilter = pd.DataFrame([cohort(intubation=1.0)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (26, 15)]))
        status = assess_scenario_support(
            "s5_with_intubation",
            "mimic3",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(status.support, "unsupported")
        self.assertFalse(status.model_run_allowed)
        self.assertIn("planned versus unplanned", status.reason)

    def test_eicu_unadjusted_endpoint_is_not_blocked_by_intubation(self) -> None:
        prefilter = pd.DataFrame([cohort(intubation=np.nan)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        for scenario in ("s1_baseline_max", "s6_landmark_24h"):
            with self.subTest(scenario=scenario):
                status = assess_scenario_support(
                    scenario,
                    "eicu",
                    prefilter=prefilter,
                    gcs=gcs,
                )
                self.assertEqual(status.support, "supported")
                self.assertTrue(status.model_run_allowed)
        unadjusted = assess_scenario_support(
            "s8_eicu_gcs_unadjusted",
            "eicu",
            prefilter=prefilter,
            gcs=gcs,
        )
        self.assertEqual(unadjusted.support, "supported")
        self.assertFalse(
            effective_include_intubation("s8_eicu_gcs_unadjusted", "eicu")
        )

    def test_eicu_prepared_endpoint_is_explicitly_gcs_or_death(self) -> None:
        prefilter = pd.DataFrame([cohort(intubation=np.nan)])
        gcs = pd.DataFrame(gcs_rows(1, [(2, 15), (25, 15), (65, 15)]))
        prepared = prepare_scenario(
            "s1_baseline_max",
            prefilter,
            gcs,
            dataset="eicu",
        )
        self.assertEqual(prepared.status.support, "supported")
        self.assertFalse(
            bool(prepared.frame.loc[0, "intubation_in_endpoint"])
        )
        self.assertEqual(
            prepared.frame.loc[0, "endpoint_label"],
            "GCS deterioration or death",
        )


if __name__ == "__main__":
    unittest.main()
