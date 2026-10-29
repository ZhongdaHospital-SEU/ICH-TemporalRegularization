"""Disjoint contract tests for outline-aligned reporting code."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.ich_dynamic import make_manuscript
from src.ich_dynamic import make_tripod
from work import build_outline_compliance_matrix as compliance


METRIC_NAMES = (
    "macro_auc",
    "auc_grade_0",
    "auc_grade_1",
    "auc_grade_2",
    "auc_any_deterioration",
    "auc_severe_deterioration",
    "multiclass_brier",
    "calibration_slope_any",
    "expected_calibration_error",
)


def metric_block(offset: float = 0.0) -> dict:
    point = {"n": 100}
    point.update(
        {
            name: 0.5 + offset + index / 100
            for index, name in enumerate(METRIC_NAMES)
        }
    )
    return {
        "point": point,
        "ci": {
            name: [point[name] - 0.02, point[name] + 0.02]
            for name in METRIC_NAMES
        },
    }


def outline_metrics() -> dict:
    block = {
        model: metric_block(index / 1000)
        for index, model in enumerate(make_manuscript.OUTLINE_MODEL_ORDER)
    }
    block["paired_differences_vs_dggru_tr"] = {
        model: [-0.01, 0.01]
        for model in make_manuscript.OUTLINE_MODEL_ORDER
        if model != "dggru_tr"
    }
    return {
        "development": block,
        "external": {
            "mimic3": block.copy(),
            "eicu": block.copy(),
        },
    }


class OutlineModelContractTests(unittest.TestCase):
    def test_exact_comparison_set_uses_current_model_identifiers(self) -> None:
        self.assertEqual(
            set(make_manuscript.OUTLINE_MODEL_ORDER),
            {
                "xgb_static",
                "xgb_summary",
                "gcs_slope",
                "gru",
                "grud",
                "transformer",
                "dggru",
                "dggru_tr",
                "nested_stack",
                "dggru_tr_treverse",
                "dggru_tr_tpermute",
                "dggru_laplace",
            },
        )
        self.assertEqual(
            set(make_manuscript.OUTLINE_BASELINE_MODELS),
            {
                "xgb_static",
                "xgb_summary",
                "gru",
                "grud",
                "transformer",
            },
        )
        self.assertNotIn("dggru_smooth", make_manuscript.MODEL_LABELS)
        self.assertNotIn("dggru_tr_anti", make_manuscript.MODEL_LABELS)

    def test_manifest_validation_rejects_a_missing_outline_model(self) -> None:
        metrics = outline_metrics()
        manifest = {"models": list(make_manuscript.OUTLINE_MODEL_ORDER)}
        ordered = make_manuscript.outline_model_names(
            metrics["development"],
            manifest,
        )
        self.assertEqual(ordered, make_manuscript.OUTLINE_MODEL_ORDER)

        broken = metrics["development"].copy()
        broken.pop("transformer")
        with self.assertRaises(ValueError):
            make_manuscript.outline_model_names(broken, manifest)

    def test_table3_contains_all_models_for_all_three_cohorts(self) -> None:
        table = make_manuscript.table3_comparison_frame(
            outline_metrics(),
            make_manuscript.OUTLINE_MODEL_ORDER,
        )
        self.assertEqual(
            len(table),
            3 * len(make_manuscript.OUTLINE_MODEL_ORDER),
        )
        self.assertEqual(
            set(table["Model"]),
            {
                make_manuscript.model_label(model)
                for model in make_manuscript.OUTLINE_MODEL_ORDER
            },
        )
        self.assertIn(
            "DG-GRU + TR minus comparator: delta macro AUC (95% CI)",
            table.columns,
        )

    def test_external_cohort_missing_a_model_is_rejected(self) -> None:
        metrics = outline_metrics()
        metrics["external"]["eicu"].pop("dggru_tr_treverse")
        with self.assertRaises(ValueError):
            make_manuscript.table3_comparison_frame(
                metrics,
                make_manuscript.OUTLINE_MODEL_ORDER,
            )

    def test_tripod_dependency_gate_rejects_incomplete_external_metrics(self) -> None:
        metrics = outline_metrics()
        metrics["external"]["mimic3"].pop("grud")
        manifest = {
            "models": list(make_manuscript.OUTLINE_MODEL_ORDER),
            "model_seeds": [0, 1, 2, 3, 4],
        }
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            (results / "metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
            with patch.object(make_tripod, "RESULTS", results):
                with self.assertRaises(ValueError):
                    make_tripod.validate_reporting_dependency()

    def test_eight_sensitivity_groups_cover_every_variant(self) -> None:
        self.assertEqual(len(make_manuscript.OUTLINE_SENSITIVITY_ANALYSES), 8)
        self.assertEqual(len(make_manuscript.OUTLINE_SENSITIVITY_SCENARIOS), 17)
        self.assertEqual(
            len(make_manuscript.OUTLINE_SENSITIVITY_SCENARIOS),
            len(set(make_manuscript.OUTLINE_SENSITIVITY_SCENARIOS)),
        )

    def test_trajectory_table_requires_all_three_controls(self) -> None:
        frame = pd.DataFrame(
            {
                "model": [
                    "dggru",
                    "dggru_tr",
                    "dggru_tr_treverse",
                    "dggru_tr_tpermute",
                ],
                "negative_step_fraction_median": [0.1, 0.2, 0.3, 0.4],
                "roughness_median": [0.1, 0.2, 0.3, 0.4],
                "total_variation_median": [0.1, 0.2, 0.3, 0.4],
                "final_risk_auc_any": [0.6, 0.7, 0.6, 0.5],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            tables = results / "tables"
            tables.mkdir()
            frame.to_csv(
                tables / "table5_trajectory_regularization.csv",
                index=False,
            )
            with patch.object(make_manuscript, "RESULTS", results):
                with self.assertRaises(ValueError):
                    make_manuscript.trajectory_frame()

    def test_time_importance_requires_complete_channel_hour_grid(self) -> None:
        channels = [f"channel_{index}" for index in range(14)]
        config = {
            "channels": channels,
            "sequence_hours": 24,
        }
        frame = pd.DataFrame(
            [
                {
                    "channel": channel,
                    "hour": hour,
                    "mean_abs_gradient": 0.1,
                }
                for channel in channels
                for hour in range(24)
                if not (channel == channels[-1] and hour == 23)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            tables = results / "tables"
            tables.mkdir()
            frame.to_csv(
                tables / "table_s_time_dependent_importance.csv",
                index=False,
            )
            with patch.object(make_manuscript, "RESULTS", results):
                with self.assertRaises(ValueError):
                    make_manuscript.time_importance_frame(config)


class ReportingSourceContractTests(unittest.TestCase):
    @staticmethod
    def _source_texts() -> dict[str, str]:
        root = Path(make_manuscript.__file__).parents[2]
        return {
            "manuscript": Path(make_manuscript.__file__).read_text(
                encoding="utf-8"
            ),
            "tripod": (
                root / "src" / "ich_dynamic" / "make_tripod.py"
            ).read_text(encoding="utf-8"),
            "protocol": (
                root / "work" / "build_study_protocol.py"
            ).read_text(encoding="utf-8"),
            "compliance": (
                root / "work" / "build_outline_compliance_matrix.py"
            ).read_text(encoding="utf-8"),
        }

    def test_tripod_interpretation_item_uses_current_figure_set(self) -> None:
        item = next(row for row in make_tripod.ITEMS if row[0] == "15")
        self.assertEqual(
            item[2],
            "Results 3.4 and 3.6; Figures 5 and 6A; Table 5; Tables S9-S10",
        )
        subgroup_item = next(row for row in make_tripod.ITEMS if row[0] == "14")
        self.assertEqual(
            subgroup_item[2],
            "Results 3.6; Figure 6B; Supplementary Table S6",
        )

    def test_outline_figure_registry_uses_real_generator_outputs(self) -> None:
        expected = (
            "figure1_cohort_flow",
            "figure2_model_architecture",
            "figure3_model_performance",
            "figure4_calibration",
            "figure4_decision_curve",
            "figure5_temporal_structure",
            "figure6_interpretation",
            "figure6_subgroups",
        )
        self.assertEqual(tuple(make_manuscript.FIGURES), expected)
        self.assertEqual(
            tuple(
                (spec["number"], spec["panel"], spec["stem"])
                for spec in make_manuscript.FIGURE_SPECS
            ),
            (
                (1, "", "figure1_cohort_flow"),
                (2, "", "figure2_model_architecture"),
                (3, "", "figure3_model_performance"),
                (4, "A", "figure4_calibration"),
                (4, "B", "figure4_decision_curve"),
                (5, "", "figure5_temporal_structure"),
                (6, "A", "figure6_interpretation"),
                (6, "B", "figure6_subgroups"),
            ),
        )
        labels = [
            make_manuscript.figure_label(spec)
            for spec in make_manuscript.FIGURE_SPECS
        ]
        self.assertEqual(
            labels,
            [
                "Figure 1",
                "Figure 2",
                "Figure 3",
                "Figure 4A",
                "Figure 4B",
                "Figure 5",
                "Figure 6A",
                "Figure 6B",
            ],
        )
        root = Path(make_manuscript.__file__).parents[2]
        generator_source = "\n".join(
            (root / "src" / "ich_dynamic" / name).read_text(encoding="utf-8")
            for name in (
                "make_tables_figures.py",
                "export_trajectories.py",
                "explain.py",
            )
        )
        for stem in expected:
            self.assertIn(stem, generator_source)

    def test_missing_figure_outputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                make_manuscript.require_figure_outputs(Path(directory))

    def test_reporting_sources_do_not_use_obsolete_figure_keys(self) -> None:
        paths = (
            Path(make_manuscript.__file__),
            Path(make_manuscript.__file__).with_name("make_tripod.py"),
            Path(make_manuscript.__file__).parents[2]
            / "work"
            / "build_study_protocol.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for stale in (
                "figure5_decision_curve",
                "figure7_risk_trajectories",
                "Figures 6-8",
            ):
                self.assertNotIn(stale, source, f"{stale} remains in {path.name}")

    def test_reporting_sources_do_not_use_obsolete_model_or_channel_text(self) -> None:
        paths = (
            Path(make_manuscript.__file__),
            Path(make_manuscript.__file__).with_name("make_tripod.py"),
            Path(make_manuscript.__file__).parents[2]
            / "work"
            / "build_study_protocol.py",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for stale in (
                "dggru_smooth",
                "dggru_tr_anti",
                "Eight channels",
                "eight channels",
                "16 value-and-mask",
            ):
                self.assertNotIn(stale, source, f"{stale} remains in {path.name}")

    def test_tripod_and_protocol_state_fourteen_channels_and_five_seeds(self) -> None:
        root = Path(make_manuscript.__file__).parents[2]
        tripod = (root / "src" / "ich_dynamic" / "make_tripod.py").read_text(
            encoding="utf-8"
        )
        protocol = (root / "work" / "build_study_protocol.py").read_text(
            encoding="utf-8"
        )
        for source in (tripod, protocol):
            self.assertIn("14", source)
            self.assertIn("five", source.lower())

    def test_reporting_sources_require_two_post_landmark_gcs_hours_for_deaths(self) -> None:
        for name, source in self._source_texts().items():
            normalized = " ".join(source.split()).lower()
            with self.subTest(source=name):
                self.assertIn("two distinct", normalized)
                self.assertIn("24 and 72", normalized)
                self.assertIn("sparse-gcs", normalized)
                self.assertIn("excluded", normalized)
                self.assertNotIn("even when fewer than two", normalized)
                self.assertNotIn("death_only_grade2", normalized)
                self.assertNotIn("sparse_post_gcs_death_grade2", normalized)

    def test_primary_inference_is_hospital_clustered_in_eicu(self) -> None:
        for name, source in self._source_texts().items():
            normalized = " ".join(source.split()).lower()
            with self.subTest(source=name):
                self.assertIn("hospital", normalized)
                self.assertIn("hospital_key", normalized)
                self.assertIn("bootstrap", normalized)
                self.assertIn("primary", normalized)
                self.assertIn("inference", normalized)
                self.assertIn("delong", normalized)
                self.assertIn("unclustered", normalized)
                self.assertIn("secondary", normalized)
        self.assertNotIn(
            "DeLong tests are reported with Holm correction.",
            self._source_texts()["tripod"],
        )

    def test_holm_family_and_seed_aggregated_selection_contract(self) -> None:
        for name, source in self._source_texts().items():
            normalized = " ".join(source.split()).lower()
            with self.subTest(source=name):
                self.assertIn(
                    "fixed primary holm family contains 10 tests",
                    normalized,
                )
                self.assertIn("p024/p039", normalized)
                self.assertIn("xgb_static", normalized)
                self.assertIn("xgb_summary", normalized)
                self.assertIn("gru", normalized)
                self.assertIn("grud", normalized)
                self.assertIn("transformer", normalized)
                self.assertIn("any/severe", normalized)
                self.assertIn(
                    "exploratory_not_in_primary_holm_family",
                    normalized,
                )
                self.assertIn("validation probabilities", normalized)
                self.assertIn("averaged", normalized)
                self.assertIn("lambda", normalized)
                self.assertNotIn("all comparators holm", normalized)
                self.assertNotIn(
                    "holm correction across the protocol-defined comparisons",
                    normalized,
                )

    def test_compliance_matrix_reads_outline_docx_directly(self) -> None:
        source = self._source_texts()["compliance"]
        self.assertIn('OUTLINE_DOCX = ROOT / "outline.docx"', source)
        self.assertIn("Document(OUTLINE_DOCX)", source)
        self.assertIn('"outline_source_format": "docx"', source)
        self.assertNotIn("outline_extracted.txt", source)
        self.assertNotIn("FACTS = _cohort_facts()", source)
        self.assertIn("3,688/1,795/3,601", source)
        self.assertIn("2,851/1,322/2,032", source)

    def test_compliance_matrix_renders_rebuilt_flow_dynamically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory)
            for index, cohort in enumerate(compliance.COHORTS, start=1):
                target = processed / cohort
                target.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "outcome": [0, 1, 2],
                        "death_within_72h": [False, False, True],
                        "min_rass_24h": [1.0, float("nan"), 0.0],
                        "hospital_expire_flag": [0, 0, 1],
                    }
                ).to_csv(target / "cohort.csv", index=False)
                pd.DataFrame(
                    {
                        "reason_code": [
                            "source_ich_icu_stays",
                            "source_common_landmark_candidates",
                            "primary_complete_followup",
                        ],
                        "n": [10 * index, 7 * index, 3 * index],
                    }
                ).to_csv(target / "cohort_flow.csv", index=False)

            with patch.object(compliance, "PROCESSED", processed):
                report = compliance.build_report()

        rows = {row["paragraph_id"]: row for row in report["rows"]}
        self.assertIn("10 -> 7 -> 3", rows["P018"]["evidence"])
        self.assertIn("20 -> 14 -> 6", rows["P018"]["evidence"])
        self.assertIn("30 -> 21 -> 9", rows["P018"]["evidence"])
        self.assertIn("66.7%", rows["P022"]["evidence"])
        self.assertIn("P020 = PARTIAL", rows["P020"]["evidence"])
        self.assertIn("p020_complete = false", rows["P020"]["evidence"])
        self.assertNotIn("sparse-post-GCS death subset", rows["P020"]["evidence"])

    def test_endpoint_preprocessing_and_channel_reporting_contract(self) -> None:
        sources = self._source_texts()
        for name, source in sources.items():
            normalized = " ".join(source.split())
            with self.subTest(source=name):
                self.assertIn("P020", normalized)
                self.assertIn("PARTIAL", normalized)
                self.assertIn("SequencePreprocessor", normalized)
                self.assertIn("causal", normalized.lower())
                self.assertIn("forward fill", normalized.lower())
                self.assertIn("14", normalized)
                self.assertIn("matched", normalized.lower())
                self.assertIn("ablation", normalized.lower())

        tripod = sources["tripod"]
        self.assertNotIn("116/80/73", tripod)
        self.assertNotIn("45/30/32", tripod)
        self.assertNotIn("93.1%, 28.6%, and 38.2%", tripod)
        self.assertNotIn("fully available in eICU", tripod)

        protocol = sources["protocol"]
        self.assertNotIn("death_only_grade2", protocol)
        self.assertNotIn("sparse_post_gcs_death_grade2", protocol)
        self.assertIn("rass_coverage", protocol)
        self.assertNotIn(
            "Composite endpoint with versus without recorded "
            "post-landmark intubation",
            protocol,
        )

    def test_tripod_marks_p020_as_partial(self) -> None:
        item = next(row for row in make_tripod.ITEMS if row[0] == "6")
        self.assertEqual(item[3], "Partially reported")
        self.assertIn("P020 = PARTIAL", item[4])
        self.assertIn("p020_complete = false", item[4])
        self.assertIn("sparse-GCS death is excluded", item[4])

    def test_laplace_control_is_generic_observed_input_noise(self) -> None:
        root = Path(make_manuscript.__file__).parents[2]
        sources = {
            **self._source_texts(),
            "figures": (
                root / "src" / "ich_dynamic" / "make_tables_figures.py"
            ).read_text(encoding="utf-8"),
            "readme": (root / "README.md").read_text(encoding="utf-8"),
        }
        for name, source in sources.items():
            normalized = " ".join(source.split()).lower()
            with self.subTest(source=name):
                self.assertIn("generic laplace", normalized)
                self.assertIn("0.01", normalized)
                self.assertNotIn("smoothness-only", normalized)
                self.assertNotIn("l1 total-variation", normalized)

        for phrase in (
            "observed standardized",
            "no temporal smoothness penalty",
        ):
            self.assertIn(
                phrase,
                " ".join(sources["manuscript"].split()).lower(),
            )


if __name__ == "__main__":
    unittest.main()
