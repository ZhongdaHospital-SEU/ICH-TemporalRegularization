from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from work import transport_documentation as transport


def make_record(
    dataset: str,
    stratum: str,
    *,
    model: str = "dggru_tr",
) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "documentation_stratum": stratum,
        "q1_cutpoint": 4.0,
        "q2_cutpoint": 8.0,
        "n": 12,
        "grade_0": 6,
        "grade_1": 3,
        "grade_2": 3,
        "events": 6,
        "event_rate": 0.5,
        "event_rate_ci_low": 0.25,
        "event_rate_ci_high": 0.75,
        "any_events": 6,
        "any_event_rate": 0.5,
        "any_event_rate_ci_low": 0.25,
        "any_event_rate_ci_high": 0.75,
        "severe_events": 3,
        "severe_event_rate": 0.25,
        "severe_event_rate_ci_low": 0.08,
        "severe_event_rate_ci_high": 0.53,
        "event_rate_ci_method": "wilson_score_95_unclustered_deterministic",
        "macro_auc": 0.7,
        "macro_auc_ci_low": 0.6,
        "macro_auc_ci_high": 0.8,
        "bootstrap_clusters": 6,
        "bootstrap_valid_replicates": 100,
        "cluster_column": "subject_key",
        "cumulative_any": {
            "events": 6,
            "slope": 1.0,
            "intercept": 0.0,
            "ece": 0.05,
        },
        "cumulative_severe": {
            "events": 3,
            "slope": 1.0,
            "intercept": 0.0,
            "ece": 0.04,
        },
    }


def complete_records() -> list[dict]:
    return [
        make_record(dataset, stratum)
        for dataset in ("mimic3", "eicu")
        for stratum in transport.STRATA
    ]


class DocumentationTransportTests(unittest.TestCase):
    def test_build_analysis_outputs_all_six_external_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = pd.DataFrame(
                {
                    "stay_id": np.arange(1, 7),
                    "subject_key": [f"dev-{value}" for value in range(1, 7)],
                    "outcome": [0, 0, 1, 1, 2, 2],
                    "monitoring_intensity_24h": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            )
            development_path = root / "development.csv"
            development.to_csv(development_path, index=False)

            external_inputs = []
            for dataset, cluster_column in (
                ("mimic3", "subject_key"),
                ("eicu", "hospital_key"),
            ):
                cohort = pd.DataFrame(
                    {
                        "stay_id": np.arange(10, 16),
                        "subject_key": [
                            f"{dataset}-subject-{value}" for value in range(6)
                        ],
                        "outcome": [0, 1, 1, 2, 2, 0],
                        "monitoring_intensity_24h": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        cluster_column: [
                            f"{dataset}-cluster-{value}" for value in range(6)
                        ],
                    }
                )
                cohort_path = root / f"{dataset}.csv"
                cohort.to_csv(cohort_path, index=False)
                prediction_path = root / f"{dataset}.npz"
                np.savez(
                    prediction_path,
                    labels=cohort["outcome"].to_numpy(),
                    stay_id=cohort["stay_id"].to_numpy(),
                    subject_key=cohort["subject_key"].astype(str).to_numpy(),
                    dggru_tr=np.tile(
                        np.array([[0.6, 0.2, 0.2]]),
                        (len(cohort), 1),
                    ),
                )
                external_inputs.append(
                    (dataset, cohort_path, prediction_path, cluster_column)
                )

            records, audit = transport.build_analysis(
                development_path=development_path,
                external_inputs=external_inputs,
                models=("dggru_tr",),
                cutpoint_path=root / "cutpoints.json",
                bootstrap_replicates=10,
                seed=11,
            )

            self.assertEqual(len(records), 6)
            self.assertEqual(audit["completeness"]["status"], "pass")
            self.assertEqual(
                audit["completeness"]["observed_group_count"],
                6,
            )
            table = transport.records_to_table(records)
            self.assertEqual(len(table), 6)
            self.assertEqual(set(table["documentation_stratum"]), set(transport.STRATA))
            self.assertTrue(table["event_rate"].notna().all())
            self.assertFalse(audit["interpretation"]["case_mix_explained"])

    def test_all_six_external_groups_are_required(self) -> None:
        records = complete_records()
        result = transport.validate_external_strata(
            records,
            datasets=("mimic3", "eicu"),
            models=("dggru_tr",),
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["observed_group_count"], 6)

        missing = [
            record
            for record in records
            if not (
                record["dataset"] == "eicu"
                and record["documentation_stratum"] == "high"
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "missing groups",
        ):
            transport.validate_external_strata(
                missing,
                datasets=("mimic3", "eicu"),
                models=("dggru_tr",),
            )

    def test_empty_duplicate_or_incomplete_group_fails(self) -> None:
        duplicate = complete_records()
        duplicate.append(make_record("eicu", "high"))
        with self.assertRaisesRegex(ValueError, "duplicate groups"):
            transport.validate_external_strata(
                duplicate,
                datasets=("mimic3", "eicu"),
                models=("dggru_tr",),
            )

        empty = complete_records()
        empty[0]["n"] = 0
        with self.assertRaisesRegex(ValueError, "empty stratum"):
            transport.validate_external_strata(
                empty,
                datasets=("mimic3", "eicu"),
                models=("dggru_tr",),
            )

        missing_rate = complete_records()
        del missing_rate[2]["any_event_rate_ci_low"]
        with self.assertRaisesRegex(ValueError, "event-rate fields"):
            transport.validate_external_strata(
                missing_rate,
                datasets=("mimic3", "eicu"),
                models=("dggru_tr",),
            )

    def test_event_rates_have_reproducible_interval_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "_outcome": [0, 0, 1, 1, 2, 2],
                "probability_0": [0.7, 0.6, 0.3, 0.2, 0.1, 0.2],
                "probability_1": [0.2, 0.3, 0.5, 0.4, 0.2, 0.2],
                "probability_2": [0.1, 0.1, 0.2, 0.4, 0.7, 0.6],
                "_cluster_key": ["a", "a", "b", "b", "c", "c"],
            }
        )
        record = transport._record_for_stratum(
            dataset="mimic3",
            model="dggru_tr",
            stratum="low",
            frame=frame,
            cutpoints=(4.0, 8.0),
            cluster_column="subject_key",
            bootstrap_replicates=20,
            seed=7,
        )
        self.assertEqual(record["any_events"], 4)
        self.assertAlmostEqual(record["any_event_rate"], 4 / 6)
        self.assertLessEqual(
            record["any_event_rate_ci_low"],
            record["any_event_rate"],
        )
        self.assertGreaterEqual(
            record["any_event_rate_ci_high"],
            record["any_event_rate"],
        )
        self.assertEqual(record["severe_events"], 2)
        self.assertEqual(
            record["event_rate_ci_method"],
            "wilson_score_95_unclustered_deterministic",
        )

        table = transport.records_to_table([record])
        for column in (
            "event_rate",
            "event_rate_ci_low",
            "event_rate_ci_high",
            "any_event_rate",
            "any_event_rate_ci_low",
            "any_event_rate_ci_high",
            "severe_event_rate",
            "severe_event_rate_ci_low",
            "severe_event_rate_ci_high",
            "interpretation_scope",
        ):
            self.assertIn(column, table.columns)

    def test_claim_boundary_does_not_assert_explained_differences(self) -> None:
        caveat = transport.DOCUMENTATION_GRADIENT_CAVEAT.lower()
        self.assertIn("documentation-associated", caveat)
        self.assertIn("do not identify or explain", caveat)
        self.assertIn("calendar period", caveat)
        self.assertIn("case-mix", caveat)


if __name__ == "__main__":
    unittest.main()
