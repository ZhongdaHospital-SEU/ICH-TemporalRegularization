"""Contract tests for Results 3.6 documentation-intensity transport."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from work import transport_documentation as transport


def cohort_frame(
    intensities: list[float],
    outcomes: list[int],
    cluster_prefix: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, len(intensities) + 1),
            "subject_key": [
                f"{cluster_prefix}_subject_{index}"
                for index in range(len(intensities))
            ],
            "hospital_key": [
                f"{cluster_prefix}_hospital_{index // 2}"
                for index in range(len(intensities))
            ],
            "outcome": outcomes,
            "monitoring_intensity_24h": intensities,
        }
    )


def prediction_probabilities(outcomes: list[int]) -> np.ndarray:
    rows = []
    for outcome in outcomes:
        row = np.full(3, 0.05, dtype=float)
        row[int(outcome)] = 0.90
        rows.append(row)
    return np.asarray(rows)


def write_prediction_file(
    path: Path,
    cohort: pd.DataFrame,
    outcomes: list[int],
) -> None:
    np.savez_compressed(
        path,
        labels=np.asarray(outcomes, dtype=np.int32),
        stay_id=cohort["stay_id"].to_numpy(),
        subject_key=cohort["subject_key"].to_numpy(dtype=str),
        dggru_tr=prediction_probabilities(outcomes),
    )


class DocumentationCutpointTests(unittest.TestCase):
    def test_cutpoints_are_persisted_and_boundary_rule_is_fixed(self) -> None:
        development = cohort_frame(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "dev",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutpoints.json"
            validated = transport.load_cohort(
                self._write_cohort(development, Path(directory)),
                cluster_column="subject_key",
            )
            payload = transport.load_or_create_cutpoints(validated, path)
            self.assertTrue(path.exists())
            self.assertEqual(
                transport.compute_development_cutpoints(validated),
                tuple(payload["cutpoints"]),
            )
            self.assertFalse(payload["external_requantilization_permitted"])

            strata = transport.assign_documentation_strata(
                [1, 4, 7, 100],
                payload["cutpoints"],
            )
            self.assertEqual(strata.tolist(), ["low", "middle", "high", "high"])

            second = transport.load_or_create_cutpoints(validated, path)
            self.assertEqual(second["cutpoints"], payload["cutpoints"])

    def test_changed_development_cohort_cannot_reuse_cutpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = transport.load_cohort(
                self._write_cohort(
                    cohort_frame(
                        [1, 2, 3, 4, 5, 6],
                        [0, 0, 1, 1, 2, 2],
                        "original",
                    ),
                    root,
                    "original.csv",
                ),
                cluster_column="subject_key",
            )
            path = root / "cutpoints.json"
            transport.load_or_create_cutpoints(original, path)

            changed = cohort_frame(
                [2, 3, 4, 5, 6, 7],
                [0, 0, 1, 1, 2, 2],
                "changed",
            )
            changed_frame = transport.load_cohort(
                self._write_cohort(changed, root, "changed.csv"),
                cluster_column="subject_key",
            )
            with self.assertRaisesRegex(ValueError, "different development cohort"):
                transport.load_or_create_cutpoints(changed_frame, path)

    @staticmethod
    def _write_cohort(
        frame: pd.DataFrame,
        directory: Path,
        name: str = "cohort.csv",
    ) -> Path:
        path = directory / name
        frame.to_csv(path, index=False)
        return path


class DocumentationTransportTests(unittest.TestCase):
    def test_macro_auc_is_na_when_a_grade_is_absent(self) -> None:
        labels = np.asarray([0, 0, 2, 2])
        probabilities = prediction_probabilities(labels.tolist())
        self.assertTrue(np.isnan(transport.macro_auc(labels, probabilities)))

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = prediction_probabilities(labels.tolist())
        clusters = np.asarray(["a", "a", "b", "b", "c", "c"])
        first = transport.cluster_bootstrap_auc(
            labels, probabilities, clusters, replicates=40, seed=123
        )
        second = transport.cluster_bootstrap_auc(
            labels, probabilities, clusters, replicates=40, seed=123
        )
        self.assertEqual(first, second)
        self.assertGreater(first[2], 0)
        self.assertLessEqual(first[0], first[1])

    def test_prediction_cohort_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort_frame_value = cohort_frame(
                [1, 2, 3], [0, 1, 2], "mismatch"
            )
            cohort_path = root / "cohort.csv"
            cohort_frame_value.to_csv(cohort_path, index=False)
            cohort = transport.load_cohort(
                cohort_path, cluster_column="hospital_key"
            )
            truncated = cohort_frame_value.iloc[:2]
            prediction_path = root / "predictions.npz"
            write_prediction_file(
                prediction_path,
                truncated,
                [0, 1],
            )
            with self.assertRaisesRegex(ValueError, "same stays"):
                transport.align_predictions(
                    cohort,
                    prediction_path,
                    "dggru_tr",
                )

    def test_cli_writes_fixed_strata_and_explicit_na(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = cohort_frame(
                list(range(1, 10)),
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "dev",
            )
            mimic3 = cohort_frame(
                list(range(1, 10)),
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "m3",
            )
            eicu_outcomes = [0, 0, 0, 0, 1, 2, 0, 1, 2]
            eicu = cohort_frame(
                list(range(1, 10)),
                eicu_outcomes,
                "eicu",
            )
            paths = {
                "dev": root / "dev.csv",
                "mimic3": root / "mimic3.csv",
                "eicu": root / "eicu.csv",
            }
            dev.to_csv(paths["dev"], index=False)
            mimic3.to_csv(paths["mimic3"], index=False)
            eicu.to_csv(paths["eicu"], index=False)
            mimic3_predictions = root / "mimic3_external.npz"
            eicu_predictions = root / "eicu_external.npz"
            write_prediction_file(
                mimic3_predictions,
                mimic3,
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
            )
            write_prediction_file(
                eicu_predictions,
                eicu,
                eicu_outcomes,
            )

            table_path = root / "table.csv"
            audit_path = root / "audit.json"
            cutpoint_path = root / "cutpoints.json"
            exit_code = transport.main(
                [
                    "--development-cohort",
                    str(paths["dev"]),
                    "--mimic3-cohort",
                    str(paths["mimic3"]),
                    "--mimic3-predictions",
                    str(mimic3_predictions),
                    "--eicu-cohort",
                    str(paths["eicu"]),
                    "--eicu-predictions",
                    str(eicu_predictions),
                    "--models",
                    "dggru_tr",
                    "--bootstrap-replicates",
                    "40",
                    "--cutpoint-output",
                    str(cutpoint_path),
                    "--table-output",
                    str(table_path),
                    "--audit-output",
                    str(audit_path),
                ]
            )
            self.assertEqual(exit_code, 0)

            table = pd.read_csv(table_path, keep_default_na=False)
            self.assertEqual(len(table), 6)
            self.assertEqual(
                table["documentation_stratum"].tolist(),
                ["low", "middle", "high", "low", "middle", "high"],
            )
            self.assertTrue((table["n"] == 3).all())
            self.assertTrue(
                table.loc[table["dataset"].eq("mimic3"), "macro_auc"]
                .astype(float)
                .eq(1.0)
                .all()
            )
            eicu_low = table.loc[
                table["dataset"].eq("eicu")
                & table["documentation_stratum"].eq("low")
            ].iloc[0]
            self.assertEqual(eicu_low["macro_auc"], "NA")

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["outline_section"], "Results 3.6")
            self.assertFalse(
                audit["stratification"]["external_requantilization"]
            )
            self.assertEqual(
                audit["stratification"]["cutpoints"],
                [float(value) for value in np.quantile(
                    dev["monitoring_intensity_24h"],
                    [1 / 3, 2 / 3],
                    method="linear",
                )],
            )
            self.assertEqual(
                audit["external_datasets"][1]["cluster_column"],
                "hospital_key",
            )
            self.assertTrue(cutpoint_path.exists())
            json.dumps(audit, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
