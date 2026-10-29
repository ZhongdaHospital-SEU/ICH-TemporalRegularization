from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic import train_evaluate
from src.ich_dynamic.common import CHANNELS, build_sequence_tensor
from work.build_landmark_inputs import (
    LandmarkInputError,
    LandmarkReconciliationError,
    build_landmark_dataset,
    write_landmark_dataset,
)


def _gcs_row(
    stay_id: int,
    time_hour: float,
    gcs: float,
) -> dict[str, float | int]:
    return {"stay_id": stay_id, "time_hour": time_hour, "gcs": gcs}


def _candidate_row(
    stay_id: int,
    *,
    icu_out_hour: float = 80.0,
    death_within_72h: bool = False,
    death_hour_from_icu: float = np.nan,
    lab_sodium_first: float = 140.0,
) -> dict[str, object]:
    return {
        "stay_id": stay_id,
        "subject_key": f"subject_{stay_id}",
        "hospital_key": f"hospital_{stay_id}",
        "icu_out_hour": icu_out_hour,
        "death_within_72h": death_within_72h,
        "death_hour_from_icu": death_hour_from_icu,
        "age": 60.0 + stay_id,
        "age_gt89": 0.0,
        "sex_male": float(stay_id % 2),
        "race_white": 1.0,
        "emergency_admission": 1.0,
        "min_rass_24h": -2.0,
        "any_rass_le_minus2_24h": 1.0,
        "any_deep_sedation_24h": 0.0,
        "deep_sedation_observed_24h": 1.0,
        "mechanical_ventilation_24h": 1.0,
        "recorded_intubation_24_72h": np.nan,
        "lab_sodium_first": lab_sodium_first,
    }


def _hourly_rows(
    stay_id: int,
    *,
    hour_values: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    values = hour_values or {3: 80.0, 6: 120.0, 24: 150.0, 50: 170.0}
    return [
        {"stay_id": stay_id, "hour": hour, "channel": "hr", "value": value}
        for hour, value in values.items()
    ]


def _write_dataset(
    root: Path,
    *,
    candidates: list[dict[str, object]],
    gcs: list[dict[str, float | int]],
    hourly: list[dict[str, object]],
    reference: bool = False,
) -> Path:
    dataset_root = root / "mimic4"
    dataset_root.mkdir(parents=True)
    pd.DataFrame(candidates).to_csv(
        dataset_root / "cohort_pre_filter.csv", index=False
    )
    pd.DataFrame(gcs).to_csv(dataset_root / "gcs_events.csv", index=False)
    pd.DataFrame(hourly).to_csv(dataset_root / "hourly.csv", index=False)
    if reference:
        built = build_landmark_dataset(
            "mimic4",
            24,
            processed_root=root,
            channels=CHANNELS,
            strict_reference=False,
        )
        built.cohort.to_csv(dataset_root / "cohort.csv", index=False)
        np.savez_compressed(
            dataset_root / "sequences.npz",
            values=built.values,
            masks=built.masks,
            stay_id=built.stay_ids,
        )
    return dataset_root


def _write_validated_landmark_fixture(root: Path) -> Path:
    dataset_root = _write_dataset(
        root,
        candidates=[_candidate_row(1)],
        gcs=[
            _gcs_row(1, 1.0, 15.0),
            _gcs_row(1, 24.0, 13.0),
            _gcs_row(1, 25.0, 13.0),
        ],
        hourly=_hourly_rows(1),
        reference=True,
    )
    built = build_landmark_dataset(
        "mimic4",
        24,
        processed_root=root,
        channels=CHANNELS,
        strict_reference=True,
    )
    paths = write_landmark_dataset(built, root / "landmark_inputs")
    return paths["audit"].parent


class LandmarkInputTests(unittest.TestCase):
    def test_model_loader_requires_current_independent_source_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_validated_landmark_fixture(root)
            with patch.object(train_evaluate, "PROCESSED", root):
                loaded = train_evaluate.load_dataset("mimic4")
                self.assertEqual(loaded.labels.tolist(), [1])

                hourly_path = root / "mimic4" / "hourly.csv"
                hourly = pd.read_csv(hourly_path)
                hourly.loc[hourly.index[-1], "value"] += 1.0
                hourly.to_csv(hourly_path, index=False)
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

    def test_main_24h_loader_cannot_bypass_audit_or_landmark_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            landmark_dir = _write_validated_landmark_fixture(root)
            audit_path = landmark_dir / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["landmark_hours"] = 48
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

            audit_path.unlink()
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

    def test_loader_rejects_duplicate_or_missing_source_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            landmark_dir = _write_validated_landmark_fixture(root)
            audit_path = landmark_dir / "audit.json"
            original = json.loads(audit_path.read_text(encoding="utf-8"))

            duplicate = json.loads(json.dumps(original))
            duplicate["source_files"]["hourly_copy"] = duplicate[
                "source_files"
            ]["hourly"]
            audit_path.write_text(json.dumps(duplicate), encoding="utf-8")
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

            missing = json.loads(json.dumps(original))
            del missing["source_files"]["gcs_events"]
            audit_path.write_text(json.dumps(missing), encoding="utf-8")
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

            serialized = json.dumps(original)
            needle = '"hourly": {'
            self.assertIn(needle, serialized)
            duplicate_key = (
                '"hourly": {"path": "duplicate", '
                f'"sha256": "{"0" * 64}"}}, '
            )
            audit_path.write_text(
                serialized.replace(needle, duplicate_key + needle, 1),
                encoding="utf-8",
            )
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

    def test_loader_rejects_sequence_cohort_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_validated_landmark_fixture(root)
            sequence_path = root / "mimic4" / "sequences.npz"
            with np.load(sequence_path, allow_pickle=True) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["stay_id"] = np.asarray([99], dtype=np.int64)
            np.savez_compressed(sequence_path, **arrays)
            with patch.object(train_evaluate, "PROCESSED", root):
                with self.assertRaises(train_evaluate.LandmarkLineageError):
                    train_evaluate.load_dataset("mimic4")

    def test_sequences_use_only_pre_landmark_hours_and_preserve_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[_candidate_row(1)],
                gcs=[
                    _gcs_row(1, 1.0, 15.0),
                    _gcs_row(1, 7.0, 14.0),
                    _gcs_row(1, 24.0, 13.0),
                    _gcs_row(1, 25.0, 12.0),
                ],
                hourly=_hourly_rows(1),
            )
            built = build_landmark_dataset(
                "mimic4",
                6,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            self.assertEqual(built.values.shape, (1, 6, 14))
            hr_index = CHANNELS.index("hr")
            self.assertEqual(built.values[0, 3, hr_index], 80.0)
            self.assertEqual(built.masks[0, 3, hr_index], 1.0)
            self.assertTrue(np.isnan(built.values[0, 5, hr_index]))
            self.assertEqual(built.masks[0, 5, hr_index], 0.0)
            self.assertTrue(
                np.isnan(built.values[0, :, CHANNELS.index("gcs_motor")]).all()
            )
            self.assertIn(
                "never imputed", built.audit["sequence"]["missing_value_encoding"]
            )

    def test_48h_filters_deaths_at_or_before_landmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[
                    _candidate_row(
                        1,
                        icu_out_hour=48.0,
                        death_within_72h=True,
                        death_hour_from_icu=48.0,
                    ),
                    _candidate_row(
                        2,
                        icu_out_hour=50.0,
                        death_within_72h=True,
                        death_hour_from_icu=50.0,
                    ),
                    _candidate_row(3),
                ],
                gcs=[
                    _gcs_row(1, 2.0, 14.0),
                    _gcs_row(2, 2.0, 14.0),
                    _gcs_row(2, 50.0, 14.0),
                    _gcs_row(2, 51.0, 14.0),
                    _gcs_row(3, 2.0, 14.0),
                    _gcs_row(3, 50.0, 12.0),
                    _gcs_row(3, 51.0, 12.0),
                ],
                hourly=(
                    _hourly_rows(1)
                    + _hourly_rows(2)
                    + _hourly_rows(3)
                ),
            )
            built = build_landmark_dataset(
                "mimic4",
                48,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            self.assertNotIn(1, set(built.stay_ids.tolist()))
            labels = dict(zip(built.stay_ids.tolist(), built.cohort["outcome"]))
            self.assertEqual(labels[2], 2)
            self.assertEqual(labels[3], 1)
            self.assertEqual(
                built.audit["eligibility"]["excluded"]["not_alive_at_landmark_n"],
                1,
            )

    def test_landmark_requires_documented_icu_presence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[
                    _candidate_row(
                        1,
                        icu_out_hour=20.0,
                        death_within_72h=True,
                        death_hour_from_icu=40.0,
                    ),
                    _candidate_row(2),
                ],
                gcs=[
                    _gcs_row(2, 2.0, 14.0),
                    _gcs_row(2, 25.0, 12.0),
                    _gcs_row(2, 26.0, 12.0),
                ],
                hourly=_hourly_rows(1) + _hourly_rows(2),
            )
            built = build_landmark_dataset(
                "mimic4",
                24,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            self.assertEqual(built.stay_ids, [2])
            self.assertEqual(
                built.audit["eligibility"]["excluded"][
                    "not_in_icu_at_landmark_n"
                ],
                1,
            )

    def test_first_day_features_are_withheld_at_6h_and_safe_later(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[_candidate_row(1, lab_sodium_first=142.0)],
                gcs=[
                    _gcs_row(1, 1.0, 15.0),
                    _gcs_row(1, 24.0, 14.0),
                    _gcs_row(1, 25.0, 14.0),
                    _gcs_row(1, 49.0, 14.0),
                    _gcs_row(1, 50.0, 14.0),
                ],
                hourly=_hourly_rows(1),
            )
            built_6 = build_landmark_dataset(
                "mimic4",
                6,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            self.assertTrue(built_6.cohort["lab_sodium_first"].isna().all())
            self.assertEqual(
                built_6.audit["static_inputs"]["laboratory_status"]["status"],
                "unavailable_unsupported",
            )
            self.assertEqual(
                built_6.audit["static_inputs"]["context_status"]["status"],
                "unavailable_unsupported",
            )
            self.assertNotIn(
                "lab_sodium_first",
                built_6.audit["static_inputs"]["predictor_columns"],
            )
            self.assertNotIn(
                "mechanical_ventilation_24h",
                built_6.audit["static_inputs"]["predictor_columns"],
            )
            self.assertNotIn(
                "min_rass_24h",
                built_6.audit["static_inputs"]["predictor_columns"],
            )

            for landmark_hours in (24, 48):
                built = build_landmark_dataset(
                    "mimic4",
                    landmark_hours,
                    processed_root=root,
                    channels=CHANNELS,
                    strict_reference=False,
                )
                self.assertEqual(
                    built.cohort["lab_sodium_first"].iloc[0], 142.0
                )
                self.assertEqual(
                    built.audit["static_inputs"]["laboratory_status"]["status"],
                    "available",
                )
                self.assertEqual(
                    built.audit["static_inputs"]["context_status"]["status"],
                    "available",
                )
                self.assertIn(
                    "lab_sodium_first",
                    built.audit["static_inputs"]["predictor_columns"],
                )
                self.assertIn(
                    "mechanical_ventilation_24h",
                    built.audit["static_inputs"]["predictor_columns"],
                )
                for column in (
                    "min_rass_24h",
                    "any_rass_le_minus2_24h",
                    "any_deep_sedation_24h",
                    "deep_sedation_observed_24h",
                ):
                    self.assertIn(
                        column,
                        built.audit["static_inputs"]["predictor_columns"],
                    )
                self.assertNotIn(
                    "n_gcs_pre",
                    built.audit["static_inputs"]["predictor_columns"],
                )

    def test_24h_reconstructs_existing_cohort_and_tensor_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gcs = [
                _gcs_row(1, 1.0, 15.0),
                _gcs_row(1, 24.0, 13.0),
                _gcs_row(1, 25.0, 13.0),
                _gcs_row(2, 2.0, 14.0),
                _gcs_row(2, 24.0, 14.0),
                _gcs_row(2, 25.0, 14.0),
            ]
            _write_dataset(
                root,
                candidates=[_candidate_row(1), _candidate_row(2)],
                gcs=gcs,
                hourly=_hourly_rows(1) + _hourly_rows(2),
                reference=True,
            )
            built = build_landmark_dataset(
                "mimic4",
                24,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=True,
            )
            self.assertTrue(
                built.audit["reconciliation_24h"]["all_checks_passed"]
            )
            self.assertTrue(
                built.audit["reconciliation_24h"]["checks"]["sequences"][
                    "values_match_exactly"
                ]
            )

    def test_24h_reconciliation_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = _write_dataset(
                root,
                candidates=[_candidate_row(1)],
                gcs=[
                    _gcs_row(1, 1.0, 15.0),
                    _gcs_row(1, 24.0, 13.0),
                    _gcs_row(1, 25.0, 13.0),
                ],
                hourly=_hourly_rows(1),
                reference=True,
            )
            reference = pd.read_csv(dataset_root / "cohort.csv")
            reference.loc[reference["stay_id"].eq(1), "outcome"] = 0
            reference.to_csv(dataset_root / "cohort.csv", index=False)
            with self.assertRaises(LandmarkReconciliationError):
                build_landmark_dataset(
                    "mimic4",
                    24,
                    processed_root=root,
                    channels=CHANNELS,
                    strict_reference=True,
                )

    def test_6h_audit_uses_the_common_unrestricted_candidate_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[_candidate_row(1)],
                gcs=[
                    _gcs_row(1, 1.0, 15.0),
                    _gcs_row(1, 24.0, 14.0),
                    _gcs_row(1, 25.0, 14.0),
                ],
                hourly=_hourly_rows(1),
            )
            built = build_landmark_dataset(
                "mimic4",
                6,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            self.assertIn(
                "not restricted to 24-hour survivors",
                built.audit["eligibility"]["scope_note"],
            )
            self.assertFalse(
                built.audit["endpoint"]["unplanned_intubation_included"]
            )

    def test_writes_three_requested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[_candidate_row(1)],
                gcs=[
                    _gcs_row(1, 1.0, 15.0),
                    _gcs_row(1, 7.0, 14.0),
                    _gcs_row(1, 24.0, 13.0),
                    _gcs_row(1, 25.0, 13.0),
                ],
                hourly=_hourly_rows(1),
            )
            built = build_landmark_dataset(
                "mimic4",
                6,
                processed_root=root,
                channels=CHANNELS,
                strict_reference=False,
            )
            paths = write_landmark_dataset(built, root / "outputs")
            self.assertTrue(paths["cohort"].is_file())
            self.assertTrue(paths["sequences"].is_file())
            self.assertTrue(paths["audit"].is_file())
            with np.load(paths["sequences"], allow_pickle=True) as npz:
                self.assertEqual(npz["values"].shape, (1, 6, 14))
                self.assertEqual(npz["landmark_hours"].item(), 6)
            with paths["audit"].open(encoding="utf-8") as stream:
                audit = json.load(stream)
            self.assertEqual(audit["landmark_hours"], 6)

    def test_rejects_non_14_channel_contract_and_bad_landmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(
                root,
                candidates=[_candidate_row(1)],
                gcs=[_gcs_row(1, 1.0, 15.0)],
                hourly=_hourly_rows(1),
            )
            with self.assertRaises(LandmarkInputError):
                build_landmark_dataset(
                    "mimic4",
                    6,
                    processed_root=root,
                    channels=("hr",),
                    strict_reference=False,
                )
            with self.assertRaises(LandmarkInputError):
                build_landmark_dataset(
                    "mimic4",
                    12,
                    processed_root=root,
                    channels=CHANNELS,
                    strict_reference=False,
                )


if __name__ == "__main__":
    unittest.main()
