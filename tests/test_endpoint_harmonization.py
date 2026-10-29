from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ich_dynamic import build_cohorts
from src.ich_dynamic.common import add_outcome, death_in_icu_within_window


def _events_without_deterioration(stay_id: int) -> list[dict[str, float | int]]:
    return [
        {"stay_id": stay_id, "time_hour": 1.0, "gcs": 10.0},
        {"stay_id": stay_id, "time_hour": 24.2, "gcs": 10.0},
        {"stay_id": stay_id, "time_hour": 25.2, "gcs": 10.0},
    ]


def _finish_cohort_result(
    cohort: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    count = len(cohort)
    values = np.zeros((count, 24, 1), dtype=float)
    masks = np.ones((count, 24, 1), dtype=float)

    def fake_sequence(
        hourly: pd.DataFrame,
        stay_ids: pd.Series,
        config: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        del hourly, config
        return values, masks, [int(value) for value in stay_ids]

    config = {"channels": ["gcs_total"], "sequence_hours": 24}
    with tempfile.TemporaryDirectory() as temporary:
        with (
            patch.object(build_cohorts, "PROCESSED", Path(temporary)),
            patch.object(build_cohorts, "sequence_from_hourly", fake_sequence),
        ):
            return build_cohorts.finish_cohort(
                cohort,
                pd.DataFrame(columns=["stay_id", "hour", "channel", "value"]),
                pd.DataFrame(columns=["stay_id"]),
                config,
                name,
            )


def _endpoint_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "outcome": [0.0, 0.0, 1.0, 2.0, np.nan, 2.0],
            "recorded_intubation_24_72h": [
                0.0,
                1.0,
                0.0,
                np.nan,
                1.0,
                0.0,
            ],
            "age": [60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
            "icu_out_hour": [60.0, 80.0, 80.0, 30.0, 80.0, 40.0],
            "death_hour_from_icu": [
                np.nan,
                np.nan,
                np.nan,
                40.0,
                np.nan,
                40.0,
            ],
            "death_within_72h": [
                False,
                False,
                False,
                True,
                False,
                True,
            ],
        }
    )


def _flow_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6, 7],
            "icu_out_hour": [20.0, 20.0, 80.0, 80.0, 60.0, 80.0, 80.0],
            "death_hour_from_icu": [
                20.0,
                np.nan,
                30.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "death_within_72h": [
                True,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            "in_icu_at_primary_landmark": [
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            "alive_at_primary_landmark": [
                False,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
            "baseline_gcs_max": [10.0, 10.0, 10.0, np.nan, 10.0, 10.0, 10.0],
            "outcome": [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 0.0],
            "death_in_icu_within_72h": [
                True,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        }
    )


class EndpointHarmonizationTests(unittest.TestCase):
    def test_csv_rounding_does_not_exclude_an_inclusive_death_boundary(
        self,
    ) -> None:
        result = death_in_icu_within_window(
            pd.Series([True]),
            pd.Series([59.18333333333333]),
            pd.Series([59.16666666666667]),
        )
        self.assertTrue(bool(result.iloc[0]))

    def test_source_uses_recorded_field_and_exclusive_72_hour_bound(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ich_dynamic"
            / "build_cohorts.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertNotIn("unplanned_intubation_24_72h", source)
        self.assertNotIn("outcome_with_intubation", source)
        self.assertEqual(source.count("p.itemid = 224385"), 2)
        self.assertEqual(
            source.count("p.starttime < c.intime + INTERVAL 72 HOUR"),
            2,
        )
        self.assertNotIn(
            "p.starttime <= c.intime + INTERVAL 72 HOUR",
            source,
        )

    def test_all_three_cohort_builders_disable_intubation_for_primary(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ich_dynamic"
            / "build_cohorts.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "add_outcome"
        ]
        self.assertEqual(len(calls), 3)
        for call in calls:
            keyword = next(
                (
                    item
                    for item in call.keywords
                    if item.arg == "include_intubation"
                ),
                None,
            )
            self.assertIsNotNone(keyword)
            self.assertIsInstance(keyword.value, ast.Constant)
            self.assertIs(keyword.value.value, False)

    def test_primary_endpoint_ignores_ordinary_recorded_intubation(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "recorded_intubation_24_72h": [1.0],
            }
        )
        events = pd.DataFrame(_events_without_deterioration(1))

        result = add_outcome(cohort, events, include_intubation=False)

        self.assertEqual(result["outcome"].iloc[0], 0)

    def test_primary_endpoint_accepts_only_explicit_verified_event(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "recorded_intubation_24_72h": [0.0],
                "verified_unplanned_intubation_24_72h": [1.0],
            }
        )
        events = pd.DataFrame(
            [
                {"stay_id": 1, "time_hour": 1.0, "gcs": 10.0},
                {"stay_id": 1, "time_hour": 25.0, "gcs": 10.0},
                {"stay_id": 1, "time_hour": 26.0, "gcs": 10.0},
            ]
        )

        result = add_outcome(
            cohort,
            events,
            include_intubation=True,
            verified_unplanned_intubation_column=(
                "verified_unplanned_intubation_24_72h"
            ),
        )

        self.assertEqual(result["outcome"].iloc[0], 2)

    def test_recorded_intubation_composite_is_identical_across_datasets(
        self,
    ) -> None:
        for name in ("mimic4", "mimic3", "eicu"):
            with self.subTest(name=name):
                result = _finish_cohort_result(_endpoint_fixture(), name=name)

                # A live ICU discharge before hour 72 with an observed
                # outcome is retained as grade 0 because eligibility must be
                # decidable at the 24-hour landmark.
                self.assertEqual(result["stay_id"].tolist(), [1, 2, 3, 4, 6])
                self.assertEqual(result["outcome"].tolist(), [0, 0, 1, 2, 2])
                self.assertEqual(
                    result["outcome_recorded_intubation_composite"].tolist(),
                    [0.0, 2.0, 1.0, 2.0, 2.0],
                )
                self.assertNotIn("outcome_with_intubation", result)
                self.assertNotIn("unplanned_intubation_24_72h", result)
                self.assertTrue(
                    (
                        result["recorded_intubation_composite_role"]
                        == "sensitivity_proxy_only"
                    ).all()
                )

    def test_missing_recorded_intubation_leaves_proxy_unchanged(self) -> None:
        cohort = _endpoint_fixture()
        cohort["recorded_intubation_24_72h"] = np.nan

        result = _finish_cohort_result(cohort, name="eicu")

        pd.testing.assert_series_equal(
            result["outcome_recorded_intubation_composite"],
            result["outcome"],
            check_dtype=False,
            check_names=False,
        )
        self.assertTrue(
            (
                result["recorded_intubation_composite_status"]
                == "unavailable_not_verified_unplanned"
            ).all()
        )

    def test_p020_is_explicitly_partial_and_primary_excludes_intubation(
        self,
    ) -> None:
        result = _finish_cohort_result(_endpoint_fixture(), name="mimic4")

        self.assertTrue((result["p020_status"] == "partial").all())
        self.assertFalse(result["p020_complete"].astype(bool).any())
        self.assertFalse(result["primary_outcome_includes_intubation"].any())

    def test_flow_uses_one_fixed_priority_reason_per_stay(self) -> None:
        prefilter = _flow_fixture()
        included = prefilter[prefilter["stay_id"].eq(7)].copy()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "mimic4"
            dataset_root.mkdir()
            prefilter.to_csv(
                dataset_root / "cohort_pre_filter.csv", index=False
            )
            with patch.object(build_cohorts, "PROCESSED", root):
                flow = build_cohorts.make_flow(
                    "mimic4",
                    raw_n=11,
                    included=included,
                    prefilter_n=len(prefilter),
                )

        excluded = flow[flow["step"].str.startswith("Excluded:")]
        self.assertEqual(excluded["n"].sum(), len(prefilter) - len(included))
        self.assertEqual(int(excluded["n"].sum()), 6)
        self.assertTrue(excluded["reason_code"].notna().all())
        self.assertEqual(excluded["reason_code"].nunique(), len(excluded))

        counts = excluded.set_index("reason_code")["n"].astype(int)
        self.assertEqual(counts["early_death_at_or_before_24h"], 1)
        self.assertEqual(counts["not_in_icu_at_24h"], 1)
        self.assertEqual(counts["not_alive_at_24h"], 1)
        self.assertEqual(counts["missing_pre_landmark_gcs"], 1)
        self.assertEqual(counts["live_icu_discharge_before_72h"], 1)
        self.assertEqual(
            counts["insufficient_post_landmark_gcs_or_unresolved_outcome"],
            1,
        )

    def test_live_transfer_before_landmark_is_never_eligible(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "outcome": [2.0],
                "recorded_intubation_24_72h": [0.0],
                "age": [60.0],
                "icu_out_hour": [20.0],
                "death_hour_from_icu": [40.0],
                "death_within_72h": [True],
            }
        )
        result = _finish_cohort_result(cohort, name="mimic4")
        self.assertTrue(result.empty)

    def test_classifiable_live_transfer_is_retained_as_grade_zero(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "outcome": [0.0],
                "recorded_intubation_24_72h": [0.0],
                "age": [60.0],
                "icu_out_hour": [30.0],
                "death_hour_from_icu": [np.nan],
                "death_within_72h": [False],
            }
        )
        result = _finish_cohort_result(cohort, name="mimic4")

        self.assertEqual(result["stay_id"].tolist(), [1])
        self.assertEqual(int(result["outcome"].iloc[0]), 0)
        self.assertTrue(
            bool(result["early_live_icu_transfer_before_72h"].iloc[0])
        )
        self.assertFalse(bool(result["complete_followup_only"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
