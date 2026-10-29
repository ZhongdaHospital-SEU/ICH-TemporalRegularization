import unittest

import pandas as pd

from src.ich_dynamic.common import add_outcome, build_sequence_tensor, classify_end


def gcs_frame(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["time_hour", "gcs"])


class EndpointClassificationTests(unittest.TestCase):
    def test_death_without_two_followup_gcs_hours_is_ineligible(self) -> None:
        result = classify_end(
            gcs_frame([]),
            baseline=10,
            death_within_72h=True,
        )
        self.assertTrue(pd.isna(result["outcome"]))

    def test_verified_intubation_without_two_followup_gcs_hours_is_ineligible(
        self,
    ) -> None:
        result = classify_end(
            gcs_frame([]),
            baseline=10,
            death_within_72h=False,
            verified_unplanned_intubation=True,
        )
        self.assertTrue(pd.isna(result["outcome"]))

    def test_gcs_only_with_sparse_followup_remains_missing(self) -> None:
        result = classify_end(
            gcs_frame([(25.0, 10.0)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertTrue(pd.isna(result["outcome"]))

    def test_death_is_severe_with_two_followup_gcs(self) -> None:
        result = classify_end(
            gcs_frame([(25.0, 9), (26.0, 8)]),
            baseline=10,
            death_within_72h=True,
        )
        self.assertEqual(result["outcome"], 2)

    def test_severity_requires_the_same_sustained_pair(self) -> None:
        result = classify_end(
            gcs_frame([(24.2, 6), (25.3, 8), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 2)
        self.assertEqual(result["max_sustained_drop"], 4)

    def test_two_severely_low_measurements_are_grade_two(self) -> None:
        result = classify_end(
            gcs_frame([(24.2, 6), (25.3, 5), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 2)
        self.assertEqual(result["max_sustained_drop"], 5)

    def test_short_elapsed_span_does_not_qualify(self) -> None:
        result = classify_end(
            gcs_frame([(30.1, 5), (30.9, 5), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 0)

    def test_non_adjacent_hourly_bins_do_not_confirm_persistence(self) -> None:
        result = classify_end(
            gcs_frame([(30.1, 5), (32.2, 5), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 0)

    def test_adjacent_hourly_bins_confirm_persistence(self) -> None:
        result = classify_end(
            gcs_frame([(30.1, 7), (31.9, 7), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 1)
        self.assertEqual(result["max_sustained_drop"], 3)

    def test_single_low_measurement_scenario_requires_no_pair(self) -> None:
        result = classify_end(
            gcs_frame([(24.2, 7), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
            confirmation_consecutive_hours=1,
        )
        self.assertEqual(result["outcome"], 1)

    def test_strict_persistence_requires_adjacent_hourly_bins(self) -> None:
        result = classify_end(
            gcs_frame([(24.2, 7), (26.3, 7), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
            confirmation_consecutive_hours=2,
        )
        self.assertEqual(result["outcome"], 0)
        accepted = classify_end(
            gcs_frame([(24.2, 7), (25.3, 7), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
            confirmation_consecutive_hours=2,
        )
        self.assertEqual(accepted["outcome"], 1)

    def test_add_outcome_replaces_preexisting_outcome_columns(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "outcome": [2],
            }
        )
        events = pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "time_hour": [10.0, 24.2, 70.0],
                "gcs": [10.0, 10.0, 10.0],
            }
        )
        result = add_outcome(cohort, events)
        self.assertEqual(result["outcome"].tolist(), [0])

    def test_add_outcome_ignores_ordinary_recorded_intubation(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "recorded_intubation_24_72h": [1.0],
            }
        )
        events = pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "time_hour": [1.0, 25.0, 26.0],
                "gcs": [10.0, 10.0, 10.0],
            }
        )

        result = add_outcome(cohort, events)

        self.assertEqual(result["outcome"].tolist(), [0])

    def test_add_outcome_uses_only_an_explicit_verified_event_column(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "recorded_intubation_24_72h": [1.0],
                "verified_unplanned_intubation_24_72h": [1.0],
            }
        )

        result = add_outcome(
            cohort,
            pd.DataFrame(
                {
                    "stay_id": [1, 1, 1],
                    "time_hour": [1.0, 25.0, 26.0],
                    "gcs": [10.0, 10.0, 10.0],
                }
            ),
            include_intubation=True,
            verified_unplanned_intubation_column=(
                "verified_unplanned_intubation_24_72h"
            ),
        )

        self.assertEqual(result["outcome"].tolist(), [2])

    def test_followup_does_not_require_a_gcs_after_hour_60(self) -> None:
        result = classify_end(
            gcs_frame([(30.0, 7), (31.0, 7)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 1)

    def test_duplicate_values_in_one_hour_are_a_single_median_observation(self) -> None:
        result = classify_end(
            gcs_frame([(30.1, 6), (30.6, 8), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 0)
        self.assertEqual(result["max_sustained_drop"], 0)

    def test_outcome_uses_hourly_median_rather_than_raw_duplicates(self) -> None:
        result = classify_end(
            gcs_frame([(30.1, 6), (30.6, 8), (31.6, 8), (70.0, 10)]),
            baseline=10,
            death_within_72h=False,
        )
        self.assertEqual(result["outcome"], 1)
        self.assertEqual(result["max_sustained_drop"], 3)

    def test_sequence_builder_uses_median_for_duplicate_hours(self) -> None:
        hourly = pd.DataFrame(
            {
                "stay_id": [1, 1],
                "hour": [0, 0],
                "channel": ["hr", "hr"],
                "value": [80.0, 100.0],
            }
        )
        values, masks, ids = build_sequence_tensor(
            hourly,
            [1],
            channels=["hr"],
            max_hour=24,
        )
        self.assertEqual(ids, [1])
        self.assertAlmostEqual(float(values[0, 0, 0]), 90.0)
        self.assertEqual(float(masks[0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
