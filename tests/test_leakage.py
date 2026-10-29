"""Regression tests for predictor-time boundaries."""
from __future__ import annotations

import unittest

from src.ich_dynamic.train_evaluate import (
    EXCLUDED_COLUMNS,
    POST_LANDMARK_COLUMNS,
    predictor_column_allowed,
)


class LeakageBoundaryTests(unittest.TestCase):
    def test_all_post_landmark_columns_are_excluded(self) -> None:
        self.assertTrue(POST_LANDMARK_COLUMNS.issubset(EXCLUDED_COLUMNS))

    def test_death_hour_cannot_enter_predictors(self) -> None:
        self.assertIn("death_hour_from_icu", EXCLUDED_COLUMNS)

    def test_untimed_neurosurgery_flag_cannot_enter_predictors(self) -> None:
        self.assertIn("neurosurgery_admission", EXCLUDED_COLUMNS)

    def test_cohort_definition_flags_are_not_predictors(self) -> None:
        self.assertIn("has_ich", EXCLUDED_COLUMNS)

    def test_post_landmark_outcome_and_icu_exit_are_excluded(self) -> None:
        for column in (
            "outcome",
            "outcome_recorded_intubation_composite",
            "outcome_gcs_death_only",
            "death_within_72h",
            "death_after_landmark",
            "recorded_intubation_24_72h",
            "verified_unplanned_intubation_24_72h",
            "endpoint_intubation",
            "intubation_component_included",
            "intubation_composite_available",
            "primary_outcome_includes_intubation",
            "p020_status",
            "p020_complete",
            "recorded_intubation_composite_role",
            "recorded_intubation_composite_status",
            "recorded_intubation_composite_note",
            "max_sustained_drop",
            "last_gcs_hour",
            "icu_out_hour",
            "primary_followup_complete",
            "landmark_alive",
            "followup_window_complete",
            "analysis_weight",
            "censoring_probability",
        ):
            self.assertIn(column, EXCLUDED_COLUMNS)

    def test_boolean_landmark_state_is_rejected_by_name_not_dtype(self) -> None:
        for column in (
            "death_in_icu_within_72h",
            "alive_at_primary_landmark",
            "in_icu_at_primary_landmark",
            "eligible_at_primary_landmark",
            "outcome_recorded_intubation_composite",
        ):
            self.assertIn(column, EXCLUDED_COLUMNS)
            self.assertFalse(predictor_column_allowed(column))
        self.assertFalse(predictor_column_allowed("landmark_future_state"))
        self.assertTrue(predictor_column_allowed("gcs_total_min"))


if __name__ == "__main__":
    unittest.main()
