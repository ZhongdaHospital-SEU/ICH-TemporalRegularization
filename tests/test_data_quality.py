"""Regression tests for cohort-state and sensitivity-data handling."""
from __future__ import annotations

import unittest

import pandas as pd

from src.ich_dynamic.build_cohorts import summarize_deep_sedation
from src.ich_dynamic.sensitivity_analysis import scenario_frame


class DeepSedationTests(unittest.TestCase):
    def test_missing_rass_remains_missing(self) -> None:
        events = pd.DataFrame(
            {"stay_id": [1], "time_hour": [3.0], "value": [0.0]}
        )
        result = summarize_deep_sedation(events)
        self.assertEqual(
            result.loc[result["stay_id"].eq(1), "any_deep_sedation_24h"].iloc[0],
            0.0,
        )

    def test_deep_sedation_marks_observed_and_positive(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [1, 1],
                "time_hour": [2.0, 4.0],
                "value": [-1.0, -3.0],
            }
        )
        result = summarize_deep_sedation(events).iloc[0]
        self.assertEqual(result["any_deep_sedation_24h"], 1.0)
        self.assertEqual(result["deep_sedation_observed_24h"], 1.0)


class EarlyDischargeSensitivityTests(unittest.TestCase):
    def test_early_transfer_is_coded_grade_zero_in_the_censoring_sensitivity(
        self,
    ) -> None:
        prefilter = pd.DataFrame(
            {
                "stay_id": [1, 2],
                "outcome": [1.0, None],
                "icu_out_hour": [40.0, 40.0],
                "death_within_72h": [False, False],
            }
        )
        gcs = pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 2],
                "time_hour": [25.0, 26.0, 25.0, 26.0],
                "gcs": [7.0, 7.0, 10.0, 10.0],
            }
        )
        result = scenario_frame("early_discharge_stable", prefilter, gcs)
        labels = dict(zip(result["stay_id"], result["outcome"]))
        self.assertEqual(labels[1], 0.0)
        self.assertEqual(labels[2], 0.0)


if __name__ == "__main__":
    unittest.main()
