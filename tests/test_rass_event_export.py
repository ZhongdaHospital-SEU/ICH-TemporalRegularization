"""Tests for event-level RASS normalization and export."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.build_cohorts import (
    eicu_rass_events_from_nurse,
    prepare_rass_events,
    write_rass_events,
)


class RassEventNormalizationTests(unittest.TestCase):
    def test_invalid_and_missing_values_are_not_coded_as_unsedated(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [1, 1, 2, 3],
                "time_hour": [25.0, 26.0, 27.0, 28.0],
                "value": [-4.0, 0.0, np.nan, 5.0],
            }
        )
        result = prepare_rass_events(events)
        self.assertEqual(result["stay_id"].tolist(), [1, 1])
        self.assertEqual(result["rass"].tolist(), [-4.0, 0.0])

    def test_event_times_are_not_hourly_aggregated(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [1, 1],
                "time_hour": [25.1, 25.9],
                "value": [-3.0, -2.0],
            }
        )
        result = prepare_rass_events(events)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["time_hour"].tolist(), [25.1, 25.9])

    def test_rass_bounds_are_preserved(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [1, 1],
                "time_hour": [0.0, 71.9],
                "value": [-5.0, 4.0],
            }
        )
        result = prepare_rass_events(events)
        self.assertEqual(result["rass"].tolist(), [-5.0, 4.0])

    def test_empty_input_returns_the_canonical_schema(self) -> None:
        result = prepare_rass_events(pd.DataFrame())
        self.assertEqual(list(result.columns), ["stay_id", "time_hour", "rass"])
        self.assertTrue(result.empty)

    def test_already_normalized_rass_column_is_accepted(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [1],
                "time_hour": [30.0],
                "rass": [-1.0],
            }
        )
        result = prepare_rass_events(events)
        self.assertEqual(float(result.loc[0, "rass"]), -1.0)


class EicuRassExtractionTests(unittest.TestCase):
    def test_only_rass_labelled_paired_scores_are_accepted(self) -> None:
        nurse = pd.DataFrame(
            {
                "stay_id": [7] * 8,
                "time_hour": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0],
                "nursingchartoffset": [60, 60, 60, 120, 120, 180, 240, 300],
                "label": [
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "Sedation Scale/Score/Goal",
                    "RASS",
                ],
                "item": [
                    "Sedation Scale",
                    "Sedation Score",
                    "Sedation Goal",
                    "Sedation Scale",
                    "Sedation Score",
                    "Sedation Score",
                    "Sedation Scale",
                    "Value",
                ],
                "nursingchartvalue": [
                    "RASS",
                    "0",
                    "0",
                    "RASS",
                    "-2",
                    "1",
                    "SAS",
                    "-1",
                ],
                "value": [np.nan, 0.0, 0.0, np.nan, -2.0, 1.0, np.nan, -1.0],
            }
        )
        result = eicu_rass_events_from_nurse(nurse)
        self.assertEqual(result["time_hour"].tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(result["rass"].tolist(), [0.0, -2.0, -1.0])

    def test_unpaired_generic_sedation_score_is_not_imputed_as_rass(self) -> None:
        nurse = pd.DataFrame(
            {
                "stay_id": [8],
                "time_hour": [2.0],
                "nursingchartoffset": [120],
                "label": ["Sedation Scale/Score/Goal"],
                "item": ["Sedation Score"],
                "nursingchartvalue": ["0"],
                "value": [0.0],
            }
        )
        result = eicu_rass_events_from_nurse(nurse)
        self.assertTrue(result.empty)


class RassEventWriteTests(unittest.TestCase):
    def test_csv_contains_only_observed_events(self) -> None:
        events = pd.DataFrame(
            {
                "stay_id": [10, 11],
                "time_hour": [24.5, 25.5],
                "value": [-4.0, np.nan],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mimic4" / "rass_events.csv"
            write_rass_events(path, events)
            result = pd.read_csv(path)
            self.assertEqual(list(result.columns), ["stay_id", "time_hour", "rass"])
            self.assertEqual(result["stay_id"].tolist(), [10])
            self.assertFalse(path.with_name("rass_events.csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
