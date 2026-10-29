"""Regression tests for eICU death timing relative to ICU admission."""
from __future__ import annotations

import unittest

import pandas as pd

from src.ich_dynamic.build_cohorts import harmonize_eicu_death


class EICUTimingTests(unittest.TestCase):
    def test_hospital_death_uses_icu_relative_offset(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stay_id": 1,
                    "unitdischargestatus": "alive",
                    "hospitaldischargestatus": "expired",
                    "unitdischargeoffset": 4000,
                    "hospitaldischargeoffset": 4000,
                    "hospitaladmitoffset": -10000,
                }
            ]
        )
        result = harmonize_eicu_death(frame)
        self.assertAlmostEqual(result.iloc[0]["death_hour_from_icu"], 4000 / 60)
        self.assertTrue(bool(result.iloc[0]["death_within_72h"]))

    def test_earlier_unit_death_is_used_when_both_are_expired(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stay_id": 2,
                    "unitdischargestatus": "expired",
                    "hospitaldischargestatus": "expired",
                    "unitdischargeoffset": 1800,
                    "hospitaldischargeoffset": 3600,
                    "hospitaladmitoffset": -3000,
                }
            ]
        )
        result = harmonize_eicu_death(frame)
        self.assertAlmostEqual(result.iloc[0]["death_hour_from_icu"], 30.0)

    def test_death_at_or_before_landmark_is_retained_for_earlier_landmarks(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stay_id": 3,
                    "unitdischargestatus": "expired",
                    "hospitaldischargestatus": "expired",
                    "unitdischargeoffset": 1440,
                    "hospitaldischargeoffset": 1440,
                    "hospitaladmitoffset": -2000,
                }
            ]
        )
        result = harmonize_eicu_death(frame)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["death_hour_from_icu"], 24.0)

    def test_observed_hospital_death_offset_is_already_icu_relative(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stay_id": 2896176,
                    "unitdischargestatus": "Expired",
                    "hospitaldischargestatus": "Expired",
                    "unitdischargeoffset": 1729,
                    "hospitaldischargeoffset": 1729,
                    "hospitaladmitoffset": -1647,
                }
            ]
        )
        result = harmonize_eicu_death(frame)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result.iloc[0]["death_hour_from_icu"],
            1729 / 60,
        )
        self.assertTrue(bool(result.iloc[0]["death_within_72h"]))


if __name__ == "__main__":
    unittest.main()
