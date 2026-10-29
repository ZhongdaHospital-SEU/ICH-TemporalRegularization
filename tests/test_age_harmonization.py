"""Tests for transparent handling of MIMIC date-shift age sentinels."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.ich_dynamic.build_cohorts import harmonize_mimic_age, parse_eicu_age


class AgeHarmonizationTests(unittest.TestCase):
    def test_mimic3_gt89_sentinel_is_not_treated_as_recorded_age(self) -> None:
        result = harmonize_mimic_age(pd.Series([18.0, 89.0, 300.0, 311.0]))
        np.testing.assert_allclose(result.to_numpy(), [18.0, 89.0, 91.0, 91.0])

    def test_ordinary_age_is_unchanged(self) -> None:
        values = pd.Series([18.0, 64.0, 89.0, float("nan")])
        np.testing.assert_allclose(
            harmonize_mimic_age(values).to_numpy(),
            values.to_numpy(),
            equal_nan=True,
        )

    def test_eicu_gt89_is_preserved_instead_of_capped_at_89(self) -> None:
        self.assertEqual(parse_eicu_age("> 89"), 91.0)
        self.assertEqual(parse_eicu_age(">89"), 91.0)
        self.assertEqual(parse_eicu_age("89"), 89.0)


if __name__ == "__main__":
    unittest.main()
