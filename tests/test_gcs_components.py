"""Tests for GCS component assembly."""
import unittest

import pandas as pd

from src.ich_dynamic.common import classify_end, combine_gcs_components


class GCSComponentTests(unittest.TestCase):
    def test_empty_input_preserves_downstream_timestamp_columns(self):
        result = combine_gcs_components(pd.DataFrame())
        self.assertEqual(
            result.columns.tolist(),
            ["stay_id", "charttime", "gcs", "ett_verbal"],
        )

    def test_sustained_drop_uses_adjacent_hourly_medians_after_csv_round_trip(self):
        post = pd.DataFrame(
            {
                "time_hour": [
                    26.14638888888889,
                    29.92972222222222,
                    34.929722222222225,
                    35.929722222222225,
                    37.92972222222223,
                    39.92972222222223,
                    45.92972222222223,
                    61.92972222222223,
                ],
                "gcs": [9.0, 8.0, 4.0, 4.0, 7.0, 11.0, 10.0, 9.0],
            }
        )
        result = classify_end(post, baseline=8.0, death_within_72h=False)
        self.assertEqual(result["outcome"], 2)
        self.assertEqual(result["max_sustained_drop"], 4.0)


if __name__ == "__main__":
    unittest.main()
