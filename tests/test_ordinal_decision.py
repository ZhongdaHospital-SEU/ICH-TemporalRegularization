"""Tests for the ordinal median decision rule."""
import unittest

import numpy as np

from src.ich_dynamic.train_evaluate import ordinal_median_prediction


class OrdinalDecisionTests(unittest.TestCase):
    def test_median_can_select_middle_grade_when_mode_cannot(self):
        probabilities = np.array([[0.45, 0.30, 0.25]])
        prediction = ordinal_median_prediction(probabilities)
        self.assertEqual(prediction.tolist(), [1])
        self.assertEqual(probabilities.argmax(axis=1).tolist(), [0])

    def test_median_respects_cumulative_thresholds(self):
        probabilities = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.30, 0.50, 0.20],
                [0.10, 0.20, 0.70],
            ]
        )
        self.assertEqual(
            ordinal_median_prediction(probabilities).tolist(),
            [0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
