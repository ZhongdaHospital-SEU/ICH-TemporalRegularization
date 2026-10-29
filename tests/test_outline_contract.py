"""Contract tests for methods explicitly fixed by the study outline."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.ich_dynamic.build_cohorts import (
    derive_comorbidity_flags,
    is_ich_diagnosis,
    is_neurosurgery_code,
)
from src.ich_dynamic.common import add_outcome, load_config
from src.ich_dynamic.models import two_branch_probabilities


class OutlineChannelTests(unittest.TestCase):
    def test_fourteen_channels_include_all_named_modalities(self) -> None:
        config = load_config()
        self.assertEqual(len(config["channels"]), 14)
        self.assertEqual(len(set(config["channels"])), 14)
        self.assertEqual(
            set(config["channels"]),
            {
                "gcs_eye",
                "gcs_verbal",
                "gcs_motor",
                "gcs_total",
                "hr",
                "rr",
                "spo2",
                "sbp",
                "dbp",
                "mbp",
                "temp",
                "urine_output",
                "fio2",
                "rass",
            },
        )


class DiagnosisDefinitionTests(unittest.TestCase):
    def test_outline_ich_codes_are_included(self) -> None:
        for code in ("431", "43111", "432", "4321"):
            self.assertTrue(is_ich_diagnosis(code, 9))
        for code in ("I61", "I61.9", "I62", "I62.9"):
            self.assertTrue(is_ich_diagnosis(code, 10))

    def test_non_ich_hemorrhage_codes_are_excluded(self) -> None:
        for code in ("430", "433", "434"):
            self.assertFalse(is_ich_diagnosis(code, 9))
        for code in ("I60", "I63", "I64"):
            self.assertFalse(is_ich_diagnosis(code, 10))

    def test_comorbidity_and_neurosurgery_flags(self) -> None:
        flags = derive_comorbidity_flags(
            7,
            [("I10", "10"), ("E11.9", "10"), ("N18.3", "10")],
        )
        self.assertEqual(flags["stay_id"], 7)
        self.assertEqual(flags["comorb_hypertension"], 1.0)
        self.assertEqual(flags["comorb_diabetes"], 1.0)
        self.assertEqual(flags["comorb_ckd"], 1.0)
        self.assertTrue(is_neurosurgery_code("01.39", 9))
        self.assertTrue(is_neurosurgery_code("00B00ZZ", 10))
        self.assertFalse(is_neurosurgery_code("0DBH0ZZ", 10))


class OrdinalHeadTests(unittest.TestCase):
    def test_two_branches_produce_valid_ordered_probabilities(self) -> None:
        generator = torch.Generator().manual_seed(12)
        centre = torch.randn(512, generator=generator)
        gaps = torch.randn(512, 2, generator=generator) * 2
        probabilities = two_branch_probabilities(centre, gaps)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(512),
                atol=1e-6,
            )
        )
        cumulative = torch.stack(
            [1.0 - probabilities[:, 0], probabilities[:, 2]],
            dim=1,
        )
        self.assertTrue((cumulative[:, 0] >= cumulative[:, 1]).all())


class CompositeEndpointTests(unittest.TestCase):
    def test_unplanned_intubation_is_grade_two_even_without_low_gcs(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "verified_unplanned_intubation_24_72h": [1.0],
            }
        )
        events = pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "time_hour": [10.0, 25.0, 30.0],
                "gcs": [10.0, 10.0, 10.0],
            }
        )
        result = add_outcome(
            cohort,
            events,
            include_intubation=True,
            verified_unplanned_intubation_column=(
                "verified_unplanned_intubation_24_72h"
            ),
        )
        self.assertEqual(int(result.iloc[0]["outcome"]), 2)

    def test_intubation_can_be_removed_for_sensitivity_analysis(self) -> None:
        cohort = pd.DataFrame(
            {
                "stay_id": [1],
                "death_within_72h": [False],
                "verified_unplanned_intubation_24_72h": [1.0],
            }
        )
        events = pd.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "time_hour": [10.0, 25.0, 61.0],
                "gcs": [10.0, 10.0, 10.0],
            }
        )
        result = add_outcome(cohort, events, include_intubation=False)
        self.assertEqual(int(result.iloc[0]["outcome"]), 0)


if __name__ == "__main__":
    unittest.main()
