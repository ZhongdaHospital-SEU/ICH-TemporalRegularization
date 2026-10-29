"""Regression tests for rank-safe time-order negative controls."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.ich_dynamic.models import (
    DynamicFeatureGraph,
    GRUDOrdinalNet,
    SequenceDataset,
    TemporalOrdinalNet,
    add_laplace_input_noise,
    predict_temporal_model,
)
from src.ich_dynamic.train_evaluate import (
    MODEL_META,
    permute_time_order_by_subject,
)


class TimeControlShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequences = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
        self.masks = np.ones((2, 6), dtype=np.float32)
        self.static = np.ones((2, 4), dtype=np.float32)
        self.labels = np.array([0, 2], dtype=np.int64)

    def test_reverse_accepts_two_dimensional_hour_mask(self) -> None:
        dataset = SequenceDataset(
            self.sequences,
            self.masks,
            self.static,
            self.labels,
            time_mode="reverse",
        )
        sequence, mask, _, _, _ = dataset[0]
        np.testing.assert_array_equal(
            sequence[:, 0].numpy(),
            np.array([15.0, 12.0, 9.0, 6.0, 3.0, 0.0]),
        )
        self.assertEqual(tuple(mask.shape), (6,))

    def test_permute_uses_the_time_axis(self) -> None:
        dataset = SequenceDataset(
            self.sequences,
            self.masks,
            self.static,
            self.labels,
            time_mode="permute",
            time_seed=17,
        )
        sequence, mask, _, _, _ = dataset[0]
        self.assertEqual(tuple(sequence.shape), (6, 3))
        self.assertEqual(tuple(mask.shape), (6,))
        np.testing.assert_array_equal(
            np.sort(sequence[:, 0].numpy()),
            np.arange(0.0, 18.0, 3.0),
        )

    def test_time_permutation_is_stable_within_patient(self) -> None:
        base = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
        sequences = np.stack([base, base, base + 100.0])
        original = sequences.copy()
        masks = np.ones((3, 6), dtype=np.float32)
        subject_keys = np.array(["p1", "p1", "p2"])
        permuted, permuted_masks = permute_time_order_by_subject(
            sequences,
            masks,
            subject_keys,
            seed=17,
        )
        np.testing.assert_array_equal(permuted[0], permuted[1])
        np.testing.assert_array_equal(permuted_masks[0], permuted_masks[1])
        self.assertFalse(np.array_equal(permuted[0, :, 0], sequences[0, :, 0]))
        np.testing.assert_array_equal(sequences, original)

    def test_reverse_control_changes_only_the_time_order(self) -> None:
        primary = MODEL_META["dggru_tr"]
        for control in ("dggru_tr_treverse", "dggru_tr_tpermute"):
            metadata = MODEL_META[control]
            for field in (
                "architecture",
                "graph",
                "regularized",
                "static_mode",
                "sequence_mode",
                "mask_mode",
            ):
                self.assertEqual(metadata[field], primary[field])
        self.assertEqual(
            MODEL_META["dggru_tr_treverse"]["time_mode"],
            "reverse",
        )
        self.assertEqual(
            MODEL_META["dggru_tr_tpermute"]["time_mode"],
            "permute",
        )
        self.assertEqual(MODEL_META["dggru_laplace"]["static_mode"], "core")

    def test_prediction_and_network_forward_accept_hour_mask(self) -> None:
        model = TemporalOrdinalNet(
            n_sequence=6,
            n_static=4,
            hidden=8,
            use_graph=True,
        )
        probabilities, trajectory = predict_temporal_model(
            model,
            np.concatenate([self.sequences, self.sequences], axis=2),
            self.masks,
            self.static,
            time_mode="reverse",
        )
        self.assertEqual(probabilities.shape, (2, 3))
        self.assertEqual(trajectory.shape, (2, 6))

        sequence = torch.from_numpy(
            np.concatenate([self.sequences, self.sequences], axis=2)
        )
        mask = torch.from_numpy(self.masks)
        probabilities, _, _ = model(
            sequence,
            mask,
            torch.from_numpy(self.static),
            reverse_time=True,
        )
        self.assertEqual(tuple(probabilities.shape), (2, 3))

    def test_dynamic_graph_does_not_use_unobserved_values_as_edges(self) -> None:
        graph = DynamicFeatureGraph(n_features=3).eval()
        observed = torch.tensor(
            [[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        first = torch.tensor(
            [[[0.2, 0.0, -0.3], [0.5, 0.1, 0.0]]],
            dtype=torch.float32,
        )
        second = first.clone()
        second[observed == 0] = 8.0
        with torch.no_grad():
            first_values, first_adjacency = graph(first, observed)
            second_values, second_adjacency = graph(second, observed)
        torch.testing.assert_close(first_values, second_values)
        torch.testing.assert_close(first_adjacency, second_adjacency)

    def test_generic_laplace_control_perturbs_only_observed_values(self) -> None:
        torch.manual_seed(11)
        sequence = torch.zeros((1, 4, 6), dtype=torch.float32)
        sequence[:, :, :3] = torch.arange(
            12, dtype=torch.float32
        ).reshape(1, 4, 3)
        sequence[:, :, 3:] = 1.0
        sequence[:, 1, 1] = 0.0
        sequence[:, 1, 4] = 0.0
        trajectory_mask = torch.ones((1, 4), dtype=torch.float32)
        perturbed = add_laplace_input_noise(
            sequence,
            trajectory_mask,
            scale=0.01,
        )
        torch.testing.assert_close(perturbed[:, :, 3:], sequence[:, :, 3:])
        self.assertFalse(
            torch.equal(perturbed[:, :, :3], sequence[:, :, :3])
        )
        torch.testing.assert_close(
            perturbed[:, 1, 1],
            sequence[:, 1, 1],
        )
        torch.testing.assert_close(
            add_laplace_input_noise(
                sequence,
                trajectory_mask,
                scale=0.0,
            ),
            sequence,
        )

    def test_all_missing_graph_and_grud_rows_remain_finite(self) -> None:
        graph = DynamicFeatureGraph(n_features=3).eval()
        values = torch.zeros((1, 2, 3), dtype=torch.float32)
        observed = torch.zeros_like(values)
        with torch.no_grad():
            aggregated, adjacency = graph(values, observed)
        self.assertTrue(torch.isfinite(aggregated).all())
        self.assertTrue(torch.isfinite(adjacency).all())
        self.assertTrue(torch.equal(aggregated, torch.zeros_like(aggregated)))

        model = GRUDOrdinalNet(
            n_sequence=6,
            n_static=4,
            hidden=8,
        ).eval()
        sequence = torch.zeros((1, 2, 6), dtype=torch.float32)
        mask = torch.zeros((1, 2), dtype=torch.float32)
        with torch.no_grad():
            probabilities, risk, _ = model(
                sequence,
                mask,
                torch.zeros((1, 4), dtype=torch.float32),
            )
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.isfinite(risk).all())


if __name__ == "__main__":
    unittest.main()
