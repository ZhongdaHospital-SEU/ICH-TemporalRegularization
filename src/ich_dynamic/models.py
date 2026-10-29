"""Model definitions and preprocessing for ordinal ICH deterioration."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASSES = 3
N_THRESHOLDS = N_CLASSES - 1


def _flip_time(
    sequence: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip the time axis for three-dimensional values and either mask rank."""
    sequence_dims = [1] if sequence.dim() == 3 else [0]
    mask_dims = [1] if mask.dim() == 3 else [0]
    return (
        torch.flip(sequence, dims=sequence_dims),
        torch.flip(mask, dims=mask_dims),
    )


def two_branch_probabilities(
    centre: torch.Tensor,
    branch_gaps: torch.Tensor,
) -> torch.Tensor:
    """Three-class probabilities from two non-proportional cumulative logits."""
    lower = centre + F.softplus(branch_gaps[:, 0])
    upper = centre - F.softplus(branch_gaps[:, 1])
    cumulative = torch.sigmoid(torch.stack([lower, upper], dim=1))
    probabilities = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1],
        ],
        dim=1,
    )
    probabilities = probabilities.clamp_min(1e-7)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


class TwoBranchOrdinalHead(nn.Module):
    """Shared latent score with two separately parameterised cumulative branches."""

    def __init__(self, hidden: int):
        super().__init__()
        self.centre = nn.Linear(hidden, 1)
        self.branch_gaps = nn.Linear(hidden, N_THRESHOLDS)

    def forward(
        self,
        representation: torch.Tensor,
        shared_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centre = shared_score + self.centre(representation).squeeze(1)
        gaps = self.branch_gaps(representation)
        return two_branch_probabilities(centre, gaps), centre


class DynamicFeatureGraph(nn.Module):
    """Value- and observation-aware dynamic feature graph.

    Missing channels remain available to the recurrent layer through their
    observation indicators, but they do not contribute pseudo-values or
    pseudo-edges to the graph at that hour.
    """

    def __init__(self, n_features: int, edge_dim: int = 16):
        super().__init__()
        self.edge_dim = edge_dim
        self.edge_embedding = nn.Parameter(torch.randn(n_features, edge_dim) * 0.1)
        self.query = nn.Linear(edge_dim + 2, edge_dim)
        self.key = nn.Linear(edge_dim + 2, edge_dim)

    def forward(
        self,
        x: torch.Tensor,
        observed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, features = x.shape
        if observed is None:
            observed = torch.ones_like(x)
        else:
            observed = observed.to(device=x.device, dtype=x.dtype)
            if observed.shape != x.shape:
                raise ValueError(
                    "Graph observation mask must match the value tensor"
                )
            observed = observed.clamp(0.0, 1.0)
        node_values = x * observed
        edge = self.edge_embedding[None, None].expand(batch, steps, features, -1)
        values = node_values.unsqueeze(-1)
        masks = observed.unsqueeze(-1)
        query = self.query(torch.cat([edge, values, masks], dim=-1))
        key = self.key(torch.cat([edge, values, masks], dim=-1))
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.edge_dim)
        valid_keys = observed > 0
        all_missing = ~valid_keys.any(dim=-1, keepdim=True)
        allowed = valid_keys | all_missing
        logits = logits.masked_fill(
            ~allowed.unsqueeze(-2),
            torch.finfo(logits.dtype).min,
        )
        adjacency = torch.softmax(
            logits,
            dim=-1,
        )
        adjacency = (
            adjacency
            * (~all_missing).to(adjacency.dtype).unsqueeze(-1)
        )
        aggregated = adjacency @ node_values.unsqueeze(-1)
        return aggregated.squeeze(-1) + node_values, adjacency


class TemporalOrdinalNet(nn.Module):
    """GRU or DG-GRU with two cumulative-logit ordinal branches.

    The preprocessor concatenates standardised values and channel-observation
    indicators. Only the value channels enter the dynamic feature graph; the
    observation indicators are supplied to the recurrent layer as missingness
    features.
    """

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        use_graph: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_graph = use_graph
        if n_sequence % 2:
            raise ValueError("Sequence tensor must contain values and observation masks")
        self.n_values = n_sequence // 2
        if use_graph:
            self.graph = DynamicFeatureGraph(self.n_values)
        self.gru = nn.GRU(
            input_size=n_sequence,
            hidden_size=hidden,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden, 1)
        self.ordinal_head = TwoBranchOrdinalHead(hidden * 2)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_time:
            sequence, mask = _flip_time(sequence, mask)
        values = sequence[:, :, : self.n_values]
        observed = sequence[:, :, self.n_values :]
        if self.use_graph:
            values, _ = self.graph(values, observed)
        sequence = torch.cat([values, observed], dim=-1)
        lengths = torch.full(
            (sequence.shape[0],),
            sequence.shape[1],
            dtype=torch.int64,
            device=sequence.device,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        hidden = F.gelu(self.projection(self.norm(output)))
        risk = self.risk_head(hidden).squeeze(-1)
        final_risk = risk[:, -1]
        final_hidden = hidden[:, -1]
        static_hidden = self.static_encoder(static)
        representation = torch.cat(
            [self.dropout(final_hidden), static_hidden], dim=1
        )
        probabilities, _ = self.ordinal_head(representation, final_risk)
        return probabilities, risk, mask


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative P(y > k) logits into three class probabilities."""
    cumulative = torch.sigmoid(logits)
    first = 1.0 - cumulative[:, 0]
    middle = cumulative[:, 0] - cumulative[:, 1]
    last = cumulative[:, 1]
    probabilities = torch.stack([first, middle, last], dim=1)
    probabilities = probabilities.clamp_min(1e-7)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def temporal_regularization(
    risk: torch.Tensor,
    mask: torch.Tensor,
    lambda_monotone: float,
    lambda_smooth: float,
    max_step: float = 0.5,
    direction: str = "increasing",
) -> torch.Tensor:
    """Penalize trajectory reversals, rough changes, and unbounded risk jumps."""
    valid = (
        (mask.sum(dim=-1) > 0).float()
        if mask.dim() == 3
        else (mask > 0).float()
    )
    pair_valid = valid[:, 1:] * valid[:, :-1]
    differences = risk[:, 1:] - risk[:, :-1]
    if direction == "decreasing":
        differences = -differences
    monotone = (F.relu(-differences) * pair_valid).sum() / pair_valid.sum().clamp(
        min=1.0
    )
    smooth = (differences.square() * pair_valid).sum() / pair_valid.sum().clamp(
        min=1.0
    )
    bound = (
        F.relu(differences - max_step).square() * pair_valid
    ).sum() / pair_valid.sum().clamp(min=1.0)
    return lambda_monotone * monotone + lambda_smooth * (smooth + bound)


def add_laplace_input_noise(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Add generic Laplace noise to observed standardised sequence values.

    Training sequences concatenate standardised values with their per-channel
    observation masks. The separate ``mask`` tensor is the rank-2 or rank-3
    trajectory mask used by the network; it is validated here but is not a
    substitute for the per-channel mask stored in the sequence tensor.
    """
    if scale < 0 or not math.isfinite(scale):
        raise ValueError("Laplace input-noise scale must be finite and non-negative")
    if sequence.dim() != 3:
        raise ValueError("Sequence tensors must have rank 3")
    expected_mask_shape = (
        sequence.shape if mask.dim() == 3 else sequence.shape[:2]
    )
    if mask.dim() not in (2, 3) or tuple(mask.shape) != tuple(
        expected_mask_shape
    ):
        raise ValueError(
            "Mask tensors must match the sequence time axis as either a "
            "rank-2 trajectory mask or a rank-3 concatenated mask"
        )
    if scale == 0:
        return sequence
    n_values = sequence.shape[2] // 2
    if n_values * 2 != sequence.shape[2]:
        raise ValueError("Sequence tensor must contain values and observation masks")
    values = sequence[:, :, :n_values]
    observed = (sequence[:, :, n_values:] > 0).to(
        device=sequence.device,
        dtype=sequence.dtype,
    )
    noise = torch.distributions.Laplace(
        torch.zeros((), device=sequence.device, dtype=sequence.dtype),
        torch.full((), scale, device=sequence.device, dtype=sequence.dtype),
    ).sample(values.shape)
    perturbed = values + noise * observed
    return torch.cat([perturbed, sequence[:, :, n_values:]], dim=-1)


def weighted_nll_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class NLL with patient-level weights normalized within each batch."""
    losses = F.nll_loss(torch.log(probabilities), labels, reduction="none")
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight.to(device=losses.device, dtype=losses.dtype)
    if weights.shape != losses.shape:
        raise ValueError("Sample weights and loss values must have equal shape")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Sample weights must be finite and non-negative")
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class GRUDOrdinalNet(nn.Module):
    """GRU-D with exponential input and hidden-state decay."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        if n_sequence % 2:
            raise ValueError("Sequence tensor must contain values and observation masks")
        self.n_values = n_sequence // 2
        self.hidden = hidden
        self.register_buffer(
            "imputation_mean",
            torch.zeros(self.n_values, dtype=torch.float32),
        )
        self.gamma_x = nn.Linear(self.n_values, self.n_values)
        self.gamma_h = nn.Linear(self.n_values, hidden)
        self.gru = nn.GRUCell(self.n_values * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden, 1)
        self.ordinal_head = TwoBranchOrdinalHead(hidden * 2)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_time:
            sequence, mask = _flip_time(sequence, mask)
        values = sequence[:, :, : self.n_values]
        observed = sequence[:, :, self.n_values :]
        batch, steps, _ = values.shape
        hidden = torch.zeros(batch, self.hidden, device=values.device)
        imputation_mean = self.imputation_mean.to(
            device=values.device,
            dtype=values.dtype,
        )
        last_value = imputation_mean.unsqueeze(0).expand(batch, -1).clone()
        delta = torch.ones(batch, self.n_values, device=values.device)
        hidden_trajectory = []
        for step in range(steps):
            obs = observed[:, step, :]
            current = values[:, step, :]
            gamma_x = torch.exp(-F.softplus(self.gamma_x(delta)))
            decayed = (
                gamma_x * last_value
                + (1.0 - gamma_x) * imputation_mean.unsqueeze(0)
            )
            imputed = obs * current + (1.0 - obs) * decayed
            gamma_h = torch.exp(-F.softplus(self.gamma_h(delta)))
            hidden = gamma_h * hidden
            hidden = self.gru(torch.cat([imputed, obs], dim=1), hidden)
            hidden_trajectory.append(hidden)
            last_value = obs * current + (1.0 - obs) * last_value
            delta = torch.where(
                obs > 0,
                torch.ones_like(delta),
                delta + 1.0,
            )
        hidden_steps = torch.stack(hidden_trajectory, dim=1)
        hidden_steps = F.gelu(
            self.projection(self.norm(hidden_steps))
        )
        risk = self.risk_head(hidden_steps).squeeze(-1)
        final_hidden = hidden_steps[:, -1]
        representation = torch.cat(
            [self.dropout(final_hidden), self.static_encoder(static)],
            dim=1,
        )
        final_risk = risk[:, -1]
        probabilities, _ = self.ordinal_head(representation, final_risk)
        return probabilities, risk, mask


class TransformerOrdinalNet(nn.Module):
    """Transformer encoder with two cumulative-logit ordinal branches."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        dropout: float = 0.2,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden = hidden
        self.input_projection = nn.Linear(n_sequence, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden, 1)
        self.ordinal_head = TwoBranchOrdinalHead(hidden * 2)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_time:
            sequence, mask = _flip_time(sequence, mask)
        steps = sequence.shape[1]
        position = torch.arange(steps, device=sequence.device).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, self.hidden, 2, device=sequence.device)
            * (-math.log(10000.0) / self.hidden)
        )
        encoding = torch.zeros(steps, self.hidden, device=sequence.device)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        hidden = self.input_projection(sequence) + encoding.unsqueeze(0)
        hidden = self.encoder(hidden)
        hidden = F.gelu(self.projection(self.norm(hidden)))
        risk = self.risk_head(hidden).squeeze(-1)
        final_hidden = hidden[:, -1]
        representation = torch.cat(
            [self.dropout(final_hidden), self.static_encoder(static)],
            dim=1,
        )
        final_risk = risk[:, -1]
        probabilities, _ = self.ordinal_head(representation, final_risk)
        return probabilities, risk, mask


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
        labels: np.ndarray,
        reverse_time: bool = False,
        time_mode: str = "forward",
        time_seed: int = 0,
        sample_weight: np.ndarray | None = None,
    ):
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.masks = torch.from_numpy(masks.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        if sample_weight is None:
            sample_weight = np.ones(len(labels), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != (len(labels),):
            raise ValueError("Sample weights must have one value per patient")
        if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
            raise ValueError("Sample weights must be finite and non-negative")
        self.sample_weight = torch.from_numpy(sample_weight)
        self.reverse_time = reverse_time
        self.time_mode = time_mode
        self.time_seed = time_seed

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        sequence = self.sequences[index]
        mask = self.masks[index]
        if self.reverse_time or self.time_mode == "reverse":
            sequence, mask = _flip_time(sequence, mask)
        elif self.time_mode == "permute":
            rng = np.random.default_rng(self.time_seed + index)
            order = torch.from_numpy(
                rng.permutation(sequence.shape[0]).astype(np.int64)
            )
            sequence = sequence[order]
            mask = mask[order]
        return (
            sequence,
            mask,
            self.static[index],
            self.labels[index],
            self.sample_weight[index],
        )


@dataclass
class SequencePreprocessor:
    channel_mean: np.ndarray
    channel_std: np.ndarray
    static_median: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray

    @classmethod
    def fit(
        cls,
        sequences: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
    ) -> "SequencePreprocessor":
        channel_mean = np.array(
            [
                np.nanmean(sequences[:, :, j][masks[:, :, j] > 0])
                if np.any(masks[:, :, j] > 0)
                else 0.0
                for j in range(sequences.shape[2])
            ],
            dtype=np.float32,
        )
        channel_std = np.array(
            [
                np.nanstd(sequences[:, :, j][masks[:, :, j] > 0])
                if np.sum(masks[:, :, j] > 0) > 1
                else 1.0
                for j in range(sequences.shape[2])
            ],
            dtype=np.float32,
        )
        channel_std[~np.isfinite(channel_std) | (channel_std < 1e-6)] = 1.0
        static_median = np.nanmedian(static, axis=0)
        static_median[~np.isfinite(static_median)] = 0.0
        filled = np.where(np.isfinite(static), static, static_median)
        static_mean = np.mean(filled, axis=0)
        static_std = np.std(filled, axis=0)
        static_std[static_std < 1e-6] = 1.0
        return cls(
            channel_mean=channel_mean,
            channel_std=channel_std,
            static_median=static_median,
            static_mean=static_mean,
            static_std=static_std,
        )

    def transform_sequence(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        *,
        mask_mode: str = "any_channel",
        gcs_channel_indices: tuple[int, ...] = (0, 1, 2, 3),
    ) -> tuple[np.ndarray, np.ndarray]:
        standardized = (sequences - self.channel_mean[None, None, :]) / (
            self.channel_std[None, None, :]
        )
        out = standardized.copy().astype(np.float32)
        for i in range(len(out)):
            valid_hours = masks[i].sum(axis=1) > 0
            if valid_hours.sum() == 0:
                out[i] = 0.0
                continue
            frame = pd.DataFrame(out[i])
            frame = frame.ffill()
            out[i] = frame.to_numpy(dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if mask_mode == "any_channel":
            trajectory_mask = (masks.sum(axis=2) > 0).astype(np.float32)
        elif mask_mode == "gcs":
            valid_indices = [
                index
                for index in gcs_channel_indices
                if 0 <= int(index) < masks.shape[2]
            ]
            if not valid_indices:
                raise ValueError(
                    "GCS mask mode requires at least one valid GCS channel index"
                )
            trajectory_mask = (
                masks[:, :, valid_indices].sum(axis=2) > 0
            ).astype(np.float32)
        elif mask_mode == "none":
            trajectory_mask = np.zeros(masks.shape[:2], dtype=np.float32)
        else:
            raise ValueError(f"Unknown sequence observation-mask mode: {mask_mode}")
        observed_channels = (masks > 0).astype(np.float32)
        return (
            np.concatenate([out, observed_channels], axis=2),
            trajectory_mask,
        )

    def transform_static(self, static: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(static), static, self.static_median)
        standardized = (filled - self.static_mean) / self.static_std
        return np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)


def build_network(
    architecture: str,
    n_sequence: int,
    n_static: int,
    *,
    hidden: int = 64,
    use_graph: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """Create one sequence architecture for the shared training loop."""
    if architecture in {"gru", "dggru"}:
        return TemporalOrdinalNet(
            n_sequence=n_sequence,
            n_static=n_static,
            hidden=hidden,
            use_graph=use_graph,
            dropout=dropout,
        )
    if architecture == "grud":
        return GRUDOrdinalNet(
            n_sequence=n_sequence,
            n_static=n_static,
            hidden=hidden,
            dropout=dropout,
        )
    if architecture == "transformer":
        return TransformerOrdinalNet(
            n_sequence=n_sequence,
            n_static=n_static,
            hidden=hidden,
            dropout=dropout,
        )
    raise ValueError(f"Unknown sequence architecture: {architecture}")


def train_temporal_model(
    train_sequences: np.ndarray,
    train_masks: np.ndarray,
    train_static: np.ndarray,
    train_labels: np.ndarray,
    valid_sequences: np.ndarray,
    valid_masks: np.ndarray,
    valid_static: np.ndarray,
    valid_labels: np.ndarray,
    *,
    use_graph: bool,
    use_regularization: bool,
    architecture: str = "dggru",
    regularization_direction: str = "increasing",
    lambda_monotone: float = 0.05,
    lambda_smooth: float = 0.02,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    dropout: float = 0.2,
    seed: int = 42,
    patience: int = 6,
    reverse_time_augmentation: bool = False,
    time_mode: str = "forward",
    sample_weight: np.ndarray | None = None,
    laplace_only: bool = False,
    laplace_input_scale: float = 0.01,
    return_model: bool = False,
):
    """Train one network and return validation probabilities plus trajectory.

    Early stopping uses unregularized validation NLL in every arm. The temporal
    penalty remains part of the training objective but does not become an
    additional model-selection target for regularized arms.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = build_network(
        architecture,
        n_sequence=train_sequences.shape[2],
        n_static=train_static.shape[1],
        hidden=hidden,
        use_graph=use_graph,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_weights = None
    train_dataset = SequenceDataset(
        train_sequences,
        train_masks,
        train_static,
        train_labels,
        reverse_time=reverse_time_augmentation,
        time_mode=time_mode,
        time_seed=seed,
        sample_weight=sample_weight,
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_dataset = SequenceDataset(
        valid_sequences,
        valid_masks,
        valid_static,
        valid_labels,
        time_mode=time_mode,
        time_seed=seed + 100_000,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        batches = 0
        for sequence, mask, static, labels, weights in loader:
            sequence = sequence.to(device)
            mask = mask.to(device)
            static = static.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            if laplace_only:
                sequence = add_laplace_input_noise(
                    sequence,
                    mask,
                    scale=laplace_input_scale,
                )
            probabilities, risk, valid = model(sequence, mask, static)
            loss = weighted_nll_loss(probabilities, labels, weights)
            if use_regularization and not laplace_only:
                loss = loss + temporal_regularization(
                    risk,
                    valid,
                    lambda_monotone=lambda_monotone,
                    lambda_smooth=lambda_smooth,
                    direction=regularization_direction,
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        valid_loss = evaluate_loss(
            model,
            valid_loader,
            class_weights,
            device=device,
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": total / max(batches, 1), "valid_loss": valid_loss}
        )
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    probabilities, trajectories = predict_temporal_model(
        model,
        valid_sequences,
        valid_masks,
        valid_static,
        batch_size=256,
    )
    if return_model:
        return probabilities, trajectories, history, model
    return probabilities, trajectories, history


def evaluate_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor | None,
    *,
    include_regularization: bool = False,
    lambda_monotone: float = 0.0,
    lambda_smooth: float = 0.0,
    regularization_direction: str = "increasing",
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    weight_total = 0.0
    regularization_total = 0.0
    count = 0
    with torch.no_grad():
        for sequence, mask, static, labels, weights in loader:
            sequence = sequence.to(device)
            mask = mask.to(device)
            static = static.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            probabilities, risk, valid = model(sequence, mask, static)
            losses = F.nll_loss(
                torch.log(probabilities),
                labels,
                reduction="none",
            )
            weighted_loss += float((losses * weights).sum().item())
            weight_total += float(weights.sum().item())
            if include_regularization:
                penalty = temporal_regularization(
                    risk,
                    valid,
                    lambda_monotone=lambda_monotone,
                    lambda_smooth=lambda_smooth,
                    direction=regularization_direction,
                )
                regularization_total += float(penalty.item()) * len(labels)
            count += len(labels)
    nll = weighted_loss / max(weight_total, 1e-12)
    return nll + regularization_total / max(count, 1)


def predict_temporal_model(
    model: nn.Module,
    sequences: np.ndarray,
    masks: np.ndarray,
    static: np.ndarray,
    batch_size: int = 256,
    time_mode: str = "forward",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            stop = min(start + batch_size, len(sequences))
            sequence = torch.from_numpy(sequences[start:stop].astype(np.float32))
            mask = torch.from_numpy(masks[start:stop].astype(np.float32))
            static_batch = torch.from_numpy(static[start:stop].astype(np.float32))
            if time_mode == "reverse":
                sequence, mask = _flip_time(sequence, mask)
            elif time_mode == "permute":
                reordered = []
                reordered_masks = []
                for row in range(len(sequence)):
                    rng = np.random.default_rng(7_000_000 + start + row)
                    order = torch.from_numpy(
                        rng.permutation(sequence.shape[1]).astype(np.int64)
                    )
                    reordered.append(sequence[row, order])
                    reordered_masks.append(mask[row, order])
                sequence = torch.stack(reordered)
                mask = torch.stack(reordered_masks)
            probabilities, risk, _ = model(sequence, mask, static_batch)
            probability_parts.append(probabilities.numpy())
            trajectory_parts.append(risk.numpy())
    return np.concatenate(probability_parts), np.concatenate(trajectory_parts)


def fit_full_temporal_model(
    sequences: np.ndarray,
    masks: np.ndarray,
    static: np.ndarray,
    labels: np.ndarray,
    *,
    use_graph: bool,
    use_regularization: bool,
    architecture: str = "dggru",
    seed: int,
    regularization_direction: str = "increasing",
    lambda_monotone: float = 0.05,
    lambda_smooth: float = 0.02,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    dropout: float = 0.2,
    time_mode: str = "forward",
    sample_weight: np.ndarray | None = None,
    laplace_only: bool = False,
    laplace_input_scale: float = 0.01,
) -> nn.Module:
    """Fit on the complete development cohort for external transport."""
    torch.manual_seed(seed)
    model = build_network(
        architecture,
        n_sequence=sequences.shape[2],
        n_static=static.shape[1],
        hidden=hidden,
        use_graph=use_graph,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_weights = None
    dataset = SequenceDataset(
        sequences,
        masks,
        static,
        labels,
        time_mode=time_mode,
        time_seed=seed,
        sample_weight=sample_weight,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for sequence, mask, static_batch, batch_labels, weights in loader:
            if laplace_only:
                sequence = add_laplace_input_noise(
                    sequence,
                    mask,
                    scale=laplace_input_scale,
                )
            probabilities, risk, valid = model(
                sequence, mask, static_batch
            )
            loss = weighted_nll_loss(probabilities, batch_labels, weights)
            if use_regularization and not laplace_only:
                loss = loss + temporal_regularization(
                    risk,
                    valid,
                    lambda_monotone=lambda_monotone,
                    lambda_smooth=lambda_smooth,
                    direction=regularization_direction,
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model
