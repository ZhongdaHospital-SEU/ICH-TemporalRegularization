"""Experimental temporal models for frozen-partition improvement screens.

The classes here are intentionally kept outside ``src`` so that completed
formal runs retain their source hash.  They preserve the public forward
signature used by ``src.ich_dynamic.models.train_temporal_model``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ich_dynamic import models as dm


class LearnedGraphV2(nn.Module):
    """Value-transformed, gated, observation-aware feature graph."""

    def __init__(self, n_features: int, edge_dim: int = 16):
        super().__init__()
        self.edge_dim = edge_dim
        self.edge_embedding = nn.Parameter(
            torch.randn(n_features, edge_dim) * 0.1
        )
        self.query = nn.Linear(edge_dim + 2, edge_dim)
        self.key = nn.Linear(edge_dim + 2, edge_dim)
        self.value_projection = nn.Linear(n_features, n_features)
        self.gate = nn.Linear(n_features * 2, n_features)
        self.norm = nn.LayerNorm(n_features)

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
        edge = self.edge_embedding[None, None].expand(
            batch, steps, features, -1
        )
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
        adjacency = torch.softmax(logits, dim=-1)
        adjacency = (
            adjacency
            * (~all_missing).to(adjacency.dtype).unsqueeze(-1)
        )
        transformed = self.value_projection(node_values)
        aggregated = (adjacency @ transformed.unsqueeze(-1)).squeeze(-1)
        gate = torch.sigmoid(
            self.gate(torch.cat([node_values, aggregated], dim=-1))
        )
        output = gate * aggregated + (1.0 - gate) * node_values
        return self.norm(output), adjacency


class MaskedPoolingTemporalNet(nn.Module):
    """Temporal model with masked mean/max/last/attention pooling."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        use_graph: bool = False,
        dropout: float = 0.2,
        learned_graph: bool = False,
    ):
        super().__init__()
        self.use_graph = use_graph
        if n_sequence % 2:
            raise ValueError(
                "Sequence tensor must contain values and observation masks"
            )
        self.n_values = n_sequence // 2
        if use_graph:
            self.graph = (
                LearnedGraphV2(self.n_values)
                if learned_graph
                else dm.DynamicFeatureGraph(self.n_values)
            )
        self.gru = nn.GRU(
            input_size=n_sequence,
            hidden_size=hidden,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.pool_score = nn.Linear(hidden, 1)
        self.pool_projection = nn.Linear(hidden * 4, hidden)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.static_skip = nn.Linear(n_static, hidden)
        self.risk_head = nn.Linear(hidden, 1)
        self.ordinal_head = dm.TwoBranchOrdinalHead(hidden * 2)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_time:
            sequence, mask = dm._flip_time(sequence, mask)
        values = sequence[:, :, : self.n_values]
        observed = sequence[:, :, self.n_values :]
        if self.use_graph:
            values, _ = self.graph(values, observed)
        sequence = torch.cat([values, observed], dim=-1)
        output, _ = self.gru(sequence)
        hidden = F.gelu(self.projection(self.norm(output)))
        risk = self.risk_head(hidden).squeeze(-1)

        if mask.dim() == 3:
            observed_hours = mask.sum(dim=-1) > 0
        else:
            observed_hours = mask > 0
        has_observation = observed_hours.any(dim=1)
        valid = observed_hours.unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        mean_pool = (hidden * valid).sum(dim=1) / denominator
        max_pool = hidden.masked_fill(
            ~valid, torch.finfo(hidden.dtype).min
        ).max(dim=1).values
        last_index = (
            observed_hours.shape[1]
            - 1
            - torch.flip(observed_hours, dims=[1]).to(torch.int64).argmax(dim=1)
        )
        last_index = torch.where(
            has_observation,
            last_index,
            torch.zeros_like(last_index),
        )
        gather_index = last_index[:, None, None].expand(
            -1, 1, hidden.shape[-1]
        )
        last_pool = hidden.gather(1, gather_index).squeeze(1)
        scores = self.pool_score(hidden).squeeze(-1)
        scores = scores.masked_fill(
            ~observed_hours, torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=1)
        attention = torch.where(
            has_observation[:, None],
            attention,
            torch.zeros_like(attention),
        )
        attention_pool = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        mean_pool = torch.where(
            has_observation[:, None],
            mean_pool,
            torch.zeros_like(mean_pool),
        )
        max_pool = torch.where(
            has_observation[:, None],
            max_pool,
            torch.zeros_like(max_pool),
        )
        last_pool = torch.where(
            has_observation[:, None],
            last_pool,
            torch.zeros_like(last_pool),
        )
        attention_pool = torch.where(
            has_observation[:, None],
            attention_pool,
            torch.zeros_like(attention_pool),
        )
        pooled = self.pool_projection(
            torch.cat(
                [last_pool, mean_pool, max_pool, attention_pool],
                dim=1,
            )
        )
        static_hidden = (
            self.static_encoder(static) + self.static_skip(static)
        )
        representation = torch.cat(
            [self.dropout(pooled), static_hidden], dim=1
        )
        final_risk = risk.gather(1, last_index[:, None]).squeeze(1)
        final_risk = torch.where(
            has_observation,
            final_risk,
            torch.zeros_like(final_risk),
        )
        probabilities, _ = self.ordinal_head(representation, final_risk)
        return probabilities, risk, mask


class LearnedGraphMaskedPoolingNet(MaskedPoolingTemporalNet):
    """Masked-pooling model with the learned value-transformed graph."""

    def __init__(self, *args, **kwargs):
        kwargs["learned_graph"] = True
        super().__init__(*args, **kwargs)
