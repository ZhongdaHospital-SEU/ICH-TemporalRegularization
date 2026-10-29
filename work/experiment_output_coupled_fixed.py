"""Output-coupled temporal regularization: penalty on the ordinal score itself."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("ICH_TORCH_THREADS", "12")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


class OutputCoupledTemporalOrdinalNet(dm.TemporalOrdinalNet):
    """Temporal penalty applied to the per-hour ordinal score.

    The unmodified model regularises ``risk_head(hidden_t)``, a scalar that
    enters the prediction only through the last hour. Here every hour gets a
    full ordinal score from the same linear map (hidden_t, static), and a
    single linear map converts the score to cumulative logits. The penalty
    therefore constrains the function that generates the reported
    probabilities rather than an auxiliary side output.
    """

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        use_graph: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__(n_sequence, n_static, hidden, use_graph, dropout)
        self.ordinal_score = nn.Linear(hidden * 2, 1)
        self.threshold_increments = nn.Parameter(torch.zeros(dm.N_THRESHOLDS))

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
        lengths = torch.full(
            (sequence.shape[0],),
            sequence.shape[1],
            dtype=torch.int64,
            device=sequence.device,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=sequence.shape[1]
        )
        hidden = F.gelu(self.projection(self.norm(output)))
        static_hidden = self.static_encoder(static)
        context = static_hidden.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        score_steps = (
            self.risk_head(hidden).squeeze(-1)
            + self.ordinal_score(torch.cat([self.dropout(hidden), context], -1)).squeeze(-1)
        )
        final_score = score_steps[:, -1]
        increments = F.softplus(self.threshold_increments) + 1e-3
        thresholds = torch.cumsum(increments, dim=0)
        logits = final_score.unsqueeze(1) - thresholds.unsqueeze(0)
        probabilities = dm.ordinal_probabilities(logits)
        return probabilities, score_steps, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--metric-workers", type=int, default=2)
    parser.add_argument("--models", default="dggru_tr")
    parser.add_argument("--lambda-monotone", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    models = tuple(p.strip() for p in args.models.split(",") if p.strip())

    development = te.load_dataset("mimic4", feature_columns=list(te.CORE_STATIC))
    dm.TemporalOrdinalNet = OutputCoupledTemporalOrdinalNet
    prefix = f"experiment_output_coupled{args.suffix}"
    predictions, _, cv_best_epochs, _, _ = te.run_cv(
        development,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=models,
        lambda_monotone=float(
            args.lambda_monotone if args.lambda_monotone is not None else config["lambda_monotone"]
        ),
        lambda_smooth=float(
            args.lambda_smooth if args.lambda_smooth is not None else config["lambda_smooth"]
        ),
        workers=1,
        worker_threads=12,
        model_seeds=tuple(range(args.seed_count)),
        repeats=args.cv_repeats,
        repeat_prefix=prefix,
    )
    repeat_path = te.RESULTS / "predictions" / f"{prefix}_repeats.npz"
    metrics = te.evaluate_repeated_cv(
        development,
        repeat_path,
        models,
        args.cv_repeats,
        replicates=args.replicates,
        seed=seed,
        metric_workers=args.metric_workers,
    )
    payload = {
        "purpose": "Output-coupled temporal regularization screen on frozen partitions.",
        "models": list(models),
        "lambda_monotone": float(
            args.lambda_monotone if args.lambda_monotone is not None else config["lambda_monotone"]
        ),
        "lambda_smooth": float(
            args.lambda_smooth if args.lambda_smooth is not None else config["lambda_smooth"]
        ),
        "cv_best_epochs": cv_best_epochs,
        "metrics": metrics,
    }
    out = te.RESULTS / "experiments" / f"{prefix}.json"
    out.write_text(json.dumps(payload, indent=1))
    for name in models:
        point = metrics[name]["point"]
        print(
            f"{name}: macro {point['macro_auc']:.4f}  "
            f"g1 {point['auc_grade_1']:.4f}  "
            f"any {point['auc_any_deterioration']:.4f}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
