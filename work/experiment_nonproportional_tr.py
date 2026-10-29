"""Test representation-dependent ordinal thresholds in DG-GRU + TR."""
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

os.environ.setdefault("ICH_TORCH_THREADS", "3")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


class NonProportionalTemporalOrdinalNet(dm.TemporalOrdinalNet):
    """Ordinal head whose cumulative-logit gap varies across patients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gap_head = nn.Linear(kwargs.get("hidden", 64) * 2, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_time:
            sequence = torch.flip(sequence, dims=[1])
            mask = torch.flip(mask, dims=[1])
        if self.use_graph:
            sequence, _ = self.graph(sequence)

        observed_hours = (
            mask.sum(dim=-1) > 0 if mask.dim() == 3 else mask > 0
        )
        has_observation = observed_hours.any(dim=1)
        last_hour = (
            observed_hours.shape[1]
            - 1
            - torch.flip(observed_hours, dims=[1]).to(torch.int64).argmax(dim=1)
        )
        last_hour = torch.where(
            has_observation,
            last_hour,
            torch.zeros_like(last_hour),
        )
        lengths = (last_hour + 1).clamp(min=1, max=sequence.shape[1])
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
        last_index = (lengths - 1).clamp(min=0).unsqueeze(1)
        final_risk = risk.gather(1, last_index).squeeze(1)
        final_hidden = hidden.gather(
            1, last_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        ).squeeze(1)
        static_hidden = self.static_encoder(static)
        representation = torch.cat(
            [self.dropout(final_hidden), static_hidden], dim=1
        )

        first_logit = (
            final_risk + self.ordinal_score(representation).squeeze(1)
        )
        cumulative_gap = F.softplus(
            self.gap_head(representation).squeeze(1)
        ) + 1e-4
        logits = torch.stack(
            [first_logit, first_logit - cumulative_gap],
            dim=1,
        )
        return dm.ordinal_probabilities(logits), risk, mask


# This assignment is intentionally at import time so Windows process workers
# instantiate the experimental head when te.run_cv creates its process pool.
dm.TemporalOrdinalNet = NonProportionalTemporalOrdinalNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--worker-threads", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--metric-workers", type=int, default=2)
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    development = te.load_dataset("mimic4")
    model_names = ("dggru_tr",)
    prefix = "experiment_nonproportional_tr"

    predictions, _, cv_best_epochs, _, _ = te.run_cv(
        development,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=model_names,
        lambda_monotone=float(config["lambda_monotone"]),
        lambda_smooth=float(config["lambda_smooth"]),
        model_seeds=tuple(range(args.seed_count)),
        workers=args.workers,
        worker_threads=args.worker_threads,
        repeats=args.cv_repeats,
        repeat_prefix=prefix,
    )
    repeat_path = te.RESULTS / "predictions" / f"{prefix}_repeats.npz"
    metrics = te.evaluate_repeated_cv(
        development,
        repeat_path,
        model_names,
        args.cv_repeats,
        replicates=args.replicates,
        seed=seed,
        metric_workers=args.metric_workers,
    )
    metrics["cv_repeat_variability"] = te.repeated_cv_variability(
        development,
        repeat_path,
        model_names,
        args.cv_repeats,
    )

    output = {
        "purpose": (
            "Fixed non-proportional ordinal-head screen. The two cumulative "
            "logits share a patient score, but their positive gap is learned "
            "from the patient representation. All splits, regularization "
            "weights, epochs, and model seeds match the formal DG-GRU + TR run."
        ),
        "development_n": int(len(development.labels)),
        "outcomes": {
            str(grade): int(np.sum(development.labels == grade))
            for grade in (0, 1, 2)
        },
        "models": list(model_names),
        "folds": args.folds,
        "cv_repeats": args.cv_repeats,
        "epochs": args.epochs,
        "seed_count": args.seed_count,
        "bootstrap_replicates": args.replicates,
        "lambda_monotone": float(config["lambda_monotone"]),
        "lambda_smooth": float(config["lambda_smooth"]),
        "cv_best_epochs": cv_best_epochs,
        "metrics": metrics,
    }
    output_dir = te.RESULTS / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "nonproportional_tr_predictions.npz",
        labels=development.labels,
        stay_id=development.cohort["stay_id"].to_numpy(),
        subject_key=development.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    output_path = output_dir / "nonproportional_tr.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
