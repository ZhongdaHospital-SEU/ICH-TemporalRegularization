"""Test temporal regularization applied directly to ordinal output scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("ICH_TORCH_THREADS", "12")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


class OutputCoupledTemporalOrdinalNet(dm.TemporalOrdinalNet):
    """Apply the temporal penalty to the ordinal score used for prediction."""

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
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.gru(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        hidden = F.gelu(self.projection(self.norm(output)))
        risk_steps = self.risk_head(hidden).squeeze(-1)
        static_hidden = self.static_encoder(static)
        static_context = static_hidden.unsqueeze(1).expand(
            -1, hidden.shape[1], -1
        )
        score_input = torch.cat(
            [self.dropout(hidden), static_context],
            dim=-1,
        )
        score_steps = (
            risk_steps + self.ordinal_score(score_input).squeeze(-1)
        )
        last_index = (lengths - 1).clamp(min=0).unsqueeze(1)
        final_score = score_steps.gather(1, last_index).squeeze(1)
        increments = F.softplus(self.threshold_increments)
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
    parser.add_argument(
        "--models",
        default="dggru,dggru_tr",
        help="Comma-separated temporal model names.",
    )
    parser.add_argument(
        "--static-mode",
        choices=("current", "full"),
        default="current",
    )
    args = parser.parse_args()

    te.ensure_dirs()
    config = te.load_config()
    seed = int(config["random_seed"])
    selected_models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    unsupported = [
        name
        for name in selected_models
        if te.MODEL_META[name]["kind"] != "temporal"
    ]
    if unsupported:
        raise ValueError(
            "This experiment is restricted to temporal models: "
            + ", ".join(unsupported)
        )

    development_raw = te.load_dataset("mimic4")
    if args.static_mode == "full":
        feature_columns = list(development_raw.static_all.columns)
        te.DEEP_STATIC = feature_columns
    else:
        feature_columns = list(development_raw.static_all.columns)
    development = te.load_dataset(
        "mimic4",
        feature_columns=feature_columns,
    )

    dm.TemporalOrdinalNet = OutputCoupledTemporalOrdinalNet
    prefix = f"experiment_output_coupled_{args.static_mode}"
    predictions, _, cv_best_epochs, _, _ = te.run_cv(
        development,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=selected_models,
        lambda_monotone=float(config["lambda_monotone"]),
        lambda_smooth=float(config["lambda_smooth"]),
        workers=1,
        worker_threads=1,
        model_seeds=tuple(range(args.seed_count)),
        repeats=args.cv_repeats,
        repeat_prefix=prefix,
    )
    repeat_path = te.RESULTS / "predictions" / f"{prefix}_repeats.npz"
    if args.cv_repeats > 1:
        metrics = te.evaluate_repeated_cv(
            development,
            repeat_path,
            selected_models,
            args.cv_repeats,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )
        metrics["cv_repeat_variability"] = te.repeated_cv_variability(
            development,
            repeat_path,
            selected_models,
            args.cv_repeats,
        )
    else:
        metrics = te.evaluate_predictions(
            development,
            predictions,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )

    output_dir = te.RESULTS / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"output_coupled_{args.static_mode}_predictions.npz",
        labels=development.labels,
        stay_id=development.cohort["stay_id"].to_numpy(),
        subject_key=development.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Sensitivity experiment: apply temporal regularization to the "
            "hourly ordinal score used for final prediction rather than only "
            "to the auxiliary risk trajectory."
        ),
        "development_n": len(development.labels),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "models": list(selected_models),
        "folds": args.folds,
        "cv_repeats": args.cv_repeats,
        "epochs": args.epochs,
        "seed_count": args.seed_count,
        "bootstrap_replicates": args.replicates,
        "cv_best_epochs": cv_best_epochs,
        "metrics": metrics,
    }
    output_path = output_dir / f"output_coupled_{args.static_mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
