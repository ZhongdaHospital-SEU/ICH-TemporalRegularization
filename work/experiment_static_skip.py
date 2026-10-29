"""Screen direct static residual links into the ordinal head."""
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

os.environ.setdefault("ICH_TORCH_THREADS", "2")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


class StaticSkipTemporalOrdinalNet(dm.TemporalOrdinalNet):
    """Add a linear static residual to the ordinal score or its thresholds."""

    skip_mode = os.environ.get("ICH_STATIC_SKIP_MODE", "centre")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_static = int(kwargs["n_static"] if "n_static" in kwargs else args[1])
        self.static_centre = nn.Linear(n_static, 1)
        if self.skip_mode == "full":
            self.static_gap = nn.Linear(n_static, dm.N_THRESHOLDS)

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
        static_hidden = self.static_encoder(static)
        representation = torch.cat(
            [self.dropout(hidden[:, -1]), static_hidden], dim=1
        )
        centre = (
            final_risk
            + self.ordinal_head.centre(representation).squeeze(1)
            + self.static_centre(static).squeeze(1)
        )
        gaps = self.ordinal_head.branch_gaps(representation)
        if self.skip_mode == "full":
            gaps = gaps + self.static_gap(static)
        probabilities = dm.two_branch_probabilities(centre, gaps)
        return probabilities, risk, mask


# Windows spawn workers import this module without executing main(). The
# assignment must therefore happen at import time so every fold uses the
# experimental decoder.
dm.TemporalOrdinalNet = StaticSkipTemporalOrdinalNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--mode", choices=("centre", "full"), default="centre")
    parser.add_argument("--static", choices=("core", "all"), default="core")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--lambda-monotone", type=float, default=0.15)
    parser.add_argument("--lambda-smooth", type=float, default=0.06)
    parser.add_argument("--models", default="dggru")
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    if any(te.MODEL_META[name]["kind"] != "temporal" for name in models):
        raise ValueError("Static-skip screen is restricted to temporal models")

    if args.static == "core":
        feature_columns = list(te.DEEP_STATIC)
        dataset = te.load_dataset("mimic4", feature_columns=feature_columns)
    else:
        dataset = te.load_dataset("mimic4")
        feature_columns = list(dataset.static_all.columns)

    os.environ["ICH_STATIC_SKIP_MODE"] = args.mode
    StaticSkipTemporalOrdinalNet.skip_mode = args.mode
    dm.TemporalOrdinalNet = StaticSkipTemporalOrdinalNet

    prefix = f"correction_{args.tag}"
    predictions, _, best_epochs, selected_lambdas, _ = te.run_cv(
        dataset,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=models,
        lambda_monotone=args.lambda_monotone,
        lambda_smooth=args.lambda_smooth,
        model_seeds=tuple(range(args.seeds)),
        workers=args.workers,
        worker_threads=args.threads,
        repeats=args.repeats,
        repeat_prefix=prefix,
    )
    metrics = te.evaluate_predictions(
        dataset,
        predictions,
        replicates=args.replicates,
        seed=seed,
        metric_workers=1,
    )

    output_dir = te.RESULTS / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{prefix}_predictions.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        subject_key=dataset.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Exploratory static-residual screen on frozen development CV "
            "partitions. External cohorts were not used."
        ),
        "skip_mode": args.mode,
        "static_subset": args.static,
        "feature_count": len(feature_columns),
        "models": list(models),
        "folds": args.folds,
        "cv_repeats": args.repeats,
        "epochs": args.epochs,
        "model_seeds": list(range(args.seeds)),
        "bootstrap_replicates": args.replicates,
        "lambda_monotone": args.lambda_monotone,
        "lambda_smooth": args.lambda_smooth,
        "cv_best_epochs": best_epochs,
        "cv_selected_lambdas": selected_lambdas,
        "metrics": metrics,
    }
    output_path = output_dir / f"{prefix}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
