"""Screen a GCS-anchored readout for the dynamic-graph model.

The endpoint is defined on the Glasgow Coma Scale, yet the shipped
``TemporalOrdinalNet`` reads the hidden state and the shared ordinal score at
the last padded hour. About 40% of stays have no GCS value in that hour, so the
readout is then driven by non-neurological channels. This script keeps the
network, loss, regularisation and partitions unchanged and moves only the
readout to the last hour with an observed GCS value, falling back to the last
padded hour when GCS is never observed.

The forward pass mirrors ``src.ich_dynamic.models.TemporalOrdinalNet.forward``;
``src`` is not modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as md
from src.ich_dynamic import train_evaluate as te

GCS_TOTAL_INDEX = 3


def _gcs_anchored_forward(
    self,
    sequence: torch.Tensor,
    mask: torch.Tensor,
    static: torch.Tensor,
    reverse_time: bool = False,
):
    if reverse_time:
        sequence, mask = md._flip_time(sequence, mask)
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
    risk = self.risk_head(hidden).squeeze(-1)
    steps = torch.arange(hidden.shape[1], device=hidden.device)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    if observed.shape[2] > GCS_TOTAL_INDEX:
        gcs_observed = observed[:, :, GCS_TOTAL_INDEX] > 0
        last_gcs = (gcs_observed.long() * (steps + 1)).max(dim=1).values - 1
        readout = torch.where(
            gcs_observed.any(dim=1),
            last_gcs,
            steps[-1].expand(hidden.shape[0]),
        ).clamp_min(0)
    else:
        readout = steps[-1].expand(hidden.shape[0])
    final_hidden = hidden[batch, readout]
    final_risk = risk[batch, readout]
    static_hidden = self.static_encoder(static)
    representation = torch.cat(
        [self.dropout(final_hidden), static_hidden], dim=1
    )
    probabilities, _ = self.ordinal_head(representation, final_risk)
    return probabilities, risk, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="readout_gcs_anchor")
    parser.add_argument("--models", default="dggru_tr")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()

    md.TemporalOrdinalNet.forward = _gcs_anchored_forward

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    dataset = te.load_dataset("mimic4")
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    predictions, _, best_epochs, selected_lambdas, _ = te.run_cv(
        dataset,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=models,
        lambda_monotone=1.5,
        lambda_smooth=0.6,
        model_seeds=tuple(range(args.seeds)),
        workers=args.workers,
        worker_threads=args.threads,
        repeats=args.repeats,
        repeat_prefix=args.tag,
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
        output_dir / f"{args.tag}_predictions.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        subject_key=dataset.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Exploratory frozen-partition screen reading the ordinal head at "
            "the last hour with an observed GCS value instead of the last "
            "padded hour. Network, loss, regularisation and partitions are "
            "unchanged. External cohorts were not read."
        ),
        "models": list(models),
        "readout": "last_observed_gcs_hour",
        "gcs_channel_index": GCS_TOTAL_INDEX,
        "folds": args.folds,
        "cv_repeats": args.repeats,
        "epochs": args.epochs,
        "model_seeds": list(range(args.seeds)),
        "workers": args.workers,
        "worker_threads": args.threads,
        "bootstrap_replicates": args.replicates,
        "lambda_monotone": 1.5,
        "lambda_smooth": 0.6,
        "cv_best_epochs": best_epochs,
        "cv_selected_lambdas": selected_lambdas,
        "metrics": metrics,
    }
    output_path = output_dir / f"{args.tag}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    for model in models:
        point = metrics[model]["point"]
        print(
            f"{model:14s} macro={point['macro_auc']:.4f} "
            f"grade1={point['auc_grade_1']:.4f} "
            f"brier={point['multiclass_brier']:.4f}"
        )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
