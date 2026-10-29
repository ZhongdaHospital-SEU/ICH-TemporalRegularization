"""Screen targeted corrections on the frozen development CV partitions.

The arms are deliberately narrow:

* ``monitor_drop`` removes the 24-hour monitoring-intensity summary, which is
  correlated with how often GCS is measured after the landmark.
* ``gcs_sequence`` feeds only the four GCS channels to the temporal model.
* ``auc_stop`` replaces inner validation NLL with macro one-vs-rest AUC for
  early stopping and epoch selection.
* ``cumulative_loss`` trains the existing ordinal head with two binary
  cumulative risks rather than three-class NLL.

External cohorts are not read. The script is a screening tool; it is not a
formal replacement for the registered run.
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
from sklearn.metrics import roc_auc_score

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


def _macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < te.N_CLASSES:
        return float("nan")
    return float(
        np.mean(
            [
                roc_auc_score(
                    (labels == grade).astype(int),
                    probabilities[:, grade],
                )
                for grade in range(te.N_CLASSES)
            ]
        )
    )


def _install_auc_early_stopping() -> None:
    """Use negative inner macro-AUC as the epoch-selection loss."""

    def evaluate_auc_loss(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        class_weights: torch.Tensor | None,
        *,
        include_regularization: bool = False,
        lambda_monotone: float = 0.0,
        lambda_smooth: float = 0.0,
        regularization_direction: str = "increasing",
        device: torch.device,
    ) -> float:
        del (
            class_weights,
            include_regularization,
            lambda_monotone,
            lambda_smooth,
            regularization_direction,
        )
        model.eval()
        label_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        with torch.no_grad():
            for sequence, mask, static, labels, _ in loader:
                probabilities, _, _ = model(
                    sequence.to(device),
                    mask.to(device),
                    static.to(device),
                )
                label_parts.append(labels.numpy())
                probability_parts.append(probabilities.numpy())
        labels = np.concatenate(label_parts)
        probabilities = np.concatenate(probability_parts)
        auc = _macro_auc(labels, probabilities)
        return -auc if np.isfinite(auc) else float("inf")

    dm.evaluate_loss = evaluate_auc_loss
    te.evaluate_loss = evaluate_auc_loss


def _install_cumulative_loss() -> None:
    """Train with BCE on P(grade >= 1) and P(grade >= 2)."""

    def cumulative_loss(
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        any_target = (labels >= 1).to(probabilities.dtype)
        severe_target = (labels >= 2).to(probabilities.dtype)
        any_probability = probabilities[:, 1:].sum(dim=1).clamp(1e-7, 1 - 1e-7)
        severe_probability = probabilities[:, 2].clamp(1e-7, 1 - 1e-7)
        losses = F.binary_cross_entropy(
            any_probability,
            any_target,
            reduction="none",
        ) + F.binary_cross_entropy(
            severe_probability,
            severe_target,
            reduction="none",
        )
        if sample_weight is None:
            return losses.mean()
        weights = sample_weight.to(
            device=losses.device,
            dtype=losses.dtype,
        )
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)

    dm.weighted_nll_loss = cumulative_loss
    te.weighted_nll_loss = cumulative_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "baseline",
            "monitor_drop",
            "gcs_sequence",
            "auc_stop",
            "cumulative_loss",
            "auc_stop_cumulative",
            "monitor_drop_gcs_auc",
            "monitor_drop_nomask",
        ),
    )
    parser.add_argument("--models", default="xgb_summary,dggru_tr")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--lambda-monotone", type=float, default=1.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.6)
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    unsupported = [
        name for name in models if te.MODEL_META[name]["kind"] == "stacking"
    ]
    if unsupported:
        raise ValueError("Stacking is not supported in this screen")

    feature_columns = list(te.DEEP_STATIC)
    dataset = te.load_dataset("mimic4", feature_columns=feature_columns)
    if args.arm in {
        "monitor_drop",
        "monitor_drop_gcs_auc",
        "monitor_drop_nomask",
    }:
        feature_columns = [
            column
            for column in feature_columns
            if column != "monitoring_intensity_24h"
        ]
        dataset = te.load_dataset("mimic4", feature_columns=feature_columns)
    if args.arm == "monitor_drop_nomask":
        dataset.masks = np.ones_like(dataset.masks, dtype=np.float32)
    if args.arm in {"gcs_sequence", "monitor_drop_gcs_auc"}:
        dataset.sequences = dataset.sequences[:, :, :4].copy()
        dataset.masks = dataset.masks[:, :, :4].copy()
    if args.arm in {"auc_stop", "auc_stop_cumulative", "monitor_drop_gcs_auc"}:
        _install_auc_early_stopping()
        if args.workers != 1:
            raise ValueError(
                "AUC early-stopping screens must use --workers 1 so the "
                "in-process monitor applies to every fold"
            )
    if args.arm in {"cumulative_loss", "auc_stop_cumulative"}:
        _install_cumulative_loss()

    prefix = f"strength_{args.tag}"
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
            "Exploratory correction screen on frozen development CV "
            "partitions. External cohorts were not used."
        ),
        "arm": args.arm,
        "models": list(models),
        "feature_count": len(feature_columns),
        "sequence_channels": (
            ["gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total"]
            if args.arm in {"gcs_sequence", "monitor_drop_gcs_auc"}
            else list(te.CHANNELS)
        ),
        "observation_masks": (
            "constant_one"
            if args.arm == "monitor_drop_nomask"
            else "source_observation"
        ),
        "folds": args.folds,
        "cv_repeats": args.repeats,
        "epochs": args.epochs,
        "model_seeds": list(range(args.seeds)),
        "workers": args.workers,
        "worker_threads": args.threads,
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
