"""Screen static-input and class-balance corrections on the frozen CV splits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ICH_TORCH_THREADS", "2")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


def class_balance_weights(
    labels: np.ndarray,
    train_index: np.ndarray,
    method: str,
) -> np.ndarray:
    """Return deterministic class weights for one outer training partition."""
    weights = np.ones(len(labels), dtype=np.float64)
    if method == "none":
        return weights
    counts = np.bincount(labels[train_index], minlength=te.N_CLASSES).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every outcome grade must occur in the training partition")
    if method == "sqrt":
        class_weight = 1.0 / np.sqrt(counts)
    elif method == "inverse":
        class_weight = 1.0 / counts
    else:
        raise ValueError(f"Unknown class-balance method: {method}")
    class_weight /= np.average(class_weight, weights=counts)
    weights[train_index] = class_weight[labels[train_index]]
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        required=True,
        help="Unique name for this screening run.",
    )
    parser.add_argument(
        "--static",
        choices=("all", "core"),
        default="all",
        help="Use all 210 eligible static features or the 23-variable core set.",
    )
    parser.add_argument(
        "--class-balance",
        choices=("none", "sqrt", "inverse"),
        default="none",
    )
    parser.add_argument("--models", default="dggru")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--lambda-monotone", type=float, default=0.15)
    parser.add_argument("--lambda-smooth", type=float, default=0.06)
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    unsupported = [
        name for name in models if te.MODEL_META[name]["kind"] != "temporal"
    ]
    if unsupported:
        raise ValueError(
            "Screen is restricted to temporal models: " + ", ".join(unsupported)
        )

    if args.static == "core":
        feature_columns = list(te.DEEP_STATIC)
        dataset = te.load_dataset("mimic4", feature_columns=feature_columns)
    else:
        dataset = te.load_dataset("mimic4")
        feature_columns = list(dataset.static_all.columns)

    fold_weight_builder = None
    if args.class_balance != "none":
        fold_weight_builder = lambda train_index: class_balance_weights(
            dataset.labels,
            train_index,
            args.class_balance,
        )

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
        fold_weight_builder=fold_weight_builder,
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
            "Exploratory correction screen on the frozen development CV "
            "partitions. External cohorts were not used."
        ),
        "static_subset": args.static,
        "feature_count": len(feature_columns),
        "class_balance": args.class_balance,
        "models": list(models),
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
