"""Screen whether the neural model benefits from the full static feature set."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=2)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument(
        "--models",
        default="dggru,dggru_tr",
        help="Comma-separated temporal model names.",
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
    feature_columns = list(development_raw.static_all.columns)
    te.DEEP_STATIC = feature_columns
    development = te.load_dataset("mimic4", feature_columns=feature_columns)

    prefix = "experiment_full_static"
    predictions, _, cv_best_epochs, _, _ = te.run_cv(
        development,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=selected_models,
        lambda_monotone=float(config["lambda_monotone"]),
        lambda_smooth=float(config["lambda_smooth"]),
        workers=args.workers,
        worker_threads=args.worker_threads,
        model_seeds=tuple(range(args.seed_count)),
        repeats=args.cv_repeats,
        repeat_prefix=prefix,
    )
    repeat_path = (
        te.RESULTS / "predictions" / f"{prefix}_repeats.npz"
    )
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
        output_dir / "full_static_screen_predictions.npz",
        labels=development.labels,
        stay_id=development.cohort["stay_id"].to_numpy(),
        subject_key=development.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Sensitivity experiment: replace the 16-variable neural static "
            "subset with all eligible pre-landmark static features."
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
    output_path = output_dir / "full_static_screen.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
