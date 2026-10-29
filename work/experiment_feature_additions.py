"""Screen fixed additions to the 23-variable core static set."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


ADDITION_SETS = {
    "treatment2": (
        "mechanical_ventilation_24h",
        "sedation_exposure_24h",
    ),
    "rass3": (
        "min_rass_24h",
        "any_rass_le_minus2_24h",
        "rass_n",
    ),
    "treatment_rass5": (
        "mechanical_ventilation_24h",
        "sedation_exposure_24h",
        "min_rass_24h",
        "any_rass_le_minus2_24h",
        "rass_n",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--addition-set",
        required=True,
        choices=tuple(ADDITION_SETS),
    )
    parser.add_argument("--models", default="xgb_summary,dggru_tr")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--lambda-monotone", type=float, default=1.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.6)
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    configured = list(te.CORE_STATIC)
    additions = list(ADDITION_SETS[args.addition_set])
    feature_columns = configured + [
        column for column in additions if column not in configured
    ]
    available = set(te.load_dataset("mimic4").static_all.columns)
    missing = [column for column in feature_columns if column not in available]
    if missing:
        raise ValueError("Unknown added features: " + ", ".join(missing))

    te.DEEP_STATIC = feature_columns
    dataset = te.load_dataset("mimic4", feature_columns=feature_columns)
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
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
            "Exploratory fixed feature-addition screen on frozen development "
            "CV partitions. The primary 23-variable set is unchanged."
        ),
        "addition_set": args.addition_set,
        "features": feature_columns,
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
