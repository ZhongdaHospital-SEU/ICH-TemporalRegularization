"""Frozen-partition screens for the v2 temporal architectures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ICH_TORCH_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te
from work.v2_models import (
    LearnedGraphMaskedPoolingNet,
    MaskedPoolingTemporalNet,
)


ARMS = {
    "pool": MaskedPoolingTemporalNet,
    "gat_pool": LearnedGraphMaskedPoolingNet,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()

    # This must happen before run_cv constructs the workers.  The screen uses
    # workers=1 by default so the in-process patch is unambiguous on Windows.
    dm.TemporalOrdinalNet = ARMS[args.arm]
    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    dataset = te.load_dataset("mimic4")
    tag = f"v2_{args.arm}"
    predictions, _, best_epochs, selected_lambdas, _ = te.run_cv(
        dataset,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=("dggru_tr",),
        lambda_monotone=1.5,
        lambda_smooth=0.6,
        model_seeds=tuple(range(args.seeds)),
        workers=args.workers,
        worker_threads=args.threads,
        repeats=args.repeats,
        repeat_prefix=tag,
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
        output_dir / f"{tag}_predictions.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        subject_key=dataset.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Exploratory frozen-partition screen of the v2 temporal "
            "architecture. External cohorts were not read."
        ),
        "arm": args.arm,
        "model": "dggru_tr",
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
    output_path = output_dir / f"{tag}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"]["dggru_tr"]["point"], indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
