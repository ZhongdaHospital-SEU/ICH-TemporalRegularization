"""Screen a strict post-landmark incident deterioration endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("ICH_TORCH_THREADS", "4")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
for path in (ROOT, WORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_output_coupled_tr import OutputCoupledTemporalOrdinalNet
from src.ich_dynamic import models as dm
from src.ich_dynamic import sensitivity_analysis as sa
from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import PROCESSED, add_outcome


def incident_frame() -> pd.DataFrame:
    prefilter = pd.read_csv(PROCESSED / "mimic4" / "cohort_pre_filter.csv")
    gcs = pd.read_csv(PROCESSED / "mimic4" / "gcs_events.csv")
    frame = add_outcome(
        prefilter,
        gcs,
        baseline_mode="last",
        confirmation_consecutive_hours=2,
        include_death=True,
    )
    return frame[sa.complete_followup(frame)].copy()


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
    args = parser.parse_args()

    te.ensure_dirs()
    config = te.load_config()
    seed = int(config["random_seed"])
    manifest = json.loads(
        (te.RESULTS / "run_manifest.json").read_text(encoding="utf-8")
    )
    feature_columns = manifest["features"]
    selected_models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )

    frame = incident_frame()
    development = sa.build_dataset(
        "landmark_incident_strict",
        frame,
        feature_columns,
    )
    dm.TemporalOrdinalNet = OutputCoupledTemporalOrdinalNet

    prefix = "experiment_landmark_incident"
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
        output_dir / "landmark_incident_predictions.npz",
        labels=development.labels,
        stay_id=development.cohort["stay_id"].to_numpy(),
        subject_key=development.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    payload = {
        "purpose": (
            "Landmark-incident endpoint sensitivity: last pre-landmark GCS "
            "and confirmation at least 2 hours apart within 3 hours."
        ),
        "development_n": len(development.labels),
        "outcomes": {
            str(grade): int((development.labels == grade).sum())
            for grade in (0, 1, 2)
        },
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
    output_path = output_dir / "landmark_incident.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
