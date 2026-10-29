"""Screen explicit pre-landmark trajectory-shape features."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


def _trajectory_features(
    values: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    index = np.flatnonzero(observed)
    if len(index) == 0:
        return {
            "gcs_traj_first": 0.0,
            "gcs_traj_last": 0.0,
            "gcs_traj_delta": 0.0,
            "gcs_traj_auc": 0.0,
            "gcs_traj_time_min": 0.0,
            "gcs_traj_time_max": 0.0,
            "gcs_traj_max_drop": 0.0,
            "gcs_traj_max_rise": 0.0,
            "gcs_traj_volatility": 0.0,
            "gcs_traj_hours_below8": 0.0,
            "gcs_traj_hours_below13": 0.0,
            "gcs_traj_first_hour": 0.0,
            "gcs_traj_last_hour": 0.0,
        }
    v = values[index].astype(float)
    h = index.astype(float)
    diffs = np.diff(v)
    running_max = np.maximum.accumulate(v)
    running_min = np.minimum.accumulate(v)
    return {
        "gcs_traj_first": float(v[0]),
        "gcs_traj_last": float(v[-1]),
        "gcs_traj_delta": float(v[-1] - v[0]),
        "gcs_traj_auc": float(np.trapz(v, h) / 24.0),
        "gcs_traj_time_min": float(h[int(np.argmin(v))]),
        "gcs_traj_time_max": float(h[int(np.argmax(v))]),
        "gcs_traj_max_drop": float(np.max(running_max - v)),
        "gcs_traj_max_rise": float(np.max(v - running_min)),
        "gcs_traj_volatility": (
            float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
        ),
        "gcs_traj_hours_below8": float(np.sum(v < 8)),
        "gcs_traj_hours_below13": float(np.sum(v < 13)),
        "gcs_traj_first_hour": float(h[0]),
        "gcs_traj_last_hour": float(h[-1]),
    }


def _augment(dataset: te.Dataset, feature_group: str) -> list[str]:
    sequences = dataset.sequences
    masks = dataset.masks
    gcs_total = sequences[:, :, 3]
    gcs_observed = masks[:, :, 3] > 0
    rows = [
        _trajectory_features(gcs_total[i], gcs_observed[i])
        for i in range(len(dataset.labels))
    ]
    frame = pd.DataFrame(rows, index=dataset.static_all.index)
    any_observed = masks.sum(axis=2) > 0
    frame["observed_channel_count_mean"] = masks.sum(axis=2).mean(axis=1)
    frame["observed_hours_any"] = any_observed.sum(axis=1).astype(float)
    frame["observed_hours_last"] = any_observed[:, -1].astype(float)
    gcs_last_hour = np.where(
        gcs_observed.any(axis=1),
        np.argmax(gcs_observed[:, ::-1], axis=1),
        24,
    )
    frame["gcs_last_gap"] = gcs_last_hour.astype(float)
    if feature_group == "gcs_shape":
        frame = frame[
            [column for column in frame.columns if column.startswith("gcs_traj_")]
        ]
    elif feature_group == "monitoring":
        frame = frame[
            [
                "observed_channel_count_mean",
                "observed_hours_any",
                "observed_hours_last",
                "gcs_last_gap",
            ]
        ]
    elif feature_group == "both":
        pass
    else:
        raise ValueError(f"Unknown feature group: {feature_group}")
    new_columns = list(frame.columns)
    for column in new_columns:
        dataset.static_all[column] = frame[column].to_numpy(dtype=np.float32)
        dataset.static_deep[column] = frame[column].to_numpy(dtype=np.float32)
    return new_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="traj_features")
    parser.add_argument(
        "--feature-group",
        default="both",
        choices=("gcs_shape", "monitoring", "both"),
    )
    parser.add_argument("--models", default="xgb_summary,dggru_tr")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()

    te.ensure_dirs()
    config, _ = te.load_config_snapshot()
    seed = int(config["random_seed"])
    dataset = te.load_dataset("mimic4")
    new_columns = _augment(dataset, args.feature_group)
    models = tuple(
        part.strip() for part in args.models.split(",") if part.strip()
    )
    tag = args.tag
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
            "Exploratory frozen-partition screen adding pre-landmark GCS "
            "trajectory-shape features to the core static set. External "
            "cohorts were not read."
        ),
        "models": list(models),
        "feature_group": args.feature_group,
        "new_features": new_columns,
        "new_feature_count": len(new_columns),
        "core_feature_count_before": len(te.CORE_STATIC),
        "core_feature_count_after": int(dataset.static_deep.shape[1]),
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
