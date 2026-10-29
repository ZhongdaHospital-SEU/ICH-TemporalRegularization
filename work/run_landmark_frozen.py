"""Run the core models at a non-primary landmark with frozen tuning."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-hours", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=3)
    parser.add_argument("--models", default="xgb_summary,gru,dggru,dggru_tr")
    parser.add_argument(
        "--freeze-from",
        type=Path,
        default=ROOT / "results" / "run_manifest_core.json",
    )
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    landmark_root = (
        ROOT
        / "data"
        / "processed"
        / "landmark_inputs"
    )
    if args.landmark_hours not in {6, 24, 48}:
        raise ValueError("Only the audited 6, 24, and 48 h landmarks are supported")
    for dataset in ("mimic4", "mimic3", "eicu"):
        path = landmark_root / dataset / f"landmark_{args.landmark_hours}h"
        if not path.is_dir():
            raise FileNotFoundError(path)

    original_load_dataset = te.load_dataset

    def load_landmark_dataset(
        name: str,
        feature_columns: list[str] | None = None,
    ):
        path = landmark_root / name / f"landmark_{args.landmark_hours}h"
        dataset = original_load_dataset(
            str(path),
            feature_columns=feature_columns,
        )
        if args.landmark_hours != 24:
            source = f"monitoring_intensity_{args.landmark_hours}h"
            if source in dataset.cohort.columns:
                dataset.static_all["monitoring_intensity_24h"] = (
                    dataset.cohort[source].to_numpy()
                )
                if source in dataset.static_all.columns:
                    dataset.static_all = dataset.static_all.drop(
                        columns=[source]
                    )
                dataset.static_deep = dataset.static_all.reindex(
                    columns=[
                        column
                        for column in te.DEEP_STATIC
                        if column in dataset.static_all.columns
                    ]
                )
        return dataset

    te.load_dataset = load_landmark_dataset
    tag = args.tag or f"landmark{args.landmark_hours}"
    argv = [
        str(ROOT / "src" / "ich_dynamic" / "train_evaluate.py"),
        "--models",
        args.models,
        "--epochs",
        str(args.epochs),
        "--folds",
        str(args.folds),
        "--cv-repeats",
        str(args.cv_repeats),
        "--replicates",
        str(args.replicates),
        "--freeze-from",
        str(args.freeze_from.resolve()),
        "--tag",
        tag,
        "--workers",
        str(args.workers),
        "--worker-threads",
        str(args.worker_threads),
        "--metric-workers",
        str(args.metric_workers),
    ]
    if not args.include_external:
        argv.append("--skip-external")
    os.environ.setdefault("ICH_TORCH_THREADS", str(args.worker_threads))
    os.environ.setdefault("ICH_XGB_THREADS", str(args.worker_threads))
    sys.argv = argv
    te.main()


if __name__ == "__main__":
    main()
