"""Transport the output-coupled TR experiment without retuning on external data."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ICH_TORCH_THREADS", "3")
os.environ.setdefault("ICH_XGB_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
for path in (ROOT, WORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_output_coupled_tr import OutputCoupledTemporalOrdinalNet
from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument(
        "--cohorts",
        default="mimic3,eicu",
        help="Comma-separated external cohorts.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results" / "experiments" / "output_coupled_current.json",
    )
    args = parser.parse_args()

    te.ensure_dirs()
    config = te.load_config()
    seed = int(config["random_seed"]) + 5000
    source = json.loads(args.source.read_text(encoding="utf-8"))
    feature_columns = list(source["feature_columns"])
    selected_models = tuple(source["models"])
    if not set(selected_models).issubset({"dggru", "dggru_tr"}):
        raise ValueError("External screen is restricted to dggru,dggru_tr")

    development = te.load_dataset("mimic4", feature_columns=feature_columns)
    external_names = tuple(
        part.strip() for part in args.cohorts.split(",") if part.strip()
    )
    if not external_names or not set(external_names).issubset({"mimic3", "eicu"}):
        raise ValueError("External cohorts must be mimic3 and/or eicu")
    external = {
        name: te.load_dataset(name, feature_columns=feature_columns)
        for name in external_names
    }
    dm.TemporalOrdinalNet = OutputCoupledTemporalOrdinalNet

    predictions, metrics = te.train_full_and_external(
        development,
        external,
        models=selected_models,
        epochs=int(source["epochs"]),
        seed=seed,
        lambda_monotone=float(config["lambda_monotone"]),
        lambda_smooth=float(config["lambda_smooth"]),
        cv_best_epochs=source["cv_best_epochs"],
        model_seeds=tuple(range(int(source["seed_count"]))),
        metric_workers=args.metric_workers,
        replicates=args.replicates,
    )

    output_dir = te.RESULTS / "experiments"
    for cohort, cohort_predictions in predictions.items():
        te.save_predictions(
            external[cohort],
            cohort_predictions,
            f"experiment_output_coupled_{cohort}_external",
        )
    payload = {
        "purpose": (
            "External transport of the output-coupled temporal-regularization "
            "screen using fixed development-fold stopping epochs. No external "
            "outcome selected a model, weight, or endpoint."
        ),
        "source_experiment": str(args.source),
        "development_n": len(development.labels),
        "external_n": {
            name: int(len(dataset.labels)) for name, dataset in external.items()
        },
        "models": list(selected_models),
        "model_seeds": list(range(int(source["seed_count"]))),
        "lambda_monotone": float(config["lambda_monotone"]),
        "lambda_smooth": float(config["lambda_smooth"]),
        "bootstrap_replicates": args.replicates,
        "metrics": metrics,
    }
    suffix = "_".join(external_names)
    output_path = output_dir / f"output_coupled_external_{suffix}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
