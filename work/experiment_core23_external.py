"""External check for the internally selected 23-variable core static set."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ICH_TORCH_THREADS", "4")
os.environ.setdefault("ICH_XGB_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


def main() -> None:
    core = list(te.DEEP_STATIC)
    development = te.load_dataset("mimic4", feature_columns=core)
    external = {
        name: te.load_dataset(name, feature_columns=core)
        for name in ("mimic3", "eicu")
    }
    dggru_run = json.loads(
        (te.RESULTS / "experiments" / "correction_core23.json").read_text(
            encoding="utf-8"
        )
    )
    dggru_tr_run = json.loads(
        (te.RESULTS / "experiments" / "correction_core23_tr.json").read_text(
            encoding="utf-8"
        )
    )
    cv_epochs = {
        "dggru": dggru_run["cv_best_epochs"]["dggru"],
        "dggru_tr": dggru_tr_run["cv_best_epochs"]["dggru_tr"],
    }
    models = ("xgb_summary", "dggru", "dggru_tr")
    predictions, metrics = te.train_full_and_external(
        development,
        external,
        models=models,
        epochs=25,
        seed=int(te.load_config_snapshot()[0]["random_seed"]),
        lambda_monotone=0.15,
        lambda_smooth=0.06,
        cv_best_epochs=cv_epochs,
        cv_selected_lambdas={"dggru_tr": [0.15]},
        lambda_smooth_ratio=0.4,
        model_seeds=(0, 1, 2),
        metric_workers=1,
        replicates=500,
        stacking_members=(),
    )
    for name, cohort_predictions in predictions.items():
        te.save_predictions(
            external[name],
            cohort_predictions,
            f"correction_core23_{name}_external",
        )
    payload = {
        "purpose": (
            "Exploratory external check after internal-CV selection of the "
            "23-variable core static set. The default 64-unit sequence "
            "configuration was used; no external outcome selected features, "
            "weights, or epochs."
        ),
        "feature_count": len(core),
        "models": list(models),
        "model_seeds": [0, 1, 2],
        "cv_best_epochs": cv_epochs,
        "bootstrap_replicates": 500,
        "metrics": metrics,
    }
    output_path = te.RESULTS / "experiments" / "core23_external_screen.json"
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
