"""External validation of the unfitted XGBoost + DG-GRU+TR average."""
from __future__ import annotations

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

PREDICTIONS = ROOT / "results" / "predictions"
LANDMARKS = ROOT / "data" / "processed" / "landmark_inputs"
BOOTSTRAP_SEED = 20261024 + 5000


def analyse(cohort: str, reference: dict) -> dict:
    payload = np.load(
        PREDICTIONS / f"{cohort}_external_core.npz", allow_pickle=True
    )
    labels = payload["labels"]
    if cohort == "eicu":
        frame = pd.read_csv(
            LANDMARKS / cohort / "landmark_24h" / "cohort.csv",
            usecols=["stay_id", "hospital_key"],
        ).set_index("stay_id")
        groups = frame.loc[payload["stay_id"], "hospital_key"].astype(str).to_numpy()
        if len(np.unique(groups)) <= 1:
            groups = payload["subject_key"].astype(str)
    else:
        groups = payload["subject_key"].astype(str)
    xgb = payload["xgb_summary"]
    dggru = payload["dggru_tr"]
    blend = 0.5 * xgb + 0.5 * dggru
    reference_auc = reference[cohort]["xgb_summary"]["point"]["macro_auc"]
    computed_auc = float(te.multiclass_auc(labels, xgb))
    print(
        f"[{cohort}] xgb reference={reference_auc:.4f} recomputed={computed_auc:.4f} "
        f"n={len(labels)} grades={np.bincount(labels).tolist()}"
    )
    rows = {}
    for name, probability in (
        ("xgb_summary", xgb),
        ("dggru_tr", dggru),
        ("xgb_plus_dggru_tr", blend),
    ):
        point = te.point_metrics(labels, probability)
        interval = te.group_bootstrap(
            labels,
            probability,
            groups,
            replicates=2000,
            seed=BOOTSTRAP_SEED,
        )
        rows[name] = {
            key: float(point[key])
            for key in (
                "macro_auc",
                "auc_grade_0",
                "auc_grade_1",
                "auc_grade_2",
                "multiclass_brier",
                "expected_calibration_error",
                "calibration_slope_any",
                "calibration_slope_severe",
            )
        }
        rows[name]["macro_auc_ci"] = [
            float(interval["macro_auc"][0]),
            float(interval["macro_auc"][1]),
        ]
    interval = te.paired_auc_difference(
        labels,
        blend,
        xgb,
        groups,
        replicates=2000,
        seed=BOOTSTRAP_SEED + 1,
    )
    rows["xgb_plus_dggru_tr"]["paired_difference_ci_vs_xgb"] = [
        float(interval[0]),
        float(interval[1]),
    ]
    print(
        f"[{cohort}] blend macro={rows['xgb_plus_dggru_tr']['macro_auc']:.4f} "
        f"CI vs xgb=[{interval[0]:+.4f},{interval[1]:+.4f}]"
    )
    return rows


def main() -> None:
    core = json.loads(
        (ROOT / "results" / "metrics_core.json").read_text(encoding="utf-8")
    )
    reference = core["external"]
    output = {
        "purpose": (
            "Direct transfer of the unfitted 1:1 average of XGBoost and "
            "DG-GRU+TR to the two external cohorts. No refitting, no weight "
            "estimation, no recalibration."
        ),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_cluster": {
            "mimic3": "patient",
            "eicu": "hospital",
        },
        "cohorts": {},
    }
    for cohort in ("mimic3", "eicu"):
        output["cohorts"][cohort] = analyse(cohort, reference)
    destination = ROOT / "results" / "experiments" / "external_ensemble.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
