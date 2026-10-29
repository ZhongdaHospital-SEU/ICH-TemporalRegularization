"""Recompute external confidence intervals with the analysis grouping rules."""
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
CORE = ROOT / "results" / "metrics_core.json"
DESTINATION = ROOT / "results" / "experiments" / "table6_consistency.json"

EXTERNAL_SEED = 20261024 + 5000
REPLICATES = 2000


def cohort_groups(cohort: str, payload) -> np.ndarray:
    if cohort == "eicu":
        frame = pd.read_csv(
            LANDMARKS / cohort / "landmark_24h" / "cohort.csv",
            usecols=["stay_id", "hospital_key"],
        ).set_index("stay_id")
        groups = frame.loc[payload["stay_id"], "hospital_key"].astype(str)
        if groups.nunique(dropna=True) <= 1:
            return payload["subject_key"].astype(str)
        return groups.to_numpy()
    return payload["subject_key"].astype(str)


def main() -> None:
    reference = json.loads(CORE.read_text(encoding="utf-8"))["external"]
    output = {
        "purpose": (
            "Recompute Table 6 intervals with the grouping and seed used by "
            "the external evaluation pipeline."
        ),
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": EXTERNAL_SEED,
        "paired_seed": EXTERNAL_SEED + 1,
        "cohorts": {},
    }
    for cohort in ("mimic3", "eicu"):
        payload = np.load(
            PREDICTIONS / f"{cohort}_external_core.npz", allow_pickle=True
        )
        labels = payload["labels"]
        groups = cohort_groups(cohort, payload)
        probabilities = {
            "xgb_summary": payload["xgb_summary"],
            "dggru_tr": payload["dggru_tr"],
            "equal_weight": (
                0.5 * payload["xgb_summary"] + 0.5 * payload["dggru_tr"]
            ),
        }
        rows = {}
        for name, probability in probabilities.items():
            interval = te.group_bootstrap(
                labels,
                probability,
                groups,
                replicates=REPLICATES,
                seed=EXTERNAL_SEED,
            )["macro_auc"]
            rows[name] = {
                "macro_auc": float(te.multiclass_auc(labels, probability)),
                "macro_auc_ci": [float(interval[0]), float(interval[1])],
            }
            print(
                f"{cohort} {name}: "
                f"{rows[name]['macro_auc']:.6f} "
                f"({rows[name]['macro_auc_ci'][0]:.6f}, "
                f"{rows[name]['macro_auc_ci'][1]:.6f})",
                flush=True,
            )
        paired = te.paired_auc_difference(
            labels,
            probabilities["equal_weight"],
            probabilities["xgb_summary"],
            groups,
            replicates=REPLICATES,
            seed=EXTERNAL_SEED + 1,
        )
        rows["paired_difference_equal_weight_vs_xgb"] = [
            float(paired[0]),
            float(paired[1]),
        ]
        rows["clusters"] = int(len(np.unique(groups)))
        rows["cluster_type"] = (
            "hospital" if cohort == "eicu" else "patient"
        )
        for model in ("xgb_summary", "dggru_tr"):
            observed = rows[model]["macro_auc_ci"]
            expected = reference[cohort][model]["ci"]["macro_auc"]
            difference = float(np.max(np.abs(np.asarray(observed) - expected)))
            rows.setdefault("single_model_ci_max_abs_difference", {})[model] = (
                difference
            )
        output["cohorts"][cohort] = rows
        print(
            f"{cohort} paired equal-weight minus XGBoost: "
            f"{paired[0]:+.6f} to {paired[1]:+.6f}",
            flush=True,
        )
    DESTINATION.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
