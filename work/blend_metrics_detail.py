"""Discrimination and calibration of the unfitted XGBoost + DG-GRU+TR average."""
from __future__ import annotations

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

KEYS = (
    "macro_auc",
    "auc_grade_0",
    "auc_grade_1",
    "auc_grade_2",
    "multiclass_brier",
    "log_loss",
    "expected_calibration_error",
    "calibration_slope_any",
    "calibration_intercept_any",
)


def describe(name: str, probabilities: list[np.ndarray], labels, weights) -> dict:
    pooled = np.mean(probabilities, axis=0)
    pooled_metrics = te.point_metrics(labels, pooled, sample_weight=weights)
    per_repeat = [
        te.point_metrics(labels, prob, sample_weight=weights)
        for prob in probabilities
    ]
    row = {
        "pooled": {key: float(pooled_metrics[key]) for key in KEYS},
        "per_repeat_mean": {
            key: float(np.mean([metrics[key] for metrics in per_repeat]))
            for key in KEYS
        },
    }
    print(
        f"{name:18s} pooled macro={row['pooled']['macro_auc']:.4f} "
        f"g1={row['pooled']['auc_grade_1']:.4f} "
        f"brier={row['pooled']['multiclass_brier']:.4f} "
        f"ece={row['pooled']['expected_calibration_error']:.4f} "
        f"slope={row['pooled']['calibration_slope_any']:.3f}"
    )
    print(
        f"{'':18s} mean   macro={row['per_repeat_mean']['macro_auc']:.4f} "
        f"g1={row['per_repeat_mean']['auc_grade_1']:.4f} "
        f"brier={row['per_repeat_mean']['multiclass_brier']:.4f} "
        f"ece={row['per_repeat_mean']['expected_calibration_error']:.4f}"
    )
    return row


def main() -> None:
    payload = np.load(
        ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz",
        allow_pickle=True,
    )
    labels = payload["labels"]
    weights = payload["repeat0_analysis_weight"]
    xgb = [payload[f"repeat{index}_xgb_summary"] for index in range(5)]
    dggru = [payload[f"repeat{index}_dggru_tr"] for index in range(5)]
    blend = [0.5 * a + 0.5 * b for a, b in zip(xgb, dggru)]
    output = {
        "purpose": (
            "Discrimination and calibration of the unfitted 1:1 average of "
            "XGBoost and DG-GRU+TR, using the formal folds and repeats."
        ),
        "xgb_summary": describe("xgb_summary", xgb, labels, weights),
        "xgb_plus_dggru_tr": describe("xgb+dggru_tr", blend, labels, weights),
    }
    destination = (
        ROOT / "results" / "experiments" / "blend_metrics_detail.json"
    )
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
