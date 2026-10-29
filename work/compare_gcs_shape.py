"""Paired comparison of the GCS-shape feature screen against its control."""
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

EXPERIMENTS = ROOT / "results" / "experiments"


def main() -> None:
    control = np.load(
        EXPERIMENTS / "strength_xgb_core3_predictions.npz", allow_pickle=True
    )
    core = np.load(
        ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz",
        allow_pickle=True,
    )
    labels = control["labels"]
    groups = control["subject_key"].astype(str)
    weights = core["repeat0_analysis_weight"]
    control_probability = control["xgb_summary"]
    output: dict[str, object] = {
        "purpose": (
            "Paired subject-cluster bootstrap of the GCS-shape feature screen "
            "against the frozen-partition control. Single repeat, three seeds, "
            "500 replicates; exploratory."
        ),
        "comparisons": {},
    }
    for tag in ("traj_gcs_shape_xgb", "traj_monitoring_xgb", "traj_features"):
        candidate = np.load(
            EXPERIMENTS / f"{tag}_predictions.npz", allow_pickle=True
        )
        if not (candidate["subject_key"] == control["subject_key"]).all():
            raise SystemExit(f"{tag}: subject order differs from the control")
        probability = candidate["xgb_summary"]
        interval = te.paired_auc_difference(
            labels,
            probability,
            control_probability,
            groups,
            replicates=2000,
            seed=20261024,
            sample_weight=weights,
        )
        auc = float(
            te.multiclass_auc(labels, probability, sample_weight=weights)
        )
        control_auc = float(
            te.multiclass_auc(labels, control_probability, sample_weight=weights)
        )
        output["comparisons"][tag] = {
            "macro_auc": auc,
            "control_macro_auc": control_auc,
            "difference": auc - control_auc,
            "paired_difference_ci": [float(interval[0]), float(interval[1])],
        }
        print(
            f"{tag:22s} auc={auc:.4f} control={control_auc:.4f} "
            f"delta={auc - control_auc:+.4f} "
            f"CI=[{interval[0]:+.4f},{interval[1]:+.4f}]"
        )
    destination = EXPERIMENTS / "gcs_shape_comparison.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
