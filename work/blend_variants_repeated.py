"""Fixed equal-weight blends from saved core OOF predictions.

Every blend uses weights fixed a priori at 1/k; no weight is fitted on the
held-out predictions. The paired interval follows the repeated-CV
subject-cluster bootstrap used by the formal audit.
"""
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


def main() -> None:
    source = ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz"
    payload = np.load(source, allow_pickle=True)
    labels = payload["labels"]
    groups = payload["subject_key"].astype(str)
    weights = payload["repeat0_analysis_weight"]
    repeats = sorted(
        {
            int(key.split("_")[0].replace("repeat", ""))
            for key in payload.files
            if key.startswith("repeat")
        }
    )

    def collect(model: str) -> list[np.ndarray]:
        return [payload[f"repeat{index}_{model}"] for index in repeats]

    xgb = collect("xgb_summary")
    gru = collect("gru")
    dggru = collect("dggru_tr")
    candidates = {
        "xgb": xgb,
        "xgb+dggru_tr": [0.5 * a + 0.5 * b for a, b in zip(xgb, dggru)],
        "xgb+gru": [0.5 * a + 0.5 * b for a, b in zip(xgb, gru)],
        "xgb+gru+dggru_tr": [
            (a + b + c) / 3.0 for a, b, c in zip(xgb, gru, dggru)
        ],
        "xgb+mean(gru,dggru_tr)": [
            0.5 * a + 0.25 * b + 0.25 * c
            for a, b, c in zip(xgb, gru, dggru)
        ],
    }
    output: dict[str, object] = {
        "purpose": (
            "Fixed equal-weight blends of saved out-of-fold predictions. "
            "Weights are not fitted; the interval is the repeated-CV "
            "subject-cluster paired bootstrap against XGBoost."
        ),
        "repeats": len(repeats),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20261024,
        "blends": {},
    }
    for name, probabilities in candidates.items():
        per_repeat = [
            float(te.multiclass_auc(labels, prob, sample_weight=weights))
            for prob in probabilities
        ]
        pooled = np.mean(probabilities, axis=0)
        difference = [
            te.multiclass_auc(labels, a, sample_weight=weights)
            - te.multiclass_auc(labels, b, sample_weight=weights)
            for a, b in zip(probabilities, xgb)
        ]
        interval = te.paired_auc_difference_repeated(
            labels,
            probabilities,
            xgb,
            groups,
            replicates=2000,
            seed=20261024,
            sample_weight=weights,
        )
        output["blends"][name] = {
            "per_repeat_macro_auc": per_repeat,
            "mean_macro_auc": float(np.mean(per_repeat)),
            "mean_macro_auc_sd": float(np.std(per_repeat, ddof=1)),
            "pooled_probability_macro_auc": float(
                te.multiclass_auc(labels, pooled, sample_weight=weights)
            ),
            "per_repeat_difference_vs_xgb": [float(value) for value in difference],
            "mean_difference_vs_xgb": float(np.mean(difference)),
            "paired_difference_ci": [float(interval[0]), float(interval[1])],
        }
        print(
            f"{name:24s} mean={np.mean(per_repeat):.4f} "
            f"pooled={te.multiclass_auc(labels, pooled, sample_weight=weights):.4f} "
            f"delta={np.mean(difference):+.4f} "
            f"CI=[{interval[0]:+.4f},{interval[1]:+.4f}]"
        )
    destination = ROOT / "results" / "experiments" / "blend_variants_repeated.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
