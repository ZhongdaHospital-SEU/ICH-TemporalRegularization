"""Check whether the tabular+sequence blend gain depends on the architecture.

Each blend is the unfitted equal-weight average of XGBoost and one sequence
member. If every sequence member gives a similar gain, the gain reflects
complementarity between the two model families rather than a selected member.
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

REPEATS = range(5)


def main() -> None:
    core = np.load(
        ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz",
        allow_pickle=True,
    )
    baselines = np.load(
        ROOT / "results" / "predictions" / "mimic4_cv_baselines_repeats.npz",
        allow_pickle=True,
    )
    labels = core["labels"]
    weights = core["repeat0_analysis_weight"]
    groups = core["subject_key"].astype(str)

    def series(source, model: str) -> list[np.ndarray]:
        return [source[f"repeat{index}_{model}"] for index in REPEATS]

    xgb = series(core, "xgb_summary")
    members = {
        "gru": series(core, "gru"),
        "dggru": series(core, "dggru"),
        "dggru_tr": series(core, "dggru_tr"),
        "grud": series(baselines, "grud"),
        "transformer": series(baselines, "transformer"),
    }

    def repeated_mean(probabilities: list[np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    te.multiclass_auc(labels, prob, sample_weight=weights)
                    for prob in probabilities
                ]
            )
        )

    def pooled(probabilities: list[np.ndarray]) -> float:
        return float(
            te.multiclass_auc(
                labels, np.mean(probabilities, axis=0), sample_weight=weights
            )
        )

    output: dict[str, object] = {
        "purpose": (
            "Architecture-generality check for the unfitted equal-weight "
            "tabular+sequence average. Repeats and partitions are the formal "
            "ones; no weight is fitted on held-out predictions."
        ),
        "xgb_mean_macro_auc": repeated_mean(xgb),
        "xgb_pooled_macro_auc": pooled(xgb),
        "blends": {},
    }
    for name, member in members.items():
        blend = [0.5 * a + 0.5 * b for a, b in zip(xgb, member)]
        interval = te.paired_auc_difference_repeated(
            labels,
            blend,
            xgb,
            groups,
            replicates=2000,
            seed=20261024,
            sample_weight=weights,
        )
        output["blends"][name] = {
            "member_mean_macro_auc": repeated_mean(member),
            "blend_mean_macro_auc": repeated_mean(blend),
            "blend_pooled_macro_auc": pooled(blend),
            "mean_difference_vs_xgb": repeated_mean(blend) - repeated_mean(xgb),
            "paired_difference_ci": [float(interval[0]), float(interval[1])],
        }
        print(
            f"xgb+{name:12s} member={repeated_mean(member):.4f} "
            f"blend={repeated_mean(blend):.4f} pooled={pooled(blend):.4f} "
            f"CI=[{interval[0]:+.4f},{interval[1]:+.4f}]"
        )
    destination = ROOT / "results" / "experiments" / "blend_architecture_generality.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"xgb alone: mean={repeated_mean(xgb):.4f} pooled={pooled(xgb):.4f}")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
