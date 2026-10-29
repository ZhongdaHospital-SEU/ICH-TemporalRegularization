"""Print the formal numbers needed for the manuscript."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def show_block(title: str, block: dict, keys: tuple[str, ...]) -> None:
    print(f"\n== {title} ==")
    for model, metrics in block.items():
        if not isinstance(metrics, dict) or "point" not in metrics:
            continue
        point = metrics["point"]
        ci = metrics.get("ci", {})
        text = []
        for key in keys:
            if key not in point:
                continue
            value = point[key]
            interval = ci.get(key)
            if interval and all(isinstance(item, (int, float)) for item in interval):
                text.append(f"{key}={value:.4f} [{interval[0]:.4f},{interval[1]:.4f}]")
            else:
                text.append(f"{key}={value:.4f}")
        print(f"  {model}: " + "; ".join(text))


def main() -> None:
    core = load("metrics_core.json")
    keys = (
        "macro_auc",
        "auc_grade_0",
        "auc_grade_1",
        "auc_grade_2",
        "accuracy",
        "quadratic_weighted_kappa",
        "multiclass_brier",
        "expected_calibration_error",
        "calibration_slope_any",
        "calibration_slope_severe",
    )
    show_block("core development", core["development"], keys)
    for cohort in ("mimic3", "eicu"):
        show_block(f"core external {cohort}", core["external"][cohort], keys)
    show_block("baselines", load("metrics_baselines.json")["development"], keys)
    show_block("ablation", load("metrics_ablation.json")["development"], keys)
    stack = load("metrics_stack1.json")
    show_block("stack", stack["development"], keys)
    variability = core["development"].get("cv_repeat_variability")
    if variability:
        print("\n== repeat variability ==")
        for model, value in variability.items():
            print(f"  {model}: {value}")


if __name__ == "__main__":
    main()
