"""Build the supplementary 6/24/48 h landmark figure from saved metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "tables" / "multilandmark_summary.csv"
OUTPUT = ROOT / "results" / "figures" / "figure_landmark_dynamics.png"
MODELS = {
    "xgb_summary": ("Summary XGBoost", "#3b6ea8"),
    "gru": ("GRU", "#d08b2f"),
    "dggru": ("DG-GRU", "#6f8f3d"),
    "dggru_tr": ("DG-GRU+TR", "#b43b4b"),
}
PANELS = (
    ("macro_auc", "Macro-AUC"),
    ("grade_1_auc", "Grade 1 AUC"),
    ("calibration_slope_any", "Calibration slope"),
    ("ece_any", "ECE (any deterioration)"),
)


def main() -> None:
    with SUMMARY.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No landmark rows in {SUMMARY}")
    available = {int(float(row["landmark_h"])) for row in rows}
    if not {6, 24, 48}.issubset(available):
        raise SystemExit(
            f"Landmark figure needs 6, 24, and 48 h; available={sorted(available)}"
        )
    n_by_landmark = {}
    for row in rows:
        n_by_landmark[int(float(row["landmark_h"]))] = int(float(row["n"]))
    positions = {6: 0, 24: 1, 48: 2}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    for panel, (axis, (field, label)) in enumerate(
        zip(axes.flat, PANELS)
    ):
        axis.text(
            -0.14,
            1.05,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        for model, (model_label, color) in MODELS.items():
            xs = []
            ys = []
            lows = []
            highs = []
            for hours in (6, 24, 48):
                match = [
                    row
                    for row in rows
                    if row["model"] == model
                    and int(float(row["landmark_h"])) == hours
                ]
                if not match:
                    continue
                row = match[0]
                value = float(row[field])
                xs.append(positions[hours])
                ys.append(value)
                if field == "macro_auc":
                    lows.append(value - float(row["macro_auc_ci_low"]))
                    highs.append(float(row["macro_auc_ci_high"]) - value)
            if not xs:
                continue
            if field == "macro_auc":
                axis.errorbar(
                    xs,
                    ys,
                    yerr=np.vstack([lows, highs]),
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                    color=color,
                    label=model_label,
                )
            else:
                axis.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    color=color,
                    label=model_label,
                )
        axis.set_xticks([0, 1, 2])
        axis.set_xticklabels(
            [
                f"6 h\n(n={n_by_landmark[6]:,})",
                f"24 h\n(n={n_by_landmark[24]:,})",
                f"48 h\n(n={n_by_landmark[48]:,})",
            ],
            fontsize=8,
        )
        axis.set_xlabel("Prediction landmark", fontsize=8)
        axis.set_ylabel(label, fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        axis.set_axisbelow(True)
        if field == "macro_auc":
            axis.axhline(0.80, color="#9a9a9a", linestyle="--", linewidth=0.8)
        if field == "calibration_slope_any":
            axis.axhline(1.0, color="#9a9a9a", linestyle="--", linewidth=0.8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.02),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
