"""Build the supplementary site-calibration figure from saved site tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "results" / "tables" / "eicu_site_calibration_sites.csv"
PAIRWISE = ROOT / "results" / "tables" / "eicu_site_calibration_pairwise.csv"
OUTPUT = ROOT / "results" / "figures" / "figure_site_calibration_updates.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    sites = read_rows(SITES)
    pairwise = read_rows(PAIRWISE)
    panel_specs = [
        ("any_deterioration", "xgb_summary", "Any deterioration: Summary XGBoost"),
        ("any_deterioration", "dggru_tr", "Any deterioration: DG-GRU+TR"),
        ("severe_deterioration", "xgb_summary", "Severe deterioration: Summary XGBoost"),
        ("severe_deterioration", "dggru_tr", "Severe deterioration: DG-GRU+TR"),
    ]
    summary = {
        (row["endpoint"], row["model"], row["method"]): row
        for row in pairwise
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), constrained_layout=True)
    for panel, (axis, (endpoint, model, title)) in enumerate(
        zip(axes.flat, panel_specs)
    ):
        axis.text(
            -0.16,
            1.05,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        by_site: dict[int, dict[str, dict[str, str]]] = {}
        for row in sites:
            if row["endpoint"] != endpoint or row["model"] != model:
                continue
            hospital = int(row["hospitalid"])
            by_site.setdefault(hospital, {})[row["method"]] = row
        hospitals = sorted(
            hospital
            for hospital, methods in by_site.items()
            if "uncalibrated" in methods and "slope_intercept" in methods
        )
        y = np.arange(len(hospitals))
        before = np.array(
            [float(by_site[h]["uncalibrated"]["ece"]) for h in hospitals]
        )
        after = np.array(
            [float(by_site[h]["slope_intercept"]["ece"]) for h in hospitals]
        )
        for index, (left, right) in enumerate(zip(before, after)):
            axis.plot(
                [left, right],
                [index, index],
                color="#bdbdbd",
                linewidth=1.0,
                zorder=1,
            )
        axis.scatter(
            before,
            y,
            s=18,
            color="#7f7f7f",
            label="No calibration",
            zorder=2,
        )
        axis.scatter(
            after,
            y,
            s=22,
            color="#1f6fb2",
            label="Site slope and intercept",
            zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels([str(h) for h in hospitals], fontsize=7)
        axis.set_xlabel("Expected calibration error", fontsize=8)
        axis.set_ylabel("eICU hospital ID", fontsize=8)
        row = summary.get((endpoint, model, "slope_intercept"))
        if row is not None:
            title = (
                f"{title}\n{row['sites_with_lower_ece']}/{row['sites']} sites "
                f"lower; median change "
                f"{float(row['median_site_ece_change']):+.3f}"
            )
        axis.set_title(title, fontsize=8.5, loc="left")
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="x", color="#e6e6e6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.set_xlim(0, max(0.25, float(np.max(np.r_[before, after])) * 1.08))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
