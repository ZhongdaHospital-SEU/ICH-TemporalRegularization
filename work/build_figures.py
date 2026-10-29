"""Build the manuscript figures from the saved formal predictions."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

os.environ.setdefault("ICH_TORCH_THREADS", "1")
os.environ.setdefault("ICH_XGB_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te

OUTPUT = ROOT / "results" / "figures"
PREDICTIONS = ROOT / "results" / "predictions"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLOURS = {
    "xgb_summary": "#2E5E8C",
    "dggru_tr": "#B4472B",
    "blend": "#1F6F4A",
    "gru": "#7A7A7A",
    "dggru": "#9C6B3F",
}


def load_repeats() -> dict:
    return np.load(
        PREDICTIONS / "mimic4_cv_core_repeats.npz", allow_pickle=True
    )


def pooled(payload, model: str) -> np.ndarray:
    repeats = [payload[f"repeat{index}_{model}"] for index in range(5)]
    return np.mean(repeats, axis=0)


def roc_curve(labels: np.ndarray, scores: np.ndarray, positive: int):
    truth = (labels == positive).astype(int)
    order = np.argsort(-scores, kind="mergesort")
    truth = truth[order]
    scores = scores[order]
    distinct = np.where(np.diff(scores))[0]
    threshold_index = np.r_[distinct, truth.size - 1]
    true_positive = np.cumsum(truth)[threshold_index]
    false_positive = 1 + threshold_index - true_positive
    tpr = np.r_[0, true_positive / max(true_positive[-1], 1)]
    fpr = np.r_[0, false_positive / max(false_positive[-1], 1)]
    return fpr, tpr


def bin_calibration(
    labels: np.ndarray, probability: np.ndarray, positive: int, bins: int = 10
):
    truth = (labels >= positive).astype(float)
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    observed, predicted = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high)
        if mask.sum() < 10:
            continue
        observed.append(truth[mask].mean())
        predicted.append(probability[mask].mean())
    return np.asarray(predicted), np.asarray(observed)


def figure_one() -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 4.8), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    left.set_axis_off()
    left.set_xlim(0, 10)
    left.set_ylim(0, 10)
    boxes = [
        (0.2, 8.70, 9.6, 0.85, "MIMIC-IV ICU stays with spontaneous ICH\nn = 4,622"),
        (0.2, 7.40, 9.6, 0.85, "First stay per patient and adult\nn = 3,446"),
        (
            0.2,
            5.95,
            9.6,
            1.10,
            "Excluded n = 773\ndeath at or before hour 24, not in ICU at hour 24,\n"
            "no valid pre-landmark GCS, <2 post-landmark GCS hours,\n"
            "unclassifiable discharge before 72 h",
        ),
        (
            0.2,
            4.45,
            9.6,
            1.00,
            "Development cohort, n = 2,673\n"
            "grades 0 / 1 / 2 = 2,031 / 243 / 399",
        ),
        (
            0.2,
            3.05,
            9.6,
            0.85,
            "Development evaluation\nfive-fold patient-grouped CV, five repeats, five seeds",
        ),
        (
            0.2,
            1.85,
            9.6,
            0.70,
            "Fitted model applied unchanged (no refitting or recalibration)",
        ),
        (
            0.2,
            0.36,
            4.6,
            1.16,
            "MIMIC-III, within-institution\ntransport\nn = 1,136\n756 / 123 / 257",
        ),
        (
            5.2,
            0.36,
            4.6,
            1.16,
            "eICU-CRD, external transport\nn = 1,725\n1,315 / 156 / 254",
        ),
    ]
    for x, y, width, height, text in boxes:
        left.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor="#F2F2F2",
                edgecolor="#4A4A4A",
                linewidth=0.7,
            )
        )
        left.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.6,
            linespacing=1.35,
        )
    for start, end in (
        ((5.0, 8.70), (5.0, 8.25)),
        ((5.0, 7.40), (5.0, 7.05)),
        ((5.0, 5.95), (5.0, 5.45)),
        ((5.0, 4.45), (5.0, 3.90)),
        ((5.0, 3.05), (5.0, 2.55)),
        ((2.8, 1.85), (2.5, 1.58)),
        ((7.2, 1.85), (7.5, 1.58)),
    ):
        left.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=8, color="#4A4A4A", linewidth=0.7
            )
        )
    left.text(0.2, 9.95, "A", fontsize=10, fontweight="bold", va="top")

    right.set_axis_off()
    right.set_xlim(0, 10)
    right.set_ylim(0, 10)
    right.text(0.2, 9.95, "B", fontsize=10, fontweight="bold", va="top")
    blocks = [
        (0.2, 9.00, 9.6, 0.72, "ICU hours 0-23 (24 hourly bins)\n14 channels + observation masks"),
        (0.2, 7.90, 9.6, 0.72, "Dynamic feature graph at each hour\n(value and mask aware)"),
        (0.2, 6.80, 9.6, 0.72, "GRU, 64 units\nhidden states h$_1$ ... h$_{24}$"),
        (0.2, 5.00, 4.2, 0.95, "Risk score r$_t$\nat each hidden state h$_t$"),
        (5.6, 5.00, 4.2, 0.95, "Static predictors\nat the landmark\n(23 variables)"),
        (0.2, 3.30, 3.55, 1.20, "Temporal penalty\n(training only)\nmonotone, smooth,\nbounded step"),
        (5.6, 3.55, 4.2, 0.90, "Concatenate h$_{24}$\nwith the encoded static\nrepresentation"),
        (
            2.0,
            1.95,
            6.0,
            0.95,
            "Two-branch ordinal head\nshared score r$_{24}$ + offset; two cut points\n"
            "P(grade 0 / 1 / 2) for hours 24-72",
        ),
        (2.0, 0.55, 6.0, 0.80, "Predicted ordinal grade for hours 24-72"),
    ]
    for x, y, width, height, text in blocks:
        right.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor="#F2F2F2" if "penalty" in text.lower() else "#FFFFFF",
                edgecolor="#4A4A4A",
                linewidth=0.7,
            )
        )
        right.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.2,
            linespacing=1.35,
        )
    arrows = (
        ((5.0, 9.00), (5.0, 8.62), "-"),
        ((5.0, 7.90), (5.0, 7.52), "-"),
        ((2.60, 6.80), (2.30, 5.95), "-"),
        ((5.00, 6.80), (5.00, 4.00), "-"),
        ((5.00, 4.00), (5.60, 4.00), "-"),
        ((7.70, 5.00), (7.70, 4.45), "-"),
        ((2.30, 5.00), (2.30, 4.50), "--"),
        ((7.70, 3.55), (7.10, 2.90), "-"),
        ((4.05, 5.00), (4.05, 2.90), "-"),
        ((5.0, 1.95), (5.0, 1.35), "-"),
    )
    for start, end, style in arrows:
        right.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                color="#4A4A4A",
                linewidth=0.7,
                linestyle=style,
            )
        )
    right.text(
        0.40,
        4.72,
        r"r$_t$ (all hours)",
        fontsize=6.0,
        ha="left",
        va="center",
    )
    right.text(4.86, 5.58, "h$_{24}$", fontsize=6.0, ha="right", va="center")
    right.text(
        4.16,
        3.30,
        "r$_{24}$ (base of the shared score)",
        fontsize=6.0,
        ha="left",
        va="center",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure1_design.png")
    plt.close(figure)


def figure_two() -> None:
    payload = load_repeats()
    labels = payload["labels"]
    weights = payload["repeat0_analysis_weight"]
    core = json.loads((ROOT / "results" / "metrics_core.json").read_text())
    baselines = json.loads(
        (ROOT / "results" / "metrics_baselines.json").read_text()
    )
    blend = 0.5 * pooled(payload, "xgb_summary") + 0.5 * pooled(payload, "dggru_tr")

    figure = plt.figure(figsize=(7.2, 5.6))
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    forest = figure.add_subplot(grid[0, 0])
    roc = figure.add_subplot(grid[0, 1])
    calibration = figure.add_subplot(grid[1, 0])
    decision = figure.add_subplot(grid[1, 1])

    rows = [
        ("GCS slope only", baselines["development"]["gcs_slope"], "#8A8A8A"),
        ("Static XGBoost", baselines["development"]["xgb_static"], "#8A8A8A"),
        ("GRU-D", baselines["development"]["grud"], "#8A8A8A"),
        ("Transformer", baselines["development"]["transformer"], "#8A8A8A"),
        ("DG-GRU", core["development"]["dggru"], "#9C6B3F"),
        ("DG-GRU+TR", core["development"]["dggru_tr"], "#B4472B"),
        ("GRU", core["development"]["gru"], "#7A7A7A"),
        ("Summary XGBoost", core["development"]["xgb_summary"], "#2E5E8C"),
    ]
    ensemble_auc = float(te.multiclass_auc(labels, blend, sample_weight=weights))
    for position, (name, block, colour) in enumerate(rows):
        point = block["point"]["macro_auc"]
        low, high = block["ci"]["macro_auc"]
        forest.plot([low, high], [position, position], color=colour, linewidth=1.4)
        forest.plot(point, position, "o", color=colour, markersize=4)
    forest.plot(
        ensemble_auc,
        len(rows),
        "o",
        color=COLOURS["blend"],
        markersize=4.5,
    )
    forest.plot(
        [ensemble_auc, ensemble_auc],
        [len(rows) - 0.25, len(rows) + 0.25],
        color=COLOURS["blend"],
        linewidth=1.4,
    )
    forest.axvline(0.80, color="#BBBBBB", linewidth=0.7, linestyle="--")
    forest.set_yticks(
        list(range(len(rows))) + [len(rows)],
        [name for name, _, _ in rows] + ["Equal-weight average"],
    )
    forest.set_xlim(0.5, 0.84)
    forest.set_xlabel("Macro-AUC (95% CI)")
    forest.set_title("A  Development discrimination")
    forest.text(
        ensemble_auc,
        len(rows) + 0.32,
        f"{ensemble_auc:.3f} (pooled)",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLOURS["blend"],
    )

    for model, colour, label in (
        ("xgb_summary", COLOURS["xgb_summary"], "Summary XGBoost"),
        ("dggru_tr", COLOURS["dggru_tr"], "DG-GRU+TR"),
    ):
        probability = pooled(payload, model)
        for grade, style in ((0, "-"), (1, "--"), (2, ":")):
            fpr, tpr = roc_curve(labels, probability[:, grade], grade)
            roc.plot(
                fpr,
                tpr,
                linestyle=style,
                color=colour,
                label=f"{label}, grade {grade}",
            )
    roc.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.7, linestyle="-.")
    roc.set_xlabel("1 - specificity")
    roc.set_ylabel("Sensitivity")
    roc.set_xlim(0, 1)
    roc.set_ylim(0, 1)
    roc.set_title("B  One-versus-rest ROC")
    roc.legend(frameon=False, loc="lower right", handlelength=1.4)

    any_probability = pooled(payload, "xgb_summary")[:, 1:].sum(axis=1)
    observed_predicted = bin_calibration(labels, any_probability, 1)
    calibration.plot(
        observed_predicted[0],
        observed_predicted[1],
        "o-",
        color=COLOURS["xgb_summary"],
        markersize=3,
        label="Summary XGBoost",
    )
    any_probability = pooled(payload, "dggru_tr")[:, 1:].sum(axis=1)
    observed_predicted = bin_calibration(labels, any_probability, 1)
    calibration.plot(
        observed_predicted[0],
        observed_predicted[1],
        "s-",
        color=COLOURS["dggru_tr"],
        markersize=3,
        label="DG-GRU+TR",
    )
    any_probability = blend[:, 1:].sum(axis=1)
    observed_predicted = bin_calibration(labels, any_probability, 1)
    calibration.plot(
        observed_predicted[0],
        observed_predicted[1],
        "^-",
        color=COLOURS["blend"],
        markersize=3,
        label="Equal-weight average",
    )
    calibration.plot([0, 0.6], [0, 0.6], color="#BBBBBB", linewidth=0.7, linestyle=":")
    calibration.set_xlabel("Predicted probability of any deterioration")
    calibration.set_ylabel("Observed proportion")
    calibration.set_title("C  Calibration of P(y >= 1)")
    calibration.legend(frameon=False, loc="upper left")

    prevalence = float((labels >= 1).mean())
    thresholds = np.linspace(0.05, 0.55, 51)
    treat_all = None
    for name, probability, colour in (
        (
            "Summary XGBoost",
            pooled(payload, "xgb_summary")[:, 1:].sum(axis=1),
            COLOURS["xgb_summary"],
        ),
        (
            "DG-GRU+TR",
            pooled(payload, "dggru_tr")[:, 1:].sum(axis=1),
            COLOURS["dggru_tr"],
        ),
        ("Equal-weight average", blend[:, 1:].sum(axis=1), COLOURS["blend"]),
    ):
        curve = te.decision_curve_net_benefit(
            (labels >= 1).astype(int), probability, thresholds
        )
        decision.plot(thresholds, curve["model"], color=colour, label=name)
        treat_all = curve["treat_all"]
    decision.plot(
        thresholds,
        treat_all,
        color="#BBBBBB",
        linestyle="--",
        label="Treat all",
    )
    decision.axhline(0.0, color="#BBBBBB", linewidth=0.7, linestyle=":")
    decision.set_xlabel("Threshold probability")
    decision.set_ylabel("Net benefit")
    decision.set_title("D  Decision curves for any deterioration")
    decision.legend(frameon=False, loc="upper right", fontsize=6.5)

    figure.savefig(OUTPUT / "figure2_development.png")
    plt.close(figure)


def figure_three() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    external = json.loads(
        (ROOT / "results" / "experiments" / "external_ensemble.json").read_text()
    )
    labels_by_plot = (
        ("MIMIC-III", "mimic3"),
        ("eICU-CRD", "eicu"),
    )
    model_order = (
        ("xgb_summary", "Summary XGBoost", COLOURS["xgb_summary"]),
        ("dggru_tr", "DG-GRU+TR", COLOURS["dggru_tr"]),
        ("xgb_plus_dggru_tr", "Equal-weight average", COLOURS["blend"]),
    )
    forest = axes[0]
    offsets = {"xgb_summary": -0.22, "dggru_tr": 0.0, "xgb_plus_dggru_tr": 0.22}
    for cohort_index, (cohort_name, cohort) in enumerate(labels_by_plot):
        block = external["cohorts"][cohort]
        for model, display, colour in model_order:
            position = cohort_index + offsets[model]
            auc = block[model]["macro_auc"]
            low, high = block[model]["macro_auc_ci"]
            forest.plot([position, position], [low, high], color=colour, linewidth=1.1)
            forest.plot(position, auc, "o", color=colour, markersize=4)
        forest.axvline(cohort_index + 0.5, color="#DDDDDD", linewidth=0.7)
    forest.set_xticks([0, 1], [name for name, _ in labels_by_plot])
    forest.set_xlim(-0.55, 1.55)
    forest.set_ylabel("Macro-AUC (95% CI)")
    forest.set_title("A  Transported discrimination")
    handles = [
        plt.Line2D([], [], color=colour, marker="o", markersize=4, linewidth=1.1, label=display)
        for _, display, colour in model_order
    ]
    forest.set_ylim(0.735, 0.845)
    forest.legend(handles=handles, frameon=False, loc="upper left")

    roc_axis = axes[1]
    for cohort_name, cohort, colour in (
        ("MIMIC-III", "mimic3", "#7A5AA6"),
        ("eICU-CRD", "eicu", "#2A7F62"),
    ):
        payload = np.load(
            PREDICTIONS / f"{cohort}_external_core.npz", allow_pickle=True
        )
        cohort_labels = payload["labels"]
        blend = 0.5 * payload["xgb_summary"] + 0.5 * payload["dggru_tr"]
        for grade, grade_style in ((0, ":"), (1, "-"), (2, "-.")):
            score = blend[:, grade]
            truth = (cohort_labels == grade).astype(int)
            order = np.argsort(-score)
            truth = truth[order]
            tpr = np.cumsum(truth) / max(truth.sum(), 1)
            fpr = np.cumsum(1 - truth) / max((1 - truth).sum(), 1)
            roc_axis.plot(
                np.r_[0, fpr],
                np.r_[0, tpr],
                color=colour,
                linestyle=grade_style,
                linewidth=1.0 if grade else 0.8,
            )
    roc_axis.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.7, linestyle="-.")
    roc_axis.legend(
        handles=[
            plt.Line2D([], [], color="#7A5AA6", linewidth=1.0, label="MIMIC-III"),
            plt.Line2D([], [], color="#2A7F62", linewidth=1.0, label="eICU-CRD"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=6,
    )
    roc_axis.set_xlim(0, 1)
    roc_axis.set_ylim(0, 1)
    roc_axis.set_xlabel("1 - specificity")
    roc_axis.set_ylabel("Sensitivity")
    roc_axis.set_title("B  Equal-weight average ROC by grade")

    calibration_axis = axes[2]
    for cohort_name, cohort, colour in (
        ("MIMIC-III", "mimic3", "#7A5AA6"),
        ("eICU-CRD", "eicu", "#2A7F62"),
    ):
        payload = np.load(
            PREDICTIONS / f"{cohort}_external_core.npz", allow_pickle=True
        )
        cohort_labels = payload["labels"]
        blend = 0.5 * payload["xgb_summary"] + 0.5 * payload["dggru_tr"]
        probability = blend[:, 1:].sum(axis=1)
        predicted, observed = bin_calibration(cohort_labels, probability, 1)
        calibration_axis.plot(
            predicted, observed, "o-", color=colour, markersize=3, label=cohort_name
        )
    calibration_axis.plot([0, 0.6], [0, 0.6], color="#BBBBBB", linewidth=0.7, linestyle=":")
    calibration_axis.set_xlabel("Predicted probability")
    calibration_axis.set_ylabel("Observed proportion")
    calibration_axis.set_title("C  Transported calibration")
    calibration_axis.legend(frameon=False, loc="upper left")

    figure.tight_layout()
    figure.savefig(OUTPUT / "figure3_external.png")
    plt.close(figure)


def figure_four() -> None:
    ablation = json.loads(
        (ROOT / "results" / "metrics_ablation.json").read_text()
    )["development"]
    intact = json.loads(
        (ROOT / "results" / "metrics_ablation_intact.json").read_text()
    )["development"]
    ablation["dggru_tr"] = intact["dggru_tr"]
    payload = load_repeats()
    labels = payload["labels"]
    weights = payload["repeat0_analysis_weight"]

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    forest = axes[0]
    order = [
        ("dggru_tr_static", "Admission-level static inputs"),
        ("dggru_tr_traj", "No static covariates"),
        ("dggru_tr_noseq", "Sequence ablated"),
        ("dggru_tr_treverse", "Time-reversed input"),
        ("dggru_tr_tpermute", "Time-permuted input"),
        ("dggru_laplace", "Laplace input noise"),
        ("dggru_tr_gcsmask", "GCS-only penalty mask"),
        ("dggru_tr", "Intact DG-GRU+TR"),
    ]
    for position, (key, display) in enumerate(order):
        block = ablation[key]
        point = block["point"]["macro_auc"]
        low, high = block["ci"]["macro_auc"]
        colour = COLOURS["dggru_tr"] if key == "dggru_tr" else "#6E6E6E"
        forest.plot([low, high], [position, position], color=colour, linewidth=1.3)
        forest.plot(point, position, "o", color=colour, markersize=4)
    forest.set_yticks(list(range(len(order))), [display for _, display in order])
    forest.tick_params(axis="y", labelsize=6.0)
    forest.set_xlim(0.72, 0.83)
    forest.set_xlabel("Macro-AUC (95% CI)")
    forest.set_title("A  Ablation and controls")

    lines = axes[1]
    repeats = range(5)
    xgb_values = [
        te.multiclass_auc(
            labels, payload[f"repeat{index}_xgb_summary"], sample_weight=weights
        )
        for index in repeats
    ]
    dggru_values = [
        te.multiclass_auc(
            labels, payload[f"repeat{index}_dggru_tr"], sample_weight=weights
        )
        for index in repeats
    ]
    blend_values = [
        te.multiclass_auc(
            labels,
            0.5 * payload[f"repeat{index}_xgb_summary"]
            + 0.5 * payload[f"repeat{index}_dggru_tr"],
            sample_weight=weights,
        )
        for index in repeats
    ]
    lines.plot(range(1, 6), xgb_values, "o-", color=COLOURS["xgb_summary"], label="Summary XGBoost")
    lines.plot(range(1, 6), dggru_values, "s-", color=COLOURS["dggru_tr"], label="DG-GRU+TR")
    lines.plot(range(1, 6), blend_values, "^-", color=COLOURS["blend"], label="Equal-weight average")
    lines.set_xticks(range(1, 6))
    lines.set_xlabel("Cross-validation repeat")
    lines.set_ylabel("Macro-AUC")
    lines.set_ylim(0.785, 0.815)
    lines.set_title("B  Repeat-level stability")
    lines.legend(frameon=False, loc="lower left")

    shape = json.loads(
        (ROOT / "results" / "experiments" / "gcs_shape_comparison.json").read_text()
    )["comparisons"]
    bars = axes[2]
    names = [
        ("traj_monitoring_xgb", "Monitoring\nproxies"),
        ("traj_gcs_shape_xgb", "GCS\nshape"),
        ("traj_features", "Both"),
    ]
    values = [shape[key]["difference"] for key, _ in names]
    lower = [shape[key]["paired_difference_ci"][0] for key, _ in names]
    upper = [shape[key]["paired_difference_ci"][1] for key, _ in names]
    positions = np.arange(len(names))
    bars.bar(positions, values, color=["#9A9A9A", "#2E5E8C", "#1F6F4A"], width=0.55)
    bars.errorbar(
        positions,
        values,
        yerr=[np.array(values) - np.array(lower), np.array(upper) - np.array(values)],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=0.9,
    )
    bars.axhline(0, color="#333333", linewidth=0.7)
    bars.set_xticks(positions, [display for _, display in names], fontsize=6.8)
    bars.set_ylabel("Difference in macro-AUC")
    bars.set_title("C  Added-feature increment")

    figure.tight_layout()
    figure.savefig(OUTPUT / "figure4_secondary.png")
    plt.close(figure)


def figure_ordinal() -> None:
    payload = load_repeats()
    labels = payload["labels"]
    repeated = {
        "Summary XGBoost": [
            payload[f"repeat{index}_xgb_summary"] for index in range(5)
        ],
        "DG-GRU+TR": [
            payload[f"repeat{index}_dggru_tr"] for index in range(5)
        ],
        "Equal-weight average": [
            0.5 * payload[f"repeat{index}_xgb_summary"]
            + 0.5 * payload[f"repeat{index}_dggru_tr"]
            for index in range(5)
        ],
    }
    heatmap_colours = LinearSegmentedColormap.from_list(
        "ordinal_blue", ["#F7F8F5", "#D9E7E2", "#7FB2A6", "#2D6D73"]
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for panel, (axis, (name, probabilities)) in zip(
        "ABC", zip(axes, repeated.items())
    ):
        matrices = []
        for probability in probabilities:
            prediction = te.ordinal_median_prediction(probability)
            matrix = np.zeros((3, 3), dtype=float)
            for truth, predicted in zip(labels, prediction):
                matrix[int(truth), int(predicted)] += 1.0
            matrices.append(matrix)
        matrix = np.mean(matrices, axis=0)
        row_percent = 100.0 * matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(
            row_percent,
            cmap=heatmap_colours,
            vmin=0,
            vmax=100,
            aspect="equal",
        )
        for row in range(3):
            for column in range(3):
                value = row_percent[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value >= 45 else "#24323A",
                )
        axis.set_xticks([0, 1, 2], ["Grade 0", "Grade 1", "Grade 2"])
        axis.set_yticks([0, 1, 2], ["Grade 0", "Grade 1", "Grade 2"])
        axis.set_xlabel("Predicted grade")
        axis.set_title(f"{panel}  {name}", loc="left")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
    axes[0].set_ylabel("Observed grade")
    colourbar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.042,
        pad=0.22,
        aspect=38,
    )
    colourbar.set_label("Row-normalized percentage")
    colourbar.ax.tick_params(labelsize=6.5, length=2)
    figure.savefig(OUTPUT / "figure_ordinal.png")
    plt.close(figure)


def figure_crosscohort() -> None:
    core = json.loads(
        (ROOT / "results" / "metrics_core.json").read_text()
    )
    baselines = json.loads(
        (ROOT / "results" / "metrics_baselines.json").read_text()
    )
    external = json.loads(
        (ROOT / "results" / "experiments" / "external_ensemble.json").read_text()
    )
    blend_detail = json.loads(
        (ROOT / "results" / "experiments" / "blend_metrics_detail.json").read_text()
    )
    architecture = json.loads(
        (
            ROOT
            / "results"
            / "experiments"
            / "blend_architecture_generality.json"
        ).read_text()
    )
    repeated_blends = json.loads(
        (ROOT / "results" / "experiments" / "blend_variants_repeated.json").read_text()
    )

    development = {
        "Summary XGBoost": core["development"]["xgb_summary"]["point"],
        "DG-GRU+TR": core["development"]["dggru_tr"]["point"],
        "Equal-weight average": blend_detail["xgb_plus_dggru_tr"]["per_repeat_mean"],
    }
    rows = []
    labels = []
    for cohort, block in (
        ("MIMIC-IV", development),
        ("MIMIC-III", external["cohorts"]["mimic3"]),
        ("eICU-CRD", external["cohorts"]["eicu"]),
    ):
        for model in ("Summary XGBoost", "DG-GRU+TR", "Equal-weight average"):
            if cohort == "MIMIC-IV":
                values = block[model]
            else:
                key = {
                    "Summary XGBoost": "xgb_summary",
                    "DG-GRU+TR": "dggru_tr",
                    "Equal-weight average": "xgb_plus_dggru_tr",
                }[model]
                values = block[key]
            rows.append(
                [
                    values["auc_grade_0"],
                    values["auc_grade_1"],
                    values["auc_grade_2"],
                ]
            )
            labels.append(f"{cohort} · {model}")
    values = np.asarray(rows)

    figure = plt.figure(figsize=(7.3, 4.6))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.55,
    )
    heat_axis = figure.add_subplot(grid[:, 0])
    blend_axis = figure.add_subplot(grid[0, 1])
    contrast_axis = figure.add_subplot(grid[1, 1])

    heat_colours = LinearSegmentedColormap.from_list(
        "grade_auc", ["#F4F1E8", "#C7DCD3", "#5C9B90", "#234E5A"]
    )
    image = heat_axis.imshow(
        values,
        cmap=heat_colours,
        vmin=0.70,
        vmax=0.83,
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            heat_axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value >= 0.792 else "#24323A",
            )
    heat_axis.set_xticks([0, 1, 2], ["Grade 0", "Grade 1", "Grade 2"])
    heat_axis.set_yticks(range(len(labels)), labels)
    heat_axis.tick_params(length=0)
    for boundary in (2.5, 5.5):
        heat_axis.axhline(boundary, color="white", linewidth=2.0)
    for spine in heat_axis.spines.values():
        spine.set_visible(False)
    heat_axis.set_title("A  Grade-specific discrimination across cohorts", loc="left")
    colourbar = figure.colorbar(
        image,
        ax=heat_axis,
        orientation="horizontal",
        fraction=0.055,
        pad=0.12,
        aspect=34,
    )
    colourbar.set_label("One-versus-rest AUC")
    colourbar.ax.tick_params(labelsize=6.5, length=2)

    blend_order = [
        ("transformer", "Transformer"),
        ("dggru", "DG-GRU"),
        ("dggru_tr", "DG-GRU+TR"),
        ("grud", "GRU-D"),
        ("gru", "GRU"),
    ]
    blend_axis.axvline(0.0, color="#8C8C8C", linewidth=0.8)
    for position, (key, display) in enumerate(blend_order):
        block = architecture["blends"][key]
        difference = block["mean_difference_vs_xgb"]
        low, high = block["paired_difference_ci"]
        blend_axis.hlines(position, 0, difference, color="#7D9D93", linewidth=1.2)
        blend_axis.plot(difference, position, "o", color="#1F6F4A", markersize=4.5)
        blend_axis.plot([low, high], [position, position], color="#1F6F4A", linewidth=1.1)
        blend_axis.text(
            high + 0.00035,
            position,
            f"{difference:+.3f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color="#1F4F3B",
        )
    blend_axis.set_yticks(
        range(len(blend_order)), [display for _, display in blend_order]
    )
    blend_axis.set_xlim(-0.0005, 0.0146)
    blend_axis.set_ylim(-0.6, len(blend_order) - 0.4)
    blend_axis.set_xlabel("Difference in macro-AUC vs XGBoost")
    blend_axis.set_title("B  Generality across sequence members", loc="left")
    blend_axis.tick_params(axis="y", length=0)

    contrast_rows = [
        (
            "MIMIC-IV",
            repeated_blends["blends"]["xgb+dggru_tr"]["mean_difference_vs_xgb"],
            repeated_blends["blends"]["xgb+dggru_tr"]["paired_difference_ci"],
        ),
        (
            "MIMIC-III",
            external["cohorts"]["mimic3"]["xgb_plus_dggru_tr"]["macro_auc"]
            - external["cohorts"]["mimic3"]["xgb_summary"]["macro_auc"],
            external["cohorts"]["mimic3"]["xgb_plus_dggru_tr"][
                "paired_difference_ci_vs_xgb"
            ],
        ),
        (
            "eICU-CRD",
            external["cohorts"]["eicu"]["xgb_plus_dggru_tr"]["macro_auc"]
            - external["cohorts"]["eicu"]["xgb_summary"]["macro_auc"],
            external["cohorts"]["eicu"]["xgb_plus_dggru_tr"][
                "paired_difference_ci_vs_xgb"
            ],
        ),
    ]
    contrast_axis.axvline(0.0, color="#8C8C8C", linewidth=0.8)
    for position, (cohort, difference, interval) in enumerate(contrast_rows):
        low, high = interval
        contrast_axis.plot(
            [low, high], [position, position], color="#1F6F4A", linewidth=1.2
        )
        contrast_axis.plot(
            difference, position, "D", color="#1F6F4A", markersize=4.2
        )
        contrast_axis.text(
            high + 0.00035,
            position,
            f"{difference:+.3f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color="#1F4F3B",
        )
    contrast_axis.set_yticks(
        range(len(contrast_rows)), [name for name, _, _ in contrast_rows]
    )
    contrast_axis.set_xlim(-0.006, 0.0142)
    contrast_axis.set_ylim(-0.55, len(contrast_rows) - 0.45)
    contrast_axis.set_xlabel("Equal-weight average minus XGBoost (macro-AUC)")
    contrast_axis.set_title("C  Paired difference by cohort", loc="left")
    contrast_axis.tick_params(axis="y", length=0)

    figure.savefig(OUTPUT / "figure_crosscohort.png")
    plt.close(figure)


def figure_ascertainment() -> None:
    audit = json.loads(
        (ROOT / "results" / "audits" / "monitoring_ascertainment.json").read_text()
    )
    dependence = json.loads(
        (
            ROOT
            / "results"
            / "audits"
            / "ascertainment_model_dependence.json"
        ).read_text()
    )
    cohorts = [
        ("MIMIC-IV", "mimic4", "#245A7A"),
        ("MIMIC-III", "mimic3", "#8B6A2E"),
        ("eICU-CRD", "eicu", "#7A3F5D"),
    ]
    strata = ["2-5", "6-11", "12-23", "24+"]
    positions = np.arange(len(strata))

    figure, axes = plt.subplots(2, 2, figsize=(7.3, 5.35))
    any_axis, composition_axis, model_axis, baseline_axis = axes.ravel()

    for name, key, colour in cohorts:
        block = audit["datasets"][key]
        values = [
            row["any_deterioration_percent"]
            for row in block["by_post_landmark_gcs_hours"]
        ]
        sizes = [
            18 + 2.8 * np.sqrt(row["n"])
            for row in block["by_post_landmark_gcs_hours"]
        ]
        any_axis.plot(
            positions,
            values,
            "-",
            color=colour,
            linewidth=1.15,
            marker="o",
            markersize=3.2,
            label=name,
        )
        any_axis.scatter(
            positions,
            values,
            s=sizes,
            facecolor="white",
            edgecolor=colour,
            linewidth=0.7,
            zorder=3,
        )
    any_axis.set_xticks(positions, strata)
    any_axis.set_ylim(0, 58)
    any_axis.set_xlabel("Post-landmark hours with a recorded GCS")
    any_axis.set_ylabel("Any deterioration (%)")
    any_axis.set_title("A  Outcome ascertainment", loc="left")
    any_axis.legend(frameon=False, loc="upper left")

    development_rows = audit["datasets"]["mimic4"]["by_post_landmark_gcs_hours"]
    grade_zero = np.asarray([row["grade_0_percent"] for row in development_rows])
    grade_one = np.asarray([row["grade_1_percent"] for row in development_rows])
    grade_two = np.asarray([row["grade_2_percent"] for row in development_rows])
    lower = np.zeros_like(grade_zero)
    composition_axis.bar(
        positions,
        grade_zero,
        width=0.62,
        color="#D7E3EA",
        edgecolor="white",
        linewidth=0.7,
        label="Grade 0",
    )
    composition_axis.bar(
        positions,
        grade_one,
        bottom=grade_zero,
        width=0.62,
        color="#D2B36B",
        edgecolor="white",
        linewidth=0.7,
        label="Grade 1",
    )
    composition_axis.bar(
        positions,
        grade_two,
        bottom=grade_zero + grade_one,
        width=0.62,
        color="#A9573D",
        edgecolor="white",
        linewidth=0.7,
        label="Grade 2",
    )
    composition_axis.set_xticks(
        positions,
        [
            f"{stratum}\n{row['n']}"
            for stratum, row in zip(strata, development_rows)
        ],
    )
    composition_axis.set_ylim(0, 100)
    composition_axis.set_xlabel("Post-landmark GCS hours and n")
    composition_axis.set_ylabel("Grade composition (%)")
    composition_axis.set_title("B  Grade mix in MIMIC-IV", loc="left", pad=22)
    composition_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.00),
        ncol=3,
        handlelength=0.9,
        columnspacing=0.8,
    )

    performance = dependence["post_landmark_monitoring_strata"]
    strata_performance = ["2-5", "6-11", "12-23", "24+"]
    x_performance = np.arange(len(strata_performance))
    for key, display, colour, marker in (
        ("xgb_summary", "Summary XGBoost", COLOURS["xgb_summary"], "o"),
        ("dggru_tr", "DG-GRU+TR", COLOURS["dggru_tr"], "s"),
        ("xgb_plus_gru", "XGBoost + GRU average", COLOURS["blend"], "^"),
    ):
        model_axis.plot(
            x_performance,
            [performance[stratum][key] for stratum in strata_performance],
            color=colour,
            marker=marker,
            markersize=3.5,
            label=display,
        )
    model_axis.axhline(0.80, color="#B0B0B0", linewidth=0.8, linestyle=(0, (2, 2)))
    model_axis.set_xticks(
        x_performance,
        [
            f"{stratum}\nn={performance[stratum]['n']}"
            for stratum in strata_performance
        ],
    )
    model_axis.set_ylim(0.68, 0.99)
    model_axis.set_xlabel("Post-landmark GCS hours and n")
    model_axis.set_ylabel("Macro-AUC")
    model_axis.set_title("C  Model performance by documentation", loc="left")
    model_axis.legend(frameon=False, loc="upper right")
    model_axis.text(
        0.99,
        0.03,
        "Single repeat, three seeds",
        transform=model_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#5A5A5A",
    )

    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    x_quartiles = np.arange(len(quartiles))
    for name, key, colour in cohorts:
        block = audit["datasets"][key]
        values = [
            row["any_deterioration_percent"]
            for row in block["by_pre_landmark_gcs_quartile"]
        ]
        baseline_axis.plot(
            x_quartiles,
            values,
            color=colour,
            marker="o",
            markersize=3.2,
            label=name,
        )
    baseline_axis.set_xticks(x_quartiles, quartiles)
    baseline_axis.set_ylim(0, 52)
    baseline_axis.set_xlabel("Pre-landmark maximum GCS quartile")
    baseline_axis.set_ylabel("Any deterioration (%)")
    baseline_axis.set_title("D  Baseline severity gradient", loc="left")
    baseline_axis.legend(frameon=False, loc="upper left")

    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_ascertainment.png")
    plt.close(figure)


def figure_lambda_sensitivity() -> None:
    rows = []
    for path in sorted((ROOT / "results").glob("metrics_lambda_*.json")):
        encoded = path.stem.replace("metrics_lambda_m", "")
        first, second = encoded.split("_s")
        row = {
            "lambda1": float(first.replace("p", ".")),
            "lambda2": float(second.replace("p", ".")),
            "point": json.loads(path.read_text())["development"]["dggru_tr"]["point"],
        }
        rows.append(row)
    rows.sort(key=lambda row: row["lambda1"])

    positions = np.arange(len(rows))
    macro = np.asarray([row["point"]["macro_auc"] for row in rows])
    grade_one = np.asarray([row["point"]["auc_grade_1"] for row in rows])
    calibration = np.asarray(
        [row["point"]["expected_calibration_error"] for row in rows]
    )
    selected = next(
        index
        for index, row in enumerate(rows)
        if row["lambda1"] == 1.5 and row["lambda2"] == 0.6
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)
    panels = [
        (
            "A  Macro-AUC",
            macro,
            "Macro-AUC",
            (0.7895, 0.7965),
            COLOURS["xgb_summary"],
            "o",
        ),
        (
            "B  Grade 1 AUC",
            grade_one,
            "Grade 1 AUC",
            (0.7660, 0.7770),
            COLOURS["dggru_tr"],
            "s",
        ),
        (
            "C  Calibration error",
            calibration,
            "Expected calibration error",
            (0.0120, 0.0230),
            "#6B6B6B",
            "D",
        ),
    ]
    for panel, (title, values, ylabel, limits, colour, marker), axis in zip(
        "ABC", panels, axes
    ):
        axis.plot(
            positions,
            values,
            color=colour,
            marker=marker,
            markersize=3.2,
            linewidth=1.1,
        )
        axis.scatter(
            [selected],
            [values[selected]],
            s=62,
            facecolor="none",
            edgecolor=colour,
            linewidth=0.9,
            zorder=5,
        )
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.set_xticks([0, 4, 8, 13, 17])
        axis.set_xticklabels(["0", "0.005", "0.05", "0.3", "1.5"])
        axis.set_xlabel("Grid point (increasing lambda1)")

    axes[0].annotate(
        "selected\n(1.5, 0.6)",
        xy=(selected, macro[selected]),
        xytext=(selected - 6.2, 0.7957),
        fontsize=6.4,
        color="#2A2A2A",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "#5A5A5A"},
    )
    axes[0].text(
        0.02,
        0.06,
        "Grid span 0.0042",
        transform=axes[0].transAxes,
        fontsize=6.4,
        color="#5A5A5A",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_lambda_sensitivity.png")
    plt.close(figure)


def figure_channel_coverage() -> None:
    datasets = [
        ("MIMIC-IV", "mimic4"),
        ("MIMIC-III", "mimic3"),
        ("eICU-CRD", "eicu"),
    ]
    labels = [
        "GCS eye",
        "GCS verbal",
        "GCS motor",
        "GCS total",
        "Heart rate",
        "Respiratory rate",
        "SpO2",
        "SBP",
        "DBP",
        "MBP",
        "Temperature",
        "Urine output",
        "FiO2",
        "RASS",
    ]
    colour_map = LinearSegmentedColormap.from_list(
        "coverage_teal", ["#FFFFFF", "#E2EFEC", "#93C2BA", "#2D6D73"]
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.3, 3.55), sharey=True)
    image = None
    for panel, (axis, (display, key)) in zip(
        "ABC", zip(axes, datasets)
    ):
        payload = np.load(
            ROOT
            / "data"
            / "processed"
            / "landmark_inputs"
            / key
            / "landmark_24h"
            / "sequences.npz",
            allow_pickle=True,
        )
        masks = payload["masks"].astype(float)
        coverage = 100.0 * masks.mean(axis=0).T
        image = axis.imshow(
            coverage,
            aspect="auto",
            cmap=colour_map,
            vmin=0,
            vmax=100,
            interpolation="nearest",
        )
        axis.axhline(3.5, color="#4D4D4D", linewidth=0.7)
        axis.set_xticks([0, 5, 11, 17, 23])
        axis.set_xticklabels(["0", "5", "11", "17", "23"])
        axis.set_xlabel("ICU hour")
        axis.set_title(
            f"{panel}  {display} ({100.0 * masks.mean():.1f}%)", loc="left"
        )
        axis.tick_params(length=0)

    axes[0].set_yticks(range(len(labels)), labels)
    figure.subplots_adjust(
        left=0.17, right=0.90, top=0.90, bottom=0.19, wspace=0.12
    )
    colour_axis = figure.add_axes([0.925, 0.22, 0.012, 0.62])
    colourbar = figure.colorbar(image, cax=colour_axis, orientation="vertical")
    colourbar.set_label("Stay-hours with a recorded value (%)")
    colourbar.ax.tick_params(labelsize=6.5, length=2)
    figure.savefig(OUTPUT / "figure_channel_coverage.png")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_one()
    figure_two()
    figure_three()
    figure_four()
    figure_ordinal()
    figure_crosscohort()
    figure_ascertainment()
    figure_lambda_sensitivity()
    figure_channel_coverage()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
