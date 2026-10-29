"""Create manuscript tables, calibration plots, DCA, and subgroup summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import roc_auc_score

from .common import CHANNELS, PROCESSED, RESULTS, ensure_dirs
from .train_evaluate import cumulative_auc, multiclass_auc, point_metrics


TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
COHORTS = ("mimic4", "mimic3", "eicu")
COHORT_LABELS = {
    "mimic4": "MIMIC-IV development",
    "mimic3": "MIMIC-III same-institution external",
    "eicu": "eICU geographic external",
}
PLOT_LABELS = {
    "development": "MIMIC-IV development",
    "mimic3": "MIMIC-III same-institution external",
    "eicu": "eICU geographic external",
}
MODELS = (
    "xgb_static",
    "xgb_summary",
    "gcs_slope",
    "gru",
    "grud",
    "transformer",
    "dggru",
    "dggru_tr",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
    "nested_stack",
)
MODEL_LABELS = {
    "xgb_static": "Static XGBoost",
    "xgb_summary": "Summary XGBoost",
    "gcs_slope": "GCS-slope-only logistic",
    "gru": "GRU",
    "grud": "GRU-D",
    "transformer": "Transformer",
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU + temporal risk-trajectory regularization",
    "dggru_tr_treverse": "DG-GRU + time reversal",
    "dggru_tr_tpermute": "DG-GRU + time permutation",
    "dggru_laplace": (
        "DG-GRU + generic Laplace input noise (scale=0.01)"
    ),
    "nested_stack": "Nested logistic stacking",
}
SENSITIVITY_DATASETS = ("mimic4", "mimic3", "eicu")
SENSITIVITY_DATASET_LABELS = {
    "mimic4": "MIMIC-IV",
    "mimic3": "MIMIC-III",
    "eicu": "eICU-CRD",
}
SENSITIVITY_MODELS = {
    "xgb_summary": "XGB",
    "gru": "GRU",
    "dggru": "DG-GRU",
    "dggru_tr": "Temporal reg.",
}
LANDMARK_SENSITIVITY_MODELS = {
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU + TR",
    "dggru_tr_treverse": "TR time reversed",
    "dggru_tr_tpermute": "TR time permuted",
    "dggru_laplace": "Laplace input noise",
}
DCA_THRESHOLD_MIN = 0.10
DCA_THRESHOLD_MAX = 0.40
DCA_THRESHOLDS = np.linspace(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX, 61)
CALIBRATION_TARGET_LABELS = {
    "grade_0": "Grade 0, one-vs-rest: P(Y = 0)",
    "grade_1": "Grade 1, one-vs-rest: P(Y = 1)",
    "grade_2": "Grade 2, one-vs-rest: P(Y = 2)",
    "cumulative_any": "Cumulative any: P(Y >= 1)",
    "cumulative_severe": "Cumulative severe: P(Y >= 2)",
}
CALIBRATION_TARGETS = tuple(CALIBRATION_TARGET_LABELS)
ARCHITECTURE_SEQUENCE_TEXT = "14 channels + observation masks"
ARCHITECTURE_ORDINAL_TEXT = (
    "Final risk + shared representation\n"
    "with two cumulative-logit branches"
)
FIGURE3_STEM = "figure3_model_performance"
FIGURE3_CALIBRATION_MODELS = (
    "xgb_summary",
    "gcs_slope",
    "gru",
    "dggru_tr",
)
FIGURE4_CALIBRATION_STEM = "figure4_calibration"
FIGURE4_DCA_STEM = "figure4_decision_curve"
FIGURE6_SUBGROUP_STEM = "figure6_subgroups"
COLORS = {
    "xgb_static": "#7C6E65",
    "xgb_summary": "#D55E00",
    "gcs_slope": "#8C564B",
    "gru": "#0072B2",
    "grud": "#56B4E9",
    "transformer": "#332288",
    "dggru": "#009E73",
    "dggru_tr": "#CC79A7",
    "dggru_tr_treverse": "#999933",
    "dggru_tr_tpermute": "#44AA99",
    "dggru_laplace": "#E69F00",
    "nested_stack": "#000000",
}


def ensure_analysis_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def fmt_median(series: pd.Series, digits: int = 1) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    low, high = np.percentile(values, [25, 75])
    return f"{values.median():.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def fmt_count(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = int(values.sum())
    denominator = len(values)
    return f"{n} ({100.0 * n / denominator:.1f}%)" if denominator else "NA"


def fmt_binary(mask: pd.Series) -> str:
    values = pd.Series(mask, dtype=bool)
    n = int(values.sum())
    denominator = len(values)
    return f"{n} ({100.0 * n / denominator:.1f}%)" if denominator else "NA"


def p_value_continuous(frame: pd.DataFrame, column: str) -> str:
    groups = [
        pd.to_numeric(group[column], errors="coerce").dropna()
        for _, group in frame.groupby("outcome")
    ]
    groups = [values for values in groups if len(values) > 0]
    if len(groups) < 2:
        return "NA"
    value = float(kruskal(*groups).pvalue)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def p_value_categorical(frame: pd.DataFrame, column: str) -> str:
    table = pd.crosstab(frame["outcome"], frame[column].fillna(0))
    table = table.loc[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return "NA"
    value = float(chi2_contingency(table).pvalue)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _post_landmark_gcs_hours(dataset: str) -> pd.Series:
    """Count distinct ICU hours with a valid total GCS in [24, 72)."""
    path = PROCESSED / dataset / "gcs_events.csv"
    if not path.is_file():
        return pd.Series(dtype=float, name="post_landmark_gcs_hours")
    events = pd.read_csv(path, usecols=["stay_id", "time_hour", "gcs"])
    events["stay_id"] = pd.to_numeric(events["stay_id"], errors="coerce")
    events["time_hour"] = pd.to_numeric(events["time_hour"], errors="coerce")
    events["gcs"] = pd.to_numeric(events["gcs"], errors="coerce")
    events = events[
        events["time_hour"].ge(24)
        & events["time_hour"].lt(72)
        & events["gcs"].between(3, 15)
    ]
    events["hour_bin"] = np.floor(events["time_hour"])
    return (
        events.groupby("stay_id")["hour_bin"]
        .nunique()
        .rename("post_landmark_gcs_hours")
    )


def create_table1() -> None:
    cohort_columns = tuple(COHORT_LABELS[name] for name in COHORTS)
    source_rows = [
        {
            "Characteristic": "Database version",
            cohort_columns[0]: "3.1",
            cohort_columns[1]: "1.4",
            cohort_columns[2]: "2.0",
        },
        {
            "Characteristic": "Database coverage period",
            cohort_columns[0]: "2008-2022",
            cohort_columns[1]: "2001-2012",
            cohort_columns[2]: "2014-2015",
        },
        {
            "Characteristic": "GCS documentation",
            cohort_columns[0]: (
                "Itemized eye, verbal, and motor components in MetaVision; "
                "total GCS reconstructed"
            ),
            cohort_columns[1]: (
                "Itemized components plus CareVue total-GCS records; total GCS "
                "harmonized"
            ),
            cohort_columns[2]: (
                "Nursing-recorded GCS entries; itemized or total values "
                "harmonized"
            ),
        },
        {
            "Characteristic": "P020 endpoint status",
            **{
                COHORT_LABELS[name]: (
                    "Partial: GCS reduction or death; verified unplanned "
                    "intubation unavailable"
                )
                for name in COHORTS
            },
        },
        {"Characteristic": "Primary analytic cohort, n"},
    ]
    variables = [
        ("Age, years", "age", "continuous", 1),
        ("Male sex", "sex_male", "binary", None),
        ("White race", "race_white", "binary", None),
        ("Emergency admission", "emergency_admission", "binary", None),
        ("Baseline GCS, maximum 0-24 h", "baseline_gcs_max", "continuous", 1),
        ("Last GCS before landmark", "baseline_gcs_last", "continuous", 1),
        (
            "Recorded deep sedation (RASS <= -3) in first 24 h",
            "any_deep_sedation_24h",
            "binary",
            None,
        ),
        (
            "Hours with observed total GCS in 0-24 h",
            "gcs_total_n",
            "continuous",
            1,
        ),
        ("Observed total-GCS hours 0-5, n (%)", "gcs_total_n", "le5", None),
        ("Observed total-GCS hours 6-11, n (%)", "gcs_total_n", "6to11", None),
        ("Observed total-GCS hours 12-17, n (%)", "gcs_total_n", "12to17", None),
        ("Observed total-GCS hours 18-23, n (%)", "gcs_total_n", "18to23", None),
        ("Observed total-GCS hours 24, n (%)", "gcs_total_n", "eq24", None),
        (
            "Hours with observed total GCS in 24-72 h",
            "__post_gcs__",
            "continuous",
            0,
        ),
        (
            "Post-landmark total GCS recorded in >=24 h, n (%)",
            "__post_gcs_ge24__",
            "binary",
            None,
        ),
        ("Heart rate, mean", "hr_mean", "continuous", 1),
        ("Systolic blood pressure, mean", "sbp_mean", "continuous", 1),
        ("Oxygen saturation, mean", "spo2_mean", "continuous", 1),
        ("Temperature, mean", "temp_mean", "continuous", 1),
        ("Any missing 14-channel summary", "__any_channel__", "any_missing", None),
        ("Missing age", "age", "missing", None),
        ("ICU length of stay, hours", "icu_out_hour", "continuous", 1),
        ("Death within 72 h", "death_within_72h", "binary", None),
    ]
    frames = {name: pd.read_csv(PROCESSED / name / "cohort.csv") for name in COHORTS}
    post_hours = {
        name: _post_landmark_gcs_hours(name) for name in COHORTS
    }
    source_rows[4] = {
        "Characteristic": "Primary analytic cohort, n",
        **{
            COHORT_LABELS[name]: f"{len(frames[name]):,}"
            for name in COHORTS
        },
    }
    rows: list[dict[str, str]] = source_rows
    for label, column, kind, digits in variables:
        row = {"Characteristic": label}
        for name, frame in frames.items():
            if column == "__post_gcs__":
                counts = frame["stay_id"].map(post_hours[name]).fillna(0)
                row[COHORT_LABELS[name]] = fmt_median(counts, 0)
                continue
            if column == "__post_gcs_ge24__":
                counts = frame["stay_id"].map(post_hours[name]).fillna(0)
                row[COHORT_LABELS[name]] = fmt_binary(counts.ge(24))
                continue
            if column == "__any_channel__":
                channel_columns = [
                    f"{channel}_mean" for channel in CHANNELS if f"{channel}_mean" in frame
                ]
                if not channel_columns:
                    row[COHORT_LABELS[name]] = "NA"
                    continue
                missing = frame[channel_columns].isna().any(axis=1)
                row[COHORT_LABELS[name]] = fmt_binary(missing)
                continue
            if column not in frame:
                row[COHORT_LABELS[name]] = "NA"
                continue
            if kind == "continuous":
                row[COHORT_LABELS[name]] = fmt_median(frame[column], digits)
            elif kind == "missing":
                row[COHORT_LABELS[name]] = fmt_binary(frame[column].isna())
            elif kind == "le5":
                row[COHORT_LABELS[name]] = fmt_binary(frame[column] <= 5)
            elif kind == "6to11":
                row[COHORT_LABELS[name]] = fmt_binary(
                    frame[column].between(6, 11)
                )
            elif kind == "12to17":
                row[COHORT_LABELS[name]] = fmt_binary(
                    frame[column].between(12, 17)
                )
            elif kind == "18to23":
                row[COHORT_LABELS[name]] = fmt_binary(
                    frame[column].between(18, 23)
                )
            elif kind == "eq24":
                row[COHORT_LABELS[name]] = fmt_binary(frame[column].eq(24))
            else:
                row[COHORT_LABELS[name]] = fmt_count(frame[column])
        rows.append(row)

    development = frames["mimic4"].copy()
    outcome_rows: list[dict[str, str]] = []
    for label, column, kind, digits in variables:
        if kind not in {"continuous", "binary"}:
            continue
        if column not in development:
            continue
        row = {"Characteristic": label}
        for grade in (0, 1, 2):
            subset = development[development["outcome"] == grade]
            if kind == "continuous":
                row[f"Grade {grade}"] = fmt_median(subset[column], digits)
            else:
                row[f"Grade {grade}"] = fmt_count(subset[column])
        row["p value"] = (
            p_value_continuous(development, column)
            if kind == "continuous"
            else p_value_categorical(development, column)
        )
        outcome_rows.append(row)

    cohort_table = pd.DataFrame(rows)
    outcome_table = pd.DataFrame(outcome_rows)
    cohort_table.to_csv(TABLES / "table1a_cohorts.csv", index=False, encoding="utf-8-sig")
    outcome_table.to_csv(TABLES / "table1b_outcomes.csv", index=False, encoding="utf-8-sig")
    (TABLES / "table1.md").write_text(
        "# Table 1. Cohort characteristics\n\n"
        + cohort_table.to_markdown(index=False)
        + "\n\n# Development cohort by outcome grade\n\n"
        + outcome_table.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def metric_block(cohort: str) -> dict:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    return (
        metrics["development"]
        if cohort == "development"
        else metrics["external"][cohort]
    )


def create_performance_tables() -> None:
    rows = []
    for cohort in ("development", "mimic3", "eicu"):
        block = metric_block(cohort)
        for model in MODELS:
            point = block[model]["point"]
            ci = block[model]["ci"]
            grade_one_predictions = point["predicted_grade_1"]
            grade_one_display = (
                f"{grade_one_predictions:.1f}"
                if cohort == "development"
                else str(int(grade_one_predictions))
            )
            rows.append(
                {
                    "Cohort": COHORT_LABELS[
                        "mimic4" if cohort == "development" else cohort
                    ],
                    "Model": MODEL_LABELS[model],
                    "n": int(point["n"]),
                    "Macro AUC (95% CI)": (
                        f"{point['macro_auc']:.3f} "
                        f"({ci['macro_auc'][0]:.3f}-{ci['macro_auc'][1]:.3f})"
                    ),
                    "Any deterioration AUC (95% CI)": (
                        f"{point['auc_any_deterioration']:.3f} "
                        f"({ci['auc_any_deterioration'][0]:.3f}-"
                        f"{ci['auc_any_deterioration'][1]:.3f})"
                    ),
                    "Severe deterioration AUC (95% CI)": (
                        f"{point['auc_severe_deterioration']:.3f} "
                        f"({ci['auc_severe_deterioration'][0]:.3f}-"
                        f"{ci['auc_severe_deterioration'][1]:.3f})"
                    ),
                    "Grade 1 AUC (95% CI)": (
                        f"{point['auc_grade_1']:.3f} "
                        f"({ci['auc_grade_1'][0]:.3f}-"
                        f"{ci['auc_grade_1'][1]:.3f})"
                    ),
                    "Predicted grade 1, mean n": grade_one_display,
                    "QWK": f"{point['quadratic_weighted_kappa']:.3f}",
                    "Brier score": f"{point['multiclass_brier']:.3f}",
                    "Calibration slope": f"{point['calibration_slope_any']:.3f}",
                    "Calibration intercept": f"{point['calibration_intercept_any']:.3f}",
                    "Modal ECE": f"{point['expected_calibration_error']:.3f}",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table3_model_performance.csv", index=False, encoding="utf-8-sig")
    (TABLES / "table3.md").write_text(
        "# Table 3. Model performance\n\n" + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def create_grade_calibration_table() -> None:
    """Export calibration slopes and intercepts for every ordinal target."""
    rows = []
    for cohort in ("development", "mimic3", "eicu"):
        block = metric_block(cohort)
        for model in MODELS:
            point = block[model]["point"]
            row = {
                "Cohort": COHORT_LABELS[
                    "mimic4" if cohort == "development" else cohort
                ],
                "Model": MODEL_LABELS[model],
                "n": int(point["n"]),
            }
            for target in (
                "grade_0",
                "grade_1",
                "grade_2",
                "any",
                "severe",
            ):
                slope_key = f"calibration_slope_{target}"
                intercept_key = f"calibration_intercept_{target}"
                if slope_key not in point or intercept_key not in point:
                    raise ValueError(
                        f"{cohort}/{model} lacks {slope_key} or "
                        f"{intercept_key}."
                    )
                row[f"{target} slope"] = float(point[slope_key])
                row[f"{target} intercept"] = float(point[intercept_key])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(
        TABLES / "table_s12_grade_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / "table_s12_grade_calibration.md").write_text(
        "# Supplementary Table S12. Ordinal-grade calibration\n\n"
        + table.to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )


def create_lambda_table() -> None:
    rows = []
    paths = sorted(RESULTS.glob("metrics_lambda_*.json"))
    if not paths:
        raise FileNotFoundError(
            "No lambda-grid metrics were found. Run run_lambda_grid.ps1 "
            "before make_tables_figures."
        )
    for path in paths:
        tag = path.stem.removeprefix("metrics_")
        manifest_path = RESULTS / f"run_manifest_{tag}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing lambda-grid manifest for {path.name}: "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        point = json.loads(path.read_text(encoding="utf-8"))["development"][
            "dggru_tr"
        ]["point"]
        rows.append(
            {
                "lambda_monotone": float(manifest["lambda_monotone"]),
                "lambda_smooth": float(manifest["lambda_smooth"]),
                "Macro AUC": point["macro_auc"],
                "Any AUC": point["auc_any_deterioration"],
                "Severe AUC": point["auc_severe_deterioration"],
                "Brier score": point["multiclass_brier"],
                "Calibration slope": point["calibration_slope_any"],
                "ECE": point["expected_calibration_error"],
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["lambda_smooth", "lambda_monotone"]
        ).reset_index(drop=True)
        table.to_csv(TABLES / "table_s4_lambda_sensitivity.csv", index=False)
        (TABLES / "table_s4.md").write_text(
            "# Supplementary Table S4. Temporal-regularization sensitivity\n\n"
            + table.to_markdown(index=False, floatfmt=".4f")
            + "\n",
            encoding="utf-8",
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_cohort_flow_table_and_figure() -> None:
    flow_parts = []
    for cohort in COHORTS:
        flow = pd.read_csv(PROCESSED / cohort / "cohort_flow.csv")
        flow_parts.append(flow)
    flow = pd.concat(flow_parts, ignore_index=True)
    flow.to_csv(TABLES / "table_s1_cohort_flow.csv", index=False, encoding="utf-8-sig")

    figure = plt.figure(figsize=(13.0, 10.2))
    grid = figure.add_gridspec(2, 1, height_ratios=(4.2, 1.45), hspace=0.20)
    axis = figure.add_subplot(grid[0])
    axis.set_xlim(0, 11.8)
    axis.set_ylim(0, 7.6)
    axis.axis("off")
    axis.text(
        0.0,
        7.5,
        "A. Cohort flow",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    x_positions = (0.25, 4.15, 8.05)
    box_width = 3.25
    labels = (
        "ICH ICU stays",
        "First ICH ICU stay per\npatient and adult",
        "Primary analytic cohort\ncomplete 24-72 h follow-up",
    )
    source_step = (
        "First ICH ICU stay per patient and adult "
        "(common landmark candidate source)"
    )
    for row, cohort in enumerate(COHORTS):
        center_y = 6.0 - row * 2.0
        ordered_flow = flow_parts[row].reset_index(drop=True)
        frame = ordered_flow.set_index("step")["n"]
        counts = (
            int(frame["ICH ICU stays"]),
            int(frame[source_step]),
            int(frame["Primary complete follow-up and valid outcome"]),
        )
        source_position = int(
            ordered_flow.index[
                ordered_flow["step"].eq(source_step)
            ][0]
        )
        primary_position = int(
            ordered_flow.index[
                ordered_flow["step"].eq(
                    "Primary complete follow-up and valid outcome"
                )
            ][0]
        )
        initial_rows = ordered_flow.iloc[:source_position]
        landmark_rows = ordered_flow.iloc[source_position + 1 : primary_position]
        initial_reasons = [
            (str(record.step), int(record.n))
            for record in initial_rows.itertuples(index=False)
            if str(record.step).startswith("Excluded:")
        ]
        landmark_reasons = [
            (str(record.step), int(record.n))
            for record in landmark_rows.itertuples(index=False)
            if str(record.step).startswith("Excluded:")
        ]
        initial_sum = sum(value for _, value in initial_reasons)
        if initial_sum != counts[0] - counts[1]:
            if initial_reasons:
                raise ValueError(
                    f"{cohort} initial screening exclusions sum to "
                    f"{initial_sum}, expected {counts[0] - counts[1]}."
                )
            initial_reasons = [
                ("Excluded: initial eligibility not met", counts[0] - counts[1])
            ]
        landmark_sum = sum(value for _, value in landmark_reasons)
        expected_landmark = counts[1] - counts[2]
        if landmark_sum != expected_landmark:
            raise ValueError(
                f"{cohort} landmark exclusions sum to {landmark_sum}, "
                f"expected {expected_landmark}; duplicated or missing "
                "eligibility reasons must be fixed before Figure 1."
            )
        reason_text = "; ".join(
            f"{reason.removeprefix('Excluded: ')} {value:,}"
            for reason, value in initial_reasons + landmark_reasons
        )
        axis.text(
            0.15,
            center_y + 0.78,
            COHORT_LABELS[cohort],
            fontsize=10,
            fontweight="bold",
            ha="left",
        )
        for column, (x, label, count) in enumerate(zip(x_positions, labels, counts)):
            color = "#DCEAF2" if column == 2 else "#F3F4F5"
            rect = Rectangle(
                (x, center_y - 0.38),
                box_width,
                0.98,
                facecolor=color,
                edgecolor="#4D4D4D",
                linewidth=0.8,
            )
            axis.add_patch(rect)
            axis.text(
                x + box_width / 2,
                center_y + 0.18,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            axis.text(
                x + box_width / 2,
                center_y - 0.13,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
            )
            if column < 2:
                arrow = FancyArrowPatch(
                    (x + box_width + 0.04, center_y + 0.1),
                    (x_positions[column + 1] - 0.04, center_y + 0.1),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.8,
                    color="#666666",
                )
                axis.add_patch(arrow)
                excluded = (
                    counts[0] - counts[1]
                    if column == 0
                    else counts[1] - counts[2]
                )
                if column == 0:
                    axis.text(
                        x + box_width + 0.33,
                        center_y + 0.36,
                        f"excluded: {excluded:,}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#8A3B2E",
                    )
        axis.text(
            4.15,
            center_y - 0.72,
            f"Excluded: {reason_text}",
            ha="left",
            va="top",
            fontsize=6.8,
            color="#8A3B2E",
            wrap=True,
        )

    outcome_axis = figure.add_subplot(grid[1])
    outcome_axis.set_title(
        "B. Ordinal outcome distribution",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    cohort_labels = [COHORT_LABELS[cohort] for cohort in COHORTS]
    outcomes = np.array(
        [
            [
                int(
                    flow_parts[index]
                    .set_index("step")
                    .loc[f"Outcome {grade}", "n"]
                )
                for grade in (0, 1, 2)
            ]
            for index in range(len(COHORTS))
        ],
        dtype=float,
    )
    proportions = outcomes / outcomes.sum(axis=1, keepdims=True)
    y_positions = np.arange(len(COHORTS))
    left = np.zeros(len(COHORTS), dtype=float)
    grade_colors = ("#4C78A8", "#F2A541", "#B64040")
    for grade, color in enumerate(grade_colors):
        outcome_axis.barh(
            y_positions,
            proportions[:, grade],
            left=left,
            color=color,
            edgecolor="white",
            height=0.55,
            label=f"Grade {grade}",
        )
        for row in y_positions:
            value = proportions[int(row), grade]
            if value >= 0.07:
                outcome_axis.text(
                    left[int(row)] + value / 2,
                    row,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if grade != 1 else "#222222",
                )
        left += proportions[:, grade]
    outcome_axis.set_yticks(y_positions)
    outcome_axis.set_yticklabels(
        [
            f"{label}\n(n={int(total):,})"
            for label, total in zip(cohort_labels, outcomes.sum(axis=1))
        ],
        fontsize=8,
    )
    outcome_axis.set_xlim(0, 1)
    outcome_axis.set_xlabel("Proportion of analytic cohort")
    outcome_axis.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{100 * value:.0f}%")
    )
    outcome_axis.legend(
        ncol=3,
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    outcome_axis.grid(axis="x", alpha=0.2)
    outcome_axis.invert_yaxis()
    figure.tight_layout()
    save_figure(figure, "figure1_cohort_flow")


def create_missingness_table() -> None:
    manifest = json.loads(
        (RESULTS / "run_manifest.json").read_text(encoding="utf-8")
    )
    feature_columns = list(manifest["features"])
    rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(PROCESSED / cohort / "cohort.csv")
        for column in feature_columns:
            rows.append(
                {
                    "Cohort": COHORT_LABELS[cohort],
                    "Variable": column,
                    "Missing, n (%)": (
                        f"{int(frame[column].isna().sum())} "
                        f"({100 * frame[column].isna().mean():.1f}%)"
                        if column in frame
                        else "Not harmonized"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table_s2_missingness.csv", index=False, encoding="utf-8-sig")


def create_measurement_audit_table() -> None:
    rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(PROCESSED / cohort / "cohort.csv")
        gcs = pd.read_csv(PROCESSED / cohort / "gcs_events.csv")
        post = gcs[gcs["time_hour"].between(24, 72, inclusive="left")].copy()
        counts = post.groupby("stay_id").size().rename("post_gcs_n")
        if cohort == "eicu":
            any_ett = pd.DataFrame(
                {
                    "stay_id": frame["stay_id"].to_numpy(),
                    "any_ett": np.nan,
                }
            )
        else:
            post["ett_verbal"] = pd.to_numeric(
                post["ett_verbal"], errors="coerce"
            ).fillna(0)
            any_ett = (
                post.groupby("stay_id")["ett_verbal"].max().rename("any_ett")
            )
        frame = frame.merge(counts, on="stay_id", how="left").merge(
            any_ett, on="stay_id", how="left"
        )
        for grade in (0, 1, 2):
            subset = frame[frame["outcome"] == grade]
            rows.append(
                {
                    "Cohort": COHORT_LABELS[cohort],
                    "Grade": grade,
                    "n": len(subset),
                    "Death by 72 h, n": int(
                        subset["death_within_72h"].fillna(False).astype(bool).sum()
                    ),
                    "Patients with any ETT-flagged post-landmark GCS, n (%)": (
                        "Not harmonized"
                        if cohort == "eicu"
                        else (
                            f"{int(subset['any_ett'].fillna(False).sum())} "
                            f"({100 * subset['any_ett'].fillna(False).mean():.1f}%)"
                        )
                    ),
                    "Median post-landmark GCS measurements": float(
                        subset["post_gcs_n"].median()
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        TABLES / "table_s4_measurement_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _sensitivity_audit_records(results_root: Path) -> list[tuple[str, dict]]:
    """Load current dataset-scoped audits without discarding failed variants."""
    records: list[tuple[str, dict]] = []
    for dataset in SENSITIVITY_DATASETS:
        audit_path = results_root / "sensitivity" / dataset / "scenario_audit.json"
        if not audit_path.exists():
            raise FileNotFoundError(
                "Missing current sensitivity audit: "
                f"{audit_path}. Run sensitivity_analysis before reporting."
            )
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{audit_path} must contain a JSON list.")
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError(f"{audit_path} contains a non-object record.")
            records.append((dataset, record))
    return records


def _sensitivity_outcome_counts(record: dict) -> tuple[int | None, str]:
    outcomes = record.get("outcomes")
    if isinstance(outcomes, dict):
        return (
            int(sum(int(value) for value in outcomes.values())),
            "/".join(str(int(outcomes.get(str(grade), 0))) for grade in range(3)),
        )
    n = record.get("n")
    return (int(n) if n is not None else None, "NA")


def _scenario_metrics(metrics: dict) -> dict:
    """Support both current flat metrics and older development-wrapped metrics."""
    development = metrics.get("development")
    return development if isinstance(development, dict) else metrics


def create_sensitivity_table(
    results_root: Path = RESULTS,
    tables_dir: Path = TABLES,
) -> None:
    """Render every audited sensitivity variant and its exact support status."""
    rows = []
    for dataset, record in _sensitivity_audit_records(results_root):
        scenario = record.get("scenario") or {}
        status = record.get("status") or {}
        scenario_id = str(scenario.get("scenario_id", "")).strip()
        if not scenario_id:
            raise ValueError(
                f"A sensitivity audit record for {dataset} has no scenario_id."
            )
        support = str(status.get("support", "unknown"))
        model_run_allowed = bool(status.get("model_run_allowed", False))
        n, grade_counts = _sensitivity_outcome_counts(record)
        row = {
            "Outline analysis": str(
                scenario.get("analysis_title", scenario.get("analysis_id", ""))
            ),
            "Scenario ID": scenario_id,
            "Variant": str(scenario.get("variant_title", scenario_id)),
            "Dataset": SENSITIVITY_DATASET_LABELS.get(dataset, dataset),
            "Support": support,
            "Model run allowed": model_run_allowed,
            "n": n if n is not None else "NA",
            "Grade 0/1/2": grade_counts,
            "Status reason": str(status.get("reason", "")),
        }
        if model_run_allowed:
            metrics_path = (
                results_root / "sensitivity" / dataset / f"{scenario_id}.json"
            )
            prediction_path = (
                results_root
                / "sensitivity"
                / dataset
                / f"{scenario_id}_predictions.npz"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"The audit marks {dataset}/{scenario_id} runnable, but "
                    f"scenario metrics are missing: {metrics_path}"
                )
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"The audit marks {dataset}/{scenario_id} runnable, but "
                    f"scenario predictions are missing: {prediction_path}"
                )
            metrics = _scenario_metrics(
                json.loads(metrics_path.read_text(encoding="utf-8"))
            )
            with np.load(prediction_path, allow_pickle=True) as data:
                if "labels" not in data:
                    raise ValueError(
                        f"Sensitivity predictions lack labels: {prediction_path}"
                    )
                labels = data["labels"].astype(int)
            row["n"] = int(len(labels))
            row["Grade 0/1/2"] = "/".join(
                str(int((labels == grade).sum())) for grade in range(3)
            )
            n_labels = len(labels)
            any_count = int((labels >= 1).sum())
            severe_count = int((labels == 2).sum())
            row["Any deterioration, n (%)"] = (
                f"{any_count} ({100 * any_count / n_labels:.1f}%)"
                if n_labels
                else "NA"
            )
            row["Severe deterioration, n (%)"] = (
                f"{severe_count} ({100 * severe_count / n_labels:.1f}%)"
                if n_labels
                else "NA"
            )
            model_labels = (
                LANDMARK_SENSITIVITY_MODELS
                if scenario_id.startswith("s6_landmark_")
                else SENSITIVITY_MODELS
            )
            for model, prefix in model_labels.items():
                if model not in metrics:
                    raise ValueError(
                        f"{dataset}/{scenario_id} is model-runnable but "
                        f"does not contain the prescribed model {model}."
                    )
                model_point = metrics[model]["point"]
                row[f"{prefix} macro AUC"] = model_point["macro_auc"]
                row[f"{prefix} any AUC"] = model_point[
                    "auc_any_deterioration"
                ]
                row[f"{prefix} Brier"] = model_point["multiclass_brier"]
        elif support in {"supported", "outcome_only", "unsupported"}:
            row["Status reason"] = row["Status reason"] or (
                "Model artifacts are unavailable for this support level."
            )
        rows.append(row)

    if not rows:
        raise ValueError("The current sensitivity audits contain no records.")
    table = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        tables_dir / "table_s3_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (tables_dir / "table_s3_sensitivity.md").write_text(
        "# Supplementary Table S3. Sensitivity analyses\n\n"
        + table.to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )


def create_model_performance_figure() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 6.2),
        gridspec_kw={"width_ratios": (1.25, 1.0, 1.0)},
    )
    y = np.arange(len(MODELS))

    development = metric_block("development")
    for index, model in enumerate(MODELS):
        point = development[model]["point"]["macro_auc"]
        ci = development[model]["ci"]["macro_auc"]
        axes[0].errorbar(
            point,
            index,
            xerr=[[point - ci[0]], [ci[1] - point]],
            fmt="o",
            color=COLORS[model],
            capsize=3,
            markersize=5,
        )
    axes[0].set_xlim(0.45, 0.9)
    axes[0].set_title("A. Development discrimination", fontsize=10)
    axes[0].set_xlabel("Macro one-vs-rest AUC")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [MODEL_LABELS[model] for model in MODELS],
        fontsize=8,
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)

    calibration_data = load_predictions("mimic4_cv")
    calibration_labels = calibration_data["labels"].astype(int)
    calibration_repeated = load_repeated_predictions("mimic4_cv")
    for model in FIGURE3_CALIBRATION_MODELS:
        expected, observed = calibration_points_repeated(
            calibration_labels,
            calibration_repeated[model],
            target="cumulative_any",
        )
        axes[1].plot(
            expected,
            observed,
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#888888",
        linewidth=1,
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B. Development calibration: cumulative any", fontsize=10)
    axes[1].set_xlabel("Predicted probability, P(Y >= 1)")
    axes[1].set_ylabel("Observed proportion")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")

    for cohort, marker, offset in (
        ("mimic3", "s", -0.12),
        ("eicu", "D", 0.12),
    ):
        block = metric_block(cohort)
        for index, model in enumerate(MODELS):
            point = block[model]["point"]["macro_auc"]
            ci = block[model]["ci"]["macro_auc"]
            axes[2].errorbar(
                point,
                index + offset,
                xerr=[[point - ci[0]], [ci[1] - point]],
                fmt=marker,
                color=COLORS[model],
                capsize=2,
                markersize=4,
                label=PLOT_LABELS[cohort] if index == 0 else None,
            )
    axes[2].set_xlim(0.45, 0.9)
    axes[2].set_title("C. External transport discrimination", fontsize=10)
    axes[2].set_xlabel("Macro one-vs-rest AUC")
    axes[2].set_yticks(y)
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle(
        "Figure 3. Performance, calibration, and cross-database transport",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, FIGURE3_STEM)


def create_model_architecture_figure() -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, axis = plt.subplots(figsize=(13.0, 6.0))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 6)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        detail: str,
        color: str,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8,
            edgecolor="#4D4D4D",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.30,
            detail,
            ha="center",
            va="center",
            fontsize=7.2,
            linespacing=1.2,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        style: str = "-",
        curve: float = 0.0,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                linestyle=style,
                color="#4D4D4D",
                connectionstyle=f"arc3,rad={curve}",
            )
        )

    box(
        0.15,
        3.95,
        2.15,
        1.10,
        "ICU hours 0-23",
        ARCHITECTURE_SEQUENCE_TEXT,
        "#E8F1F4",
    )
    box(2.75, 3.95, 2.10, 1.10, "Dynamic feature graph", "Value-aware hourly adjacency", "#E6F0EA")
    box(5.30, 3.95, 1.90, 1.10, "GRU encoder", "Masked 24-hour hidden states", "#E6F0EA")
    box(7.65, 3.95, 2.10, 1.10, "Latent risk trajectory", "r(t), t = 1, ..., 24", "#F5E8EC")
    box(10.20, 4.65, 2.55, 0.82, "Temporal regularizer", "Monotonicity + smoothness\n(training only)", "#F5E8EC")

    box(0.15, 0.80, 2.15, 1.35, "Static predictors", "Age, sex, race, GCS,\nvital and laboratory summaries", "#F3F1E8")
    box(2.75, 0.80, 2.10, 1.35, "Static MLP", "Layer-normalized static\nrepresentation", "#F3F1E8")
    box(
        5.75,
        0.80,
        2.10,
        1.35,
        "Cumulative ordinal head",
        ARCHITECTURE_ORDINAL_TEXT,
        "#E8F1F4",
    )
    box(9.30, 0.80, 2.55, 1.35, "Three-level output", "Grade 0 / Grade 1 / Grade 2\nprobabilities", "#E8F1F4")

    arrow((2.30, 4.50), (2.75, 4.50))
    arrow((4.85, 4.50), (5.30, 4.50))
    arrow((7.20, 4.50), (7.65, 4.50))
    arrow((9.75, 4.78), (10.20, 5.00), style="--")
    arrow((2.30, 1.48), (2.75, 1.48))
    arrow((4.85, 1.48), (5.75, 1.48))
    arrow((6.45, 3.95), (6.45, 2.15), curve=-0.12)
    arrow((8.70, 3.95), (7.25, 2.15), curve=0.18)
    arrow((7.85, 1.48), (9.30, 1.48))

    axis.text(
        9.80,
        3.40,
        "All prediction inputs precede the ICU hour-24 landmark",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#8A3B2E",
    )
    fig.suptitle(
        "Figure 2. Dynamic graph network with two cumulative-logit branches",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, "figure2_model_architecture")


def _calibration_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the predicted probability and binary target for one calibration scale."""
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Ordinal probabilities must have shape (n, 3).")
    if len(labels) != len(probabilities):
        raise ValueError("Calibration labels and probabilities are misaligned.")
    if target_name == "grade_0":
        return probabilities[:, 0], (labels == 0).astype(int)
    if target_name == "grade_1":
        return probabilities[:, 1], (labels == 1).astype(int)
    if target_name == "grade_2":
        return probabilities[:, 2], (labels == 2).astype(int)
    if target_name == "cumulative_any":
        return probabilities[:, 1:].sum(axis=1), (labels >= 1).astype(int)
    if target_name == "cumulative_severe":
        return probabilities[:, 2], (labels >= 2).astype(int)
    raise ValueError(
        f"Unknown calibration target {target_name!r}; expected one of "
        f"{', '.join(CALIBRATION_TARGETS)}."
    )


def calibration_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 8,
    *,
    target: str = "cumulative_any",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed-bin points for an explicitly named ordinal calibration scale."""
    predicted, binary_target = _calibration_arrays(labels, probabilities, target)
    edges = np.linspace(0, 1, bins + 1)
    observed = []
    expected = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted >= low) & (
            predicted < high if high < 1 else predicted <= high
        )
        if mask.sum() >= 10:
            observed.append(binary_target[mask].mean())
            expected.append(predicted[mask].mean())
    return np.asarray(expected), np.asarray(observed)


def calibration_points_repeated(
    labels: np.ndarray,
    repeated_probabilities: list[np.ndarray],
    bins: int = 8,
    *,
    target: str = "cumulative_any",
) -> tuple[np.ndarray, np.ndarray]:
    """Average fixed-bin calibration points over repeated CV partitions."""
    if not repeated_probabilities:
        raise ValueError("Repeated calibration requires at least one prediction set.")
    edges = np.linspace(0, 1, bins + 1)
    expected: list[float] = []
    observed: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        repeat_expected = []
        repeat_observed = []
        for probabilities in repeated_probabilities:
            predicted, binary_target = _calibration_arrays(
                labels,
                probabilities,
                target,
            )
            mask = (predicted >= low) & (
                predicted < high if high < 1 else predicted <= high
            )
            if mask.sum() >= 10:
                repeat_expected.append(float(predicted[mask].mean()))
                repeat_observed.append(float(binary_target[mask].mean()))
        if repeat_expected:
            expected.append(float(np.mean(repeat_expected)))
            observed.append(float(np.mean(repeat_observed)))
    return np.asarray(expected), np.asarray(observed)


def _expected_prediction_cohort(prefix: str) -> str:
    if prefix.startswith("mimic4"):
        return "mimic4"
    if prefix.startswith("mimic3"):
        return "mimic3"
    if prefix.startswith("eicu"):
        return "eicu"
    raise ValueError(f"Cannot infer cohort from prediction prefix: {prefix}")


def _validate_prediction_alignment(
    data: dict[str, np.ndarray],
    prefix: str,
) -> None:
    cohort_name = _expected_prediction_cohort(prefix)
    cohort = pd.read_csv(PROCESSED / cohort_name / "cohort.csv")
    expected_ids = cohort["stay_id"].to_numpy(dtype=np.int64)
    if "stay_id" in data:
        stored_ids = data["stay_id"].astype(np.int64)
        if not np.array_equal(stored_ids, expected_ids):
            raise ValueError(
                f"{prefix}: prediction stay_id order does not match the "
                "current cohort"
            )
    if "labels" in data:
        expected_labels = cohort["outcome"].astype(int).to_numpy()
        if not np.array_equal(data["labels"].astype(int), expected_labels):
            raise ValueError(
                f"{prefix}: prediction labels do not match the current cohort"
            )


def load_predictions(prefix: str) -> dict[str, np.ndarray]:
    path = RESULTS / "predictions" / f"{prefix}.npz"
    with np.load(path, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    _validate_prediction_alignment(data, prefix)
    return data


def load_repeated_predictions(prefix: str) -> dict[str, list[np.ndarray]]:
    path = RESULTS / "predictions" / f"{prefix}_repeats.npz"
    with np.load(path, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    _validate_prediction_alignment(data, prefix)
    repeat_ids = sorted(
        {
            int(name.split("_", 1)[0].replace("repeat", ""))
            for name in data
            if name.startswith("repeat")
        }
    )
    models = sorted(
        {
            name.split("_", 1)[1]
            for name in data
            if name.startswith("repeat")
        }
    )
    return {
        model: [
            data[f"repeat{repeat}_{model}"].astype(float)
            for repeat in repeat_ids
            if f"repeat{repeat}_{model}" in data
        ]
        for model in models
    }


def create_calibration_figure() -> None:
    selected = ("xgb_summary", "gru", "dggru_tr")
    cohorts = (
        ("development", "mimic4_cv"),
        ("mimic3", "mimic3_external"),
        ("eicu", "eicu_external"),
    )
    fig, axes = plt.subplots(
        len(cohorts),
        len(CALIBRATION_TARGETS),
        figsize=(20.0, 10.8),
        sharex=True,
        sharey=True,
    )
    for row, (cohort, prefix) in enumerate(cohorts):
        data = load_predictions(prefix)
        labels = data["labels"].astype(int)
        repeated = (
            load_repeated_predictions(prefix)
            if cohort == "development"
            else None
        )
        for column, target_name in enumerate(CALIBRATION_TARGETS):
            axis = axes[row, column]
            for model in selected:
                if repeated is None:
                    expected, observed = calibration_points(
                        labels,
                        data[model],
                        target=target_name,
                    )
                else:
                    expected, observed = calibration_points_repeated(
                        labels,
                        repeated[model],
                        target=target_name,
                    )
                axis.plot(
                    expected,
                    observed,
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=COLORS[model],
                    label=MODEL_LABELS[model],
                )
            axis.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="#888888",
                linewidth=1,
            )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(CALIBRATION_TARGET_LABELS[target_name], fontsize=8)
            if column == 0:
                axis.set_ylabel(
                    f"{PLOT_LABELS[cohort]}\nObserved proportion",
                    fontsize=8,
                )
            if row == len(cohorts) - 1:
                axis.set_xlabel("Predicted probability", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(selected),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Figure 4. Calibration by ordinal grade and cumulative outcome",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, FIGURE4_CALIBRATION_STEM)


def net_benefit(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    target = (labels >= 1).astype(int)
    predicted = probabilities[:, 1:].sum(axis=1)
    n = len(target)
    values = []
    for threshold in thresholds:
        positive = predicted >= threshold
        tp = np.sum(positive & (target == 1))
        fp = np.sum(positive & (target == 0))
        values.append(tp / n - fp / n * threshold / (1.0 - threshold))
    return np.asarray(values)


def net_benefit_repeated(
    labels: np.ndarray,
    repeated_probabilities: list[np.ndarray],
    thresholds: np.ndarray,
) -> np.ndarray:
    return np.mean(
        [
            net_benefit(labels, probabilities, thresholds)
            for probabilities in repeated_probabilities
        ],
        axis=0,
    )


def create_dca_figure() -> None:
    thresholds = DCA_THRESHOLDS
    selected = ("xgb_summary", "gru", "dggru_tr")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, (cohort, prefix) in zip(
        axes,
        (("development", "mimic4_cv"), ("mimic3", "mimic3_external"), ("eicu", "eicu_external")),
    ):
        data = load_predictions(prefix)
        labels = data["labels"].astype(int)
        repeated = (
            load_repeated_predictions(prefix)
            if cohort == "development"
            else None
        )
        prevalence = np.mean(labels >= 1)
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        axis.plot(thresholds, treat_all, color="#999999", linestyle=":", label="Treat all")
        axis.axhline(0, color="#666666", linewidth=1, label="Treat none")
        for model in selected:
            values = (
                net_benefit_repeated(labels, repeated[model], thresholds)
                if repeated is not None
                else net_benefit(labels, data[model], thresholds)
            )
            axis.plot(
                thresholds,
                values,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_ylim(-0.08, max(0.75, prevalence + 0.05))
        axis.set_xlim(DCA_THRESHOLD_MIN, DCA_THRESHOLD_MAX)
        axis.set_title(PLOT_LABELS[cohort], fontsize=9)
        axis.set_xlabel("Risk threshold for any deterioration")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Net benefit")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Figure 4. Decision-curve analysis for any deterioration",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, FIGURE4_DCA_STEM)


def subgroup_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    replicates: int = 2000,
    seed: int = 20261024,
) -> tuple[float, float, float, int]:
    sub_labels = labels[mask]
    sub_probabilities = probabilities[mask]
    n = int(mask.sum())
    if n < 20 or len(np.unique(sub_labels)) < 2:
        return np.nan, np.nan, np.nan, n
    target = (sub_labels >= 1).astype(int)
    prediction = sub_probabilities[:, 1:].sum(axis=1)
    auc = roc_auc_score(target, prediction)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        index = rng.integers(0, n, n)
        if len(np.unique(target[index])) < 2:
            continue
        samples.append(roc_auc_score(target[index], prediction[index]))
    return auc, np.nanpercentile(samples, 2.5), np.nanpercentile(samples, 97.5), n


def subgroup_auc_repeated(
    labels: np.ndarray,
    repeated_probabilities: list[np.ndarray],
    mask: np.ndarray,
    replicates: int = 2000,
    seed: int = 20261024,
) -> tuple[float, float, float, int]:
    sub_labels = labels[mask]
    n = int(mask.sum())
    if n < 20 or len(np.unique(sub_labels)) < 2:
        return np.nan, np.nan, np.nan, n
    target = (sub_labels >= 1).astype(int)
    repeat_values = []
    repeat_predictions = []
    for probabilities in repeated_probabilities:
        prediction = probabilities[mask][:, 1:].sum(axis=1)
        repeat_values.append(float(roc_auc_score(target, prediction)))
        repeat_predictions.append(prediction)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        index = rng.integers(0, n, n)
        if len(np.unique(target[index])) < 2:
            continue
        values = [
            roc_auc_score(target[index], prediction[index])
            for prediction in repeat_predictions
        ]
        samples.append(float(np.mean(values)))
    return (
        float(np.mean(repeat_values)),
        float(np.nanpercentile(samples, 2.5)),
        float(np.nanpercentile(samples, 97.5)),
        n,
    )


def create_subgroup_table_and_figure() -> None:
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    data = load_predictions("mimic4_cv")
    repeated = load_repeated_predictions("mimic4_cv")
    labels = data["labels"].astype(int)
    strata = {
        "Age <65 years": cohort["age"] < 65,
        "Age >=65 years": cohort["age"] >= 65,
        "No mechanical ventilation in first 24 h": (
            cohort["mechanical_ventilation_24h"] < 0.5
        ),
        "Mechanical ventilation in first 24 h": (
            cohort["mechanical_ventilation_24h"] >= 0.5
        ),
        "No anticoagulation-related hemorrhage": (
            cohort["anticoagulation_related_hemorrhage"] < 0.5
        ),
        "Anticoagulation-related hemorrhage": (
            cohort["anticoagulation_related_hemorrhage"] >= 0.5
        ),
        "No recorded neurosurgical intervention": (
            cohort["neurosurgery_24h"] < 0.5
        ),
        "Recorded neurosurgical intervention": (
            cohort["neurosurgery_24h"] >= 0.5
        ),
        "No recorded sedative infusion in first 24 h": (
            cohort["sedation_exposure_24h"] < 0.5
        ),
        "Recorded sedative infusion in first 24 h": (
            cohort["sedation_exposure_24h"] >= 0.5
        ),
    }
    rows = []
    for label, mask in strata.items():
        mask_array = mask.fillna(False).to_numpy(dtype=bool)
        auc, low, high, n = subgroup_auc_repeated(
            labels, repeated["nested_stack"], mask_array
        )
        rows.append(
            {
                "Subgroup": label,
                "n": n,
                "Any deterioration AUC": auc,
                "95% CI low": low,
                "95% CI high": high,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table4_subgroups.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for i, row in table.iterrows():
        if not np.isfinite(row["Any deterioration AUC"]):
            continue
        axis.errorbar(
            row["Any deterioration AUC"],
            i,
            xerr=[
                [row["Any deterioration AUC"] - row["95% CI low"]],
                [row["95% CI high"] - row["Any deterioration AUC"]],
            ],
            fmt="o",
            color="#0072B2",
            capsize=3,
        )
    axis.set_yticks(np.arange(len(table)))
    axis.set_yticklabels(
        [f"{row['Subgroup']} (n={int(row['n'])})" for _, row in table.iterrows()],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.35, 0.9)
    axis.set_xlabel("Any-deterioration AUC")
    axis.set_title(
        "Figure 6B. Development-cohort subgroup performance",
        fontsize=11,
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURE6_SUBGROUP_STEM)


def bootstrap_auc_difference(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    groups: np.ndarray,
    kind: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    def metric(
        subset_labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        if kind == "macro":
            if len(np.unique(subset_labels)) < 3:
                return np.nan
            return float(
                np.mean(
                    [
                        roc_auc_score(
                            (subset_labels == grade).astype(int),
                            probabilities[:, grade],
                        )
                        for grade in range(3)
                    ]
                )
            )
        threshold = 1 if kind == "any" else 2
        target = (subset_labels >= threshold).astype(int)
        if target.min() == target.max():
            return np.nan
        prediction = probabilities[:, threshold:].sum(axis=1)
        return float(roc_auc_score(target, prediction))

    point = metric(labels, primary) - metric(labels, comparator)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    differences = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([group_indices[group] for group in sampled])
        subset_labels = labels[index]
        value = metric(subset_labels, primary[index]) - metric(
            subset_labels, comparator[index]
        )
        if np.isfinite(value):
            differences.append(value)
    return (
        point,
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def bootstrap_auc_difference_repeated(
    labels: np.ndarray,
    primary: list[np.ndarray],
    comparator: list[np.ndarray],
    groups: np.ndarray,
    kind: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    def metric(
        subset_labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        if kind == "macro":
            if len(np.unique(subset_labels)) < 3:
                return np.nan
            return float(
                np.mean(
                    [
                        roc_auc_score(
                            (subset_labels == grade).astype(int),
                            probabilities[:, grade],
                        )
                        for grade in range(3)
                    ]
                )
            )
        threshold = 1 if kind == "any" else 2
        target = (subset_labels >= threshold).astype(int)
        if target.min() == target.max():
            return np.nan
        return float(
            roc_auc_score(
                target,
                probabilities[:, threshold:].sum(axis=1),
            )
        )

    point = float(
        np.mean(
            [
                metric(labels, primary_repeat)
                - metric(labels, comparator_repeat)
                for primary_repeat, comparator_repeat in zip(primary, comparator)
            ]
        )
    )
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    differences = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([group_indices[group] for group in sampled])
        subset_labels = labels[index]
        repeat_differences = [
            metric(subset_labels, primary_repeat[index])
            - metric(subset_labels, comparator_repeat[index])
            for primary_repeat, comparator_repeat in zip(primary, comparator)
        ]
        finite = [value for value in repeat_differences if np.isfinite(value)]
        if finite:
            differences.append(float(np.mean(finite)))
    return (
        point,
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def create_paired_comparison_table() -> None:
    rows = []
    cohorts = (
        ("development", "mimic4_cv"),
        ("mimic3", "mimic3_external"),
        ("eicu", "eicu_external"),
    )
    comparators = (
        "dggru",
        "dggru_laplace",
        "dggru_tr_treverse",
        "dggru_tr_tpermute",
        "xgb_summary",
    )
    for cohort, prefix in cohorts:
        data = load_predictions(prefix)
        labels = data["labels"].astype(int)
        repeated = (
            load_repeated_predictions(prefix)
            if cohort == "development"
            else None
        )
        if cohort == "eicu":
            frame = pd.read_csv(PROCESSED / "eicu" / "cohort.csv")
            groups = frame["hospital_key"].astype(str).to_numpy()
        else:
            groups = data["subject_key"].astype(str)
        for comparator in comparators:
            for kind in ("macro", "any", "severe"):
                if repeated is None:
                    point, low, high = bootstrap_auc_difference(
                        labels,
                        data["dggru_tr"],
                        data[comparator],
                        groups,
                        kind=kind,
                        replicates=2000,
                        seed=20261024,
                    )
                else:
                    point, low, high = bootstrap_auc_difference_repeated(
                        labels,
                        repeated["dggru_tr"],
                        repeated[comparator],
                        groups,
                        kind=kind,
                        replicates=2000,
                        seed=20261024,
                    )
                rows.append(
                    {
                        "Cohort": PLOT_LABELS[cohort],
                        "Comparison": (
                            "temporal regularization - "
                            + MODEL_LABELS[comparator]
                        ),
                        "Metric": kind,
                        "Difference": point,
                        "95% CI low": low,
                        "95% CI high": high,
                    }
                )
                print(
                    f"[paired] {cohort} {comparator} {kind}: "
                    f"{point:.4f} ({low:.4f}, {high:.4f})",
                    flush=True,
                )
    table = pd.DataFrame(rows)
    table.to_csv(
        TABLES / "table6_paired_auc_differences.csv", index=False
    )
    (TABLES / "table6.md").write_text(
        "# Table 6. Paired AUC differences: DG-GRU + TR minus comparator\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    ensure_analysis_dirs()
    create_cohort_flow_table_and_figure()
    create_table1()
    create_missingness_table()
    create_measurement_audit_table()
    create_performance_tables()
    create_grade_calibration_table()
    create_lambda_table()
    create_sensitivity_table()
    create_model_architecture_figure()
    create_model_performance_figure()
    create_calibration_figure()
    create_dca_figure()
    create_subgroup_table_and_figure()
    create_paired_comparison_table()
    print("Saved tables to results/tables and figures to results/figures")


if __name__ == "__main__":
    main()
