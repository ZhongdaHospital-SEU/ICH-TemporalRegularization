"""Build the manuscript, tables, figure package, and rendered documents."""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import (
    WD_ALIGN_PARAGRAPH,
    WD_BREAK,
    WD_COLOR_INDEX,
    WD_LINE_SPACING,
)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from .common import CHANNELS, OUTPUTS, PROCESSED, RESULTS, load_config


TITLE = (
    "Temporal Regularization of a Dynamic Graph Network for Ordinal "
    "Prediction of Early Neurological Deterioration After Intracerebral "
    "Hemorrhage in Intensive Care"
)
RUNNING_TITLE = "Temporal regularization for ordinal ICH deterioration"
TITLE_WITHOUT_GAIN = (
    "A Dynamic Graph Network for Ordinal Prediction of Early Neurological "
    "Deterioration After Intracerebral Hemorrhage in Intensive Care"
)
RUNNING_TITLE_WITHOUT_GAIN = "Dynamic graph ordinal prediction for ICH deterioration"

COHORTS = ("mimic4", "mimic3", "eicu")
DATABASE_SPANS = {
    "mimic4": "2008-2022",
    "mimic3": "2001-2012",
    "eicu": "2014-2015",
}
DATABASE_GCS_DOCUMENTATION = {
    "mimic4": "Itemized GCS components in MetaVision; total reconstructed",
    "mimic3": "Itemized components and CareVue total-GCS records",
    "eicu": "Nursing-recorded GCS entries; itemized/total harmonized",
}
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
MODEL_LABELS = {
    "xgb_static": "Static XGBoost",
    "xgb_summary": "Summary XGBoost",
    "gcs_slope": "GCS temporal slope only",
    "gru": "SeqOrdinal GRU",
    "grud": "GRU-D",
    "transformer": "Transformer",
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU + TR",
    "nested_stack": "Nested logistic stacking",
    "dggru_tr_treverse": "DG-GRU + TR, time reversed",
    "dggru_tr_tpermute": "DG-GRU + TR, time permuted",
    "dggru_laplace": (
        "DG-GRU + generic Laplace input noise (scale=0.01)"
    ),
    "dggru_tr_gcsmask": "DG-GRU + TR, GCS-observation mask",
    "dggru_tr_noseq": "DG-GRU + TR, sequence ablated",
    "dggru_tr_static": "DG-GRU + TR, non-summary static inputs",
    "dggru_tr_traj": "DG-GRU + TR, trajectory only",
}
OUTLINE_BASELINE_MODELS = (
    "xgb_static",
    "xgb_summary",
    "gru",
    "grud",
    "transformer",
)
OUTLINE_MODEL_ORDER = OUTLINE_BASELINE_MODELS + (
    "gcs_slope",
    "dggru",
    "dggru_tr",
    "nested_stack",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)
OUTLINE_REQUIRED_MODELS = frozenset(OUTLINE_MODEL_ORDER)
OUTLINE_CONTROL_MODELS = (
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)
SECONDARY_ABLATION_MODELS = (
    "dggru_tr_gcsmask",
    "dggru_tr_noseq",
    "dggru_tr_static",
    "dggru_tr_traj",
)
TABLE2_MODEL_LABEL_ALIASES = {
    "dggru_tr": {
        "DG-GRU + TR",
        "DG-GRU + temporal risk-trajectory regularization",
    },
}
OUTLINE_SENSITIVITY_ANALYSES = (
    {
        "analysis": "1. Baseline GCS definition",
        "dataset": "mimic4",
        "variants": (
            ("s1_baseline_max", "Maximum pre-landmark GCS"),
            ("s1_baseline_last", "Final pre-landmark GCS"),
        ),
    },
    {
        "analysis": "2. Persistence threshold",
        "dataset": "mimic4",
        "variants": (
            ("s2_persistence_1h", "At least 1 h"),
            ("s2_persistence_2h", "At least 2 h"),
        ),
    },
    {
        "analysis": "3. Sedation handling",
        "dataset": "mimic4",
        "variants": (
            ("s3_exclude_continuous_sedation", "Exclude continuous sedation windows"),
            ("s3_unsedated_only", "Restrict to observed unsedated subgroup"),
        ),
    },
    {
        "analysis": "4. Early ICU transfer censoring",
        "dataset": "mimic4",
        "variants": (
            ("s4_censor_exclude", "Primary exclusion"),
            ("s4_censor_grade0", "Code early transfer as grade 0"),
            ("s4_censor_ipw", "Inverse-probability weighting"),
        ),
    },
    {
        "analysis": "5. Unplanned intubation",
        "dataset": "mimic4",
        "variants": (
            ("s5_with_intubation", "Included in grade 2"),
            ("s5_without_intubation", "Excluded from endpoint"),
        ),
    },
    {
        "analysis": "6. Landmark window",
        "dataset": "mimic4",
        "variants": (
            ("s6_landmark_6h", "6 h landmark"),
            ("s6_landmark_24h", "24 h landmark"),
            ("s6_landmark_48h", "48 h landmark"),
        ),
    },
    {
        "analysis": "7. Death exclusion",
        "dataset": "mimic4",
        "variants": (("s7_exclude_death72h", "Exclude deaths within 72 h"),),
    },
    {
        "analysis": "8. eICU GCS and RASS",
        "dataset": "eicu",
        "variants": (
            ("s8_eicu_gcs_unadjusted", "Recorded GCS without RASS adjustment"),
            ("s8_eicu_gcs_rass_adjusted", "Explicit RASS-adjusted GCS"),
        ),
    },
)
OUTLINE_SENSITIVITY_SCENARIOS = tuple(
    scenario
    for analysis in OUTLINE_SENSITIVITY_ANALYSES
    for scenario, _ in analysis["variants"]
)
OUTLINE_SUBGROUP_FACTORS = (
    "Age",
    "Mechanical ventilation",
    "Anticoagulation-related hemorrhage",
    "Neurosurgical intervention",
    "Sedation exposure",
)
FIGURE_SPECS = (
    {
        "number": 1,
        "panel": "",
        "stem": "figure1_cohort_flow",
        "caption": (
            "Cohort construction (A) and ordinal outcome distribution (B) "
            "across MIMIC-IV, MIMIC-III, and eICU-CRD."
        ),
    },
    {
        "number": 2,
        "panel": "",
        "stem": "figure2_model_architecture",
        "caption": (
            "Temporal-regularized dynamic graph architecture. All prediction "
            "inputs precede the ICU hour-24 landmark."
        ),
    },
    {
        "number": 3,
        "panel": "",
        "stem": "figure3_model_performance",
        "caption": (
            "Development discrimination (A), development cumulative "
            "calibration (B), and external transport discrimination (C). "
            "Points are point estimates and horizontal lines are "
            "2,000-replicate cluster-bootstrap 95% confidence intervals."
        ),
    },
    {
        "number": 4,
        "panel": "A",
        "stem": "figure4_calibration",
        "caption": (
            "Calibration by ordinal grade and for the cumulative probability "
            "of any deterioration. The dashed diagonal is perfect calibration."
        ),
    },
    {
        "number": 4,
        "panel": "B",
        "stem": "figure4_decision_curve",
        "caption": (
            "Decision-curve analysis for any deterioration across the "
            "fixed 10%-40% threshold range. Treat-none and treat-all "
            "strategies are shown for context."
        ),
    },
    {
        "number": 5,
        "panel": "",
        "stem": "figure5_temporal_structure",
        "caption": (
            "Temporal structure and regularization effects. Macro AUC is shown "
            "for the 6-, 24-, and 48-hour landmark models using the same 92 "
            "predictors available at the 6-hour landmark. Each landmark uses "
            "landmark-specific eligibility reconstructed from the same first "
            "ICU episode, with no 24-hour survivor restriction in the 6-hour "
            "analysis. "
            "Latent risk trajectories and reversal summaries compare the "
            "primary model with time-reversed, time-permuted, and generic "
            "Laplace input-noise controls at the ICU hour-24 landmark."
        ),
    },
    {
        "number": 6,
        "panel": "A",
        "stem": "figure6_interpretation",
        "caption": (
            "Model interpretation. The left panel ranks the top 15 Summary "
            "XGBoost predictors by mean absolute TreeSHAP value; the right "
            "panel shows DG-GRU + TR time-dependent importance across ICU "
            "hours 0-23."
        ),
    },
    {
        "number": 6,
        "panel": "B",
        "stem": "figure6_subgroups",
        "caption": (
            "Development-cohort subgroup discrimination for nested stacking. "
            "Error bars are exploratory nonparametric bootstrap 95% confidence "
            "intervals for the five outline subgroup factors."
        ),
    },
)
FIGURES = {spec["stem"]: spec["caption"] for spec in FIGURE_SPECS}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_ci(block: dict, model: str, metric: str, digits: int = 3) -> str:
    point = float(block[model]["point"][metric])
    interval = block[model]["ci"].get(metric)
    if not interval or len(interval) != 2:
        return fmt(point, digits)
    return f"{point:.{digits}f} ({interval[0]:.{digits}f}-{interval[1]:.{digits}f})"


def median_iqr(series: pd.Series, digits: int = 1) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "NA"
    low, high = np.percentile(values, [25, 75])
    return f"{values.median():.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def n_percent(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce")
    denominator = int(values.notna().sum())
    if denominator == 0:
        return "NA"
    numerator = int(values.fillna(0).astype(bool).sum())
    percentage = 100.0 * numerator / denominator
    return f"{numerator} ({percentage:.1f}%)"


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def seed_count(manifest: dict) -> int:
    values = manifest.get("model_seeds", [])
    if isinstance(values, (list, tuple, set)):
        return len(values)
    if isinstance(values, int):
        return values
    raise ValueError("The run manifest does not define a valid model_seeds value.")


def feature_count(manifest: dict) -> int:
    """Return the size of the static set the primary models actually used.

    ``features`` lists every eligible static predictor in the development
    matrix. The primary models and summary XGBoost receive the pre-specified
    core subset, so the manuscript must quote ``temporal_static_feature_count``
    rather than the size of the screened matrix.
    """
    count = manifest.get("temporal_static_feature_count")
    if isinstance(count, int) and count > 0:
        return count
    features = manifest.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("The run manifest contains no eligible static predictors.")
    return len(features)


def eligible_feature_count(manifest: dict) -> int:
    features = manifest.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("The run manifest contains no eligible static predictors.")
    return len(features)


def dynamic_summary_count() -> int:
    """Number of 24-hour trajectory summaries excluded from the static arm."""
    return 1 + 6 * len(CHANNELS)


def configured_channels(config: dict) -> tuple[str, ...]:
    channels = tuple(config.get("channels", []))
    if len(channels) != 14 or len(set(channels)) != 14:
        raise ValueError(
            "outline.docx requires 14 unique configured time-series channels; "
            f"configs/study.json contains {len(channels)} entries and "
            f"{len(set(channels))} unique names."
        )
    return channels


def shared_static_sentence(n_static: int) -> str:
    return (
        "Summary XGBoost and the temporal neural models both used the same "
        f"{n_static} pre-specified core first-24-hour static predictors; the neural "
        "models also received the hourly value-and-mask sequence."
    )


def shared_static_discussion_sentence(n_static: int) -> str:
    return (
        "Both representation families received the same "
        f"{n_static}-variable core static matrix; cross-family differences "
        "therefore reflect representation and optimization rather than "
        "unequal static-feature access."
    )


def metric_block(metrics: dict, cohort: str) -> dict:
    return (
        metrics["development"]
        if cohort == "development"
        else metrics["external"][cohort]
    )


def title_context(metrics: dict) -> dict[str, object]:
    """Choose title and abstract emphasis from the paired development effect."""
    development = metric_block(metrics, "development")
    interval = paired_interval(development, "dggru")
    gain_supported = (
        all(np.isfinite(interval))
        and float(interval[0]) > 0.0
    )
    if gain_supported:
        return {
            "title": TITLE,
            "running_title": RUNNING_TITLE,
            "keywords": (
                "intracerebral hemorrhage; early neurological deterioration; "
                "ordinal prediction; dynamic graph neural network; temporal "
                "regularization; external validation"
            ),
            "temporal_regularization_primary": True,
        }
    return {
        "title": TITLE_WITHOUT_GAIN,
        "running_title": RUNNING_TITLE_WITHOUT_GAIN,
        "keywords": (
            "intracerebral hemorrhage; early neurological deterioration; "
            "ordinal prediction; dynamic graph neural network; external "
            "validation"
        ),
        "temporal_regularization_primary": False,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".3f")


def table1_frame() -> pd.DataFrame:
    frames = {
        cohort: pd.read_csv(PROCESSED / cohort / "cohort.csv")
        for cohort in COHORTS
    }
    rows = [
        {
            "Characteristic": "Database version",
            **{
                COHORT_LABELS["mimic4"]: "3.1",
                COHORT_LABELS["mimic3"]: "1.4",
                COHORT_LABELS["eicu"]: "2.0",
            },
        },
        {
            "Characteristic": "Database coverage period",
            **{
                COHORT_LABELS[cohort]: DATABASE_SPANS[cohort]
                for cohort in COHORTS
            },
        },
        {
            "Characteristic": "GCS documentation",
            **{
                COHORT_LABELS[cohort]: DATABASE_GCS_DOCUMENTATION[cohort]
                for cohort in COHORTS
            },
        },
        {
            "Characteristic": "P020 endpoint status",
            **{
                COHORT_LABELS[cohort]: (
                    "Partial: GCS reduction or death; verified unplanned "
                    "intubation unavailable"
                )
                for cohort in COHORTS
            },
        },
    ]
    variables = [
        ("Patients, n", lambda x: f"{len(x):,}"),
        ("Age, years", lambda x: median_iqr(x["age"])),
        ("Male sex", lambda x: n_percent(x["sex_male"])),
        ("White race", lambda x: n_percent(x["race_white"])),
        ("Emergency admission", lambda x: n_percent(x["emergency_admission"])),
        ("Maximum GCS in hours 0-24", lambda x: median_iqr(x["baseline_gcs_max"])),
        (
            "Hours with observed total GCS in 0-24 h",
            lambda x: median_iqr(x["gcs_total_n"]),
        ),
        (
            "Observed total-GCS hours 0-5",
            lambda x: n_percent(x["gcs_total_n"].le(5)),
        ),
        (
            "Observed total-GCS hours 6-11",
            lambda x: n_percent(x["gcs_total_n"].between(6, 11)),
        ),
        (
            "Observed total-GCS hours 12-17",
            lambda x: n_percent(x["gcs_total_n"].between(12, 17)),
        ),
        (
            "Observed total-GCS hours 18-23",
            lambda x: n_percent(x["gcs_total_n"].between(18, 23)),
        ),
        (
            "Observed total-GCS hours 24",
            lambda x: n_percent(x["gcs_total_n"].eq(24)),
        ),
        (
            "Any missing 14-channel summary",
            lambda x: n_percent(
                x[
                    [
                        f"{channel}_mean"
                        for channel in CHANNELS
                        if f"{channel}_mean" in x
                    ]
                ]
                .isna()
                .any(axis=1)
            ),
        ),
        ("Missing age", lambda x: n_percent(x["age"].isna())),
        ("ICU length of stay, hours", lambda x: median_iqr(x["icu_out_hour"])),
        ("Grade 0", lambda x: f"{int((x['outcome'] == 0).sum()):,}"),
        ("Grade 1", lambda x: f"{int((x['outcome'] == 1).sum()):,}"),
        ("Grade 2", lambda x: f"{int((x['outcome'] == 2).sum()):,}"),
        ("Death by hour 72", lambda x: n_percent(x["death_within_72h"])),
    ]
    for label, function in variables:
        row = {"Characteristic": label}
        for cohort, frame in frames.items():
            row[COHORT_LABELS[cohort]] = function(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def table1_outcome_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    rows = []
    variables = [
        ("Patients, n", lambda x: f"{len(x):,}"),
        ("Age, years", lambda x: median_iqr(x["age"])),
        ("Male sex", lambda x: n_percent(x["sex_male"])),
        ("Maximum GCS in hours 0-24", lambda x: median_iqr(x["baseline_gcs_max"])),
        ("Last GCS before landmark", lambda x: median_iqr(x["baseline_gcs_last"])),
        (
            "GCS measurements in hours 0-24",
            lambda x: median_iqr(x["gcs_total_n"]),
        ),
        (
            "Observed deep sedation (RASS <= -3) in first 24 h",
            lambda x: n_percent(x["any_deep_sedation_24h"]),
        ),
        ("Death by hour 72", lambda x: n_percent(x["death_within_72h"])),
    ]
    for label, function in variables:
        row = {"Characteristic": label}
        for grade in (0, 1, 2):
            row[f"Grade {grade}"] = function(frame[frame["outcome"] == grade])
        rows.append(row)
    return pd.DataFrame(rows)


def table2_outcome_frame() -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(PROCESSED / cohort / "cohort.csv")
        outcomes = pd.to_numeric(frame["outcome"], errors="coerce")
        n = int(outcomes.notna().sum())
        if n == 0:
            raise ValueError(f"{cohort} has no evaluable ordinal outcomes.")
        row = {
            "Cohort": COHORT_LABELS[cohort],
            "Patients, n": n,
        }
        for grade in (0, 1, 2):
            count = int(outcomes.eq(grade).sum())
            row[f"Grade {grade}, n (%)"] = f"{count} ({100 * count / n:.1f}%)"
        any_count = int(outcomes.ge(1).sum())
        severe_count = int(outcomes.eq(2).sum())
        row["Any deterioration, n (%)"] = (
            f"{any_count} ({100 * any_count / n:.1f}%)"
        )
        row["Severe deterioration, n (%)"] = (
            f"{severe_count} ({100 * severe_count / n:.1f}%)"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def performance_frame(
    metrics: dict,
    models: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if models is None:
        models = tuple(
            name
            for name, block in metric_block(metrics, "development").items()
            if isinstance(block, dict) and isinstance(block.get("point"), dict)
        )
    rows = []
    for cohort in ("development", "mimic3", "eicu"):
        block = metric_block(metrics, cohort)
        for model in models:
            point = block[model]["point"]
            rows.append(
                {
                    "Cohort": PLOT_LABELS[cohort],
                    "Model": model_label(model),
                    "n": int(point["n"]),
                    "Macro AUC (95% CI)": fmt_ci(block, model, "macro_auc"),
                    "Any AUC (95% CI)": fmt_ci(
                        block, model, "auc_any_deterioration"
                    ),
                    "Severe AUC (95% CI)": fmt_ci(
                        block, model, "auc_severe_deterioration"
                    ),
                    "Brier": fmt(point["multiclass_brier"]),
                    "Slope": fmt(point["calibration_slope_any"]),
                    "ECE": fmt(point["expected_calibration_error"]),
                }
            )
    return pd.DataFrame(rows)


def lambda_frame() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(RESULTS.glob("metrics_lambda_*.json")):
        tag = metrics_path.stem.removeprefix("metrics_")
        manifest_path = RESULTS / f"run_manifest_{tag}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing lambda-grid manifest for {metrics_path.name}: "
                f"{manifest_path}"
            )
        manifest = load_json(manifest_path)
        if (
            "lambda_monotone" not in manifest
            or "lambda_smooth" not in manifest
        ):
            raise ValueError(
                f"{manifest_path.name} does not record both lambda weights."
            )
        block = load_json(metrics_path)["development"]
        models = ordered_model_names(block, manifest)
        model = (
            "dggru_tr"
            if "dggru_tr" in models
            else models[0]
        )
        point = block[model]["point"]
        rows.append(
            {
                "lambda monotone": float(manifest["lambda_monotone"]),
                "lambda smooth": float(manifest["lambda_smooth"]),
                "Macro AUC": point["macro_auc"],
                "Any AUC": point["auc_any_deterioration"],
                "Severe AUC": point["auc_severe_deterioration"],
                "Brier": point["multiclass_brier"],
                "Slope": point["calibration_slope_any"],
                "ECE": point["expected_calibration_error"],
            }
        )
    columns = [
        "lambda monotone",
        "lambda smooth",
        "Macro AUC",
        "Any AUC",
        "Severe AUC",
        "Brier",
        "Slope",
        "ECE",
    ]
    if rows:
        frame = pd.DataFrame(rows)
        return (
            frame.drop_duplicates(
                subset=["lambda monotone", "lambda smooth"],
                keep="last",
            )
            .sort_values(["lambda monotone", "lambda smooth"])
            .reset_index(drop=True)
        )

    manifest_path = RESULTS / "run_manifest.json"
    manifest = load_json(manifest_path)
    selection = manifest.get("cv_selected_lambdas", {}).get("dggru_tr", {})
    raw_values = selection.get("independent_values") or selection.get("values", [])
    values = [
        float(value)
        for value in raw_values
        if np.isfinite(float(value))
    ]
    if not values:
        raise FileNotFoundError(
            "No lambda-grid metrics or development lambda-selection records "
            "were available."
        )
    ratio = float(manifest.get("lambda_smooth_ratio", 0.4))
    counts: dict[tuple[float, float], int] = {}
    for value in values:
        key = (float(value), float(value) * ratio)
        counts[key] = counts.get(key, 0) + 1
    return pd.DataFrame(
        [
            {
                "lambda monotone": key[0],
                "lambda smooth": key[1],
                "Selected fit records": count,
                "Selection proportion": count / len(values),
            }
            for key, count in sorted(counts.items())
        ]
    )


def sensitivity_frame() -> pd.DataFrame:
    """Render all eight outline analyses without dropping unsupported variants."""
    records: dict[tuple[str, str], dict] = {}
    for dataset in ("mimic4", "eicu"):
        audit_path = RESULTS / "sensitivity" / dataset / "scenario_audit.json"
        if not audit_path.exists():
            raise FileNotFoundError(
                "The outline sensitivity audit is missing: "
                f"{audit_path}. Run all eight analyses before reporting."
            )
        payload = load_json(audit_path)
        if not isinstance(payload, list):
            raise ValueError(f"{audit_path} must contain a JSON list.")
        for record in payload:
            scenario = str(record.get("scenario", {}).get("scenario_id", ""))
            if scenario:
                records[(dataset, scenario)] = record

    rows = []
    for analysis in OUTLINE_SENSITIVITY_ANALYSES:
        dataset = str(analysis["dataset"])
        for scenario, variant_label in analysis["variants"]:
            record = records.get((dataset, scenario))
            if record is None:
                raise FileNotFoundError(
                    "The sensitivity audit does not include the prescribed "
                    f"variant {dataset}/{scenario}."
                )
            status = record.get("status", {})
            support = str(status.get("support", "unknown"))
            model_run_allowed = bool(status.get("model_run_allowed", False))
            outcomes = record.get("outcomes") or {}
            row = {
                "Outline analysis": analysis["analysis"],
                "Variant": variant_label,
                "Dataset": "MIMIC-IV" if dataset == "mimic4" else "eICU-CRD",
                "Support": support,
                "n": record.get("n"),
                "Grade 0/1/2": (
                    f"{int(outcomes.get('0', 0))}/"
                    f"{int(outcomes.get('1', 0))}/"
                    f"{int(outcomes.get('2', 0))}"
                    if outcomes
                    else "NA"
                ),
                "_dataset": dataset,
                "_scenario": scenario,
            }
            if not model_run_allowed:
                row["Note"] = str(status.get("reason", "Not model-runnable."))
                rows.append(row)
                continue

            metrics_path = RESULTS / "sensitivity" / dataset / f"{scenario}.json"
            prediction_path = (
                RESULTS / "sensitivity" / dataset / f"{scenario}_predictions.npz"
            )
            if not metrics_path.exists() or not prediction_path.exists():
                raise FileNotFoundError(
                    f"The audit marks {dataset}/{scenario} runnable, but its "
                    "metrics or predictions are missing."
                )
            block = load_json(metrics_path)
            scenario_block = block.get("development", block)
            with np.load(prediction_path, allow_pickle=True) as data:
                labels = data["labels"].astype(int)
            row["n"] = int(len(labels))
            row["Grade 0/1/2"] = (
                f"{int((labels == 0).sum())}/"
                f"{int((labels == 1).sum())}/"
                f"{int((labels == 2).sum())}"
            )
            n_labels = len(labels)
            any_count = int((labels >= 1).sum())
            severe_count = int((labels == 2).sum())
            row["Any deterioration, n (%)"] = (
                f"{any_count} ({100 * any_count / n_labels:.1f}%)"
            )
            row["Severe deterioration, n (%)"] = (
                f"{severe_count} ({100 * severe_count / n_labels:.1f}%)"
            )
            row["_any_rate"] = any_count / n_labels
            row["_severe_rate"] = severe_count / n_labels
            required_models = (
                (
                    "dggru",
                    "dggru_tr",
                    "dggru_tr_treverse",
                    "dggru_tr_tpermute",
                    "dggru_laplace",
                )
                if scenario.startswith("s6_landmark_")
                else ("xgb_summary", "gru", "dggru_tr")
            )
            for model in required_models:
                if model not in scenario_block:
                    raise ValueError(
                        f"{scenario} does not contain the prescribed model {model}."
                    )
                point = scenario_block[model]["point"]
                label_model = {
                    "xgb_summary": "Summary XGBoost",
                    "gru": "SeqOrdinal GRU",
                    "dggru_tr": "DG-GRU + TR",
                    "dggru": "DG-GRU",
                    "dggru_tr_treverse": "DG-GRU + TR time reversed",
                    "dggru_tr_tpermute": "DG-GRU + TR time permuted",
                    "dggru_laplace": "Generic Laplace input-noise control",
                }[model]
                row[f"{label_model} macro AUC"] = point["macro_auc"]
                row[f"{label_model} any AUC"] = point[
                    "auc_any_deterioration"
                ]
                if model == "dggru_tr":
                    row["_dggru_tr_macro_auc"] = point["macro_auc"]
            row["Note"] = ""
            rows.append(row)
    if len(rows) != len(OUTLINE_SENSITIVITY_SCENARIOS):
        raise ValueError(
            "The rendered sensitivity table does not cover all protocol-defined "
            f"variants ({len(rows)} of {len(OUTLINE_SENSITIVITY_SCENARIOS)})."
        )
    by_scenario = {
        (row["_dataset"], row["_scenario"]): row
        for row in rows
    }
    reference_scenarios = {
        "mimic4": "s1_baseline_max",
        "eicu": "s8_eicu_gcs_unadjusted",
    }
    for dataset, reference_scenario in reference_scenarios.items():
        reference = by_scenario.get((dataset, reference_scenario))
        if reference is None or not np.isfinite(
            float(reference.get("_any_rate", np.nan))
        ):
            raise ValueError(
                f"{dataset} lacks its primary reference "
                f"{reference_scenario} needed for event-rate and "
                "discrimination deltas."
            )
        if not np.isfinite(
            float(reference.get("_dggru_tr_macro_auc", np.nan))
        ):
            raise ValueError(
                f"{dataset} primary reference lacks a runnable DG-GRU + TR "
                "macro AUC."
            )
        reference_any = float(reference["_any_rate"])
        reference_severe = float(reference["_severe_rate"])
        reference_auc = float(reference["_dggru_tr_macro_auc"])
        for row in rows:
            if row["_dataset"] != dataset:
                continue
            any_rate = row.get("_any_rate")
            severe_rate = row.get("_severe_rate")
            macro_auc = row.get("_dggru_tr_macro_auc")
            if any_rate is not None and severe_rate is not None:
                row["Change in any event rate vs primary (pp)"] = (
                    100.0 * (float(any_rate) - reference_any)
                )
                row["Change in severe event rate vs primary (pp)"] = (
                    100.0 * (float(severe_rate) - reference_severe)
                )
            if macro_auc is not None:
                row["Change in DG-GRU + TR macro AUC vs primary"] = (
                    float(macro_auc) - reference_auc
                )
    frame = pd.DataFrame(rows)
    return frame.drop(
        columns=[
            column
            for column in frame.columns
            if column.startswith("_")
        ]
    )


def trajectory_frame() -> pd.DataFrame:
    path = RESULTS / "tables" / "table5_trajectory_regularization.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Trajectory-regularization table is missing. Run "
            "export_trajectories before make_manuscript."
        )
    frame = pd.read_csv(path)
    if "model" not in frame.columns:
        raise ValueError("The trajectory table is missing the model column.")
    required_models = {
        "dggru",
        "dggru_tr",
        *OUTLINE_CONTROL_MODELS,
    }
    observed_models = set(frame["model"].dropna().astype(str))
    if observed_models != required_models:
        missing = required_models.difference(observed_models)
        unexpected = observed_models.difference(required_models)
        problems = []
        if missing:
            problems.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            problems.append(
                "unexpected or obsolete " + ", ".join(sorted(unexpected))
            )
        raise ValueError(
            "The trajectory table does not match the outline control set: "
            + "; ".join(problems)
        )
    if frame["model"].duplicated().any():
        raise ValueError(
            "The trajectory table contains duplicate control-model rows."
        )
    frame["Model"] = frame["model"].map(
        {
            "dggru": "DG-GRU",
            "dggru_tr": "DG-GRU + TR",
            "dggru_tr_treverse": "Time reversed",
            "dggru_tr_tpermute": "Time permuted",
            "dggru_laplace": "Generic Laplace input noise",
        }
    )
    if frame["Model"].isna().any():
        stale = sorted(
            set(frame.loc[frame["Model"].isna(), "model"].dropna().astype(str))
        )
        raise ValueError(
            "The trajectory table contains obsolete or unknown model "
            f"identifiers: {', '.join(stale)}"
        )
    frame = frame.rename(
        columns={
            "negative_step_fraction_median": "Decreasing-step fraction, median",
            "roughness_median": "Roughness, median",
            "monotone_violation_median": "Monotone violation, median",
            "total_variation_median": "Total variation, median",
            "final_risk_auc_any": "Final-risk any-deterioration AUC",
        }
    )
    return frame[
        [
            "Model",
            "Decreasing-step fraction, median",
            "Roughness, median",
            "Monotone violation, median",
            "Total variation, median",
            "Final-risk any-deterioration AUC",
        ]
    ]


def subgroup_frame() -> pd.DataFrame:
    path = RESULTS / "tables" / "table4_subgroups.csv"
    if not path.exists():
        raise FileNotFoundError(
            "The outline-required subgroup table is missing. Run "
            "make_tables_figures before make_manuscript: "
            f"{path}"
        )
    frame = pd.read_csv(path)
    required = {
        "Subgroup",
        "n",
        "Any deterioration AUC",
        "95% CI low",
        "95% CI high",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The subgroup table is missing columns: "
            + ", ".join(sorted(missing))
        )
    labels = " ".join(frame["Subgroup"].astype(str))
    absent = [
        factor
        for factor in OUTLINE_SUBGROUP_FACTORS
        if factor.lower() not in labels.lower()
    ]
    if absent:
        raise ValueError(
            "The subgroup table does not cover all five outline factors: "
            + ", ".join(absent)
        )
    return frame


def grade_calibration_frame() -> pd.DataFrame:
    path = RESULTS / "tables" / "table_s12_grade_calibration.csv"
    if not path.exists():
        raise FileNotFoundError(
            "The grade-specific calibration table is missing. Run "
            f"make_tables_figures before make_manuscript: {path}"
        )
    frame = pd.read_csv(path)
    required = {
        "Cohort",
        "Model",
        "n",
        "grade_0 slope",
        "grade_0 intercept",
        "grade_1 slope",
        "grade_1 intercept",
        "grade_2 slope",
        "grade_2 intercept",
        "any slope",
        "any intercept",
        "severe slope",
        "severe intercept",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The grade-specific calibration table is missing columns: "
            + ", ".join(sorted(missing))
        )
    if frame[list(required)].isna().any().any():
        raise ValueError(
            "The grade-specific calibration table contains missing values."
        )
    return frame


def shap_top15_frame() -> pd.DataFrame:
    path = RESULTS / "tables" / "table_s_shap_top15.csv"
    if not path.exists():
        raise FileNotFoundError(
            "The outline-required TreeSHAP table is missing. Run explain.py: "
            f"{path}"
        )
    frame = pd.read_csv(path)
    required = {"feature", "mean_abs_tree_shap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The TreeSHAP table is missing columns: "
            + ", ".join(sorted(missing))
        )
    if len(frame) != 15 or frame["feature"].astype(str).nunique() != 15:
        raise ValueError(
            "The outline requires exactly 15 unique ranked TreeSHAP features."
        )
    return frame


def time_importance_frame(config: dict) -> pd.DataFrame:
    path = RESULTS / "tables" / "table_s_time_dependent_importance.csv"
    if not path.exists():
        raise FileNotFoundError(
            "The outline-required time-dependent importance table is missing. "
            f"Run explain.py: {path}"
        )
    frame = pd.read_csv(path)
    required = {"hour", "channel", "mean_abs_gradient"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The time-dependent importance table is missing columns: "
            + ", ".join(sorted(missing))
        )
    expected_channels = set(configured_channels(config))
    observed_channels = set(frame["channel"].astype(str))
    if observed_channels != expected_channels:
        raise ValueError(
            "The time-dependent importance table does not match the 14 "
            "configured channels."
        )
    expected_hours = set(range(int(config["sequence_hours"])))
    hours = pd.to_numeric(frame["hour"], errors="coerce")
    if hours.isna().any() or not np.isfinite(hours).all():
        raise ValueError(
            "The time-dependent importance table contains invalid hours."
        )
    frame["hour"] = hours.astype(int)
    observed_hours = set(frame["hour"])
    if observed_hours != expected_hours:
        raise ValueError(
            "The time-dependent importance table does not cover ICU hours 0-23."
        )
    pairs = list(zip(frame["channel"].astype(str), frame["hour"]))
    if len(pairs) != len(set(pairs)):
        raise ValueError(
            "The time-dependent importance table contains duplicate "
            "channel-hour rows."
        )
    expected_pairs = {
        (channel, hour)
        for channel in expected_channels
        for hour in expected_hours
    }
    observed_pairs = set(pairs)
    if observed_pairs != expected_pairs:
        raise ValueError(
            "The time-dependent importance table does not contain the full "
            "14-channel by ICU hour 0-23 grid."
        )
    gradients = pd.to_numeric(frame["mean_abs_gradient"], errors="coerce")
    if gradients.isna().any() or not np.isfinite(gradients).all():
        raise ValueError(
            "The time-dependent importance table contains non-finite "
            "importance values."
        )
    return frame


def missingness_frame() -> pd.DataFrame:
    manifest = load_json(RESULTS / "run_manifest.json")
    feature_columns = list(manifest["features"])
    rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(PROCESSED / cohort / "cohort.csv")
        for column in feature_columns:
            if column not in frame:
                value = "Not harmonized"
            else:
                missing = int(frame[column].isna().sum())
                value = f"{missing} ({100 * frame[column].isna().mean():.1f}%)"
            rows.append(
                {
                    "Cohort": COHORT_LABELS[cohort],
                    "Variable": column,
                    "Missing, n (%)": value,
                }
            )
    return pd.DataFrame(rows)


def flow_frame() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(PROCESSED / cohort / "cohort_flow.csv")
            for cohort in COHORTS
        ],
        ignore_index=True,
    )


def flow_count(frame: pd.DataFrame, reason_code: str) -> int:
    """Return one unique flow count and fail closed on ambiguous artifacts."""
    match = frame[frame["reason_code"].astype(str).eq(reason_code)]
    if len(match) != 1:
        raise ValueError(
            f"Cohort flow must contain one row for {reason_code!r}; "
            f"observed {len(match)}."
        )
    return int(match["n"].iloc[0])


def measurement_audit_frame() -> pd.DataFrame:
    path = RESULTS / "tables" / "table_s4_measurement_audit.csv"
    if not path.exists():
        raise FileNotFoundError(
            "The measurement audit is missing. Run make_tables_figures before "
            f"make_manuscript: {path}"
        )
    frame = pd.read_csv(path)
    required = {
        "Cohort",
        "Patients with any ETT-flagged post-landmark GCS, n (%)",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The measurement audit is missing columns: "
            + ", ".join(sorted(missing))
        )
    return frame


def monitoring_ascertainment_statement() -> str:
    """Describe how recorded GCS intensity relates to the endpoint."""
    path = RESULTS / "audits" / "monitoring_ascertainment.json"
    if not path.exists():
        return (
            "Recording intensity can affect whether a GCS decline is observed, "
            "and that dependence was not quantified because the monitoring "
            "audit was absent."
        )
    block = load_json(path)["datasets"]
    parts = []
    for name in ("mimic4", "mimic3", "eicu"):
        strata = {
            row["stratum"]: row
            for row in block[name]["by_post_landmark_gcs_hours"]
        }
        low = strata["2-5"]["any_deterioration_percent"]
        high = strata["24+"]["any_deterioration_percent"]
        parts.append(f"{low:.1f}% versus {high:.1f}% in {COHORT_LABELS[name]}")
    return (
        "The probability of a recorded deterioration also depended on how often "
        "GCS was recorded: any-deterioration rates for stays with 2-5 versus at "
        "least 24 post-landmark GCS hours were "
        + ", ".join(parts[:-1])
        + f", and {parts[-1]}. The endpoint therefore measures documented "
        "deterioration, and between-database differences partly reflect "
        "documentation practice. Monitoring intensity entered the predictor "
        "matrix, but that adjustment cannot remove dependence in the label "
        "itself."
    )


REFERENCES = [
    "Leira R, Davalos A, Silva Y, Gil-Peralta A, Tejada J, Garcia M, et al. Early neurologic deterioration in intracerebral hemorrhage: predictors and associated factors. Neurology. 2004;63(3):461-467. doi:10.1212/01.wnl.0000133204.81153.ac",
    "Moon JS, Janjua N, Ahmed S, Kirmani JF, Harris-Lane P, Jacob M, et al. Prehospital neurologic deterioration in patients with intracerebral hemorrhage. Crit Care Med. 2008;36(1):172-175. doi:10.1097/01.CCM.0000297876.62464.6B",
    "Law ZK, Dineen R, England TJ, Cala L, Mistri AK, Appleton JP, et al. Predictors and outcomes of neurological deterioration in intracerebral hemorrhage: results from the TICH-2 randomized controlled trial. Transl Stroke Res. 2021;12(2):275-283. doi:10.1007/s12975-020-00845-6",
    "Kuohn LR, Witsch J, Steiner T, Sheth KN, Kamel H, Navi BB, et al. Early deterioration, hematoma expansion, and outcomes in deep versus lobar intracerebral hemorrhage: the FAST trial. Stroke. 2022;53(8):2441-2448. doi:10.1161/STROKEAHA.121.037974",
    "Zhou H, Dai D, Xie K, Li A. Risk factors and model construction for early neurological deterioration in patients with intracerebral hemorrhage. Front Neurol. 2025;16. doi:10.3389/fneur.2025.1663347",
    "Zhu W, Zhou J, Ma B, Fan C. Predictors of early neurological deterioration in patients with intracerebral hemorrhage: a systematic review and meta-analysis. J Neurol. 2024;271(6):2980-2991. doi:10.1007/s00415-024-12230-6",
    "Pareja A, Domeniconi G, Chen J, Ma T, Suzumura T, Kanezashi H, et al. EvolveGCN: evolving graph convolutional networks for dynamic graphs. Proc AAAI Conf Artif Intell. 2020;34(4):5363-5370. doi:10.1609/aaai.v34i04.5984",
    "Ma M, Sun P, Li Y, Huo W. Predicting the risk of mortality in ICU patients based on dynamic graph attention network of patient similarity. Math Biosci Eng. 2023;20(8):15326-15344. doi:10.3934/mbe.2023685",
    "Feng X, Zhu S, Shen Y, Zhu H, Yan M, Cai G, et al. Multi-organ spatiotemporal information aware model for sepsis mortality prediction. Artif Intell Med. 2024;147:102746. doi:10.1016/j.artmed.2023.102746",
    "Xu Z, Guo J, Qin L, Xie Y, Xiao Y, Lin X, et al. Predicting ICU interventions: a transparent decision support model based on multivariate time series graph convolutional neural network. IEEE J Biomed Health Inform. 2024;28(6):3709-3720. doi:10.1109/JBHI.2024.3379998",
    "McCullagh P. Regression models for ordinal data. J R Stat Soc Series B. 1980;42(2):109-127. doi:10.1111/j.2517-6161.1980.tb01109.x",
    "Van Hoorde K, Vergouwe Y, Timmerman D, Van Huffel S, Steyerberg EW, Van Calster B. Assessing calibration of multinomial risk prediction models. Stat Med. 2014;33(15):2585-2596. doi:10.1002/sim.6114",
    "Edlinger M, van Smeden M, Alber HF, Wanitschek M, Van Calster B. Risk prediction models for discrete ordinal outcomes: calibration and the impact of the proportional odds assumption. Stat Med. 2022;41(8):1334-1360. doi:10.1002/sim.9281",
    "Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17(1):230. doi:10.1186/s12916-019-1466-7",
    "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574. doi:10.1177/0272989X06295361",
    "Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378",
    "Collins GS, Dhiman P, Ma J, Schlussel MM, Archer L, Van Calster B, et al. Evaluation of clinical prediction models (part 1): from development to external validation. BMJ. 2024;384:e074819. doi:10.1136/bmj-2023-074819",
    "Riley RD, Archer L, Snell KIE, Ensor J, Dhiman P, Martin GP, et al. Evaluation of clinical prediction models (part 2): how to undertake an external validation study. BMJ. 2024;384:e074820. doi:10.1136/bmj-2023-074820",
    "Riley RD, Snell KIE, Archer L, Ensor J, Debray TPA, van Calster B, et al. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ. 2024;384:e074821. doi:10.1136/bmj-2023-074821",
    "Van Calster B, Steyerberg EW, Wynants L, van Smeden M. There is no such thing as a validated prediction model. BMC Med. 2023;21(1):70. doi:10.1186/s12916-023-02779-w",
    "Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. doi:10.1186/1471-2105-7-91",
    "Runje D, Shankaranarayana SM. Constrained monotonic neural networks. Proc Mach Learn Res. 2023;202:29338-29353. arXiv:2205.11775",
    "Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10(1):1. doi:10.1038/s41597-022-01899-x",
    "Johnson AEW, Pollard TJ, Shen L, Lehman LH, Feng M, Ghassemi M, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035. doi:10.1038/sdata.2016.35",
    "Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Sci Data. 2018;5:180178. doi:10.1038/sdata.2018.178",
    "Teasdale G, Jennett B. Assessment of coma and impaired consciousness: a practical scale. Lancet. 1974;304(7872):81-84. doi:10.1016/S0140-6736(74)91639-0",
]


def metric(block: dict, model: str, name: str) -> float:
    return float(block[model]["point"][name])


def point_difference(primary: dict, comparator: dict, model: str, name: str) -> float:
    return metric(primary, "dggru_tr", name) - metric(comparator, model, name)


def paired_interval(block: dict, comparator: str) -> list[float]:
    interval = block.get("paired_differences_vs_dggru_tr", {}).get(
        comparator, [math.nan, math.nan]
    )
    if len(interval) != 2:
        return [math.nan, math.nan]
    return interval


def paired_effect_statement(
    block: dict,
    comparator: str,
    cohort_label: str,
) -> str:
    delta = (
        metric(block, "dggru_tr", "macro_auc")
        - metric(block, comparator, "macro_auc")
    )
    interval = paired_interval(block, comparator)
    if not all(np.isfinite(interval)):
        return (
            f"In {cohort_label}, the paired macro-AUC difference versus "
            f"{model_label(comparator)} was {delta:+.3f}; a paired interval "
            "was not available."
        )
    if interval[0] > 0:
        evidence = "the paired interval was above zero"
    elif interval[1] < 0:
        evidence = "the paired interval was below zero"
    else:
        evidence = "the paired interval included zero"
    return (
        f"In {cohort_label}, the paired macro-AUC difference versus "
        f"{model_label(comparator)} was {delta:+.3f} "
        f"(95% interval {interval[0]:+.3f} to {interval[1]:+.3f}); "
        f"{evidence}."
    )


def ordered_model_names(block: dict, manifest: dict) -> tuple[str, ...]:
    """Return evaluated models without fixing the model count in prose."""
    available = {
        name
        for name, value in block.items()
        if isinstance(value, dict) and isinstance(value.get("point"), dict)
    }
    declared = manifest.get("models", [])
    if isinstance(declared, str):
        declared = [declared]
    ordered = [
        str(name)
        for name in declared
        if str(name) in available
    ]
    ordered.extend(sorted(available.difference(ordered)))
    if not ordered:
        raise ValueError("No evaluated models were found in metrics.json.")
    return tuple(ordered)


def outline_model_names(block: dict, manifest: dict) -> tuple[str, ...]:
    """Return the exact outline comparison set or stop before rendering."""
    available = set(ordered_model_names(block, manifest))
    missing = OUTLINE_REQUIRED_MODELS.difference(available)
    unexpected = available.difference(
        OUTLINE_REQUIRED_MODELS.union(SECONDARY_ABLATION_MODELS)
    )
    problems = []
    if missing:
        problems.append("missing " + ", ".join(sorted(missing)))
    if unexpected:
        problems.append(
            "unexpected or obsolete " + ", ".join(sorted(unexpected))
        )
    if problems:
        raise ValueError(
            "metrics.json does not match the outline Table 3 comparison set: "
            + "; ".join(problems)
        )
    declared = manifest.get("models", [])
    if isinstance(declared, str):
        declared = [declared]
    declared_set = {str(name) for name in declared}
    declared_missing = OUTLINE_REQUIRED_MODELS.difference(declared_set)
    if declared_missing:
        raise ValueError(
            "run_manifest.json does not declare the full outline model set: "
            + ", ".join(sorted(declared_missing))
        )
    declared_unexpected = declared_set.difference(
        OUTLINE_REQUIRED_MODELS.union(SECONDARY_ABLATION_MODELS)
    )
    if declared_unexpected:
        raise ValueError(
            "run_manifest.json declares unexpected or obsolete models: "
            + ", ".join(sorted(declared_unexpected))
        )
    return OUTLINE_MODEL_ORDER


def table3_comparison_frame(
    metrics: dict,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Build the outline Table 3 directly from current metric identifiers."""
    if set(models) != OUTLINE_REQUIRED_MODELS:
        raise ValueError(
            "Table 3 requires exactly the five baselines, GCS-slope "
            "comparator, DG-GRU, "
            "DG-GRU + TR, nested stacking, and three controls."
        )
    rows = []
    for cohort in ("development", "mimic3", "eicu"):
        block = metric_block(metrics, cohort)
        available = {
            name
            for name, value in block.items()
            if name in OUTLINE_REQUIRED_MODELS
            and isinstance(value, dict)
            and isinstance(value.get("point"), dict)
        }
        missing = OUTLINE_REQUIRED_MODELS.difference(available)
        if missing:
            raise ValueError(
                f"{cohort} is missing outline models: "
                + ", ".join(sorted(missing))
            )
        unexpected = available.difference(OUTLINE_REQUIRED_MODELS)
        if unexpected:
            raise ValueError(
                f"{cohort} contains unexpected or obsolete models: "
                + ", ".join(sorted(unexpected))
            )
        paired = block.get("paired_differences_vs_dggru_tr", {})
        comparisons = block.get("comparisons_vs_primary", {})
        comparators = OUTLINE_REQUIRED_MODELS.difference({"dggru_tr"})
        missing_paired = {
            name
            for name in comparators
            if name not in paired
            or not isinstance(paired[name], (list, tuple))
            or len(paired[name]) != 2
            or not all(np.isfinite(paired[name]))
        }
        if missing_paired:
            raise ValueError(
                f"{cohort} lacks finite paired macro-AUC intervals versus "
                "DG-GRU + TR for: "
                + ", ".join(sorted(missing_paired))
            )
        for model in models:
            point = block[model]["point"]
            comparison = comparisons.get(model, {})
            if model == "dggru_tr":
                delta_display = "Reference"
            else:
                delta = float(
                    comparison.get(
                        "delta_macro_auc",
                        metric(block, "dggru_tr", "macro_auc")
                        - metric(block, model, "macro_auc"),
                    )
                )
                interval = paired_interval(block, model)
                if all(np.isfinite(interval)):
                    delta_display = (
                        f"{delta:+.3f} ({interval[0]:+.3f} to "
                        f"{interval[1]:+.3f})"
                    )
                else:
                    delta_display = f"{delta:+.3f} (NA)"
            if model == "dggru_tr":
                nri_any = nri_severe = idi_any = idi_severe = "Reference"
            else:
                nri_any = fmt(comparison.get("nri_any_continuous"))
                nri_severe = fmt(comparison.get("nri_severe_continuous"))
                idi_any = fmt(comparison.get("idi_any"))
                idi_severe = fmt(comparison.get("idi_severe"))
            rows.append(
                {
                    "Cohort": PLOT_LABELS[cohort],
                    "Model": model_label(model),
                    "n": int(point["n"]),
                    "Macro AUC (95% CI)": fmt_ci(
                        block, model, "macro_auc"
                    ),
                    "Grade 0 AUC (95% CI)": fmt_ci(
                        block, model, "auc_grade_0"
                    ),
                    "Grade 1 AUC (95% CI)": fmt_ci(
                        block, model, "auc_grade_1"
                    ),
                    "Grade 2 AUC (95% CI)": fmt_ci(
                        block, model, "auc_grade_2"
                    ),
                    "Any deterioration AUC (95% CI)": fmt_ci(
                        block, model, "auc_any_deterioration"
                    ),
                    "Severe deterioration AUC (95% CI)": fmt_ci(
                        block, model, "auc_severe_deterioration"
                    ),
                    "DG-GRU + TR minus comparator: delta macro AUC "
                    "(95% CI)": delta_display,
                    "Continuous NRI, any deterioration": nri_any,
                    "Continuous NRI, severe deterioration": nri_severe,
                    "IDI, any deterioration": idi_any,
                    "IDI, severe deterioration": idi_severe,
                    "Brier score": fmt(point["multiclass_brier"]),
                    "Calibration slope": fmt(
                        point["calibration_slope_any"]
                    ),
                    "ECE": fmt(point["expected_calibration_error"]),
                }
            )
    return pd.DataFrame(rows)


def documentation_transport_statement() -> str:
    """Summarize the MIMIC-IV-fixed documentation-stratum transport analysis."""
    path = RESULTS / "tables" / "table_s_transport_documentation.csv"
    if not path.exists():
        return (
            "The documentation-stratified transport analysis has "
            "not yet been generated."
        )
    table = pd.read_csv(path)
    required = {
        "dataset",
        "model",
        "documentation_stratum",
        "n",
        "macro_auc",
        "calibration_slope_any",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Documentation-transport table is missing columns: "
            + ", ".join(sorted(missing))
        )
    primary = table[table["model"].eq("dggru_tr")].copy()
    if primary.empty:
        raise ValueError(
            "Documentation-transport table has no DG-GRU + TR results."
        )
    parts = []
    for dataset, label in (("mimic3", "MIMIC-III"), ("eicu", "eICU")):
        rows = primary[primary["dataset"].eq(dataset)]
        auc = pd.to_numeric(rows["macro_auc"], errors="coerce").dropna()
        slope = pd.to_numeric(
            rows["calibration_slope_any"], errors="coerce"
        ).dropna()
        if auc.empty:
            parts.append(
                f"{label}: macro AUC was not estimable in the documentation strata"
            )
            continue
        auc_text = (
            f"{auc.min():.3f}"
            if np.isclose(auc.min(), auc.max())
            else f"{auc.min():.3f}-{auc.max():.3f}"
        )
        slope_text = (
            "not estimable"
            if slope.empty
            else (
                f"{slope.min():.3f}"
                if np.isclose(slope.min(), slope.max())
                else f"{slope.min():.3f}-{slope.max():.3f}"
            )
        )
        parts.append(
            f"{label}: macro AUC {auc_text} and any-deterioration calibration "
            f"slope {slope_text} across the low, middle, and high strata"
        )
    return (
        "Using the three GCS-documentation-intensity strata fixed in MIMIC-IV, "
        + "; ".join(parts)
        + " (Supplementary Table S7). Strata with fewer than three outcome "
        "grades were not assigned a macro AUC."
    )


def temporal_ablation_frame(metrics: dict) -> pd.DataFrame:
    """Summarize matched temporal-input ablations without mixing external arms."""
    block = metric_block(metrics, "development")
    models = (
        "dggru_tr",
        "dggru_tr_noseq",
        "dggru_tr_static",
        "dggru_tr_traj",
        "dggru_tr_gcsmask",
    )
    rows = []
    primary = block.get("dggru_tr", {}).get("point")
    for name in models:
        point = block.get(name, {}).get("point")
        if not isinstance(point, dict):
            continue
        rows.append(
            {
                "Model": MODEL_LABELS[name],
                "Macro AUC": point.get("macro_auc", math.nan),
                "Any AUC": point.get("auc_any_deterioration", math.nan),
                "Severe AUC": point.get("auc_severe_deterioration", math.nan),
                "Brier": point.get("multiclass_brier", math.nan),
                "Delta macro AUC vs primary": (
                    float(point.get("macro_auc", math.nan))
                    - float(primary.get("macro_auc", math.nan))
                    if isinstance(primary, dict)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def table2_output_frame(
    metrics: dict,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Load the table emitted by make_tables_figures for both outputs."""
    development = metrics.get("development", {}) if isinstance(metrics, dict) else {}
    complete_metrics = OUTLINE_REQUIRED_MODELS.issubset(
        {
            name
            for name, value in development.items()
            if isinstance(value, dict) and isinstance(value.get("point"), dict)
        }
    )
    if complete_metrics:
        return table3_comparison_frame(metrics, models)

    path = RESULTS / "tables" / "table3_model_performance.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Table 3 model-performance output is missing. Run "
            "make_tables_figures before "
            "make_manuscript."
        )
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Cohort", "Model", "n"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Table 3 model-performance output is missing columns: "
            + ", ".join(sorted(missing))
        )
    cohort_counts = table.groupby("Cohort", dropna=False)["Model"].nunique()
    expected_cohorts = len(("development", "mimic3", "eicu"))
    if len(cohort_counts) != expected_cohorts:
        raise ValueError(
            "Table 3 model-performance output does not contain all three "
            "analysis cohorts."
        )
    if not cohort_counts.eq(len(OUTLINE_MODEL_ORDER)).all():
        raise ValueError(
            "Table 3 model-performance output is stale: model counts per "
            "cohort do not match "
            f"the {len(OUTLINE_MODEL_ORDER)} outline models."
        )
    observed = set(table["Model"].dropna().astype(str))
    if set(models) != OUTLINE_REQUIRED_MODELS:
        raise ValueError(
            "The requested Table 3 models do not equal the outline comparison set."
        )
    missing_labels = []
    for name in models:
        accepted = TABLE2_MODEL_LABEL_ALIASES.get(
            name,
            {MODEL_LABELS[name]},
        )
        if not accepted.intersection(observed):
            missing_labels.append(MODEL_LABELS[name])
    if missing_labels:
        raise ValueError(
            "Table 3 model-performance output is stale: missing model labels "
            + ", ".join(missing_labels)
        )
    table["Cohort"] = table["Cohort"].replace(
        {
            "MIMIC-III same-institution external": PLOT_LABELS["mimic3"],
        }
    )
    return table


def manuscript_markdown() -> tuple[str, dict[str, pd.DataFrame]]:
    config = load_config()
    metrics = load_json(RESULTS / "metrics.json")
    manifest = load_json(RESULTS / "run_manifest.json")
    title_info = title_context(metrics)
    display_title = str(title_info["title"])
    display_running_title = str(title_info["running_title"])
    display_keywords = str(title_info["keywords"])
    temporal_regularization_primary = bool(
        title_info["temporal_regularization_primary"]
    )
    dev = metric_block(metrics, "development")
    mimic3 = metric_block(metrics, "mimic3")
    eicu = metric_block(metrics, "eicu")
    models = outline_model_names(dev, manifest)
    outline_model_names(mimic3, manifest)
    outline_model_names(eicu, manifest)
    n_static = feature_count(manifest)
    n_eligible = eligible_feature_count(manifest)
    n_static_summary = eligible_feature_count(manifest) - dynamic_summary_count()
    n_seeds = seed_count(manifest)
    if n_seeds != 5:
        raise ValueError("The outline requires five fixed model seeds.")
    channels = configured_channels(config)
    channel_labels = config["channel_labels"]
    channel_sentence = ", ".join(
        channel_labels[channel].rstrip(".") for channel in channels[:-1]
    ) + f", and {channel_labels[channels[-1]].rstrip('.')}"
    cohort_frames = {
        cohort: pd.read_csv(PROCESSED / cohort / "cohort.csv")
        for cohort in COHORTS
    }
    cohort_flows = {
        cohort: pd.read_csv(PROCESSED / cohort / "cohort_flow.csv")
        for cohort in COHORTS
    }
    source_counts = {
        cohort: flow_count(
            cohort_flows[cohort],
            "source_ich_icu_stays",
        )
        for cohort in COHORTS
    }
    counts = {
        cohort: {
            grade: int((frame["outcome"] == grade).sum())
            for grade in (0, 1, 2)
        }
        for cohort, frame in cohort_frames.items()
    }
    death = {
        cohort: int(
            frame["death_within_72h"].fillna(False).astype(bool).sum()
        )
        for cohort, frame in cohort_frames.items()
    }
    rass_coverage = {
        cohort: (
            100.0
            * pd.to_numeric(
                frame.get("min_rass_24h", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            ).notna().mean()
        )
        for cohort, frame in cohort_frames.items()
    }
    any_events = counts["mimic4"][1] + counts["mimic4"][2]
    severe_events = counts["mimic4"][2]
    epv_any = any_events / n_static if n_static else np.nan
    epv_severe = severe_events / n_static if n_static else np.nan

    tables = {
        "table1": table1_frame(),
        "table1b": table1_outcome_frame(),
        "table2_outcome": table2_outcome_frame(),
        "table2": table2_output_frame(metrics, models),
        "table3_lambda": lambda_frame(),
        "temporal_ablations": temporal_ablation_frame(metrics),
        "table4": sensitivity_frame(),
        "table5": trajectory_frame(),
        "grade_calibration": grade_calibration_frame(),
        "subgroup": subgroup_frame(),
        "shap_top15": shap_top15_frame(),
        "time_importance": time_importance_frame(config),
        "flow": flow_frame(),
        "missingness": missingness_frame(),
    }

    dev_tr_macro = metric(dev, "dggru_tr", "macro_auc")
    dev_graph_macro = metric(dev, "dggru", "macro_auc")
    dev_summary_macro = metric(dev, "xgb_summary", "macro_auc")
    dev_delta_graph = dev_tr_macro - dev_graph_macro
    dev_delta_summary = dev_tr_macro - dev_summary_macro
    m3_delta_graph = (
        metric(mimic3, "dggru_tr", "macro_auc")
        - metric(mimic3, "dggru", "macro_auc")
    )
    ei_delta_graph = (
        metric(eicu, "dggru_tr", "macro_auc")
        - metric(eicu, "dggru", "macro_auc")
    )
    repeat_variability = metrics.get("development", {}).get(
        "cv_repeat_variability",
        {},
    )
    repeat_tr_sd = (
        repeat_variability.get("dggru_tr", {})
        .get("macro_auc", {})
        .get("sd")
    )
    repeat_delta_sd = (
        repeat_variability.get(
            "paired_macro_auc_differences_vs_dggru_tr",
            {},
        )
        .get("dggru", {})
        .get("sd")
    )
    if (
        repeat_tr_sd is not None
        and repeat_delta_sd is not None
        and math.isfinite(float(repeat_tr_sd))
        and math.isfinite(float(repeat_delta_sd))
    ):
        repeat_variability_statement = (
            "The SD across repeated CV partitions was "
            f"{float(repeat_tr_sd):.3f} for the DG-GRU + TR macro AUC "
            "and "
            f"{float(repeat_delta_sd):.3f} for its paired difference "
            "versus DG-GRU; this SD summarizes combined CV-partition "
            "and training stochasticity."
        )
    else:
        repeat_variability_statement = (
            "Repeat-specific SDs were not available in this results file."
        )

    lambda_values = tables["table3_lambda"]
    if not lambda_values.empty and "Macro AUC" in lambda_values:
        lambda_auc_range = (
            f"{lambda_values['Macro AUC'].min():.3f}-"
            f"{lambda_values['Macro AUC'].max():.3f}"
        )
        lambda_delta = (
            float(lambda_values["Macro AUC"].max())
            - float(lambda_values["Macro AUC"].min())
        )
        lambda_selection_statement = (
            "The lambda grid produced development macro AUCs from "
            f"{lambda_auc_range}, a range of {fmt(lambda_delta, 4)} "
            "(Supplementary Table S8)."
        )
        primary_lambda_monotone_median = float(manifest["lambda_monotone"])
        primary_lambda_smooth_median = float(manifest["lambda_smooth"])
    elif not lambda_values.empty:
        lambda_monotone_values = pd.to_numeric(
            lambda_values["lambda monotone"],
            errors="coerce",
        ).dropna()
        lambda_smooth_values = pd.to_numeric(
            lambda_values["lambda smooth"],
            errors="coerce",
        ).dropna()
        primary_lambda_monotone_median = float(lambda_monotone_values.median())
        primary_lambda_smooth_median = float(lambda_smooth_values.median())
        lambda_selection_statement = (
            "The 18-point grid was searched independently within each outer "
            "training fold. Selected monotone weights ranged from "
            f"{lambda_monotone_values.min():.4g} to "
            f"{lambda_monotone_values.max():.4g} "
            f"(median {primary_lambda_monotone_median:.4g}); the paired "
            "smoothness weights were fixed at "
            f"{manifest.get('lambda_smooth_ratio', 0.4):.2f} times the selected "
            "monotone weight (Supplementary Table S8)."
        )
    else:
        lambda_selection_statement = (
            "Temporal-regularization selection records were unavailable."
        )
        primary_lambda_monotone_median = float(manifest["lambda_monotone"])
        primary_lambda_smooth_median = float(manifest["lambda_smooth"])

    sensitivity = tables["table4"]
    if sensitivity.empty:
        sensitivity_statement = "Sensitivity analyses had not completed when this draft was rendered."
        sensitivity_range = "not available"
    else:
        auc_values = pd.to_numeric(
            sensitivity.get(
                "DG-GRU + TR macro AUC",
                pd.Series(index=sensitivity.index, dtype=float),
            ),
            errors="coerce",
        ).dropna()
        if auc_values.empty:
            sensitivity_range = "not available"
            sensitivity_statement = (
                "All eight sensitivity analyses were audited, but no variant "
                "was model-runnable in this run; Table 4 reports the explicit "
                "support status for each requested variant."
            )
        else:
            sensitivity_range = (
                f"{auc_values.min():.3f}-{auc_values.max():.3f}"
            )
            runnable_variants = int(
                sensitivity["Support"].astype(str).eq("supported").sum()
            )
            sensitivity_statement = (
                f"Across {len(OUTLINE_SENSITIVITY_ANALYSES)} protocol-defined "
                f"analyses ({runnable_variants} model-runnable variants), "
                "the macro AUC of DG-GRU + TR ranged from "
                f"{sensitivity_range}. Table 4 preserves unsupported or "
                "outcome-only variants with their stated reason."
            )

    trajectories = tables["table5"]
    if trajectories.empty:
        trajectory_statement = (
            "Trajectory exports were not available at the time of rendering."
        )
    else:
        tr_row = trajectories[
            trajectories["Model"].eq("DG-GRU + TR")
        ].iloc[0]
        graph_row = trajectories[
            trajectories["Model"].eq("DG-GRU")
        ].iloc[0]
        trajectory_statement = (
            "The median fraction of decreasing hourly latent-risk steps was "
            f"{tr_row['Decreasing-step fraction, median']:.3f} for DG-GRU + TR "
            f"and {graph_row['Decreasing-step fraction, median']:.3f} for "
            "unregularized DG-GRU; the corresponding median monotone "
            f"violations were {tr_row['Monotone violation, median']:.4f} and "
            f"{graph_row['Monotone violation, median']:.4f}, median roughness "
            f"values were {tr_row['Roughness, median']:.4f} and "
            f"{graph_row['Roughness, median']:.4f}, and median total "
            f"variations were {tr_row['Total variation, median']:.4f} and "
            f"{graph_row['Total variation, median']:.4f}. These are latent "
            "trajectory variation metrics, not raw-input step counts."
        )

    ablations = tables["temporal_ablations"]
    if ablations.empty or not {
        "dggru_tr",
        "dggru_tr_noseq",
        "dggru_tr_static",
        "dggru_tr_traj",
    }.issubset(ablations["Model"].tolist()):
        ablation_statement = (
            "Matched temporal-input ablations were not available in this run."
        )
    else:
        ablation_by_label = ablations.set_index("Model")
        primary_auc = float(
            ablation_by_label.loc[MODEL_LABELS["dggru_tr"], "Macro AUC"]
        )
        no_sequence_auc = float(
            ablation_by_label.loc[
                MODEL_LABELS["dggru_tr_noseq"],
                "Macro AUC",
            ]
        )
        static_only_auc = float(
            ablation_by_label.loc[
                MODEL_LABELS["dggru_tr_static"],
                "Macro AUC",
            ]
        )
        trajectory_only_auc = float(
            ablation_by_label.loc[
                MODEL_LABELS["dggru_tr_traj"],
                "Macro AUC",
            ]
        )
        ablation_statement = (
            "In matched development ablations, the primary model achieved "
            f"macro AUC {primary_auc:.3f}; the same {n_static} core static inputs with the "
            f"sequence zeroed achieved {no_sequence_auc:.3f}, the observed "
            f"sequence with 125 non-summary static inputs achieved "
            f"{static_only_auc:.3f}, and the sequence-only arm achieved "
            f"{trajectory_only_auc:.3f}. These comparisons bound the "
            "incremental temporal contribution and are reported without "
            "selecting a replacement primary model."
        )

    audit = measurement_audit_frame()
    if audit.empty:
        ett_statement = (
            "ETT-flagged GCS observations were audited separately in the "
            "supplementary material."
        )
    else:
        ett_values = audit[
            audit["Cohort"].eq(COHORT_LABELS["mimic4"])
        ][
            "Patients with any ETT-flagged post-landmark GCS, n (%)"
        ].tolist()
        ett_statement = (
            "In MIMIC-IV, the proportion with any ETT-flagged post-landmark "
            "GCS was "
            + "; ".join(ett_values)
            + " across grades 0, 1, and 2, respectively."
        )

    summary_vs_tr = (
        "higher" if dev_summary_macro > dev_tr_macro else "lower"
    )
    eicu_summary_vs_tr = (
        "higher"
        if metric(eicu, "xgb_summary", "macro_auc") > metric(eicu, "dggru_tr", "macro_auc")
        else "lower"
    )
    oof_top_point_model = max(
        models,
        key=lambda name: metric(dev, name, "macro_auc"),
    )
    oof_top_point_label = model_label(oof_top_point_model)
    oof_top_point_auc = metric(dev, oof_top_point_model, "macro_auc")
    dev_effect_statement = paired_effect_statement(
        dev,
        "dggru",
        "MIMIC-IV development",
    )
    m3_effect_statement = paired_effect_statement(
        mimic3,
        "dggru",
        "MIMIC-III",
    )
    eicu_effect_statement = paired_effect_statement(
        eicu,
        "dggru",
        "eICU-CRD",
    )
    dev_interval = paired_interval(dev, "dggru")
    if all(np.isfinite(dev_interval)) and dev_interval[0] > 0:
        primary_effect_conclusion = (
            "The protocol-defined paired interval favored DG-GRU + TR over "
            "unregularized DG-GRU in development, but the external estimates "
            "and calibration results determine whether that difference "
            "transported."
        )
    elif all(np.isfinite(dev_interval)) and dev_interval[1] < 0:
        primary_effect_conclusion = (
            "The protocol-defined paired interval favored unregularized DG-GRU "
            "over DG-GRU + TR in development; the data therefore do not show "
            "an incremental discrimination benefit from temporal regularization."
        )
    else:
        primary_effect_conclusion = (
            "The protocol-defined paired interval did not establish an incremental "
            "discrimination benefit from temporal regularization."
        )
    top_model_statement = (
        "this descriptive ranking does not identify a validated best model"
        if oof_top_point_model != "dggru_tr"
        else (
            "DG-GRU + TR had the highest descriptive point estimate, but this "
            "ranking alone does not establish a validated best model"
        )
    )
    if temporal_regularization_primary:
        abstract_background = (
            "Early neurological deterioration after nontraumatic "
            "intracerebral hemorrhage is common, but existing prediction "
            "models seldom combine ordinal outcome structure, intensive care "
            "time series, and explicit temporal regularization. It is also "
            "unclear whether temporal regularization of a latent risk "
            "trajectory improves transport across hospitals and database "
            "generations."
        )
    else:
        abstract_background = (
            "Early neurological deterioration after nontraumatic "
            "intracerebral hemorrhage is common, but existing prediction "
            "models seldom combine ordinal outcome structure with intensive "
            "care time series. This study evaluated a dynamic graph ordinal "
            "prediction model and tested whether temporal regularization "
            "added reproducible discrimination."
        )
    abstract = f"""
## Abstract

**Background.** {abstract_background}

**Methods.** We developed and transported a dynamic graph gated recurrent unit with temporal risk-trajectory regularization (DG-GRU + TR) for the common endpoint of confirmed Glasgow Coma Scale (GCS) deterioration or death from ICU hours 24 to 72. The development cohort comprised {len(cohort_frames['mimic4']):,} MIMIC-IV stays ({counts['mimic4'][0]}/{counts['mimic4'][1]}/{counts['mimic4'][2]} for grades 0/1/2). Same-institution transport used {len(cohort_frames['mimic3']):,} MIMIC-III stays from a different database generation, and geographic transport used {len(cohort_frames['eicu']):,} eICU-CRD stays. Procedure start times identified recorded intubation but did not establish whether it was unplanned; because that is the required endpoint component, the intubation-augmented analysis was retained in the audit as unsupported. Predictors preceded the ICU hour-24 landmark. Patient-grouped five-fold cross-validation repeated {manifest['cv_repeats']} times used {n_seeds} fixed seeds whose averaged validation probabilities selected inner hyperparameters and lambda_monotone; internal early stopping, nested out-of-fold stacking, and conditional cluster-bootstrap uncertainty were also used. Primary uncertainty resampled patients in MIMIC-IV and MIMIC-III and hospital_key clusters in eICU. Standard unclustered first-partition DeLong tests were secondary; the P024/P039 fixed primary Holm family was limited to the five baselines ({', '.join(model_label(name) for name in OUTLINE_BASELINE_MODELS)}) with any and severe outcomes, and all other comparisons were `exploratory_not_in_primary_holm_family`. The comparison set also contained a GCS-slope-only comparator, DG-GRU, DG-GRU + TR, nested stacking, and three negative controls.

**Results.** In MIMIC-IV, DG-GRU + TR achieved a macro AUC of {fmt_ci(dev, 'dggru_tr', 'macro_auc')} and any-deterioration AUC of {fmt_ci(dev, 'dggru_tr', 'auc_any_deterioration')}. Summary XGBoost reached a macro AUC of {fmt_ci(dev, 'xgb_summary', 'macro_auc')}. Temporal regularization changed development macro AUC by {dev_delta_graph:+.3f} versus DG-GRU and by {dev_delta_summary:+.3f} versus summary XGBoost. The corresponding changes versus DG-GRU were {m3_delta_graph:+.3f} in MIMIC-III and {ei_delta_graph:+.3f} in the {len(cohort_frames['eicu']):,}-stay eICU cohort. External calibration slopes for DG-GRU + TR were {fmt(metric(mimic3, 'dggru_tr', 'calibration_slope_any'))} and {fmt(metric(eicu, 'dggru_tr', 'calibration_slope_any'))}, respectively.

**Conclusions.** {primary_effect_conclusion} In this run, the highest out-of-fold point estimate for macro AUC was {oof_top_point_label} ({oof_top_point_auc:.3f}); {top_model_statement}. The GCS-based endpoint remains vulnerable to sedation, ventilation, and care limitations.
"""

    results = f"""
## 1. Introduction

Nontraumatic intracranial hemorrhage under predefined ICH diagnosis codes accounts for a substantial share of stroke-related critical illness, and the first days after admission often determine later care [1-6]. Early neurological deterioration is associated with hematoma expansion, edema, impaired consciousness, and worse functional outcome [1-4]. Its reported frequency and predictors vary because studies use different GCS thresholds, observation windows, imaging endpoints, and treatment settings [5,6]. This heterogeneity matters for critical care: a prediction made at ICU hour 24 must use only the information available at that time, and the target should be specified as an observable clinical event rather than an unmeasured biological process.

Ordinal outcome structure is clinically natural. No confirmed deterioration, moderate confirmed deterioration, and severe deterioration or death form an ordered sequence, yet many prediction models collapse the endpoint into a binary label or treat the three grades as unrelated classes. Ordinal models can share information across thresholds and place the cut points on a common latent scale [11-13]. Calibration is central to this approach because a model with useful ranking may still assign probabilities that are too high or too low when transported to a new database [14,15]. Discrimination without calibration would therefore provide an incomplete account of transport.

Intensive care data add a second structure. Vital signs and GCS are measured repeatedly, irregularly, and with different missingness patterns. Recurrent models can represent this history, and dynamic graphs can express changing relations among physiological channels at each hour [7-10]. A graph alone, however, does not define how a latent deterioration risk should behave over time. Smoothness, bounded change, and monotonicity can be added as soft constraints, but their benefit is an empirical question. A constraint may stabilize a trajectory and improve transfer, or it may suppress informative reversals and reduce discrimination. Time-reversal, time-permutation, and generic Laplace input-noise controls are needed to distinguish a directional prior from a temporal-order artifact or generic input perturbation [22].

We therefore evaluated temporal risk-trajectory regularization in a dynamic graph gated recurrent unit (DG-GRU + TR) for ordinal prediction of GCS-based early neurological deterioration or death after spontaneous ICH. The prediction landmark was ICU hour 24, and the outcome window was ICU hours 24 to 72. We developed the model in MIMIC-IV, transported the complete pipeline to MIMIC-III as a same-institution external cohort from a different database generation, and transported it to eICU-CRD as a multi-hospital geographic cohort [23-25]. The study had three aims: to estimate discrimination and calibration without post-landmark leakage; to isolate temporal regularization from graph structure and generic smoothness; and to determine whether any change persisted across transport cohorts and endpoint, sedation, intubation, and early-discharge analyses. The central comparison was deliberately bounded. We estimated whether temporal regularization added reproducible discrimination beyond an otherwise matched dynamic graph model, not whether the model was ready for clinical use.

## 2. Methods

### 2.1 Study design and data sources

This retrospective prediction-model study followed TRIPOD+AI [16]. MIMIC-IV version 3.1 was the development database [23]. MIMIC-III version 1.4 was used as a same-institution external cohort from a different database generation [24]; this contrast represents database-generation and source shift. MIMIC-IV, MIMIC-III, and eICU-CRD source records cover {DATABASE_SPANS['mimic4']}, {DATABASE_SPANS['mimic3']}, and {DATABASE_SPANS['eicu']}, respectively. These are database coverage periods, not a patient-level temporal sequence, and MIMIC dates are shifted. eICU-CRD version 2.0 was used as a multi-hospital geographic external cohort [25]. The databases contain de-identified ICU records collected under credentialed PhysioNet access. The analysis used existing data and did not require new patient contact. Reporting covered model development, internal validation, external transport, and post hoc sensitivity analyses; no model was updated on external outcomes.

### 2.2 Cohort construction

Nontraumatic intracranial hemorrhage was identified from discharge diagnoses using the outline codes: ICD-9 431/432 and ICD-10 I61/I62. These codes include hemorrhages beyond strictly spontaneous intracerebral hemorrhage, so the cohort is named by the implemented code set rather than by an unverified etiology. The same code set was applied in MIMIC-IV, MIMIC-III, and eICU-CRD. Each patient's first ICU episode was identified before applying the ICH, adult, landmark-survival, and GCS eligibility filters. A later ICH stay was therefore not substituted when the first ICU episode did not satisfy the cohort definition. The implemented query used inclusion codes only and did not exclude an otherwise qualifying stay because of a co-coded diagnosis.

Patients were at least 18 years old, alive at ICU hour 24, and had at least one valid total GCS before hour 24. Every included patient, including a patient who died in the ICU by hour 72, was required to have at least two distinct ICU hours with valid total-GCS values between ICU hours 24 and 72; the same requirement applied to a death after a live ICU discharge. Sparse-GCS deaths were therefore excluded rather than promoted to Grade 2. A live ICU discharge before hour 72 was not an exclusion criterion: such a stay was retained when the observed post-landmark window yielded a classifiable grade, and the complete-follow-up subset was analysed as a protocol-defined sensitivity analysis, because requiring the full window at entry would have conditioned on future observation. The outline's earlier draft sample sizes are not carried forward because they do not match the rebuilt audit. The audited source counts were {source_counts['mimic4']:,}, {source_counts['mimic3']:,}, and {source_counts['eicu']:,} ICH ICU stays, and the final analytic cohorts contained {len(cohort_frames['mimic4']):,}, {len(cohort_frames['mimic3']):,}, and {len(cohort_frames['eicu']):,} stays, respectively. These values were read directly from the rebuilt cohort-flow and cohort artifacts; Figure 1 summarizes the unique exclusion sequence.

### 2.3 Composite ordinal endpoint

The common endpoint across all three databases was confirmed GCS deterioration or death, where deterioration was measured against the maximum valid total GCS in ICU hours 0-24 [26]. The endpoint is explicitly coded `P020 = PARTIAL` with `p020_complete = false`, not as a complete P020 composite. GCS values were reduced to hourly medians. A deterioration signal required two adjacent hourly medians, each at least 2 points below baseline; a missing intervening hour broke persistence. Grade 0 denoted no confirmed reduction of at least 2 points and no severe event by hour 72. Grade 1 denoted a confirmed reduction of 2 or 3 points. Grade 2 denoted a confirmed reduction of at least 4 points or death within 72 h of ICU admission, including a death after a live ICU discharge. Procedure records identified post-landmark intubation without a reliable planned versus unplanned distinction. Because the outline requested unplanned intubation, that component was not substituted with the broader recorded-intubation proxy; the corresponding sensitivity analysis is reported as unsupported.

Because the reference was the pre-landmark maximum, a patient could have deteriorated before hour 24. The grades therefore describe the severity observed during hours 24-72 relative to the pre-landmark maximum, not strictly the first incident deterioration. A last-pre-landmark GCS reference was evaluated as a sensitivity analysis. Death was treated as Grade 2 only after the same P018 eligibility rule, including at least two distinct post-landmark GCS hours, had been satisfied. No intubation component entered the transported endpoint. The endpoint was not interpreted as neurological deterioration independent of sedation, analgesia, neuromuscular blockade, tracheal intubation, or limitations of care.

### 2.4 Predictors and preprocessing

{len(channels)} time-varying channels were extracted on an hourly grid: {channel_sentence}. The outline states that there are 14 channels but its inline list names 13 modalities; total GCS was therefore added to reconcile the stated count without omitting a named channel. Physiologically implausible values were removed before tensor construction. `SequencePreprocessor` was fitted inside each outer training fold using training rows only. After that fold-specific fit, causal forward fill was applied within each stay across ICU hours 0-23; backward filling was prohibited. Each sequence therefore included {len(channels)} standardized values and {len(channels)} explicit channel-observation masks. The masks allowed the model to distinguish an imputed value from a measured value. RASS was not fully observed in any database, including eICU-CRD. eICU-CRD also did not provide structured urine-output and FiO2 tables in the accessed source; those channels remain explicitly missing and are represented by their masks rather than cross-database imputation.

Static predictors included age, an indicator for age older than 89 years, sex, race, emergency admission, GCS summaries, vital-sign summaries, monitoring intensity, laboratory summaries from the first 24 hours, and mechanical-ventilation and anticoagulation-related hemorrhage indicators when harmonized with observed variation. MIMIC-III shifts the date of birth for patients older than 89 years to exactly 300 years before the first admission; the resulting age-300 sentinel was recoded to 91 years to match MIMIC-IV `anchor_age`, and the `age_gt89` indicator preserved the interval-censoring information. The eICU `> 89` category was handled identically rather than being collapsed to 89. Discharge location, post-landmark events, procedure counts after the landmark, and outcome-derived variables were excluded. The eligible matrix contained {n_eligible} numeric pre-landmark variables and was fixed by the source schema; no missingness- or variance-based variable selection was applied, and outcome labels were not used. The primary models received a pre-specified core subset of {n_static} variables covering demographics, baseline and summary GCS, vital-sign summaries, urine output, FiO2, RASS, and monitoring intensity. The core subset was fixed before any outcome comparison and was shared by every primary model. Static medians and continuous standardization parameters were estimated from training data only. Supplementary Table S2 reports missingness for the eligible matrix and the core subset.

### 2.5 DG-GRU + TR

The temporal network received {2 * len(channels)} value-and-mask inputs for ICU hours 0-23. At each hour, a value-aware dynamic feature graph formed pairwise relations among the {len(channels)} value channels from learned edge embeddings conditioned on the current values. Graph aggregation was residual and preceded a single-layer GRU whose hidden size was selected from the protocol-defined 48, 64, and 96-unit candidates. A two-layer static encoder mapped the {n_static}-variable core static set into the same hidden dimension. The final recurrent state and static representation were concatenated.

The ordinal head used two cumulative-logit branches, for grade 1 or higher and grade 2, with shared hidden states [11]. Softplus-transformed gaps constrained the cumulative risks to be ordered. The auxiliary latent risk trajectory was regularized in the increasing direction with three terms: a hinge penalty on negative hourly steps, a squared-step roughness penalty, and a penalty on steps larger than 0.5. Class labels were obtained from the median of the predicted ordinal distribution, which minimizes expected absolute ordinal error; modal predictions were retained as a secondary descriptive output. The training objective was

`L = L_NLL + lambda_monotone * L_reversal + lambda_smooth * (L_roughness + L_step_bound)`.

The 18-point grid was searched within each outer training fold. Across folds, the median selected weights were lambda_monotone = {primary_lambda_monotone_median:.4g} and lambda_smooth = {primary_lambda_smooth_median:.4g}. The monotone direction was treated as an inductive bias, not as a biological law. Predefined controls reversed the input time order and permuted the input time order. The third control added independent generic Laplace noise with location 0 and scale 0.01 only to observed, standardized time-series values; missing entries were not noised, and no temporal smoothness penalty was applied in that arm.

### 2.6 Comparators and matched ablations

{len(models)} models were evaluated using a common evaluation pipeline: Static XGBoost, Summary XGBoost, a GCS-temporal-slope-only logistic model, SeqOrdinal GRU, GRU-D, Transformer, DG-GRU, DG-GRU + TR, nested logistic stacking, time-reversed DG-GRU + TR, time-permuted DG-GRU + TR, and DG-GRU with generic Laplace input noise. Static XGBoost used a compact demographic and baseline GCS feature set. The slope-only comparator used one predefined feature, the ordinary least-squares slope of observed pre-landmark total GCS against ICU hour. {shared_static_sentence(n_static)} The matrix included vital-sign, GCS, laboratory, monitoring, and available RASS summaries; the neural models additionally received the {len(channels)} value-and-mask pairs for ICU hours 0-23. The SeqOrdinal GRU used the temporal inputs and static fusion without a graph. GRU-D added learned decay for irregular observation times. The Transformer used self-attention without a graph. DG-GRU added the dynamic feature graph. DG-GRU + TR added monotone and smoothness regularization. The three controls tested reversed time order, permuted time order, and generic Laplace noise added only to observed standardized temporal inputs at scale 0.01 without a temporal penalty. Nested logistic stacking combined Summary XGBoost, SeqOrdinal GRU, and DG-GRU + TR using inner out-of-fold probabilities.

Four secondary internal-development ablations made the temporal contribution identifiable: DG-GRU + TR with the {n_static} core static variables and the sequence and trajectory mask set to zero; DG-GRU + TR with the observed sequence and the {n_static_summary} static variables that exclude the 24-hour trajectory summaries; DG-GRU + TR with the observed sequence and one placeholder static column; and the primary model with a GCS-observation rather than any-channel trajectory mask. These arms were not externally transported. XGBoost and sequence-network hyperparameters were selected by inner patient-grouped macro AUC from three explicit protocol-defined candidate dictionaries per family. The 18-point temporal-regularization grid was searched after sequence-parameter selection. All selection used outer-training data only; external outcomes never informed a hyperparameter, epoch, or lambda choice. Cross-family comparisons therefore describe complete pipelines rather than isolated architecture effects.

### 2.7 Development and transport

    Development used patient-grouped stratified five-fold cross-validation repeated {manifest['cv_repeats']} times for the primary models and nested stacking. Each outer training fold contained its own patient-grouped stratified inner holdout for early stopping and preprocessing [17,21]. No outer-fold information was used to fit medians, scaling parameters, graph parameters, cut points, or stopping epochs. The {n_seeds} fixed random seeds were used jointly in inner model selection: validation probabilities were averaged across seeds before candidate hyperparameters and lambda_monotone were scored. After selection, each seed was fitted separately for the outer out-of-fold predictions and then averaged. The 18-weight lambda screen used one five-fold partition per weight; the selected weight was then carried into the five-repeat primary analysis and was not reselected on external cohorts. The three negative controls and four secondary ablations used one fixed CV partition, with lambda_monotone and the stopping epoch frozen at the primary median so that the controls isolate the architectural or temporal manipulation rather than a new tuning path. For nested stacking, five patient-grouped inner folds generated out-of-fold member probabilities strictly inside each outer training split; a multinomial logistic meta-model was fitted to those probabilities and applied only after the members were refitted on the complete outer training split. The inner out-of-fold folds used one fixed model seed, whereas the outer member predictions retained five-seed averaging. The reported out-of-fold predictions therefore combined repeat-, fold-, seed-, and model-specific estimates without using held-out labels.

For external transport, each fixed model seed was fitted on the complete development cohort and applied without external refitting; predictions were then averaged across the {n_seeds} fixed seeds. For nested stacking, member out-of-fold probabilities were generated within the complete development cohort, the meta-model was fitted there, and the frozen development pipeline was then applied to each external cohort. Temporal networks used the median best epoch from the development outer folds, and tabular imputation used development medians. The eICU cohort retained hospital identifiers only for uncertainty estimation. MIMIC-III was interpreted as same-institution transport under database-generation and source shift; the comparison does not imply a patient-level ordering between databases [18,20].

Transport heterogeneity by GCS documentation intensity was fixed before evaluation. The low, middle, and high strata were defined by the tertiles of monitoring intensity in the complete MIMIC-IV development cohort, and the same absolute cut points were applied unchanged in MIMIC-III and eICU. External cohorts were not requantilized. Macro AUC and calibration were estimated within each stratum, with uncertainty from 2,000 cluster-bootstrap resamples within the external dataset.

### 2.8 Performance, calibration, and uncertainty

The primary discrimination measure was macro one-versus-rest AUC, the unweighted mean of the three class-specific AUCs. Cumulative AUCs were computed for grade 1 or higher (any deterioration) and grade 2 (severe deterioration). Additional measures were adjacent-class and non-adjacent-class error, mean ordinal error, ordinal-median accuracy, quadratic weighted kappa, multiclass Brier score, log loss, calibration slope and intercept, continuous and categorical net reclassification improvement and integrated discrimination improvement, expected calibration error, and decision-curve net benefit across 10%-40% thresholds [12-15]. Calibration curves used fixed probability bins and were descriptive.

Primary inferential uncertainty used 2,000 percentile paired cluster-bootstrap replicates conditional on the fitted out-of-fold predictions and the realized cross-validation partitions. Patients were resampled as clusters in MIMIC-IV and MIMIC-III. Hospitals were resampled as clusters in eICU, with the hospital_key as the primary eICU cluster. Paired AUC differences paired model predictions within the same resampled cluster. Standard paired DeLong tests were unclustered, based only on the first realized CV partition, and secondary. The P024/P039 fixed primary Holm family contains 10 tests: the five baseline models xgb_static, xgb_summary, gru, grud, and transformer, each for any and severe outcomes. DG-GRU, the GCS slope comparator, nested stacking, and all three controls are marked `exploratory_not_in_primary_holm_family` and are not members of that family. The DeLong family does not replace the eICU hospital-cluster bootstrap as the primary inference. The bootstrap did not re-estimate models, generate new cross-validation partitions, reselect stopping epochs, or resample optimization seeds. Separate SDs across repeated cross-validation partitions describe combined partition and training stochasticity.

The cohort sizes were fixed by the available databases and were not justified against target precision criteria for external validation [19]. External estimates are therefore reported with their observed uncertainty rather than as a powered test of transport superiority.

### 2.9 Protocol-defined sensitivity analyses

The 18-point temporal-regularization grid included a true no-regularization reference, component-isolation weights, and higher weights. Selected lambda_monotone and lambda_smooth values were read from the development folds, and Supplementary Table S8 lists their realized combinations. No external outcome was used to select a weight. The eight outline analyses were evaluated as follows: maximum versus final pre-landmark GCS; 1-hour versus 2-hour persistence; exclusion of continuous sedation windows versus restriction to an observed unsedated subgroup; primary exclusion, grade-0 assignment, and inverse-probability weighting for early ICU transfer; the composite endpoint with versus without unplanned intubation; 6-, 24-, and 48-hour landmarks; exclusion of deaths within 72 hours; and eICU GCS with versus without an explicit RASS adjustment. The intubation-augmented variant was unsupported because the sources recorded intubation timing but not whether it was unplanned. Sedation analyses were restricted to datasets with adequate event-level RASS coverage. A missing RASS was not imputed as no deep sedation. The eICU RASS-adjusted variant required an explicit contemporaneous adjusted-GCS field and was marked unsupported otherwise.

### 2.10 Interpretation and subgroups

For post hoc interpretation, Summary XGBoost and DG-GRU + TR were refit on the complete development cohort; these explanation models were not the frozen transported models. TreeSHAP ranked the top 15 predictors for the Summary XGBoost refit, and absolute-gradient importance summarized the DG-GRU + TR refit by ICU hour and channel. The five protocol-defined subgroup factors were age, mechanical ventilation, anticoagulation-related hemorrhage, neurosurgical intervention, and sedation exposure. These analyses were exploratory. No subgroup or explanation result was used to select the primary model.

Analyses used Python 3.10, PyTorch 2.4.1, XGBoost 3.2.0, scikit-learn 1.7.2, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, and DuckDB 1.5.5.

## 3. Results

### 3.1 Cohort and endpoint distribution

The MIMIC-IV development cohort contained {len(cohort_frames['mimic4']):,} stays. Grades 0, 1, and 2 occurred in {counts['mimic4'][0]:,}, {counts['mimic4'][1]:,}, and {counts['mimic4'][2]:,} patients, respectively; {death['mimic4']} patients died by hour 72. The same-institution MIMIC-III external cohort contained {len(cohort_frames['mimic3']):,} stays ({counts['mimic3'][0]}/{counts['mimic3'][1]}/{counts['mimic3'][2]}), with {death['mimic3']} deaths. The eICU geographic cohort contained {len(cohort_frames['eicu']):,} stays ({counts['eicu'][0]}/{counts['eicu'][1]}/{counts['eicu'][2]}), with {death['eicu']} deaths. Table 1 summarizes cohort characteristics, Table 2 gives the ordinal outcome distribution, and Supplementary Table S5 gives characteristics by development outcome grade.

The development predictor matrix contained {n_static} pre-landmark variables. The {any_events:,} any-deterioration events and {severe_events:,} severe events yielded crude events-per-variable ratios of {epv_any:.2f} and {epv_severe:.2f}, respectively. These ratios describe the static matrix and are not interpreted as the effective degrees of freedom of the sequence models.

At least one observed first-24-hour RASS value was available for {rass_coverage['mimic4']:.1f}%, {rass_coverage['mimic3']:.1f}%, and {rass_coverage['eicu']:.1f}% of the MIMIC-IV, MIMIC-III, and eICU analytic cohorts, respectively. This contradicts the outline's statement that RASS was fully available in eICU-CRD and therefore remains a P022 data-limitation. Missing RASS was retained as missing and was never interpreted as the absence of sedation.

### 3.2 Discrimination

In MIMIC-IV, DG-GRU + TR achieved a macro AUC of {fmt_ci(dev, 'dggru_tr', 'macro_auc')}, an any-deterioration AUC of {fmt_ci(dev, 'dggru_tr', 'auc_any_deterioration')}, and a severe-deterioration AUC of {fmt_ci(dev, 'dggru_tr', 'auc_severe_deterioration')}. Summary XGBoost achieved a macro AUC of {fmt_ci(dev, 'xgb_summary', 'macro_auc')}, the GCS-slope comparator achieved {fmt_ci(dev, 'gcs_slope', 'macro_auc')}, and unregularized DG-GRU achieved {fmt_ci(dev, 'dggru', 'macro_auc')}. The development macro AUC was therefore {dev_delta_graph:+.3f} with temporal regularization versus DG-GRU and {dev_delta_summary:+.3f} versus summary XGBoost. {dev_effect_statement} Table 3 reports the five baselines, GCS-slope comparator, DG-GRU, DG-GRU + TR, nested stacking, all three controls, per-grade AUCs, and paired differences versus DG-GRU + TR.

In MIMIC-III, DG-GRU + TR achieved a macro AUC of {fmt_ci(mimic3, 'dggru_tr', 'macro_auc')} and an any-deterioration AUC of {fmt_ci(mimic3, 'dggru_tr', 'auc_any_deterioration')}; the changes versus DG-GRU were {m3_delta_graph:+.3f} for macro AUC and {point_difference(mimic3, mimic3, 'dggru', 'auc_any_deterioration'):+.3f} for any-deterioration AUC. In eICU, DG-GRU + TR achieved a macro AUC of {fmt_ci(eicu, 'dggru_tr', 'macro_auc')} and an any-deterioration AUC of {fmt_ci(eicu, 'dggru_tr', 'auc_any_deterioration')}; the changes versus DG-GRU were {ei_delta_graph:+.3f} and {point_difference(eicu, eicu, 'dggru', 'auc_any_deterioration'):+.3f}. {m3_effect_statement} {eicu_effect_statement} Figure 3A and Table 3 present all {len(models)} outline models across cohorts.

The paired macro-AUC difference for DG-GRU + TR versus unregularized DG-GRU was {dev_delta_graph:+.3f} with a 95% interval of {paired_interval(dev, 'dggru')[0]:+.3f} to {paired_interval(dev, 'dggru')[1]:+.3f} in development, {m3_delta_graph:+.3f} with an interval of {paired_interval(mimic3, 'dggru')[0]:+.3f} to {paired_interval(mimic3, 'dggru')[1]:+.3f} in MIMIC-III, and {ei_delta_graph:+.3f} with an interval of {paired_interval(eicu, 'dggru')[0]:+.3f} to {paired_interval(eicu, 'dggru')[1]:+.3f} in eICU. In MIMIC-IV and MIMIC-III, each estimate is a conditional patient-cluster bootstrap interval; in eICU, it is a conditional hospital-cluster bootstrap interval and remains the primary eICU inference. These intervals condition on realized repeated-CV out-of-fold predictions and do not re-estimate models or resample CV partitions, stopping epochs, or optimization seeds. {repeat_variability_statement} Standard paired DeLong tests are reported secondarily as unclustered, first-CV-partition tests. The P024/P039 fixed primary Holm family contains only the 10 tests for the five baseline models with any and severe outcomes; DG-GRU, the GCS slope comparator, nested stacking, and the three controls are marked `exploratory_not_in_primary_holm_family`. They are not the primary inferential basis for any cohort, and the DeLong tests are not a hospital-clustered eICU analysis.

### 3.3 Calibration and decision curves

The any-deterioration calibration slope for DG-GRU + TR was {fmt(metric(dev, 'dggru_tr', 'calibration_slope_any'))} in development, {fmt(metric(mimic3, 'dggru_tr', 'calibration_slope_any'))} in MIMIC-III, and {fmt(metric(eicu, 'dggru_tr', 'calibration_slope_any'))} in eICU. The intercepts were {fmt(metric(dev, 'dggru_tr', 'calibration_intercept_any'))}, {fmt(metric(mimic3, 'dggru_tr', 'calibration_intercept_any'))}, and {fmt(metric(eicu, 'dggru_tr', 'calibration_intercept_any'))}, respectively. Grade-specific one-versus-rest slopes and intercepts for grades 0, 1, and 2 are reported for every model and cohort in Supplementary Table S12. Top-label expected calibration errors were {fmt(metric(dev, 'dggru_tr', 'expected_calibration_error'))}, {fmt(metric(mimic3, 'dggru_tr', 'expected_calibration_error'))}, and {fmt(metric(eicu, 'dggru_tr', 'expected_calibration_error'))}. Figure 3B and Figure 4A show cumulative and grade-specific calibration, and Figure 4B shows decision curves across the fixed 10%-40% threshold range.

Summary XGBoost had a {summary_vs_tr} development macro AUC than DG-GRU + TR ({fmt(metric(dev, 'xgb_summary', 'macro_auc'))} versus {fmt(dev_tr_macro)}) and a {eicu_summary_vs_tr} eICU macro AUC ({fmt(metric(eicu, 'xgb_summary', 'macro_auc'))} versus {fmt(metric(eicu, 'dggru_tr', 'macro_auc'))}). Both representation families received the same {n_static}-variable core static matrix, including available RASS and urine-output summaries; the neural models additionally received the hourly value and observation-mask sequence. The comparison is therefore pipeline-level and does not isolate graph structure from optimization and sequence representation. Multiclass Brier scores for DG-GRU + TR versus summary XGBoost were {fmt(metric(dev, 'dggru_tr', 'multiclass_brier'))} versus {fmt(metric(dev, 'xgb_summary', 'multiclass_brier'))} in development, {fmt(metric(mimic3, 'dggru_tr', 'multiclass_brier'))} versus {fmt(metric(mimic3, 'xgb_summary', 'multiclass_brier'))} in MIMIC-III, and {fmt(metric(eicu, 'dggru_tr', 'multiclass_brier'))} versus {fmt(metric(eicu, 'xgb_summary', 'multiclass_brier'))} in eICU. The decision curves in Figure 4B are therefore reported as descriptive net-benefit comparisons rather than evidence of clinical superiority.

### 3.4 Temporal regularization, direction, and smoothness

{lambda_selection_statement} The selected weights are reported by their realized monotone and smoothness values. {trajectory_statement} {ablation_statement} Time-reversed, time-permuted, and generic Laplace input-noise controls are shown alongside the primary model in Figure 5 and Table 5. The Laplace arm adds noise only to observed standardized temporal values at scale 0.01 and applies no temporal smoothness penalty. These controls test whether apparent temporal structure survives destruction of time order and whether the directional penalty differs from a generic input-perturbation comparator.

These results separate trajectory structure from discrimination. A more regularized latent path can coexist with unchanged class ranking; therefore the trajectory summaries are interpreted as model behavior, not as evidence that neurological deterioration follows an invariant monotone biological process.

### 3.5 Endpoint and measurement sensitivity

{sensitivity_statement} Table 4 preserves all eight outline analyses and every requested variant, including explicit support status when an exact source field or model-fitting interface was unavailable. Each scenario is interpreted descriptively, and no scenario was used to select the primary model. The 6-, 24-, and 48-hour landmark inputs were generated from the same adult first-ICH candidate source and each landmark applied its own survival and follow-up eligibility rule.

{ett_statement} This measurement dependence is a substantive limitation. ETT flags indicate that a verbal GCS response may be recorded differently during ventilation; they do not remove the effects of sedation or neuromuscular blockade. The eICU data had no harmonized ETT flag, so absence of a flag in eICU cannot be interpreted as absence of intubation.

### 3.6 Subgroups and transport heterogeneity

Development subgroup estimates for age, mechanical ventilation, anticoagulation-related hemorrhage, neurosurgical intervention, and sedation exposure are shown in Figure 6B and Supplementary Table S6. These exploratory intervals were not powered for interaction tests. In complete-development refits used only for interpretation, TreeSHAP ranked the top 15 Summary XGBoost predictors, and absolute-gradient importance showed how DG-GRU + TR depended on each of the 14 channels across ICU hours 0-23 (Figure 6A and Supplementary Tables S9-S10). The eICU cohort contained {cohort_frames['eicu']['hospital_key'].nunique()} hospitals, and the primary analysis included {len(cohort_frames['eicu'])} stays. This sample supports exploratory transport analysis only; it does not estimate hospital-level calibration or broad geographic generalization.

{documentation_transport_statement()}

## 4. Discussion

### 4.1 Principal findings

We evaluated a temporally regularized dynamic graph model for ordinal deterioration after spontaneous ICH using a fixed ICU-hour-24 landmark and two external transport cohorts. Temporal regularization changed development macro AUC by {dev_delta_graph:+.3f} relative to DG-GRU and by {dev_delta_summary:+.3f} relative to summary XGBoost. Transport changes versus DG-GRU were {m3_delta_graph:+.3f} in MIMIC-III and {ei_delta_graph:+.3f} in eICU. In this run, the highest out-of-fold point estimate for macro AUC was {oof_top_point_label} (macro AUC {oof_top_point_auc:.3f}); {top_model_statement}. {primary_effect_conclusion}

Calibration results provide complementary information to the discrimination estimates. Calibration slopes, intercepts, and their intervals are interpreted by cohort without assuming that either discrimination or calibration is the more informative result. A transported model cannot be recommended for absolute risk use without local recalibration, even when its ranking appears acceptable [14,15,18]. The decision curves were therefore used to display consequences of alternative thresholds, not to declare a preferred treatment strategy [15].

### 4.2 Relation to prior work

Prior studies of early deterioration after ICH have identified low GCS, hematoma expansion, intraventricular hemorrhage, blood pressure, and inflammatory markers as correlates, but definitions and observation windows differ [1-6]. Our endpoint focused on a routinely available ICU signal and an explicit 24-72-hour window. This design supports standardized landmark-based external evaluation, but it does not capture hematoma expansion or imaging phenotypes. The model should not be interpreted as a substitute for serial imaging or a neurological examination.

Dynamic graph and recurrent models have been applied to ICU mortality and intervention prediction [7-10]. Their reported gains often combine representation learning, static information, and cohort-specific preprocessing. Our controls were designed to expose this ambiguity. The highest out-of-fold point estimate among the compared models was {oof_top_point_label}. Time-reversed, time-permuted, and generic Laplace input-noise comparisons therefore describe whether the proposed temporal structure adds prediction rather than merely changing optimization behavior.

### 4.3 Interpretation of temporal regularization

Temporal regularization may act as an optimization constraint rather than a new source of predictive information. It can stabilize the latent trajectory, reduce abrupt changes, and make model behavior easier to describe. Those effects do not imply that the observed clinical course should be monotone. Neurological status can improve, fluctuate, or worsen in response to sedation, seizures, hydrocephalus, edema, and treatment transitions. A soft monotone prior therefore has to be judged by calibration and transport, not by the visual appearance of a smoother curve.

The direction, time-order, and input-noise controls are important in this context. If the proposed model and its negative controls produce similar discrimination, the penalty may act mainly as shrinkage. Generic input noise can offer a similarly useful perturbation benchmark without imposing temporal smoothness. Future work could learn patient-specific change points, incorporate explicit medication and ventilation states, or regularize a trajectory against clinically measured transitions rather than a global direction.

### 4.4 Limitations

Several limitations bound the findings. First, GCS is not a pure neurological measurement. Sedation, analgesia, neuromuscular blockade, intubation, aphasia, and care limitations can alter recorded values. The ETT audit quantified how flagged post-landmark observations differed across grades, and the endpoint does not eliminate this measurement dependence.

Second, the baseline was the maximum pre-landmark GCS. Some patients may already have deteriorated before hour 24, so the outcome describes severity observed in the 24-72-hour window rather than strictly incident first deterioration. The last-pre-landmark reference was one sensitivity analysis, but it cannot recover unmeasured trajectories.

Third, the study used three public databases with different documentation systems, coding eras, and case mix. MIMIC-III is a same-institution external cohort from a different database generation, so it captures database-generation and source shift but does not provide independent institutional validation. eICU was multi-hospital but small after exclusions, with {len(cohort_frames['eicu'])} analytic stays and {death['eicu']} deaths. Its confidence intervals are exploratory. {monitoring_ascertainment_statement()}

Fourth, the models used only variables available through the structured record. Hematoma volume, location, intraventricular extension, midline shift, edema, seizures, anticoagulation reversal, and treatment limitations were incomplete or absent. These variables may carry substantial prognostic information. {shared_static_discussion_sentence(n_static)} Fifth, temporal tuning used an 18-point grid and the remaining search was restricted to three protocol-defined candidate dictionaries per model family. The comparison estimates the behavior of a defined pipeline and does not exclude a better configuration outside those dictionaries. Predictor availability and variance screening used all development records without outcome labels; this minor transductive step could affect uncertainty. Sixth, no external recalibration or model updating was performed. Finally, the study did not evaluate workflow, clinician trust, downstream treatment, or patient outcomes. It is a model-development and transport study, not a clinical impact trial.

### 4.5 Implications and future work

For research on temporal clinical prediction, the results argue for reporting negative and control results alongside the primary model. Temporal regularization can be made to produce a smoother latent path, but smoothness and monotonicity should not be presented as evidence of predictive gain without calibrated comparisons. Future studies should define direction controls before analysis, report calibration across transport cohorts, and include measurement-state variables such as sedation and ventilation in the endpoint audit.

For ICH care, a model at ICU hour 24 could support structured review of patients whose GCS trajectory is falling. The model output would need local recalibration, verification of medication and ventilation status, and a prospective silent evaluation before any workflow use. It should not be used to limit care, allocate scarce resources, or replace repeated neurological assessment.

## 5. Conclusions

DG-GRU + TR was compared with five baselines, a GCS-slope-only comparator, unregularized DG-GRU, nested stacking, and three protocol-defined negative controls. {primary_effect_conclusion} External discrimination and calibration must therefore be interpreted together, and the measurement dependence of the GCS composite must be retained. External evaluation with measurement-state-aware endpoints and local recalibration is needed before clinical use.
"""

    declarations = f"""
## Declarations

**Ethics approval and consent to participate.** This study used de-identified, credentialed-access databases. Under the data-use agreements, no additional informed consent was required for the secondary analysis. The author's institution must confirm the applicable ethics determination before submission.

**Consent for publication.** Not applicable.

**Data availability.** MIMIC-IV v3.1, MIMIC-III v1.4, and eICU-CRD v2.0 are available from PhysioNet to credentialed users. Raw patient-level records are not redistributed. The public analysis package is limited to aggregate tables, model code, manifests, and figure-level aggregated data. It contains no patient-level predictions, stay_id values, subject_key values, hospital identifiers, or row-level source data. Patient-level prediction files are used only in the local audit workspace and are not redistributed.

**Code availability.** Analysis code, unit tests, and reproduction commands are included in the package. The workflow supports computational reproduction from the manifest stored at `results/run_manifest.json`. Exact numerical identity is not guaranteed because hardware, software versions, thread scheduling, and stochastic optimization can vary.

**Funding.** [AUTHOR_INPUT_NEEDED: supply funding source and funder role.]

**Competing interests.** [AUTHOR_INPUT_NEEDED: confirm the competing-interest declaration.]

**Author contributions.** [AUTHOR_INPUT_NEEDED: provide author initials and contributions.]

**Use of artificial intelligence.** An AI coding assistant supported code development, analysis documentation, and language editing. Analysis scripts and aggregate outputs were reviewed for internal consistency. No claim of bitwise deterministic reproduction is made. No patient-level record was submitted to an external generative service.

**Reporting guideline.** The study follows TRIPOD+AI. The completed 27-item checklist is supplied separately.

**Registration.** Not registered; this was a retrospective secondary analysis of existing databases.

**Patient and public involvement.** No patients or members of the public were involved in the design, conduct, or reporting of this analysis.
"""

    references = "## References\n\n" + "\n\n".join(
        f"{index}. {reference}"
        for index, reference in enumerate(REFERENCES, start=1)
    )

    table_text = f"""
[[LANDSCAPE]]
## Tables

**Table 1. Characteristics of the development and external cohorts.**

{markdown_table(tables['table1'])}

**Table 2. Ordinal outcome distribution across cohorts.**

{markdown_table(tables['table2_outcome'])}

[[PAGEBREAK]]
**Table 3. Discrimination and calibration for the five baselines, GCS-slope comparator, DG-GRU, DG-GRU + TR, nested stacking, and protocol-defined controls.**

{markdown_table(tables['table2'])}

[[PAGEBREAK]]
**Table 4. The eight outline sensitivity analyses.**

{markdown_table(tables['table4'])}

[[PAGEBREAK]]
**Table 5. Cross-fitted latent trajectory behavior.**

{markdown_table(tables['table5'])}
"""

    front = f"""
# {display_title}

**Running title:** {display_running_title}

**Article type:** Original research

**Authors and affiliations:** [AUTHOR_INPUT_NEEDED: author names, degrees, affiliations, ORCID identifiers, and corresponding-author contact details.]

**Keywords:** {display_keywords}

**Study version:** 2026-10-24

**Reporting guideline:** TRIPOD+AI
"""

    markdown = (
        front
        + "\n\n[[PAGEBREAK]]\n"
        + abstract
        + "\n\n"
        + results
        + "\n\n"
        + declarations
        + "\n\n"
        + references
        + "\n\n"
        + table_text
    )
    return markdown, tables


def set_run_font(
    run,
    *,
    size: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_rich_text(paragraph, text: str, size: float, bold_all: bool = False) -> None:
    text = text.replace("`", "")
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if not part:
            continue
        is_bold = part.startswith("**") and part.endswith("**")
        content = part[2:-2] if is_bold else part
        run = paragraph.add_run(content)
        set_run_font(run, size=size, bold=True if is_bold else bold_all)
        if "AUTHOR_INPUT_NEEDED" in content:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def configure_section(section, landscape: bool = False) -> None:
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    set_run_font(run, size=9)
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_end)


def add_footer(document, text: str = "") -> None:
    for section in document.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.text = ""
        if text:
            run = footer.add_run(text + "  |  ")
            set_run_font(run, size=8, italic=True)
        add_page_number(footer)


def set_cell_margins(cell, top: int = 30, start: int = 45, bottom: int = 30, end: int = 45) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in {
        "top": top,
        "start": start,
        "bottom": bottom,
        "end": end,
    }.items():
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_markdown_table(document, rows: list[list[str]]) -> None:
    if len(rows) < 2:
        return
    header = rows[0]
    body = rows[2:] if len(rows) > 2 else []
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat_header = OxmlElement("w:tblHeader")
    repeat_header.set(qn("w:val"), "true")
    header_properties.append(repeat_header)
    for cell, value in zip(table.rows[0].cells, header):
        cell.text = value
        shade_cell(cell, "D9E2F3")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                set_run_font(run, size=7.4, bold=True)
        set_cell_margins(cell)
    for row_values in body:
        row = table.add_row()
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        cells = row.cells
        for cell, value in zip(cells, row_values):
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    set_run_font(run, size=6.8)
            set_cell_margins(cell)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def render_markdown_document(
    document: Document,
    markdown: str,
    *,
    title_heading: bool = False,
) -> None:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped == "[[PAGEBREAK]]":
            document.add_page_break()
            index += 1
            continue
        if stripped == "[[LANDSCAPE]]":
            section = document.add_section(WD_SECTION.NEW_PAGE)
            configure_section(section, landscape=True)
            index += 1
            continue
        if stripped == "[[PORTRAIT]]":
            section = document.add_section(WD_SECTION.NEW_PAGE)
            configure_section(section, landscape=False)
            index += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            parsed = []
            for table_line in table_lines:
                cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                parsed.append(cells)
            add_markdown_table(document, parsed)
            continue
        if stripped.startswith("### "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(10)
            paragraph.paragraph_format.space_after = Pt(4)
            add_rich_text(paragraph, stripped[4:], 11.5, bold_all=True)
            index += 1
            continue
        if stripped.startswith("## "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(12)
            paragraph.paragraph_format.space_after = Pt(5)
            add_rich_text(paragraph, stripped[3:], 13, bold_all=True)
            index += 1
            continue
        if stripped.startswith("# "):
            paragraph = document.add_paragraph()
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if title_heading else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.space_after = Pt(18)
            add_rich_text(paragraph, stripped[2:], 17, bold_all=True)
            index += 1
            continue
        if stripped.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            paragraph.paragraph_format.space_after = Pt(3)
            add_rich_text(paragraph, stripped[2:], 10.5)
            index += 1
            continue
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.line_spacing = 2.0
        paragraph.paragraph_format.space_after = Pt(0)
        is_reference = bool(re.match(r"^\d+\. ", stripped))
        if is_reference:
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.first_line_indent = Inches(-0.25)
            paragraph.paragraph_format.space_after = Pt(3)
        add_rich_text(paragraph, stripped, 10.5)
        index += 1


def figure_label(spec: dict) -> str:
    return f"Figure {spec['number']}{spec['panel']}"


def missing_figure_outputs(results: Path | None = None) -> list[str]:
    root = RESULTS if results is None else Path(results)
    missing = []
    for stem in FIGURES:
        for suffix in (".png", ".pdf"):
            if not (root / "figures" / f"{stem}{suffix}").exists():
                missing.append(f"{stem}{suffix}")
    return missing


def require_figure_outputs(results: Path | None = None) -> None:
    missing = missing_figure_outputs(results)
    if missing:
        raise FileNotFoundError(
            "Every outline figure must exist in both PNG and PDF form; "
            "missing: " + ", ".join(missing)
        )


def add_figure_pages(document: Document) -> None:
    require_figure_outputs()
    section = document.add_section(WD_SECTION.NEW_PAGE)
    configure_section(section, landscape=True)
    heading = document.add_paragraph()
    heading.paragraph_format.space_after = Pt(10)
    add_rich_text(heading, "Figures", 14, bold_all=True)
    for index, spec in enumerate(FIGURE_SPECS):
        image_path = RESULTS / "figures" / f"{spec['stem']}.png"
        if index > 0:
            document.add_page_break()
        picture_paragraph = document.add_paragraph()
        picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture_paragraph.add_run().add_picture(str(image_path), width=Inches(9.4))
        caption_paragraph = document.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        caption_paragraph.paragraph_format.space_before = Pt(7)
        caption_paragraph.paragraph_format.line_spacing = 1.15
        add_rich_text(
            caption_paragraph,
            f"**{figure_label(spec)}. {spec['caption']}**",
            8.5,
        )


def write_main_docx(markdown: str) -> Path:
    document = Document()
    configure_section(document.sections[0], landscape=False)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(0)
    render_markdown_document(document, markdown, title_heading=True)
    add_figure_pages(document)
    metrics = load_json(RESULTS / "metrics.json")
    add_footer(document, str(title_context(metrics)["running_title"]))
    path = OUTPUTS / "manuscript.docx"
    document.save(path)
    return path


def supplement_markdown(tables: dict[str, pd.DataFrame]) -> str:
    config = load_config()
    manifest = load_json(RESULTS / "run_manifest.json")
    metrics = load_json(RESULTS / "metrics.json")
    display_title = str(title_context(metrics)["title"])
    n_seeds = seed_count(manifest)
    channels = configured_channels(config)
    subgroup = tables["subgroup"]
    subgroup = subgroup.rename(
        columns={
            "Subgroup": "Subgroup",
            "n": "n",
            "Any deterioration AUC": "Any-deterioration AUC",
            "95% CI low": "95% CI low",
            "95% CI high": "95% CI high",
        }
    )
    audit = measurement_audit_frame()
    channel_table = pd.DataFrame(
        [
            {
                "Channel": channel,
                "Definition": config["channel_labels"][channel],
            }
            for channel in channels
        ]
    )
    codes = pd.DataFrame(
        [
            {
                "Cohort": "MIMIC-IV",
                "Inclusion": "ICD-9 431/432 or ICD-10 I61/I62",
                "Exclusion": "None at the diagnosis-code step",
            },
            {
                "Cohort": "MIMIC-III",
                "Inclusion": "ICD-9 431/432",
                "Exclusion": "None at the diagnosis-code step",
            },
            {
                "Cohort": "eICU-CRD",
                "Inclusion": "ICD-9 431/432 or ICD-10 I61/I62",
                "Exclusion": "None at the diagnosis-code step",
            },
        ]
    )
    design = pd.DataFrame(
        [
            ("Prediction landmark", "ICU hour 24"),
            ("Sequence window", "ICU hours 0-23"),
            ("Outcome window", "ICU hours 24-71"),
            (
                "Outer validation",
                f"Patient-grouped stratified five-fold CV repeated "
                f"{manifest['cv_repeats']} times",
            ),
            (
                "Inner validation",
                "Patient-grouped stratified 20% holdout for early stopping",
            ),
            (
                "Inner-selection probability",
                "Mean validation probabilities across the five fixed model "
                "seeds before scoring candidate hyperparameters and lambda",
            ),
            (
                "Nested stacking",
                "Patient-grouped five-fold OOF within each outer training split",
            ),
            ("Seeds", f"{n_seeds} per outer model"),
            ("Maximum epochs", "25"),
            ("Early-stopping patience", "6 epochs"),
            (
                "Development median monotone weight",
                f"{float(manifest.get('cv_selected_lambdas', {}).get('dggru_tr', {}).get('median', manifest['lambda_monotone'])):.4g}",
            ),
            (
                "Development smoothness-to-monotone ratio",
                f"{float(manifest.get('lambda_smooth_ratio', 0.4)):.2f}",
            ),
            ("Bootstrap replicates", "2,000"),
            ("Development bootstrap cluster", "Patient"),
            ("MIMIC-III bootstrap cluster", "Patient"),
            ("eICU bootstrap cluster", "Hospital"),
            (
                "Primary paired inference",
                "2,000-replicate conditional cluster bootstrap; patients for "
                "MIMIC-IV/MIMIC-III and hospitals for eICU",
            ),
            (
                "Secondary paired test",
                "Unclustered standard DeLong on the first CV partition; "
                "secondary and not used as primary inference. The P024/P039 "
                "fixed primary Holm family is limited to the five baselines (xgb_static, "
                "xgb_summary, gru, grud, transformer) x any/severe, 10 tests; "
                "DG-GRU, GCS slope, stacking, and the three controls are "
                "exploratory_not_in_primary_holm_family",
            ),
            (
                "Matched ablations",
                "Four internal-development DG-GRU + TR ablations; not "
                "externally transported",
            ),
        ],
        columns=["Design element", "Specification"],
    )
    return f"""
# Supplementary Material

## {display_title}

### Supplementary Methods S1. Cohort definitions

{markdown_table(codes)}

### Supplementary Methods S2. Analysis settings

{markdown_table(design)}

### Supplementary Methods S3. Time-series channels

{markdown_table(channel_table)}

### Supplementary Table S1. Cohort flow

{markdown_table(tables['flow'])}

### Supplementary Table S2. Missingness in all model variables

{markdown_table(tables['missingness'])}

### Supplementary Table S3. Endpoint and measurement sensitivity analyses

{markdown_table(tables['table4'])}

### Supplementary Table S4. Measurement and outcome audit

{markdown_table(audit)}

### Supplementary Table S5. Development-cohort characteristics by outcome grade

{markdown_table(tables['table1b'])}

### Supplementary Table S6. Exploratory development-cohort subgroups

{markdown_table(subgroup)}

### Supplementary Table S7. Full model performance

{markdown_table(tables['table2'])}

### Supplementary Table S8. Development-fold temporal-regularization weights

{markdown_table(tables['table3_lambda'])}

### Supplementary Table S9. Top 15 TreeSHAP predictors

{markdown_table(tables['shap_top15'])}

### Supplementary Table S10. Time-dependent DG-GRU + TR importance

{markdown_table(tables['time_importance'])}

### Supplementary Table S11. Matched temporal-input ablations

{markdown_table(tables['temporal_ablations'])}

### Supplementary Table S12. Ordinal-grade calibration slopes and intercepts

{markdown_table(tables['grade_calibration'])}
"""


def write_supplement_docx(tables: dict[str, pd.DataFrame]) -> Path:
    document = Document()
    configure_section(document.sections[0], landscape=True)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(9)
    markdown = supplement_markdown(tables)
    render_markdown_document(document, markdown, title_heading=True)
    add_footer(document, "Supplementary material")
    path = OUTPUTS / "Supplementary_Material.docx"
    document.save(path)
    return path


def copy_deliverable_assets() -> None:
    figure_output = OUTPUTS / "figures"
    table_output = OUTPUTS / "tables"
    figure_output.mkdir(parents=True, exist_ok=True)
    table_output.mkdir(parents=True, exist_ok=True)
    for path in figure_output.iterdir():
        if path.is_file():
            path.unlink()
    for path in table_output.iterdir():
        if path.is_file():
            path.unlink()
    for path in (RESULTS / "figures").glob("figure*.*"):
        shutil.copy2(path, figure_output / path.name)
    for path in (RESULTS / "tables").glob("*"):
        if path.is_file():
            shutil.copy2(path, table_output / path.name)
    for name in ("metrics.json", "run_manifest.json"):
        source = RESULTS / name
        if source.exists():
            shutil.copy2(source, OUTPUTS / name)


def convert_documents_to_pdf(paths: list[Path]) -> list[Path]:
    from docx2pdf import convert

    outputs = []
    failures = []
    for path in paths:
        if not path.exists():
            failures.append(f"{path.name}: document does not exist")
            continue
        try:
            convert(str(path), str(path.with_suffix(".pdf")))
            outputs.append(path.with_suffix(".pdf"))
        except Exception as error:
            failures.append(f"{path.name}: {error}")
    if failures:
        raise RuntimeError(
            "PDF conversion failed; refusing to report a complete package: "
            + "; ".join(failures)
        )
    return outputs


def word_count(markdown: str) -> int:
    body = markdown.split("## References", 1)[0]
    body = re.sub(r"\[\[.*?\]\]", " ", body)
    body = re.sub(r"[|#*_`-]", " ", body)
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", body))


def audit_outputs(markdown: str, tables: dict[str, pd.DataFrame]) -> str:
    issues = []
    config = load_config()
    configured_channels(config)
    body = markdown.split("## References", 1)[0]
    cited: set[int] = set()
    for token in re.findall(r"\[([0-9,\-\s]+)\]", body):
        for part in token.split(","):
            part = part.strip()
            if "-" in part:
                low, high = [int(value) for value in part.split("-", 1)]
                cited.update(range(low, high + 1))
            elif part.isdigit():
                cited.add(int(part))
    for citation in range(1, len(REFERENCES) + 1):
        if citation not in cited:
            issues.append(f"Reference {citation} is listed but not cited.")
    issues.extend(
        f"Missing figure output for {name}."
        for name in missing_figure_outputs()
    )
    for name, frame in tables.items():
        if frame.empty:
            issues.append(f"Table {name} is empty.")
    metrics = load_json(RESULTS / "metrics.json")
    manifest = load_json(RESULTS / "run_manifest.json")
    outline_model_names(metric_block(metrics, "development"), manifest)
    if len(sensitivity_frame()) != len(OUTLINE_SENSITIVITY_SCENARIOS):
        issues.append(
            "Sensitivity output does not cover every outline variant."
        )
    observed_subgroups = " ".join(
        tables["subgroup"]["Subgroup"].astype(str)
    ).lower()
    for factor in OUTLINE_SUBGROUP_FACTORS:
        if factor.lower() not in observed_subgroups:
            issues.append(f"Missing outline subgroup factor: {factor}.")
    if len(tables["shap_top15"]) != 15:
        issues.append("TreeSHAP output does not contain 15 features.")
    if tables["time_importance"]["channel"].nunique() != 14:
        issues.append(
            "Time-dependent importance does not cover 14 channels."
        )
    post_landmark = {
        "outcome",
        "outcome_with_intubation",
        "outcome_gcs_death_only",
        "outcome_primary",
        "outcome_reconstructed",
        "outcome_recorded_intubation_composite",
        "death_within_72h",
        "death_hour_from_icu",
        "death_after_landmark",
        "unplanned_intubation_24_72h",
        "recorded_intubation_24_72h",
        "verified_unplanned_intubation_24_72h",
        "endpoint_intubation",
        "intubation_component_included",
        "intubation_composite_available",
        "primary_outcome_includes_intubation",
        "p020_status",
        "p020_complete",
        "recorded_intubation_composite_role",
        "recorded_intubation_composite_status",
        "recorded_intubation_composite_note",
        "max_sustained_drop",
        "last_gcs_hour",
        "icu_out_hour",
        "unitdischargeoffset",
        "unitdischargestatus",
        "hospitaldischargestatus",
        "hospitaldischargeoffset",
        "hospitaladmitoffset",
        "hospital_expire_flag",
        "primary_followup_complete",
        "complete_followup_only",
        "primary_included",
        "landmark_alive",
        "followup_window_complete",
        "followup_complete",
        "early_icu_transfer",
        "early_live_icu_transfer_before_72h",
        "analysis_weight",
        "analysis_weight_raw",
        "censoring_probability",
    }
    leaked = sorted(post_landmark.intersection(manifest.get("features", [])))
    if leaked:
        issues.append(
            "Post-landmark variables are present in the model manifest: "
            + ", ".join(leaked)
        )
    for cohort, key in (
        ("mimic4", "development"),
        ("mimic3", "mimic3"),
        ("eicu", "eicu"),
    ):
        frame = pd.read_csv(PROCESSED / cohort / "cohort.csv")
        block_n = int(metric_block(metrics, key)["dggru_tr"]["point"]["n"])
        if block_n != len(frame):
            issues.append(
                f"{cohort}: metric n={block_n} differs from cohort n={len(frame)}."
            )
    placeholder_lines = [
        line.strip()
        for line in markdown.splitlines()
        if "AUTHOR_INPUT_NEEDED" in line
    ]
    report = [
        "# Manuscript consistency audit",
        "",
        f"- Body word count before references: {word_count(markdown):,}",
        f"- Main figures: {len({spec['number'] for spec in FIGURE_SPECS})}",
        f"- Figure output files: {len(FIGURES)}",
        f"- Main and supplementary tables generated: {len(tables)}",
        f"- References: {len(REFERENCES)}",
        f"- Author-input placeholders: {len(placeholder_lines)}",
        "",
        "## Required author input",
        "",
    ]
    if placeholder_lines:
        report.extend(f"- {line}" for line in placeholder_lines)
    else:
        report.append("- None.")
    report.extend(["", "## Automated checks", ""])
    if issues:
        report.extend(f"- ISSUE: {issue}" for issue in issues)
    else:
        report.append("- No internal consistency issue detected.")
    report.extend(
        [
            "",
            "## Scientific boundaries",
            "",
            "- The endpoint is always named as GCS-based early neurological deterioration or death.",
            "- The pre-landmark maximum GCS is the primary reference.",
            "- GCS dependence on sedation, analgesia, neuromuscular blockade, and ventilation is stated.",
            "- eICU transport is labelled exploratory.",
            "- No clinical-impact or deployment claim is made.",
            "",
        ]
    )
    text = "\n".join(report)
    (OUTPUTS / "manuscript_audit.md").write_text(text, encoding="utf-8")
    if issues:
        raise RuntimeError(
            "Manuscript consistency audit failed; no complete reporting "
            f"package was written. Resolve the {len(issues)} issue(s) listed "
            "in outputs/manuscript_audit.md."
        )
    return text


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    markdown, tables = manuscript_markdown()
    audit_outputs(markdown, tables)
    (OUTPUTS / "manuscript.md").write_text(markdown, encoding="utf-8")
    copy_deliverable_assets()
    main_docx = write_main_docx(markdown)
    supplement_docx = write_supplement_docx(tables)
    pdfs = convert_documents_to_pdf([main_docx, supplement_docx])
    print(
        f"Saved manuscript ({word_count(markdown):,} words), "
        f"supplement, figure/table package, and {len(pdfs)} PDFs to {OUTPUTS}"
    )


if __name__ == "__main__":
    main()
