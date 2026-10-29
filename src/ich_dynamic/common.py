"""Shared constants and preprocessing helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get("ICH_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
OUTPUTS = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "configs" / "study.json"

DATASET_NAMES = ("mimic4", "mimic3", "eicu")
OUTCOME_END_HOURS = 72.0
ICU_DEATH_TIME_TOLERANCE_HOURS = 1.0 / 60.0
P020_STATUS = "partial"
P020_COMPLETE = False
CHANNELS = (
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "gcs_total",
    "hr",
    "rr",
    "spo2",
    "sbp",
    "dbp",
    "mbp",
    "temp",
    "urine_output",
    "fio2",
    "rass",
)


def death_in_icu_within_window(
    death_within_window: pd.Series,
    death_hour: pd.Series,
    icu_out_hour: pd.Series,
    *,
    end_hour: float = OUTCOME_END_HOURS,
    tolerance_hours: float = ICU_DEATH_TIME_TOLERANCE_HOURS,
) -> pd.Series:
    """Return deaths occurring in the ICU within the prespecified window.

    The primary endpoint counts any death within the window, including a death
    after a live ICU discharge. This helper answers a narrower question: did
    the patient die before ICU discharge, which is the condition for a
    completed ICU observation window. It is used for follow-up completeness and
    censoring, not for assigning the endpoint.
    """
    death = death_within_window.fillna(False).astype(bool)
    death_hour = pd.to_numeric(death_hour, errors="coerce")
    icu_out_hour = pd.to_numeric(icu_out_hour, errors="coerce")
    # CSV round-tripping can move an exactly one-minute discrepancy just above
    # the binary tolerance. Preserve the intended inclusive boundary.
    effective_tolerance = float(tolerance_hours) + 1e-9
    return (
        death
        & death_hour.notna()
        & icu_out_hour.notna()
        & death_hour.le(end_hour)
        & death_hour.le(icu_out_hour + effective_tolerance)
    )


def complete_icu_followup(
    outcome: pd.Series,
    icu_out_hour: pd.Series,
    death_in_icu: pd.Series,
    *,
    end_hour: float = OUTCOME_END_HOURS,
) -> pd.Series:
    """Return whether the ICU outcome window was observed to its end."""
    values = pd.to_numeric(outcome, errors="coerce")
    icu_out = pd.to_numeric(icu_out_hour, errors="coerce")
    return values.notna() & (icu_out.ge(end_hour) | death_in_icu.fillna(False))


def recorded_intubation_sensitivity_proxy(
    outcome: pd.Series,
    recorded_intubation: pd.Series,
) -> pd.Series:
    """Build an explicitly labelled recorded-intubation sensitivity proxy.

    This helper never asserts that a recorded intubation was unplanned. It only
    raises an otherwise observed ordinal outcome to grade 2 for sensitivity
    analysis when the ordinary intubation record is present.
    """
    primary = pd.to_numeric(outcome, errors="coerce")
    recorded = pd.to_numeric(recorded_intubation, errors="coerce")
    proxy = primary.where(~(primary.notna() & recorded.eq(1)), 2.0)
    return proxy.rename("outcome_recorded_intubation_composite")


LAB_ITEMS = {
    "glucose": (50809, 50931),
    "creatinine": (50912,),
    "sodium": (50983,),
    "potassium": (50971,),
    "bicarbonate": (50882,),
    "bun": (51006,),
    "wbc": (51301,),
    "hemoglobin": (51222,),
    "platelets": (51265,),
    "inr": (51237,),
    "lactate": (50813,),
    "calcium": (50893,),
    "albumin": (50862,),
    "total_bilirubin": (50885,),
    "alt": (50861,),
    "ast": (50878,),
    "alkaline_phosphatase": (50863,),
    "ptt": (51275,),
    "ph": (50820,),
    "pao2": (50821,),
}

EICU_LABS = {
    "glucose": ("glucose", "bedside glucose"),
    "creatinine": ("creatinine",),
    "sodium": ("sodium",),
    "potassium": ("potassium",),
    "bicarbonate": ("bicarbonate",),
    "bun": ("BUN",),
    "wbc": ("WBC x 1000",),
    "hemoglobin": ("Hgb",),
    "platelets": ("platelets x 1000", "platelets"),
    "inr": ("PT - INR",),
    "lactate": ("lactate",),
    "calcium": ("calcium",),
    "albumin": ("albumin",),
    "total_bilirubin": ("total bilirubin",),
    "alt": ("ALT (SGPT)", "ALT"),
    "ast": ("AST (SGOT)", "AST"),
    "alkaline_phosphatase": ("alkaline phosphatase",),
    "ptt": ("PTT",),
    "ph": ("pH",),
    "pao2": ("paO2",),
}

LAB_RANGES = {
    "glucose": (10, 1000),
    "creatinine": (0.1, 30),
    "sodium": (80, 200),
    "potassium": (1, 15),
    "bicarbonate": (1, 80),
    "bun": (1, 300),
    "wbc": (0, 300),
    "hemoglobin": (1, 30),
    "platelets": (0, 2000),
    "inr": (0.1, 30),
    "lactate": (0.1, 50),
    "calcium": (1, 30),
    "albumin": (0.5, 8),
    "total_bilirubin": (0.1, 80),
    "alt": (1, 20000),
    "ast": (1, 20000),
    "alkaline_phosphatase": (1, 5000),
    "ptt": (5, 300),
    "ph": (6.5, 7.9),
    "pao2": (20, 800),
}


def validate_study_config(config: dict) -> dict:
    """Reject duplicated settings that could silently diverge at runtime."""
    if not isinstance(config, dict):
        raise ValueError("Study configuration must be a JSON object")
    temporal = (
        config.get("hyperparameters", {})
        .get("temporal_regularization", {})
    )
    if temporal:
        canonical_grid = tuple(config.get("lambda_grid", ()))
        nested_grid = tuple(temporal.get("lambda_monotone", ()))
        if canonical_grid != nested_grid:
            raise ValueError(
                "lambda_grid must exactly match "
                "hyperparameters.temporal_regularization.lambda_monotone"
            )

        canonical_ratio = config.get("lambda_smooth_ratio")
        nested_ratios = tuple(temporal.get("smooth_to_monotone_ratio", ()))
        if nested_ratios:
            if len(nested_ratios) != 1:
                raise ValueError(
                    "smooth_to_monotone_ratio must contain exactly one value"
                )
            if float(canonical_ratio) != float(nested_ratios[0]):
                raise ValueError(
                    "lambda_smooth_ratio must exactly match "
                    "hyperparameters.temporal_regularization."
                    "smooth_to_monotone_ratio"
                )

    stacking = config.get("hyperparameters", {}).get("stacking", {})
    if stacking:
        for configuration_key, nested_key in (
            ("stacking_c", "regularization_c"),
            ("nested_inner_folds", "inner_folds"),
        ):
            nested_values = tuple(stacking.get(nested_key, ()))
            if not nested_values:
                continue
            if len(nested_values) != 1:
                raise ValueError(f"{nested_key} must contain exactly one value")
            canonical = config.get(configuration_key)
            if canonical is None:
                raise ValueError(
                    f"{configuration_key} is required when {nested_key} is set"
                )
            if float(canonical) != float(nested_values[0]):
                raise ValueError(
                    f"{configuration_key} must exactly match "
                    f"hyperparameters.stacking.{nested_key}"
                )
    return config


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return validate_study_config(json.load(fh))


def ensure_dirs() -> None:
    for path in (PROCESSED, RESULTS, OUTPUTS, RESULTS / "figures", RESULTS / "predictions"):
        path.mkdir(parents=True, exist_ok=True)


def clamp_series(values: pd.Series, low: float, high: float) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    return out.where((out >= low) & (out <= high))


def combine_gcs_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a de-duplicated GCS total from same-time component rows.

    Expected component item IDs are eye=220739, verbal=223900, motor=223901.
    A component sum is accepted only when all three components share a timestamp.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "gcs", "ett_verbal"]
        )
    components = frame[frame["itemid"].isin([220739, 223900, 223901])].copy()
    totals = frame[frame["itemid"] == 198].copy()
    if components.empty:
        out = totals.rename(columns={"valuenum": "gcs"}).copy()
        out["ett_verbal"] = 0.0
    else:
        components["value_num"] = pd.to_numeric(components["valuenum"], errors="coerce")
        ranges = {220739: (1, 4), 223900: (1, 5), 223901: (1, 6)}
        for item, (low, high) in ranges.items():
            mask = components["itemid"] == item
            components.loc[mask, "value_num"] = components.loc[mask, "value_num"].where(
                components.loc[mask, "value_num"].between(low, high)
            )
        pivot = components.pivot_table(
            index=["stay_id", "charttime"],
            columns="itemid",
            values="value_num",
            aggfunc="median",
        )
        pivot = pivot.reindex(columns=[220739, 223900, 223901])
        complete = pivot.notna().all(axis=1)
        comp = pivot[complete].sum(axis=1).rename("gcs").reset_index()
        verbal_text = components[components["itemid"] == 223900].copy()
        verbal_text["ett_verbal"] = (
            verbal_text["value"]
            .fillna("")
            .astype(str)
            .str.contains(r"ET|ETT|trach|intub", case=False, regex=True)
            .astype(float)
        )
        ett = verbal_text.groupby(["stay_id", "charttime"], as_index=False)[
            "ett_verbal"
        ].max()
        comp = comp.merge(ett, on=["stay_id", "charttime"], how="left")
        if not totals.empty:
            total = totals.rename(columns={"valuenum": "gcs"})[
                ["stay_id", "charttime", "gcs"]
            ].copy()
            total["ett_verbal"] = np.nan
            comp = pd.concat([comp, total], ignore_index=True)
            comp = (
                comp.sort_values(["stay_id", "charttime"])
                .groupby(["stay_id", "charttime"], as_index=False)
                .agg({"gcs": "last", "ett_verbal": "max"})
            )
        out = comp
    out["gcs"] = pd.to_numeric(out["gcs"], errors="coerce")
    out = out[out["gcs"].between(3, 15)].copy()
    out["ett_verbal"] = out["ett_verbal"].fillna(0.0)
    return out[["stay_id", "charttime", "gcs", "ett_verbal"]]


def hourly_medians(
    events: pd.DataFrame,
    value_col: str = "value",
    time_col: str = "time_hour",
) -> pd.DataFrame:
    """Return stay_id x integer hour medians for one or more channels."""
    if events.empty:
        return pd.DataFrame(columns=["stay_id", "hour", value_col])
    work = events.dropna(subset=[time_col, value_col]).copy()
    work["hour"] = np.floor(pd.to_numeric(work[time_col], errors="coerce")).astype("Int64")
    work = work[work["hour"].between(0, 71)]
    out = (
        work.groupby(["stay_id", "hour"], as_index=False)[value_col]
        .median()
        .sort_values(["stay_id", "hour"])
    )
    return out


def build_sequence_tensor(
    hourly_long: pd.DataFrame,
    stay_ids: Iterable[int],
    channels: Iterable[str] = CHANNELS,
    max_hour: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build (N, 24, channels) arrays and channel-level observation masks."""
    channels = list(channels)
    ids = [int(x) for x in stay_ids]
    values = np.full((len(ids), max_hour, len(channels)), np.nan, dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.float32)
    id_pos = {stay_id: i for i, stay_id in enumerate(ids)}
    work = hourly_long[
        hourly_long["channel"].isin(channels)
        & hourly_long["hour"].between(0, max_hour - 1)
    ].copy()
    if not work.empty:
        work = (
            work.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
            .median()
            .sort_values(["stay_id", "hour", "channel"])
        )
    for row in work.itertuples(index=False):
        i = id_pos.get(int(row.stay_id))
        if i is None:
            continue
        j = channels.index(row.channel)
        values[i, int(row.hour), j] = float(row.value)
        masks[i, int(row.hour), j] = 1.0
    return values, masks, ids


def sequence_summary(
    values: np.ndarray,
    masks: np.ndarray,
    stay_ids: Iterable[int],
    channels: Iterable[str] = CHANNELS,
) -> pd.DataFrame:
    """Summarize first 24 hours of sequence data for tabular baselines."""
    rows: list[dict[str, float | int]] = []
    channels = list(channels)
    for i, stay_id in enumerate(stay_ids):
        rec: dict[str, float | int] = {"stay_id": int(stay_id)}
        for j, channel in enumerate(channels):
            x = values[i, :, j]
            observed = (masks[i, :, j] > 0) & np.isfinite(x)
            vals = x[observed]
            rec[f"{channel}_n"] = int(observed.sum())
            if len(vals) == 0:
                rec[f"{channel}_mean"] = np.nan
                rec[f"{channel}_min"] = np.nan
                rec[f"{channel}_max"] = np.nan
                rec[f"{channel}_sd"] = np.nan
                rec[f"{channel}_slope"] = np.nan
                continue
            rec[f"{channel}_mean"] = float(np.mean(vals))
            rec[f"{channel}_min"] = float(np.min(vals))
            rec[f"{channel}_max"] = float(np.max(vals))
            rec[f"{channel}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if len(vals) > 1:
                hours = np.flatnonzero(observed).astype(float)
                rec[f"{channel}_slope"] = float(np.polyfit(hours, vals, 1)[0])
            else:
                rec[f"{channel}_slope"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_gcs_end(
    gcs: pd.DataFrame,
    baseline: float,
    *,
    start_hour: float,
    end_hour: float = 72.0,
    death_within_window: bool = False,
    verified_unplanned_intubation: bool = False,
    unplanned_intubation: bool | None = None,
    min_followup_hour: float | None = None,
    min_post_measurements: int = 2,
    confirmation_consecutive_hours: int = 2,
) -> dict[str, float | int]:
    """Classify GCS deterioration in a fixed landmark-to-endpoint window.

    GCS observations are first reduced to hourly medians. The primary rule
    requires two adjacent hourly bins whose medians are both at least two
    points below baseline. Missing intervening hours therefore break
    persistence. Severity is the largest reduction observed within a qualifying
    consecutive run. Stays with fewer than two valid post-landmark GCS hours
    remain ineligible. Among eligible stays, death or a verified unplanned
    intubation is grade 2 regardless of the observed GCS reduction.
    The ``unplanned_intubation`` keyword is retained only as a compatibility
    alias for an explicitly verified event; ordinary recorded intubation must
    not be passed here.
    """
    verified_event = bool(verified_unplanned_intubation)
    if unplanned_intubation is not None:
        verified_event = verified_event or bool(unplanned_intubation)
    if confirmation_consecutive_hours < 1:
        raise ValueError("confirmation_consecutive_hours must be at least 1")
    if start_hour >= end_hour:
        raise ValueError("start_hour must be earlier than end_hour")
    post = gcs[
        (gcs["time_hour"] >= start_hour) & (gcs["time_hour"] < end_hour)
    ].copy()
    post = post.dropna(subset=["gcs"]).sort_values("time_hour")
    if not np.isfinite(baseline):
        return {"outcome": np.nan, "max_sustained_drop": np.nan, "last_gcs_hour": np.nan}
    if not post.empty:
        post["hour_bin"] = np.floor(
            pd.to_numeric(post["time_hour"], errors="coerce")
        )
        post = (
            post.dropna(subset=["hour_bin"])
            .groupby("hour_bin", as_index=False)
            .agg(time_hour=("time_hour", "median"), gcs=("gcs", "median"))
            .sort_values("time_hour")
        )
    last_hour = (
        float(post["time_hour"].max()) if not post.empty else np.nan
    )
    # The prespecified cohort requires at least two valid post-landmark GCS
    # hours before any outcome can be assigned. Death is a severity override
    # for eligible stays, not an exception to the cohort eligibility rule.
    if len(post) < min_post_measurements:
        return {
            "outcome": np.nan,
            "max_sustained_drop": np.nan,
            "last_gcs_hour": last_hour,
        }
    if death_within_window or verified_event:
        observed_drop = (
            float(baseline - post["gcs"].min()) if not post.empty else np.nan
        )
        return {
            "outcome": 2,
            "max_sustained_drop": observed_drop,
            "last_gcs_hour": last_hour,
        }
    if min_followup_hour is not None and (
        not np.isfinite(last_hour) or last_hour < min_followup_hour
    ):
        return {
            "outcome": np.nan,
            "max_sustained_drop": np.nan,
            "last_gcs_hour": last_hour,
        }

    low = post[post["gcs"] <= (baseline - 2)].copy()
    max_drop = 0.0
    if not low.empty:
        run_id = low["hour_bin"].diff().ne(1).cumsum()
        for _, run in low.groupby(run_id, sort=False):
            if len(run) < confirmation_consecutive_hours:
                continue
            run_drop = float(baseline - run["gcs"].min())
            max_drop = max(max_drop, run_drop)
    if max_drop >= 4:
        grade = 2
    elif max_drop >= 2:
        grade = 1
    else:
        grade = 0
    return {
        "outcome": int(grade),
        "max_sustained_drop": max_drop,
        "last_gcs_hour": last_hour,
    }


def classify_end(
    gcs: pd.DataFrame,
    baseline: float,
    death_within_72h: bool,
    verified_unplanned_intubation: bool = False,
    unplanned_intubation: bool | None = None,
    min_followup_hour: float | None = None,
    min_post_measurements: int = 2,
    confirmation_consecutive_hours: int = 2,
) -> dict[str, float | int]:
    """Classify the primary 24-to-72-hour ordinal endpoint."""
    return classify_gcs_end(
        gcs,
        baseline,
        start_hour=24,
        end_hour=72,
        death_within_window=death_within_72h,
        verified_unplanned_intubation=verified_unplanned_intubation,
        unplanned_intubation=unplanned_intubation,
        min_followup_hour=min_followup_hour,
        min_post_measurements=min_post_measurements,
        confirmation_consecutive_hours=confirmation_consecutive_hours,
    )


def add_outcome(
    cohort: pd.DataFrame,
    gcs_events: pd.DataFrame,
    baseline_mode: str = "max",
    confirmation_consecutive_hours: int = 2,
    include_death: bool = True,
    exclude_death: bool = False,
    include_intubation: bool = False,
    verified_unplanned_intubation_column: str | None = None,
) -> pd.DataFrame:
    """Attach baseline, follow-up, and three-level primary outcome columns.

    Ordinary recorded intubation is deliberately excluded from the primary
    endpoint. ``include_intubation=True`` only has an effect when
    ``verified_unplanned_intubation_column`` names an explicit verified-event
    column supplied by the caller.
    """
    if (
        include_intubation
        and verified_unplanned_intubation_column is not None
        and verified_unplanned_intubation_column not in cohort.columns
    ):
        raise KeyError(
            "verified unplanned-intubation column requested but absent: "
            f"{verified_unplanned_intubation_column}"
        )
    stale_columns = [
        "outcome",
        "baseline_gcs_max",
        "baseline_gcs_last",
        "n_gcs_pre",
        "max_sustained_drop",
        "last_gcs_hour",
    ]
    cohort = cohort.drop(columns=stale_columns, errors="ignore").copy()
    rows = []
    by_stay = {int(k): v for k, v in gcs_events.groupby("stay_id")}
    for row in cohort.itertuples(index=False):
        g = by_stay.get(int(row.stay_id), gcs_events.iloc[0:0])
        pre = g[(g["time_hour"] >= 0) & (g["time_hour"] < 24)].dropna(subset=["gcs"])
        baseline_max = float(pre["gcs"].max()) if not pre.empty else np.nan
        baseline_last = (
            float(pre.sort_values("time_hour")["gcs"].iloc[-1]) if not pre.empty else np.nan
        )
        baseline = baseline_max if baseline_mode == "max" else baseline_last
        death = bool(getattr(row, "death_within_72h", False)) and include_death
        verified_intubation = False
        if include_intubation and verified_unplanned_intubation_column:
            verified_value = getattr(
                row, verified_unplanned_intubation_column, np.nan
            )
            verified_intubation = (
                pd.notna(verified_value)
                and float(
                    pd.to_numeric(
                        pd.Series([verified_value]), errors="coerce"
                    ).iloc[0]
                )
                == 1.0
            )
        label = classify_end(
            g,
            baseline,
            death_within_72h=death,
            verified_unplanned_intubation=bool(verified_intubation),
            min_followup_hour=None,
            min_post_measurements=2,
            confirmation_consecutive_hours=confirmation_consecutive_hours,
        )
        if exclude_death:
            label["outcome"] = (
                np.nan if bool(getattr(row, "death_within_72h", False)) else label["outcome"]
            )
        rows.append(
            {
                "stay_id": int(row.stay_id),
                "baseline_gcs_max": baseline_max,
                "baseline_gcs_last": baseline_last,
                "n_gcs_pre": int(len(pre)),
                **label,
            }
        )
    return cohort.merge(pd.DataFrame(rows), on="stay_id", how="left")


def lab_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-24-hour laboratory values with common feature names."""
    if long.empty:
        return pd.DataFrame(columns=["stay_id"])
    work = long.dropna(subset=["value"]).copy()
    grouped = (
        work.groupby(["stay_id", "lab"], as_index=False)
        .agg(
            first=("value", "first"),
            last=("value", "last"),
            min=("value", "min"),
            max=("value", "max"),
            mean=("value", "mean"),
        )
        .sort_values(["stay_id", "lab"])
    )
    wide = grouped.pivot(index="stay_id", columns="lab")
    wide.columns = [f"lab_{lab}_{stat}" for stat, lab in wide.columns]
    return wide.reset_index()


def save_dataset(
    name: str,
    cohort: pd.DataFrame,
    gcs_events: pd.DataFrame,
    hourly: pd.DataFrame,
    flow: pd.DataFrame,
    config: dict,
) -> None:
    outdir = PROCESSED / name
    outdir.mkdir(parents=True, exist_ok=True)
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)
    cohort.to_csv(outdir / "cohort.csv", index=False)
    gcs_events.to_csv(outdir / "gcs_events.csv", index=False)
    hourly.to_csv(outdir / "hourly.csv", index=False)
    flow.to_csv(outdir / "cohort_flow.csv", index=False)
    values, masks, ids = build_sequence_tensor(
        hourly,
        cohort["stay_id"],
        channels=config["channels"],
        max_hour=config["sequence_hours"],
    )
    np.savez_compressed(
        outdir / "sequences.npz",
        values=values,
        masks=masks,
        stay_id=np.asarray(ids, dtype=np.int64),
        channels=np.asarray(config["channels"]),
    )
    print(
        f"[{name}] saved cohort={len(cohort)}, sequences={values.shape}, "
        f"outcome counts={cohort['outcome'].value_counts(dropna=False).to_dict()}"
    )
