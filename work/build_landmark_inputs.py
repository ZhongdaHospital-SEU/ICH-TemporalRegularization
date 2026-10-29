"""Build leakage-safe model inputs at the 6, 24, and 48 hour landmarks.

The builder uses only the processed artifacts already created by
``src.ich_dynamic.build_cohorts``. It deliberately does not preprocess, impute,
or forward fill. First-24-hour summaries are withheld at 6 h because they
would contain future data; at 24 h and 48 h they are valid pre-landmark
covariates.

The 24-hour output is reconciled against the existing processed cohort and
sequence tensor. A reconciliation failure is treated as a build failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.ich_dynamic.common import (
    CHANNELS,
    PROCESSED,
    build_sequence_tensor,
    classify_gcs_end,
    death_in_icu_within_window,
    load_config,
    recorded_intubation_sensitivity_proxy,
    sequence_summary,
)


SUPPORTED_LANDMARKS = (6, 24, 48)
OUTCOME_WINDOW_END_HOURS = 72
MINIMUM_POST_LANDMARK_GCS = 2
SCHEMA_VERSION = 1

BASE_FEATURE_COLUMNS = (
    "age",
    "age_gt89",
    "sex_male",
    "race_white",
    "emergency_admission",
)
RASS_CONTEXT_COLUMNS = (
    "min_rass_24h",
    "any_rass_le_minus2_24h",
    "any_deep_sedation_24h",
    "deep_sedation_observed_24h",
)
PRE_24H_CONTEXT_COLUMNS = (
    "comorb_hypertension",
    "comorb_diabetes",
    "comorb_ckd",
    "comorb_liver_disease",
    "comorb_atrial_fibrillation",
    "comorb_coronary_disease",
    "comorb_heart_failure",
    "comorb_copd",
    "coagulopathy_diagnosis",
    "neurosurgery_admission",
    "mechanical_ventilation_24h",
    "neurosurgery_24h",
    "sedation_exposure_24h",
    "anticoagulant_use_24h",
    "anticoagulation_related_hemorrhage",
)
IDENTIFIER_COLUMNS = ("stay_id", "subject_key", "hospital_key")
OUTCOME_COLUMNS = (
    "outcome",
    "outcome_recorded_intubation_composite",
    "outcome_gcs_death_only",
    "recorded_intubation_24_72h",
    "verified_unplanned_intubation_24_72h",
    "recorded_intubation_composite_role",
    "recorded_intubation_composite_status",
    "p020_status",
    "p020_complete",
    "max_sustained_drop",
    "last_gcs_hour",
    "death_within_72h",
    "death_after_landmark",
    "endpoint_intubation",
    "intubation_component_included",
    "landmark_alive",
    "followup_window_complete",
)


class LandmarkInputError(ValueError):
    """Raised when processed inputs cannot support a valid landmark build."""


class LandmarkReconciliationError(RuntimeError):
    """Raised when the reconstructed 24-hour inputs do not match processed data."""


@dataclass(frozen=True)
class LandmarkInputs:
    """One landmark-specific cohort, sequence tensor, and audit record."""

    dataset: str
    landmark_hours: int
    cohort: pd.DataFrame
    values: np.ndarray
    masks: np.ndarray
    stay_ids: np.ndarray
    audit: dict[str, Any]


def _json_ready(value: Any) -> Any:
    """Convert numpy and non-finite values into strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce CSV boolean values without treating missing as true."""

    def convert(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value != 0)
        return str(value).strip().lower() in {"true", "t", "yes", "y", "1"}

    return series.map(convert).astype(bool)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise LandmarkInputError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def _read_sources(processed_root: Path, dataset: str) -> dict[str, Any]:
    root = processed_root / dataset
    required = {
        "cohort_pre_filter": root / "cohort_pre_filter.csv",
        "gcs_events": root / "gcs_events.csv",
        "hourly": root / "hourly.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed landmark inputs for {dataset}: {', '.join(missing)}"
        )
    sources: dict[str, Any] = {
        name: pd.read_csv(path)
        for name, path in required.items()
    }
    sources["paths"] = required
    reference_paths = {
        "cohort": root / "cohort.csv",
        "sequences": root / "sequences.npz",
    }
    sources["reference_paths"] = reference_paths
    _require_columns(
        sources["cohort_pre_filter"],
        (
            "stay_id",
            "death_within_72h",
            "death_hour_from_icu",
            "icu_out_hour",
        ),
        "cohort_pre_filter.csv",
    )
    _require_columns(
        sources["gcs_events"],
        ("stay_id", "time_hour", "gcs"),
        "gcs_events.csv",
    )
    _require_columns(
        sources["hourly"],
        ("stay_id", "hour", "channel", "value"),
        "hourly.csv",
    )
    return sources


def _normalize_channels(channels: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(channels or CHANNELS)
    if len(selected) != 14 or len(set(selected)) != 14:
        raise LandmarkInputError(
            "Landmark sequence contract requires exactly 14 unique channels."
        )
    return selected


def _baseline_table(
    cohort: pd.DataFrame,
    gcs_events: pd.DataFrame,
    landmark_hours: int,
) -> pd.DataFrame:
    """Compute the maximum and last GCS strictly before the landmark."""
    events = gcs_events[["stay_id", "time_hour", "gcs"]].copy()
    events["stay_id"] = pd.to_numeric(events["stay_id"], errors="coerce")
    events["time_hour"] = pd.to_numeric(events["time_hour"], errors="coerce")
    events["gcs"] = pd.to_numeric(events["gcs"], errors="coerce")
    events = events.dropna(subset=["stay_id", "time_hour", "gcs"])
    events = events[
        events["time_hour"].ge(0)
        & events["time_hour"].lt(landmark_hours)
        & events["gcs"].between(3, 15)
    ].copy()
    events["stay_id"] = events["stay_id"].astype(np.int64)
    events = events.sort_values(["stay_id", "time_hour"])

    rows: list[dict[str, Any]] = []
    for stay_id, group in events.groupby("stay_id", sort=False):
        values = group["gcs"].to_numpy(dtype=float)
        rows.append(
            {
                "stay_id": int(stay_id),
                "baseline_gcs_max": float(np.max(values)),
                "baseline_gcs_last": float(values[-1]),
                "n_gcs_pre": int(len(values)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "baseline_gcs_max",
                "baseline_gcs_last",
                "n_gcs_pre",
            ]
        )
    return pd.DataFrame(rows)


def classify_landmark_end(
    gcs: pd.DataFrame,
    baseline: float,
    landmark_hours: int,
    *,
    death_after_landmark: bool,
    unplanned_intubation: bool = False,
    minimum_post_measurements: int = MINIMUM_POST_LANDMARK_GCS,
) -> dict[str, float | int]:
    """Apply the study's ordinal GCS rule to an arbitrary landmark window.

    The implementation mirrors ``common.classify_end`` for the 24-hour
    landmark, but parameterizes the lower window bound. Only observations in
    ``[landmark_hours, 72)`` contribute to the outcome.
    """
    if landmark_hours not in SUPPORTED_LANDMARKS:
        raise LandmarkInputError(
            f"Unsupported landmark {landmark_hours}; expected one of "
            f"{SUPPORTED_LANDMARKS}."
        )
    return classify_gcs_end(
        gcs,
        baseline,
        start_hour=landmark_hours,
        end_hour=OUTCOME_WINDOW_END_HOURS,
        death_within_window=death_after_landmark,
        unplanned_intubation=unplanned_intubation,
        min_post_measurements=minimum_post_measurements,
        confirmation_consecutive_hours=2,
    )


def _intubation_support(
    candidate: pd.DataFrame,
    landmark_hours: int,
    dataset: str,
) -> tuple[pd.Series, pd.Series, bool, str]:
    recorded_column = "recorded_intubation_24_72h"
    verified_column = "verified_unplanned_intubation_24_72h"
    recorded = (
        pd.to_numeric(candidate[recorded_column], errors="coerce")
        if recorded_column in candidate
        else pd.Series(np.nan, index=candidate.index, dtype=float)
    )
    verified = (
        pd.to_numeric(candidate[verified_column], errors="coerce")
        if verified_column in candidate
        else pd.Series(np.nan, index=candidate.index, dtype=float)
    )
    if landmark_hours != 24:
        return (
            recorded,
            pd.Series(False, index=candidate.index),
            False,
            (
                "event timestamps are unavailable; the stored 24-72 h flag "
                "cannot establish an unplanned-intubation component for this "
                "landmark without temporal leakage or an incomplete endpoint"
            ),
        )
    if verified.notna().any():
        return (
            recorded,
            verified.eq(1),
            True,
            "an explicit verified unplanned-intubation field is available",
        )
    if not recorded.notna().any():
        return (
            recorded,
            pd.Series(False, index=candidate.index),
            False,
            "no recorded or verified post-landmark intubation field is available",
        )
    return (
        recorded,
        pd.Series(False, index=candidate.index),
        False,
        (
            f"{dataset} contains an ordinary recorded-intubation field but no "
            "planned versus unplanned indication; it is retained only for an "
            "explicitly labelled sensitivity proxy"
        ),
    )


def _attach_outcomes(
    candidate: pd.DataFrame,
    gcs_events: pd.DataFrame,
    baseline: pd.DataFrame,
    landmark_hours: int,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stale = [
        "baseline_gcs_max",
        "baseline_gcs_last",
        "n_gcs_pre",
        "outcome",
        "outcome_recorded_intubation_composite",
        "outcome_gcs_death_only",
        "max_sustained_drop",
        "last_gcs_hour",
    ]
    work = candidate.drop(columns=stale, errors="ignore").merge(
        baseline, on="stay_id", how="left"
    )
    work["death_within_72h"] = _as_bool(work["death_within_72h"])
    work["death_hour_from_icu"] = pd.to_numeric(
        work["death_hour_from_icu"], errors="coerce"
    )
    work["icu_out_hour"] = pd.to_numeric(work["icu_out_hour"], errors="coerce")
    work["death_in_icu_within_72h"] = death_in_icu_within_window(
        work["death_within_72h"],
        work["death_hour_from_icu"],
        work["icu_out_hour"],
    )
    # Admission-level death flag: the cohort builder's endpoint treats any
    # death within 72 h of ICU admission as grade 2, including deaths that
    # occurred after a live ICU discharge. Restricting the flag to deaths
    # observed inside the ICU would give a second, incompatible definition.
    work["followup_window_complete"] = (
        work["death_in_icu_within_72h"]
        | work["icu_out_hour"].ge(OUTCOME_WINDOW_END_HOURS)
    )
    work["in_icu_at_landmark"] = work["icu_out_hour"].ge(landmark_hours)
    work["death_after_landmark"] = work[
        "death_within_72h"
    ] & work["death_hour_from_icu"].gt(landmark_hours)
    work["landmark_alive"] = (
        ~work["death_within_72h"]
        | work["death_hour_from_icu"].gt(landmark_hours)
    )

    (
        recorded_intubation,
        verified_intubation,
        intubation_supported,
        intubation_reason,
    ) = _intubation_support(work, landmark_hours, dataset)
    work["recorded_intubation_24_72h"] = recorded_intubation
    work["verified_unplanned_intubation_24_72h"] = np.where(
        intubation_supported, verified_intubation.astype(float), np.nan
    )
    work["endpoint_intubation"] = work["verified_unplanned_intubation_24_72h"]
    work["intubation_component_included"] = intubation_supported
    work["intubation_composite_available"] = intubation_supported
    work["primary_outcome_includes_intubation"] = intubation_supported
    work["p020_status"] = "complete" if intubation_supported else "partial"
    work["p020_complete"] = intubation_supported
    work["recorded_intubation_composite_role"] = "sensitivity_proxy_only"
    work["recorded_intubation_composite_status"] = np.where(
        recorded_intubation.notna(),
        "ordinary_recorded_intubation_available",
        "unavailable_not_verified_unplanned",
    )

    gcs = gcs_events[["stay_id", "time_hour", "gcs"]].copy()
    gcs["stay_id"] = pd.to_numeric(gcs["stay_id"], errors="coerce")
    gcs["time_hour"] = pd.to_numeric(gcs["time_hour"], errors="coerce")
    gcs["gcs"] = pd.to_numeric(gcs["gcs"], errors="coerce")
    gcs = gcs.dropna(subset=["stay_id", "time_hour", "gcs"]).copy()
    gcs["stay_id"] = gcs["stay_id"].astype(np.int64)
    by_stay = {int(key): group for key, group in gcs.groupby("stay_id", sort=False)}

    rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        stay_id = int(row.stay_id)
        events = by_stay.get(stay_id, gcs.iloc[0:0])
        death_after_landmark = bool(row.death_after_landmark)
        gcs_death = classify_landmark_end(
            events,
            float(row.baseline_gcs_max)
            if pd.notna(row.baseline_gcs_max)
            else np.nan,
            landmark_hours,
            death_after_landmark=death_after_landmark,
            unplanned_intubation=False,
        )
        endpoint = classify_landmark_end(
            events,
            float(row.baseline_gcs_max)
            if pd.notna(row.baseline_gcs_max)
            else np.nan,
            landmark_hours,
            death_after_landmark=death_after_landmark,
            unplanned_intubation=bool(
                intubation_supported and bool(row.endpoint_intubation == 1)
            ),
        )
        proxy_outcome = (
            float(recorded_intubation_sensitivity_proxy(
                pd.Series([gcs_death["outcome"]]),
                pd.Series([row.recorded_intubation_24_72h]),
            ).iloc[0])
            if pd.notna(gcs_death["outcome"])
            else np.nan
        )
        rows.append(
            {
                "stay_id": stay_id,
                **gcs_death,
                "outcome": endpoint["outcome"],
                "outcome_gcs_death_only": gcs_death["outcome"],
                "outcome_recorded_intubation_composite": proxy_outcome,
            }
        )
    outcomes = pd.DataFrame(rows)
    work = work.merge(outcomes, on="stay_id", how="left")
    details = {
        "unplanned_intubation_supported": bool(intubation_supported),
        "unplanned_intubation_reason": intubation_reason,
        "recorded_intubation_available": bool(
            recorded_intubation.notna().any()
        ),
    }
    return work, details


def _summary_columns(channels: Sequence[str]) -> list[str]:
    return [
        f"{channel}_{statistic}"
        for channel in channels
        for statistic in ("n", "mean", "min", "max", "sd", "slope")
    ]


def _static_frame(
    selected: pd.DataFrame,
    summaries: pd.DataFrame,
    candidate_source: pd.DataFrame,
    channels: Sequence[str],
    landmark_hours: int,
    lab_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    identifiers = [
        column for column in IDENTIFIER_COLUMNS if column in selected.columns
    ]
    context_allowed = landmark_hours >= 24
    rass_columns = (
        [
            column
            for column in RASS_CONTEXT_COLUMNS
            if column in selected.columns
        ]
        if context_allowed
        else []
    )
    context_columns = (
        [
            column
            for column in PRE_24H_CONTEXT_COLUMNS
            if column in selected.columns
        ]
        if context_allowed
        else []
    )
    static_parts = selected[
        identifiers
        + [column for column in BASE_FEATURE_COLUMNS if column in selected.columns]
        + rass_columns
        + context_columns
        + [
            "baseline_gcs_max",
            "baseline_gcs_last",
            "n_gcs_pre",
        ]
    ].copy()
    static_parts = static_parts.merge(summaries, on="stay_id", how="left")
    static_parts[f"monitoring_intensity_{landmark_hours}h"] = (
        pd.to_numeric(static_parts["gcs_total_n"], errors="coerce").fillna(0.0)
        / float(landmark_hours)
    )

    # The processed laboratory summaries cover ICU hours [0, 24). They are
    # available at the 24 h landmark and remain valid pre-landmark covariates
    # at 48 h, but they cannot be used at 6 h.
    labs_supported = landmark_hours >= 24
    if labs_supported:
        available_labs = [column for column in lab_columns if column in candidate_source]
        lab_values = candidate_source[["stay_id", *available_labs]].copy()
        static_parts = static_parts.merge(lab_values, on="stay_id", how="left")
        lab_status = {
            "status": "available",
            "source_window_hours": [0, 24],
            "columns": available_labs,
            "allowed_for_model": True,
            "reason": (
                "Processed laboratory summaries were constructed from events "
                "with 0 <= time_hour < 24, which precede this landmark."
            ),
        }
    else:
        available_labs = [column for column in lab_columns if column in candidate_source]
        missing_labs = pd.DataFrame(
            np.nan,
            index=static_parts.index,
            columns=available_labs,
        )
        static_parts = pd.concat([static_parts, missing_labs], axis=1)
        lab_status = {
            "status": "unavailable_unsupported",
            "source_window_hours": [0, landmark_hours],
            "columns": available_labs,
            "allowed_for_model": False,
            "reason": (
                "Only first-24-hour aggregate laboratory summaries exist in "
                "processed data. Because the 6 h landmark precedes that "
                "aggregation window, the columns are retained as missing and "
                "must not enter the predictor matrix."
            ),
        }

    outcome_part = selected[
        [
            "stay_id",
            *[
                column
                for column in OUTCOME_COLUMNS
                if column in selected.columns
            ],
        ]
    ].copy()
    frame = static_parts.merge(outcome_part, on="stay_id", how="left")
    sequence_columns = _summary_columns(channels)
    predictor_columns = [
        column
        for column in (
            [column for column in BASE_FEATURE_COLUMNS if column in frame.columns]
            + rass_columns
            + context_columns
            + [
                "baseline_gcs_max",
                "baseline_gcs_last",
                *sequence_columns,
            ]
            + (available_labs if labs_supported else [])
            + [f"monitoring_intensity_{landmark_hours}h"]
        )
    ]
    context_status = {
        "status": "available" if context_allowed else "unavailable_unsupported",
        "source_window_hours": [0, 24],
        "columns": rass_columns + context_columns,
        "allowed_for_model": context_allowed,
        "reason": (
            "The stored RASS/deep-sedation summaries, comorbidity, and "
            "first-24-hour treatment flags precede this landmark."
            if context_allowed
            else (
                "The stored RASS/deep-sedation summaries, comorbidity, and "
                "first-24-hour treatment flags include information after the "
                "6 h landmark and are withheld."
            )
        ),
    }
    return frame, predictor_columns, lab_status, context_status


def _eligibility_audit(work: pd.DataFrame, landmark_hours: int) -> dict[str, Any]:
    candidate_n = int(len(work))
    after_icu = work["in_icu_at_landmark"]
    after_alive = after_icu & work["landmark_alive"]
    after_baseline = after_alive & work["baseline_gcs_max"].notna()
    after_outcome = after_baseline & work["outcome"].notna()
    final = after_outcome
    return {
        "candidate_source_n": candidate_n,
        "in_icu_at_landmark_n": int(after_icu.sum()),
        "alive_at_landmark_n": int(after_alive.sum()),
        "valid_pre_landmark_gcs_n": int(after_baseline.sum()),
        "classifiable_outcome_n": int(after_outcome.sum()),
        "analysis_n": int(final.sum()),
        "excluded": {
            "not_in_icu_at_landmark_n": int((~after_icu).sum()),
            "not_alive_at_landmark_n": int(
                (after_icu & ~work["landmark_alive"]).sum()
            ),
            "no_valid_pre_landmark_gcs_n": int(
                (after_alive & work["baseline_gcs_max"].isna()).sum()
            ),
            "unclassifiable_landmark_outcome_n": int(
                (after_baseline & work["outcome"].isna()).sum()
            ),
        },
        # The 72-hour window is not an eligibility criterion: requiring it at
        # the landmark would select on future observation and would drop
        # patients whose outcome was already classifiable. Completeness is
        # reported here for transparency only.
        "followup_status": {
            "complete_72h_window_n": int(
                (after_alive & work["followup_window_complete"]).sum()
            ),
            "live_icu_discharge_before_72h_n": int(
                (
                    after_alive
                    & ~work["followup_window_complete"]
                    & ~work["death_within_72h"]
                ).sum()
            ),
            "note": (
                "Live ICU discharge before 72 h is retained when the observed "
                "post-landmark GCS window still yields a classifiable outcome; "
                "it is not grounds for exclusion."
            ),
        },
        "outcome_counts": {
            str(grade): int((work.loc[final, "outcome"] == grade).sum())
            for grade in (0, 1, 2)
        },
        "scope_note": (
            "The common candidate source contains first ICH ICU stays in adults "
            "and is not restricted to 24-hour survivors. Each landmark applies "
            "its own survival and pre-landmark GCS rules; the 72-hour window is "
            "an observation window, not an inclusion criterion."
        ),
    }


def _sequence_audit(
    values: np.ndarray,
    masks: np.ndarray,
    channels: Sequence[str],
    landmark_hours: int,
) -> dict[str, Any]:
    observed = masks > 0
    channel_missing = {
        channel: float((~observed[:, :, index]).mean())
        for index, channel in enumerate(channels)
    }
    return {
        "shape": list(values.shape),
        "hour_window": [0, landmark_hours],
        "hour_window_stop": "exclusive",
        "channels": list(channels),
        "mask_definition": "1 when an unaltered processed hourly value was observed",
        "missing_value_encoding": "NaN; never imputed, carried forward, or backfilled",
        "observed_cell_fraction": float(observed.mean()) if observed.size else 0.0,
        "missing_fraction_by_channel": channel_missing,
    }


def _numeric_difference(
    reconstructed: pd.Series,
    reference: pd.Series,
) -> dict[str, Any]:
    reconstructed = pd.to_numeric(reconstructed, errors="coerce")
    reference = pd.to_numeric(reference, errors="coerce")
    unequal_missing = reconstructed.isna().ne(reference.isna())
    both = reconstructed.notna() & reference.notna()
    absolute = (reconstructed[both] - reference[both]).abs()
    maximum = float(absolute.max()) if not absolute.empty else 0.0
    n_unequal = int(
        unequal_missing.sum()
        + (absolute.gt(1e-4)).sum()
    )
    return {
        "n_unequal": n_unequal,
        "max_absolute_difference": maximum,
    }


def _reconcile_24h(
    dataset: str,
    cohort: pd.DataFrame,
    values: np.ndarray,
    masks: np.ndarray,
    stay_ids: np.ndarray,
    sources: Mapping[str, Any],
    channels: Sequence[str],
) -> dict[str, Any]:
    paths = sources["reference_paths"]
    if not paths["cohort"].is_file() or not paths["sequences"].is_file():
        return {
            "status": "not_available",
            "all_checks_passed": False,
            "reason": "Existing 24-hour reference files were not found.",
        }

    reference_cohort = pd.read_csv(paths["cohort"])
    reference_cohort["stay_id"] = pd.to_numeric(
        reference_cohort["stay_id"], errors="raise"
    ).astype(np.int64)
    reference_cohort = reference_cohort.sort_values("stay_id").reset_index(drop=True)
    reconstructed = cohort.sort_values("stay_id").reset_index(drop=True)
    checks: dict[str, Any] = {
        "reference_cohort_n": int(len(reference_cohort)),
        "reconstructed_cohort_n": int(len(reconstructed)),
        "stay_id_order_and_values_match": bool(
            np.array_equal(
                reconstructed["stay_id"].to_numpy(dtype=np.int64),
                reference_cohort["stay_id"].to_numpy(dtype=np.int64),
            )
        ),
    }
    common_numeric = [
        "outcome",
        "baseline_gcs_max",
        "baseline_gcs_last",
        "n_gcs_pre",
        "max_sustained_drop",
        "last_gcs_hour",
        "monitoring_intensity_24h",
        *_summary_columns(channels),
    ]
    for column in common_numeric:
        if column not in reconstructed or column not in reference_cohort:
            checks[column] = {
                "n_unequal": None,
                "max_absolute_difference": None,
                "status": "missing_from_one_side",
            }
            continue
        comparison = _numeric_difference(
            reconstructed[column], reference_cohort[column]
        )
        comparison["status"] = (
            "match" if comparison["n_unequal"] == 0 else "mismatch"
        )
        checks[column] = comparison

    lab_columns = [
        column
        for column in reference_cohort.columns
        if column.startswith("lab_") and column in reconstructed.columns
    ]
    lab_mismatches = 0
    lab_max_difference = 0.0
    for column in lab_columns:
        comparison = _numeric_difference(
            reconstructed[column], reference_cohort[column]
        )
        lab_mismatches += int(comparison["n_unequal"])
        lab_max_difference = max(
            lab_max_difference, float(comparison["max_absolute_difference"])
        )
    checks["laboratory_columns"] = {
        "status": "match" if lab_mismatches == 0 else "mismatch",
        "n_columns": int(len(lab_columns)),
        "n_unequal": int(lab_mismatches),
        "max_absolute_difference": lab_max_difference,
    }

    with np.load(paths["sequences"], allow_pickle=True) as npz:
        reference_values = npz["values"].astype(np.float32)
        reference_masks = npz["masks"].astype(np.float32)
        reference_ids = npz["stay_id"].astype(np.int64)
    id_to_row = {int(value): index for index, value in enumerate(reference_ids)}
    ordered = [
        id_to_row[int(stay_id)]
        for stay_id in stay_ids
        if int(stay_id) in id_to_row
    ]
    sequence_ids_match = len(ordered) == len(reference_ids) == len(stay_ids)
    if sequence_ids_match:
        ordered_values = reference_values[ordered]
        ordered_masks = reference_masks[ordered]
        values_match = bool(
            np.array_equal(values, ordered_values, equal_nan=True)
        )
        values_close = bool(
            np.allclose(
                values,
                ordered_values,
                rtol=1e-5,
                atol=1e-4,
                equal_nan=True,
            )
        )
        shared = np.isfinite(values) & np.isfinite(ordered_values)
        maximum_value_difference = (
            float(np.max(np.abs(values[shared] - ordered_values[shared])))
            if shared.any()
            else 0.0
        )
        masks_match = bool(np.array_equal(masks, ordered_masks))
    else:
        values_match = False
        values_close = False
        maximum_value_difference = None
        masks_match = False
    checks["sequences"] = {
        "stay_ids_match": bool(sequence_ids_match),
        "values_match_exactly": values_match,
        "values_match_within_float32_tolerance": values_close,
        "maximum_absolute_value_difference": maximum_value_difference,
        "value_tolerance": {"rtol": 1e-5, "atol": 1e-4},
        "masks_match_exactly": masks_match,
        "reference_shape": list(reference_values.shape),
        "reconstructed_shape": list(values.shape),
    }

    scalar_checks = [
        checks["stay_id_order_and_values_match"],
        checks["sequences"]["stay_ids_match"],
        checks["sequences"]["values_match_within_float32_tolerance"],
        checks["sequences"]["masks_match_exactly"],
        checks["laboratory_columns"]["status"] == "match",
    ]
    for column in common_numeric:
        item = checks.get(column)
        if isinstance(item, Mapping):
            scalar_checks.append(item.get("status") == "match")
    all_passed = bool(all(scalar_checks))
    return {
        "status": "passed" if all_passed else "failed",
        "all_checks_passed": all_passed,
        "dataset": dataset,
        "reference_files": {
            name: str(path) for name, path in paths.items()
        },
        "checks": checks,
    }


def build_landmark_dataset(
    dataset: str,
    landmark_hours: int,
    *,
    processed_root: Path | str = PROCESSED,
    channels: Sequence[str] | None = None,
    strict_reference: bool = True,
) -> LandmarkInputs:
    """Build one leakage-safe landmark cohort and sequence tensor."""
    if landmark_hours not in SUPPORTED_LANDMARKS:
        raise LandmarkInputError(
            f"Unsupported landmark {landmark_hours}; expected {SUPPORTED_LANDMARKS}."
        )
    processed_root = Path(processed_root)
    channels = _normalize_channels(channels)
    sources = _read_sources(processed_root, dataset)
    candidate_source = sources["cohort_pre_filter"].copy()
    candidate_source["stay_id"] = pd.to_numeric(
        candidate_source["stay_id"], errors="raise"
    ).astype(np.int64)
    if landmark_hours >= 24:
        _require_columns(
            candidate_source,
            RASS_CONTEXT_COLUMNS,
            "cohort_pre_filter.csv at the 24 h or later landmark",
        )
    if candidate_source["stay_id"].duplicated().any():
        raise LandmarkInputError("cohort_pre_filter.csv contains duplicate stay_id.")
    candidate_source = candidate_source.sort_values("stay_id").reset_index(drop=True)

    baseline = _baseline_table(
        candidate_source, sources["gcs_events"], landmark_hours
    )
    work, endpoint_details = _attach_outcomes(
        candidate_source,
        sources["gcs_events"],
        baseline,
        landmark_hours,
        dataset,
    )
    eligibility = _eligibility_audit(work, landmark_hours)
    selected = work[
        work["in_icu_at_landmark"]
        & work["landmark_alive"]
        & work["baseline_gcs_max"].notna()
        & work["outcome"].notna()
    ].copy()
    selected["outcome"] = pd.to_numeric(selected["outcome"], errors="raise").astype(int)
    selected = selected.sort_values("stay_id").reset_index(drop=True)

    values, masks, stay_ids = build_sequence_tensor(
        sources["hourly"],
        selected["stay_id"],
        channels=channels,
        max_hour=landmark_hours,
    )
    summaries = sequence_summary(values, masks, stay_ids, channels)
    lab_columns = [
        column
        for column in candidate_source.columns
        if column.startswith("lab_")
    ]
    cohort, predictor_columns, lab_status, context_status = _static_frame(
        selected,
        summaries,
        candidate_source,
        channels,
        landmark_hours,
        lab_columns,
    )
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)

    reconciliation: dict[str, Any]
    if landmark_hours == 24:
        reconciliation = _reconcile_24h(
            dataset,
            cohort,
            values,
            masks,
            np.asarray(stay_ids, dtype=np.int64),
            sources,
            channels,
        )
        if strict_reference and not reconciliation["all_checks_passed"]:
            raise LandmarkReconciliationError(
                "Reconstructed 24-hour cohort or sequences do not match the "
                "existing processed artifacts. See reconciliation details in "
                "the audit when running with strict_reference=False."
            )
    else:
        reconciliation = {
            "status": "not_applicable",
            "all_checks_passed": None,
            "reason": "Reference reconciliation is defined for the 24-hour landmark.",
        }

    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "landmark_hours": landmark_hours,
        "time_origin": "ICU intime",
        "eligibility": eligibility,
        "endpoint": {
            "window_hours": [landmark_hours, OUTCOME_WINDOW_END_HOURS],
            "window_stop": "exclusive",
            "baseline": "maximum total GCS in [0, landmark_hours)",
            "ordinal_rule": (
                "Two adjacent hourly median GCS values at least 2 points below "
                "baseline define confirmed deterioration; the maximum drop in the "
                "qualifying run defines grade 1 versus grade 2. Death after the "
                "landmark is grade 2."
            ),
            "primary_outcome": "confirmed GCS deterioration or death",
            "unplanned_intubation_included": bool(
                endpoint_details["unplanned_intubation_supported"]
            ),
            "unplanned_intubation_composite_available": bool(
                endpoint_details["unplanned_intubation_supported"]
            ),
            "unplanned_intubation_status": (
                "supported"
                if endpoint_details["unplanned_intubation_supported"]
                else "unavailable_unsupported"
            ),
            "unplanned_intubation_reason": endpoint_details[
                "unplanned_intubation_reason"
            ],
            "p020_status": (
                "complete"
                if endpoint_details["unplanned_intubation_supported"]
                else "partial"
            ),
            "recorded_intubation_composite_available": endpoint_details[
                "recorded_intubation_available"
            ],
            "recorded_intubation_composite_role": "sensitivity_proxy_only",
        },
        "sequence": _sequence_audit(values, masks, channels, landmark_hours),
        "static_inputs": {
            "predictor_columns": predictor_columns,
            "laboratory_status": lab_status,
            "context_status": context_status,
            "laboratory_columns_withheld_from_model": (
                lab_columns if landmark_hours < 24 else []
            ),
            "administrative_exclusions": (
                [
                    "comorbidity and treatment flags built from the first 24 h",
                    "first-24-hour laboratory summaries",
                    "recorded and verified intubation fields are excluded from "
                    "all landmark predictor matrices",
                    "post-landmark laboratory, procedure, and outcome fields",
                ]
                if landmark_hours < 24
                else [
                    "recorded and verified intubation fields are endpoint "
                    "metadata, not predictors",
                    "post-landmark laboratory, procedure, and outcome fields",
                ]
            ),
        },
        "reconciliation_24h": reconciliation,
        "source_files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in sources["paths"].items()
        },
        "limitations": [
            (
                "Processed laboratory data contain only first-24-hour aggregate "
                "summaries. They are withheld at 6 h because that window would "
                "include future information; at 48 h they are valid historical "
                "covariates rather than a 48 h laboratory update."
            )
            if landmark_hours != 24
            else "",
            (
                "The displayed outcome uses the stored 24-72 h intubation flag "
                "only at the 24 h landmark."
            )
            if landmark_hours == 24
            else (
                "The outcome is GCS deterioration or death only because the "
                "available intubation flag cannot be aligned to this landmark."
            ),
        ],
    }
    audit["limitations"] = [
        item for item in audit["limitations"] if str(item).strip()
    ]
    audit = _json_ready(audit)
    return LandmarkInputs(
        dataset=dataset,
        landmark_hours=landmark_hours,
        cohort=cohort,
        values=values.astype(np.float32),
        masks=masks.astype(np.float32),
        stay_ids=np.asarray(stay_ids, dtype=np.int64),
        audit=audit,
    )


def write_landmark_dataset(
    built: LandmarkInputs,
    output_root: Path | str,
) -> dict[str, Path]:
    """Write cohort CSV, sequence NPZ, and audit JSON for one landmark."""
    output_dir = (
        Path(output_root) / built.dataset / f"landmark_{built.landmark_hours}h"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "cohort.csv"
    sequence_path = output_dir / "sequences.npz"
    audit_path = output_dir / "audit.json"
    built.cohort.to_csv(cohort_path, index=False)
    np.savez_compressed(
        sequence_path,
        values=built.values,
        masks=built.masks,
        stay_id=built.stay_ids,
        outcome=built.cohort["outcome"].to_numpy(dtype=np.int64),
        channels=np.asarray(built.audit["sequence"]["channels"], dtype=str),
        landmark_hours=np.asarray(built.landmark_hours, dtype=np.int64),
    )
    built.audit["artifact_hashes"] = {
        "cohort.csv": _sha256(cohort_path),
        "sequences.npz": _sha256(sequence_path),
        "hash_algorithm": "sha256",
    }
    with audit_path.open("w", encoding="utf-8") as stream:
        json.dump(built.audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "cohort": cohort_path,
        "sequences": sequence_path,
        "audit": audit_path,
    }


def _parse_csv_argument(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="mimic4,mimic3,eicu",
        help="Comma-separated processed datasets.",
    )
    parser.add_argument(
        "--landmarks",
        default="6,24,48",
        help="Comma-separated landmark hours from 6,24,48.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROCESSED,
        help="Root containing dataset-specific processed artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED / "landmark_inputs",
        help="Output root for landmark_<hours>h directories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional study configuration; defaults to configs/study.json.",
    )
    args = parser.parse_args(argv)

    if args.config is None:
        config = load_config()
    else:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Study configuration not found: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    channels = _normalize_channels(config.get("channels"))
    datasets = _parse_csv_argument(args.datasets)
    landmarks = [int(value) for value in _parse_csv_argument(args.landmarks)]
    if not datasets:
        raise LandmarkInputError("At least one dataset is required.")
    if not landmarks:
        raise LandmarkInputError("At least one landmark is required.")

    written: list[dict[str, Path]] = []
    for dataset in datasets:
        for landmark_hours in landmarks:
            built = build_landmark_dataset(
                dataset,
                landmark_hours,
                processed_root=args.processed_root,
                channels=channels,
                strict_reference=True,
            )
            written.append(write_landmark_dataset(built, args.output_root))
            print(
                f"{dataset} landmark {landmark_hours}h: "
                f"n={len(built.cohort)}, shape={built.values.shape}, "
                f"audit={written[-1]['audit']}"
            )


if __name__ == "__main__":
    main()
