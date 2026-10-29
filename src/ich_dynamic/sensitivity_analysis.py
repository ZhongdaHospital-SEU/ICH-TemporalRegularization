"""Prespecified sensitivity analyses for outline.docx section 2.8.

The registry in this module is fixed before model fitting.  It must not be
edited in response to an observed result.  Scenarios that need source data or
model-fitting support that is not present are reported as unsupported; this
module never substitutes a convenient proxy for a prespecified analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .common import (
    CHANNELS,
    PROCESSED,
    RESULTS,
    build_sequence_tensor,
    classify_gcs_end,
    complete_icu_followup,
    death_in_icu_within_window,
    ensure_dirs,
    load_config,
)
from .train_evaluate import (
    DEEP_STATIC,
    Dataset,
    SEQUENCE_DEFAULTS,
    XGBOOST_DEFAULTS,
    configured_candidates,
    evaluate_predictions,
    evaluate_repeated_cv,
    run_cv,
)


PRIMARY_LANDMARK_HOURS = 24
OUTCOME_END_HOURS = 72
LANDMARK_INPUT_ROOT = PROCESSED / "landmark_inputs"
MINIMUM_POST_LANDMARK_GCS = 2
PRIMARY_CONFIRMATION_CONSECUTIVE_HOURS = 2

SupportLevel = Literal["supported", "outcome_only", "unsupported", "not_applicable"]


@dataclass(frozen=True)
class ScenarioSpec:
    """One prespecified variant within one of the eight outline analyses."""

    analysis_id: str
    analysis_title: str
    scenario_id: str
    variant_title: str
    prespecification: str
    dataset_scope: str = "all"
    baseline_mode: Literal["max", "last"] = "max"
    landmark_hours: int = PRIMARY_LANDMARK_HOURS
    confirmation_consecutive_hours: int = PRIMARY_CONFIRMATION_CONSECUTIVE_HOURS
    include_intubation: bool = False
    exclude_death_within_72h: bool = False
    sedation_strategy: Literal["none", "exclude_windows", "unsedated_only"] = "none"
    # ``primary`` mirrors the main analysis: early live ICU transfers are kept
    # when the observed post-landmark window yields a classifiable outcome.
    # The other strategies are alternatives used only by scenario 4.
    censoring_strategy: Literal["primary", "exclude", "grade0", "ipw"] = "primary"
    requires_rass_adjustment: bool = False

    @property
    def changes_model_inputs(self) -> bool:
        return (
            self.landmark_hours != PRIMARY_LANDMARK_HOURS
            or self.sedation_strategy != "none"
            or self.requires_rass_adjustment
        )

    @property
    def requires_sample_weight(self) -> bool:
        return self.censoring_strategy == "ipw"


_ANALYSIS_TITLES = {
    "1": "Alternative baseline GCS definition",
    "2": "Alternative persistence threshold",
    "3": "Sedation handling",
    "4": "Early ICU transfer censoring",
    "5": "Unplanned intubation in the composite endpoint",
    "6": "Landmark window",
    "7": "Exclusion of deaths within 72 hours",
    "8": "eICU GCS adjustment for RASS",
}


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "1",
        _ANALYSIS_TITLES["1"],
        "s1_baseline_max",
        "Maximum pre-landmark GCS",
        "Baseline is the maximum total GCS from ICU hour 0 to the 24 h landmark.",
    ),
    ScenarioSpec(
        "1",
        _ANALYSIS_TITLES["1"],
        "s1_baseline_last",
        "Final pre-landmark GCS",
        "Baseline is the final total GCS recorded before the 24 h landmark.",
        baseline_mode="last",
    ),
    ScenarioSpec(
        "2",
        _ANALYSIS_TITLES["2"],
        "s2_persistence_1h",
        "At least one low hourly observation",
        "A >=2-point reduction is confirmed by one low GCS observation.",
        confirmation_consecutive_hours=1,
    ),
    ScenarioSpec(
        "2",
        _ANALYSIS_TITLES["2"],
        "s2_persistence_2h",
        "At least two consecutive hourly hours",
        "A >=2-point reduction must be present in two adjacent hourly GCS medians.",
    ),
    ScenarioSpec(
        "3",
        _ANALYSIS_TITLES["3"],
        "s3_exclude_continuous_sedation",
        "Exclude GCS observations in continuous sedation windows",
        "RASS <= -2 in two hourly observations separated by one to two hours "
        "defines a continuous sedation window; all GCS observations inside that "
        "window are excluded before endpoint reconstruction.",
        sedation_strategy="exclude_windows",
    ),
    ScenarioSpec(
        "3",
        _ANALYSIS_TITLES["3"],
        "s3_unsedated_only",
        "Restrict to the unsedated subgroup",
        "Patients with no RASS <= -2 observation through ICU hour 72 are included; "
        "patients without observed RASS are not classified as unsedated.",
        sedation_strategy="unsedated_only",
    ),
    ScenarioSpec(
        "4",
        _ANALYSIS_TITLES["4"],
        "s4_censor_exclude",
        "Exclude early live ICU transfers",
        "Alive patients transferred out of the ICU before 72 h are excluded.",
        censoring_strategy="exclude",
    ),
    ScenarioSpec(
        "4",
        _ANALYSIS_TITLES["4"],
        "s4_censor_grade0",
        "Code early live ICU transfers as grade 0",
        "Alive patients transferred out of the ICU before 72 h are retained and "
        "assigned grade 0 regardless of their observed pre-transfer trajectory. "
        "The primary analysis instead keeps the grade observed in the available "
        "post-landmark window.",
        censoring_strategy="grade0",
    ),
    ScenarioSpec(
        "4",
        _ANALYSIS_TITLES["4"],
        "s4_censor_ipw",
        "Inverse-probability weighting for early ICU transfer",
        "Complete-follow-up observations are weighted by the stabilized inverse "
        "probability of remaining under observation through 72 h. The propensity "
        "model uses only prespecified pre-landmark covariates.",
        censoring_strategy="ipw",
    ),
    ScenarioSpec(
        "5",
        _ANALYSIS_TITLES["5"],
        "s5_with_intubation",
        "Composite endpoint includes unplanned intubation",
        "Unplanned intubation after the landmark is grade 2, in addition to GCS "
        "decline and death.",
        include_intubation=True,
    ),
    ScenarioSpec(
        "5",
        _ANALYSIS_TITLES["5"],
        "s5_without_intubation",
        "GCS/death endpoint excludes unplanned intubation",
        "Unplanned intubation after the landmark is ignored; GCS decline and death "
        "are retained.",
        include_intubation=False,
    ),
    ScenarioSpec(
        "6",
        _ANALYSIS_TITLES["6"],
        "s6_landmark_6h",
        "6 h landmark",
        "Baseline and input sequence end at 6 h; deterioration is ascertained from "
        "6 to 72 h.",
        landmark_hours=6,
    ),
    ScenarioSpec(
        "6",
        _ANALYSIS_TITLES["6"],
        "s6_landmark_24h",
        "24 h landmark",
        "Baseline and input sequence end at 24 h; deterioration is ascertained from "
        "24 to 72 h.",
        landmark_hours=24,
    ),
    ScenarioSpec(
        "6",
        _ANALYSIS_TITLES["6"],
        "s6_landmark_48h",
        "48 h landmark",
        "Baseline and input sequence end at 48 h; deterioration is ascertained from "
        "48 to 72 h.",
        landmark_hours=48,
    ),
    ScenarioSpec(
        "7",
        _ANALYSIS_TITLES["7"],
        "s7_exclude_death72h",
        "Exclude deaths within 72 h",
        "Patients who die at or before ICU hour 72 are excluded from both endpoint "
        "ascertainment and model fitting.",
        exclude_death_within_72h=True,
    ),
    ScenarioSpec(
        "8",
        _ANALYSIS_TITLES["8"],
        "s8_eicu_gcs_unadjusted",
        "eICU GCS without RASS adjustment",
        "The recorded total GCS is used without a sedation correction.",
        dataset_scope="eicu",
    ),
    ScenarioSpec(
        "8",
        _ANALYSIS_TITLES["8"],
        "s8_eicu_gcs_rass_adjusted",
        "eICU GCS with RASS adjustment",
        "Requires an explicit contemporaneous GCS-RASS adjustment field. Imputed "
        "hourly RASS is not an acceptable substitute.",
        dataset_scope="eicu",
        requires_rass_adjustment=True,
    ),
)

SCENARIO_INDEX = {scenario.scenario_id: scenario for scenario in SCENARIOS}

# Compatibility names from the superseded analysis script. They point to the
# closest prespecified scenario and do not create additional analyses.
LEGACY_SCENARIO_ALIASES = {
    "last_gcs_baseline": "s1_baseline_last",
    "single_low_gcs": "s2_persistence_1h",
    "strict_persistence_2h": "s2_persistence_2h",
    "exclude_death": "s7_exclude_death72h",
    "early_discharge_stable": "s4_censor_grade0",
    "exclude_deep_sedation": "s3_unsedated_only",
    "intubation_composite": "s5_with_intubation",
}

MODELS = ("xgb_summary", "gru", "dggru_tr")
LANDMARK_MODELS = (
    "dggru",
    "dggru_tr",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)


def models_for_scenario(spec: ScenarioSpec) -> tuple[str, ...]:
    """Return the prespecified model matrix for one sensitivity scenario."""
    return LANDMARK_MODELS if spec.analysis_id == "6" else MODELS
ENDPOINT_CAPABILITIES = {
    "mimic4": {
        "unplanned_intubation": False,
        "recorded_intubation": True,
        "intubation_field_label": (
            "procedureevents starttime after the landmark; planned versus "
            "unplanned status is not identifiable"
        ),
        "outcome_label": "GCS deterioration or death",
    },
    "mimic3": {
        "unplanned_intubation": False,
        "recorded_intubation": True,
        "intubation_field_label": (
            "procedureevents starttime after the landmark; planned versus "
            "unplanned status is not identifiable"
        ),
        "outcome_label": "GCS deterioration or death",
    },
    "eicu": {
        "unplanned_intubation": False,
        "recorded_intubation": False,
        "intubation_field_label": "no harmonized intubation timing field",
        "outcome_label": "GCS deterioration or death",
    },
}
ADJUSTED_GCS_COLUMNS = ("gcs_rass_adjusted", "adjusted_gcs")
RASS_COLUMNS = ("rass", "rass_value", "value")
IPW_COVARIATES = (
    "age",
    "sex_male",
    "race_white",
    "emergency_admission",
    "baseline_gcs_max",
    "min_rass_24h",
    "any_deep_sedation_24h",
)
IPW_CROSSFIT_FOLDS = 5
IPW_PROBABILITY_FLOOR = 0.05
IPW_PROBABILITY_CEILING = 0.95
MINIMUM_SEDATION_COVERAGE = 0.80


class UnsupportedScenarioError(RuntimeError):
    """Raised instead of silently substituting data for a prespecified scenario."""


@dataclass(frozen=True)
class ScenarioStatus:
    scenario_id: str
    dataset: str
    support: SupportLevel
    model_run_allowed: bool
    reason: str


@dataclass
class PreparedScenario:
    spec: ScenarioSpec
    status: ScenarioStatus
    frame: pd.DataFrame
    rass_adjustment_applied: bool = False
    weights_estimated: bool = False


def resolve_scenario(scenario: str | ScenarioSpec) -> ScenarioSpec:
    if isinstance(scenario, ScenarioSpec):
        return scenario
    key = LEGACY_SCENARIO_ALIASES.get(scenario, scenario)
    try:
        return SCENARIO_INDEX[key]
    except KeyError as exc:
        available = ", ".join(SCENARIO_INDEX)
        raise ValueError(f"Unknown scenario {scenario!r}. Available: {available}") from exc


def scenario_registry_manifest() -> dict:
    """Return the fixed, JSON-serializable section 2.8 registry."""
    analyses = []
    for analysis_id, title in _ANALYSIS_TITLES.items():
        variants = [
            scenario.scenario_id
            for scenario in SCENARIOS
            if scenario.analysis_id == analysis_id
        ]
        analyses.append(
            {
                "analysis_id": analysis_id,
                "title": title,
                "variants": variants,
            }
        )
    return {
        "outline_section": "2.8",
        "analysis_count": len(analyses),
        "analyses": analyses,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "endpoint_capabilities": ENDPOINT_CAPABILITIES,
        "endpoint_selection_rule": (
            "All scenarios are fixed before model fitting; no scenario is selected "
            "or altered according to model performance."
        ),
    }


def effective_include_intubation(
    scenario: str | ScenarioSpec,
    dataset: str,
) -> bool:
    """Return the endpoint component available for a specific database.

    The eICU extract has no reliable unplanned-intubation field. Its
    transported endpoint is therefore explicitly GCS deterioration or death.
    Analysis 5 remains the one place where the presence of that component is
    tested; other analyses use the available harmonized endpoint.
    """
    spec = resolve_scenario(scenario)
    if not spec.include_intubation:
        return False
    capability = ENDPOINT_CAPABILITIES.get(dataset, {})
    return bool(capability.get("unplanned_intubation", True))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return (numeric.eq(1) | text.isin({"true", "t", "yes", "y", "1"})).fillna(False)


def _scalar_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "yes", "y", "1"}
    return bool(value)


def _normalise_gcs(gcs: pd.DataFrame) -> pd.DataFrame:
    required = {"stay_id", "gcs"}
    missing = required.difference(gcs.columns)
    if missing:
        raise ValueError(f"GCS events are missing columns: {sorted(missing)}")
    out = gcs.copy()
    if "time_hour" not in out:
        if "charttime" in out:
            raise ValueError(
                "GCS events with charttime but no time_hour require ICU intime; "
                "reconstruct time_hour before running sensitivity analyses."
            )
        raise ValueError("GCS events are missing time_hour")
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce")
    out["time_hour"] = pd.to_numeric(out["time_hour"], errors="coerce")
    out["gcs"] = pd.to_numeric(out["gcs"], errors="coerce")
    out = out.dropna(subset=["stay_id", "time_hour"]).copy()
    out["stay_id"] = out["stay_id"].astype(np.int64)
    if "ett_verbal" in out:
        out["ett_verbal"] = pd.to_numeric(out["ett_verbal"], errors="coerce").fillna(0)
    return out.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)


def _normalise_rass(rass_events: pd.DataFrame | None) -> pd.DataFrame | None:
    if rass_events is None or rass_events.empty:
        return None
    required = {"stay_id", "time_hour"}
    missing = required.difference(rass_events.columns)
    if missing:
        return None
    value_column = next(
        (column for column in RASS_COLUMNS if column in rass_events.columns),
        None,
    )
    if value_column is None:
        return None
    out = rass_events.copy()
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce")
    out["time_hour"] = pd.to_numeric(out["time_hour"], errors="coerce")
    out["rass"] = pd.to_numeric(out[value_column], errors="coerce")
    out = out.dropna(subset=["stay_id", "time_hour", "rass"]).copy()
    if out.empty:
        return None
    out["stay_id"] = out["stay_id"].astype(np.int64)
    return out.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)


def _explicit_adjusted_gcs(gcs: pd.DataFrame) -> pd.Series | None:
    """Return an explicit adjusted GCS field only when it is complete.

    Partial adjustment is not patched with the unadjusted value. Doing so would
    silently mix two endpoint definitions.
    """
    column = next(
        (name for name in ADJUSTED_GCS_COLUMNS if name in gcs.columns),
        None,
    )
    if column is None:
        return None
    values = pd.to_numeric(gcs[column], errors="coerce")
    if values.isna().any() or not values.between(3, 15).all():
        return None
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_landmark_artifacts(
    dataset: str,
    landmark_hours: int,
) -> tuple[bool, str]:
    """Validate the complete landmark artifact contract before model fitting."""
    landmark_dir = (
        LANDMARK_INPUT_ROOT / dataset / f"landmark_{landmark_hours}h"
    )
    paths = {
        "cohort": landmark_dir / "cohort.csv",
        "sequences": landmark_dir / "sequences.npz",
        "audit": landmark_dir / "audit.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return False, f"missing files: {', '.join(missing)}"
    try:
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        if audit.get("schema_version") != 1:
            return False, "audit schema_version is not 1"
        if audit.get("dataset") != dataset:
            return False, "audit dataset does not match the requested dataset"
        if int(audit.get("landmark_hours", -1)) != int(landmark_hours):
            return False, "audit landmark_hours does not match the request"
        hashes = audit.get("artifact_hashes", {})
        if hashes.get("hash_algorithm") != "sha256":
            return False, "artifact hashes are missing or use an unknown algorithm"
        for name, path in (
            ("cohort.csv", paths["cohort"]),
            ("sequences.npz", paths["sequences"]),
        ):
            if hashes.get(name) != _sha256(path):
                return False, f"{name} hash does not match audit.json"

        cohort = pd.read_csv(paths["cohort"])
        required_columns = {"stay_id", "outcome"}
        if not required_columns.issubset(cohort.columns):
            return False, "cohort.csv lacks stay_id or outcome"
        if cohort.empty:
            return False, "cohort.csv is empty"
        if cohort["stay_id"].duplicated().any():
            return False, "cohort.csv contains duplicate stay_id values"
        cohort_ids = pd.to_numeric(
            cohort["stay_id"], errors="coerce"
        ).to_numpy(dtype=np.int64)
        outcomes = pd.to_numeric(
            cohort["outcome"], errors="coerce"
        ).to_numpy(dtype=np.int64)
        if not set(np.unique(outcomes)).issubset({0, 1, 2}):
            return False, "cohort outcome values are not exclusively 0, 1, or 2"

        with np.load(paths["sequences"], allow_pickle=False) as archive:
            required_arrays = {
                "values",
                "masks",
                "stay_id",
                "outcome",
                "channels",
                "landmark_hours",
            }
            missing_arrays = required_arrays.difference(archive.files)
            if missing_arrays:
                return False, (
                    "sequences.npz lacks arrays: "
                    + ", ".join(sorted(missing_arrays))
                )
            values = archive["values"]
            masks = archive["masks"]
            sequence_ids = archive["stay_id"].astype(np.int64)
            sequence_outcomes = archive["outcome"].astype(np.int64)
            channels = archive["channels"].astype(str).tolist()
            stored_landmark = int(archive["landmark_hours"].item())
        expected_shape = (len(cohort), int(landmark_hours), len(CHANNELS))
        if values.shape != expected_shape or masks.shape != expected_shape:
            return False, (
                f"sequence shape {values.shape}/{masks.shape} does not match "
                f"expected {expected_shape}"
            )
        if not np.array_equal(sequence_ids, cohort_ids):
            return False, "sequence stay_id order differs from cohort.csv"
        if len(np.unique(sequence_ids)) != len(sequence_ids):
            return False, "sequence stay_id values are not unique"
        if not np.array_equal(sequence_outcomes, outcomes):
            return False, "sequence outcomes differ from cohort.csv"
        if stored_landmark != int(landmark_hours):
            return False, "stored sequence landmark_hours differs from the request"
        if channels != list(CHANNELS):
            return False, "sequence channels differ from the study channel contract"
        if not np.isfinite(masks).all() or not np.isin(masks, (0.0, 1.0)).all():
            return False, "sequence masks must be finite and binary"
        finite_values = values[np.isfinite(values)]
        if not np.isfinite(finite_values).all():
            return False, "sequence values contain invalid numeric entries"

        sequence_audit = audit.get("sequence", {})
        if sequence_audit.get("channels") != list(CHANNELS):
            return False, "audit sequence channels differ from the study contract"
        if sequence_audit.get("shape") != list(expected_shape):
            return False, "audit sequence shape does not match the arrays"
        eligibility = audit.get("eligibility", {})
        if int(eligibility.get("analysis_n", -1)) != len(cohort):
            return False, "audit analysis_n does not match cohort.csv"
        if landmark_hours == PRIMARY_LANDMARK_HOURS:
            reconciliation = audit.get("reconciliation_24h", {})
            if reconciliation.get("all_checks_passed") is not True:
                return False, "24 h reconciliation did not pass"
    except Exception as exc:
        return False, f"artifact validation failed: {exc}"
    return True, "schema, hashes, shapes, identifiers, outcomes, and audit passed"


def validate_landmark_model_inputs(
    dataset: str,
    landmark_hours: int,
) -> tuple[bool, str]:
    """Validate the target landmark and its 6 h predictor-anchor artifact."""
    validations = {
        hour: validate_landmark_artifacts(dataset, hour)
        for hour in sorted({int(landmark_hours), 6})
    }
    invalid = {
        hour: message
        for hour, (valid, message) in validations.items()
        if not valid
    }
    if invalid:
        details = "; ".join(
            f"{hour} h: {message}" for hour, message in invalid.items()
        )
        return False, details
    return (
        True,
        (
            f"{int(landmark_hours)} h target and 6 h predictor-anchor "
            "artifacts passed schema, hash, shape, identifier, outcome, "
            "and audit validation"
        ),
    )


def _censoring_support_status(
    spec: ScenarioSpec,
    prefilter: pd.DataFrame,
) -> str | None:
    """Return an unsupported reason when a censoring design is not estimable."""
    if spec.censoring_strategy == "primary":
        return None
    work = prefilter.drop_duplicates("stay_id").copy()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce")
    work = work.dropna(subset=["stay_id"]).copy()
    death_hour = pd.to_numeric(
        work.get(
            "death_hour_from_icu",
            pd.Series(np.nan, index=work.index, dtype=float),
        ),
        errors="coerce",
    )
    icu_out_hour = pd.to_numeric(
        work.get(
            "icu_out_hour",
            pd.Series(np.nan, index=work.index, dtype=float),
        ),
        errors="coerce",
    )
    death = _as_bool(work["death_within_72h"])
    death_in_icu = death_in_icu_within_window(
        work["death_within_72h"],
        death_hour,
        icu_out_hour,
    )
    landmark_alive = ~death | death_hour.gt(float(spec.landmark_hours))
    in_icu = icu_out_hour.ge(float(spec.landmark_hours))
    work = work[landmark_alive & in_icu].copy()
    outcome = pd.to_numeric(work.get("outcome", np.nan), errors="coerce")
    observed = complete_icu_followup(
        outcome,
        icu_out_hour.loc[work.index],
        death_in_icu.loc[work.index],
    )
    administrative_censor = (
        ~death_in_icu.loc[work.index]
        & icu_out_hour.loc[work.index].ge(float(spec.landmark_hours))
        & icu_out_hour.loc[work.index].lt(OUTCOME_END_HOURS)
    )
    if spec.censoring_strategy == "exclude":
        if int(observed.sum()) == 0:
            return "The 24 h landmark risk set has no observed outcomes."
        return None
    if int(administrative_censor.sum()) == 0:
        return (
            "The early-transfer sensitivity analysis requires at least one "
            "post-landmark administratively censored patient."
        )
    if spec.censoring_strategy == "grade0":
        return None
    if int(observed.sum()) < IPW_CROSSFIT_FOLDS or int(
        administrative_censor.sum()
    ) < IPW_CROSSFIT_FOLDS:
        return (
            "IPW requires at least five observed and five post-landmark "
            "administratively censored patients."
        )
    covariates = [column for column in IPW_COVARIATES if column in work.columns]
    usable = [
        column
        for column in covariates
        if pd.to_numeric(work[column], errors="coerce").notna().any()
    ]
    if not usable:
        return "No prespecified IPW covariate is observed in the risk set."
    group_column = (
        "subject_key"
        if "subject_key" in work.columns
        else "stay_id"
    )
    if work[group_column].astype(str).nunique() < IPW_CROSSFIT_FOLDS:
        return (
            "IPW requires at least five patient groups for patient-grouped "
            "cross-fitting."
        )
    return None


def assess_scenario_support(
    scenario: str | ScenarioSpec,
    dataset: str,
    prefilter: pd.DataFrame | None = None,
    gcs: pd.DataFrame | None = None,
    rass_events: pd.DataFrame | None = None,
) -> ScenarioStatus:
    """Assess whether the prespecified analysis can be run exactly."""
    spec = resolve_scenario(scenario)
    if spec.dataset_scope != "all" and dataset != spec.dataset_scope:
        return ScenarioStatus(
            spec.scenario_id,
            dataset,
            "not_applicable",
            False,
            f"Outline analysis {spec.analysis_id} applies only to {spec.dataset_scope}.",
        )
    if prefilter is None or gcs is None:
        return ScenarioStatus(
            spec.scenario_id,
            dataset,
            "unsupported",
            False,
            "Prefilter and GCS event data are required for a support decision.",
        )

    if spec.analysis_id == "4":
        censoring_reason = _censoring_support_status(spec, prefilter)
        if censoring_reason is not None:
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "unsupported",
                False,
                censoring_reason,
            )

    if spec.requires_rass_adjustment:
        if _explicit_adjusted_gcs(gcs) is None:
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "unsupported",
                False,
                "No complete explicit contemporaneous GCS-RASS adjustment field "
                "(gcs_rass_adjusted or adjusted_gcs) is present. Forward-filled "
                "hourly RASS is not a valid substitute.",
            )

    endpoint_is_reconstructed = spec.analysis_id != "4"
    intubation_required = effective_include_intubation(spec, dataset)
    if (
        spec.include_intubation
        and not ENDPOINT_CAPABILITIES.get(dataset, {}).get(
            "unplanned_intubation", False
        )
    ):
        return ScenarioStatus(
            spec.scenario_id,
            dataset,
            "unsupported",
            False,
            "Reliable planned versus unplanned intubation after the landmark is "
            "unavailable in this source. A procedure start time records "
            "intubation but cannot establish that it was unplanned, so the "
            "intubation-augmented composite is not substituted with that proxy.",
        )
    if (
        endpoint_is_reconstructed
        and intubation_required
        and "verified_unplanned_intubation_24_72h" in prefilter
    ):
        if (
            prefilter["verified_unplanned_intubation_24_72h"]
            .notna()
            .sum()
            == 0
        ):
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "unsupported",
                False,
                "The prespecified composite endpoint includes unplanned intubation, "
                "but no nonmissing intubation observations are available.",
            )
    elif (
        endpoint_is_reconstructed
        and intubation_required
        and "verified_unplanned_intubation_24_72h" not in prefilter
    ):
        return ScenarioStatus(
            spec.scenario_id,
            dataset,
            "unsupported",
            False,
            "The prespecified composite endpoint includes unplanned intubation, "
            "but the column is absent.",
        )

    if spec.sedation_strategy != "none":
        rass = _normalise_rass(rass_events)
        if rass is None:
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "unsupported",
                False,
                "Event-level RASS with stay_id and time_hour is required for the "
                "sedation analysis; patient-level any_sedation flags cannot define "
                "continuous windows or an observed unsedated subgroup.",
            )
        death = _as_bool(prefilter["death_within_72h"])
        death_hour = pd.to_numeric(
            prefilter.get(
                "death_hour_from_icu",
                pd.Series(np.nan, index=prefilter.index, dtype=float),
            ),
            errors="coerce",
        )
        icu_out_hour = pd.to_numeric(
            prefilter.get(
                "icu_out_hour",
                pd.Series(np.nan, index=prefilter.index, dtype=float),
            ),
            errors="coerce",
        )
        eligible_at_landmark = (
            (~death | death_hour.gt(float(spec.landmark_hours)))
            & icu_out_hour.ge(float(spec.landmark_hours))
        )
        eligible_ids = set(
            pd.to_numeric(
                prefilter.loc[eligible_at_landmark, "stay_id"],
                errors="coerce",
            )
            .dropna()
            .astype(np.int64)
        )
        observed_ids = set(
            rass.loc[
                rass["time_hour"].between(
                    0, OUTCOME_END_HOURS, inclusive="both"
                ),
                "stay_id",
            ].astype(np.int64)
        )
        coverage = (
            len(eligible_ids & observed_ids) / len(eligible_ids)
            if eligible_ids
            else 0.0
        )
        if coverage < MINIMUM_SEDATION_COVERAGE:
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "unsupported",
                False,
                (
                    "Event-level RASS covers only "
                    f"{coverage:.1%} of landmark-eligible stays; the "
                    "prespecified sedation scenarios require at least "
                    f"{MINIMUM_SEDATION_COVERAGE:.0%}. Missing RASS cannot be "
                    "interpreted as no sedation."
                ),
            )

    if spec.analysis_id == "6":
        valid, validation_message = validate_landmark_model_inputs(
            dataset, spec.landmark_hours
        )
        if valid:
            return ScenarioStatus(
                spec.scenario_id,
                dataset,
                "supported",
                True,
                "Landmark-specific eligibility, sequence, and static inputs "
                "and the common 6 h predictor anchor were rebuilt without "
                "post-landmark leakage. "
                + validation_message.capitalize()
                + ".",
            )
        return ScenarioStatus(
            spec.scenario_id,
            dataset,
            "outcome_only",
            False,
            f"Landmark-specific model inputs for {spec.landmark_hours} h are "
            f"invalid or missing ({validation_message}). Run "
            "work.build_landmark_inputs before this analysis; both the target "
            "landmark and 6 h anchor are required, and the 24 h prefilter and "
            "static table are not reusable without leakage.",
        )

    return ScenarioStatus(
        spec.scenario_id,
        dataset,
        "supported",
        True,
        "All required fields and the 24 h prediction inputs are available.",
    )


def continuous_sedation_windows(
    rass_events: pd.DataFrame | None,
    *,
    threshold: float = -2.0,
    start_hour: float = 0.0,
    end_hour: float = OUTCOME_END_HOURS,
) -> pd.DataFrame:
    """Return merged continuous RASS <= threshold windows per ICU stay."""
    rass = _normalise_rass(rass_events)
    if rass is None:
        raise UnsupportedScenarioError(
            "Event-level RASS is unavailable for continuous sedation windows."
        )
    work = rass[rass["time_hour"].between(start_hour, end_hour, inclusive="both")]
    low = work[work["rass"].le(threshold)].copy()
    records: list[dict[str, float | int]] = []
    for stay_id, group in low.groupby("stay_id", sort=False):
        times = group["time_hour"].sort_values().to_numpy(dtype=float)
        for first in times:
            later = times[(times >= first + 1.0) & (times <= first + 2.0)]
            if not len(later):
                continue
            records.append(
                {
                    "stay_id": int(stay_id),
                    "sedation_start_hour": float(first),
                    "sedation_end_hour": float(later[-1]),
                }
            )
    columns = ["stay_id", "sedation_start_hour", "sedation_end_hour"]
    if not records:
        return pd.DataFrame(columns=columns)
    windows = pd.DataFrame(records).sort_values(
        ["stay_id", "sedation_start_hour", "sedation_end_hour"]
    )
    merged = []
    for stay_id, group in windows.groupby("stay_id", sort=False):
        current_start = float(group["sedation_start_hour"].iloc[0])
        current_end = float(group["sedation_end_hour"].iloc[0])
        for row in group.iloc[1:].itertuples(index=False):
            start = float(row.sedation_start_hour)
            end = float(row.sedation_end_hour)
            if start <= current_end + 1e-9:
                current_end = max(current_end, end)
            else:
                merged.append((int(stay_id), current_start, current_end))
                current_start, current_end = start, end
        merged.append((int(stay_id), current_start, current_end))
    return pd.DataFrame(merged, columns=columns)


def _exclude_sedation_windows(
    gcs: pd.DataFrame,
    rass_events: pd.DataFrame | None,
    landmark_hours: int,
) -> pd.DataFrame:
    windows = continuous_sedation_windows(rass_events, end_hour=OUTCOME_END_HOURS)
    if windows.empty:
        return gcs
    keep = np.ones(len(gcs), dtype=bool)
    by_stay = {
        int(stay_id): group
        for stay_id, group in windows.groupby("stay_id", sort=False)
    }
    for index, row in enumerate(gcs.itertuples(index=False)):
        group = by_stay.get(int(row.stay_id))
        if group is None:
            continue
        within = (
            group["sedation_start_hour"].le(float(row.time_hour))
            & group["sedation_end_hour"].ge(float(row.time_hour))
        )
        keep[index] = not bool(within.any())
    return gcs.loc[keep].copy()


def _unsedated_stay_ids(rass_events: pd.DataFrame | None) -> set[int]:
    rass = _normalise_rass(rass_events)
    if rass is None:
        raise UnsupportedScenarioError(
            "Observed RASS is required to define the prespecified unsedated subgroup."
        )
    study = rass[rass["time_hour"].between(0, OUTCOME_END_HOURS, inclusive="both")]
    if study.empty:
        return set()
    minimum = study.groupby("stay_id")["rass"].min()
    return {int(stay_id) for stay_id, value in minimum.items() if float(value) > -2.0}


def _classify_landmark_end(
    post: pd.DataFrame,
    baseline: float,
    *,
    landmark_hours: int,
    death_within_72h: bool,
    unplanned_intubation: bool,
    confirmation_consecutive_hours: int,
) -> tuple[float, float]:
    """Return ordinal grade and maximum confirmed drop.

    At least two post-landmark GCS hours are required for every stay. Among
    eligible stays, death or the verified intubation component is grade 2
    regardless of the observed GCS reduction. Primary persistence requires two
    adjacent hourly medians.
    """
    result = classify_gcs_end(
        post,
        baseline,
        start_hour=landmark_hours,
        end_hour=OUTCOME_END_HOURS,
        death_within_window=death_within_72h,
        unplanned_intubation=unplanned_intubation,
        min_post_measurements=MINIMUM_POST_LANDMARK_GCS,
        confirmation_consecutive_hours=confirmation_consecutive_hours,
    )
    outcome = result["outcome"]
    maximum_drop = result["max_sustained_drop"]
    return float(outcome), float(maximum_drop)


def _reconstruct_frame(
    spec: ScenarioSpec,
    prefilter: pd.DataFrame,
    gcs: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    frame = prefilter.drop_duplicates("stay_id").copy()
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
    frame = frame.dropna(subset=["stay_id"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(np.int64)
    death_hour = pd.to_numeric(
        frame.get(
            "death_hour_from_icu",
            pd.Series(np.nan, index=frame.index, dtype=float),
        ),
        errors="coerce",
    )
    death_within_window = _as_bool(frame["death_within_72h"])
    icu_out_hour = pd.to_numeric(
        frame.get(
            "icu_out_hour",
            pd.Series(np.nan, index=frame.index, dtype=float),
        ),
        errors="coerce",
    )
    frame["landmark_alive"] = (
        ~death_within_window
        | death_hour.gt(float(spec.landmark_hours))
    )
    frame["death_in_icu_within_72h"] = death_in_icu_within_window(
        frame["death_within_72h"],
        death_hour,
        icu_out_hour,
    )
    frame["in_icu_at_landmark"] = icu_out_hour.ge(
        float(spec.landmark_hours)
    )
    # Endpoint death follows the primary definition: any death within 72 h of
    # ICU admission, including a death after a live ICU discharge. The
    # in-ICU flag is retained separately because it defines completion of the
    # ICU observation window used for censoring.
    frame["death_by_endpoint_window"] = (
        death_within_window
        & death_hour.gt(float(spec.landmark_hours))
    )
    frame = frame[
        frame["landmark_alive"] & frame["in_icu_at_landmark"]
    ].copy()
    frame["outcome_primary"] = pd.to_numeric(
        frame.get("outcome", np.nan), errors="coerce"
    )

    events = _normalise_gcs(gcs)
    adjusted = False
    if spec.requires_rass_adjustment:
        replacement = _explicit_adjusted_gcs(events)
        if replacement is None:
            raise UnsupportedScenarioError(
                "The prespecified eICU RASS-adjusted GCS endpoint is unavailable: "
                "no complete explicit adjustment field exists."
            )
        events.loc[:, "gcs"] = replacement.to_numpy()
        adjusted = True

    if spec.sedation_strategy == "exclude_windows":
        raise UnsupportedScenarioError(
            "Internal error: sedation events were not supplied to reconstruction."
        )

    by_stay = {
        int(stay_id): group
        for stay_id, group in events.groupby("stay_id", sort=False)
    }
    empty = events.iloc[0:0]
    records: list[dict[str, float | int]] = []
    for row in frame.itertuples(index=False):
        group = by_stay.get(int(row.stay_id), empty)
        pre = group[group["time_hour"].between(0, spec.landmark_hours, inclusive="left")]
        post = group[
            group["time_hour"].between(
                spec.landmark_hours, OUTCOME_END_HOURS, inclusive="left"
            )
        ]
        baseline_max = float(pre["gcs"].max()) if not pre.empty else np.nan
        baseline_last = (
            float(pre.sort_values("time_hour")["gcs"].iloc[-1])
            if not pre.empty
            else np.nan
        )
        baseline = baseline_max if spec.baseline_mode == "max" else baseline_last
        death = _scalar_bool(
            getattr(row, "death_by_endpoint_window", False)
        )
        intubation_raw = getattr(
            row,
            "verified_unplanned_intubation_24_72h",
            np.nan,
        )
        intubation = (
            pd.notna(intubation_raw)
            and float(intubation_raw) == 1.0
            and spec.include_intubation
        )
        outcome, maximum_drop = _classify_landmark_end(
            post,
            baseline,
            landmark_hours=spec.landmark_hours,
            death_within_72h=death,
            unplanned_intubation=intubation,
            confirmation_consecutive_hours=spec.confirmation_consecutive_hours,
        )
        records.append(
            {
                "stay_id": int(row.stay_id),
                "baseline_gcs_max": baseline_max,
                "baseline_gcs_last": baseline_last,
                "n_gcs_pre": int(len(pre)),
                "n_gcs_post": int(len(post)),
                "outcome_reconstructed": outcome,
                "max_sustained_drop": maximum_drop,
                "last_gcs_hour": (
                    float(post["time_hour"].max()) if not post.empty else np.nan
                ),
            }
        )
    reconstructed = pd.DataFrame(records)
    frame = frame.drop(
        columns=[
            "baseline_gcs_max",
            "baseline_gcs_last",
            "n_gcs_pre",
            "n_gcs_post",
            "outcome_reconstructed",
            "max_sustained_drop",
            "last_gcs_hour",
        ],
        errors="ignore",
    ).merge(reconstructed, on="stay_id", how="left")
    return frame, adjusted


def _add_censoring_metadata(
    frame: pd.DataFrame,
    spec: ScenarioSpec,
    dataset: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["sensitivity_analysis_id"] = spec.analysis_id
    out["sensitivity_scenario"] = spec.scenario_id
    out["censoring_strategy"] = spec.censoring_strategy
    out["baseline_definition"] = spec.baseline_mode
    out["landmark_hours"] = spec.landmark_hours
    out["intubation_in_endpoint"] = effective_include_intubation(spec, dataset)
    out["endpoint_label"] = ENDPOINT_CAPABILITIES.get(
        dataset, {}
    ).get("outcome_label", "GCS deterioration or death")
    return out


def _primary_outcome_for_censoring(frame: pd.DataFrame) -> pd.Series:
    primary = (
        pd.to_numeric(frame["outcome_primary"], errors="coerce")
        if "outcome_primary" in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    reconstructed = pd.to_numeric(
        frame.get("outcome_reconstructed", np.nan), errors="coerce"
    )
    return primary.where(primary.notna(), reconstructed)


def _apply_censoring(frame: pd.DataFrame, spec: ScenarioSpec) -> pd.DataFrame:
    if spec.censoring_strategy not in {"primary", "exclude", "grade0", "ipw"}:
        raise ValueError(f"Unknown censoring strategy: {spec.censoring_strategy}")
    work = frame.copy()
    death_hour = pd.to_numeric(
        work.get(
            "death_hour_from_icu",
            pd.Series(np.nan, index=work.index, dtype=float),
        ),
        errors="coerce",
    )
    work["death_within_72h"] = _as_bool(work["death_within_72h"])
    work["death_in_icu_within_72h"] = death_in_icu_within_window(
        work["death_within_72h"],
        death_hour,
        work.get(
            "icu_out_hour",
            pd.Series(np.nan, index=work.index, dtype=float),
        ),
    )
    work["landmark_alive"] = (
        ~work["death_within_72h"]
        | death_hour.gt(float(spec.landmark_hours))
    )
    work = work[work["landmark_alive"]].copy()
    # Scenarios 1, 2, 3 and 5 redefine the endpoint and must use the
    # reconstructed label. Scenario 4 changes only which stays are analysed, so
    # its variants start from the primary outcome.
    if spec.analysis_id in {"1", "2", "3", "5"}:
        work["outcome"] = pd.to_numeric(
            work["outcome_reconstructed"], errors="coerce"
        )
    else:
        work["outcome"] = _primary_outcome_for_censoring(work)
    work["death_within_72h"] = _as_bool(work["death_within_72h"])
    work["icu_out_hour"] = pd.to_numeric(work["icu_out_hour"], errors="coerce")
    work["death_in_icu_within_72h"] = death_in_icu_within_window(
        work["death_within_72h"],
        death_hour,
        work["icu_out_hour"],
    )
    work["in_icu_at_landmark"] = work["icu_out_hour"].ge(
        float(spec.landmark_hours)
    )
    early_transfer = (
        ~work["death_in_icu_within_72h"]
        & work["icu_out_hour"].lt(OUTCOME_END_HOURS)
        & work["icu_out_hour"].ge(spec.landmark_hours)
    )
    work["early_icu_transfer"] = early_transfer.astype(bool)
    work["followup_complete"] = complete_icu_followup(
        work["outcome"],
        work["icu_out_hour"],
        work["death_in_icu_within_72h"],
    )
    work["unresolved_outcome"] = (
        work["outcome"].isna()
        & ~work["early_icu_transfer"]
    )
    work["protocol_excluded_before_landmark"] = (
        ~_as_bool(work["landmark_alive"]) | ~work["in_icu_at_landmark"]
    )

    if spec.exclude_death_within_72h:
        work = work[~work["death_within_72h"]].copy()

    if spec.censoring_strategy == "primary":
        selected = work[work["outcome"].notna()].copy()
        selected["analysis_weight"] = 1.0
        selected["censoring_probability"] = np.nan
    elif spec.censoring_strategy == "exclude":
        selected = work[work["followup_complete"]].copy()
        selected["analysis_weight"] = 1.0
        selected["censoring_probability"] = np.nan
    elif spec.censoring_strategy == "grade0":
        selected = work[
            work["followup_complete"] | work["early_icu_transfer"]
        ].copy()
        selected.loc[selected["early_icu_transfer"], "outcome"] = 0.0
        selected = selected[selected["outcome"].notna()].copy()
        selected["analysis_weight"] = 1.0
        selected["censoring_probability"] = np.nan
    else:
        risk_set = work[
            work["followup_complete"] | work["early_icu_transfer"]
        ].copy()
        risk_set.loc[risk_set["early_icu_transfer"], "outcome"] = 0.0
        risk_set = risk_set[risk_set["outcome"].notna()].copy()
        selected = estimate_censoring_weights(
            risk_set,
            retain_censored=True,
        )
        selected.attrs["ipw_risk_set"] = risk_set.reset_index(drop=True)

    selected["outcome"] = pd.to_numeric(selected["outcome"], errors="raise").astype(int)
    weighting_audit = selected.attrs.get("ipw_audit")
    ipw_risk_set = selected.attrs.get("ipw_risk_set")
    selected = selected.reset_index(drop=True)
    if weighting_audit is not None:
        selected.attrs["ipw_audit"] = weighting_audit
    if ipw_risk_set is not None:
        selected.attrs["ipw_risk_set"] = ipw_risk_set
    return selected


def estimate_censoring_weights(
    eligible: pd.DataFrame,
    covariates: Iterable[str] = IPW_COVARIATES,
    *,
    seed: int = 20261024,
    n_splits: int = IPW_CROSSFIT_FOLDS,
    retain_censored: bool = False,
    training_subject_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Estimate patient-grouped cross-fitted observation weights.

    The design matrix is fixed by ``IPW_COVARIATES`` and uses only variables
    available before the landmark. Every patient's weight is predicted by a
    propensity model fitted without that patient's group. This prevents a
    patient's own censoring status from contributing to their weight while
    retaining the prespecified stabilized inverse-probability estimand.
    Covariates with no nonmissing value in the eligible cohort are removed and
    recorded before fitting. Probabilities are clipped to [0.05, 0.95] as a
    prespecified finite-weight rule. The returned ``analysis_weight`` is
    consumed by every model-training path in ``train_evaluate``.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    work = eligible.copy()
    observed = work["followup_complete"].astype(bool)
    if observed.nunique() < 2:
        raise UnsupportedScenarioError(
            "IPW cannot be estimated because the censoring indicator has one level."
        )
    if observed.sum() < 5 or (~observed).sum() < 5:
        raise UnsupportedScenarioError(
            "IPW requires at least five observed and five censored patients."
        )
    prespecified = [column for column in covariates if column in work.columns]
    if not prespecified:
        raise UnsupportedScenarioError("No prespecified IPW covariates are present.")
    numeric = work[prespecified].apply(pd.to_numeric, errors="coerce")
    columns = [
        column for column in prespecified if numeric[column].notna().any()
    ]
    excluded_all_missing = [
        column for column in prespecified if column not in columns
    ]
    if not columns:
        raise UnsupportedScenarioError(
            "All prespecified IPW covariates are missing in the eligible cohort."
        )

    if "subject_key" in work.columns:
        groups = work["subject_key"].astype(str).to_numpy()
    else:
        groups = work["stay_id"].astype(str).to_numpy()
    design = numeric[columns].to_numpy(dtype=float)
    outcome = observed.to_numpy(dtype=int)
    probability = np.full(len(work), np.nan, dtype=float)
    analysis_weight = np.full(len(work), np.nan, dtype=float)
    fold_assignment = np.full(len(work), -1, dtype=int)
    fold_prevalence: list[float] = []

    def fit_and_predict(train_index: np.ndarray, test_index: np.ndarray) -> None:
        if len(np.unique(outcome[train_index])) < 2:
            raise UnsupportedScenarioError(
                "IPW propensity training lacks both censoring levels."
            )
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=seed,
            ),
        )
        model.fit(design[train_index], outcome[train_index])
        fold_probability = np.clip(
            model.predict_proba(design[test_index])[:, 1],
            IPW_PROBABILITY_FLOOR,
            IPW_PROBABILITY_CEILING,
        )
        prevalence = float(outcome[train_index].mean())
        fold_prevalence.append(prevalence)
        probability[test_index] = fold_probability
        analysis_weight[test_index] = prevalence / fold_probability

    if training_subject_keys is None:
        n_groups = len(np.unique(groups))
        eligible_splits = min(
            int(n_splits),
            n_groups,
            int(observed.sum()),
            int((~observed).sum()),
        )
        if eligible_splits < 2:
            raise UnsupportedScenarioError(
                "IPW cross-fitting requires at least two patient groups and two "
                "patients in each censoring stratum."
            )
        splitter = StratifiedGroupKFold(
            n_splits=eligible_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(design, outcome, groups=groups)
        ):
            fit_and_predict(train_index, test_index)
            fold_assignment[test_index] = fold
        estimator = "patient_grouped_cross_fitted_logistic"
    else:
        training_keys = np.asarray(list(training_subject_keys)).astype(str)
        train_index = np.flatnonzero(np.isin(groups, training_keys))
        if len(train_index) == 0:
            raise UnsupportedScenarioError(
                "Outer training fold contains no eligible IPW risk-set rows."
            )
        eligible_splits = 1
        fit_and_predict(
            train_index,
            np.arange(len(work), dtype=int),
        )
        fold_assignment[:] = 0
        estimator = "nested_outer_training_fold_logistic"

    if not np.isfinite(probability).all() or not np.isfinite(
        analysis_weight
    ).all():
        raise RuntimeError("IPW cross-fitting did not assign every eligible row.")

    work["censoring_probability"] = probability
    work["analysis_weight"] = analysis_weight
    work["analysis_weight_raw"] = 1.0 / probability
    work["ipw_crossfit_fold"] = fold_assignment
    work["ipw_crossfit_splits"] = eligible_splits
    work["ipw_covariate_count"] = len(columns)
    retained = (
        work.copy().reset_index(drop=True)
        if retain_censored
        else work.loc[observed].copy().reset_index(drop=True)
    )
    retained_weight = retained["analysis_weight"].to_numpy(dtype=float)
    ess = float(
        np.square(retained_weight.sum())
        / np.square(retained_weight).sum()
    )
    retained.attrs["ipw_audit"] = {
        "estimator": estimator,
        "crossfit_splits": eligible_splits,
        "prespecified_covariates": prespecified,
        "covariates_used": columns,
        "excluded_all_missing_covariates": excluded_all_missing,
        "probability_clip": [
            IPW_PROBABILITY_FLOOR,
            IPW_PROBABILITY_CEILING,
        ],
        "stabilization_numerator": "training-fold observation prevalence",
        "fold_observation_prevalence": fold_prevalence,
        "observed_n": int(observed.sum()),
        "censored_n": int((~observed).sum()),
        "retained_censored_rows": bool(retain_censored),
        "retained_n": int(len(retained)),
        "effective_sample_size": ess,
    }
    return retained


def prepare_scenario(
    scenario: str | ScenarioSpec,
    prefilter: pd.DataFrame,
    gcs: pd.DataFrame,
    *,
    dataset: str = "mimic4",
    rass_events: pd.DataFrame | None = None,
) -> PreparedScenario:
    """Prepare one prespecified sensitivity cohort and its support decision."""
    spec = resolve_scenario(scenario)
    status = assess_scenario_support(
        spec,
        dataset,
        prefilter=prefilter,
        gcs=gcs,
        rass_events=rass_events,
    )
    if status.support in {"unsupported", "not_applicable"}:
        return PreparedScenario(spec, status, pd.DataFrame())

    events = _normalise_gcs(gcs)
    frame = prefilter.copy()
    adjusted = False
    if spec.sedation_strategy == "exclude_windows":
        events = _exclude_sedation_windows(events, rass_events, spec.landmark_hours)
    elif spec.sedation_strategy == "unsedated_only":
        unsedated = _unsedated_stay_ids(rass_events)
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
        frame = frame[frame["stay_id"].isin(unsedated)].copy()

    reconstruct_spec = (
        replace(spec, sedation_strategy="none")
        if spec.sedation_strategy == "exclude_windows"
        else spec
    )
    frame, adjusted = _reconstruct_frame(reconstruct_spec, frame, events)
    frame = _apply_censoring(frame, spec)
    frame = _add_censoring_metadata(frame, spec, dataset)
    return PreparedScenario(
        spec=spec,
        status=status,
        frame=frame,
        rass_adjustment_applied=adjusted,
        weights_estimated=bool(frame["analysis_weight"].notna().any()),
    )


def scenario_frame(
    scenario: str | ScenarioSpec,
    prefilter: pd.DataFrame,
    gcs: pd.DataFrame,
    *,
    dataset: str = "mimic4",
    rass_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper returning the prepared sensitivity cohort."""
    prepared = prepare_scenario(
        scenario,
        prefilter,
        gcs,
        dataset=dataset,
        rass_events=rass_events,
    )
    if prepared.status.support in {"unsupported", "not_applicable"}:
        raise UnsupportedScenarioError(prepared.status.reason)
    return prepared.frame


def complete_followup(frame: pd.DataFrame) -> pd.Series:
    """Primary complete-follow-up indicator used by the superseded module."""
    outcome = pd.to_numeric(frame.get("outcome"), errors="coerce")
    icu_out = pd.to_numeric(frame["icu_out_hour"], errors="coerce")
    death_hour = pd.to_numeric(
        frame.get("death_hour_from_icu", np.nan), errors="coerce"
    )
    death_in_icu = death_in_icu_within_window(
        frame["death_within_72h"],
        death_hour,
        icu_out,
    )
    return complete_icu_followup(outcome, icu_out, death_in_icu)


def post_landmark_counts(
    gcs: pd.DataFrame,
    landmark_hours: int = PRIMARY_LANDMARK_HOURS,
) -> pd.Series:
    events = _normalise_gcs(gcs)
    post = events[
        events["time_hour"].between(
            landmark_hours, OUTCOME_END_HOURS, inclusive="left"
        )
        & events["gcs"].notna()
    ]
    return post.groupby("stay_id").size().rename("n_gcs_post")


def matched_landmark_predictor_columns(
    dataset: str,
    landmark_hours: int,
    available_columns: Iterable[str],
) -> list[str]:
    """Return the 6-hour-available predictor set for every landmark window.

    Figure 5 compares prediction landmarks. Using all 211 predictors at 24 and
    48 hours but only 92 at 6 hours would confound the landmark comparison with
    laboratory and exposure availability. The common set is therefore defined
    once from the 6-hour artifact, with only the time-window suffix of
    monitoring intensity mapped to the target landmark.
    """
    anchor_path = (
        LANDMARK_INPUT_ROOT / dataset / "landmark_6h" / "audit.json"
    )
    if not anchor_path.is_file():
        raise UnsupportedScenarioError(
            f"The 6 h landmark audit is required to align predictors: {anchor_path}"
        )
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_columns = [
        str(column)
        for column in anchor.get("static_inputs", {}).get(
            "predictor_columns", []
        )
    ]
    available = {str(column) for column in available_columns}
    selected: list[str] = []
    for column in anchor_columns:
        candidates = [column]
        if column == "monitoring_intensity_6h":
            candidates.insert(
                0,
                f"monitoring_intensity_{int(landmark_hours)}h",
            )
        target = next(
            (candidate for candidate in candidates if candidate in available),
            None,
        )
        if target is not None:
            selected.append(target)
    if len(selected) != len(anchor_columns):
        missing = [
            column
            for column in anchor_columns
            if column == "monitoring_intensity_6h"
            or column not in available
        ]
        raise ValueError(
            "The target landmark cannot represent the 6-hour predictor set: "
            + ", ".join(missing)
        )
    return selected


def build_dataset(
    scenario: str | ScenarioSpec,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    dataset: str = "mimic4",
    allow_outcome_only: bool = False,
) -> Dataset:
    spec = resolve_scenario(scenario)
    if (
        spec.analysis_id == "6"
        or spec.landmark_hours != PRIMARY_LANDMARK_HOURS
    ):
        landmark_dir = (
            LANDMARK_INPUT_ROOT
            / dataset
            / f"landmark_{spec.landmark_hours}h"
        )
        if allow_outcome_only:
            raise UnsupportedScenarioError(
                "Outcome-only landmark data cannot be passed to a model."
            )
        cohort_path = landmark_dir / "cohort.csv"
        sequence_path = landmark_dir / "sequences.npz"
        audit_path = landmark_dir / "audit.json"
        if not all(
            path.is_file()
            for path in (cohort_path, sequence_path, audit_path)
        ):
            raise UnsupportedScenarioError(
                f"Landmark inputs are missing under {landmark_dir}."
            )
        cohort = (
            pd.read_csv(cohort_path)
            .drop_duplicates("stay_id")
            .sort_values("stay_id")
            .reset_index(drop=True)
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        landmark_predictors = matched_landmark_predictor_columns(
            dataset,
            spec.landmark_hours,
            cohort.columns,
        )
        if not landmark_predictors:
            raise ValueError(
                f"{audit_path} contains no usable static predictor columns."
            )
        with np.load(sequence_path, allow_pickle=True) as sequence_data:
            values = sequence_data["values"].astype(np.float32)
            masks = sequence_data["masks"].astype(np.float32)
            stay_ids = sequence_data["stay_id"].astype(np.int64)
        if not np.array_equal(stay_ids, cohort["stay_id"].to_numpy(dtype=np.int64)):
            raise ValueError(
                f"{sequence_path} is not aligned to {cohort_path}."
            )
        numeric = cohort.select_dtypes(include=[np.number])
        static_all = numeric.reindex(columns=landmark_predictors)
        static_deep = static_all.reindex(
            columns=[
                column
                for column in DEEP_STATIC
                if column in static_all.columns
            ]
        )
        labels = pd.to_numeric(cohort["outcome"], errors="raise").astype(int)
        analysis_weight = (
            pd.to_numeric(cohort["analysis_weight"], errors="raise").to_numpy(
                dtype=np.float64
            )
            if "analysis_weight" in cohort
            else np.ones(len(cohort), dtype=np.float64)
        )
        return Dataset(
            name=f"sensitivity_{spec.scenario_id}",
            cohort=cohort,
            sequences=values,
            masks=masks,
            labels=labels.to_numpy(),
            static_all=static_all,
            static_deep=static_deep,
            analysis_weight=analysis_weight,
        )

    cohort = frame.drop_duplicates("stay_id").sort_values("stay_id").reset_index(drop=True)
    hourly = pd.read_csv(PROCESSED / dataset / "hourly.csv")
    config = load_config()
    values, masks, ids = build_sequence_tensor(
        hourly,
        cohort["stay_id"],
        channels=config["channels"],
        max_hour=spec.landmark_hours,
    )
    numeric = cohort.select_dtypes(include=[np.number])
    static_all = numeric.reindex(columns=feature_columns)
    static_deep = numeric.reindex(
        columns=[column for column in DEEP_STATIC if column in numeric]
    )
    if "analysis_weight" not in cohort:
        raise ValueError("Sensitivity cohorts must include analysis_weight")
    analysis_weight = pd.to_numeric(
        cohort["analysis_weight"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(analysis_weight).all() or (analysis_weight <= 0).any():
        raise ValueError(
            "Sensitivity analysis weights must be finite and strictly positive"
        )
    return Dataset(
        name=f"sensitivity_{spec.scenario_id}",
        cohort=cohort,
        sequences=values.astype(np.float32),
        masks=masks.astype(np.float32),
        labels=cohort["outcome"].astype(int).to_numpy(),
        static_all=static_all,
        static_deep=static_deep,
        analysis_weight=analysis_weight,
    )


def _load_rass_events(dataset: str) -> pd.DataFrame | None:
    path = PROCESSED / dataset / "rass_events.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def run_scenario(
    scenario: str | ScenarioSpec,
    *,
    dataset: str,
    epochs: int,
    folds: int,
    repeats: int,
    replicates: int,
    workers: int,
    worker_threads: int,
) -> dict:
    """Run one exactly supported scenario and return an audit record."""
    spec = resolve_scenario(scenario)
    prefilter = pd.read_csv(PROCESSED / dataset / "cohort_pre_filter.csv")
    gcs = pd.read_csv(PROCESSED / dataset / "gcs_events.csv")
    rass_events = _load_rass_events(dataset)
    status = assess_scenario_support(
        spec,
        dataset,
        prefilter=prefilter,
        gcs=gcs,
        rass_events=rass_events,
    )
    record = {
        "scenario": asdict(spec),
        "status": asdict(status),
        "n": None,
        "outcomes": None,
        "artifact": None,
        "weighting": None,
    }
    if not status.model_run_allowed:
        print(
            f"[{dataset}/{spec.scenario_id}] {status.support}: {status.reason}",
            flush=True,
        )
        return record

    prepared = prepare_scenario(
        spec,
        prefilter,
        gcs,
        dataset=dataset,
        rass_events=rass_events,
    )
    ipw_risk_set = prepared.frame.attrs.get("ipw_risk_set")
    config = load_config()
    manifest = json.loads(
        (RESULTS / "run_manifest.json").read_text(encoding="utf-8")
    )
    feature_columns = manifest["features"]
    data = build_dataset(
        spec,
        prepared.frame,
        feature_columns,
        dataset=dataset,
    )
    record["n"] = int(len(data.labels))
    record["outcomes"] = {
        str(key): int(value)
        for key, value in pd.Series(data.labels).value_counts().sort_index().items()
    }
    record["weighting"] = prepared.frame.attrs.get("ipw_audit")
    print(
        f"[{dataset}/{spec.scenario_id}] n={record['n']} "
        f"outcomes={record['outcomes']}",
        flush=True,
    )

    scenario_models = models_for_scenario(spec)
    xgb_candidates = configured_candidates(
        config,
        "xgboost",
        set(XGBOOST_DEFAULTS),
    )
    sequence_candidates = configured_candidates(
        config,
        "sequence_models",
        set(SEQUENCE_DEFAULTS),
    )
    fold_weight_builder = None
    if spec.censoring_strategy == "ipw":
        if not isinstance(ipw_risk_set, pd.DataFrame):
            raise RuntimeError(
                "IPW scenario is missing its prespecified censoring risk set"
            )
        risk_set = ipw_risk_set.copy().reset_index(drop=True)
        selected_ids = data.cohort["stay_id"].astype(np.int64).to_numpy()

        def fold_weight_builder(train_index: np.ndarray) -> np.ndarray:
            training_subjects = (
                data.cohort["subject_key"]
                .astype(str)
                .to_numpy()[train_index]
            )
            weighted = estimate_censoring_weights(
                risk_set,
                seed=int(config["random_seed"]),
                retain_censored=True,
                training_subject_keys=np.unique(training_subjects),
            )
            lookup = weighted.set_index("stay_id")["analysis_weight"]
            aligned = pd.Series(selected_ids).map(lookup)
            if aligned.isna().any():
                raise RuntimeError(
                    "Nested IPW failed to assign a weight to every analysis row"
                )
            return aligned.to_numpy(dtype=np.float64)

        record["weighting"] = {
            **(record["weighting"] or {}),
            "nesting": "re-estimated_within_each_outer_training_fold",
            "censored_outcomes": "grade_0",
        }
    predictions, _, _, _, _ = run_cv(
        data,
        folds=folds,
        epochs=epochs,
        seed=int(config["random_seed"]),
        models=scenario_models,
        lambda_monotone=float(config["lambda_monotone"]),
        lambda_smooth=float(config["lambda_smooth"]),
        lambda_grid=tuple(float(value) for value in config["lambda_grid"]),
        lambda_smooth_ratio=float(config["lambda_smooth_ratio"]),
        model_seeds=tuple(range(int(config.get("model_seeds", 5)))),
        workers=workers,
        worker_threads=worker_threads,
        repeats=repeats,
        repeat_prefix=f"sensitivity_{dataset}_{spec.scenario_id}",
        xgb_candidates=xgb_candidates,
        sequence_hyperparameters=sequence_candidates,
        fold_weight_builder=fold_weight_builder,
    )
    outdir = RESULTS / "sensitivity" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    prediction_path = outdir / f"{spec.scenario_id}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        labels=data.labels,
        stay_id=data.cohort["stay_id"].to_numpy(),
        subject_key=data.cohort["subject_key"].astype(str).to_numpy(),
        analysis_weight=data.cohort["analysis_weight"].to_numpy(dtype=float),
        **predictions,
    )
    metrics = evaluate_repeated_cv(
        data,
        RESULTS
        / "predictions"
        / f"sensitivity_{dataset}_{spec.scenario_id}_repeats.npz",
        scenario_models,
        max(1, repeats),
        replicates=replicates,
        seed=int(config["random_seed"]),
    )
    metrics_path = outdir / f"{spec.scenario_id}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    record["artifact"] = str(metrics_path)
    return record


def _scenarios_for_analysis(analysis_id: str) -> tuple[ScenarioSpec, ...]:
    if analysis_id != "all" and analysis_id not in _ANALYSIS_TITLES:
        raise ValueError(f"Unknown outline analysis id: {analysis_id}")
    if analysis_id == "all":
        return SCENARIOS
    return tuple(
        scenario for scenario in SCENARIOS if scenario.analysis_id == analysis_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        choices=("all",) + tuple(_ANALYSIS_TITLES),
        default="all",
        help="Outline section 2.8 analysis number.",
    )
    parser.add_argument("--scenario", default=None, help="Specific scenario id.")
    parser.add_argument(
        "--dataset",
        choices=("mimic4", "mimic3", "eicu"),
        default="mimic4",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-threads", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    repeats = (
        int(config.get("cv_repeats", 1))
        if args.cv_repeats is None
        else int(args.cv_repeats)
    )
    workers = (
        int(config.get("parallel_fold_workers", 1))
        if args.workers is None
        else max(1, int(args.workers))
    )
    worker_threads = (
        int(config.get("worker_threads", 3))
        if args.worker_threads is None
        else max(1, int(args.worker_threads))
    )
    scenarios = (
        (resolve_scenario(args.scenario),)
        if args.scenario
        else _scenarios_for_analysis(args.analysis)
    )
    outdir = RESULTS / "sensitivity" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    registry_path = outdir / "scenario_registry.json"
    registry_path.write_text(
        json.dumps(scenario_registry_manifest(), indent=2),
        encoding="utf-8",
    )
    records = []
    for scenario in scenarios:
        records.append(
            run_scenario(
                scenario,
                dataset=args.dataset,
                epochs=args.epochs,
                folds=args.folds,
                repeats=repeats,
                replicates=args.replicates,
                workers=workers,
                worker_threads=worker_threads,
            )
        )
    with (outdir / "scenario_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
