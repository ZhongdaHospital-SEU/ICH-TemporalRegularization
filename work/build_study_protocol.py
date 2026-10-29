"""Render the final operational analysis plan without retrospective claims."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "outputs" / "study_protocol.md"
FIGURE_OUTPUTS = (
    ("Figure 1", "figure1_cohort_flow"),
    ("Figure 2", "figure2_model_architecture"),
    ("Figure 3A-C", "figure3_model_performance"),
    ("Figure 4A", "figure4_calibration"),
    ("Figure 4B", "figure4_decision_curve"),
    ("Figure 5", "figure5_temporal_structure"),
    ("Figure 6A", "figure6_interpretation"),
    ("Figure 6B", "figure6_subgroups"),
)

DATABASE_SPANS = {
    "mimic4": "2008-2022",
    "mimic3": "2001-2012",
    "eicu": "2014-2015",
}
OUTLINE_BASELINES = (
    "static XGBoost",
    "summary-feature XGBoost",
    "SeqOrdinal GRU",
    "GRU-D",
    "Transformer",
)
OUTLINE_MODELS = (
    "xgb_static",
    "xgb_summary",
    "gcs_slope",
    "gru",
    "grud",
    "transformer",
    "dggru",
    "dggru_tr",
    "nested_stack",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)
SECONDARY_ABLATIONS = (
    "dggru_tr_gcsmask",
    "dggru_tr_noseq",
    "dggru_tr_static",
    "dggru_tr_traj",
)
P020_STATUS = "PARTIAL"
OUTLINE_SENSITIVITIES = (
    (
        "1. Baseline GCS",
        "Maximum pre-landmark GCS versus final pre-landmark GCS",
    ),
    (
        "2. Persistence",
        "At least 1 hour versus at least 2 hours",
    ),
    (
        "3. Sedation",
        "Exclude continuous sedation windows versus observed unsedated subgroup",
    ),
    (
        "4. Early transfer",
        "Primary exclusion, grade-0 assignment, and inverse-probability weighting",
    ),
    (
        "5. Intubation",
        "Required unplanned-intubation composite is unsupported because planned "
        "status is unavailable; ordinary recorded intubation is not substituted",
    ),
    (
        "6. Landmark",
        "6-hour, 24-hour, and 48-hour landmark windows",
    ),
    (
        "7. Death",
        "Exclude patients who die within 72 hours",
    ),
    (
        "8. eICU RASS",
        "Recorded GCS versus explicit RASS-adjusted GCS",
    ),
)
OUTLINE_SUBGROUPS = (
    "age",
    "mechanical ventilation",
    "anticoagulation-related hemorrhage",
    "neurosurgical intervention",
    "sedation exposure",
)


def main() -> None:
    missing_figures = [
        f"{stem}{suffix}"
        for _, stem in FIGURE_OUTPUTS
        for suffix in (".png", ".pdf")
        if not (RESULTS / "figures" / f"{stem}{suffix}").exists()
    ]
    if missing_figures:
        raise FileNotFoundError(
            "Study protocol generation requires every outline figure in both "
            "PNG and PDF form; missing: " + ", ".join(missing_figures)
        )
    config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    manifest_path = RESULTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {
        name: pd.read_csv(PROCESSED / name / "cohort.csv")
        for name in ("mimic4", "mimic3", "eicu")
    }
    flows = {
        name: pd.read_csv(PROCESSED / name / "cohort_flow.csv")
        for name in ("mimic4", "mimic3", "eicu")
    }
    declared_models = tuple(manifest.get("models", []))
    if set(declared_models) != set(OUTLINE_MODELS + SECONDARY_ABLATIONS):
        raise ValueError(
            "run_manifest.json does not contain the exact outline-plus-ablation set: "
            f"expected {OUTLINE_MODELS}, observed {declared_models}."
        )
    channels = tuple(config.get("channels", []))
    if len(channels) != 14 or len(set(channels)) != 14:
        raise ValueError("configs/study.json must define 14 unique channels.")
    model_seeds = manifest.get("model_seeds", [])
    if not isinstance(model_seeds, list) or len(model_seeds) != 5:
        raise ValueError("The outline requires five fixed model seeds.")
    n_features = len(manifest.get("features", []))
    if n_features == 0:
        raise ValueError("run_manifest.json contains no static predictors.")
    n_primary = int(
        manifest.get("temporal_static_feature_count", n_features)
    )
    if n_primary <= 0:
        raise ValueError("run_manifest.json declares no primary static set.")

    rass_coverage = {
        name: 100.0
        * pd.to_numeric(frame["min_rass_24h"], errors="coerce").notna().mean()
        for name, frame in frames.items()
    }
    def counts(name: str) -> str:
        values = frames[name]["outcome"].value_counts().sort_index()
        return "/".join(str(int(values.get(grade, 0))) for grade in (0, 1, 2))

    def screened(name: str) -> int:
        flow = flows[name]
        match = flow[
            flow["reason_code"].astype(str).eq("source_ich_icu_stays")
        ]
        if len(match) != 1:
            raise ValueError(
                f"{name}/cohort_flow.csv lacks one unique "
                "source_ich_icu_stays row."
            )
        return int(match.iloc[0]["n"])

    channel_lines = "\n".join(
        f"- {channel}: {config['channel_labels'][channel]}"
        for channel in channels
    )
    sensitivity_lines = "\n".join(
        f"{index}. {title}: {description}."
        for index, (title, description) in enumerate(
            OUTLINE_SENSITIVITIES,
            start=1,
        )
    )
    subgroup_text = ", ".join(OUTLINE_SUBGROUPS)
    baseline_text = ", ".join(OUTLINE_BASELINES)
    model_text = ", ".join(OUTLINE_MODELS)
    figure_lines = "\n".join(
        f"- {label}: `results/figures/{stem}.png` and `{stem}.pdf`."
        for label, stem in FIGURE_OUTPUTS
    )

    text = f"""# Analysis Plan and Operational Definitions

## Status and objective

Version: 2026-10-24.

This document records the operational definitions implemented for the final
analysis. It is not a registration record and does not claim prospective
registration. The objective is to estimate whether temporal regularization
adds reproducible discrimination beyond an otherwise matched dynamic-graph
model, while preserving all outline baselines, controls, transport cohorts,
and sensitivity variants regardless of their direction.

The prediction landmark is ICU hour {config['landmark_hours']}. The outcome
window is ICU hours {config['landmark_hours']} through
{config['outcome_end_hours']}. The temporal network uses information from ICU
hours 0-23.

## Cohorts

| Role | Database | Source span | Screened ICH stays | Analyzable stays | Grade 0/1/2 |
|---|---|---|---:|---:|---:|
| Development | MIMIC-IV v3.1 | {DATABASE_SPANS['mimic4']} | {screened('mimic4'):,} | {len(frames['mimic4']):,} | {counts('mimic4')} |
| Same-institution source shift | MIMIC-III v1.4 | {DATABASE_SPANS['mimic3']} | {screened('mimic3'):,} | {len(frames['mimic3']):,} | {counts('mimic3')} |
| Multi-hospital transport | eICU-CRD v2.0 | {DATABASE_SPANS['eicu']} | {screened('eicu'):,} | {len(frames['eicu']):,} | {counts('eicu')} |

MIMIC-IV and MIMIC-III are different database generations collected at the
same institution. Their source periods are not used as patient-level time
coordinates. eICU is a multi-hospital transport cohort and is interpreted
with the precision supported by its observed event count.

The outline's earlier draft sample sizes are not carried forward because they
differ from the rebuilt audit. The source and analytic counts above are read
directly from the rebuilt `cohort_flow.csv` and `cohort.csv` artifacts.

## Eligibility

- Adult ICH ICU admission with ICD-9 431/432 or ICD-10 I61/I62.
- The first ICU episode per patient is ranked before applying ICH, adult,
  landmark-survival, and GCS eligibility filters.
- An otherwise qualifying stay is not excluded solely because of a co-coded
  diagnosis in the implemented query.
- Alive at ICU hour 24.
- At least one valid total GCS before hour 24.
- At least two distinct ICU hours with valid total-GCS values between ICU
  hours 24 and 72. This requirement applies to every included patient,
  including those who died by hour 72; sparse-GCS deaths are excluded rather
  than assigned grade 2.
- A live ICU discharge before hour 72 is not an exclusion criterion. Such a
  stay is retained when the observed post-landmark GCS window yields a
  classifiable grade. The complete-follow-up subset is a prespecified
  sensitivity analysis, because requiring the full window would select on
  future observation.

## Ordinal endpoint

The reference is the maximum valid total GCS in ICU hours 0-24. GCS values are
reduced to hourly medians. A positive signal requires two adjacent hourly
medians, each at least 2 points below the reference. A missing intervening hour
breaks persistence. The maximum drop within the qualifying consecutive run
defines:

- Grade 0: no qualifying confirmed drop, without death by hour 72.
- Grade 1: confirmed drop of 2-3 points.
- Grade 2: confirmed drop of at least 4 points, or death within 72 h of ICU
  admission, including a death after a live ICU discharge.

The endpoint is explicitly `P020 = {P020_STATUS}` with `p020_complete = false`.
It is an operational signal, not an unconfounded measure of neurological
function. Sedation, analgesia, neuromuscular blockade, intubation, aphasia, and
care limitations can affect recorded GCS. The outline also requested an
intubation-augmented grade-2 endpoint. The available procedure record identifies
post-landmark intubation without a reliable planned versus unplanned
distinction. Because the outline explicitly requires unplanned intubation, that
variant is reported as unsupported and is not replaced by the broader
recorded-intubation proxy. No intubation component is used in the transported
endpoint.

## Predictors and missing data

Fourteen channels are represented hourly with values and explicit observation
masks:

{channel_lines}

The outline states 14 channels but names 13 modalities. Total GCS is included
to reconcile that discrepancy without dropping a named channel. The 14-channel
tensor was confirmed in all three cohorts. `SequencePreprocessor` is fitted
inside each outer training fold using training rows only, then applies causal
within-stay forward filling over ICU hours 0-23; no backward filling is used,
and the original observation mask remains available to every model. Observed
first-24-hour RASS coverage was {rass_coverage['mimic4']:.1f}%,
{rass_coverage['mimic3']:.1f}%, and {rass_coverage['eicu']:.1f}% in MIMIC-IV,
MIMIC-III, and eICU-CRD. The outline's statement that RASS was fully available
in eICU-CRD is therefore not supported. eICU-CRD also lacks structured
urine-output and FiO2 tables in the accessed source; those channels remain
explicitly missing and are not imputed across databases.

Static predictors include age, an indicator for age older than 89 years, sex,
race, admission status, first-24-hour vital and GCS summaries, monitoring
intensity, available laboratory summaries, and mechanical-ventilation and
anticoagulation-related hemorrhage indicators when harmonized with observed
variation. MIMIC-III patients older than 89 years have DOB shifted to exactly
300 years before the first admission; the resulting sentinel age is recoded to
91 years to match MIMIC-IV `anchor_age`, and an `age_gt89` indicator is
retained. The eICU `> 89` category is handled identically rather than being
collapsed to 89. All {n_features} eligible static predictors end before the
landmark.

Static medians, standardization parameters, and early-stopping epochs are
estimated inside outer training folds. The eligible predictor matrix was
fixed by the source schema after the landmark boundary blocklist; no
missingness- or variance-based variable selection was applied, and outcome
labels were not used. Summary
XGBoost and the temporal neural models use the same pre-specified
{n_primary}-variable core subset of that eligible matrix, covering
demographics, baseline and summary GCS, vital-sign summaries, urine output,
FiO2, RASS, and monitoring intensity; the subset was fixed before any outcome
comparison. Discharge location, post-landmark events, and outcome-derived
variables are excluded.

## Models and comparison

The five outline baselines are {baseline_text}. The complete comparison set is
{model_text}. It includes DG-GRU, DG-GRU with temporal risk-trajectory
regularization (DG-GRU + TR), nested logistic stacking, time-reversed DG-GRU +
TR, time-permuted DG-GRU + TR, and DG-GRU with generic Laplace input noise at
scale 0.01. The Laplace control adds independent noise only to observed,
standardized time-series values and applies no temporal smoothness penalty.

Matched secondary ablations separate temporal information from the 24-hour
static summaries: DG-GRU + TR with the core static variables and the sequence
zeroed (`dggru_tr_noseq`), the static variables that exclude the 24-hour
trajectory summaries plus the observed sequence (`dggru_tr_static`), the
observed sequence with one placeholder static column (`dggru_tr_traj`), and a
GCS-observation trajectory mask
(`dggru_tr_gcsmask`). These ablations are internal-development analyses and are
not part of the external transport model set.

The ordinal head uses two cumulative-logit branches with shared hidden states,
corresponding to grade >=1 and grade 2. The TR arm constrains the auxiliary
latent deterioration-risk path; it does not directly constrain the ordinal
probability path. The primary mechanistic comparison is DG-GRU + TR versus
unregularized DG-GRU; other model differences are descriptive.

## Validation and uncertainty

- Development: patient-grouped stratified five-fold cross-validation repeated
  {config['cv_repeats']} times.
- Each outer training fold uses a patient-grouped inner holdout for early
  stopping. All neural arms use unregularized validation NLL for early stopping.
- XGBoost and sequence-network hyperparameters are selected from explicit
  prespecified candidate dictionaries by inner patient-grouped macro AUC. The
  18-point lambda grid is then searched with the selected sequence parameters.
  For every inner selection step, validation probabilities are averaged across
  the five fixed model seeds before candidate hyperparameters and
  lambda_monotone are scored. Selection never uses outer-test or external
  outcomes.
- The five fixed seeds are then fitted separately for outer out-of-fold
  predictions and averaged.
- Nested stacking uses five patient-grouped inner folds to generate member OOF
  probabilities within each outer training split, fits a multinomial logistic
  meta-model, and then refits members on the complete outer training split.
- Development point estimates are unweighted means of repeat-specific OOF
  metrics. Probability averaging across repeats is a secondary analysis.
- Each external model seed is fitted once in development and predictions are
  averaged over the five fixed seeds. No external refitting is used,
  and neural transport epochs use the median development best epoch.
- The nested stacking meta-model is fitted from development OOF predictions
  before transport and is not updated on external outcomes.
- Primary paired inference uses 2,000 conditional cluster-bootstrap replicates:
  patients for MIMIC-IV and MIMIC-III and hospitals for eICU. This interval
  conditions on the fitted OOF predictions and fixed CV partitions; it does not
  include retraining or epoch-selection uncertainty.
- Standard two-sided paired DeLong tests are unclustered, use the first CV
  partition only, and are secondary. The P024/P039 fixed primary Holm family
  contains 10 tests: the five baseline models (`xgb_static`, `xgb_summary`,
  `gru`, `grud`, `transformer`) x any/severe. DG-GRU, the GCS slope comparator,
  nested stacking, and the three temporal or smoothing controls are marked
  `exploratory_not_in_primary_holm_family` and are not members of that family.
  The DeLong family is not the primary eICU inference and does not replace
  `hospital_key`-clustered bootstrap intervals.
- Repeat-to-repeat SD summarizes combined CV-partition and training-seed
  variation.

## Performance measures

- Macro one-versus-rest AUC across grades 0, 1, and 2.
- One-versus-rest AUC for each individual grade.
- Cumulative AUC for grade 1 or higher and grade 2.
- Adjacent-class and non-adjacent-class error and mean ordinal error.
- Median-ordinal accuracy and quadratic weighted kappa.
- Multiclass Brier score and log loss.
- Calibration slope and intercept for any deterioration.
- Top-label and cumulative expected calibration error.
- Continuous and categorical NRI and IDI.
- Decision-curve net benefit for any deterioration over thresholds 10%-40%.
- Subgroup summaries for {subgroup_text}.

## Reporting figures

{figure_lines}

Figure 1 combines cohort construction with the ordinal outcome distribution.
Figure 3 contains development discrimination (A), development cumulative
calibration (B), and external transport discrimination (C). Figure 4 contains
calibration (A) and decision-curve analysis over the prespecified 10%-40%
threshold range (B). Figure 5 reports temporal structure and the three
regularization controls. Figure 6 contains model interpretation (A) and
subgroup analyses (B). Manuscript and checklist generation fail closed if any
listed PNG or PDF output is absent.

## Sensitivity analyses

{sensitivity_lines}

A missing RASS is never treated as evidence of the absence of sedation. A
variant that lacks the exact required source field or model-fitting interface
is retained in the audit table with an explicit unsupported or outcome-only
status; it is not silently replaced by a convenient proxy.

The lambda comparison includes a true no-regularization point
(lambda_monotone=0, lambda_smooth=0), component-isolation points, the primary
weights, and higher weights on the prespecified 18-point grid. No external
outcome is used to choose weights.

## Interpretation and subgroups

TreeSHAP ranks the top 15 Summary XGBoost predictors. Time-dependent absolute
gradient importance summarizes DG-GRU + TR by the 14 channels and ICU hours
0-23. Subgroup analyses cover the five factors listed above and are
exploratory. No explanation or subgroup result is used to select the primary
model.

## Reporting boundaries

The study does not claim that temporal regularization improves ordinal
prediction unless a stable incremental estimate is observed. A near-zero or
direction-inconsistent difference remains visible in the abstract, tables,
and conclusions. The study does not establish equivalence, causal effects,
clinical utility, or readiness for deployment. Patient-level predictions
remain in the local audit workspace and are not distributed in the reproduction
archive.
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
