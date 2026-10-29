"""Validate the rebuilt cohorts before any landmark or model run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ich_dynamic.common import CHANNELS, PROCESSED, RESULTS


DATASETS = ("mimic4", "mimic3", "eicu")
LEGACY_ENDPOINT_COLUMNS = {
    "unplanned_intubation_24_72h",
    "outcome_with_intubation",
}
REQUIRED_ENDPOINT_COLUMNS = {
    "recorded_intubation_24_72h",
    "outcome_recorded_intubation_composite",
    "primary_outcome_includes_intubation",
    "p020_status",
    "p020_complete",
    "recorded_intubation_composite_role",
    "recorded_intubation_composite_status",
}


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _coverage(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").notna().mean())


def _positive_coverage(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.gt(0).fillna(False).mean())


def verify_dataset(dataset: str) -> dict[str, Any]:
    root = PROCESSED / dataset
    cohort = pd.read_csv(root / "cohort.csv")
    prefilter = pd.read_csv(root / "cohort_pre_filter.csv")
    flow = pd.read_csv(root / "cohort_flow.csv")
    gcs_events = pd.read_csv(root / "gcs_events.csv")
    values = np.load(root / "sequences.npz")

    errors: list[str] = []
    legacy = sorted(LEGACY_ENDPOINT_COLUMNS.intersection(cohort.columns))
    if legacy:
        errors.append(f"legacy endpoint columns remain: {legacy}")
    missing = sorted(REQUIRED_ENDPOINT_COLUMNS.difference(cohort.columns))
    if missing:
        errors.append(f"required endpoint columns are missing: {missing}")

    candidate_rows = flow[flow["reason_code"].eq("source_common_landmark_candidates")]
    included_rows = flow[
        flow["reason_code"].isin(
            ("primary_analytic_cohort", "primary_complete_followup")
        )
    ]
    if len(candidate_rows) != 1 or len(included_rows) != 1:
        errors.append("cohort flow must contain one candidate and one included row")
        candidate_n = included_n = -1
    else:
        candidate_n = int(candidate_rows.iloc[0]["n"])
        included_n = int(included_rows.iloc[0]["n"])
    exclusions = flow[flow["step"].astype(str).str.startswith("Excluded:")]
    exclusion_sum = int(pd.to_numeric(exclusions["n"], errors="raise").sum())
    if candidate_n >= 0 and candidate_n - exclusion_sum != included_n:
        errors.append(
            "flow exclusions do not sum to candidate minus included: "
            f"{candidate_n} - {exclusion_sum} != {included_n}"
        )
    if included_n != len(cohort):
        errors.append(f"flow included n={included_n} differs from cohort n={len(cohort)}")
    if flow["reason_code"].duplicated().any():
        errors.append("cohort flow reason codes are not unique")

    counts = {
        int(grade): int(cohort["outcome"].eq(grade).sum())
        for grade in (0, 1, 2)
    }
    for grade, count in counts.items():
        recorded = flow[flow["reason_code"].eq(f"outcome_{grade}")]
        if len(recorded) != 1 or int(recorded.iloc[0]["n"]) != count:
            errors.append(f"outcome_{grade} flow count differs from cohort")

    if "p020_status" in cohort and set(cohort["p020_status"].dropna()) != {"partial"}:
        errors.append("P020 status is not uniformly partial")
    if "p020_complete" in cohort and _bool(cohort["p020_complete"]).any():
        errors.append("P020 is marked complete despite no verified unplanned field")
    if (
        "primary_outcome_includes_intubation" in cohort
        and _bool(cohort["primary_outcome_includes_intubation"]).any()
    ):
        errors.append("primary outcome is marked as including intubation")

    # The endpoint counts any death within 72 h of ICU admission; a death after
    # a live ICU discharge is still an observed outcome in the window.
    death = _bool(cohort["death_within_72h"])
    drop = pd.to_numeric(cohort["max_sustained_drop"], errors="coerce")
    severe = drop.ge(4).fillna(False)
    verified = (
        _bool(cohort["verified_unplanned_intubation_24_72h"])
        if "verified_unplanned_intubation_24_72h" in cohort
        else pd.Series(False, index=cohort.index)
    )
    outcome = pd.to_numeric(cohort["outcome"], errors="raise")
    if not (death | severe | verified).loc[outcome.eq(2)].all():
        errors.append("some Grade 2 records lack death, severe GCS drop, or verified event")
    if (death | severe | verified).loc[outcome.lt(2)].any():
        errors.append("some Grade 0/1 records satisfy a Grade 2 criterion")
    death_only_grade2 = int((outcome.eq(2) & death & ~severe).sum())
    post = gcs_events[
        gcs_events["time_hour"].between(24.0, 72.0, inclusive="left")
        & pd.to_numeric(gcs_events["gcs"], errors="coerce").between(3.0, 15.0)
    ].copy()
    if post.empty:
        post_hour_count = pd.Series(0, index=cohort.index, dtype=np.int64)
    else:
        post["hour_bin"] = np.floor(
            pd.to_numeric(post["time_hour"], errors="raise")
        )
        hours_by_stay = post.groupby("stay_id")["hour_bin"].nunique()
        post_hour_count = (
            cohort["stay_id"].map(hours_by_stay).fillna(0).astype(np.int64)
        )
    sparse_post_gcs_death_grade2 = int(
        (outcome.eq(2) & death & post_hour_count.lt(2)).sum()
    )

    if tuple(values["values"].shape) != (len(cohort), 24, len(CHANNELS)):
        errors.append(
            f"sequence values shape {values['values'].shape} does not match "
            f"{(len(cohort), 24, len(CHANNELS))}"
        )
    if tuple(values["masks"].shape) != tuple(values["values"].shape):
        errors.append("sequence masks and values have different shapes")
    if not np.array_equal(
        values["stay_id"].astype(np.int64),
        cohort["stay_id"].astype(np.int64).to_numpy(),
    ):
        errors.append("sequence stay_ids differ from cohort.csv order")

    age_sentinel: dict[str, Any] | None = None
    if "age_raw" in cohort:
        sentinel = pd.to_numeric(cohort["age_raw"], errors="coerce").ge(300)
        mapped = pd.to_numeric(cohort.loc[sentinel, "age"], errors="coerce")
        age_sentinel = {
            "n": int(sentinel.sum()),
            "mapped_values": sorted(mapped.dropna().unique().tolist()),
        }
        if sentinel.any() and not mapped.dropna().between(90, 92).all():
            errors.append("age >= 300 sentinel was not mapped to 90-92 years")

    rass_observed = (
        _bool(cohort["deep_sedation_observed_24h"])
        if "deep_sedation_observed_24h" in cohort
        else pd.Series(False, index=cohort.index)
    )
    result = {
        "dataset": dataset,
        "candidate_n": candidate_n,
        "cohort_n": len(cohort),
        "excluded_n": int(candidate_n - len(cohort)),
        "outcome_counts": counts,
        "exclusion_sum": exclusion_sum,
        "prefilter_n": len(prefilter),
        "p020_status": sorted(cohort["p020_status"].dropna().astype(str).unique()),
        "p020_complete_any": bool(_bool(cohort["p020_complete"]).any()),
        "sequence_shape": list(values["values"].shape),
        "age_sentinel": age_sentinel,
        "death_only_grade2_without_severe_gcs_n": death_only_grade2,
        "grade2_with_fewer_than_two_postlandmark_gcs_n": (
            sparse_post_gcs_death_grade2
        ),
        "rass_observed_n": int(rass_observed.sum()),
        "rass_observed_proportion": float(rass_observed.mean()),
        "urine_output_observed_proportion": _positive_coverage(
            cohort, "urine_output_n"
        ),
        "fio2_observed_proportion": _positive_coverage(cohort, "fio2_n"),
        "errors": errors,
    }
    if errors:
        raise ValueError(f"{dataset} cohort validation failed: {'; '.join(errors)}")
    return result


def main() -> None:
    report = {
        "datasets": {dataset: verify_dataset(dataset) for dataset in DATASETS}
    }
    output = RESULTS / "audits" / "strict_cohort_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for dataset, record in report["datasets"].items():
        print(
            f"{dataset}: n={record['cohort_n']} "
            f"grades={record['outcome_counts']} "
            "death_only_grade2="
            f"{record['death_only_grade2_without_severe_gcs_n']} "
            "grade2_lt2_post_gcs="
            f"{record['grade2_with_fewer_than_two_postlandmark_gcs_n']} "
            f"P020={record['p020_status']}"
        )
    print(output)


if __name__ == "__main__":
    main()
