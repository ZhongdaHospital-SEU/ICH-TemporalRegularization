"""Decompose the landmark-to-analysis cohort attrition without overlap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.common import (
    complete_icu_followup,
    death_in_icu_within_window,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "audits" / "cohort_exclusion_decomposition.json"


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def decompose(name: str) -> dict:
    frame = pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv")
    gcs = pd.read_csv(PROCESSED / name / "gcs_events.csv")
    n = len(frame)
    icu_out_hour = pd.to_numeric(frame["icu_out_hour"], errors="coerce")
    death_hour = pd.to_numeric(frame["death_hour_from_icu"], errors="coerce")
    death = _bool(frame["death_within_72h"])
    if "in_icu_at_primary_landmark" in frame:
        in_icu = _bool(frame["in_icu_at_primary_landmark"])
    else:
        in_icu = icu_out_hour.ge(24)
    if "alive_at_primary_landmark" in frame:
        alive = _bool(frame["alive_at_primary_landmark"])
    else:
        alive = ~death | death_hour.gt(24)
    eligible = in_icu & alive
    outcome_known = frame["outcome"].notna()
    baseline_known = pd.to_numeric(
        frame["baseline_gcs_max"], errors="coerce"
    ).notna()
    death_in_icu = death_in_icu_within_window(
        frame["death_within_72h"],
        death_hour,
        icu_out_hour,
    )
    followup_72h = icu_out_hour.ge(72) | death_in_icu
    complete_followup = (
        eligible
        & baseline_known
        & outcome_known
        & complete_icu_followup(
            frame["outcome"],
            icu_out_hour,
            death_in_icu,
        )
    )
    post = gcs[["stay_id", "time_hour", "gcs"]].copy()
    post["stay_id"] = pd.to_numeric(post["stay_id"], errors="coerce")
    post["time_hour"] = pd.to_numeric(post["time_hour"], errors="coerce")
    post["gcs"] = pd.to_numeric(post["gcs"], errors="coerce")
    post = post[
        post["stay_id"].notna()
        & post["time_hour"].ge(24)
        & post["time_hour"].lt(72)
        & post["gcs"].between(3, 15)
    ].copy()
    post["hour_bin"] = np.floor(post["time_hour"])
    post_counts = (
        post.groupby("stay_id")["hour_bin"]
        .nunique()
        .rename("n_gcs_post")
    )
    frame = frame.merge(post_counts, left_on="stay_id", right_index=True, how="left")
    frame["n_gcs_post"] = pd.to_numeric(
        frame["n_gcs_post"], errors="coerce"
    ).fillna(0).astype(int)

    assignments = pd.Series("included_primary", index=frame.index)
    assignments.loc[death & death_hour.le(24)] = (
        "death_at_or_before_24h_landmark"
    )
    assignments.loc[
        ~(death & death_hour.le(24)) & ~in_icu
    ] = "not_in_icu_at_24h_landmark"
    assignments.loc[
        in_icu & ~alive
    ] = "not_alive_at_24h_landmark"
    assignments.loc[in_icu & alive & ~baseline_known] = (
        "alive_at_landmark_without_valid_prelandmark_gcs"
    )
    known_baseline = in_icu & alive & baseline_known
    early_transfer = (
        known_baseline
        & icu_out_hour.lt(72)
        & ~death_in_icu
    )
    assignments.loc[known_baseline & ~outcome_known & early_transfer] = (
        "early_live_icu_transfer_before_72h_without_classifiable_outcome"
    )
    assignments.loc[
        known_baseline
        & ~outcome_known
        & ~early_transfer
    ] = "complete_followup_but_fewer_than_two_valid_postlandmark_gcs"
    primary = known_baseline & outcome_known
    assignments.loc[primary] = "included_primary"

    counts = assignments.value_counts().to_dict()
    included = frame.loc[primary]
    return {
        "dataset": name,
        "prefilter_n": int(n),
        "included_n": int(primary.sum()),
        "excluded_n": int((~primary).sum()),
        "included_complete_followup_n": int(complete_followup.sum()),
        "mutually_exclusive_reasons": {
            reason: int(counts.get(reason, 0))
            for reason in (
                "death_at_or_before_24h_landmark",
                "not_in_icu_at_24h_landmark",
                "not_alive_at_24h_landmark",
                "alive_at_landmark_without_valid_prelandmark_gcs",
                "complete_followup_but_fewer_than_two_valid_postlandmark_gcs",
                "early_live_icu_transfer_before_72h_without_classifiable_outcome",
                "included_primary",
            )
        },
        "included_outcomes": {
            str(grade): int(included["outcome"].eq(grade).sum())
            for grade in (0, 1, 2)
        },
        "included_deaths_within_72h": int(
            _bool(included["death_within_72h"]).sum()
        ),
        "included_deaths_with_at_least_two_postlandmark_gcs_n": int(
            (
                _bool(included["death_within_72h"])
                & pd.to_numeric(
                    included["n_gcs_post"], errors="coerce"
                ).ge(2)
            ).sum()
        ),
        "included_grade2_without_two_postlandmark_gcs_n": int(
            (
                included["outcome"].eq(2)
                & pd.to_numeric(
                    included["n_gcs_post"], errors="coerce"
                ).lt(2)
            ).sum()
        ),
    }


def main() -> None:
    report = {
        name: decompose(name) for name in ("mimic4", "mimic3", "eicu")
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, values in report.items():
        print(
            f"{name}: {values['prefilter_n']} -> {values['included_n']} "
            f"(excluded {values['excluded_n']})"
        )
        for reason, count in values["mutually_exclusive_reasons"].items():
            print(f"  {reason}: {count}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
