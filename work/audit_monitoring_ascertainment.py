"""Quantify how recorded GCS intensity relates to the ordinal endpoint.

The endpoint is defined on recorded GCS. A stay with more post-landmark GCS
observations has more opportunity to meet the confirmation rule, so the grade
distribution can depend on monitoring intensity. This audit reports that
relationship for each database and writes a machine-readable artifact for the
manuscript's limitations section.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED = Path("data/processed")
OUTPUT = Path("results/audits/monitoring_ascertainment.json")
MARKDOWN = Path("results/audits/monitoring_ascertainment.md")
DATASETS = ("mimic4", "mimic3", "eicu")
BINS = (1, 5, 11, 23, np.inf)
LABELS = ("2-5", "6-11", "12-23", "24+")


def _post_landmark_counts(gcs: pd.DataFrame, stay_ids: pd.Series) -> pd.Series:
    events = gcs[["stay_id", "time_hour", "gcs"]].copy()
    events["stay_id"] = pd.to_numeric(events["stay_id"], errors="coerce")
    events["time_hour"] = pd.to_numeric(events["time_hour"], errors="coerce")
    events["gcs"] = pd.to_numeric(events["gcs"], errors="coerce")
    events = events[
        events["stay_id"].isin(stay_ids)
        & events["time_hour"].ge(24)
        & events["time_hour"].lt(72)
        & events["gcs"].between(3, 15)
    ]
    events["hour_bin"] = np.floor(events["time_hour"])
    return (
        events.groupby("stay_id")["hour_bin"]
        .nunique()
        .rename("post_landmark_gcs_hours")
    )


def _pre_landmark_counts(gcs: pd.DataFrame, stay_ids: pd.Series) -> pd.Series:
    events = gcs[["stay_id", "time_hour", "gcs"]].copy()
    events["stay_id"] = pd.to_numeric(events["stay_id"], errors="coerce")
    events["time_hour"] = pd.to_numeric(events["time_hour"], errors="coerce")
    events["gcs"] = pd.to_numeric(events["gcs"], errors="coerce")
    events = events[
        events["stay_id"].isin(stay_ids)
        & events["time_hour"].ge(0)
        & events["time_hour"].lt(24)
        & events["gcs"].between(3, 15)
    ]
    events["hour_bin"] = np.floor(events["time_hour"])
    return (
        events.groupby("stay_id")["hour_bin"]
        .nunique()
        .rename("pre_landmark_gcs_hours")
    )


def _stratified_table(frame: pd.DataFrame, column: str) -> list[dict]:
    total = len(frame)
    rows: list[dict] = []
    for label in LABELS:
        subset = frame[frame[column].eq(label)]
        if subset.empty:
            continue
        rows.append(
            {
                "stratum": label,
                "n": int(len(subset)),
                "share_percent": round(100 * len(subset) / total, 1),
                "grade_0_percent": round(100 * (subset["outcome"] == 0).mean(), 1),
                "grade_1_percent": round(100 * (subset["outcome"] == 1).mean(), 1),
                "grade_2_percent": round(100 * (subset["outcome"] == 2).mean(), 1),
                "any_deterioration_percent": round(
                    100 * (subset["outcome"] > 0).mean(), 1
                ),
            }
        )
    return rows


def _spearman(frame: pd.DataFrame, left: str, right: str) -> dict:
    values = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3:
        return {"rho": None, "n": int(len(values))}
    rho = values[left].corr(values[right], method="spearman")
    return {"rho": round(float(rho), 3), "n": int(len(values))}


def main() -> None:
    audit: dict[str, object] = {
        "definition": (
            "Distinct ICU hours with a valid total GCS in [24, 72) per stay, "
            "stratified against the three-level ordinal endpoint."
        ),
        "datasets": {},
    }
    markdown = [
        "# Monitoring intensity and endpoint ascertainment",
        "",
        "Event rates by the number of distinct post-landmark ICU hours with a",
        "valid total GCS. The endpoint is defined on recorded GCS, so these",
        "strata describe ascertainment, not a causal effect of monitoring.",
        "",
    ]
    for dataset in DATASETS:
        root = PROCESSED / dataset
        cohort = pd.read_csv(root / "cohort.csv")
        gcs = pd.read_csv(root / "gcs_events.csv")
        stay_ids = cohort["stay_id"]
        post = _post_landmark_counts(gcs, stay_ids)
        pre = _pre_landmark_counts(gcs, stay_ids)
        frame = cohort[["stay_id", "outcome"]].copy()
        frame["post_landmark_gcs_hours"] = (
            frame["stay_id"].map(post).fillna(0).astype(int)
        )
        frame["pre_landmark_gcs_hours"] = (
            frame["stay_id"].map(pre).fillna(0).astype(int)
        )
        frame["post_stratum"] = pd.cut(
            frame["post_landmark_gcs_hours"],
            bins=BINS,
            labels=LABELS,
            right=True,
        )
        pre_rank = frame["pre_landmark_gcs_hours"].rank(method="first")
        frame["pre_stratum"] = pd.qcut(
            pre_rank, 4, labels=("Q1", "Q2", "Q3", "Q4")
        )
        post_rows = _stratified_table(frame, "post_stratum")
        pre_rows = [
            {
                "stratum": label,
                "n": int((frame["pre_stratum"] == label).sum()),
                "any_deterioration_percent": round(
                    100
                    * (
                        frame.loc[frame["pre_stratum"] == label, "outcome"] > 0
                    ).mean(),
                    1,
                ),
                "grade_2_percent": round(
                    100
                    * (
                        frame.loc[frame["pre_stratum"] == label, "outcome"] == 2
                    ).mean(),
                    1,
                ),
            }
            for label in ("Q1", "Q2", "Q3", "Q4")
        ]
        audit["datasets"][dataset] = {
            "n": int(len(frame)),
            "grade_counts": {
                str(grade): int((frame["outcome"] == grade).sum())
                for grade in (0, 1, 2)
            },
            "any_deterioration_percent": round(
                100 * (frame["outcome"] > 0).mean(), 1
            ),
            "post_landmark_gcs_hours": {
                "median": float(frame["post_landmark_gcs_hours"].median()),
                "q1": float(frame["post_landmark_gcs_hours"].quantile(0.25)),
                "q3": float(frame["post_landmark_gcs_hours"].quantile(0.75)),
                "minimum": int(frame["post_landmark_gcs_hours"].min()),
            },
            "by_post_landmark_gcs_hours": post_rows,
            "by_pre_landmark_gcs_quartile": pre_rows,
            "spearman_post_hours_vs_grade": _spearman(
                frame, "post_landmark_gcs_hours", "outcome"
            ),
            "spearman_pre_hours_vs_grade": _spearman(
                frame, "pre_landmark_gcs_hours", "outcome"
            ),
        }
        markdown.extend(
            [
                f"## {dataset} (n={len(frame):,})",
                "",
                "| Post-landmark GCS hours | n | Share (%) | Grade 0 (%) | "
                "Grade 1 (%) | Grade 2 (%) | Any deterioration (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        markdown.extend(
            "| {stratum} | {n:,} | {share_percent} | {grade_0_percent} | "
            "{grade_1_percent} | {grade_2_percent} | "
            "{any_deterioration_percent} |".format(**row)
            for row in post_rows
        )
        markdown.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    MARKDOWN.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(audit["datasets"], indent=1))
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MARKDOWN}")


if __name__ == "__main__":
    main()
