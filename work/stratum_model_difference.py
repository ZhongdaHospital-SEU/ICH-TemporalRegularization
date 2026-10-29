"""Model contrasts within strata of post-landmark GCS documentation.

The endpoint is defined on recorded GCS, so a model could in principle rank
stays by how densely the score was charted.  This script asks whether the
contrast between the dynamic-graph model and the tabular summary model holds
inside each documentation stratum, and reports a patient-cluster bootstrap
interval for each stratum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "processed" / "mimic4" / "cohort.csv"
GCS = ROOT / "data" / "processed" / "mimic4" / "gcs_events.csv"
PRED = ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz"
OUTPUT = ROOT / "results" / "audits" / "stratum_model_difference.json"

BINS = [(2, 5), (6, 11), (12, 23), (24, 10_000)]
LABELS = ["2-5", "6-11", "12-23", "24+"]
REPLICATES = 2000
SEED = 20261028
RNG = np.random.default_rng(SEED)


def post_landmark_hours() -> pd.Series:
    events = pd.read_csv(GCS, usecols=["stay_id", "time_hour", "gcs"])
    events["stay_id"] = pd.to_numeric(events["stay_id"], errors="coerce")
    events["time_hour"] = pd.to_numeric(events["time_hour"], errors="coerce")
    events["gcs"] = pd.to_numeric(events["gcs"], errors="coerce")
    events = events[
        events["time_hour"].ge(24)
        & events["time_hour"].lt(72)
        & events["gcs"].between(3, 15)
    ]
    events["hour_bin"] = np.floor(events["time_hour"])
    return events.groupby("stay_id")["hour_bin"].nunique().rename("hours")


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
                for grade in range(3)
            ]
        )
    )


def mean_over_repeats(
    labels: np.ndarray,
    payload,
    keys: list[str],
    rows: np.ndarray,
) -> float:
    return float(
        np.mean([macro_auc(labels[rows], payload[key][rows]) for key in keys])
    )


def paired_interval(
    labels: np.ndarray,
    payload,
    left: list[str],
    right: list[str],
    rows: np.ndarray,
    subjects: np.ndarray,
) -> list[float]:
    unique = np.unique(subjects[rows])
    index = {key: rows[subjects[rows] == key] for key in unique}
    draws = []
    for _ in range(REPLICATES):
        chosen = RNG.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([index[key] for key in chosen])
        if np.unique(labels[sampled]).size < 3:
            continue
        draws.append(
            mean_over_repeats(labels, payload, left, sampled)
            - mean_over_repeats(labels, payload, right, sampled)
        )
    return [
        round(float(np.quantile(draws, 0.025)), 3),
        round(float(np.quantile(draws, 0.975)), 3),
    ]


def main() -> None:
    stored = np.load(PRED, allow_pickle=True)
    payload = {name: stored[name] for name in stored.files}
    labels = np.asarray(payload["labels"]).astype(int)
    subjects = np.asarray(payload["subject_key"]).astype(str)
    stay_ids = payload["stay_id"]
    hours = post_landmark_hours()
    per_stay = pd.Series(stay_ids).map(hours).fillna(0).astype(int).to_numpy()

    xgb = [f"repeat{index}_xgb_summary" for index in range(5)]
    dggru = [f"repeat{index}_dggru_tr" for index in range(5)]
    blend: list[str] = []
    for index in range(5):
        key = f"repeat{index}_blend"
        if key not in payload:
            payload[key] = 0.5 * (
                np.asarray(payload[f"repeat{index}_xgb_summary"])
                + np.asarray(payload[f"repeat{index}_dggru_tr"])
            )
        blend.append(key)

    result = {"definition": "distinct ICU hours with a valid total GCS in [24,72)"}
    for label, (low, high) in zip(LABELS, BINS):
        rows = np.flatnonzero((per_stay >= low) & (per_stay <= high))
        result[label] = {
            "n": int(len(rows)),
            "events": int((labels[rows] > 0).sum()),
            "xgb": round(mean_over_repeats(labels, payload, xgb, rows), 3),
            "dggru_tr": round(mean_over_repeats(labels, payload, dggru, rows), 3),
            "blend": round(mean_over_repeats(labels, payload, blend, rows), 3),
            "dggru_minus_xgb": round(
                mean_over_repeats(labels, payload, dggru, rows)
                - mean_over_repeats(labels, payload, xgb, rows),
                3,
            ),
            "blend_minus_xgb": round(
                mean_over_repeats(labels, payload, blend, rows)
                - mean_over_repeats(labels, payload, xgb, rows),
                3,
            ),
            "dggru_minus_xgb_ci": paired_interval(
                labels, payload, dggru, xgb, rows, subjects
            ),
            "blend_minus_xgb_ci": paired_interval(
                labels, payload, blend, xgb, rows, subjects
            ),
        }
        print(label, json.dumps(result[label]), flush=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
