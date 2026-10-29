"""Descriptive audit of imaging-feature coverage and univariate signal."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    target = labels[valid] >= 1
    if len(target) == 0 or target.min() == target.max():
        return float("nan")
    return float(roc_auc_score(target, values[valid]))


def main() -> None:
    cohort = pd.read_csv(ROOT / "data" / "processed" / "mimic4" / "cohort.csv")
    imaging = pd.read_csv(
        ROOT / "data" / "processed" / "mimic4" / "imaging_features_v2.csv"
    )
    frame = cohort.merge(imaging, on="stay_id", how="left")
    patients = pd.read_csv(
        Path(r"D:\TT paper\0811Temporal Pathway")
        / "mimic-iv-3.1"
        / "hosp"
        / "patients.csv.gz",
        usecols=["subject_id", "anchor_year_group"],
    )
    frame = frame.merge(patients, on="subject_id", how="left")
    frame["supported_note_era"] = frame["anchor_year_group"].isin(
        ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
    )
    subgroup = frame[frame["supported_note_era"]].copy()
    available = subgroup[subgroup["imaging_available"].eq(1)].copy()
    features = [
        "imaging_ivh",
        "imaging_hydrocephalus",
        "imaging_midline_shift",
        "imaging_herniation",
        "imaging_lobar",
        "imaging_deep",
        "imaging_brainstem",
        "imaging_cerebellar",
        "imaging_posterior_fossa",
    ]
    rows = []
    for feature in features:
        values = pd.to_numeric(available[feature], errors="coerce").to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "available_n": int(np.isfinite(values).sum()),
                "positive_n": int(np.nansum(values > 0)),
                "any_deterioration_auc": safe_auc(
                    available["outcome"].to_numpy(int), values
                ),
            }
        )
    payload = {
        "full_n": int(len(frame)),
        "supported_era_n": int(len(subgroup)),
        "supported_era_available_n": int(len(available)),
        "supported_era_coverage": float(len(available) / max(1, len(subgroup))),
        "available_outcomes": {
            str(grade): int((available["outcome"] == grade).sum())
            for grade in (0, 1, 2)
        },
        "missing_outcomes": {
            str(grade): int(
                (
                    subgroup.loc[
                        subgroup["imaging_available"].fillna(0).eq(0), "outcome"
                    ]
                    == grade
                ).sum()
            )
            for grade in (0, 1, 2)
        },
        "univariate_features": rows,
    }
    output = ROOT / "results" / "experiments" / "imaging_signal_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
