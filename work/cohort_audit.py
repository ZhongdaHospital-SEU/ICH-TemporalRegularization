"""Audit rebuilt cohorts for structure, leakage, and endpoint reproducibility."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from src.ich_dynamic.common import PROCESSED, add_outcome, load_config
from src.ich_dynamic.train_evaluate import POST_LANDMARK_COLUMNS, load_dataset


def audit_dataset(name: str, feature_columns: list[str] | None = None) -> dict:
    frame = pd.read_csv(PROCESSED / name / "cohort.csv")
    gcs = pd.read_csv(PROCESSED / name / "gcs_events.csv")
    hourly = pd.read_csv(PROCESSED / name / "hourly.csv")
    prefilter = pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv")
    dataset = load_dataset(name, feature_columns=feature_columns)
    config = load_config()

    recomputed = add_outcome(
        prefilter,
        gcs,
        include_intubation=False,
    )
    exact = prefilter[["stay_id", "outcome"]].merge(
        recomputed[["stay_id", "outcome"]],
        on="stay_id",
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    endpoint_mismatch = int(
        (
            exact["outcome_saved"].fillna(-1).astype(int)
            != exact["outcome_recomputed"].fillna(-1).astype(int)
        ).sum()
    )
    endpoint_mismatch_stays = [
        int(value)
        for value in exact.loc[
            exact["outcome_saved"].fillna(-1).astype(int)
            != exact["outcome_recomputed"].fillna(-1).astype(int),
            "stay_id",
        ]
    ]
    mismatch_details = []
    for stay_id in endpoint_mismatch_stays:
        saved = prefilter.loc[prefilter["stay_id"].eq(stay_id)].iloc[0]
        fresh = recomputed.loc[recomputed["stay_id"].eq(stay_id)].iloc[0]
        mismatch_details.append(
            {
                "stay_id": stay_id,
                "outcome_saved": int(saved["outcome"]),
                "outcome_recomputed": int(fresh["outcome"]),
                "baseline_gcs_max": float(saved["baseline_gcs_max"]),
                "death_within_72h": bool(saved["death_within_72h"]),
                "max_sustained_drop_saved": float(
                    saved["max_sustained_drop"]
                ),
                "max_sustained_drop_recomputed": float(
                    fresh["max_sustained_drop"]
                ),
                "post_events": gcs.loc[
                    gcs["stay_id"].eq(stay_id)
                    & gcs["time_hour"].between(24, 72, inclusive="left"),
                    ["time_hour", "gcs"],
                ].sort_values("time_hour").to_dict(orient="records"),
            }
        )
    rass = pd.read_csv(PROCESSED / name / "cohort.csv", usecols=["min_rass_24h",
        "deep_sedation_observed_24h",
        "any_rass_le_minus2_24h",
        "any_deep_sedation_24h",
    ])
    duplicate_hours = int(
        hourly.duplicated(["stay_id", "hour", "channel"]).sum()
    )
    with np.load(PROCESSED / name / "sequences.npz") as data:
        sequence_ids_match = bool(
            np.array_equal(data["stay_id"], frame["stay_id"].to_numpy())
        )
    return {
        "dataset": name,
        "n": int(len(frame)),
        "grades": {
            str(grade): int((frame["outcome"] == grade).sum())
            for grade in range(3)
        },
        "deaths_72h": int(frame["death_within_72h"].astype(bool).sum()),
        "unique_stays": bool(frame["stay_id"].is_unique),
        "sequences": list(dataset.sequences.shape),
        "masks": list(dataset.masks.shape),
        "sequence_ids_match": sequence_ids_match,
        "duplicate_hourly_keys": duplicate_hours,
        "post_landmark_leaks": sorted(
            POST_LANDMARK_COLUMNS.intersection(dataset.static_all.columns)
        ),
        "feature_count": int(dataset.static_all.shape[1]),
        "rass_observed": int(pd.to_numeric(
            rass["deep_sedation_observed_24h"], errors="coerce"
        ).fillna(0).astype(bool).sum()),
        "rass_missing": int(pd.to_numeric(
            rass["deep_sedation_observed_24h"], errors="coerce"
        ).fillna(0).eq(0).sum()),
        "deep_sedation_rass_le_minus3": int(pd.to_numeric(
            rass["any_deep_sedation_24h"], errors="coerce"
        ).fillna(0).eq(1).sum()),
        "endpoint_mismatch": endpoint_mismatch,
        "endpoint_mismatch_stays": endpoint_mismatch_stays,
        "endpoint_mismatch_details": mismatch_details,
        "primary_followup_complete": int(
            frame["primary_followup_complete"].astype(bool).sum()
        ),
        "missing_baseline_gcs": int(frame["baseline_gcs_max"].isna().sum()),
        "config": {
            "landmark_hours": config["landmark_hours"],
            "outcome_end_hours": config["outcome_end_hours"],
            "sequence_hours": config["sequence_hours"],
        },
    }


def main() -> None:
    feature_columns = list(load_dataset("mimic4").static_all.columns)
    output = {
        name: audit_dataset(name, feature_columns)
        for name in ("mimic4", "mimic3", "eicu")
    }
    path = Path("work") / "cohort_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
