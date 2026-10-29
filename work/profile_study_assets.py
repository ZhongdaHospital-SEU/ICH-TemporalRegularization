"""Profile the current analytic cohorts and sequence completeness.

The report is descriptive and read-only. It exists to prevent stale outline
counts from being copied into the manuscript after a cohort rebuild.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "audits"
OUTPUT = RESULTS / "study_asset_profile.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_dataset(name: str) -> dict:
    root = PROCESSED / name
    cohort = pd.read_csv(root / "cohort.csv")
    prefilter = pd.read_csv(root / "cohort_pre_filter.csv")
    flow = pd.read_csv(root / "cohort_flow.csv")
    with np.load(root / "sequences.npz", allow_pickle=True) as archive:
        values = archive["values"]
        masks = archive["masks"]
        ids = archive["stay_id"].astype(np.int64)
        channels = archive["channels"].astype(str).tolist()
    if not np.array_equal(ids, cohort["stay_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"{name}: sequence and cohort IDs are not aligned")
    if values.shape != masks.shape or values.shape[:2] != (len(cohort), 24):
        raise ValueError(f"{name}: unexpected sequence shape {values.shape}")
    observed = masks.sum(axis=1) > 0
    channel_coverage = {
        channel: {
            "observed_hours": int(masks[:, :, index].sum()),
            "stays_with_any_observation": int(observed[:, index].sum()),
            "stay_coverage": float(observed[:, index].mean()),
        }
        for index, channel in enumerate(channels)
    }
    outcome_counts = {
        str(grade): int((cohort["outcome"] == grade).sum())
        for grade in (0, 1, 2)
    }
    death_72h = (
        int(cohort["death_within_72h"].fillna(False).astype(bool).sum())
        if "death_within_72h" in cohort
        else None
    )
    any_event = outcome_counts["1"] + outcome_counts["2"]
    result = {
        "n": int(len(cohort)),
        "grade_counts": outcome_counts,
        "any_grade_1_or_2": any_event,
        "any_grade_1_or_2_rate": float(any_event / len(cohort)),
        "death_within_72h": death_72h,
        "death_within_72h_rate": (
            float(death_72h / len(cohort)) if death_72h is not None else None
        ),
        "sequence_shape": list(values.shape),
        "channels": channels,
        "channel_coverage": channel_coverage,
        "flow": flow.to_dict(orient="records"),
        "hashes": {
            "cohort.csv": sha256_file(root / "cohort.csv"),
            "cohort_flow.csv": sha256_file(root / "cohort_flow.csv"),
            "sequences.npz": sha256_file(root / "sequences.npz"),
        },
    }
    if "age_raw" in cohort:
        age_raw = pd.to_numeric(cohort["age_raw"], errors="coerce")
        result["mimic3_age_sentinel"] = {
            "age_300_or_more": int(age_raw.ge(300).sum()),
            "age_gt89": int(cohort["age_gt89"].fillna(0).astype(bool).sum()),
            "min_raw": float(age_raw.min()),
            "max_raw": float(age_raw.max()),
        }
    if "eligible_at_primary_landmark" in cohort:
        result["eligible_at_primary_landmark"] = {
            "false": int((~cohort["eligible_at_primary_landmark"].astype(bool)).sum()),
            "true": int(cohort["eligible_at_primary_landmark"].astype(bool).sum()),
        }
    if "eligible_at_primary_landmark" in prefilter:
        eligible = prefilter["eligible_at_primary_landmark"].astype(bool)
        result["prefilter"] = {
            "n": int(len(prefilter)),
            "eligible_at_primary_landmark_false": int((~eligible).sum()),
            "eligible_at_primary_landmark_true": int(eligible.sum()),
            "grade_counts_among_landmark_eligible": {
                str(grade): int(
                    prefilter.loc[eligible, "outcome"].eq(grade).sum()
                )
                for grade in (0, 1, 2)
            },
            "grade_counts_all": {
                str(grade): int(prefilter["outcome"].eq(grade).sum())
                for grade in (0, 1, 2)
            },
        }
    if "recorded_intubation_24_72h" in cohort:
        recorded = pd.to_numeric(
            cohort["recorded_intubation_24_72h"], errors="coerce"
        )
        result["recorded_intubation_24_72h"] = {
            "observed": int(recorded.notna().sum()),
            "positive": int(recorded.eq(1).sum()),
            "verified_unplanned": False,
        }
    if "verified_unplanned_intubation_24_72h" in cohort:
        verified = pd.to_numeric(
            cohort["verified_unplanned_intubation_24_72h"], errors="coerce"
        )
        result["verified_unplanned_intubation_24_72h"] = {
            "observed": int(verified.notna().sum()),
            "positive": int(verified.eq(1).sum()),
        }
    if "outcome_recorded_intubation_composite" in cohort:
        proxy = pd.to_numeric(
            cohort["outcome_recorded_intubation_composite"], errors="coerce"
        )
        result["outcome_recorded_intubation_composite"] = {
            "observed": int(proxy.notna().sum()),
            "grade_2": int(proxy.eq(2).sum()),
            "role": "sensitivity_proxy_only",
        }
    if "p020_status" in cohort and "p020_complete" in cohort:
        result["p020"] = {
            "status": sorted(
                cohort["p020_status"].dropna().astype(str).unique().tolist()
            ),
            "complete_any": bool(
                cohort["p020_complete"].fillna(False).astype(bool).any()
            ),
        }
    return result


def main() -> None:
    report = {
        "datasets": {
            name: profile_dataset(name) for name in ("mimic4", "mimic3", "eicu")
        }
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, profile in report["datasets"].items():
        print(
            f"{name}: n={profile['n']} grades={profile['grade_counts']} "
            f"death72={profile['death_within_72h']}"
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
