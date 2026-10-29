"""Paired uncertainty summary for the non-proportional ordinal-head screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()

    formal_path = te.RESULTS / "predictions" / "mimic4_cv_repeats.npz"
    candidate_path = (
        te.RESULTS
        / "predictions"
        / "experiment_nonproportional_tr_repeats.npz"
    )
    result_path = te.RESULTS / "experiments" / "nonproportional_tr.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    repeats = int(result["cv_repeats"])
    seed = int(te.load_config()["random_seed"])

    with np.load(formal_path, allow_pickle=True) as formal:
        labels = formal["labels"].astype(int)
        groups = formal["subject_key"].astype(str)
        reference = [
            formal[f"repeat{repeat}_dggru_tr"].astype(np.float64)
            for repeat in range(repeats)
        ]
    with np.load(candidate_path, allow_pickle=True) as candidate:
        if not np.array_equal(labels, candidate["labels"].astype(int)):
            raise ValueError("Candidate labels do not match the formal run")
        if not np.array_equal(groups, candidate["subject_key"].astype(str)):
            raise ValueError("Candidate patients do not match the formal run")
        nonproportional = [
            candidate[f"repeat{repeat}_dggru_tr"].astype(np.float64)
            for repeat in range(repeats)
        ]

    reference_auc = [
        float(te.multiclass_auc(labels, probabilities))
        for probabilities in reference
    ]
    candidate_auc = [
        float(te.multiclass_auc(labels, probabilities))
        for probabilities in nonproportional
    ]
    differences = [
        candidate_value - reference_value
        for candidate_value, reference_value in zip(
            candidate_auc, reference_auc
        )
    ]
    interval = te.paired_auc_difference_repeated(
        labels,
        nonproportional,
        reference,
        groups,
        replicates=args.replicates,
        seed=seed,
    )
    result["metrics"]["paired_nonproportional_minus_formal"] = {
        "macro_auc_difference": {
            "point": float(np.mean(differences)),
            "ci": [float(interval[0]), float(interval[1])],
            "repeat_values": [float(value) for value in differences],
            "positive_repeats": int(np.sum(np.asarray(differences) > 0)),
            "repeats": repeats,
        }
    }
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            result["metrics"]["paired_nonproportional_minus_formal"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
