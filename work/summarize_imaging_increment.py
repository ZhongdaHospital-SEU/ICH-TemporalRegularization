"""Add paired uncertainty estimates to the imaging increment experiment."""
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

    result_path = te.RESULTS / "experiments" / "imaging_increment.json"
    prediction_path = (
        te.RESULTS
        / "predictions"
        / "experiment_imaging_increment_repeats.npz"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    repeats = int(result["cv_repeats"])
    seed = int(te.load_config()["random_seed"])

    with np.load(prediction_path, allow_pickle=True) as data:
        labels = data["labels"].astype(int)
        groups = data["subject_key"].astype(str)
        baseline = [
            data[f"repeat{repeat}_baseline"].astype(np.float64)
            for repeat in range(repeats)
        ]
        imaging = [
            data[f"repeat{repeat}_imaging_added"].astype(np.float64)
            for repeat in range(repeats)
        ]

    baseline_auc = [
        float(te.point_metrics(labels, probabilities)["macro_auc"])
        for probabilities in baseline
    ]
    imaging_auc = [
        float(te.point_metrics(labels, probabilities)["macro_auc"])
        for probabilities in imaging
    ]
    differences = [
        added - reference
        for added, reference in zip(imaging_auc, baseline_auc)
    ]
    interval = te.paired_auc_difference_repeated(
        labels,
        imaging,
        baseline,
        groups,
        replicates=args.replicates,
        seed=seed,
    )
    result["metrics"]["paired_imaging_minus_baseline"] = {
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
    print(json.dumps(result["metrics"]["paired_imaging_minus_baseline"], indent=2))
    print(f"Updated {result_path}")


if __name__ == "__main__":
    main()
