"""Patient-cluster bootstrap for the repeat-averaged equal-weight blend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = np.load(
    ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz",
    allow_pickle=True,
)
LABELS = PAYLOAD["labels"]
CLUSTERS = PAYLOAD["subject_key"].astype(str)
BLENDS = [
    0.5 * PAYLOAD[f"repeat{index}_xgb_summary"]
    + 0.5 * PAYLOAD[f"repeat{index}_dggru_tr"]
    for index in range(5)
]


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
                for grade in range(3)
            ]
        )
    )


def statistic(rows: np.ndarray) -> float:
    return float(np.mean([macro_auc(LABELS[rows], blend[rows]) for blend in BLENDS]))


def main() -> None:
    point = statistic(np.arange(LABELS.size))
    unique = np.unique(CLUSTERS)
    index = {key: np.flatnonzero(CLUSTERS == key) for key in unique}
    rng = np.random.default_rng(20261029)
    draws = []
    while len(draws) < 2000:
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index[key] for key in chosen])
        if np.unique(LABELS[rows]).size < 3:
            continue
        draws.append(statistic(rows))
    low, high = np.percentile(draws, [2.5, 97.5])
    print(f"point={point:.6f} ci=({low:.6f}, {high:.6f}) replicates={len(draws)}")


if __name__ == "__main__":
    main()
