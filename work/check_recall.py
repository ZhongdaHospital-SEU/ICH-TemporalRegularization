"""Recompute the Figure 5 row-normalised confusion diagonal (recall)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ICH_TORCH_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te

payload = np.load(
    ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz",
    allow_pickle=True,
)
labels = payload["labels"]
configs = {
    "Summary XGBoost": [payload[f"repeat{i}_xgb_summary"] for i in range(5)],
    "DG-GRU+TR": [payload[f"repeat{i}_dggru_tr"] for i in range(5)],
}
configs["Ensemble"] = [
    0.5 * payload[f"repeat{i}_xgb_summary"] + 0.5 * payload[f"repeat{i}_dggru_tr"]
    for i in range(5)
]

for name, probabilities in configs.items():
    matrices = []
    for probability in probabilities:
        prediction = te.ordinal_median_prediction(probability)
        matrix = np.zeros((3, 3), dtype=float)
        for truth, predicted in zip(labels, prediction):
            matrix[int(truth), int(predicted)] += 1.0
        matrices.append(matrix)
    matrix = np.mean(matrices, axis=0)
    row = 100.0 * matrix / matrix.sum(axis=1, keepdims=True)
    print(name, "recall:", " ".join(f"{value:.1f}" for value in np.diag(row)))
    print("   predicted share:", np.round(matrix.sum(axis=0) / matrix.sum(), 3))
