"""Recompute channel coverage quoted in Section 3.9 / Figure 9."""
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
labels = [
    "GCS eye", "GCS verbal", "GCS motor", "GCS total", "Heart rate",
    "Respiratory rate", "SpO2", "SBP", "DBP", "MBP", "Temperature",
    "Urine output", "FiO2", "RASS",
]
for key in ("mimic4", "mimic3", "eicu"):
    payload = np.load(
        root / "data" / "processed" / "landmark_inputs" / key / "landmark_24h" / "sequences.npz",
        allow_pickle=True,
    )
    masks = payload["masks"].astype(float)
    coverage = 100.0 * masks.mean(axis=(0, 1))
    print(key, "overall", round(100.0 * masks.mean(), 1))
    for label, value in zip(labels, coverage):
        print(f"   {label:18s} {value:5.1f}")
