"""Print cohort columns and selected site/identifier fields."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for cohort in ("mimic4", "mimic3", "eicu"):
    path = ROOT / "data" / "processed" / cohort / "cohort.csv"
    frame = pd.read_csv(path, nrows=5)
    print(f"== {cohort} ==")
    print(", ".join(frame.columns))
    for column in frame.columns:
        lowered = column.lower()
        if any(token in lowered for token in ("hospital", "site", "unit", "stay", "subject", "patient")):
            print(f"  {column}: {frame[column].tolist()[:5]}")
    print()
