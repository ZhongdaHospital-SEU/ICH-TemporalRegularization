import pandas as pd
from src.ich_dynamic.build_cohorts import make_flow
from src.ich_dynamic.common import PROCESSED
import os
os.environ.setdefault("ICH_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
root = PROCESSED / "mimic4"
cohort = pd.read_csv(root / "cohort.csv", low_memory=False)
prefilter = pd.read_csv(root / "cohort_pre_filter.csv", low_memory=False)
flow = make_flow("mimic4", 4622, cohort, len(prefilter), True)
flow.to_csv(root / "cohort_flow.csv", index=False)
print(flow.to_string(index=False))
