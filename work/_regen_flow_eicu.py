import pandas as pd
from src.ich_dynamic.build_cohorts import make_flow
from src.ich_dynamic.common import PROCESSED
root = PROCESSED / "eicu"
cohort = pd.read_csv(root / "cohort.csv", low_memory=False)
prefilter = pd.read_csv(root / "cohort_pre_filter.csv", low_memory=False)
flow = make_flow("eicu", 4117, cohort, len(prefilter), True)
flow.to_csv(root / "cohort_flow.csv", index=False)
print(flow[["step","n"]].to_string(index=False))
