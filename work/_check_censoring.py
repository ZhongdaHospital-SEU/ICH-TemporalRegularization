import pandas as pd, numpy as np
from src.ich_dynamic.sensitivity_analysis import _apply_censoring, ScenarioSpec, SCENARIO_INDEX

frame = pd.DataFrame([
    # complete follow-up, stable
    dict(stay_id=1, outcome=0, outcome_reconstructed=0, death_within_72h=False,
         death_hour_from_icu=np.nan, icu_out_hour=96.0),
    # early live transfer, observed deterioration grade1
    dict(stay_id=2, outcome=1, outcome_reconstructed=1, death_within_72h=False,
         death_hour_from_icu=np.nan, icu_out_hour=40.0),
    # early live transfer, no classifiable outcome
    dict(stay_id=3, outcome=np.nan, outcome_reconstructed=np.nan, death_within_72h=False,
         death_hour_from_icu=np.nan, icu_out_hour=30.0),
    # early live transfer, death after discharge
    dict(stay_id=4, outcome=2, outcome_reconstructed=2, death_within_72h=True,
         death_hour_from_icu=45.0, icu_out_hour=40.0),
    # in-ICU death
    dict(stay_id=5, outcome=2, outcome_reconstructed=2, death_within_72h=True,
         death_hour_from_icu=60.0, icu_out_hour=80.0),
])
for sid in ["s1_baseline_max","s4_censor_exclude","s4_censor_grade0"]:
    spec = SCENARIO_INDEX[sid]
    out = _apply_censoring(frame.copy(), spec)
    print(sid, "strategy=",spec.censoring_strategy, "->", dict(zip(out.stay_id, out.outcome)))
