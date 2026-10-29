import pandas as pd
p=r"D:\TT paper\1024 TP Intracerebral Hemorrhage\data\processed\mimic4\cohort.csv"
d=pd.read_csv(p, low_memory=False)
print("n", len(d))
print(d["outcome"].value_counts().sort_index().to_dict())
print("early transfer included:", int(d["early_live_icu_transfer_before_72h"].astype(bool).sum()))
print("complete followup:", int(d["complete_followup_only"].astype(bool).sum()))
print("death<=72h:", int(d["death_in_icu_within_72h"].astype(bool).sum()))
print("cols", len(d.columns))
