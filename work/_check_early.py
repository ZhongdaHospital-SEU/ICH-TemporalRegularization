import pandas as pd
base=r"D:\TT paper\1024 TP Intracerebral Hemorrhage\data\processed"
for ds in ["mimic4","mimic3","eicu"]:
    d=pd.read_csv(f"{base}\\{ds}\\cohort_pre_filter.csv", low_memory=False)
    early=d[d["early_live_icu_transfer_before_72h"].astype(bool)]
    inc=d[d["primary_followup_complete"].astype(bool)]
    print("="*20, ds, "prefilter", len(d))
    print(" included", len(inc), "outcome", inc["outcome"].value_counts().sort_index().to_dict())
    print(" early transfer", len(early), "outcome", early["outcome"].value_counts(dropna=False).sort_index().to_dict())
    print(" early transfer outcome rate", round(early["outcome"].notna().mean(),3))
