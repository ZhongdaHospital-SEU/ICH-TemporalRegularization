import pandas as pd
p=r"D:\TT paper\1024 TP Intracerebral Hemorrhage\data\processed\mimic4\cohort.csv"
d=pd.read_csv(p, low_memory=False)
death=d["death_in_icu_within_72h"].fillna(False).astype(bool)
drop=pd.to_numeric(d["max_sustained_drop"], errors="coerce")
severe=drop.ge(4).fillna(False)
o=pd.to_numeric(d["outcome"])
bad=d[(o==2)&~death&~severe]
print("n grade2", int((o==2).sum()), "bad", len(bad))
print(bad[["stay_id","outcome","max_sustained_drop","death_in_icu_within_72h","death_within_72h","early_live_icu_transfer_before_72h"]].head(10).to_string())
print("early transfer grade2:", int(((o==2)&d["early_live_icu_transfer_before_72h"].astype(bool)).sum()))
