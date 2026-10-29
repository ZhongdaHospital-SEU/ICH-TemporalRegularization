import pandas as pd, numpy as np
from pathlib import Path
root=Path("data/processed")
for d in ["mimic4","mimic3","eicu"]:
    h=pd.read_csv(root/d/"hourly.csv")
    h["value"]=pd.to_numeric(h["value"],errors="coerce")
    w=h.pivot_table(index=["stay_id","hour"],columns="channel",values="value",aggfunc="first")
    need={"gcs_eye","gcs_verbal","gcs_motor","gcs_total"}
    if not need.issubset(w.columns): 
        print(d,"missing components"); continue
    v=w.dropna(subset=list(need))
    comp=v.gcs_eye+v.gcs_verbal+v.gcs_motor
    diff=(v.gcs_total-comp)
    print(d,"both present n=",len(v),
          "exact match %",round(100*(diff.abs()<1e-9).mean(),1),
          "|diff|>=2 %",round(100*(diff.abs()>=2).mean(),1),
          "median diff",diff.median())
