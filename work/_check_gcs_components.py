import pandas as pd, numpy as np
from pathlib import Path
root=Path("data/processed")
rows=[]
for d in ["mimic4","mimic3","eicu"]:
    h=pd.read_csv(root/d/"hourly.csv")
    h["stay_id"]=pd.to_numeric(h["stay_id"],errors="coerce")
    h["value"]=pd.to_numeric(h["value"],errors="coerce")
    for ch in ["gcs_eye","gcs_verbal","gcs_motor","gcs_total"]:
        s=h.loc[h.channel==ch,"value"].dropna()
        if s.empty: continue
        rows.append({"dataset":d,"channel":ch,"n":len(s),
                     "mean":round(s.mean(),2),"median":s.median(),
                     "p1":s.quantile(.01),"p5":s.quantile(.05),
                     "pct_min":round(100*(s==s.min()).mean(),1),
                     "pct_le_2":round(100*(s<=2).mean(),1)})
print(pd.DataFrame(rows).to_string(index=False))
print()
# verbal==1 conditional on total<=8 (proxy for intubated/sedated)
for d in ["mimic4","mimic3","eicu"]:
    h=pd.read_csv(root/d/"hourly.csv")
    h["stay_id"]=pd.to_numeric(h["stay_id"],errors="coerce")
    h["value"]=pd.to_numeric(h["value"],errors="coerce")
    w=h.pivot_table(index=["stay_id","hour"],columns="channel",values="value",aggfunc="first")
    if not {"gcs_verbal","gcs_total"}.issubset(w.columns): continue
    v=w.dropna(subset=["gcs_verbal","gcs_total"])
    print(d, "n_cells",len(v),
          "verbal=1 & total<=8:", round(100*((v.gcs_verbal==1)&(v.gcs_total<=8)).mean(),1),
          "verbal=1 & total>8:", round(100*((v.gcs_verbal==1)&(v.gcs_total>8)).mean(),1))
