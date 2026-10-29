import pandas as pd, numpy as np
from pathlib import Path
root=Path("data/processed")
rows=[]
for d in ["mimic4","mimic3","eicu"]:
    c=pd.read_csv(root/d/"cohort.csv")
    g=pd.read_csv(root/d/"gcs_events.csv")
    g["stay_id"]=pd.to_numeric(g["stay_id"],errors="coerce")
    g["time_hour"]=pd.to_numeric(g["time_hour"],errors="coerce")
    sub=g[g["stay_id"].isin(c["stay_id"])]
    pre=sub[(sub.time_hour>=0)&(sub.time_hour<24)].groupby("stay_id").size()
    post=sub[(sub.time_hour>=24)&(sub.time_hour<72)].groupby("stay_id").size()
    n=len(c)
    row={
        "dataset":d,"n":n,
        "g0":int((c.outcome==0).sum()),"g1":int((c.outcome==1).sum()),"g2":int((c.outcome==2).sum()),
        "event_any_%":round(100*(c.outcome>0).mean(),1),
        "severe_%":round(100*(c.outcome==2).mean(),1),
        "death72_%":round(100*c.death_within_72h.astype(str).str.lower().eq("true").mean(),1),
        "gcs_pre_med":float(c.stay_id.map(pre).fillna(0).median()),
        "gcs_post_med":float(c.stay_id.map(post).fillna(0).median()),
        "los_med_h":float(pd.to_numeric(c.icu_out_hour).median()),
        "baseline_gcs_med":float(pd.to_numeric(c.baseline_gcs_max).median()),
    }
    # rate by GCS observation quartile
    q=pd.qcut(c.stay_id.map(pre).fillna(0).rank(method="first"),4,labels=False)
    row["event_by_preGCS_quartile"]=[round(100*(c.outcome>0)[q==i].mean(),1) for i in range(4)]
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
