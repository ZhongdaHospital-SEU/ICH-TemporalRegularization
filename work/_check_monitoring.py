import pandas as pd, numpy as np
from pathlib import Path
root=Path("data/processed")
frames=[]
for d in ["mimic4","mimic3","eicu"]:
    c=pd.read_csv(root/d/"cohort.csv")
    g=pd.read_csv(root/d/"gcs_events.csv")
    g["stay_id"]=pd.to_numeric(g["stay_id"],errors="coerce")
    g["time_hour"]=pd.to_numeric(g["time_hour"],errors="coerce")
    sub=g[g["stay_id"].isin(c["stay_id"])]
    post=sub[(sub.time_hour>=24)&(sub.time_hour<72)&sub.gcs.between(3,15)].groupby("stay_id").size()
    c["post_gcs"]=c.stay_id.map(post).fillna(0).astype(int)
    c["dataset"]=d
    frames.append(c[["dataset","stay_id","outcome","post_gcs"]])
x=pd.concat(frames)
bins=[0,2,5,11,23,10**6]; labels=["<2","2-5","6-11","12-23","24+"]
x["bin"]=pd.cut(x.post_gcs,bins=bins,labels=labels,right=True)
tab=pd.crosstab([x.dataset,x.bin],x.outcome,normalize="index").mul(100).round(1)
cnt=x.groupby(["dataset","bin"],observed=True).size().rename("n")
print(pd.concat([cnt,tab],axis=1).to_string())
print()
print("post-GCS distribution (median, IQR):")
for d in ["mimic4","mimic3","eicu"]:
    s=x[x.dataset==d].post_gcs
    print(d, "median",s.median(),"IQR",s.quantile(.25),s.quantile(.75),"n<2:",int((s<2).sum()))
