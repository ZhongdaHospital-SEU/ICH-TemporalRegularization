import pandas as pd
from pathlib import Path
root=Path("data/processed")
for d in ["mimic4","mimic3","eicu"]:
    h=pd.read_csv(root/d/"hourly.csv")
    g=h[h.channel=="gcs_total"]
    print(d,"rows",len(g),"unique stay-hour",g[["stay_id","hour"]].drop_duplicates().shape[0],
          "exact dup rows",g.duplicated(["stay_id","hour","value"]).sum(),
          "multi-value stay-hours",(g.groupby(["stay_id","hour"]).value.nunique()>1).sum())
