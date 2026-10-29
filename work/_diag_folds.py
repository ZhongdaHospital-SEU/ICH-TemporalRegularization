import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

folds = pd.read_csv('results/mimic4_folds_core.csv')
d = np.load('results/predictions/mimic4_cv_core_repeats.npz', allow_pickle=True)
y = d['labels']; sid = d['stay_id']
models = ['xgb_summary','gru','dggru','dggru_tr']
def macro(p,y):
    return np.mean([roc_auc_score((y==k).astype(int), p[:,k]) for k in range(3)])

rows=[]
for r in range(5):
    f = folds[folds.repeat==r].set_index('stay_id')['fold'].reindex(sid).values
    for k in np.unique(f):
        m = f==k
        rec={'repeat':r,'fold':int(k),'n':int(m.sum())}
        for mod in models:
            rec[mod]=macro(d[f'repeat{r}_{mod}'][m], y[m])
        rows.append(rec)
df=pd.DataFrame(rows)
df['diff']=df['dggru_tr']-df['xgb_summary']
print(df.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
print()
print('folds where DG-GRU+TR > XGB:', int((df['diff']>0).sum()), '/', len(df))
print('mean diff %.4f  sd %.4f  min %.4f  max %.4f'%(df['diff'].mean(),df['diff'].std(),df['diff'].min(),df['diff'].max()))
print()
# by repeat
print(df.groupby('repeat')[['xgb_summary','dggru_tr','diff']].mean().to_string(float_format=lambda v:f'{v:.4f}'))
