import numpy as np
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(20261029)
a=np.load('results/predictions/mimic4_cv_ablation.npz',allow_pickle=True)
b=np.load('results/predictions/mimic4_cv_ablation_intact.npz',allow_pickle=True)
y=b['labels']; subj=b['subject_key']
print('arms:',[k for k in a.keys()])
mean=np.mean([a[f'repeat{r}_{m}'] for r in range(1)],axis=0) if False else None
def macro(p,y):
    return np.mean([roc_auc_score((y==k).astype(int),p[:,k]) for k in range(3)])
base=b['dggru_tr']
print('intact %.4f'%macro(base,y))
clusters=np.unique(subj); idx={c:np.where(subj==c)[0] for c in clusters}
arms=['dggru_tr_treverse','dggru_tr_tpermute','dggru_laplace']
for arm in arms:
    p=a[arm]
    d=macro(base,y)-macro(p,y)
    diffs=[]
    for _ in range(2000):
        pick=rng.choice(clusters,len(clusters),replace=True)
        ii=np.concatenate([idx[c] for c in pick])
        yy=y[ii]
        if len(np.unique(yy))<3: continue
        diffs.append(macro(base[ii],yy)-macro(p[ii],yy))
    lo,hi=np.percentile(diffs,[2.5,97.5])
    print('%-20s %.4f  diff %+.4f  CI [%+.4f, %+.4f]'%(arm,macro(p,y),d,lo,hi))
