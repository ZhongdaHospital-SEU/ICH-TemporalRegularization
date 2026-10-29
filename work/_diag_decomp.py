import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(20261029)
d = np.load('results/predictions/mimic4_cv_core_repeats.npz', allow_pickle=True)
y = d['labels']; subj = d['subject_key']
def macro(p,y): return np.mean([roc_auc_score((y==k).astype(int), p[:,k]) for k in range(3)])
X = np.mean([d[f'repeat{r}_xgb_summary'] for r in range(5)],axis=0)
G = np.mean([d[f'repeat{r}_dggru_tr'] for r in range(5)],axis=0)
print('pooled xgb %.4f  dggru_tr %.4f'%(macro(X,y),macro(G,y)))
print()
# per-class
for k in range(3):
    print('class %d  n=%4d  xgb %.4f  dg %.4f  diff %+.4f'%(k,(y==k).sum(),
        roc_auc_score((y==k).astype(int),X[:,k]),roc_auc_score((y==k).astype(int),G[:,k]),
        roc_auc_score((y==k).astype(int),G[:,k])-roc_auc_score((y==k).astype(int),X[:,k])))
print()
# blend ceiling
best=(0,None)
for w in np.linspace(0,1,101):
    a=macro(w*X+(1-w)*G,y)
    if a>best[0]: best=(a,w)
print('best in-sample blend macro %.4f at w_xgb=%.2f'%(best[0],best[1]))
print('pooled 50/50 %.4f'%macro(0.5*X+0.5*G,y))
print()
# bootstrap SE of macro diff and of grade-1 diff
clusters=np.unique(subj); idx={c:np.where(subj==c)[0] for c in clusters}
def boot(B=1000):
    dm=[];dg=[]
    for _ in range(B):
        pick=rng.choice(clusters,len(clusters),replace=True)
        ii=np.concatenate([idx[c] for c in pick])
        yy=y[ii]
        if len(np.unique(yy))<3: continue
        dm.append(macro(G[ii],yy)-macro(X[ii],yy))
        dg.append(roc_auc_score((yy==1),G[ii,1])-roc_auc_score((yy==1),X[ii,1]))
    return np.array(dm),np.array(dg)
dm,dg=boot()
print('paired macro diff  mean %+.4f  se %.4f  95%%CI [%+.4f, %+.4f]'%(dm.mean(),dm.std(),*np.percentile(dm,[2.5,97.5])))
print('paired grade1 diff mean %+.4f  se %.4f  95%%CI [%+.4f, %+.4f]'%(dg.mean(),dg.std(),*np.percentile(dg,[2.5,97.5])))
print()
# correlation / agreement
print('pearson r grade1 prob %.3f'%np.corrcoef(X[:,1],G[:,1])[0,1])
print('argmax agreement %.3f'%( (X.argmax(1)==G.argmax(1)).mean()))
