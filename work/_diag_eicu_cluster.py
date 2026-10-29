import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(20261029)
d=np.load('results/predictions/eicu_external_core.npz',allow_pickle=True)
y=d['labels']; subj=d['subject_key']; sid=d['stay_id']
coh=pd.read_csv('data/processed/eicu/cohort.csv')
cols=[c for c in coh.columns if 'hospital' in c.lower()]
print('hospital columns:',cols)
hcol=cols[0]
coh=coh.set_index('stay_id')
hosp=coh[hcol].reindex(sid).to_numpy()
print('unique hospitals',len(pd.unique(hosp)),'unique patients',len(pd.unique(subj)))
def macro(p,y): return np.mean([roc_auc_score((y==k).astype(int),p[:,k]) for k in range(3)])
def boot(key, pa, pb, B=2000):
    ks=pd.unique(key); idx={k:np.where(key==k)[0] for k in ks}
    out=[]
    for _ in range(B):
        pick=rng.choice(ks,len(ks),replace=True)
        ii=np.concatenate([idx[k] for k in pick]); yy=y[ii]
        if len(np.unique(yy))<3: continue
        out.append(macro(pa[ii],yy)-macro(pb[ii],yy))
    out=np.array(out)
    return out.mean(), np.percentile(out,[2.5,97.5])
for name,a,b in [('dggru_tr - dggru',d['dggru_tr'],d['dggru']),
                 ('dggru_tr - xgb',d['dggru_tr'],d['xgb_summary'])]:
    print('\n',name)
    for lab,key in [('patient',subj),('hospital',hosp)]:
        m,ci=boot(key,a,b)
        print('  %-9s mean %+.4f  CI [%+.4f, %+.4f]'%(lab,m,ci[0],ci[1]))
