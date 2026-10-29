import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from src.ich_dynamic import train_evaluate as te
from sklearn.metrics import roc_auc_score

folds = pd.read_csv('results/mimic4_folds_core.csv')
fold0 = folds[folds.repeat==0].set_index('stay_id')['fold']
ds_core = te.load_dataset('mimic4', feature_columns=list(te.CORE_STATIC))
ds_all  = te.load_dataset('mimic4')
print('core feats', ds_core.static_all.shape[1], ' all feats', ds_all.static_all.shape[1])
y = ds_core.labels
sid = ds_core.cohort['stay_id'].to_numpy()
f = fold0.reindex(sid).to_numpy()

def macro(p,y): return np.mean([roc_auc_score((y==k).astype(int), p[:,k]) for k in range(3)])
def run(mat, label):
    oof = np.zeros((len(y),3))
    for k in range(5):
        tr, te_ = f!=k, f==k
        Xtr = mat[tr]; Xte = mat[te_]
        med = np.nanmedian(Xtr, axis=0); med[~np.isfinite(med)] = 0.0
        Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
        oof[te_] = te.fit_xgboost(Xtr, y[tr], Xte, seed=20261024)
    m = macro(oof,y)
    print('%-6s macro %.4f  g0 %.4f  g1 %.4f  g2 %.4f'%(label,m,
        roc_auc_score((y==0),oof[:,0]),roc_auc_score((y==1),oof[:,1]),roc_auc_score((y==2),oof[:,2])))
    return oof, m

o1,m1 = run(ds_core.static_all.to_numpy(float),'core23')
o2,m2 = run(ds_all.static_all.to_numpy(float),'all')
print('difference all-core %+.4f'%(m2-m1))
