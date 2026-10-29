import numpy as np, glob, json, os
from sklearn.metrics import roc_auc_score
rows=[]
for f in sorted(glob.glob('results/predictions/grid2d_*_repeats.npz')):
    base=os.path.basename(f).replace('grid2d_','').replace('_repeats.npz','')
    a,b=base.split('_')
    l1=float(a.replace('p','.')); l2=float(b.replace('p','.'))
    d=np.load(f,allow_pickle=True); y=d['labels']; p=d['repeat0_dggru_tr']
    macro=float(np.mean([roc_auc_score((y==k).astype(int),p[:,k]) for k in range(3)]))
    rows.append((l1,l2,macro))
rows.sort(key=lambda r:(r[0],r[1]))
print('  l1   l2   macro')
for r in rows: print('%4.2f %5.2f  %.4f'%r)
best=max(rows,key=lambda r:r[2])
print('\nbest',best)
json.dump([{'l1':a,'l2':b,'macro_auc':m} for a,b,m in rows],open('results/experiments/grid2d_lambda.json','w'),indent=1)
