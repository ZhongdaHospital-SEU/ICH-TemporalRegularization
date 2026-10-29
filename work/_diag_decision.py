import numpy as np, json
from sklearn.metrics import roc_curve, roc_auc_score
rng=np.random.default_rng(7)

def pauc(y,s,lo,hi):
    fpr,tpr,_=roc_curve(y,s)
    # interpolate tpr over fpr grid in [lo,hi]
    grid=np.linspace(lo,hi,200)
    t=np.interp(grid,fpr,tpr)
    return np.trapz(t,grid)/(hi-lo)

def pauc_tpr(y,s,lo,hi):
    fpr,tpr,_=roc_curve(y,s)
    # integrate over tpr range
    grid=np.linspace(lo,hi,200)
    f=np.interp(grid,tpr,fpr)
    return 1-np.trapz(f,grid)/(hi-lo)

def net_benefit(y,p,pt):
    tp=((p>=pt)&(y==1)).sum(); fp=((p>=pt)&(y==0)).sum(); n=len(y)
    return tp/n - fp/n*(pt/(1-pt))

for cohort,path,key in [('mimic4-dev','results/predictions/mimic4_cv_core_repeats.npz','repeat{r}_'),
                        ('mimic3','results/predictions/mimic3_external_core.npz',''),
                        ('eicu','results/predictions/eicu_external_core.npz','')]:
    try:
        d=np.load(path,allow_pickle=True)
    except FileNotFoundError:
        print('missing',path); continue
    print('=====',cohort)
    y=d['labels']
    for m in ['xgb_summary','dggru','dggru_tr']:
        if key:
            P=np.mean([d[key.format(r=r)+m] for r in range(5)],axis=0)
        else:
            P=d[m]
        sev=1-(P[:,0])           # any deterioration score
        yy=(y>=1).astype(int)
        a=roc_auc_score(yy,sev)
        p1=pauc(yy,sev,0,0.10)
        p2=pauc_tpr(yy,sev,0,0.50)
        nb=[net_benefit(yy,sev,t) for t in (0.10,0.20,0.30)]
        print('  %-10s AUC %.4f  pAUC(FPR<=.10) %.4f  pAUC(TPR>=.50) %.4f  NB@.1 %.4f @.2 %.4f @.3 %.4f'%(m,a,p1,p2,*nb))
