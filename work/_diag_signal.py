import numpy as np, pandas as pd
from src.ich_dynamic.train_evaluate import load_dataset
ds = load_dataset('mimic4')
cohort = ds.cohort
labels = ds.labels
masks = ds.masks
# masks shape: (n, steps, channels) or (n, steps)
print('masks', masks.shape, 'sequences', ds.sequences.shape)
any_obs = (masks.sum(axis=2) > 0) if masks.ndim == 3 else (masks > 0)
n_obs = any_obs.sum(axis=1)
last = np.where(any_obs.any(axis=1), any_obs.shape[1] - 1 - np.argmax(any_obs[:, ::-1], axis=1), -1)
print('n_obs describe', pd.Series(n_obs).describe().round(2).to_dict())
print('last observed hour counts (top)', pd.Series(last).value_counts().sort_index().tail(8).to_dict())
print('outcome by last-hour group')
for lo, hi in [(0,11),(12,17),(18,20),(21,22),(23,23)]:
    m=(last>=lo)&(last<=hi)
    if m.sum():
        print(f'  last {lo}-{hi}: n={m.sum()} any={np.mean(labels[m]>=1):.3f} severe={np.mean(labels[m]>=2):.3f}')
print('analysis_weight describe', pd.Series(ds.analysis_weight).describe().round(3).to_dict())
print('n weights >2', int((ds.analysis_weight>2).sum()), 'max', float(ds.analysis_weight.max()))
cols=[c for c in ['gcs_min','gcs_max','gcs_mean','gcs_slope','gcs_last','monitoring_intensity_24h','age','nihss'] if c in cohort.columns]
print('available summary cols', cols)
