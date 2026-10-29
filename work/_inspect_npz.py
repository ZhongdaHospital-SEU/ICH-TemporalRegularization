import numpy as np
for p in ['results/predictions/mimic4_cv_prelim.npz','results/predictions/mimic4_cv_prelim_repeats.npz','results/predictions/mimic3_external_prelim.npz']:
    z=np.load(p, allow_pickle=True)
    print(p, len(z.files))
    print(sorted(z.files)[:40])
    print({k: (z[k].shape, z[k].dtype) for k in z.files[:6]})
