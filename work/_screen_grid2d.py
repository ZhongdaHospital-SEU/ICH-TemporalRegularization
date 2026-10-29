import os, sys, json, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("ICH_TORCH_THREADS", "3")
os.environ.setdefault("ICH_XGB_THREADS", "1")
sys.path.insert(0, '.')

L1 = [0.0, 0.5, 1.0, 1.5, 2.5]
L2 = [0.0, 0.05, 0.2, 0.6, 1.5]
COMBOS = [(a, b) for a in L1 for b in L2 if not (a == 0.0 and b != 0.0)]

def run(combo):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.ich_dynamic import train_evaluate as te
    l1, l2 = combo
    config, _ = te.load_config_snapshot()
    dev = te.load_dataset("mimic4", feature_columns=list(te.CORE_STATIC))
    tag = f"grid2d_{str(l1).replace('.','p')}_{str(l2).replace('.','p')}"
    preds, *_ = te.run_cv(
        dev, folds=5, epochs=25, seed=int(config["random_seed"]),
        models=("dggru_tr",), lambda_monotone=l1, lambda_smooth=l2,
        workers=1, worker_threads=3, model_seeds=(0,), repeats=1,
        repeat_prefix=tag,
    )
    p = preds["dggru_tr"]; y = dev.labels
    macro = float(np.mean([roc_auc_score((y == k).astype(int), p[:, k]) for k in range(3)]))
    return {"l1": l1, "l2": l2, "macro_auc": macro}

if __name__ == "__main__":
    out = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run, c): c for c in COMBOS}
        for f in as_completed(futs):
            r = f.result()
            out.append(r)
            print(f"l1={r['l1']:.2f} l2={r['l2']:.2f} macro={r['macro_auc']:.4f}", flush=True)
    out.sort(key=lambda r: -r["macro_auc"])
    json.dump(out, open("results/experiments/grid2d_lambda.json", "w"), indent=1)
    print("\nTOP5")
    for r in out[:5]:
        print(r)
