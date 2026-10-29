import os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("ICH_TORCH_THREADS", "2")
os.environ.setdefault("ICH_XGB_THREADS", "1")
sys.path.insert(0, '.')

L1 = [0.0, 0.5, 1.0, 1.5, 2.5]
L2 = [0.0, 0.05, 0.2, 0.6, 1.5]
COMBOS = [(a, b) for a in L1 for b in L2 if not (a == 0.0 and b != 0.0)]

def tag_of(l1, l2):
    return f"grid2d_{str(l1).replace('.','p')}_{str(l2).replace('.','p')}"

def run(combo):
    from src.ich_dynamic import train_evaluate as te
    l1, l2 = combo
    config, _ = te.load_config_snapshot()
    dev = te.load_dataset("mimic4", feature_columns=list(te.CORE_STATIC))
    tag = tag_of(l1, l2)
    te.run_cv(
        dev, folds=5, epochs=25, seed=int(config["random_seed"]),
        models=("dggru_tr",), lambda_monotone=l1, lambda_smooth=l2,
        workers=1, worker_threads=2, model_seeds=(0,), repeats=1,
        repeat_prefix=tag,
    )
    return l1, l2

if __name__ == "__main__":
    done = {tag_of(a, b) for a, b in COMBOS
            if os.path.exists(f"results/predictions/{tag_of(a,b)}_repeats.npz")}
    todo = [c for c in COMBOS if tag_of(*c) not in done]
    print("done", len(done), "todo", len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run, c) for c in todo]
        for f in as_completed(futs):
            print("finished", f.result(), flush=True)
    print("all runs complete", flush=True)
