import numpy as np
from src.ich_dynamic.train_evaluate import load_dataset
import work.assemble_split_run as a

ds = load_dataset("mimic4")
n = len(ds.labels)
rng = np.random.default_rng(0)
def probs():
    x = rng.random((n, 3)) + 0.1
    return x / x.sum(axis=1, keepdims=True)
ids = ds.cohort["stay_id"].astype(np.int64).to_numpy()
subs = ds.cohort["subject_key"].astype(str).to_numpy()
labs = ds.labels.astype(int)
core = {"repeats": {}, "predictions": {}}
for archive in (core["repeats"], core["predictions"]):
    archive.update(labels=labs, stay_id=ids, subject_key=subs)
for m in ("dggru_tr", "dggru"):
    core["predictions"][m] = probs()
    for r in range(5):
        core["repeats"][f"repeat{r}_{m}"] = probs()
for r in range(5):
    core["repeats"][f"repeat{r}_analysis_weight"] = np.ones(n)
abl = {"repeats": {}, "predictions": {}}
for archive in (abl["repeats"], abl["predictions"]):
    archive.update(labels=labs, stay_id=ids, subject_key=subs)
abl["predictions"]["dggru_tr_treverse"] = probs()
abl["repeats"]["repeat0_dggru_tr_treverse"] = probs()
components = {
    "core": {"repeats": core["repeats"], "predictions": core["predictions"]},
    "ablation": {"repeats": abl["repeats"], "predictions": abl["predictions"]},
}
orig = a.COMPONENT_MODELS
a.COMPONENT_MODELS = {"core": ("dggru_tr", "dggru"), "ablation": ("dggru_tr_treverse",)}
a.COMPONENT_REPEATS = {"core": 5, "ablation": 1}
try:
    merged = a._merge_repeats(components)
    print("repeat keys", len(merged), sorted(merged)[:5])
    block, repeated, weights = a._development_metrics(ds, components, replicates=20, seed=1)
    print("models", sorted(block))
    comps, paired, validity = a._comparisons(
        ds.labels.astype(int), repeated, weights,
        groups=ds.cohort["subject_key"].astype(str).to_numpy(),
        replicates=20, seed=1)
    print("comparisons", sorted(comps))
    print("paired", {k: [round(float(x), 4) for x in v] for k, v in paired.items()})
    print("valid", {k: v.get("valid_replicates") for k, v in validity.items()})
finally:
    a.COMPONENT_MODELS = orig
