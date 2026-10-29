import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")


def macro_auc(labels, probs):
    return float(np.mean([roc_auc_score(labels == g, probs[:, g]) for g in range(3)]))


base = np.load("results/experiments/strength_monitor_drop_predictions.npz", allow_pickle=True)
nomask = np.load(
    "results/experiments/strength_monitor_drop_nomask_predictions.npz", allow_pickle=True
)

labels = base["labels"]
subjects = base["subject_key"]
assert np.array_equal(base["stay_id"], nomask["stay_id"])

auc_with = macro_auc(labels, base["dggru_tr"])
auc_without = macro_auc(labels, nomask["dggru_tr"])
print("with mask", round(auc_with, 6), "no mask", round(auc_without, 6))
print("difference", round(auc_without - auc_with, 6))

unique = np.unique(subjects)
index_by_subject = {s: np.where(subjects == s)[0] for s in unique}
rng = np.random.default_rng(20261024)
diffs = []
for _ in range(2000):
    draw = rng.choice(unique, size=len(unique), replace=True)
    idx = np.concatenate([index_by_subject[s] for s in draw])
    lab = labels[idx]
    if len(np.unique(lab)) < 3:
        continue
    diffs.append(
        macro_auc(lab, nomask["dggru_tr"][idx]) - macro_auc(lab, base["dggru_tr"][idx])
    )
print("replicates", len(diffs))
print("ci", np.percentile(diffs, [2.5, 97.5]))
