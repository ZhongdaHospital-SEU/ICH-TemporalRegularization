"""Patient-cluster bootstrap CI for the one-repeat nested stacking model."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

payload = np.load(
    "results/predictions/mimic4_cv_stack1_repeats.npz", allow_pickle=True
)
labels = payload["labels"]
clusters = payload["subject_key"]
probability = payload["repeat0_nested_stack"]


def macro_auc(y_true, scores) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((y_true == grade).astype(int), scores[:, grade])
                for grade in range(3)
            ]
        )
    )


print("point", round(macro_auc(labels, probability), 4))

unique = np.unique(clusters)
index_by_cluster = {key: np.where(clusters == key)[0] for key in unique}
rng = np.random.default_rng(20240501)
values = []
for _ in range(2000):
    draw = rng.choice(unique, size=unique.size, replace=True)
    index = np.concatenate([index_by_cluster[key] for key in draw])
    if np.unique(labels[index]).size < 3:
        continue
    values.append(macro_auc(labels[index], probability[index]))

values = np.asarray(values)
print(
    "bootstrap 95% CI",
    round(float(np.percentile(values, 2.5)), 3),
    round(float(np.percentile(values, 97.5)), 3),
    "replicates",
    values.size,
)
