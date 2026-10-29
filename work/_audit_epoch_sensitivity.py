"""Fold-0 check of the inherited epoch budget in the penalty contrast."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ich_dynamic.train_evaluate import (
    SEQUENCE_DEFAULTS,
    build_repeated_splits,
    fit_temporal,
    load_dataset,
)

SEED = 20261024
EPOCHS = (5, 8, 12, 16, 25)


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == k).astype(int), probabilities[:, k])
                for k in range(3)
            ]
        )
    )


def main() -> None:
    dataset = load_dataset("mimic4")
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    splits, _ = build_repeated_splits(
        dataset.labels, groups, folds=5, repeats=1, seed=SEED
    )
    train_index, test_index = splits[0]
    labels = dataset.labels[test_index]
    print(f"fold 0: train={len(train_index)} test={len(test_index)}", flush=True)
    results: dict[int, dict[str, float]] = {}
    for epochs in EPOCHS:
        row: dict[str, float] = {}
        for arm, penalty in (("dggru", 0.0), ("dggru_tr", 1.5)):
            start = time.time()
            probabilities, used, _ = fit_temporal(
                dataset,
                train_index,
                test_index,
                {
                    "architecture": "dggru",
                    "graph": True,
                    "regularized": penalty > 0,
                    "static_mode": "core",
                    "sequence_mode": "observed",
                    "mask_mode": "any_channel",
                },
                seed=SEED,
                epochs=epochs,
                lambda_monotone=penalty,
                lambda_smooth=0.4 * penalty,
                lambda_grid=(),
                tune_lambda=False,
                model_name=arm,
                hyperparameter_override=dict(SEQUENCE_DEFAULTS),
                select_hyperparameters=False,
                fixed_epochs=epochs,
            )
            row[arm] = macro_auc(labels, probabilities)
            print(
                f"epochs={epochs:2d} {arm:9s} auc={row[arm]:.4f} "
                f"used={used} {time.time() - start:.0f}s",
                flush=True,
            )
        results[epochs] = row
    print("\nepochs  dggru   dggru_tr  diff")
    for epochs, row in results.items():
        print(
            f"{epochs:6d}  {row['dggru']:.4f}  {row['dggru_tr']:.4f}  "
            f"{row['dggru_tr'] - row['dggru']:+.4f}"
        )


if __name__ == "__main__":
    main()
