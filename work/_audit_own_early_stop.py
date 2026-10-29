"""Fold-0 contrast when each arm early-stops on its own inner validation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ich_dynamic.train_evaluate import (
    SEQUENCE_DEFAULTS,
    build_repeated_splits,
    fit_temporal,
    load_dataset,
)

SEED = 20261024


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
    scores = {}
    for arm, penalty in (("dggru", 0.0), ("dggru_tr", 1.5)):
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
            epochs=25,
            lambda_monotone=penalty,
            lambda_smooth=0.4 * penalty,
            lambda_grid=(),
            tune_lambda=False,
            model_name=arm,
            hyperparameter_override=dict(SEQUENCE_DEFAULTS),
            select_hyperparameters=False,
        )
        scores[arm] = macro_auc(labels, probabilities)
        print(f"{arm:9s} own_stop_epochs={used:2d} auc={scores[arm]:.4f}", flush=True)
    print(f"diff={scores['dggru_tr'] - scores['dggru']:+.4f}")


if __name__ == "__main__":
    main()
