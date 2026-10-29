"""Check whether the transported contrast depends on the inherited epoch count."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ich_dynamic.train_evaluate import (
    SEQUENCE_DEFAULTS,
    _fit_transport_member,
    load_dataset,
)

SEED = 20261024
EPOCHS = (6, 8, 11)


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
    development = load_dataset("mimic4")
    external = load_dataset("mimic3")
    for epochs in EPOCHS:
        row = {}
        for arm, penalty in (("dggru", 0.0), ("dggru_tr", 1.5)):
            start = time.time()
            probabilities = _fit_transport_member(
                development,
                external,
                arm,
                seed=SEED,
                epochs=epochs,
                lambda_monotone=penalty,
                lambda_smooth=0.4 * penalty,
                lambda_smooth_ratio=0.4,
                model_seeds=(0,),
                cv_best_epochs={},
                cv_selected_lambdas={},
                transport_hyperparameters={arm: dict(SEQUENCE_DEFAULTS)},
            )
            row[arm] = macro_auc(external.labels, probabilities)
            print(
                f"epochs={epochs:2d} {arm:9s} auc={row[arm]:.4f} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
        print(f"  diff={row['dggru_tr'] - row['dggru']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
