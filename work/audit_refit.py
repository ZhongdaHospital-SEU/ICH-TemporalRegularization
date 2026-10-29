"""Audit whether neural models materially change after outer-fold refitting."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ich_dynamic.common import RESULTS, load_config
from src.ich_dynamic.models import (
    SequencePreprocessor,
    fit_full_temporal_model,
    predict_temporal_model,
    train_temporal_model,
)
from src.ich_dynamic.train_evaluate import (
    _configure_training_worker,
    evaluate_predictions,
    inner_split,
    load_dataset,
)


MODELS = {
    "dggru": {
        "use_graph": True,
        "use_regularization": False,
        "direction": "increasing",
    },
    "dggru_tr": {
        "use_graph": True,
        "use_regularization": True,
        "direction": "increasing",
    },
}


def fit_fold(
    dataset,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    model_seeds: tuple[int, ...],
) -> tuple[int, dict[str, np.ndarray], dict[str, list[int]]]:
    raw_static = dataset.static_all.to_numpy(dtype=np.float32)
    fold_predictions = {
        name: np.zeros((len(test_index), 3), dtype=np.float64)
        for name in MODELS
    }
    chosen_epochs = {name: [] for name in MODELS}
    for offset in range(len(model_seeds)):
        model_seed = seed + 100 * (fold + 1) + 1000 * offset
        outer_groups = dataset.cohort["subject_key"].astype(str).to_numpy()[train_index]
        inner_train, inner_valid = inner_split(
            dataset.labels[train_index], outer_groups, model_seed
        )
        inner_preprocessor = SequencePreprocessor.fit(
            dataset.sequences[train_index][inner_train],
            dataset.masks[train_index][inner_train],
            raw_static[train_index][inner_train],
        )
        inner_sequences, inner_masks = inner_preprocessor.transform_sequence(
            dataset.sequences[train_index], dataset.masks[train_index]
        )
        inner_static = inner_preprocessor.transform_static(raw_static[train_index])

        outer_preprocessor = SequencePreprocessor.fit(
            dataset.sequences[train_index],
            dataset.masks[train_index],
            raw_static[train_index],
        )
        train_sequences, train_masks = outer_preprocessor.transform_sequence(
            dataset.sequences[train_index], dataset.masks[train_index]
        )
        test_sequences, test_masks = outer_preprocessor.transform_sequence(
            dataset.sequences[test_index], dataset.masks[test_index]
        )
        train_static = outer_preprocessor.transform_static(raw_static[train_index])
        test_static = outer_preprocessor.transform_static(raw_static[test_index])

        for name, meta in MODELS.items():
            _, _, history = train_temporal_model(
                inner_sequences[inner_train],
                inner_masks[inner_train],
                inner_static[inner_train],
                dataset.labels[train_index][inner_train],
                inner_sequences[inner_valid],
                inner_masks[inner_valid],
                inner_static[inner_valid],
                dataset.labels[train_index][inner_valid],
                use_graph=meta["use_graph"],
                use_regularization=meta["use_regularization"],
                regularization_direction=meta["direction"],
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                epochs=epochs,
                seed=model_seed,
            )
            best_epoch = int(
                min(history, key=lambda row: row["valid_loss"])["epoch"]
            )
            chosen_epochs[name].append(best_epoch)
            model = fit_full_temporal_model(
                train_sequences,
                train_masks,
                train_static,
                dataset.labels[train_index],
                use_graph=meta["use_graph"],
                use_regularization=meta["use_regularization"],
                regularization_direction=meta["direction"],
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                seed=model_seed,
                epochs=best_epoch,
            )
            probabilities, _ = predict_temporal_model(
                model, test_sequences, test_masks, test_static
            )
            fold_predictions[name] += probabilities / len(model_seeds)
    return fold, fold_predictions, chosen_epochs


def main() -> None:
    config = load_config()
    seed = int(config["random_seed"])
    epochs = 25
    workers = 5
    worker_threads = 3
    replicates = 2000
    model_seeds = (0, 1, 2)
    lambda_monotone = float(config["lambda_monotone"])
    lambda_smooth = float(config["lambda_smooth"])

    dataset = load_dataset("mimic4")
    folds = pd.read_csv(RESULTS / "mimic4_folds.csv")
    order = {
        int(stay_id): index
        for index, stay_id in enumerate(dataset.cohort["stay_id"])
    }
    folds = (
        folds.assign(_row=folds["stay_id"].map(order))
        .sort_values("_row")
        .reset_index(drop=True)
    )

    predictions = {
        name: np.zeros((len(dataset.labels), 3), dtype=np.float64)
        for name in MODELS
    }
    epoch_rows = []
    tasks = []
    for fold in sorted(folds["fold"].unique()):
        train_index = np.flatnonzero(folds["fold"].to_numpy() != fold)
        test_index = np.flatnonzero(folds["fold"].to_numpy() == fold)
        tasks.append((fold, train_index, test_index))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_configure_training_worker,
        initargs=(worker_threads,),
    ) as executor:
        futures = {
            executor.submit(
                fit_fold,
                dataset,
                fold,
                train_index,
                test_index,
                seed=seed,
                epochs=epochs,
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                model_seeds=model_seeds,
            ): fold
            for fold, train_index, test_index in tasks
        }
        for future in as_completed(futures):
            fold, fold_predictions, fold_epochs = future.result()
            test_index = tasks[fold][2]
            print(f"completed fold {fold + 1}", flush=True)
            for name in MODELS:
                predictions[name][test_index] = fold_predictions[name]
                epoch_rows.extend(
                    {
                        "model": name,
                        "fold": fold,
                        "seed_offset": offset,
                        "best_epoch": value,
                    }
                    for offset, value in enumerate(fold_epochs[name])
                )

    with np.load(RESULTS / "predictions" / "mimic4_cv.npz") as saved:
        predictions["xgb_summary"] = saved["xgb_summary"]
    metrics = evaluate_predictions(
        dataset,
        predictions,
        replicates=replicates,
        seed=seed,
        metric_workers=3,
    )
    output = {
        "description": (
            "Audit-only nested CV with neural refitting on the full outer "
            "training fold after epoch selection."
        ),
        "metrics": metrics,
    }
    (RESULTS / "audit_refit_metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=True), encoding="utf-8"
    )
    np.savez_compressed(
        RESULTS / "predictions" / "mimic4_cv_audit_refit.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        subject_key=dataset.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )
    pd.DataFrame(epoch_rows).to_csv(
        RESULTS / "audit_refit_epochs.csv", index=False
    )
    for model in ("dggru", "dggru_tr", "xgb_summary"):
        point = metrics[model]["point"]
        print(
            f"{model}: macro_auc={point['macro_auc']:.4f} "
            f"any_auc={point['auc_any_deterioration']:.4f} "
            f"brier={point['multiclass_brier']:.4f}",
            flush=True,
        )
    pairs = metrics["paired_differences_vs_dggru_tr"]
    for model in ("dggru", "xgb_summary"):
        low, high = pairs[model]
        print(f"TR - {model}: {low:.4f} to {high:.4f}", flush=True)


if __name__ == "__main__":
    main()
