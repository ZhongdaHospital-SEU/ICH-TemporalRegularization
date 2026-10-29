"""Exploratory screen: multiclass XGBoost versus cumulative binary XGBoost.

This is a frozen-partition screen, not a formal result.  Both arms use the
same 23-feature matrix, the same patient-grouped folds, the same fixed
XGBoost candidate, and the same model seeds.  The only difference is how the
three-level ordinal outcome is parameterised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import load_config


def _fit_multiclass(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    sample_weight: np.ndarray,
) -> np.ndarray:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        **te.XGBOOST_DEFAULTS,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        eval_metric="mlogloss",
    )
    model.fit(train_x, train_y, sample_weight=sample_weight, verbose=False)
    return model.predict_proba(test_x)


def _fit_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    sample_weight: np.ndarray,
) -> np.ndarray:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        **te.XGBOOST_DEFAULTS,
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        eval_metric="logloss",
    )
    model.fit(train_x, train_y, sample_weight=sample_weight, verbose=False)
    return model.predict_proba(test_x)[:, 1]


def _cumulative_to_grade(
    p_any: np.ndarray,
    p_severe: np.ndarray,
) -> np.ndarray:
    p_any = np.clip(p_any, 1e-7, 1.0 - 1e-7)
    p_severe = np.clip(p_severe, 1e-7, 1.0 - 1e-7)
    p_severe = np.minimum(p_severe, p_any)
    probabilities = np.column_stack(
        [1.0 - p_any, p_any - p_severe, p_severe]
    )
    probabilities = np.clip(probabilities, 1e-7, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()

    te.ensure_dirs()
    config = load_config()
    seed = int(config["random_seed"])
    dataset = te.load_dataset("mimic4")
    frame = te.tabular_frame(dataset, "core")
    labels = dataset.labels.astype(int)
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    keys = ("multiclass", "ordinal_cumulative")
    predictions = {
        key: np.zeros((len(frame), 3), dtype=np.float64) for key in keys
    }
    fold_rows = []
    splits, split_meta = te.build_repeated_splits(
        labels,
        groups,
        folds=args.folds,
        repeats=args.repeats,
        seed=seed,
    )
    for (train_index, test_index), (repeat, fold) in zip(splits, split_meta):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        train_x, test_x = te.impute_tabular(train, test)
        train_y = labels[train_index]
        weight = np.asarray(dataset.analysis_weight, dtype=float)[train_index]
        for offset in range(args.seed_count):
            model_seed = seed + repeat * 10_000 + fold * 100 + offset
            predictions["multiclass"][test_index] += _fit_multiclass(
                train_x, train_y, test_x, model_seed, weight
            )
            p_any = _fit_binary(
                train_x,
                (train_y >= 1).astype(int),
                test_x,
                model_seed,
                weight,
            )
            p_severe = _fit_binary(
                train_x,
                (train_y >= 2).astype(int),
                test_x,
                model_seed,
                weight,
            )
            predictions["ordinal_cumulative"][test_index] += (
                _cumulative_to_grade(p_any, p_severe)
            )
        for key in keys:
            predictions[key][test_index] /= args.seed_count
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n_test": int(len(test_index)),
            }
        )
        print(
            f"completed repeat {repeat + 1} fold {fold + 1}",
            flush=True,
        )

    metrics = {
        key: te.point_metrics(labels, probabilities)
        for key, probabilities in predictions.items()
    }
    difference = (
        metrics["ordinal_cumulative"]["macro_auc"]
        - metrics["multiclass"]["macro_auc"]
    )
    interval = te.paired_auc_difference(
        labels,
        predictions["ordinal_cumulative"],
        predictions["multiclass"],
        groups,
        replicates=args.replicates,
        seed=seed,
        sample_weight=np.asarray(dataset.analysis_weight, dtype=float),
    )
    payload = {
        "purpose": (
            "Exploratory frozen-partition screen of an ordinal cumulative "
            "XGBoost parameterisation against the prespecified multiclass "
            "XGBoost. Not a formal result."
        ),
        "n": int(len(labels)),
        "outcomes": {
            str(grade): int((labels == grade).sum()) for grade in (0, 1, 2)
        },
        "features": list(frame.columns),
        "fixed_parameters": dict(te.XGBOOST_DEFAULTS),
        "folds": args.folds,
        "repeats": args.repeats,
        "model_seeds": args.seed_count,
        "metrics": metrics,
        "paired_ordinal_minus_multiclass": {
            "macro_auc_difference": float(difference),
            "ci": [float(interval[0]), float(interval[1])],
        },
        "fold_partition": fold_rows,
    }
    output = te.RESULTS / "experiments" / "ordinal_xgb_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], indent=2, allow_nan=True))
    print(json.dumps(payload["paired_ordinal_minus_multiclass"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
