"""Exploratory screen of alternative XGBoost outcome parameterisations.

The screen uses one frozen CV repeat, three model seeds, the prespecified
23-feature matrix, and a single fixed XGBoost candidate.  It is intended only
to decide whether a formal run is worth starting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import load_config


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


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-7, None)
    return values / values.sum(axis=1, keepdims=True)


def _class_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(float)
    if mode == "none":
        return np.ones(len(labels), dtype=float)
    if mode == "sqrt":
        weights = np.sqrt(1.0 / np.maximum(counts, 1.0))
    elif mode == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
    elif mode == "grade1_x2":
        weights = np.ones(3, dtype=float)
        weights[1] = 2.0
    elif mode == "grade1_x3":
        weights = np.ones(3, dtype=float)
        weights[1] = 3.0
    else:
        raise ValueError(f"Unknown class-weight mode: {mode}")
    weights = weights / np.average(weights, weights=counts)
    return weights[labels]


def _ovr_to_grade(
    probabilities: np.ndarray,
) -> np.ndarray:
    return _normalize(probabilities)


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
    base_weight = np.asarray(dataset.analysis_weight, dtype=float)
    arms = (
        "multiclass",
        "ovr",
        "sqrt",
        "inverse",
        "grade1_x2",
        "grade1_x3",
    )
    predictions = {
        arm: np.zeros((len(frame), 3), dtype=np.float64) for arm in arms
    }
    splits, split_meta = te.build_repeated_splits(
        labels,
        groups,
        folds=args.folds,
        repeats=args.repeats,
        seed=seed,
    )
    for (train_index, test_index), (repeat, fold) in zip(splits, split_meta):
        train_x, test_x = te.impute_tabular(
            frame.iloc[train_index], frame.iloc[test_index]
        )
        train_y = labels[train_index]
        weight = base_weight[train_index]
        for offset in range(args.seed_count):
            model_seed = seed + repeat * 10_000 + fold * 100 + offset
            predictions["multiclass"][test_index] += _fit_multiclass(
                train_x, train_y, test_x, model_seed, weight
            )
            for arm in ("sqrt", "inverse", "grade1_x2", "grade1_x3"):
                arm_weight = weight * _class_weights(train_y, arm)
                predictions[arm][test_index] += _fit_multiclass(
                    train_x, train_y, test_x, model_seed, arm_weight
                )
            ovr = np.column_stack(
                [
                    _fit_binary(
                        train_x,
                        (train_y == grade).astype(int),
                        test_x,
                        model_seed,
                        weight,
                    )
                    for grade in range(3)
                ]
            )
            predictions["ovr"][test_index] += _ovr_to_grade(ovr)
        for arm in arms:
            predictions[arm][test_index] /= args.seed_count
        print(
            f"completed repeat {repeat + 1} fold {fold + 1}",
            flush=True,
        )

    metrics = {
        arm: te.point_metrics(labels, probabilities)
        for arm, probabilities in predictions.items()
    }
    comparisons = {}
    for arm in arms[1:]:
        interval = te.paired_auc_difference(
            labels,
            predictions[arm],
            predictions["multiclass"],
            groups,
            replicates=args.replicates,
            seed=seed,
            sample_weight=base_weight,
        )
        comparisons[arm] = {
            "macro_auc_difference": float(
                metrics[arm]["macro_auc"]
                - metrics["multiclass"]["macro_auc"]
            ),
            "ci": [float(interval[0]), float(interval[1])],
            "grade_1_auc_difference": float(
                metrics[arm]["auc_grade_1"]
                - metrics["multiclass"]["auc_grade_1"]
            ),
        }
    payload = {
        "purpose": (
            "Exploratory frozen-partition screen of XGBoost outcome "
            "parameterisations and training weights. Not a formal result."
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
        "paired_vs_multiclass": comparisons,
    }
    output = te.RESULTS / "experiments" / "xgb_parameterisation_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    for arm in arms:
        print(
            f"{arm:14s} macro={metrics[arm]['macro_auc']:.4f} "
            f"grade1={metrics[arm]['auc_grade_1']:.4f} "
            f"brier={metrics[arm]['multiclass_brier']:.4f}"
        )
    print(json.dumps(comparisons, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
