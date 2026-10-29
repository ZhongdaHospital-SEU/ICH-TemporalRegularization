"""Two-threshold (ordinal) intercept recalibration of transported predictions.

The transported models are used without refitting.  For each external cohort,
an intercept is fitted for the any-deterioration threshold and a second
intercept for the severe-deterioration threshold.  Both are estimated in
training folds of a patient-grouped five-fold split and applied to the
held-out fold.  Class probabilities are then reconstructed from the two
calibrated cumulative probabilities, so the target is the same three-class
endpoint as the primary analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

from eicu_site_calibration import (  # noqa: E402
    _expit,
    _fit_with_offset,
    _logit,
)

PRED = ROOT / "results" / "predictions"
OUTPUT = ROOT / "results" / "audits" / "ordinal_recalibration.json"
SEED = 20261028
RNG = np.random.default_rng(SEED)


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def top_label_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )
    return float(total)


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
                for grade in range(3)
            ]
        )
    )


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def rebuild(p_any: np.ndarray, p_severe: np.ndarray) -> np.ndarray:
    p_any = clip(p_any)
    p_severe = np.minimum(clip(p_severe), p_any)
    probabilities = np.column_stack(
        [1.0 - p_any, p_any - p_severe, p_severe]
    )
    probabilities = np.clip(probabilities, 1e-7, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def applied(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    any_minus, severe_minus = thresholds
    p_any = clip(probabilities[:, 1:].sum(axis=1))
    p_severe = clip(probabilities[:, 2])
    return rebuild(
        _expit(_logit(p_any) + any_minus),
        _expit(_logit(p_severe) + severe_minus),
    )


def recalibrate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    splits = list(
        StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=SEED
        ).split(np.zeros(len(labels)), labels, groups)
    )
    predicted = np.empty_like(probabilities)
    fitted: list[list[float]] = []
    for train, test in splits:
        train_p = probabilities[train]
        offsets = [
            _logit(clip(train_p[:, 1:].sum(axis=1))),
            _logit(clip(train_p[:, 2])),
        ]
        targets = [(labels[train] >= 1).astype(float), (labels[train] >= 2).astype(float)]
        betas = [
            float(_fit_with_offset(target, offset, cols=2)[0])
            for target, offset in zip(targets, offsets)
        ]
        fitted.append(betas)
        predicted[test] = applied(probabilities[test], np.asarray(betas))
    return predicted, [value for pair in fitted for value in pair]


def bootstrap_interval(
    labels: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    groups: np.ndarray,
    replicates: int = 1000,
) -> dict:
    unique = np.unique(groups)
    index = {group: np.flatnonzero(groups == group) for group in unique}
    ece_before, ece_after, delta = [], [], []
    for _ in range(replicates):
        chosen = RNG.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[group] for group in chosen])
        if len(np.unique(labels[rows])) < 3:
            continue
        b = top_label_ece(labels[rows], before[rows])
        a = top_label_ece(labels[rows], after[rows])
        ece_before.append(b)
        ece_after.append(a)
        delta.append(a - b)
    quantile = lambda values: [
        round(float(np.quantile(values, 0.025)), 3),
        round(float(np.quantile(values, 0.975)), 3),
    ]
    return {
        "ece_before_ci": quantile(ece_before),
        "ece_after_ci": quantile(ece_after),
        "ece_change_ci": quantile(delta),
    }


def run(cohort: str, filename: str) -> dict:
    payload = np.load(PRED / filename, allow_pickle=True)
    labels = payload["labels"].astype(int)
    subjects = payload["subject_key"].astype(str)
    if cohort == "eicu":
        bootstrap_groups = np.array(
            [value.split("_", 1)[1].split("-", 1)[0] for value in subjects]
        )
        bootstrap_cluster = "hospital"
    else:
        bootstrap_groups = subjects
        bootstrap_cluster = "patient"
    result = {
        "n": int(len(labels)),
        "bootstrap_cluster": bootstrap_cluster,
    }
    for model in ("xgb_summary", "dggru_tr"):
        probabilities = payload[model]
        after, betas = recalibrate(labels, probabilities, subjects)
        result[model] = {
            "before": {
                "ece": round(top_label_ece(labels, probabilities), 3),
                "macro_auc": round(macro_auc(labels, probabilities), 3),
                "brier": round(multiclass_brier(labels, probabilities), 3),
            },
            "after": {
                "ece": round(top_label_ece(labels, after), 3),
                "macro_auc": round(macro_auc(labels, after), 3),
                "brier": round(multiclass_brier(labels, after), 3),
            },
            "fold_intercepts_any_severe": [round(value, 3) for value in betas],
            "bootstrap": bootstrap_interval(
                labels,
                probabilities,
                after,
                bootstrap_groups,
            ),
        }
        print(cohort, model, json.dumps(result[model], indent=2), flush=True)
    return result


def main() -> None:
    report = {
        "purpose": (
            "Two-threshold ordinal intercept recalibration of transported "
            "predictions, fitted in training folds of a patient-grouped "
            "five-fold split within each external cohort. No refitting of the "
            "transported models."
        ),
        "cohorts": {
            "mimic3": run("mimic3", "mimic3_external_core.npz"),
            "eicu": run("eicu", "eicu_external_core.npz"),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
