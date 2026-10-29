"""Quantify the three shortfalls: grade detection, transport calibration, label rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

from eicu_site_calibration import (  # noqa: E402
    _apply_with_offset,
    _calibration_slope_intercept,
    _ece,
    _expit,
    _fit_with_offset,
    _logit,
    _safe_auc,
)

PRED = ROOT / "results" / "predictions"
RNG = np.random.default_rng(20261028)


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
                for grade in range(3)
            ]
        )
    )


def sensitivity_at_specificity(
    labels: np.ndarray, scores: np.ndarray, specificity: float
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(1.0 - specificity, fpr, tpr))


def detection_curve() -> dict:
    payload = np.load(PRED / "mimic4_cv_core_repeats.npz", allow_pickle=True)
    labels = payload["labels"]
    models = {
        "xgb_summary": [payload[f"repeat{i}_xgb_summary"] for i in range(5)],
        "dggru_tr": [payload[f"repeat{i}_dggru_tr"] for i in range(5)],
        "ensemble": [
            0.5 * (payload[f"repeat{i}_xgb_summary"] + payload[f"repeat{i}_dggru_tr"])
            for i in range(5)
        ],
    }
    result: dict[str, object] = {}
    for name, probabilities in models.items():
        per_grade = {}
        for grade in range(3):
            truth = (labels == grade).astype(int)
            sens90 = [
                sensitivity_at_specificity(truth, p[:, grade], 0.90)
                for p in probabilities
            ]
            sens80 = [
                sensitivity_at_specificity(truth, p[:, grade], 0.80)
                for p in probabilities
            ]
            predictions = [
                np.where(
                    p[:, 0] >= 0.5,
                    0,
                    np.where((p[:, 0] + p[:, 1]) >= 0.5, 1, 2),
                )
                for p in probabilities
            ]
            recall = [
                float(np.mean(pred[truth == 1] == grade)) for pred in predictions
            ]
            per_grade[f"grade_{grade}"] = {
                "sensitivity_at_specificity_0.90": round(float(np.mean(sens90)), 3),
                "sensitivity_at_specificity_0.80": round(float(np.mean(sens80)), 3),
                "recall_at_median_rule": round(float(np.mean(recall)), 3),
            }
        result[name] = per_grade
    return result


def external_recalibration() -> dict:
    payload = np.load(PRED / "mimic3_external_core.npz", allow_pickle=True)
    labels = payload["labels"]
    groups = payload["subject_key"].astype(str)
    truth = (labels >= 1).astype(int)
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261028).split(
            np.zeros(len(labels)), truth, groups
        )
    )
    methods = {
        "no_update": 0,
        "intercept": 2,
        "slope_intercept": 2,
        "quadratic": 3,
    }
    result: dict[str, object] = {}
    for model in ("xgb_summary", "dggru_tr"):
        probabilities = payload[model]
        p_any = np.clip(probabilities[:, 1:].sum(axis=1), 1e-6, 1 - 1e-6)
        offset = _logit(p_any)
        entry = {}
        for name, columns in methods.items():
            predicted = np.empty(len(labels))
            for train, test in splits:
                if name == "no_update":
                    predicted[test] = p_any[test]
                    continue
                if name == "intercept":
                    beta = _fit_with_offset(truth[train], offset[train], cols=2)
                    predicted[test] = _apply_with_offset(offset[test], beta)
                else:
                    beta = _fit_with_offset(truth[train], offset[train], cols=columns)
                    predicted[test] = _apply_with_offset(offset[test], beta)
            slope, intercept = _calibration_slope_intercept(truth, predicted)
            entry[name] = {
                "ece": round(_ece(truth, predicted), 3),
                "brier": round(float(np.mean((predicted - truth) ** 2)), 3),
                "auroc": round(_safe_auc(truth, predicted), 3),
                "slope": round(slope, 2),
                "intercept": round(intercept, 2),
            }
        result[model] = entry
    return result


def label_rule() -> dict:
    result: dict[str, object] = {}
    for cohort, filename, rng in (
        ("mimic3", "sensitivity_mimic3_s1_baseline_last_repeats.npz", None),
        ("eicu", "sensitivity_eicu_s1_baseline_last_repeats.npz", "hospital"),
        ("mimic4", "sensitivity_mimic4_s1_baseline_last_repeats.npz", None),
    ):
        payload = np.load(PRED / filename, allow_pickle=True)
        labels = payload["labels"]
        subjects = payload["subject_key"].astype(str)
        xgb = payload["repeat0_xgb_summary"]
        dgtr = payload["repeat0_dggru_tr"]
        blend = 0.5 * (xgb + dgtr)
        point = macro_auc(labels, blend) - macro_auc(labels, xgb)
        unique = np.unique(subjects)
        index = {subject: np.flatnonzero(subjects == subject) for subject in unique}
        draws = []
        for _ in range(2000):
            chosen = RNG.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([index[subject] for subject in chosen])
            if len(np.unique(labels[rows])) < 3:
                continue
            draws.append(
                macro_auc(labels[rows], blend[rows])
                - macro_auc(labels[rows], xgb[rows])
            )
        result[cohort] = {
            "n": int(len(labels)),
            "grades": [int(v) for v in np.bincount(labels, minlength=3)],
            "xgb": round(macro_auc(labels, xgb), 3),
            "dggru_tr": round(macro_auc(labels, dgtr), 3),
            "blend": round(macro_auc(labels, blend), 3),
            "blend_minus_xgb": round(point, 3),
            "blend_minus_xgb_ci": [
                round(float(np.quantile(draws, 0.025)), 3),
                round(float(np.quantile(draws, 0.975)), 3),
            ],
        }
    return result


def main() -> None:
    report = {
        "detection_curve": detection_curve(),
        "mimic3_recalibration": external_recalibration(),
        "baseline_last": label_rule(),
    }
    (ROOT / "results" / "audits" / "strengthen_shortfalls.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
