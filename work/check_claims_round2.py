"""Recompute the intervals and contrasts that the manuscript quotes without a source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
RNG = np.random.default_rng(20261028)
REPLICATES = 2000


def slope(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = (labels >= 1).astype(int)
    predicted = np.clip(probabilities[:, 1:].sum(axis=1), 1e-5, 1 - 1e-5)
    logit = np.log(predicted / (1 - predicted)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(logit, target)
    return float(model.coef_[0, 0])


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
                for grade in range(3)
            ]
        )
    )


def cluster_bootstrap(
    labels: np.ndarray,
    subjects: np.ndarray,
    statistic,
    replicates: int = REPLICATES,
) -> tuple[float, float, float]:
    point = statistic(np.arange(len(labels)))
    unique = np.unique(subjects)
    index = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    draws = np.empty(replicates)
    for replicate in range(replicates):
        chosen = RNG.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[subject] for subject in chosen])
        draws[replicate] = statistic(rows)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    core = np.load(PRED / "mimic4_cv_core_repeats.npz", allow_pickle=True)
    labels = core["labels"]
    subjects = core["subject_key"].astype(str)
    report: dict[str, object] = {}

    for model in ("xgb_summary", "dggru_tr"):
        per_repeat = [
            slope(labels, core[f"repeat{index}_{model}"]) for index in range(5)
        ]
        point, low, high = cluster_bootstrap(
            labels,
            subjects,
            lambda rows, model=model: float(
                np.mean(
                    [
                        slope(
                            labels[rows],
                            core[f"repeat{index}_{model}"][rows],
                        )
                        for index in range(5)
                    ]
                )
            ),
        )
        report[f"calibration_slope_any_{model}"] = {
            "per_repeat": [round(value, 4) for value in per_repeat],
            "mean": round(float(np.mean(per_repeat)), 4),
            "bootstrap_point": round(point, 4),
            "ci": [round(low, 4), round(high, 4)],
        }

    comparisons = {
        "mask_removed_vs_core": "strength_monitor_drop_nomask_repeats.npz",
        "gcs_only_vs_core": "strength_gcs_sequence_repeats.npz",
    }
    for name, filename in comparisons.items():
        other = np.load(PRED / filename, allow_pickle=True)
        assert np.array_equal(other["labels"], labels)
        assert np.array_equal(other["subject_key"].astype(str), subjects)
        repeats = [
            index
            for index in range(5)
            if f"repeat{index}_dggru_tr" in other.files
        ]

        def difference(rows: np.ndarray, other=other, repeats=repeats) -> float:
            return float(
                np.mean(
                    [
                        macro_auc(labels[rows], other[f"repeat{index}_dggru_tr"][rows])
                        - macro_auc(labels[rows], core[f"repeat{index}_dggru_tr"][rows])
                        for index in repeats
                    ]
                )
            )

        point, low, high = cluster_bootstrap(labels, subjects, difference)
        report[name] = {
            "repeats": repeats,
            "difference": round(point, 4),
            "ci": [round(low, 4), round(high, 4)],
        }

    sensitivity = pd.read_csv(ROOT / "results" / "tables" / "sensitivity_summary.csv")
    estimable = sensitivity[
        (sensitivity["fit_status"] == "saved") & sensitivity["macro_auc"].notna()
    ]
    wide = estimable.pivot_table(
        index=["dataset", "scenario"], columns="model", values="macro_auc"
    ).dropna()
    wide["delta_dgtr_minus_xgb"] = wide["dggru_tr"] - wide["xgb_summary"]
    exceptions = wide[wide["delta_dgtr_minus_xgb"] > 0.0005]
    report["sensitivity_scenarios"] = {
        "estimable": int(len(wide)),
        "dgtr_higher": [
            {
                "dataset": dataset,
                "scenario": scenario,
                "xgb": round(float(row["xgb_summary"]), 4),
                "dgtr": round(float(row["dggru_tr"]), 4),
                "delta": round(float(row["delta_dgtr_minus_xgb"]), 4),
            }
            for (dataset, scenario), row in exceptions.iterrows()
        ],
    }

    (ROOT / "results" / "audits" / "claims_round2.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
