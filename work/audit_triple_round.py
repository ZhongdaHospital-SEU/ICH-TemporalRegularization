"""Recompute the numeric claims checked in the three reviewer passes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cohort = pd.read_csv(ROOT / "data/processed/mimic4/cohort.csv")
    print("MIMIC-IV cohort", len(cohort))
    print("grades", cohort["outcome"].value_counts().sort_index().to_dict())
    print("emergency", int(cohort["emergency_admission"].sum()),
          round(100 * cohort["emergency_admission"].mean(), 1))
    print("GCS hours by grade")
    print(
        cohort.groupby("outcome")["gcs_total_n"]
        .agg(
            n="count",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .to_string()
    )
    print(
        "GCS hours overall",
        cohort["gcs_total_n"].median(),
        cohort["gcs_total_n"].quantile([0.25, 0.75]).tolist(),
    )

    sensitivity = pd.read_csv(ROOT / "results/tables/sensitivity_summary.csv")
    baseline = sensitivity[
        sensitivity["scenario"].isin(["s1_baseline_max", "s1_baseline_last"])
        & sensitivity["macro_auc"].notna()
    ].pivot_table(
        index=["dataset", "scenario"],
        columns="model",
        values="macro_auc",
    )
    print("\nBaseline-definition sensitivity")
    print(baseline.to_string())
    for dataset in ("mimic4", "mimic3", "eicu"):
        maximum = baseline.loc[(dataset, "s1_baseline_max")]
        last = baseline.loc[(dataset, "s1_baseline_last")]
        print(
            dataset,
            "drop xgb",
            round(float(maximum["xgb_summary"] - last["xgb_summary"]), 6),
            "drop dggru_tr",
            round(float(maximum["dggru_tr"] - last["dggru_tr"]), 6),
        )

    confirmation = sensitivity[
        sensitivity["scenario"].isin(["s2_persistence_2h", "s2_persistence_1h"])
        & sensitivity["macro_auc"].notna()
    ].pivot_table(
        index=["dataset", "scenario"],
        columns="model",
        values="macro_auc",
    )
    print("\nPersistence sensitivity")
    print(confirmation.to_string())
    for dataset in ("mimic4", "mimic3", "eicu"):
        two = confirmation.loc[(dataset, "s2_persistence_2h")]
        one = confirmation.loc[(dataset, "s2_persistence_1h")]
        print(
            dataset,
            "change xgb",
            round(float(one["xgb_summary"] - two["xgb_summary"]), 6),
            "change dggru_tr",
            round(float(one["dggru_tr"] - two["dggru_tr"]), 6),
        )

    claims = json.loads(
        (ROOT / "results/audits/claims_round2.json").read_text(encoding="utf-8")
    )
    print("\nFour-channel versus core")
    print(json.dumps(claims["gcs_only_vs_core"], indent=2))

    trajectory = json.loads(
        (ROOT / "results/experiments/traj_features.json").read_text(
            encoding="utf-8"
        )
    )
    print("\nAll 17 features")
    for model, block in trajectory["metrics"].items():
        if "point" not in block:
            continue
        print(model, round(float(block["point"]["macro_auc"]), 6))

    blend = json.loads(
        (ROOT / "results/experiments/blend_metrics_detail.json").read_text(
            encoding="utf-8"
        )
    )
    print("\nBlend mean and pooled")
    for key in ("xgb_plus_dggru_tr", "xgb_summary"):
        if key not in blend:
            continue
        print(
            key,
            "mean",
            round(float(blend[key]["per_repeat_mean"]["macro_auc"]), 6),
            "pooled",
            round(float(blend[key]["pooled"]["macro_auc"]), 6),
        )

    core = np.load(
        ROOT / "results/predictions/mimic4_cv_core_repeats.npz",
        allow_pickle=True,
    )
    print("\nCore prediction keys", core.files)
    print("Core repeats", sum(name.startswith("repeat") and name.endswith("_xgb_summary") for name in core.files))
    gcs_only = np.load(
        ROOT / "results/predictions/strength_gcs_sequence_repeats.npz",
        allow_pickle=True,
    )
    labels = core["labels"]
    weights = core["repeat0_analysis_weight"]

    def macro_auc(probabilities: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    roc_auc_score(
                        (labels == grade).astype(int),
                        probabilities[:, grade],
                        sample_weight=weights,
                    )
                    for grade in range(3)
                ]
            )
        )

    print(
        "repeat0 formal core DG-GRU+TR",
        round(macro_auc(core["repeat0_dggru_tr"]), 6),
    )
    print(
        "repeat0 four-channel DG-GRU+TR",
        round(macro_auc(gcs_only["repeat0_dggru_tr"]), 6),
    )
    print(
        "repeat0 difference",
        round(
            macro_auc(gcs_only["repeat0_dggru_tr"])
            - macro_auc(core["repeat0_dggru_tr"]),
            6,
        ),
    )


if __name__ == "__main__":
    main()
