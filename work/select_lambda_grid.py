"""Select the temporal-regularization weight from the screening grid.

Selection rule, fixed before running:
1. For each grid weight, compute macro-AUC within each outer fold from the
   out-of-fold predictions.
2. Average the five fold-level values.
3. Take the highest mean.
4. Break ties toward the smaller monotone weight.

External cohorts are not used for selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    values = []
    for grade in range(3):
        indicator = (labels == grade).astype(int)
        if 0 < indicator.sum() < len(indicator):
            values.append(
                float(roc_auc_score(indicator, probabilities[:, grade]))
            )
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    rows = []
    for manifest_path in sorted(
        RESULTS.glob("run_manifest_lambda_m*_s*.json")
    ):
        tag = manifest_path.stem.removeprefix("run_manifest_")
        metrics_path = RESULTS / f"metrics_{tag}.json"
        predictions_path = RESULTS / "predictions" / f"mimic4_cv_{tag}.npz"
        folds_path = RESULTS / f"mimic4_folds_{tag}.csv"
        if not (
            metrics_path.exists()
            and predictions_path.exists()
            and folds_path.exists()
        ):
            continue
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        with np.load(predictions_path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            probabilities = data["dggru_tr"].astype(np.float64)
            stay_id = data["stay_id"].astype(np.int64)
        folds = pd.read_csv(folds_path)
        if "repeat" in folds:
            folds = folds[folds["repeat"].astype(int) == 0]
        fold_lookup = dict(
            zip(
                folds["stay_id"].astype(np.int64),
                folds["fold"].astype(int),
            )
        )
        fold_values: list[float] = []
        for fold in sorted(set(fold_lookup.values())):
            mask = np.asarray(
                [fold_lookup.get(int(value), -1) == fold for value in stay_id],
                dtype=bool,
            )
            if mask.sum() < 2:
                continue
            value = macro_auc(labels[mask], probabilities[mask])
            if np.isfinite(value):
                fold_values.append(value)
        pooled = macro_auc(labels, probabilities)
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        reported = float(
            metrics["development"]["dggru_tr"]["point"]["macro_auc"]
        )
        rows.append(
            {
                "tag": tag,
                "lambda_monotone": float(manifest["lambda_monotone"]),
                "lambda_smooth": float(manifest["lambda_smooth"]),
                "fold_mean_macro_auc": (
                    float(np.mean(fold_values)) if fold_values else float("nan")
                ),
                "fold_sd_macro_auc": (
                    float(np.std(fold_values, ddof=1))
                    if len(fold_values) > 1
                    else 0.0
                ),
                "n_folds": len(fold_values),
                "pooled_macro_auc": pooled,
                "reported_macro_auc": reported,
            }
        )
    if not rows:
        raise SystemExit("No completed lambda-grid runs were found.")
    frame = pd.DataFrame(rows).sort_values(
        ["fold_mean_macro_auc", "lambda_monotone"],
        ascending=[False, True],
    )
    frame["abs_pooled_reported_difference"] = (
        frame["pooled_macro_auc"] - frame["reported_macro_auc"]
    ).abs()
    if float(frame["abs_pooled_reported_difference"].max()) > 1e-9:
        raise SystemExit(
            "Pooled grid macro-AUC does not match the reported metric; "
            "refusing to select a weight."
        )
    if frame["n_folds"].min() != 5:
        raise SystemExit("At least one grid run lacks five fold-level estimates.")
    selected = frame.iloc[0]
    output = {
        "selection_rule": {
            "metric": "mean_of_five_outer_fold_macro_auc",
            "direction": "maximise",
            "tie_break": "smaller_lambda_monotone",
            "external_cohorts_used": False,
            "cv_repeats_per_grid_weight": 1,
        },
        "selected": {
            "tag": str(selected["tag"]),
            "lambda_monotone": float(selected["lambda_monotone"]),
            "lambda_smooth": float(selected["lambda_smooth"]),
            "fold_mean_macro_auc": float(selected["fold_mean_macro_auc"]),
            "fold_sd_macro_auc": float(selected["fold_sd_macro_auc"]),
            "pooled_macro_auc": float(selected["pooled_macro_auc"]),
        },
        "grid": frame.to_dict(orient="records"),
    }
    frame.to_csv(RESULTS / "lambda_grid_selection.csv", index=False)
    with open(RESULTS / "lambda_grid_selection.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(json.dumps(output["selected"], indent=2))


if __name__ == "__main__":
    main()
