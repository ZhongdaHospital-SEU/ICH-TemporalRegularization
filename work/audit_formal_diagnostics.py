"""Audit formal predictions, paired contrasts, and finite metrics.

This script is intentionally read-only with respect to model artifacts.  It
recomputes the headline paired contrasts from the saved repeated-CV
predictions so that a missing interval cannot be mistaken for a null effect.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.common import PROCESSED, RESULTS
from src.ich_dynamic.train_evaluate import (
    multiclass_auc,
    paired_auc_difference_repeated,
    point_metrics,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _interval_status(interval: object) -> str:
    if not isinstance(interval, list) or len(interval) != 2:
        return "missing"
    if all(_finite(value) for value in interval):
        return "finite"
    return "non-finite"


def _prediction_audit(path: Path, models: list[str]) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        labels = archive["labels"].astype(int)
        records = {}
        for model in models:
            probabilities = archive[model].astype(float)
            point = point_metrics(labels, probabilities)
            records[model] = {
                "macro_auc": multiclass_auc(labels, probabilities),
                "grade_auc": {
                    f"grade_{grade}": point.get(f"auc_grade_{grade}")
                    for grade in (0, 1, 2)
                },
                "auc_any_deterioration": point.get("auc_any_deterioration"),
                "auc_severe_deterioration": point.get(
                    "auc_severe_deterioration"
                ),
                "finite": bool(np.isfinite(probabilities).all()),
                "row_sum_max_abs_error": float(
                    np.abs(probabilities.sum(axis=1) - 1.0).max()
                ),
                "class_counts": {
                    str(grade): int((labels == grade).sum())
                    for grade in (0, 1, 2)
                },
            }
    return records


def paired_repeated_audit(
    dataset: str,
    tag: str,
    models: list[str],
    replicates: int,
) -> dict:
    suffix = f"_{tag}" if tag else ""
    path = RESULTS / "predictions" / f"mimic4_cv{suffix}_repeats.npz"
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    groups = cohort["subject_key"].astype(str).to_numpy()
    with np.load(path, allow_pickle=True) as archive:
        labels = archive["labels"].astype(int)
        repeats = max(
            int(key.removeprefix("repeat").split("_", 1)[0]) + 1
            for key in archive.files
            if key.startswith("repeat")
        )
        primary = [
            archive[f"repeat{repeat}_dggru_tr"].astype(float)
            for repeat in range(repeats)
        ]
        records = {}
        for comparator in models:
            if comparator == "dggru_tr":
                continue
            comparator_values = [
                archive[f"repeat{repeat}_{comparator}"].astype(float)
                for repeat in range(repeats)
            ]
            differences = [
                multiclass_auc(labels, primary_values)
                - multiclass_auc(labels, comparator_values)
                for primary_values, comparator_values in zip(
                    primary,
                    comparator_values,
                )
            ]
            interval = paired_auc_difference_repeated(
                labels,
                primary,
                comparator_values,
                groups,
                replicates=replicates,
                seed=20261024,
            )
            records[comparator] = {
                "repeat_differences": differences,
                "mean": float(np.mean(differences)),
                "sd": float(np.std(differences, ddof=1)),
                "bootstrap_interval": interval,
                "bootstrap_interval_status": _interval_status(interval),
            }
    return {
        "dataset": dataset,
        "repeats": repeats,
        "path": str(path),
        "comparisons": records,
    }


def audit(tag: str, replicates: int) -> dict:
    suffix = f"_{tag}" if tag else ""
    metrics_path = RESULTS / f"metrics{suffix}.json"
    manifest_path = RESULTS / f"run_manifest{suffix}.json"
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    models = [str(model) for model in manifest["models"]]

    reported = metrics.get("development", {}).get(
        "paired_differences_vs_dggru_tr",
        {},
    )
    reported_status = {
        comparator: {
            "reported_interval": interval,
            "status": _interval_status(interval),
        }
        for comparator, interval in reported.items()
    }

    prediction_audits = {}
    for dataset, stem in (
        ("development", "mimic4_cv"),
        ("mimic3", "mimic3_external"),
        ("eicu", "eicu_external"),
    ):
        prediction_audits[dataset] = _prediction_audit(
            RESULTS / "predictions" / f"{stem}{suffix}.npz",
            models,
        )

    return {
        "tag": tag,
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "models": models,
        "reported_paired_differences": reported_status,
        "recomputed_paired_differences": paired_repeated_audit(
            "development",
            tag,
            models,
            replicates,
        ),
        "prediction_audits": prediction_audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="formal_round5")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.tag, args.replicates)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(args.output)
    print(
        json.dumps(
            {
                "tag": args.tag,
                "reported_paired_differences": payload[
                    "reported_paired_differences"
                ],
                "recomputed_paired_differences": {
                    key: {
                        "mean": value["mean"],
                        "bootstrap_interval": value["bootstrap_interval"],
                        "bootstrap_interval_status": value[
                            "bootstrap_interval_status"
                        ],
                    }
                    for key, value in payload[
                        "recomputed_paired_differences"
                    ]["comparisons"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
