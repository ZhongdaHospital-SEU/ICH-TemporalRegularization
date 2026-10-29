"""Fail-closed validation of one formal model run.

The validator checks artifact completeness, model coverage, repeated-CV
alignment, temporal-fit bookkeeping, probability integrity, predictor
leakage, data/source hashes, and external cohort identifiers. It does not
refit models or alter any analysis artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.common import (
    CHANNELS,
    PROCESSED,
    PROJECT_ROOT,
    RESULTS,
    death_in_icu_within_window,
)
from src.ich_dynamic.train_evaluate import (
    ALL_MODEL_ORDER,
    CORE_STATIC,
    DATA_CONTRACT_FILES,
    MODEL_META,
    OUTLINE_MODEL_ORDER,
    POST_LANDMARK_COLUMNS,
    SECONDARY_MODEL_ORDER,
    _validated_metric_weight,
    aggregate_point_metrics,
    build_repeated_splits,
    point_metrics,
    sha256_file,
)
from work.audit_source_drift import training_runtime_sources


REQUIRED_MODELS = ALL_MODEL_ORDER
TEMPORAL_MODELS = tuple(
    name for name, meta in MODEL_META.items() if meta["kind"] == "temporal"
)
RUNTIME_SOURCE_FILES = frozenset(training_runtime_sources())


class ValidationFailure(RuntimeError):
    """Raised when a formal artifact violates the run contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def _load_json(path: Path) -> dict:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _probability_checks(
    path: Path,
    models: tuple[str, ...],
    expected_ids: np.ndarray | None = None,
    expected_labels: np.ndarray | None = None,
) -> dict[str, object]:
    _require(path.is_file(), f"Missing prediction artifact: {path}")
    records: dict[str, object] = {}
    with np.load(path, allow_pickle=True) as archive:
        _require("labels" in archive, f"{path} lacks labels")
        _require("stay_id" in archive, f"{path} lacks stay_id")
        labels = archive["labels"].astype(int)
        stay_ids = archive["stay_id"].astype(np.int64)
        _require(
            len(np.unique(stay_ids)) == len(stay_ids),
            f"{path} contains duplicate stay_id values",
        )
        if expected_ids is not None:
            _require(
                np.array_equal(stay_ids, expected_ids),
                f"{path} stay_id order differs from the cohort",
            )
        if expected_labels is not None:
            _require(
                np.array_equal(labels, expected_labels),
                f"{path} labels differ from the cohort",
            )
        _require(
            set(np.unique(labels)).issubset({0, 1, 2}),
            f"{path} has labels outside 0/1/2",
        )
        for model in models:
            _require(model in archive, f"{path} lacks model {model}")
            probabilities = archive[model].astype(np.float64)
            _require(
                probabilities.shape == (len(labels), 3),
                f"{path}:{model} has shape {probabilities.shape}, "
                f"expected {(len(labels), 3)}",
            )
            _require(
                np.isfinite(probabilities).all(),
                f"{path}:{model} contains non-finite probabilities",
            )
            _require(
                (probabilities >= 0).all(),
                f"{path}:{model} contains negative probabilities",
            )
            _require(
                np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                f"{path}:{model} probabilities do not sum to one",
            )
            records[model] = {
                "rows": int(len(probabilities)),
                "min": float(probabilities.min()),
                "max": float(probabilities.max()),
            }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n": int(len(labels)),
        "outcomes": {
            str(grade): int((labels == grade).sum()) for grade in (0, 1, 2)
        },
        "models": records,
    }


def _repeated_probability_checks(
    path: Path,
    models: tuple[str, ...],
    model_repeats: dict[str, int],
    expected_ids: np.ndarray,
    expected_labels: np.ndarray,
) -> dict[str, object]:
    _require(path.is_file(), f"Missing repeated-CV predictions: {path}")
    with np.load(path, allow_pickle=True) as archive:
        _require(
            np.array_equal(archive["stay_id"].astype(np.int64), expected_ids),
            f"{path} stay_id order differs from the cohort",
        )
        _require(
            np.array_equal(archive["labels"].astype(int), expected_labels),
            f"{path} labels differ from the cohort",
        )
        for model in models:
            repeats = int(model_repeats.get(model, 0))
            _require(
                repeats >= 1,
                f"{path} has no declared CV repeat count for {model}",
            )
            for repeat in range(repeats):
                key = f"repeat{repeat}_{model}"
                _require(key in archive, f"{path} lacks {key}")
                probabilities = archive[key].astype(np.float64)
                _require(
                    probabilities.shape == (len(expected_labels), 3),
                    f"{path}:{key} has invalid shape",
                )
                _require(
                    np.isfinite(probabilities).all(),
                    f"{path}:{key} contains non-finite probabilities",
                )
                _require(
                    (probabilities >= 0).all(),
                    f"{path}:{key} contains negative probabilities",
                )
                _require(
                    np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                    f"{path}:{key} probabilities do not sum to one",
                )
    return {"path": str(path), "sha256": sha256_file(path)}


def _compare_metric_payload(
    expected,
    observed,
    *,
    context: str,
    tolerance: float = 1e-6,
) -> None:
    """Require every recomputed metric field to match the serialized metric."""
    if isinstance(expected, dict):
        _require(isinstance(observed, dict), f"{context} is not a metric object")
        missing = sorted(set(expected).difference(observed))
        _require(not missing, f"{context} lacks fields: {', '.join(missing)}")
        for key, value in expected.items():
            _compare_metric_payload(
                value,
                observed[key],
                context=f"{context}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_values = np.asarray(expected, dtype=float)
        observed_values = np.asarray(observed, dtype=float)
        _require(
            expected_values.shape == observed_values.shape,
            f"{context} shape differs after metric recomputation",
        )
        _require(
            np.allclose(
                expected_values,
                observed_values,
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            ),
            f"{context} differs after metric recomputation",
        )
        return
    expected_value = float(expected)
    observed_value = float(observed)
    if np.isnan(expected_value) and np.isnan(observed_value):
        return
    _require(
        np.isclose(
            expected_value,
            observed_value,
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        ),
        f"{context} differs after metric recomputation: "
        f"{expected_value} vs {observed_value}",
    )


def _metric_recompute_checks(
    metrics: dict,
    development_cohort: pd.DataFrame,
    *,
    suffix: str,
    model_repeats: dict[str, int],
) -> dict[str, object]:
    """Recompute point discrimination/calibration metrics from saved probabilities."""
    labels = development_cohort["outcome"].astype(int).to_numpy()
    if "analysis_weight" in development_cohort:
        weights = _validated_metric_weight(
            pd.to_numeric(
                development_cohort["analysis_weight"], errors="raise"
            ).to_numpy(dtype=float),
            len(labels),
        )
    else:
        weights = None
    repeated_path = RESULTS / "predictions" / f"mimic4_cv{suffix}_repeats.npz"
    verified_models = 0
    with np.load(repeated_path, allow_pickle=True) as archive:
        averaged = {}
        for model in REQUIRED_MODELS:
            repeats = int(model_repeats.get(model, 0))
            _require(repeats >= 1, f"No CV repeat count recorded for {model}")
            rows = [
                point_metrics(
                    labels,
                    archive[f"repeat{repeat}_{model}"].astype(np.float64),
                    sample_weight=weights,
                )
                for repeat in range(repeats)
            ]
            expected_point = aggregate_point_metrics(rows)
            observed_block = metrics["development"][model]
            _compare_metric_payload(
                expected_point,
                observed_block.get("point"),
                context=f"development.{model}.point",
            )
            expected_repeat_metrics = {
                key: [float(row[key]) for row in rows]
                for key in rows[0]
                if key not in {"n", "confusion_matrix", "mode_confusion_matrix"}
                and np.isscalar(rows[0][key])
            }
            _compare_metric_payload(
                expected_repeat_metrics,
                observed_block.get("repeat_metrics"),
                context=f"development.{model}.repeat_metrics",
            )
            averaged[model] = np.mean(
                [
                    archive[f"repeat{repeat}_{model}"]
                    for repeat in range(repeats)
                ],
                axis=0,
            )
            verified_models += 1
    ensemble = metrics.get("development_probability_ensemble", {}).get("point", {})
    for model, probabilities in averaged.items():
        _compare_metric_payload(
            point_metrics(labels, probabilities, sample_weight=weights),
            ensemble.get(model),
            context=f"development_probability_ensemble.{model}",
        )
    external_verified = {}
    for dataset in ("mimic3", "eicu"):
        cohort = pd.read_csv(PROCESSED / dataset / "cohort.csv")
        external_labels = cohort["outcome"].astype(int).to_numpy()
        path = RESULTS / "predictions" / f"{dataset}_external{suffix}.npz"
        with np.load(path, allow_pickle=True) as archive:
            for model in OUTLINE_MODEL_ORDER:
                _compare_metric_payload(
                    point_metrics(
                        external_labels,
                        archive[model].astype(np.float64),
                    ),
                    metrics["external"][dataset][model].get("point"),
                    context=f"external.{dataset}.{model}.point",
                )
        external_verified[dataset] = len(OUTLINE_MODEL_ORDER)
    return {
        "development_models": verified_models,
        "development_probability_ensemble_models": len(averaged),
        "external_model_counts": external_verified,
    }


def _primary_bootstrap_validity_checks(metrics: dict, replicates: int) -> dict:
    """Require all requested primary inference replicates to be estimable."""
    primary_metrics = (
        "macro_auc",
        "auc_any_deterioration",
        "auc_severe_deterioration",
    )
    blocks = {
        "development": metrics["development"],
        "mimic3": metrics["external"]["mimic3"],
        "eicu": metrics["external"]["eicu"],
    }
    summary: dict[str, object] = {}
    for dataset, block in blocks.items():
        checked_models = 0
        for model in REQUIRED_MODELS if dataset == "development" else OUTLINE_MODEL_ORDER:
            model_block = block.get(model, {})
            validity = model_block.get("ci_validity", {})
            counts = validity.get("metric_valid_replicates", {})
            _require(
                validity.get("requested_replicates") == replicates,
                f"{dataset}.{model} records a different bootstrap replicate count",
            )
            for metric in primary_metrics:
                _require(
                    int(counts.get(metric, 0)) == replicates,
                    f"{dataset}.{model}.{metric} has "
                    f"{counts.get(metric, 0)} valid bootstrap replicates, "
                    f"expected {replicates}",
                )
            checked_models += 1
        paired = block.get(
            "paired_differences_bootstrap_validity_vs_dggru_tr", {}
        )
        _require(
            paired,
            f"{dataset} lacks paired-bootstrap validity records",
        )
        for comparator, validity in paired.items():
            _require(
                int(validity.get("requested_replicates", 0)) == replicates,
                f"{dataset}.{comparator} records a different "
                "paired-bootstrap replicate count",
            )
            _require(
                int(validity.get("valid_replicates", 0)) == replicates,
                f"{dataset}.{comparator} has "
                f"{validity.get('valid_replicates', 0)} valid paired-bootstrap "
                f"replicates, expected {replicates}",
            )
        summary[dataset] = {
            "models": checked_models,
            "paired_comparators": len(paired),
            "requested_replicates": replicates,
        }
    return summary


def _require_post_landmark_gcs_eligibility(
    dataset: str,
    minimum_post: np.ndarray,
    death_in_icu: np.ndarray,
) -> None:
    """Require the outline's two-hour follow-up GCS rule for every stay."""
    minimum_post = np.asarray(minimum_post)
    death_in_icu = np.asarray(death_in_icu, dtype=bool)
    _require(
        minimum_post.shape == death_in_icu.shape,
        f"{dataset} sparse-GCS and death arrays differ in shape",
    )
    sparse_post = minimum_post < 2
    _require(
        not sparse_post.any(),
        f"{dataset} contains a stay with fewer than two distinct "
        "post-landmark GCS hours",
    )


def _cohort_checks(dataset: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = PROCESSED / dataset
    cohort_path = root / "cohort.csv"
    prefilter_path = root / "cohort_pre_filter.csv"
    gcs_path = root / "gcs_events.csv"
    sequence_path = root / "sequences.npz"
    _require(cohort_path.is_file(), f"Missing cohort: {cohort_path}")
    _require(prefilter_path.is_file(), f"Missing prefilter cohort: {prefilter_path}")
    _require(gcs_path.is_file(), f"Missing GCS event file: {gcs_path}")
    _require(sequence_path.is_file(), f"Missing sequences: {sequence_path}")
    cohort = pd.read_csv(cohort_path)
    prefilter = pd.read_csv(prefilter_path)
    gcs = pd.read_csv(gcs_path)
    _require(
        not cohort["stay_id"].duplicated().any(),
        f"{dataset} cohort contains duplicate stay_id values",
    )
    _require(
        cohort["outcome"].isin([0, 1, 2]).all(),
        f"{dataset} cohort has invalid outcome values",
    )
    required_eligibility = {
        "in_icu_at_primary_landmark",
        "alive_at_primary_landmark",
        "death_in_icu_within_72h",
        "early_live_icu_transfer_before_72h",
        "primary_included",
    }
    _require(
        required_eligibility.issubset(prefilter.columns),
        f"{dataset} prefilter lacks strict landmark eligibility columns",
    )
    included_prefilter = prefilter[
        prefilter["stay_id"].isin(cohort["stay_id"])
    ]
    _require(
        len(included_prefilter) == len(cohort),
        f"{dataset} analytic cohort is not a subset of prefilter stays",
    )
    for column in (
        "in_icu_at_primary_landmark",
        "alive_at_primary_landmark",
        "primary_included",
    ):
        _require(
            included_prefilter[column].fillna(False).astype(bool).all(),
            f"{dataset} contains a stay failing {column}",
        )
    _require(
        pd.to_numeric(
            cohort["icu_out_hour"], errors="coerce"
        ).ge(24).all(),
        f"{dataset} contains a stay outside the ICU at the 24 h landmark",
    )
    death = cohort["death_within_72h"].fillna(False).astype(bool)
    death_hour = pd.to_numeric(
        cohort["death_hour_from_icu"], errors="coerce"
    )
    icu_out_hour = pd.to_numeric(cohort["icu_out_hour"], errors="coerce")
    death_in_icu = death_in_icu_within_window(
        death,
        death_hour,
        icu_out_hour,
    )
    included_by_stay = included_prefilter.set_index(
        included_prefilter["stay_id"].astype(np.int64)
    ).loc[cohort["stay_id"].astype(np.int64)]
    stored_death_in_icu = (
        included_by_stay["death_in_icu_within_72h"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    _require(
        np.array_equal(stored_death_in_icu, death_in_icu.to_numpy()),
        f"{dataset} stored death_in_icu_within_72h differs from derived death timing",
    )
    _require(
        (~death | death_hour.gt(24)).all(),
        f"{dataset} contains a death at or before the 24 h landmark",
    )
    post = gcs[["stay_id", "time_hour", "gcs"]].copy()
    post["stay_id"] = pd.to_numeric(post["stay_id"], errors="coerce")
    post["time_hour"] = pd.to_numeric(post["time_hour"], errors="coerce")
    post["gcs"] = pd.to_numeric(post["gcs"], errors="coerce")
    post = post[
        post["stay_id"].notna()
        & post["time_hour"].ge(24)
        & post["time_hour"].lt(72)
        & post["gcs"].between(3, 15)
    ].copy()
    post["hour_bin"] = np.floor(post["time_hour"])
    post_counts = post.groupby("stay_id")["hour_bin"].nunique()
    minimum_post = (
        cohort["stay_id"].astype(np.int64).map(post_counts).fillna(0)
    )
    _require_post_landmark_gcs_eligibility(
        dataset,
        minimum_post.to_numpy(),
        death_in_icu.to_numpy(),
    )
    with np.load(sequence_path, allow_pickle=True) as archive:
        values = archive["values"]
        masks = archive["masks"]
        stay_ids = archive["stay_id"].astype(np.int64)
        channels = archive["channels"].astype(str).tolist()
    _require(
        values.shape == (len(cohort), 24, len(CHANNELS)),
        f"{dataset} sequence shape is {values.shape}",
    )
    _require(values.shape == masks.shape, f"{dataset} value/mask shapes differ")
    _require(
        channels == list(CHANNELS),
        f"{dataset} sequence channels differ from the study contract",
    )
    _require(
        np.array_equal(stay_ids, cohort["stay_id"].to_numpy(dtype=np.int64)),
        f"{dataset} sequence IDs differ from cohort order",
    )
    _require(
        np.isin(masks, [0.0, 1.0]).all(),
        f"{dataset} masks contain values other than 0/1",
    )
    finite_values = values[np.isfinite(values)]
    _require(
        np.isfinite(finite_values).all(),
        f"{dataset} sequences contain invalid numeric values",
    )
    return (
        cohort,
        cohort["stay_id"].to_numpy(dtype=np.int64),
        cohort["outcome"].to_numpy(dtype=int),
    )


def _fold_checks(
    path: Path,
    cohort: pd.DataFrame,
    *,
    repeats: int,
    folds: int,
    seed: int,
) -> dict[str, object]:
    _require(path.is_file(), f"Missing fold artifact: {path}")
    frame = pd.read_csv(path)
    required = {"stay_id", "subject_key", "outcome", "repeat", "fold"}
    _require(
        required.issubset(frame.columns),
        f"{path} lacks columns {sorted(required.difference(frame.columns))}",
    )
    _require(
        set(frame["repeat"].astype(int)) == set(range(repeats)),
        f"{path} repeat labels differ from 0..{repeats - 1}",
    )
    _require(
        len(frame) == len(cohort) * repeats,
        f"{path} has {len(frame)} rows, expected {len(cohort) * repeats}",
    )
    expected = cohort[["stay_id", "outcome"]].sort_values("stay_id").reset_index(
        drop=True
    )
    patient_to_group = dict(
        zip(
            cohort["stay_id"].astype(np.int64),
            cohort["subject_key"].astype(str),
        )
    )
    for repeat in range(repeats):
        block = frame[frame["repeat"].astype(int).eq(repeat)]
        observed = block[["stay_id", "outcome"]].sort_values("stay_id").reset_index(
            drop=True
        )
        _require(
            observed["stay_id"].equals(expected["stay_id"]),
            f"{path} repeat {repeat} is not a complete patient partition",
        )
        _require(
            observed["outcome"].equals(expected["outcome"]),
            f"{path} repeat {repeat} outcome labels differ from the cohort",
        )
        observed_groups = block["subject_key"].astype(str)
        expected_groups = block["stay_id"].astype(np.int64).map(patient_to_group)
        _require(
            observed_groups.equals(expected_groups),
            f"{path} repeat {repeat} subject_key mapping differs from cohort",
        )
        _require(
            set(block["fold"].astype(int)) == set(range(folds)),
            f"{path} repeat {repeat} fold labels differ from 0..{folds - 1}",
        )
        for fold in range(folds):
            fold_rows = block[block["fold"].astype(int).eq(fold)]
            _require(len(fold_rows) > 0, f"{path} repeat {repeat} fold {fold} empty")
            for other in range(fold + 1, folds):
                other_groups = set(
                    block.loc[
                        block["fold"].astype(int).eq(other), "subject_key"
                    ].astype(str)
                )
                overlap = set(fold_rows["subject_key"].astype(str)).intersection(
                    other_groups
                )
                _require(
                    not overlap,
                    f"{path} repeat {repeat} folds {fold}/{other} share patients",
                )
    expected_splits, expected_meta = build_repeated_splits(
        cohort["outcome"].astype(int).to_numpy(),
        cohort["subject_key"].astype(str).to_numpy(),
        folds=folds,
        repeats=repeats,
        seed=seed,
    )
    _require(
        len(expected_splits) == repeats * folds
        and expected_meta
        == [
            (repeat, fold)
            for repeat in range(repeats)
            for fold in range(folds)
        ],
        "Frozen repeated-CV partition contract is internally inconsistent",
    )
    for (repeat, fold), (_, test_index) in zip(expected_meta, expected_splits):
        expected_ids = np.sort(
            cohort["stay_id"].astype(np.int64).to_numpy()[test_index]
        )
        observed_ids = np.sort(
            frame.loc[
                frame["repeat"].astype(int).eq(repeat)
                & frame["fold"].astype(int).eq(fold),
                "stay_id",
            ]
            .astype(np.int64)
            .to_numpy()
        )
        _require(
            np.array_equal(observed_ids, expected_ids),
            f"{path} repeat {repeat} fold {fold} differs from the "
            "seed-derived patient-grouped partition",
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
    }


def _require_formal_design(
    *,
    folds: int,
    repeats: int,
    replicates: int,
) -> None:
    _require(folds == 5, "Formal analysis requires exactly five outer folds")
    _require(repeats == 5, "Formal analysis requires exactly five CV repeats")
    _require(
        replicates == 2000,
        "Formal analysis requires 2000 bootstrap replicates",
    )


def _manifest_checks(
    manifest: dict,
    *,
    repeats: int,
    folds: int,
) -> dict[str, object]:
    _require(
        tuple(manifest.get("models", [])) == REQUIRED_MODELS,
        "Manifest model list/order differs from the outline-plus-ablation contract",
    )
    _require(
        tuple(manifest.get("outline_models", [])) == OUTLINE_MODEL_ORDER,
        "Manifest outline model list differs from the frozen outline contract",
    )
    _require(
        tuple(manifest.get("secondary_models", [])) == SECONDARY_MODEL_ORDER,
        "Manifest secondary model list differs from the matched-ablation contract",
    )
    _require_formal_design(
        folds=folds,
        repeats=repeats,
        replicates=int(manifest.get("bootstrap_replicates", -1)),
    )
    _require(int(manifest.get("folds", -1)) == folds, "Manifest fold count mismatch")
    _require(
        int(manifest.get("cv_repeats", -1)) == repeats,
        "Manifest repeat count mismatch",
    )
    model_repeats = {
        str(name): int(value)
        for name, value in manifest.get("model_cv_repeats", {}).items()
    }
    _require(
        set(REQUIRED_MODELS).issubset(model_repeats),
        "Manifest lacks a per-model CV repeat count",
    )
    _require(
        all(value in {1, 5} for value in model_repeats.values()),
        "Per-model CV repeat counts must be 1 or 5",
    )
    primary_five = (
        "xgb_static",
        "xgb_summary",
        "gcs_slope",
        "gru",
        "grud",
        "transformer",
        "dggru",
        "dggru_tr",
        "nested_stack",
    )
    single_partition = (
        "dggru_tr_treverse",
        "dggru_tr_tpermute",
        "dggru_laplace",
        *SECONDARY_MODEL_ORDER,
    )
    _require(
        all(model_repeats[name] == repeats for name in primary_five),
        "Primary models must retain the full CV repeat count",
    )
    _require(
        all(model_repeats[name] == 1 for name in single_partition),
        "Negative controls and secondary ablations must use one CV repeat",
    )
    _require(
        int(manifest.get("stacking_inner_oof_seeds", 0)) == 1,
        "Nested stacking must record a single inner-OOF seed",
    )
    _require(
        list(manifest.get("model_seeds", [])) == [0, 1, 2, 3, 4],
        "Manifest does not contain exactly seeds 0-4",
    )
    _require(
        manifest.get("source_sha256_at_end") == manifest.get("source_sha256"),
        "Start and end model-fitting source hashes differ",
    )
    laplace = manifest.get("laplace_control", {})
    _require(
        laplace.get("implementation")
        == "generic_laplace_noise_on_observed_standardised_inputs"
        and np.isclose(float(laplace.get("scale", np.nan)), 0.01),
        "Manifest does not document the prespecified generic Laplace control",
    )
    features = [str(value) for value in manifest.get("features", [])]
    temporal_features = [
        str(value) for value in manifest.get("temporal_static_features", [])
    ]
    _require(len(features) == 210, f"Expected 210 development features, got {len(features)}")
    expected_temporal = [
        column for column in CORE_STATIC if column in features
    ]
    _require(
        temporal_features == expected_temporal,
        "The primary static matrix is not the declared core set",
    )
    _require(
        int(manifest.get("temporal_static_feature_count", -1))
        == len(expected_temporal),
        "Temporal static feature count is inconsistent",
    )
    leaked = sorted(POST_LANDMARK_COLUMNS.intersection(features))
    _require(not leaked, "Post-landmark columns entered predictors: " + ", ".join(leaked))
    _require("n_gcs_pre" not in features, "n_gcs_pre entered the predictor matrix")
    _require(
        "neurosurgery_admission" not in features,
        "The untimed neurosurgery_admission flag entered the predictor matrix",
    )
    contracts = manifest.get("model_feature_contract", {})
    _require(
        isinstance(contracts, dict)
        and set(REQUIRED_MODELS).issubset(contracts),
        "Manifest lacks per-model feature contracts",
    )
    primary_contract = contracts.get("dggru_tr", {})
    noseq_contract = contracts.get("dggru_tr_noseq", {})
    static_contract = contracts.get("dggru_tr_static", {})
    trajectory_contract = contracts.get("dggru_tr_traj", {})
    _require(
        primary_contract.get("sequence_enabled") is True,
        "Primary DG-GRU + TR must use the observed sequence",
    )
    _require(
        noseq_contract.get("sequence_enabled") is False
        and noseq_contract.get("sequence_mode") == "none"
        and noseq_contract.get("static_feature_count")
        == len(expected_temporal),
        "The matched no-sequence ablation does not use the core static set",
    )
    _require(
        static_contract.get("sequence_enabled") is True
        and static_contract.get("static_feature_count") == 125,
        "The sequence-plus-non-summary arm does not contain 125 static features",
    )
    _require(
        trajectory_contract.get("sequence_enabled") is True
        and trajectory_contract.get("static_feature_count") == 1,
        "The trajectory-only arm does not use one placeholder static feature",
    )
    search = manifest.get("hyperparameter_search", {})
    for family in ("xgboost", "sequence_models"):
        _require(
            len(search.get(family, {}).get("candidates", [])) >= 2,
            f"Manifest lacks a non-degenerate {family} search space",
        )
    selected_hyperparameters = manifest.get("cv_selected_hyperparameters", {})
    for model in ("xgb_static", "xgb_summary", "dggru_tr"):
        expected_hp_records = folds * model_repeats[model]
        records = selected_hyperparameters.get(model, [])
        _require(
            len(records) == expected_hp_records,
            f"{model} has {len(records)} hyperparameter records, "
            f"expected {expected_hp_records}",
        )
        _require(
            all(int(record.get("candidate_count", 0)) >= 2 for record in records),
            f"{model} includes a fold without a genuine candidate comparison",
        )
    inference = manifest.get("statistical_inference", {})
    _require(
        inference.get("bootstrap_estimability")
        == "rejection_sample_clusters_until_all_three_ordinal_grades_are_present"
        and int(inference.get("requested_bootstrap_replicates", -1)) == 2000,
        "Manifest does not record the primary bootstrap estimability contract",
    )
    bookkeeping = {}
    for model in TEMPORAL_MODELS:
        model_repeat_count = int(model_repeats[model])
        expected_fits = folds * model_repeat_count * 5
        epochs = manifest.get("cv_best_epochs", {}).get(model, {}).get("values", [])
        lambdas = (
            manifest.get("cv_selected_lambdas", {})
            .get(model, {})
            .get("values", [])
        )
        selection_records = (
            manifest.get("cv_selected_lambdas", {})
            .get(model, {})
            .get("independent_selection_records")
        )
        independent_values = (
            manifest.get("cv_selected_lambdas", {})
            .get(model, {})
            .get("independent_values", [])
        )
        _require(
            len(epochs) == expected_fits,
            f"{model} has {len(epochs)} epoch records, expected {expected_fits}",
        )
        _require(
            len(lambdas) == expected_fits,
            f"{model} has {len(lambdas)} lambda records, expected {expected_fits}",
        )
        _require(
            all(int(value) >= 1 for value in epochs),
            f"{model} contains an invalid selected epoch",
        )
        _require(
            all(np.isfinite(float(value)) for value in lambdas),
            f"{model} contains an invalid selected lambda",
        )
        lambda_policy = str(
            manifest.get("lambda_policy", {}).get("dggru_tr", "")
        )
        expected_selections = (
            0
            if lambda_policy == "frozen_from_lambda_grid"
            else folds * model_repeat_count
        ) if model == "dggru_tr" else 0
        _require(
            selection_records == expected_selections,
            f"{model} has {selection_records} independent lambda selections, "
            f"expected {expected_selections}",
        )
        _require(
            len(independent_values) == expected_selections,
            f"{model} has {len(independent_values)} independent lambda values, "
            f"expected {expected_selections}",
        )
        bookkeeping[model] = {
            "fit_records": len(epochs),
            "independent_lambda_selections": selection_records,
            "epoch_min": int(min(epochs)),
            "epoch_max": int(max(epochs)),
            "lambda_min": float(min(lambdas)),
            "lambda_max": float(max(lambdas)),
        }
    return {
        "features": len(features),
        "models": list(REQUIRED_MODELS),
        "temporal_bookkeeping": bookkeeping,
    }


def validate(tag: str) -> dict:
    suffix = f"_{tag}" if tag else ""
    metrics_path = RESULTS / f"metrics{suffix}.json"
    manifest_path = RESULTS / f"run_manifest{suffix}.json"
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    folds = int(manifest["folds"])
    repeats = int(manifest["cv_repeats"])

    manifest_summary = _manifest_checks(
        manifest,
        repeats=repeats,
        folds=folds,
    )
    development_cohort, development_ids, development_labels = _cohort_checks(
        "mimic4"
    )
    cohort_artifacts = {}
    for dataset in ("mimic4", "mimic3", "eicu"):
        cohort, ids, labels = _cohort_checks(dataset)
        cohort_artifacts[dataset] = {
            "n": int(len(cohort)),
            "outcomes": {
                str(grade): int((labels == grade).sum()) for grade in (0, 1, 2)
            },
            "sequence_sha256": sha256_file(PROCESSED / dataset / "sequences.npz"),
        }

    _require(
        int(manifest.get("development_n", -1)) == len(development_cohort),
        "Manifest development_n differs from the cohort",
    )
    manifest_outcomes = {
        str(key): int(value)
        for key, value in manifest.get("development_outcomes", {}).items()
    }
    expected_outcomes = {
        str(grade): int((development_labels == grade).sum())
        for grade in (0, 1, 2)
    }
    _require(
        manifest_outcomes == expected_outcomes,
        "Manifest outcome counts differ from the cohort",
    )

    input_snapshot = manifest.get("input_snapshot", {})
    _require(
        int(input_snapshot.get("data_file_count", -1))
        == 3 * len(DATA_CONTRACT_FILES),
        "Manifest input snapshot does not cover the fixed data contract",
    )
    _require(
        input_snapshot.get("config_sha256") == manifest.get("config_sha256"),
        "Manifest start/end configuration snapshots disagree",
    )
    _require(
        input_snapshot.get("capture_phase")
        == "before_data_loading_or_model_fitting"
        and input_snapshot.get("verification_phase")
        == "after_all_model_fitting_before_metrics_write",
        "Manifest input snapshot phases are not fail-closed",
    )
    expected_data_keys = {
        f"{dataset}/{filename}"
        for dataset in ("mimic4", "mimic3", "eicu")
        for filename in DATA_CONTRACT_FILES
    }
    _require(
        set(manifest.get("data_sha256", {})) == expected_data_keys,
        "Manifest data hash set differs from the fixed training input contract",
    )
    for relative, recorded in manifest.get("data_sha256", {}).items():
        path = PROCESSED / Path(relative)
        _require(path.is_file(), f"Manifest references missing data file: {path}")
        _require(
            sha256_file(path) == recorded,
            f"Data hash mismatch: {path}",
        )
    source_root = PROJECT_ROOT / "src" / "ich_dynamic"
    runtime_source_files = set(training_runtime_sources())
    recorded_sources = {
        str(filename): str(digest)
        for filename, digest in manifest.get("source_sha256", {}).items()
    }
    missing_runtime_sources = sorted(
        runtime_source_files.difference(recorded_sources)
    )
    _require(
        not missing_runtime_sources,
        "Manifest omits model-fitting runtime sources: "
        + ", ".join(missing_runtime_sources),
    )
    nonruntime_source_mismatches = []
    for filename, recorded in recorded_sources.items():
        path = source_root / filename
        _require(path.is_file(), f"Manifest references missing source: {path}")
        matches = sha256_file(path) == recorded
        if filename in runtime_source_files:
            _require(matches, f"Model-fitting source hash mismatch: {path}")
        elif not matches:
            nonruntime_source_mismatches.append(filename)

    development_metrics = metrics.get("development")
    external_metrics = metrics.get("external")
    _require(isinstance(development_metrics, dict), "Metrics lack development block")
    _require(isinstance(external_metrics, dict), "Metrics lack external block")
    _require(
        set(development_metrics).intersection(REQUIRED_MODELS) == set(REQUIRED_MODELS),
        "Development metrics do not contain the exact outline model set",
    )
    for dataset in ("mimic3", "eicu"):
        block = external_metrics.get(dataset)
        _require(isinstance(block, dict), f"Metrics lack external dataset {dataset}")
        _require(
            set(block).intersection(OUTLINE_MODEL_ORDER)
            == set(OUTLINE_MODEL_ORDER),
            f"External metrics for {dataset} lack outline models",
        )

    development_predictions = _probability_checks(
        RESULTS / "predictions" / f"mimic4_cv{suffix}.npz",
        REQUIRED_MODELS,
        expected_ids=development_ids,
        expected_labels=development_labels,
    )
    repeated_predictions = _repeated_probability_checks(
        RESULTS / "predictions" / f"mimic4_cv{suffix}_repeats.npz",
        REQUIRED_MODELS,
        model_repeats={
            str(name): int(value)
            for name, value in manifest["model_cv_repeats"].items()
        },
        expected_ids=development_ids,
        expected_labels=development_labels,
    )
    fold_artifact = _fold_checks(
        RESULTS / f"mimic4_folds{suffix}.csv",
        development_cohort,
        repeats=repeats,
        folds=folds,
        seed=int(manifest["seed"]),
    )
    recorded_fold = manifest.get("fold_artifact", {})
    _require(
        Path(str(recorded_fold.get("path", ""))).resolve()
        == (RESULTS / f"mimic4_folds{suffix}.csv").resolve(),
        "Manifest fold artifact path differs from the expected tagged path",
    )
    _require(
        recorded_fold.get("sha256") == fold_artifact["sha256"],
        "Manifest fold artifact hash differs from the current fold file",
    )
    external_predictions = {}
    for dataset in ("mimic3", "eicu"):
        cohort, ids, labels = _cohort_checks(dataset)
        external_predictions[dataset] = _probability_checks(
            RESULTS / "predictions" / f"{dataset}_external{suffix}.npz",
            OUTLINE_MODEL_ORDER,
            expected_ids=ids,
            expected_labels=labels,
        )

    recorded_results = manifest.get("result_sha256", {})
    result_paths = {
        "metrics": metrics_path,
        "development_repeated_predictions": RESULTS
        / "predictions"
        / f"mimic4_cv{suffix}_repeats.npz",
        "development_probability_ensemble": RESULTS
        / "predictions"
        / f"mimic4_cv{suffix}.npz",
        "mimic3_external_predictions": RESULTS
        / "predictions"
        / f"mimic3_external{suffix}.npz",
        "eicu_external_predictions": RESULTS
        / "predictions"
        / f"eicu_external{suffix}.npz",
    }
    for key, path in result_paths.items():
        _require(key in recorded_results, f"Manifest lacks result hash {key}")
        _require(
            sha256_file(path) == recorded_results[key],
            f"Result hash mismatch for {key}: {path}",
        )
    metric_recomputation = _metric_recompute_checks(
        metrics,
        development_cohort,
        suffix=suffix,
        model_repeats={
            str(name): int(value)
            for name, value in manifest["model_cv_repeats"].items()
        },
    )
    bootstrap_validity = _primary_bootstrap_validity_checks(
        metrics,
        replicates=int(manifest["bootstrap_replicates"]),
    )

    return {
        "status": "passed",
        "tag": tag,
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "manifest_checks": manifest_summary,
        "nonruntime_source_mismatches": nonruntime_source_mismatches,
        "cohorts": cohort_artifacts,
        "development_predictions": development_predictions,
        "repeated_predictions": repeated_predictions,
        "folds": fold_artifact,
        "external_predictions": external_predictions,
        "metric_recomputation": metric_recomputation,
        "primary_bootstrap_validity": bootstrap_validity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    output = (
        Path(args.output)
        if args.output
        else RESULTS / "audits" / f"formal_run_validation_{args.tag}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(args.tag)
    except Exception as exc:
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "tag": args.tag,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    output.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS {args.tag}")
    print(output)


if __name__ == "__main__":
    main()
