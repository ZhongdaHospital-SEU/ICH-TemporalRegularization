"""Patient-grouped development CV and direct external validation."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    CHANNELS,
    CONFIG_PATH,
    PROCESSED,
    PROJECT_ROOT,
    RESULTS,
    ensure_dirs,
    validate_study_config,
)
from .models import (
    N_CLASSES,
    SequencePreprocessor,
    fit_full_temporal_model,
    predict_temporal_model,
    train_temporal_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(
    int(
        os.environ.get(
            "ICH_TORCH_THREADS",
            str(max(1, min(6, os.cpu_count() or 1))),
        )
    )
)

POST_LANDMARK_COLUMNS = {
    "outcome",
    "outcome_recorded_intubation_composite",
    "outcome_gcs_death_only",
    "outcome_primary",
    "outcome_reconstructed",
    "death_within_72h",
    "death_in_icu_within_72h",
    "death_hour_from_icu",
    "death_after_landmark",
    "recorded_intubation_24_72h",
    "verified_unplanned_intubation_24_72h",
    "endpoint_intubation",
    "intubation_component_included",
    "intubation_composite_available",
    "primary_outcome_includes_intubation",
    "p020_status",
    "p020_complete",
    "recorded_intubation_composite_role",
    "recorded_intubation_composite_status",
    "recorded_intubation_composite_note",
    "max_sustained_drop",
    "last_gcs_hour",
    "icu_out_hour",
    "unitdischargeoffset",
    "unitdischargestatus",
    "hospitaldischargestatus",
    "hospitaldischargeoffset",
    "hospitaladmitoffset",
    "hospital_expire_flag",
    "primary_followup_complete",
    "complete_followup_only",
    "primary_included",
    "landmark_alive",
    "followup_window_complete",
    "followup_complete",
    "alive_at_primary_landmark",
    "in_icu_at_primary_landmark",
    "eligible_at_primary_landmark",
    "early_icu_transfer",
    "early_live_icu_transfer_before_72h",
    "analysis_weight",
    "analysis_weight_raw",
    "censoring_probability",
}

POST_LANDMARK_PREFIXES = (
    "outcome",
    "death_",
    "followup_",
    "p020_",
    "primary_followup_",
    "landmark_",
)


def predictor_column_allowed(column: str) -> bool:
    """Return False for outcomes, landmark state, and post-landmark bookkeeping."""
    return column not in POST_LANDMARK_COLUMNS and not column.startswith(
        POST_LANDMARK_PREFIXES
    )

EXCLUDED_COLUMNS = POST_LANDMARK_COLUMNS | {
    "stay_id",
    "hadm_id",
    "subject_id",
    "intime",
    "outtime",
    "deathtime",
    "first_careunit",
    "gender",
    "race",
    "admission_type",
    "has_ich",
    "subject_key",
    "hospital_key",
    "dataset",
    "age_raw",
    "n_gcs_pre",
    "patient_sequence",
    "hospitalid",
    "unitvisitnumber",
    # This procedure flag has no event timestamp and therefore cannot be
    # established as a pre-landmark predictor.
    "neurosurgery_admission",
    "sensitivity_analysis_id",
    "sensitivity_scenario",
    "censoring_strategy",
    "baseline_definition",
    "landmark_hours",
    "intubation_in_endpoint",
    "endpoint_label",
}

STATIC_FEATURES = [
    "age",
    "age_gt89",
    "sex_male",
    "race_white",
    "emergency_admission",
    "baseline_gcs_max",
    "min_rass_24h",
    "any_rass_le_minus2_24h",
    "any_deep_sedation_24h",
    "deep_sedation_observed_24h",
]

CORE_STATIC = [
    "age",
    "age_gt89",
    "sex_male",
    "race_white",
    "baseline_gcs_max",
    "baseline_gcs_last",
    "gcs_total_mean",
    "gcs_total_min",
    "gcs_total_max",
    "gcs_total_sd",
    "gcs_total_slope",
    "gcs_eye_mean",
    "gcs_verbal_mean",
    "gcs_motor_mean",
    "sbp_min",
    "sbp_mean",
    "spo2_min",
    "hr_mean",
    "rr_mean",
    "urine_output_mean",
    "fio2_mean",
    "rass_mean",
    "monitoring_intensity_24h",
]
DEEP_STATIC = CORE_STATIC

MODEL_META = {
    "xgb_static": {
        "kind": "xgboost",
        "features": "static",
        "static_mode": "static",
        "sequence_mode": "none",
    },
    "xgb_summary": {
        "kind": "xgboost",
        "features": "core",
        "static_mode": "core",
        "sequence_mode": "none",
    },
    "gcs_slope": {"kind": "slope_logistic", "features": "gcs_slope"},
    "gru": {
        "kind": "temporal",
        "architecture": "gru",
        "graph": False,
        "regularized": False,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "grud": {
        "kind": "temporal",
        "architecture": "grud",
        "graph": False,
        "regularized": False,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "transformer": {
        "kind": "temporal",
        "architecture": "transformer",
        "graph": False,
        "regularized": False,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": False,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru_tr": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "nested_stack": {
        "kind": "stacking",
        "members": ("xgb_summary", "gru", "dggru_tr"),
    },
    "dggru_tr_treverse": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "time_mode": "reverse",
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru_tr_tpermute": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "time_mode": "permute",
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru_laplace": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": False,
        "laplace_only": True,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru_tr_gcsmask": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "static_mode": "core",
        "sequence_mode": "observed",
        "mask_mode": "gcs",
    },
    "dggru_tr_noseq": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "static_mode": "core",
        "sequence_mode": "none",
        "mask_mode": "none",
    },
    "dggru_tr_static": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "static_mode": "static",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
    "dggru_tr_traj": {
        "kind": "temporal",
        "architecture": "dggru",
        "graph": True,
        "regularized": True,
        "static_mode": "placeholder",
        "sequence_mode": "observed",
        "mask_mode": "any_channel",
    },
}

MODEL_SEEDS = (0, 1, 2, 3, 4)
STACKING_MEMBERS = ("xgb_summary", "gru", "dggru_tr")
STACKING_INNER_FOLDS = 5
STACKING_C = 0.1
TIME_PERMUTATION_SEED = 20261024
LAPLACE_INPUT_SCALE = 0.01
DATA_CONTRACT_FILES = (
    "cohort_pre_filter.csv",
    "cohort.csv",
    "cohort_flow.csv",
    "hourly.csv",
    "gcs_events.csv",
    "rass_events.csv",
    "sequences.npz",
)
LANDMARK_SOURCE_FILES = {
    "cohort_pre_filter": "cohort_pre_filter.csv",
    "gcs_events": "gcs_events.csv",
    "hourly": "hourly.csv",
}
OUTLINE_MODEL_ORDER = (
    "xgb_static",
    "xgb_summary",
    "gcs_slope",
    "gru",
    "grud",
    "transformer",
    "dggru",
    "dggru_tr",
    "nested_stack",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)
OUTLINE_PRIMARY_BASELINES = (
    "xgb_static",
    "xgb_summary",
    "gru",
    "grud",
    "transformer",
)
OUTLINE_HOLM_COMPARATORS = OUTLINE_PRIMARY_BASELINES
OUTLINE_HOLM_ENDPOINTS = ("any", "severe")
SECONDARY_MODEL_ORDER = (
    "dggru_tr_gcsmask",
    "dggru_tr_noseq",
    "dggru_tr_static",
    "dggru_tr_traj",
)
ALL_MODEL_ORDER = OUTLINE_MODEL_ORDER + SECONDARY_MODEL_ORDER
if tuple(MODEL_META) != ALL_MODEL_ORDER:
    raise RuntimeError("MODEL_META order differs from the declared model contract")

XGBOOST_DEFAULTS = {
    "n_estimators": 260,
    "max_depth": 3,
    "learning_rate": 0.035,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
}
SEQUENCE_DEFAULTS = {
    "hidden_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 25,
    "patience": 6,
}


@dataclass
class Dataset:
    name: str
    cohort: pd.DataFrame
    sequences: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    static_all: pd.DataFrame
    static_deep: pd.DataFrame
    analysis_weight: np.ndarray | None = None


class LandmarkLineageError(ValueError):
    """Raised when model inputs cannot be tied to a current landmark audit."""


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LandmarkLineageError(f"Required lineage file is missing: {path}") from exc


def _parse_landmark_directory(path: Path) -> int | None:
    prefix = "landmark_"
    suffix = "h"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        return None
    value = path.name[len(prefix) : -len(suffix)]
    if not value.isdigit():
        return None
    return int(value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise LandmarkLineageError(
                f"Landmark audit repeats JSON key {key!r}"
            )
        result[key] = value
    return result


def _resolve_dataset_location(name: str) -> tuple[str, Path, Path, int]:
    """Resolve a canonical or landmark directory to its independent audit."""
    candidate = Path(str(name)).expanduser()
    if candidate.is_dir():
        artifact_root = candidate.resolve()
        if (artifact_root / "audit.json").is_file():
            dataset = artifact_root.parent.name
            landmark_hours = _parse_landmark_directory(artifact_root)
            if landmark_hours is None:
                raise LandmarkLineageError(
                    f"Cannot infer landmark_hours from {artifact_root}"
                )
            expected_root = (
                PROCESSED
                / "landmark_inputs"
                / dataset
                / f"landmark_{landmark_hours}h"
            )
            if artifact_root != expected_root.resolve():
                raise LandmarkLineageError(
                    f"Landmark data are outside the audited root: {artifact_root}"
                )
            audit_path = artifact_root / "audit.json"
        else:
            dataset = artifact_root.name
            if dataset not in {"mimic4", "mimic3", "eicu"}:
                raise LandmarkLineageError(
                    f"Dataset directory has no landmark audit.json: {artifact_root}"
                )
            if artifact_root != (PROCESSED / dataset).resolve():
                raise LandmarkLineageError(
                    f"Main data are outside the processed root: {artifact_root}"
                )
            landmark_hours = 24
            audit_path = (
                PROCESSED
                / "landmark_inputs"
                / dataset
                / "landmark_24h"
                / "audit.json"
            )
    else:
        dataset = str(name)
        if dataset not in {"mimic4", "mimic3", "eicu"}:
            raise LandmarkLineageError(f"Unknown dataset location: {name}")
        artifact_root = PROCESSED / dataset
        landmark_hours = 24
        audit_path = (
            PROCESSED
            / "landmark_inputs"
            / dataset
            / "landmark_24h"
            / "audit.json"
        )
    return dataset, artifact_root, audit_path, landmark_hours


def _validate_landmark_lineage(
    *,
    dataset: str,
    landmark_hours: int,
    artifact_root: Path,
    audit_path: Path,
    cohort: pd.DataFrame,
    values: np.ndarray,
    masks: np.ndarray,
    sequence_ids: np.ndarray,
) -> None:
    """Recompute source lineage and validate audit/artifact identifier alignment."""
    if not audit_path.is_file():
        raise LandmarkLineageError(
            f"Landmark audit is required before model loading: {audit_path}"
        )
    try:
        audit = json.loads(
            audit_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LandmarkLineageError(
            f"Cannot read landmark audit: {audit_path}"
        ) from exc
    if not isinstance(audit, dict):
        raise LandmarkLineageError("Landmark audit must be a JSON object")
    if int(audit.get("schema_version", -1)) != 1:
        raise LandmarkLineageError("Landmark audit schema_version must be 1")
    if audit.get("dataset") != dataset:
        raise LandmarkLineageError(
            f"Landmark audit dataset {audit.get('dataset')!r} does not match {dataset!r}"
        )
    if int(audit.get("landmark_hours", -1)) != int(landmark_hours):
        raise LandmarkLineageError(
            "Landmark audit landmark_hours does not match the requested data"
        )

    source_files = audit.get("source_files")
    if not isinstance(source_files, dict):
        raise LandmarkLineageError("Landmark audit has no source_files object")
    recorded_names = set(source_files)
    expected_names = set(LANDMARK_SOURCE_FILES)
    if recorded_names != expected_names:
        missing = sorted(expected_names - recorded_names)
        unexpected = sorted(recorded_names - expected_names)
        raise LandmarkLineageError(
            "Landmark source_files contract mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    source_root = PROCESSED / dataset
    seen_paths: dict[str, str] = {}
    for source_name, filename in LANDMARK_SOURCE_FILES.items():
        record = source_files[source_name]
        if not isinstance(record, dict):
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} must be an object"
            )
        recorded_path = record.get("path")
        recorded_hash = record.get("sha256")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} has no path"
            )
        if Path(recorded_path).name != filename:
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} does not name {filename}"
            )
        if not Path(recorded_path).is_absolute():
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} path must be absolute"
            )
        current_path = _resolved(source_root / filename)
        if Path(recorded_path).resolve() != current_path:
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} points to another path"
            )
        normalized = os.path.normcase(str(current_path))
        if normalized in seen_paths:
            raise LandmarkLineageError(
                "Landmark audit repeats a source file: "
                f"{seen_paths[normalized]} and {source_name}"
            )
        seen_paths[normalized] = source_name
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise LandmarkLineageError(
                f"Landmark source entry {source_name!r} has an invalid SHA256"
            )
        if sha256_file(current_path) != recorded_hash.lower():
            raise LandmarkLineageError(
                f"Landmark source hash drifted: {current_path}"
            )

    if cohort.empty:
        raise LandmarkLineageError("Loaded landmark cohort is empty")
    if "stay_id" not in cohort or "outcome" not in cohort:
        raise LandmarkLineageError("Loaded landmark cohort lacks stay_id or outcome")
    cohort_ids = pd.to_numeric(cohort["stay_id"], errors="coerce")
    if cohort_ids.isna().any() or cohort_ids.duplicated().any():
        raise LandmarkLineageError(
            "Loaded landmark cohort stay_id must be complete and unique"
        )
    cohort_ids = cohort_ids.to_numpy(dtype=np.int64)
    outcomes = pd.to_numeric(cohort["outcome"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not set(np.unique(outcomes)).issubset({0, 1, 2}):
        raise LandmarkLineageError(
            "Loaded landmark outcomes must be ordinal values 0, 1, or 2"
        )
    if not np.array_equal(sequence_ids.astype(np.int64), cohort_ids):
        raise LandmarkLineageError(
            "Landmark sequence stay_id order differs from cohort.csv"
        )

    expected_shape = (len(cohort), int(landmark_hours), len(CHANNELS))
    if values.shape != expected_shape or masks.shape != expected_shape:
        raise LandmarkLineageError(
            f"Landmark tensors have shapes {values.shape}/{masks.shape}; "
            f"expected {expected_shape}"
        )
    sequence_audit = audit.get("sequence")
    if not isinstance(sequence_audit, dict):
        raise LandmarkLineageError("Landmark audit has no sequence object")
    if sequence_audit.get("shape") != list(expected_shape):
        raise LandmarkLineageError(
            "Landmark audit sequence shape differs from the loaded tensors"
        )
    if sequence_audit.get("channels") != list(CHANNELS):
        raise LandmarkLineageError(
            "Landmark audit channels differ from the study contract"
        )
    eligibility = audit.get("eligibility")
    if not isinstance(eligibility, dict):
        raise LandmarkLineageError("Landmark audit has no eligibility object")
    if int(eligibility.get("analysis_n", -1)) != len(cohort):
        raise LandmarkLineageError(
            "Landmark audit analysis_n differs from the loaded cohort"
        )

    if artifact_root.resolve() == audit_path.parent.resolve():
        artifact_hashes = audit.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            raise LandmarkLineageError(
                "Landmark audit has no artifact_hashes object"
            )
        if artifact_hashes.get("hash_algorithm") != "sha256":
            raise LandmarkLineageError(
                "Landmark artifact hashes must use sha256"
            )
        for filename in ("cohort.csv", "sequences.npz"):
            artifact = _resolved(artifact_root / filename)
            if artifact_hashes.get(filename) != sha256_file(artifact):
                raise LandmarkLineageError(
                    f"Landmark artifact hash drifted: {artifact}"
                )


def load_dataset(name: str, feature_columns: list[str] | None = None) -> Dataset:
    dataset_name, root, audit_path, landmark_hours = _resolve_dataset_location(
        name
    )
    cohort = pd.read_csv(root / "cohort.csv")
    with np.load(root / "sequences.npz", allow_pickle=True) as npz:
        sequences = npz["values"].astype(np.float32, copy=False)
        masks = npz["masks"].astype(np.float32, copy=False)
        ids = npz["stay_id"].astype(np.int64, copy=False)
    _validate_landmark_lineage(
        dataset=dataset_name,
        landmark_hours=landmark_hours,
        artifact_root=root,
        audit_path=audit_path,
        cohort=cohort,
        values=sequences,
        masks=masks,
        sequence_ids=ids,
    )
    numeric = cohort.select_dtypes(include=[np.number]).copy()
    candidates = [
        column
        for column in numeric.columns
        if column not in EXCLUDED_COLUMNS and predictor_column_allowed(column)
    ]
    if feature_columns is None:
        # Column availability is fixed by the source schema. Missingness and
        # variance are learned only inside the outer training fold by the model
        # preprocessors, never from held-out development records.
        feature_columns = list(candidates)
    leaked = sorted(
        column
        for column in feature_columns
        if not predictor_column_allowed(column)
    )
    if leaked:
        raise ValueError(
            "Post-landmark variables entered the predictor matrix: "
            + ", ".join(leaked)
        )
    static_all = numeric.reindex(columns=feature_columns)
    deep_columns = [column for column in DEEP_STATIC if column in feature_columns]
    static_deep = numeric.reindex(columns=deep_columns)
    return Dataset(
        name=dataset_name,
        cohort=cohort,
        sequences=sequences.astype(np.float32),
        masks=masks.astype(np.float32),
        labels=cohort["outcome"].astype(int).to_numpy(),
        static_all=static_all,
        static_deep=static_deep,
        analysis_weight=(
            pd.to_numeric(cohort["analysis_weight"], errors="raise")
            .to_numpy(dtype=np.float64)
            if "analysis_weight" in cohort
            else np.ones(len(cohort), dtype=np.float64)
        ),
    )


def dynamic_summary_columns() -> set[str]:
    """Return the 24-hour trajectory summaries excluded from static-only models."""
    return {
        "monitoring_intensity_24h",
        *{
            f"{channel}_{statistic}"
            for channel in CHANNELS
            for statistic in ("n", "mean", "min", "max", "sd", "slope")
        },
    }


def model_static_frame(dataset: Dataset, mode: str) -> pd.DataFrame:
    """Materialize one explicitly declared static-feature arm."""
    if mode == "all":
        return dataset.static_all
    if mode == "core":
        return dataset.static_deep
    if mode == "static":
        excluded = dynamic_summary_columns()
        return dataset.static_all.reindex(
            columns=[
                column
                for column in dataset.static_all.columns
                if column not in excluded
            ]
        )
    if mode == "placeholder":
        return pd.DataFrame(
            {"__static_placeholder": np.zeros(len(dataset.labels), dtype=np.float32)},
            index=dataset.static_all.index,
        )
    raise ValueError(f"Unknown static-feature mode: {mode}")


def model_sequence_inputs(
    dataset: Dataset,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed or explicitly ablated temporal inputs."""
    if mode == "observed":
        return dataset.sequences, dataset.masks
    if mode == "none":
        return (
            np.zeros_like(dataset.sequences, dtype=np.float32),
            np.zeros_like(dataset.masks, dtype=np.float32),
        )
    raise ValueError(f"Unknown sequence-input mode: {mode}")


def _stable_subject_order(subject_key: str, n_steps: int, seed: int) -> np.ndarray:
    """Return one deterministic time permutation for a patient."""
    digest = hashlib.sha256(f"{seed}|{subject_key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.permutation(n_steps).astype(np.int64)


def permute_time_order_by_subject(
    sequences: np.ndarray,
    masks: np.ndarray,
    subject_keys: np.ndarray,
    *,
    seed: int = TIME_PERMUTATION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a patient-stable time permutation before any fold-specific subsetting."""
    sequences = np.asarray(sequences)
    masks = np.asarray(masks)
    subject_keys = np.asarray(subject_keys).astype(str)
    if sequences.ndim != 3 or masks.ndim not in (2, 3):
        raise ValueError("Sequences and masks must have patient/time/channel axes")
    if len(sequences) != len(masks) or len(sequences) != len(subject_keys):
        raise ValueError("Sequences, masks, and subject keys are not patient-aligned")
    output_sequences = np.empty_like(sequences)
    output_masks = np.empty_like(masks)
    for subject_key in np.unique(subject_keys):
        rows = np.flatnonzero(subject_keys == subject_key)
        order = _stable_subject_order(str(subject_key), sequences.shape[1], seed)
        output_sequences[rows] = sequences[rows][:, order]
        if masks.ndim == 2:
            output_masks[rows] = masks[rows][:, order]
        else:
            output_masks[rows] = masks[rows][:, order, :]
    return output_sequences, output_masks


def validate_candidate_set(
    candidates: tuple[dict, ...] | list[dict],
    *,
    family: str,
    required: set[str],
) -> tuple[dict, ...]:
    """Validate prespecified candidate dictionaries without synthesising a grid."""
    if not candidates:
        raise ValueError(f"{family} hyperparameter candidates are empty")
    normalized: list[dict] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"{family} candidate {index} must be a dictionary"
            )
        missing = required.difference(candidate)
        if missing:
            raise ValueError(
                f"{family} candidate {index} lacks: {', '.join(sorted(missing))}"
            )
        extra = set(candidate).difference(required)
        if extra:
            raise ValueError(
                f"{family} candidate {index} has unknown fields: "
                + ", ".join(sorted(extra))
            )
        canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            raise ValueError(f"{family} contains a duplicate candidate")
        seen.add(canonical)
        normalized.append(copy.deepcopy(candidate))
    return tuple(normalized)


def configured_candidates(
    config: dict,
    family: str,
    required: set[str],
) -> tuple[dict, ...]:
    """Read one explicit candidate list from the study configuration."""
    block = config.get("hyperparameters", {}).get(family, {})
    strategy = block.get("search_strategy") if isinstance(block, dict) else None
    if strategy != "explicit_prespecified_grid":
        raise ValueError(
            f"{family} must declare search_strategy='explicit_prespecified_grid'"
        )
    candidates = block.get("candidates", [])
    return validate_candidate_set(
        candidates,
        family=family,
        required=required,
    )


def append_hyperparameter_record(
    records: list[dict] | None,
    *,
    model: str,
    family: str,
    cached: dict,
    default_candidate_count: int = 1,
    default_candidate_index: int = 0,
) -> None:
    """Append one model's fold-level selection exactly once."""
    if records is None or any(record.get("model") == model for record in records):
        return
    records.append(
        {
            "model": model,
            "family": family,
            "parameters": copy.deepcopy(cached["parameters"]),
            "inner_macro_auc": float(cached.get("inner_macro_auc", np.nan)),
            "candidate_count": int(
                cached.get("candidate_count", default_candidate_count)
            ),
            "selected_candidate_index": int(
                cached.get(
                    "selected_candidate_index",
                    default_candidate_index,
                )
            ),
            **(
                {
                    "inner_best_epoch": int(
                        cached.get("inner_best_epoch", 0)
                    )
                }
                if "inner_best_epoch" in cached
                else {}
            ),
        }
    )


def modal_hyperparameters(
    records: list[dict],
    *,
    default: dict,
) -> dict:
    """Select the most frequent development-fold choice without external labels."""
    if not records:
        return copy.deepcopy(default)
    counts: dict[str, tuple[int, dict]] = {}
    for record in records:
        parameters = record.get("parameters", {})
        canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        count = counts.get(canonical, (0, parameters))[0] + 1
        counts[canonical] = (count, parameters)
    _, selected = max(counts.values(), key=lambda item: item[0])
    return copy.deepcopy(selected)


def model_feature_contract(dataset: Dataset) -> dict[str, dict]:
    """Describe exact static and sequence inputs for every registered model."""
    contract: dict[str, dict] = {}
    for name, meta in MODEL_META.items():
        if meta["kind"] == "stacking":
            contract[name] = {
                "kind": "stacking",
                "members": list(meta["members"]),
            }
            continue
        if meta["kind"] in {"xgboost", "slope_logistic"}:
            static_frame = tabular_frame(dataset, meta["features"])
            sequence_mode = "none"
        else:
            static_frame = model_static_frame(
                dataset,
                str(meta.get("static_mode", "all")),
            )
            sequence_mode = str(meta.get("sequence_mode", "observed"))
        columns = [str(column) for column in static_frame.columns]
        contract[name] = {
            "kind": meta["kind"],
            "static_mode": str(meta.get("static_mode", meta.get("features", ""))),
            "static_feature_count": len(columns),
            "static_feature_sha256": hashlib.sha256(
                "\n".join(columns).encode("utf-8")
            ).hexdigest(),
            "sequence_mode": sequence_mode,
            "sequence_enabled": sequence_mode == "observed",
            "time_mode": str(meta.get("time_mode", "forward")),
            "trajectory_mask_mode": str(meta.get("mask_mode", "none")),
        }
    return contract


def impute_tabular(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    median = train.median(axis=0, skipna=True)
    median[~np.isfinite(median)] = 0.0
    train_out = train.fillna(median).to_numpy(dtype=np.float32)
    test_out = test.fillna(median).to_numpy(dtype=np.float32)
    return train_out, test_out


def fit_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
    parameters: dict | None = None,
) -> np.ndarray:
    values = copy.deepcopy(XGBOOST_DEFAULTS)
    if parameters:
        values.update(parameters)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        **values,
        tree_method="hist",
        random_state=seed,
        n_jobs=int(
            os.environ.get(
                "ICH_XGB_THREADS",
                str(max(1, os.cpu_count() or 1)),
            )
        ),
        eval_metric="mlogloss",
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=sample_weight,
        verbose=False,
    )
    return model.predict_proba(test_x)


def fit_slope_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Three-class logistic baseline using only the pre-landmark GCS slope."""
    if train_x.ndim != 2 or train_x.shape[1] != 1:
        raise ValueError("The GCS-slope baseline requires exactly one feature")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        ),
    )
    model.fit(train_x, train_y, logisticregression__sample_weight=sample_weight)
    raw = model.predict_proba(test_x)
    probabilities = np.zeros((len(test_x), N_CLASSES), dtype=np.float64)
    probabilities[:, np.asarray(model.classes_, dtype=int)] = raw
    return probabilities


def inner_split(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a patient-grouped inner validation split for early stopping."""
    split = patient_grouped_inner_splits(
        labels,
        groups,
        n_splits=5,
        seed=seed,
    )[0]
    train_index, valid_index = split
    if len(np.unique(labels[valid_index])) < N_CLASSES:
        raise ValueError("Inner validation split does not contain all outcome levels")
    return train_index, valid_index


def build_repeated_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    """Build the frozen patient-grouped repeated-CV partition contract."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups).astype(str)
    if len(labels) != len(groups):
        raise ValueError("Labels and patient groups must have equal length")
    if folds < 2 or repeats < 1:
        raise ValueError("Repeated CV requires at least two folds and one repeat")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    split_meta: list[tuple[int, int]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat * 10_000,
        )
        repeat_splits = list(
            splitter.split(
                np.zeros((len(labels), 1), dtype=np.float32),
                labels,
                groups=groups,
            )
        )
        if len(repeat_splits) != folds:
            raise RuntimeError(
                f"Expected {folds} outer folds, got {len(repeat_splits)}"
            )
        splits.extend(repeat_splits)
        split_meta.extend((repeat, fold) for fold in range(folds))
    return splits, split_meta


def patient_grouped_inner_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = STACKING_INNER_FOLDS,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return patient-disjoint stratified folds for an outer training set."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups).astype(str)
    if len(labels) != len(groups):
        raise ValueError("Labels and patient groups must have equal length")
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f"At least {n_splits} patients are required for {n_splits}-fold CV"
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = list(
        splitter.split(np.zeros(len(labels)), labels, groups=groups)
    )
    if len(splits) != n_splits:
        raise RuntimeError(f"Expected {n_splits} inner folds, got {len(splits)}")
    return splits


def _normalized_probabilities(
    probabilities: np.ndarray,
    expected_rows: int,
    *,
    context: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (expected_rows, N_CLASSES):
        raise ValueError(
            f"{context} must have shape ({expected_rows}, {N_CLASSES}); "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite probabilities")
    if (values < -1e-8).any():
        raise ValueError(f"{context} contains negative probabilities")
    values = np.clip(values, 0.0, None)
    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError(f"{context} contains rows with zero total probability")
    return values / row_sums


def stacking_meta_features(
    member_probabilities: dict[str, np.ndarray],
    members: tuple[str, ...] = STACKING_MEMBERS,
) -> np.ndarray:
    """Concatenate member probabilities in fixed member/class order."""
    matrices: list[np.ndarray] = []
    expected_rows: int | None = None
    for member in members:
        if member not in member_probabilities:
            raise ValueError(f"Missing stacking member: {member}")
        values = np.asarray(member_probabilities[member], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != N_CLASSES:
            raise ValueError(
                f"{member} probabilities must have {N_CLASSES} columns"
            )
        if expected_rows is None:
            expected_rows = len(values)
        elif len(values) != expected_rows:
            raise ValueError("Stacking members have different row counts")
        matrices.append(
            _normalized_probabilities(
                values,
                len(values),
                context=f"{member} probabilities",
            )
        )
    if not matrices:
        raise ValueError("At least one stacking member is required")
    return np.concatenate(matrices, axis=1)


def fit_stacking_meta_model(
    oof_probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    members: tuple[str, ...] = STACKING_MEMBERS,
    c: float = STACKING_C,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    """Fit the level-1 model exclusively on inner-OOF probabilities."""
    labels = np.asarray(labels, dtype=int)
    if np.unique(labels).size != N_CLASSES:
        raise ValueError("Stacking OOF labels must contain all outcome levels")
    features = stacking_meta_features(oof_probabilities, members)
    if len(features) != len(labels):
        raise ValueError("Stacking OOF features and labels are misaligned")
    model = LogisticRegression(
        C=float(c),
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def predict_stacking_meta_model(
    model: LogisticRegression,
    member_probabilities: dict[str, np.ndarray],
    *,
    members: tuple[str, ...] = STACKING_MEMBERS,
) -> np.ndarray:
    features = stacking_meta_features(member_probabilities, members)
    raw = model.predict_proba(features)
    probabilities = np.zeros((len(features), N_CLASSES), dtype=np.float64)
    probabilities[:, np.asarray(model.classes_, dtype=int)] = raw
    return _normalized_probabilities(
        probabilities,
        len(features),
        context="Level-1 stacking probabilities",
    )


def tabular_frame(dataset: Dataset, features: str) -> pd.DataFrame:
    if features == "static":
        return model_static_frame(dataset, "static")
    if features == "core":
        return model_static_frame(dataset, "core")
    if features == "gcs_slope":
        return dataset.static_all.reindex(columns=["gcs_total_slope"])
    if features == "all":
        return model_static_frame(dataset, "all")
    if features == "placeholder":
        return model_static_frame(dataset, "placeholder")
    return dataset.static_all


def _temporal_hyperparameter_cache_key(
    meta: dict,
    *,
    model_name: str,
    mask_mode: str,
    static_count: int,
    channel_count: int,
) -> str:
    """Identify one architecture-and-input contract for shared tuning.

    Regularization, temporal controls, and model labels are intentionally
    excluded. Matched DG-GRU arms must use one inner-selected architecture
    configuration so that only the prespecified temporal operation differs.
    Secondary ablations that change the static, sequence, or mask contract
    still receive distinct keys.
    """
    return "|".join(
        (
            str(meta.get("architecture", "dggru")),
            str(meta.get("static_mode", "all")),
            str(meta.get("sequence_mode", "observed")),
            mask_mode,
            str(bool(meta.get("graph", False))),
            str(static_count),
            str(channel_count),
        )
    )


def _selection_seed_values(
    fallback_seed: int,
    selection_seeds: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    """Return the fixed model-seed ensemble used for inner selection."""
    values = (
        tuple(int(value) for value in selection_seeds)
        if selection_seeds
        else (int(fallback_seed),)
    )
    if not values:
        raise ValueError("At least one inner-selection seed is required")
    if len(set(values)) != len(values):
        raise ValueError("Inner-selection seeds must be unique")
    return values


def fit_temporal(
    dataset: Dataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    meta: dict,
    seed: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...] | None = None,
    lambda_smooth_ratio: float = 0.4,
    tune_lambda: bool = False,
    model_name: str = "temporal",
    sequence_hyperparameters: tuple[dict, ...] = (),
    hyperparameter_cache: dict[str, dict] | None = None,
    hyperparameter_records: list[dict] | None = None,
    select_hyperparameters: bool = True,
    hyperparameter_override: dict | None = None,
    fixed_epochs: int | None = None,
    selection_seeds: tuple[int, ...] | list[int] = (),
    selection_offset: int = 0,
) -> tuple[np.ndarray, int, float]:
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()[train_index]
    static_frame = model_static_frame(
        dataset,
        str(meta.get("static_mode", "all")),
    )
    raw_static = static_frame.to_numpy(dtype=np.float32)
    raw_sequences, raw_masks = model_sequence_inputs(
        dataset,
        str(meta.get("sequence_mode", "observed")),
    )
    requested_time_mode = str(meta.get("time_mode", "forward"))
    effective_time_mode = requested_time_mode
    if requested_time_mode == "permute":
        raw_sequences, raw_masks = permute_time_order_by_subject(
            raw_sequences,
            raw_masks,
            dataset.cohort["subject_key"].astype(str).to_numpy(),
            seed=TIME_PERMUTATION_SEED,
        )
        effective_time_mode = "forward"
    mask_mode = str(meta.get("mask_mode", "any_channel"))
    weights = (
        np.ones(len(dataset.labels), dtype=np.float64)
        if dataset.analysis_weight is None
        else np.asarray(dataset.analysis_weight, dtype=np.float64)
    )
    if weights.shape != (len(dataset.labels),):
        raise ValueError("Dataset analysis weights are not patient-aligned")
    train_weights = weights[train_index]
    inner_train, inner_valid = inner_split(
        dataset.labels[train_index], groups, seed
    )
    inner_preprocessor = SequencePreprocessor.fit(
        raw_sequences[train_index][inner_train],
        raw_masks[train_index][inner_train],
        raw_static[train_index][inner_train],
    )
    inner_sequences, inner_masks = inner_preprocessor.transform_sequence(
        raw_sequences[train_index],
        raw_masks[train_index],
        mask_mode=mask_mode,
    )
    inner_static = inner_preprocessor.transform_static(raw_static[train_index])
    cache = hyperparameter_cache if hyperparameter_cache is not None else {}
    selection_seed_values = _selection_seed_values(seed, selection_seeds)
    if not 0 <= int(selection_offset) < len(selection_seed_values):
        raise ValueError("selection_offset is outside the model-seed ensemble")
    # Hyperparameter selection uses one fixed seed. Five-seed averaging is
    # reserved for the final prediction ensemble, avoiding five identical
    # grid searches for every candidate and inner split.
    selection_probe_seed = int(selection_seed_values[0])
    cache_key = _temporal_hyperparameter_cache_key(
        meta,
        model_name=model_name,
        mask_mode=mask_mode,
        static_count=raw_static.shape[1],
        channel_count=raw_sequences.shape[2],
    )
    if hyperparameter_override is not None:
        overridden = copy.deepcopy(SEQUENCE_DEFAULTS)
        overridden.update(hyperparameter_override)
        validated_override = validate_candidate_set(
            (overridden,),
            family="sequence_models",
            required=set(SEQUENCE_DEFAULTS),
        )[0]
        cached = {
            "parameters": validated_override,
            "inner_macro_auc": float("nan"),
            "inner_best_epoch": int(
                fixed_epochs if fixed_epochs is not None else epochs
            ),
            "inner_best_epochs": [
                int(fixed_epochs if fixed_epochs is not None else epochs)
                for _ in selection_seed_values
            ],
            "selected_by": "fixed_development_cv_parameters",
            "candidate_count": 1,
            "selected_candidate_index": 0,
        }
        cache[cache_key] = cached
    else:
        cached = cache.get(cache_key)
    if cached is None and select_hyperparameters:
        raw_candidates = sequence_hyperparameters or (
            copy.deepcopy(SEQUENCE_DEFAULTS),
        )
        candidates = validate_candidate_set(
            raw_candidates,
            family="sequence_models",
            required=set(SEQUENCE_DEFAULTS),
        )
        evaluated: list[tuple[int, dict, float, tuple[int, ...]]] = []
        for candidate_index, candidate in enumerate(candidates):
            parameters = copy.deepcopy(SEQUENCE_DEFAULTS)
            parameters.update(candidate)
            candidate_epochs = min(
                int(epochs),
                int(parameters["max_epochs"]),
            )
            seed_probabilities: list[np.ndarray] = []
            seed_best_epochs: list[int] = []
            for selection_seed in (selection_probe_seed,):
                validation_probabilities, _, history = train_temporal_model(
                    inner_sequences[inner_train],
                    inner_masks[inner_train],
                    inner_static[inner_train],
                    dataset.labels[train_index][inner_train],
                    inner_sequences[inner_valid],
                    inner_masks[inner_valid],
                    inner_static[inner_valid],
                    dataset.labels[train_index][inner_valid],
                    use_graph=bool(meta.get("graph", False)),
                    use_regularization=False,
                    architecture=str(meta.get("architecture", "dggru")),
                    lambda_monotone=0.0,
                    lambda_smooth=0.0,
                    epochs=candidate_epochs,
                    batch_size=int(parameters["batch_size"]),
                    learning_rate=float(parameters["learning_rate"]),
                    weight_decay=float(parameters["weight_decay"]),
                    hidden=int(parameters["hidden_size"]),
                    dropout=float(parameters["dropout"]),
                    patience=int(parameters["patience"]),
                    seed=selection_seed,
                    sample_weight=train_weights[inner_train],
                    laplace_only=bool(meta.get("laplace_only", False)),
                    laplace_input_scale=LAPLACE_INPUT_SCALE,
                )
                seed_probabilities.append(validation_probabilities)
                seed_best_epochs.append(
                    int(
                        min(
                            history,
                            key=lambda row: row["valid_loss"],
                        )["epoch"]
                    )
                )
            validation_probabilities = _normalized_probabilities(
                np.mean(seed_probabilities, axis=0),
                len(inner_valid),
                context="inner hyperparameter ensemble probabilities",
            )
            validation_auc = multiclass_auc(
                dataset.labels[train_index][inner_valid],
                validation_probabilities,
                train_weights[inner_valid],
            )
            evaluated.append(
                (
                    candidate_index,
                    parameters,
                    float(validation_auc),
                    tuple(seed_best_epochs),
                )
            )
        finite = [row for row in evaluated if np.isfinite(row[2])]
        selected = (
            max(finite, key=lambda row: (row[2], -row[0]))
            if finite
            else evaluated[0]
        )
        _, selected_parameters, selected_auc, selected_epochs = selected
        cached = {
            "parameters": copy.deepcopy(selected_parameters),
            "inner_macro_auc": float(selected_auc),
            "inner_best_epoch": int(selected_epochs[selection_offset]),
            "inner_best_epochs": [int(value) for value in selected_epochs],
            "selected_by": (
                "fixed_seed_inner_patient_grouped_macro_auc_then_"
                "five_seed_final_ensemble"
            ),
            "candidate_count": len(candidates),
            "selected_candidate_index": int(selected[0]),
        }
        cache[cache_key] = cached
    if cached is None:
        cached = {
            "parameters": copy.deepcopy(SEQUENCE_DEFAULTS),
            "inner_macro_auc": float("nan"),
            "inner_best_epoch": int(epochs),
            "inner_best_epochs": [
                int(epochs) for _ in selection_seed_values
            ],
            "selected_by": "prespecified_default",
        }
        cache[cache_key] = cached
    append_hyperparameter_record(
        hyperparameter_records,
        model=model_name,
        family="sequence_models",
        cached=cached,
    )
    parameters = copy.deepcopy(cached["parameters"])
    candidate_epochs = min(int(epochs), int(parameters["max_epochs"]))
    candidates = (
        tuple(float(value) for value in lambda_grid)
        if tune_lambda
        and lambda_grid
        and bool(meta.get("regularized", False))
        and not bool(meta.get("laplace_only", False))
        else (float(lambda_monotone),)
    )
    tuned: list[tuple[float, float, tuple[int, ...]]] = []
    candidate_selection_seeds = (
        (selection_probe_seed,) if tune_lambda else (int(seed),)
    )
    candidate_selection_offset = selection_offset if tune_lambda else 0
    if fixed_epochs is not None and not tune_lambda:
        selected_lambda = float(candidates[0])
        best_epoch = int(fixed_epochs)
    else:
        for candidate in candidates:
            seed_probabilities = []
            seed_best_epochs = []
            for selection_seed in candidate_selection_seeds:
                validation_probabilities, _, history = train_temporal_model(
                    inner_sequences[inner_train],
                    inner_masks[inner_train],
                    inner_static[inner_train],
                    dataset.labels[train_index][inner_train],
                    inner_sequences[inner_valid],
                    inner_masks[inner_valid],
                    inner_static[inner_valid],
                    dataset.labels[train_index][inner_valid],
                    use_graph=bool(meta.get("graph", False)),
                    use_regularization=bool(meta.get("regularized", False)),
                    architecture=str(meta.get("architecture", "dggru")),
                    regularization_direction=meta.get(
                        "direction", "increasing"
                    ),
                    lambda_monotone=(
                        0.0
                        if meta.get("laplace_only", False)
                        else candidate
                    ),
                    lambda_smooth=(
                        lambda_smooth
                        if not tune_lambda
                        else max(0.0, float(lambda_smooth_ratio))
                        * candidate
                    ),
                    epochs=candidate_epochs,
                    batch_size=int(parameters["batch_size"]),
                    learning_rate=float(parameters["learning_rate"]),
                    weight_decay=float(parameters["weight_decay"]),
                    hidden=int(parameters["hidden_size"]),
                    dropout=float(parameters["dropout"]),
                    patience=int(parameters["patience"]),
                    seed=selection_seed,
                    time_mode=effective_time_mode,
                    sample_weight=train_weights[inner_train],
                    laplace_only=bool(meta.get("laplace_only", False)),
                    laplace_input_scale=LAPLACE_INPUT_SCALE,
                )
                seed_probabilities.append(validation_probabilities)
                seed_best_epochs.append(
                    int(
                        min(
                            history,
                            key=lambda row: row["valid_loss"],
                        )["epoch"]
                    )
                )
            validation_probabilities = _normalized_probabilities(
                np.mean(seed_probabilities, axis=0),
                len(inner_valid),
                context="inner lambda ensemble probabilities",
            )
            validation_auc = multiclass_auc(
                dataset.labels[train_index][inner_valid],
                validation_probabilities,
                train_weights[inner_valid],
            )
            tuned.append(
                (
                    candidate,
                    float(validation_auc),
                    tuple(seed_best_epochs),
                )
            )
        finite = [row for row in tuned if np.isfinite(row[1])]
        if finite:
            selected_lambda, _, best_epochs = max(
                finite,
                key=lambda row: (row[1], -row[0]),
            )
        else:
            selected_lambda, best_epochs = float(candidates[0]), tuned[0][2]
        best_epoch = int(best_epochs[candidate_selection_offset])
    final_epochs = (
        min(int(fixed_epochs), candidate_epochs)
        if fixed_epochs is not None
        else best_epoch
    )
    outer_preprocessor = SequencePreprocessor.fit(
        raw_sequences[train_index],
        raw_masks[train_index],
        raw_static[train_index],
    )
    train_sequences, train_masks = outer_preprocessor.transform_sequence(
        raw_sequences[train_index],
        raw_masks[train_index],
        mask_mode=mask_mode,
    )
    test_sequences, test_masks = outer_preprocessor.transform_sequence(
        raw_sequences[test_index],
        raw_masks[test_index],
        mask_mode=mask_mode,
    )
    train_static = outer_preprocessor.transform_static(raw_static[train_index])
    test_static = outer_preprocessor.transform_static(raw_static[test_index])
    model = fit_full_temporal_model(
        train_sequences,
        train_masks,
        train_static,
        dataset.labels[train_index],
        use_graph=bool(meta.get("graph", False)),
        use_regularization=bool(meta.get("regularized", False)),
        architecture=str(meta.get("architecture", "dggru")),
        regularization_direction=meta.get("direction", "increasing"),
        lambda_monotone=(
            0.0 if meta.get("laplace_only", False) else selected_lambda
        ),
        lambda_smooth=(
            lambda_smooth
            if not tune_lambda
            else max(0.0, float(lambda_smooth_ratio)) * selected_lambda
        ),
        seed=seed,
        epochs=final_epochs,
        batch_size=int(parameters["batch_size"]),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        hidden=int(parameters["hidden_size"]),
        dropout=float(parameters["dropout"]),
        time_mode=effective_time_mode,
        sample_weight=train_weights,
        laplace_only=bool(meta.get("laplace_only", False)),
        laplace_input_scale=LAPLACE_INPUT_SCALE,
    )
    model_probabilities, _ = predict_temporal_model(
        model,
        test_sequences,
        test_masks,
        test_static,
        time_mode=effective_time_mode,
    )
    return model_probabilities, final_epochs, selected_lambda


def _configure_training_worker(threads: int) -> None:
    torch.set_num_threads(max(1, threads))
    os.environ["ICH_XGB_THREADS"] = str(max(1, threads))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_source_files() -> list[str]:
    """Return modules reachable from this entry point by relative imports.

    Report, figure, and manuscript generators sit in the same package but
    cannot change a fitted model. Hashing them alongside the numerical code
    would invalidate a multi-hour run whenever documentation is edited, so the
    snapshot covers the import closure only.
    """
    root = Path(__file__).resolve().parent
    pending = ["__init__.py", Path(__file__).name]
    visited: set[str] = set()
    while pending:
        filename = pending.pop()
        if filename in visited:
            continue
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Training source is missing: {path}")
        visited.add(filename)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module:
                dependency = f"{node.module.split('.')[0]}.py"
                if (root / dependency).is_file():
                    pending.append(dependency)
                continue
            for alias in node.names:
                dependency = f"{alias.name}.py"
                if (root / dependency).is_file():
                    pending.append(dependency)
    return sorted(visited)


def training_source_snapshot() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        filename: sha256_file(root / filename)
        for filename in training_source_files()
    }


def load_config_snapshot(path: Path = CONFIG_PATH) -> tuple[dict, str]:
    """Parse the exact configuration bytes whose hash is bound to the run."""
    raw = path.read_bytes()
    config = validate_study_config(json.loads(raw.decode("utf-8")))
    return config, hashlib.sha256(raw).hexdigest()


def snapshot_training_inputs(
    config_path: Path = CONFIG_PATH,
) -> tuple[dict, str, dict[str, str]]:
    """Hash configuration and processed inputs before any model fitting."""
    config, config_sha256 = load_config_snapshot(config_path)
    data_hashes = {
        f"{dataset}/{filename}": sha256_file(PROCESSED / dataset / filename)
        for dataset in ("mimic4", "mimic3", "eicu")
        for filename in DATA_CONTRACT_FILES
        if (PROCESSED / dataset / filename).is_file()
    }
    missing = [
        f"{dataset}/{filename}"
        for dataset in ("mimic4", "mimic3", "eicu")
        for filename in DATA_CONTRACT_FILES
        if not (PROCESSED / dataset / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Training input snapshot is incomplete: " + ", ".join(missing)
        )
    return config, config_sha256, data_hashes


def require_inputs_unchanged(
    config_sha256: str,
    data_hashes: dict[str, str],
    *,
    config_path: Path = CONFIG_PATH,
) -> None:
    """Fail closed if configuration or processed data drift during training."""
    if sha256_file(config_path) != config_sha256:
        raise RuntimeError("Study configuration changed during model fitting")
    changed = [
        relative
        for relative, recorded in data_hashes.items()
        if sha256_file(PROCESSED / Path(relative)) != recorded
    ]
    if changed:
        raise RuntimeError(
            "Processed training inputs changed during model fitting: "
            + ", ".join(changed)
        )


def installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def select_xgboost_parameters(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    train_weight: np.ndarray,
    *,
    seed: int,
    candidates: tuple[dict, ...] | list[dict],
    raw_train_frame: pd.DataFrame | None = None,
    selection_seeds: tuple[int, ...] | list[int] = (),
) -> tuple[dict, float, int]:
    """Select one prespecified XGBoost candidate on a patient-grouped inner split."""
    inner_train, inner_valid = inner_split(train_y, train_groups, seed)
    seed_values = _selection_seed_values(seed, selection_seeds)[:1]
    validated = validate_candidate_set(
        candidates,
        family="xgboost",
        required=set(XGBOOST_DEFAULTS),
    )
    evaluated: list[tuple[int, dict, float]] = []
    for candidate_index, candidate in enumerate(validated):
        parameters = copy.deepcopy(XGBOOST_DEFAULTS)
        parameters.update(candidate)
        if raw_train_frame is None:
            candidate_train_x = train_x[inner_train]
            candidate_valid_x = train_x[inner_valid]
        else:
            candidate_train_x, candidate_valid_x = impute_tabular(
                raw_train_frame.iloc[inner_train].reset_index(drop=True),
                raw_train_frame.iloc[inner_valid].reset_index(drop=True),
            )
        seed_probabilities = [
            fit_xgboost(
                candidate_train_x,
                train_y[inner_train],
                candidate_valid_x,
                selection_seed,
                train_weight[inner_train],
                parameters=parameters,
            )
            for selection_seed in seed_values
        ]
        probabilities = _normalized_probabilities(
            np.mean(seed_probabilities, axis=0),
            len(inner_valid),
            context="inner XGBoost ensemble probabilities",
        )
        score = multiclass_auc(
            train_y[inner_valid],
            probabilities,
            train_weight[inner_valid],
        )
        evaluated.append((candidate_index, parameters, float(score)))
    finite = [row for row in evaluated if np.isfinite(row[2])]
    selected = (
        max(finite, key=lambda row: (row[2], -row[0]))
        if finite
        else evaluated[0]
    )
    return selected[1], selected[2], selected[0]


def _seed_epoch_override(
    override: int | tuple[int, ...] | None,
    offset: int,
    n_seeds: int,
) -> int | None:
    """Resolve a scalar or per-seed epoch schedule without silent truncation."""
    if override is None:
        return None
    if isinstance(override, (int, np.integer)):
        value = int(override)
        if value < 1:
            raise ValueError("Epoch override must be positive")
        return value
    values = tuple(int(value) for value in override)
    if len(values) != n_seeds:
        raise ValueError(
            "Per-seed epoch override must contain one value per model seed"
        )
    if any(value < 1 for value in values):
        raise ValueError("Every per-seed epoch override must be positive")
    return values[offset]


def _fit_member_predictions(
    dataset: Dataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    name: str,
    fold: int,
    seed: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...],
    lambda_smooth_ratio: float,
    model_seeds: tuple[int, ...],
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    hyperparameter_cache: dict[str, dict] | None = None,
    hyperparameter_records: list[dict] | None = None,
    hyperparameter_override: dict | None = None,
    lambda_override: float | None = None,
    smooth_override: float | None = None,
    epoch_override: int | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Fit one level-0 member and return its held-out probabilities."""
    meta = MODEL_META[name]
    if meta["kind"] == "stacking":
        raise ValueError("A stacking model cannot be used as a level-0 member")
    xgboost_seed_base = seed + fold
    xgboost_selection_seeds = tuple(
        xgboost_seed_base + 1000 * offset
        for offset in range(len(model_seeds))
    )
    if meta["kind"] in {"xgboost", "slope_logistic"}:
        frame = tabular_frame(dataset, meta["features"])
        train_x, test_x = impute_tabular(
            frame.iloc[train_index],
            frame.iloc[test_index],
        )
        train_weight = (
            np.ones(len(train_index), dtype=np.float64)
            if dataset.analysis_weight is None
            else np.asarray(dataset.analysis_weight, dtype=np.float64)[train_index]
        )
        parameters: dict | None = None
        if meta["kind"] == "xgboost":
            cache = hyperparameter_cache if hyperparameter_cache is not None else {}
            cache_key = (
                f"xgboost|{name}|{meta['features']}|{train_x.shape[1]}"
            )
            if hyperparameter_override is not None:
                overridden = copy.deepcopy(XGBOOST_DEFAULTS)
                overridden.update(hyperparameter_override)
                parameters = validate_candidate_set(
                    (overridden,),
                    family="xgboost",
                    required=set(XGBOOST_DEFAULTS),
                )[0]
                cached = {
                    "parameters": copy.deepcopy(parameters),
                    "inner_macro_auc": float("nan"),
                    "selected_by": "fixed_development_cv_parameters",
                    "candidate_count": 1,
                    "selected_candidate_index": 0,
                }
                cache[cache_key] = cached
            else:
                cached = cache.get(cache_key)
            if cached is None:
                raw_candidates = xgb_candidates or (
                    copy.deepcopy(XGBOOST_DEFAULTS),
                )
                train_groups = dataset.cohort["subject_key"].astype(str).to_numpy()[
                    train_index
                ]
                raw_train_frame = frame.iloc[train_index].reset_index(drop=True)
                parameters, score, candidate_index = select_xgboost_parameters(
                    train_x,
                    dataset.labels[train_index],
                    train_groups,
                    train_weight,
                    seed=seed + fold,
                    candidates=raw_candidates,
                    raw_train_frame=raw_train_frame,
                    selection_seeds=xgboost_selection_seeds,
                )
                cached = {
                    "parameters": copy.deepcopy(parameters),
                    "inner_macro_auc": float(score),
                    "selected_by": (
                        "fixed_seed_inner_patient_grouped_macro_auc_then_"
                        "five_seed_final_ensemble"
                    ),
                    "candidate_count": len(raw_candidates),
                    "selected_candidate_index": int(candidate_index),
                }
                cache[cache_key] = cached
            append_hyperparameter_record(
                hyperparameter_records,
                model=name,
                family="xgboost",
                cached=cached,
            )
            parameters = copy.deepcopy(cached["parameters"])
        fitter = fit_xgboost if meta["kind"] == "xgboost" else fit_slope_logistic
        parts = []
        for offset in range(len(model_seeds)):
            kwargs = (
                {"parameters": parameters}
                if meta["kind"] == "xgboost"
                else {}
            )
            parts.append(
                fitter(
                train_x,
                dataset.labels[train_index],
                test_x,
                xgboost_selection_seeds[offset],
                train_weight,
                **kwargs,
                )
            )
        probabilities = _normalized_probabilities(
            np.mean(parts, axis=0),
            len(test_index),
            context=f"{name} probabilities",
        )
        return probabilities, [], []

    parts = []
    epochs_used: list[int] = []
    lambdas_used: list[float] = []
    local_hyperparameter_cache = (
        hyperparameter_cache if hyperparameter_cache is not None else {}
    )
    selected_lambda = float(lambda_monotone)
    selected_smooth = float(lambda_smooth)
    fixed_lambda = (
        float(lambda_override) if lambda_override is not None else None
    )
    fixed_smooth = (
        float(smooth_override) if smooth_override is not None else None
    )
    temporal_seed_base = seed + 100 * (fold + 1)
    selection_seeds = tuple(
        temporal_seed_base + 1000 * offset
        for offset in range(len(model_seeds))
    )
    for offset in range(len(model_seeds)):
        seed_epochs = _seed_epoch_override(
            epoch_override,
            offset,
            len(model_seeds),
        )
        probabilities, best_epoch, selected_lambda = fit_temporal(
            dataset,
            train_index,
            test_index,
            meta,
            seed=selection_seeds[offset],
            epochs=epochs,
            lambda_monotone=(
                fixed_lambda if fixed_lambda is not None else selected_lambda
            ),
            lambda_smooth=(
                fixed_smooth if fixed_smooth is not None else selected_smooth
            ),
            lambda_grid=lambda_grid,
            lambda_smooth_ratio=lambda_smooth_ratio,
            tune_lambda=(
                name == "dggru_tr"
                and offset == 0
                and bool(lambda_grid)
                and fixed_lambda is None
            ),
            model_name=name,
            sequence_hyperparameters=sequence_hyperparameters,
            hyperparameter_cache=local_hyperparameter_cache,
            hyperparameter_records=hyperparameter_records,
            hyperparameter_override=hyperparameter_override,
            fixed_epochs=seed_epochs,
            selection_seeds=selection_seeds,
            selection_offset=offset,
        )
        parts.append(probabilities)
        epochs_used.append(best_epoch)
        lambdas_used.append(
            fixed_lambda if fixed_lambda is not None else selected_lambda
        )
        if offset == 0 and name == "dggru_tr" and lambda_grid:
            selected_smooth = (
                max(0.0, float(lambda_smooth_ratio)) * selected_lambda
            )
    probabilities = _normalized_probabilities(
        np.mean(parts, axis=0),
        len(test_index),
        context=f"{name} probabilities",
    )
    return probabilities, epochs_used, lambdas_used


def _stacking_oof_probabilities(
    dataset: Dataset,
    train_index: np.ndarray,
    *,
    seed: int,
    fold: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...],
    lambda_smooth_ratio: float,
    model_seeds: tuple[int, ...],
    members: tuple[str, ...],
    inner_folds: int,
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    hyperparameter_cache: dict[str, dict] | None = None,
    hyperparameter_records: list[dict] | None = None,
    hyperparameter_overrides: dict[str, dict] | None = None,
    lambda_overrides: dict[str, float] | None = None,
    smooth_overrides: dict[str, float] | None = None,
    epoch_overrides: dict[str, int | tuple[int, ...]] | None = None,
    inner_model_seeds: tuple[int, ...] | None = None,
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Generate inner patient-grouped OOF probabilities inside one outer train."""
    # The inner OOF path is a prediction generator for the level-1 model, not
    # the reported outer estimate. One fixed seed is sufficient there; the
    # outer member predictions retain the full seed ensemble.
    inner_seeds = (
        tuple(int(value) for value in inner_model_seeds)
        if inner_model_seeds is not None
        else (int(model_seeds[0]),)
    )
    if not inner_seeds:
        raise ValueError("Inner OOF stacking requires at least one model seed")
    train_index = np.asarray(train_index, dtype=int)
    train_labels = dataset.labels[train_index]
    train_groups = (
        dataset.cohort["subject_key"].astype(str).to_numpy()[train_index]
    )
    inner_splits = patient_grouped_inner_splits(
        train_labels,
        train_groups,
        n_splits=inner_folds,
        seed=seed + 50_000 + fold,
    )
    oof_probabilities = {
        member: np.full(
            (len(train_index), N_CLASSES),
            np.nan,
            dtype=np.float64,
        )
        for member in members
    }
    coverage = np.zeros(len(train_index), dtype=int)
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        if np.unique(train_labels[inner_train]).size != N_CLASSES:
            raise ValueError(
                f"Stacking inner training fold {inner_fold} lacks an outcome level"
            )
        absolute_train = train_index[inner_train]
        absolute_valid = train_index[inner_valid]
        for member in members:
            # Hyperparameters must be selected separately within each inner
            # training split. Reusing one outer-fold cache lets labels from a
            # later inner validation fold influence an earlier OOF prediction.
            probabilities, _, _ = _fit_member_predictions(
                dataset,
                absolute_train,
                absolute_valid,
                name=member,
                fold=fold * (inner_folds + 1) + inner_fold + 1,
                seed=seed,
                epochs=epochs,
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                lambda_grid=lambda_grid,
                lambda_smooth_ratio=lambda_smooth_ratio,
                model_seeds=inner_seeds,
                xgb_candidates=xgb_candidates,
                sequence_hyperparameters=sequence_hyperparameters,
                hyperparameter_cache=None,
                hyperparameter_records=None,
                hyperparameter_override=(hyperparameter_overrides or {}).get(
                    member
                ),
                lambda_override=(lambda_overrides or {}).get(member),
                smooth_override=(smooth_overrides or {}).get(member),
                epoch_override=(epoch_overrides or {}).get(member),
            )
            oof_probabilities[member][inner_valid] = probabilities
        coverage[inner_valid] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Inner OOF folds did not cover each outer-train row once")
    return oof_probabilities, inner_splits


def nested_stacking_predictions(
    dataset: Dataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    seed: int,
    fold: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...] = (),
    lambda_smooth_ratio: float = 0.4,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
    members: tuple[str, ...] = STACKING_MEMBERS,
    inner_folds: int = STACKING_INNER_FOLDS,
    stacking_c: float = STACKING_C,
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    hyperparameter_cache: dict[str, dict] | None = None,
    hyperparameter_records: list[dict] | None = None,
    hyperparameter_overrides: dict[str, dict] | None = None,
    lambda_overrides: dict[str, float] | None = None,
    smooth_overrides: dict[str, float] | None = None,
    epoch_overrides: dict[str, int | tuple[int, ...]] | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    LogisticRegression,
    list[tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, list[int]],
    dict[str, list[float]],
]:
    """Fit nested stacking without exposing the outer test set to selection."""
    train_index = np.asarray(train_index, dtype=int)
    test_index = np.asarray(test_index, dtype=int)
    if len(set(train_index).intersection(test_index)):
        raise ValueError("Outer train and test indices overlap")
    members = tuple(members)
    if not members:
        raise ValueError("At least one level-0 member is required")
    if "nested_stack" in members:
        raise ValueError("nested_stack cannot appear among its own members")
    unknown = [member for member in members if member not in MODEL_META]
    if unknown:
        raise ValueError(
            "Unknown stacking members: " + ", ".join(unknown)
        )
    oof_probabilities, inner_splits = _stacking_oof_probabilities(
        dataset,
        train_index,
        seed=seed,
        fold=fold,
        epochs=epochs,
        lambda_monotone=lambda_monotone,
        lambda_smooth=lambda_smooth,
        lambda_grid=lambda_grid,
        lambda_smooth_ratio=lambda_smooth_ratio,
        model_seeds=model_seeds,
        members=members,
        inner_folds=inner_folds,
        xgb_candidates=xgb_candidates,
        sequence_hyperparameters=sequence_hyperparameters,
        hyperparameter_overrides=hyperparameter_overrides,
        lambda_overrides=lambda_overrides,
        smooth_overrides=smooth_overrides,
        epoch_overrides=epoch_overrides,
    )
    meta_model = fit_stacking_meta_model(
        oof_probabilities,
        dataset.labels[train_index],
        members=members,
        c=stacking_c,
        seed=seed + fold,
        sample_weight=(
            None
            if dataset.analysis_weight is None
            else np.asarray(dataset.analysis_weight, dtype=np.float64)[
                train_index
            ]
        ),
    )
    outer_member_probabilities: dict[str, np.ndarray] = {}
    member_epochs: dict[str, list[int]] = {}
    member_lambdas: dict[str, list[float]] = {}
    for member in members:
        probabilities, epochs_used, lambdas_used = _fit_member_predictions(
            dataset,
            train_index,
            test_index,
            name=member,
            fold=fold,
            seed=seed,
            epochs=epochs,
            lambda_monotone=lambda_monotone,
            lambda_smooth=lambda_smooth,
            lambda_grid=lambda_grid,
            lambda_smooth_ratio=lambda_smooth_ratio,
            model_seeds=model_seeds,
            xgb_candidates=xgb_candidates,
            sequence_hyperparameters=sequence_hyperparameters,
            hyperparameter_cache=hyperparameter_cache,
            hyperparameter_records=hyperparameter_records,
            hyperparameter_override=(hyperparameter_overrides or {}).get(
                member
            ),
            lambda_override=(lambda_overrides or {}).get(member),
            smooth_override=(smooth_overrides or {}).get(member),
            epoch_override=(epoch_overrides or {}).get(member),
        )
        outer_member_probabilities[member] = probabilities
        member_epochs[member] = epochs_used
        member_lambdas[member] = lambdas_used
    stacked = predict_stacking_meta_model(
        meta_model,
        outer_member_probabilities,
        members=members,
    )
    return (
        stacked,
        oof_probabilities,
        meta_model,
        inner_splits,
        outer_member_probabilities,
        member_epochs,
        member_lambdas,
    )


def _train_fold(
    dataset: Dataset,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    epochs: int,
    seed: int,
    models: tuple[str, ...],
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...],
    lambda_smooth_ratio: float,
    model_seeds: tuple[int, ...],
    inner_folds: int = STACKING_INNER_FOLDS,
    stacking_c: float = STACKING_C,
    stacking_members: tuple[str, ...] = STACKING_MEMBERS,
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    fold_analysis_weight: np.ndarray | None = None,
    epoch_overrides: dict[str, int | tuple[int, ...]] | None = None,
    lambda_overrides: dict[str, float] | None = None,
    smooth_overrides: dict[str, float] | None = None,
) -> tuple[
    int,
    dict[str, np.ndarray],
    dict[str, list[int]],
    dict[str, list[float]],
    list[dict],
]:
    if fold_analysis_weight is not None:
        weight = np.asarray(fold_analysis_weight, dtype=np.float64)
        if weight.shape != (len(dataset.labels),):
            raise ValueError("Fold-specific analysis weights are patient-misaligned")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Fold-specific analysis weights are invalid")
        dataset = replace(dataset, analysis_weight=weight)
    fold_predictions = {
        name: np.zeros((len(dataset.labels), 3), dtype=np.float64)
        for name in models
    }
    fold_epochs = {name: [] for name in models}
    fold_lambdas = {name: [] for name in models}
    hyperparameter_cache: dict[str, dict] = {}
    hyperparameter_records: list[dict] = []
    stack_members = set(stacking_members) if "nested_stack" in models else set()
    matched_control_lambda: float | None = None
    matched_control_epochs: tuple[int, ...] | None = None
    matched_control_names = {
        "dggru_tr_treverse",
        "dggru_tr_tpermute",
        "dggru_laplace",
        "dggru_tr_gcsmask",
        "dggru_tr_noseq",
        "dggru_tr_static",
        "dggru_tr_traj",
    }
    frozen_epochs = dict(epoch_overrides or {})
    frozen_lambdas = dict(lambda_overrides or {})
    frozen_smooths = dict(smooth_overrides or {})

    # The stack members are fitted first and reused. This keeps the expensive
    # temporal models to one outer-fold fit and makes the selected dggru_tr
    # penalty available before the matched negative controls are trained.
    if "nested_stack" in models:
        (
            probabilities,
            _,
            _,
            _,
            member_probabilities,
            member_epochs,
            member_lambdas,
        ) = nested_stacking_predictions(
            dataset,
            train_index,
            test_index,
            seed=seed,
            fold=fold,
            epochs=epochs,
            lambda_monotone=lambda_monotone,
            lambda_smooth=lambda_smooth,
            lambda_grid=lambda_grid,
            lambda_smooth_ratio=lambda_smooth_ratio,
            model_seeds=model_seeds,
            members=stacking_members,
            inner_folds=inner_folds,
            stacking_c=stacking_c,
            xgb_candidates=xgb_candidates,
            sequence_hyperparameters=sequence_hyperparameters,
            hyperparameter_cache=hyperparameter_cache,
            hyperparameter_records=hyperparameter_records,
        )
        fold_predictions["nested_stack"][test_index] = probabilities
        for member in stacking_members:
            if member in models:
                fold_predictions[member][test_index] = (
                    member_probabilities[member]
                )
                fold_epochs[member].extend(member_epochs[member])
                fold_lambdas[member].extend(member_lambdas[member])
        if member_lambdas.get("dggru_tr"):
            matched_control_lambda = float(
                np.median(member_lambdas["dggru_tr"])
            )
        if member_epochs.get("dggru_tr"):
            matched_control_epochs = tuple(
                int(value) for value in member_epochs["dggru_tr"]
            )

    # The regularized arm is the canonical matched configuration. Fit it
    # before the unregularized comparator when nested stacking is not part of
    # the requested model set.
    fit_order = tuple(
        [name for name in models if name == "dggru_tr"]
        + [name for name in models if name != "dggru_tr"]
    )
    for name in fit_order:
        if MODEL_META[name]["kind"] == "stacking":
            continue
        if name in stack_members:
            continue
        member_lambda = float(lambda_monotone)
        member_smooth = float(lambda_smooth)
        if name in frozen_lambdas:
            member_lambda = float(frozen_lambdas[name])
        if name in frozen_smooths:
            member_smooth = float(frozen_smooths[name])
        if (
            name in matched_control_names
            and matched_control_lambda is not None
            and name not in frozen_lambdas
        ):
            member_lambda = matched_control_lambda
            member_smooth = lambda_smooth_ratio * matched_control_lambda
        seed_epochs = frozen_epochs.get(name)
        if (
            seed_epochs is None
            and (name == "dggru" or name in matched_control_names)
        ):
            seed_epochs = matched_control_epochs
        probabilities, epochs_used, lambdas_used = _fit_member_predictions(
            dataset,
            train_index,
            test_index,
            name=name,
            fold=fold,
            seed=seed,
            epochs=epochs,
            lambda_monotone=member_lambda,
            lambda_smooth=member_smooth,
            lambda_grid=lambda_grid,
            lambda_smooth_ratio=lambda_smooth_ratio,
            model_seeds=model_seeds,
            xgb_candidates=xgb_candidates,
            sequence_hyperparameters=sequence_hyperparameters,
            hyperparameter_cache=hyperparameter_cache,
            hyperparameter_records=hyperparameter_records,
            epoch_override=seed_epochs,
        )
        fold_predictions[name][test_index] = probabilities
        fold_epochs[name].extend(epochs_used)
        fold_lambdas[name].extend(lambdas_used)
        if name == "dggru_tr" and lambdas_used:
            matched_control_lambda = float(np.median(lambdas_used))
        if name == "dggru_tr" and epochs_used:
            matched_control_epochs = tuple(int(value) for value in epochs_used)
    return (
        fold,
        fold_predictions,
        fold_epochs,
        fold_lambdas,
        hyperparameter_records,
    )


def run_cv(
    dataset: Dataset,
    folds: int,
    epochs: int,
    seed: int,
    models: tuple[str, ...],
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_grid: tuple[float, ...] = (),
    lambda_smooth_ratio: float = 0.4,
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    workers: int = 1,
    worker_threads: int = 3,
    repeats: int = 1,
    repeat_prefix: str | None = None,
    nested_inner_folds: int = STACKING_INNER_FOLDS,
    stacking_c: float = STACKING_C,
    stacking_members: tuple[str, ...] = STACKING_MEMBERS,
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    fold_weight_builder: Callable[[np.ndarray], np.ndarray] | None = None,
    epoch_overrides: dict[str, int | tuple[int, ...]] | None = None,
    lambda_overrides: dict[str, float] | None = None,
    smooth_overrides: dict[str, float] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    list[tuple[np.ndarray, np.ndarray]],
    dict[str, list[int]],
    dict[str, list[float]],
    dict[str, list[dict]],
]:
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    splits, split_meta = build_repeated_splits(
        dataset.labels,
        groups,
        folds=folds,
        repeats=max(1, repeats),
        seed=seed,
    )
    predictions = {
        name: np.zeros((len(dataset.labels), 3), dtype=np.float64)
        for name in models
    }
    repeat_predictions = {
        repeat: {
            name: np.zeros((len(dataset.labels), 3), dtype=np.float64)
            for name in models
        }
        for repeat in range(max(1, repeats))
    }
    repeat_weights = {
        repeat: np.zeros(len(dataset.labels), dtype=np.float64)
        for repeat in range(max(1, repeats))
    }
    selected_epochs = {name: [] for name in models}
    selected_lambdas = {name: [] for name in models}
    fold_weights: list[np.ndarray] = []
    for train_index, _ in splits:
        if fold_weight_builder is None:
            weight = np.ones(len(dataset.labels), dtype=np.float64)
        else:
            weight = np.asarray(
                fold_weight_builder(train_index),
                dtype=np.float64,
            )
        if weight.shape != (len(dataset.labels),):
            raise ValueError(
                "Fold weight builder returned weights misaligned with the cohort"
            )
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Fold weight builder returned invalid weights")
        fold_weights.append(weight)
    fold_results: list[
        tuple[
            int,
            int,
            dict[str, np.ndarray],
            dict[str, list[int]],
            dict[str, list[float]],
            list[dict],
        ]
    ] = []
    tasks = [
        (
            task,
            repeat,
            fold,
            train_index,
            test_index,
            fold_weights[task],
        )
        for task, ((train_index, test_index), (repeat, fold)) in enumerate(
            zip(splits, split_meta)
        )
    ]
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_configure_training_worker,
            initargs=(worker_threads,),
        ) as executor:
            futures = {
                executor.submit(
                    _train_fold,
                    dataset,
                    fold,
                    train_index,
                    test_index,
                    epochs,
                    seed + repeat * 10_000,
                    models,
                    lambda_monotone,
                    lambda_smooth,
                    lambda_grid,
                    lambda_smooth_ratio,
                    model_seeds,
                    nested_inner_folds,
                    stacking_c,
                    stacking_members,
                    xgb_candidates,
                    sequence_hyperparameters,
                    weight,
                    epoch_overrides,
                    lambda_overrides,
                    smooth_overrides,
                ): (task, repeat, fold)
                for task, repeat, fold, train_index, test_index, weight in tasks
            }
            for future in as_completed(futures):
                _, repeat, fold = futures[future]
                (
                    result_fold,
                    result_predictions,
                    result_epochs,
                    result_lambdas,
                    result_hyperparameters,
                ) = future.result()
                fold_results.append(
                    (
                        repeat,
                        result_fold,
                        result_predictions,
                        result_epochs,
                        result_lambdas,
                        result_hyperparameters,
                    )
                )
                print(
                    f"[{dataset.name}] completed repeat {repeat + 1}/"
                    f"{max(1, repeats)} fold {result_fold + 1}/{folds}",
                    flush=True,
                )
    else:
        for task, repeat, fold, train_index, test_index, weight in tasks:
            print(
                f"[{dataset.name}] repeat {repeat + 1}/{max(1, repeats)} "
                f"fold {fold + 1}/{folds}: "
                f"train={len(train_index)} test={len(test_index)}",
                flush=True,
            )
            result = _train_fold(
                dataset,
                fold,
                train_index,
                test_index,
                epochs,
                seed + repeat * 10_000,
                models,
                lambda_monotone,
                lambda_smooth,
                lambda_grid,
                lambda_smooth_ratio,
                model_seeds,
                nested_inner_folds,
                stacking_c,
                stacking_members,
                xgb_candidates,
                sequence_hyperparameters,
                weight,
                epoch_overrides,
                lambda_overrides,
                smooth_overrides,
            )
            (
                result_fold,
                result_predictions,
                result_epochs,
                result_lambdas,
                result_hyperparameters,
            ) = result
            fold_results.append(
                (
                    repeat,
                    result_fold,
                    result_predictions,
                    result_epochs,
                    result_lambdas,
                    result_hyperparameters,
                )
            )
            print(f"  completed fold {fold + 1}", flush=True)
    # Parallel workers complete out of order. Sort before extending the
    # per-model epoch and lambda records so manifest entries map exactly to
    # repeat * folds + fold, as downstream trajectory and audit code expects.
    fold_results.sort(key=lambda result: (result[0], result[1]))
    selected_hyperparameters: dict[str, list[dict]] = {}
    for (
        repeat,
        fold,
        fold_predictions,
        fold_epochs,
        fold_lambdas,
        fold_hyperparameters,
    ) in fold_results:
        for record in fold_hyperparameters:
            enriched = copy.deepcopy(record)
            enriched["repeat"] = int(repeat)
            enriched["fold"] = int(fold)
            selected_hyperparameters.setdefault(record["model"], []).append(
                enriched
            )
        test_index = splits[repeat * folds + fold][1]
        repeat_weights[repeat][test_index] = fold_weights[
            repeat * folds + fold
        ][test_index]
        for name in models:
            predictions[name] += fold_predictions[name]
            repeat_predictions[repeat][name][test_index] = fold_predictions[name][
                test_index
            ]
            selected_epochs[name].extend(fold_epochs[name])
            selected_lambdas[name].extend(fold_lambdas[name])
        print(
            f"[{dataset.name}] assembled repeat {repeat + 1}/"
            f"{max(1, repeats)} fold {fold + 1}/{folds}",
            flush=True,
        )
    for name in models:
        predictions[name] /= max(1, repeats)
    if repeat_prefix:
        repeat_path = RESULTS / "predictions" / f"{repeat_prefix}_repeats.npz"
        repeat_payload = {
            "labels": dataset.labels,
            "stay_id": dataset.cohort["stay_id"].to_numpy(),
            "subject_key": dataset.cohort["subject_key"].astype(str).to_numpy(),
        }
        for repeat, model_predictions in repeat_predictions.items():
            for name, values in model_predictions.items():
                repeat_payload[f"repeat{repeat}_{name}"] = values
            repeat_payload[f"repeat{repeat}_analysis_weight"] = repeat_weights[
                repeat
            ]
        np.savez_compressed(repeat_path, **repeat_payload)
    return (
        predictions,
        splits,
        selected_epochs,
        selected_lambdas,
        selected_hyperparameters,
    )


def _validated_metric_weight(
    sample_weight: np.ndarray | None,
    n_observations: int,
) -> np.ndarray | None:
    """Return normalized unequal weights, or None for the legacy equal-weight path."""
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_observations,):
        raise ValueError("Metric sample_weight must be observation-aligned")
    if not np.isfinite(weights).all():
        raise ValueError("Metric sample_weight must be finite")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Metric sample_weight must be non-negative with positive sum")
    if np.all(weights == weights[0]):
        return None
    return weights / weights.mean()


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if sample_weight is None:
        return float(np.mean(values))
    if np.sum(sample_weight) <= 0:
        return np.nan
    return float(np.average(values, weights=sample_weight))


def _weighted_rate(
    indicator: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    indicator = np.asarray(indicator, dtype=np.float64)
    if sample_weight is None:
        return float(np.mean(indicator))
    if np.sum(sample_weight) <= 0:
        return np.nan
    return float(np.average(indicator, weights=sample_weight))


def multiclass_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return macro one-vs-rest AUC only when all three grades are present."""
    weights = _validated_metric_weight(sample_weight, len(labels))
    if weights is None:
        if np.unique(labels).size < N_CLASSES:
            return np.nan
    else:
        class_weight = np.bincount(labels, weights=weights, minlength=N_CLASSES)
        if np.count_nonzero(class_weight) < N_CLASSES:
            return np.nan
    values = [
        roc_auc_score(
            (labels == klass).astype(int),
            probabilities[:, klass],
            sample_weight=weights,
        )
        for klass in range(N_CLASSES)
    ]
    return float(np.mean(values))


def cumulative_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: int,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(labels))
    target = (labels >= threshold).astype(int)
    if weights is None:
        if target.min() == target.max():
            return np.nan
    else:
        weighted_target = np.bincount(target, weights=weights, minlength=2)
        if np.count_nonzero(weighted_target) < 2:
            return np.nan
    prediction = probabilities[:, threshold:].sum(axis=1)
    return float(
        roc_auc_score(target, prediction, sample_weight=weights)
    )


def ordinal_median_prediction(probabilities: np.ndarray) -> np.ndarray:
    """Return the median of the predicted ordinal distribution.

    The cumulative-logit head can assign low probability to the middle class
    even when the latent ordinal distribution has grade 1 as its median. The
    mode (argmax) then collapses to the adjacent classes. The median decision
    avoids that structural artefact and minimizes expected absolute ordinal
    error; probability ranking and calibration metrics are unchanged.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    at_least_one = probabilities[:, 1:].sum(axis=1)
    at_least_two = probabilities[:, 2]
    return (at_least_one > 0.5).astype(int) + (
        at_least_two > 0.5
    ).astype(int)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(labels))
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if weights is None and mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        elif weights is not None and weights[mask].sum() > 0:
            bin_weight = weights[mask].sum() / weights.sum()
            ece += bin_weight * abs(
                np.average(correct[mask], weights=weights[mask])
                - np.average(confidence[mask], weights=weights[mask])
            )
    return float(ece)


def binary_expected_calibration_error(
    target: np.ndarray,
    predicted: np.ndarray,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(target))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted >= low) & (
            predicted < high if high < 1 else predicted <= high
        )
        if weights is None and mask.any():
            ece += mask.mean() * abs(target[mask].mean() - predicted[mask].mean())
        elif weights is not None and weights[mask].sum() > 0:
            bin_weight = weights[mask].sum() / weights.sum()
            ece += bin_weight * abs(
                np.average(target[mask], weights=weights[mask])
                - np.average(predicted[mask], weights=weights[mask])
            )
    return float(ece)


def calibration_slope_intercept(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: int = 1,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    weights = _validated_metric_weight(sample_weight, len(labels))
    target = (labels >= threshold).astype(int)
    predicted = probabilities[:, threshold:].sum(axis=1)
    predicted = np.clip(predicted, 1e-5, 1 - 1e-5)
    if weights is None:
        if target.min() == target.max():
            return np.nan, np.nan
    else:
        weighted_target = np.bincount(target, weights=weights, minlength=2)
        if np.count_nonzero(weighted_target) < 2:
            return np.nan, np.nan
    logit = np.log(predicted / (1 - predicted)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit, target, sample_weight=weights)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def calibration_slope_intercept_grade(
    labels: np.ndarray,
    probabilities: np.ndarray,
    grade: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit calibration of one-vs-rest probability for an ordinal grade."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if grade not in range(N_CLASSES):
        raise ValueError(f"Grade must be in 0..{N_CLASSES - 1}")
    weights = _validated_metric_weight(sample_weight, len(labels))
    target = (labels == grade).astype(int)
    if weights is None:
        if target.min() == target.max():
            return np.nan, np.nan
    else:
        weighted_target = np.bincount(target, weights=weights, minlength=2)
        if np.count_nonzero(weighted_target) < 2:
            return np.nan, np.nan
    predicted = np.clip(probabilities[:, grade], 1e-5, 1 - 1e-5)
    logit = np.log(predicted / (1 - predicted)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit, target, sample_weight=weights)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def decision_curve_net_benefit(
    labels: np.ndarray,
    predicted: np.ndarray,
    thresholds: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Binary net benefit across prespecified clinically relevant thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.40, 7)
    weights = _validated_metric_weight(sample_weight, len(labels))
    denominator = float(len(labels)) if weights is None else float(weights.sum())
    prevalence = (
        float(np.mean(labels))
        if weights is None
        else float(np.average(labels, weights=weights))
    )
    model_benefit = []
    treat_all = []
    for threshold in thresholds:
        positive = predicted >= threshold
        if weights is None:
            true_positive = float(np.sum(positive & (labels == 1)))
            false_positive = float(np.sum(positive & (labels == 0)))
        else:
            true_positive = float(
                np.sum(weights * positive * (labels == 1))
            )
            false_positive = float(
                np.sum(weights * positive * (labels == 0))
            )
        weight = threshold / (1.0 - threshold)
        model_benefit.append(
            true_positive / denominator
            - false_positive / denominator * weight
        )
        treat_all.append(prevalence - (1.0 - prevalence) * weight)
    return {
        "thresholds": [float(value) for value in thresholds],
        "model": model_benefit,
        "treat_all": treat_all,
        "treat_none": [0.0 for _ in thresholds],
    }


def continuous_nri(
    labels: np.ndarray,
    new_probability: np.ndarray,
    old_probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(labels))
    events = labels == 1
    non_events = ~events
    event_gain = _weighted_mean(
        new_probability[events] - old_probability[events],
        None if weights is None else weights[events],
    )
    non_event_gain = _weighted_mean(
        old_probability[non_events] - new_probability[non_events],
        None if weights is None else weights[non_events],
    )
    return event_gain + non_event_gain


def categorical_nri(
    labels: np.ndarray,
    new_probability: np.ndarray,
    old_probability: np.ndarray,
    threshold: float = 0.2,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(labels))
    events = labels == 1
    non_events = ~events
    new_up = new_probability >= threshold
    old_up = old_probability >= threshold
    event_nri = (
        _weighted_rate(new_up[events], None if weights is None else weights[events])
        - _weighted_rate(
            old_up[events], None if weights is None else weights[events]
        )
    )
    non_event_nri = (
        _weighted_rate(
            ~new_up[non_events], None if weights is None else weights[non_events]
        )
        - _weighted_rate(
            ~old_up[non_events], None if weights is None else weights[non_events]
        )
    )
    return float(event_nri + non_event_nri)


def integrated_discrimination_improvement(
    labels: np.ndarray,
    new_probability: np.ndarray,
    old_probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = _validated_metric_weight(sample_weight, len(labels))
    events = labels == 1
    non_events = ~events
    if not events.any() or not non_events.any():
        return np.nan
    event_weights = None if weights is None else weights[events]
    non_event_weights = None if weights is None else weights[non_events]
    new_slope = _weighted_mean(
        new_probability[events], event_weights
    ) - _weighted_mean(
        new_probability[non_events], non_event_weights
    )
    old_slope = _weighted_mean(
        old_probability[events], event_weights
    ) - _weighted_mean(
        old_probability[non_events], non_event_weights
    )
    return float(new_slope - old_slope)


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1
        start = stop
    output = np.empty(len(values), dtype=float)
    output[order] = ranks
    return output


def paired_delong(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
) -> tuple[float, float, float]:
    """Paired DeLong test for two binary ROC curves on identical observations."""
    order = np.argsort(-labels.astype(int), kind="stable")
    predictions = np.vstack([primary, comparator])[:, order]
    labels = labels[order].astype(int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return np.nan, np.nan, np.nan
    tx = np.empty((2, positives))
    ty = np.empty((2, negatives))
    tz = np.empty((2, len(labels)))
    for row in range(2):
        tx[row] = _midrank(predictions[row, :positives])
        ty[row] = _midrank(predictions[row, positives:])
        tz[row] = _midrank(predictions[row])
    aucs = tz[:, :positives].sum(axis=1) / positives / negatives - (
        positives + 1.0
    ) / 2.0 / negatives
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    covariance = np.cov(v01) / positives + np.cov(v10) / negatives
    variance = (
        covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    )
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        return difference, 0.0, 1.0
    z_value = difference / np.sqrt(variance)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_value))))
    return difference, float(z_value), p_value


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        current = (len(values) - rank) * values[index]
        running = max(running, current)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def comparison_metrics(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Paired discrimination, reclassification, and DeLong comparisons."""
    weights = _validated_metric_weight(sample_weight, len(labels))
    y_any = (labels >= 1).astype(int)
    y_severe = (labels >= 2).astype(int)
    p_any_new = primary[:, 1:].sum(axis=1)
    p_any_old = comparator[:, 1:].sum(axis=1)
    p_severe_new = primary[:, 2]
    p_severe_old = comparator[:, 2]
    if weights is None:
        delong_any = paired_delong(y_any, p_any_new, p_any_old)
        delong_severe = paired_delong(y_severe, p_severe_new, p_severe_old)
    else:
        invalid = (np.nan, np.nan, np.nan)
        delong_any = invalid
        delong_severe = invalid
    return {
        "delta_macro_auc": multiclass_auc(labels, primary, weights)
        - multiclass_auc(labels, comparator, weights),
        "delta_auc_any": cumulative_auc(labels, primary, 1, weights)
        - cumulative_auc(labels, comparator, 1, weights),
        "delta_auc_severe": cumulative_auc(labels, primary, 2, weights)
        - cumulative_auc(labels, comparator, 2, weights),
        "nri_any_continuous": continuous_nri(
            y_any, p_any_new, p_any_old, weights
        ),
        "nri_severe_continuous": continuous_nri(
            y_severe, p_severe_new, p_severe_old, weights
        ),
        "nri_any_categorical": categorical_nri(
            y_any, p_any_new, p_any_old, sample_weight=weights
        ),
        "nri_severe_categorical": categorical_nri(
            y_severe, p_severe_new, p_severe_old, sample_weight=weights
        ),
        "idi_any": integrated_discrimination_improvement(
            y_any, p_any_new, p_any_old, weights
        ),
        "idi_severe": integrated_discrimination_improvement(
            y_severe, p_severe_new, p_severe_old, weights
        ),
        "delong_any_delta": delong_any[0],
        "delong_any_z": delong_any[1],
        "delong_any_p": delong_any[2],
        "delong_severe_delta": delong_severe[0],
        "delong_severe_z": delong_severe[1],
        "delong_severe_p": delong_severe[2],
        "delong_inference_status": (
            "unclustered_standard_delong_secondary; "
            "group_bootstrap_is_primary"
            if weights is None
            else "not_applicable_for_weighted_analysis_use_group_bootstrap"
        ),
    }


def comparison_metrics_repeated(
    labels: np.ndarray,
    primary: list[np.ndarray],
    comparator: list[np.ndarray],
    sample_weight: np.ndarray | list[np.ndarray] | None = None,
) -> dict[str, float | str]:
    """Aggregate paired metrics over repeat-specific predictions and weights."""
    if len(primary) != len(comparator):
        raise ValueError("Repeated primary and comparator predictions differ in length")
    n_repeats = len(primary)
    if sample_weight is None:
        weights: list[np.ndarray | None] = [None] * n_repeats
    elif isinstance(sample_weight, list):
        if len(sample_weight) != n_repeats:
            raise ValueError("Repeated sample weights are not repeat-aligned")
        weights = list(sample_weight)
    else:
        weights = [sample_weight] * n_repeats
    rows = [
        comparison_metrics(
            labels,
            primary[index],
            comparator[index],
            sample_weight=weights[index],
        )
        for index in range(n_repeats)
    ]
    output: dict[str, float | str] = {}
    for key in rows[0]:
        if isinstance(rows[0][key], str):
            output[key] = str(rows[0][key])
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        output[key] = float(finite.mean()) if finite.size else float("nan")

    # Standard DeLong treats observations as independent and is retained only
    # as a secondary first-partition check. The repeated-CV inference of record
    # is the patient- or hospital-clustered paired bootstrap.
    for endpoint in ("any", "severe"):
        for suffix in ("z", "p"):
            output[f"delong_{endpoint}_{suffix}"] = float(
                rows[0][f"delong_{endpoint}_{suffix}"]
            )
    if all(weight is None for weight in weights):
        output["delong_inference_status"] = (
            "unclustered_standard_delong_first_cv_partition_secondary"
        )
    else:
        output["delong_inference_status"] = (
            "not_applicable_for_weighted_analysis_use_group_bootstrap"
        )
    return output


def add_holm_comparisons(output: dict, primary_name: str = "dggru_tr") -> None:
    """Apply a fixed outline-wide Holm family to secondary DeLong tests.

    The family does not change when a command trains only a subset of models.
    Missing or non-finite secondary tests are conservatively ranked as p=1 so a
    partial run cannot appear more significant merely because fewer tests ran.
    """
    comparisons = output.get("comparisons_vs_primary")
    if not comparisons:
        return
    family_keys = [
        (name, endpoint)
        for name in OUTLINE_HOLM_COMPARATORS
        for endpoint in OUTLINE_HOLM_ENDPOINTS
    ]
    family_p = np.ones(len(family_keys), dtype=float)
    reportable: list[int] = []
    for index, (name, endpoint) in enumerate(family_keys):
        comparison = comparisons.get(name)
        if not isinstance(comparison, dict):
            continue
        value = comparison.get(f"delong_{endpoint}_p")
        if isinstance(value, (int, float)) and np.isfinite(value):
            family_p[index] = float(value)
            reportable.append(index)
    adjusted_family = np.asarray(holm_adjust(family_p.tolist()), dtype=float)
    for index in reportable:
        name, endpoint = family_keys[index]
        comparisons[name][f"delong_{endpoint}_p_holm"] = float(
            adjusted_family[index]
        )
    for name, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        for endpoint in OUTLINE_HOLM_ENDPOINTS:
            comparison.setdefault(f"delong_{endpoint}_p_holm", float("nan"))
        comparison["delong_holm_family"] = (
            "fixed_five_outline_baselines_x_any_severe"
        )
        comparison["inference_role"] = (
            "primary_holm_family"
            if name in OUTLINE_HOLM_COMPARATORS
            else "exploratory_not_in_primary_holm_family"
        )


def point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = _validated_metric_weight(sample_weight, len(labels))
    if weights is not None and np.any(weights == 0):
        observed = weights > 0
        labels = labels[observed]
        probabilities = probabilities[observed]
        weights = _validated_metric_weight(weights[observed], len(labels))
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    prediction = ordinal_median_prediction(probabilities)
    mode_prediction = probabilities.argmax(axis=1)
    slope, intercept = calibration_slope_intercept(
        labels, probabilities, sample_weight=weights
    )
    slope_severe, intercept_severe = calibration_slope_intercept(
        labels, probabilities, threshold=2, sample_weight=weights
    )
    grade_calibration = {
        grade: calibration_slope_intercept_grade(
            labels,
            probabilities,
            grade,
            sample_weight=weights,
        )
        for grade in range(N_CLASSES)
    }
    y_any = (labels >= 1).astype(int)
    y_severe = (labels >= 2).astype(int)
    p_any = probabilities[:, 1:].sum(axis=1)
    p_severe = probabilities[:, 2]
    class_auc = {
        f"auc_grade_{grade}": (
            float(
                roc_auc_score(
                    (labels == grade).astype(int),
                    probabilities[:, grade],
                    sample_weight=weights,
                )
            )
            if (
                0
                < (
                    int((labels == grade).sum())
                    if weights is None
                    else float(weights[labels == grade].sum())
                )
                < (
                    len(labels)
                    if weights is None
                    else float(weights.sum())
                )
            )
            else np.nan
        )
        for grade in range(N_CLASSES)
    }
    prediction_counts = {
        f"predicted_grade_{grade}": (
            int((prediction == grade).sum())
            if weights is None
            else float(weights[prediction == grade].sum())
        )
        for grade in range(N_CLASSES)
    }
    absolute_error = np.abs(prediction - labels)
    confusion = confusion_matrix(
        labels,
        prediction,
        labels=list(range(N_CLASSES)),
        sample_weight=weights,
    )
    mode_confusion = confusion_matrix(
        labels,
        mode_prediction,
        labels=list(range(N_CLASSES)),
        sample_weight=weights,
    )
    return {
        "n": float(len(labels)),
        "macro_auc": multiclass_auc(labels, probabilities, weights),
        **class_auc,
        **prediction_counts,
        "confusion_matrix": confusion.tolist(),
        "mode_predicted_grade_0": (
            int((mode_prediction == 0).sum())
            if weights is None
            else float(weights[mode_prediction == 0].sum())
        ),
        "mode_predicted_grade_1": (
            int((mode_prediction == 1).sum())
            if weights is None
            else float(weights[mode_prediction == 1].sum())
        ),
        "mode_predicted_grade_2": (
            int((mode_prediction == 2).sum())
            if weights is None
            else float(weights[mode_prediction == 2].sum())
        ),
        "mode_confusion_matrix": mode_confusion.tolist(),
        "auc_any_deterioration": cumulative_auc(
            labels, probabilities, 1, weights
        ),
        "auc_severe_deterioration": cumulative_auc(
            labels, probabilities, 2, weights
        ),
        "accuracy": float(
            accuracy_score(labels, prediction, sample_weight=weights)
        ),
        "adjacent_class_misclassification_rate": _weighted_mean(
            absolute_error == 1, weights
        ),
        "non_adjacent_misclassification_rate": _weighted_mean(
            absolute_error > 1, weights
        ),
        "mean_absolute_ordinal_error": _weighted_mean(absolute_error, weights),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                labels,
                prediction,
                weights="quadratic",
                sample_weight=weights,
            )
        ),
        "mode_accuracy": float(
            accuracy_score(labels, mode_prediction, sample_weight=weights)
        ),
        "mode_quadratic_weighted_kappa": float(
            cohen_kappa_score(
                labels,
                mode_prediction,
                weights="quadratic",
                sample_weight=weights,
            )
        ),
        "multiclass_brier": _weighted_mean(
            np.sum((probabilities - np.eye(3)[labels]) ** 2, axis=1),
            weights,
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1, 2],
                sample_weight=weights,
            )
        ),
        "brier_any_deterioration": float(
            brier_score_loss(
                y_any, p_any, sample_weight=weights
            )
        ),
        "brier_severe_deterioration": float(
            brier_score_loss(
                y_severe, p_severe, sample_weight=weights
            )
        ),
        "calibration_slope_any": slope,
        "calibration_intercept_any": intercept,
        "calibration_slope_severe": slope_severe,
        "calibration_intercept_severe": intercept_severe,
        **{
            f"calibration_slope_grade_{grade}": grade_calibration[grade][0]
            for grade in range(N_CLASSES)
        },
        **{
            f"calibration_intercept_grade_{grade}": grade_calibration[grade][1]
            for grade in range(N_CLASSES)
        },
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, sample_weight=weights
        ),
        "ece_any_deterioration": binary_expected_calibration_error(
            y_any, p_any, sample_weight=weights
        ),
        "ece_severe_deterioration": binary_expected_calibration_error(
            y_severe, p_severe, sample_weight=weights
        ),
        "dca_any": decision_curve_net_benefit(
            y_any, p_any, sample_weight=weights
        ),
        "dca_severe": decision_curve_net_benefit(
            y_severe, p_severe, sample_weight=weights
        ),
    }


def _draw_cluster_bootstrap_index(
    rng: np.random.Generator,
    labels: np.ndarray,
    unique_groups: np.ndarray,
    group_indices: dict,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Draw clustered resamples that retain every ordinal class."""
    for _ in range(100_000):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        index = np.concatenate([group_indices[group] for group in sampled_groups])
        effective_labels = labels[index]
        if sample_weight is not None:
            effective_labels = effective_labels[sample_weight[index] > 0]
        if np.unique(effective_labels).size == N_CLASSES:
            return index
    raise RuntimeError(
        "Unable to draw a cluster-bootstrap replicate containing all three "
        "ordinal outcome grades"
    )


def group_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
    validity_out: dict | None = None,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    weights = _validated_metric_weight(sample_weight, len(labels))
    if weights is None:
        unique_groups = np.unique(groups)
    else:
        group_weight = {
            group: float(weights[groups == group].sum())
            for group in np.unique(groups)
        }
        unique_groups = np.asarray(
            [group for group, total in group_weight.items() if total > 0]
        )
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    samples: dict[str, list[float]] = {}
    for _ in range(replicates):
        index = _draw_cluster_bootstrap_index(
            rng,
            labels,
            unique_groups,
            group_indices,
            sample_weight=weights,
        )
        metrics = point_metrics(
            labels[index],
            probabilities[index],
            sample_weight=None if weights is None else weights[index],
        )
        for key, value in metrics.items():
            if (
                key == "n"
                or key.startswith("predicted_grade_")
                or key.startswith("mode_predicted_grade_")
                or key == "confusion_matrix"
                or key == "mode_confusion_matrix"
            ):
                continue
            if np.isscalar(value) and np.isfinite(value):
                samples.setdefault(key, []).append(float(value))
    intervals = {
        key: [
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        ]
        for key, values in samples.items()
        if values
    }
    if validity_out is not None:
        metric_counts = {
            key: int(len(values)) for key, values in samples.items() if values
        }
        validity_out.update(
            {
                "requested_replicates": int(replicates),
                "clusters": int(len(unique_groups)),
                "metric_valid_replicates": metric_counts,
                "minimum_metric_valid_replicates": (
                    min(metric_counts.values()) if metric_counts else 0
                ),
                "all_requested_replicates_valid": bool(
                    metric_counts
                    and min(metric_counts.values()) == int(replicates)
                ),
            }
        )
    return intervals


def paired_auc_difference(
    labels: np.ndarray,
    primary: np.ndarray,
    comparator: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
    validity_out: dict | None = None,
) -> list[float]:
    rng = np.random.default_rng(seed)
    weights = _validated_metric_weight(sample_weight, len(labels))
    if weights is None:
        unique_groups = np.unique(groups)
    else:
        group_weight = {
            group: float(weights[groups == group].sum())
            for group in np.unique(groups)
        }
        unique_groups = np.asarray(
            [group for group, total in group_weight.items() if total > 0]
        )
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    differences = []
    for _ in range(replicates):
        index = _draw_cluster_bootstrap_index(
            rng,
            labels,
            unique_groups,
            group_indices,
            sample_weight=weights,
        )
        value = multiclass_auc(
            labels[index],
            primary[index],
            None if weights is None else weights[index],
        ) - multiclass_auc(
            labels[index],
            comparator[index],
            None if weights is None else weights[index],
        )
        if np.isfinite(value):
            differences.append(float(value))
    if not differences:
        if validity_out is not None:
            validity_out.update(
                {
                    "requested_replicates": int(replicates),
                    "valid_replicates": 0,
                    "clusters": int(len(unique_groups)),
                    "all_requested_replicates_valid": False,
                }
            )
        return [np.nan, np.nan]
    if validity_out is not None:
        validity_out.update(
            {
                "requested_replicates": int(replicates),
                "valid_replicates": int(len(differences)),
                "clusters": int(len(unique_groups)),
                "all_requested_replicates_valid": bool(
                    len(differences) == int(replicates)
                ),
            }
        )
    return [
        float(np.nanpercentile(differences, 2.5)),
        float(np.nanpercentile(differences, 97.5)),
    ]


def repeated_cv_variability(
    dataset: Dataset,
    repeat_path: Path,
    models: tuple[str, ...],
    repeats: int,
) -> dict[str, dict]:
    """Summarize discrimination across repeated cross-validation partitions."""
    output: dict[str, dict] = {}
    fallback_weights = _validated_metric_weight(
        dataset.analysis_weight, len(dataset.labels)
    )
    with np.load(repeat_path, allow_pickle=True) as data:
        repeated_weights: list[np.ndarray | None] = []
        for repeat in range(max(1, repeats)):
            key = f"repeat{repeat}_analysis_weight"
            repeated_weights.append(
                _validated_metric_weight(
                    data[key].astype(np.float64),
                    len(dataset.labels),
                )
                if key in data
                else fallback_weights
            )
        per_model_auc = {name: [] for name in models}
        per_model_any = {name: [] for name in models}
        for repeat in range(max(1, repeats)):
            for name in models:
                probabilities = data[f"repeat{repeat}_{name}"]
                values = point_metrics(
                    dataset.labels,
                    probabilities,
                    sample_weight=repeated_weights[repeat],
                )
                per_model_auc[name].append(float(values["macro_auc"]))
                per_model_any[name].append(
                    float(values["auc_any_deterioration"])
                )
    for name in models:
        for metric, values in (
            ("macro_auc", per_model_auc[name]),
            ("auc_any_deterioration", per_model_any[name]),
        ):
            output.setdefault(name, {})[metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    if "dggru_tr" in models:
        comparisons = {}
        for comparator in models:
            if comparator == "dggru_tr":
                continue
            differences = [
                tr - other
                for tr, other in zip(
                    per_model_auc["dggru_tr"], per_model_auc[comparator]
                )
            ]
            comparisons[comparator] = {
                "values": differences,
                "mean": float(np.mean(differences)),
                "sd": (
                    float(np.std(differences, ddof=1))
                    if len(differences) > 1
                    else 0.0
                ),
                "min": float(np.min(differences)),
                "max": float(np.max(differences)),
            }
        output["paired_macro_auc_differences_vs_dggru_tr"] = comparisons
    return output


def aggregate_point_metrics(metric_rows: list[dict]) -> dict:
    """Average scalar point metrics across repeated cross-validation runs."""
    if not metric_rows:
        raise ValueError("At least one repeated-CV metric row is required")
    keys = set(metric_rows[0])
    if any(set(row) != keys for row in metric_rows[1:]):
        raise ValueError("Repeated-CV metric rows must have identical keys")
    output = {"n": float(metric_rows[0]["n"])}
    for key in metric_rows[0]:
        if key == "n":
            continue
        values = [row[key] for row in metric_rows]
        if key.endswith("_matrix"):
            output[key] = np.mean(np.asarray(values, dtype=float), axis=0).tolist()
            continue
        if isinstance(values[0], dict):
            output[key] = {
                subkey: np.mean(
                    np.asarray([value[subkey] for value in values], dtype=float),
                    axis=0,
                ).tolist()
                for subkey in values[0]
            }
            continue
        numeric = np.asarray(values, dtype=float)
        finite = numeric[np.isfinite(numeric)]
        output[key] = float(finite.mean()) if finite.size else np.nan
    return output


def group_bootstrap_repeated(
    labels: np.ndarray,
    repeated_probabilities: list[np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
    sample_weight: np.ndarray | list[np.ndarray] | None = None,
    validity_out: dict | None = None,
) -> dict[str, list[float]]:
    """Bootstrap the mean metric over repeated-CV partitions."""
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    if sample_weight is None:
        repeated_weights = [None] * len(repeated_probabilities)
    elif isinstance(sample_weight, list):
        if len(sample_weight) != len(repeated_probabilities):
            raise ValueError("Repeated sample weights are not repeat-aligned")
        repeated_weights = [
            _validated_metric_weight(weight, len(labels))
            for weight in sample_weight
        ]
    else:
        normalized = _validated_metric_weight(sample_weight, len(labels))
        repeated_weights = [normalized] * len(repeated_probabilities)
    if all(weight is None for weight in repeated_weights):
        unique_groups = np.unique(groups)
    else:
        group_weight = {
            group: sum(
                float(weight[groups == group].sum())
                for weight in repeated_weights
                if weight is not None
            )
            for group in np.unique(groups)
        }
        unique_groups = np.asarray(
            [group for group, total in group_weight.items() if total > 0]
        )
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    effective_weight = (
        None
        if all(weight is None for weight in repeated_weights)
        else np.mean(
            np.stack(
                [weight for weight in repeated_weights if weight is not None]
            ),
            axis=0,
        )
    )
    samples: dict[str, list[float]] = {}
    for _ in range(replicates):
        index = _draw_cluster_bootstrap_index(
            rng,
            labels,
            unique_groups,
            group_indices,
            sample_weight=effective_weight,
        )
        rows = [
            point_metrics(
                labels[index],
                probabilities[index],
                sample_weight=(
                    None
                    if weight is None
                    else weight[index]
                ),
            )
            for probabilities, weight in zip(
                repeated_probabilities,
                repeated_weights,
            )
        ]
        metrics = aggregate_point_metrics(rows)
        for key, value in metrics.items():
            if (
                key == "n"
                or key.startswith("predicted_grade_")
                or key.startswith("mode_predicted_grade_")
                or key.endswith("_matrix")
            ):
                continue
            if np.isscalar(value) and np.isfinite(value):
                samples.setdefault(key, []).append(float(value))
    intervals = {
        key: [
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        ]
        for key, values in samples.items()
        if values
    }
    if validity_out is not None:
        metric_counts = {
            key: int(len(values)) for key, values in samples.items() if values
        }
        validity_out.update(
            {
                "requested_replicates": int(replicates),
                "clusters": int(len(unique_groups)),
                "metric_valid_replicates": metric_counts,
                "minimum_metric_valid_replicates": (
                    min(metric_counts.values()) if metric_counts else 0
                ),
                "all_requested_replicates_valid": bool(
                    metric_counts
                    and min(metric_counts.values()) == int(replicates)
                ),
            }
        )
    return intervals


def paired_auc_difference_repeated(
    labels: np.ndarray,
    primary: list[np.ndarray],
    comparator: list[np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
    sample_weight: np.ndarray | list[np.ndarray] | None = None,
    validity_out: dict | None = None,
) -> list[float]:
    """Bootstrap the mean paired AUC difference across CV repeats."""
    if replicates <= 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    n_repeats = len(primary)
    if len(comparator) != n_repeats:
        raise ValueError("Repeated primary and comparator predictions differ in length")
    if sample_weight is None:
        repeated_weights: list[np.ndarray | None] = [None] * n_repeats
    elif isinstance(sample_weight, list):
        if len(sample_weight) != n_repeats:
            raise ValueError("Repeated sample weights are not repeat-aligned")
        repeated_weights = [
            _validated_metric_weight(weight, len(labels))
            for weight in sample_weight
        ]
    else:
        weights = _validated_metric_weight(sample_weight, len(labels))
        repeated_weights = [weights] * n_repeats
    if all(weight is None for weight in repeated_weights):
        unique_groups = np.unique(groups)
    else:
        group_weight = {
            group: float(
                sum(
                    weight[groups == group].sum()
                    for weight in repeated_weights
                    if weight is not None
                )
            )
            for group in np.unique(groups)
        }
        unique_groups = np.asarray(
            [group for group, total in group_weight.items() if total > 0]
        )
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    effective_weight = (
        None
        if all(weight is None for weight in repeated_weights)
        else np.mean(
            np.stack(
                [weight for weight in repeated_weights if weight is not None]
            ),
            axis=0,
        )
    )
    differences = []
    for _ in range(replicates):
        index = _draw_cluster_bootstrap_index(
            rng,
            labels,
            unique_groups,
            group_indices,
            sample_weight=effective_weight,
        )
        repeat_differences = [
            multiclass_auc(
                labels[index],
                primary_repeat[index],
                (
                    None
                    if repeat_weight is None
                    else repeat_weight[index]
                ),
            )
            - multiclass_auc(
                labels[index],
                comparator_repeat[index],
                (
                    None
                    if repeat_weight is None
                    else repeat_weight[index]
                ),
            )
            for primary_repeat, comparator_repeat, repeat_weight in zip(
                primary,
                comparator,
                repeated_weights,
            )
        ]
        finite = [value for value in repeat_differences if np.isfinite(value)]
        if finite:
            differences.append(float(np.mean(finite)))
    if not differences:
        if validity_out is not None:
            validity_out.update(
                {
                    "requested_replicates": int(replicates),
                    "valid_replicates": 0,
                    "clusters": int(len(unique_groups)),
                    "all_requested_replicates_valid": False,
                }
            )
        return [np.nan, np.nan]
    if validity_out is not None:
        validity_out.update(
            {
                "requested_replicates": int(replicates),
                "valid_replicates": int(len(differences)),
                "clusters": int(len(unique_groups)),
                "all_requested_replicates_valid": bool(
                    len(differences) == int(replicates)
                ),
            }
        )
    return [
        float(np.nanpercentile(differences, 2.5)),
        float(np.nanpercentile(differences, 97.5)),
    ]


def _evaluate_repeated_model(
    name: str,
    labels: np.ndarray,
    repeated_probabilities: list[np.ndarray],
    groups: np.ndarray,
    repeated_sample_weight: list[np.ndarray | None],
    replicates: int,
    seed: int,
) -> tuple[str, dict]:
    rows = [
        point_metrics(labels, probabilities, sample_weight=sample_weight)
        for probabilities, sample_weight in zip(
            repeated_probabilities,
            repeated_sample_weight,
        )
    ]
    ci_validity: dict = {}
    intervals = group_bootstrap_repeated(
        labels,
        repeated_probabilities,
        groups,
        replicates=replicates,
        seed=seed,
        sample_weight=repeated_sample_weight,
        validity_out=ci_validity,
    )
    return name, {
        "point": aggregate_point_metrics(rows),
        "ci": intervals,
        "ci_validity": ci_validity,
        "repeat_metrics": {
            key: [float(row[key]) for row in rows]
            for key in rows[0]
            if key not in {"n", "confusion_matrix", "mode_confusion_matrix"}
            and np.isscalar(rows[0][key])
        },
    }


def evaluate_repeated_cv(
    dataset: Dataset,
    repeat_path: Path,
    models: tuple[str, ...],
    repeats: int,
    replicates: int,
    seed: int,
    metric_workers: int = 1,
) -> dict:
    """Evaluate repeated CV using the mean of repeat-specific OOF metrics."""
    weights = _validated_metric_weight(
        dataset.analysis_weight, len(dataset.labels)
    )
    with np.load(repeat_path, allow_pickle=True) as data:
        repeated = {
            model: [
                data[f"repeat{repeat}_{model}"].astype(np.float64)
                for repeat in range(repeats)
            ]
            for model in models
        }
        repeated_weights: list[np.ndarray | None] = []
        for repeat in range(repeats):
            key = f"repeat{repeat}_analysis_weight"
            if key in data:
                repeated_weights.append(
                    _validated_metric_weight(
                        data[key].astype(np.float64),
                        len(dataset.labels),
                    )
                )
            else:
                repeated_weights.append(weights)
        labels = data["labels"].astype(int)
        stay_id = data["stay_id"].astype(np.int64)
        subject_key = data["subject_key"].astype(str)
    expected_stay_id = dataset.cohort["stay_id"].astype(np.int64).to_numpy()
    expected_subject_key = dataset.cohort["subject_key"].astype(str).to_numpy()
    if not np.array_equal(stay_id, expected_stay_id):
        raise ValueError("Repeated-CV predictions are not aligned to cohort stay_id")
    if not np.array_equal(subject_key, expected_subject_key):
        raise ValueError("Repeated-CV predictions are not aligned to cohort subject_key")
    if not np.array_equal(labels, dataset.labels.astype(int)):
        raise ValueError("Repeated-CV predictions are not aligned to cohort labels")
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    output = {}
    tasks = [
        (
            model,
            labels,
            repeated[model],
            groups,
            repeated_weights,
            replicates,
            seed,
        )
        for model in models
    ]
    if metric_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=min(metric_workers, len(tasks))
        ) as executor:
            futures = {
                executor.submit(_evaluate_repeated_model, *task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                name, result = future.result()
                output[name] = result
                print(f"[metrics] development repeated-CV {name}", flush=True)
    else:
        for task in tasks:
            name, result = _evaluate_repeated_model(*task)
            output[name] = result
            print(f"[metrics] development repeated-CV {name}", flush=True)
    if "dggru_tr" in repeated:
        output["comparisons_vs_primary"] = {
            name: comparison_metrics_repeated(
                labels,
                repeated["dggru_tr"],
                repeated[name],
                sample_weight=repeated_weights,
            )
            for name in models
            if name != "dggru_tr"
        }
        add_holm_comparisons(output)
        output["paired_differences_vs_dggru_tr"] = {}
        paired_validity: dict[str, dict] = {}
        for name in models:
            if name == "dggru_tr":
                continue
            validity: dict = {}
            output["paired_differences_vs_dggru_tr"][name] = (
                paired_auc_difference_repeated(
                labels,
                repeated["dggru_tr"],
                repeated[name],
                groups,
                replicates=replicates,
                seed=seed + 1,
                sample_weight=repeated_weights,
                validity_out=validity,
                )
            )
            paired_validity[name] = validity
        output["paired_differences_bootstrap_validity_vs_dggru_tr"] = (
            paired_validity
        )
    return output


def evaluate_predictions(
    dataset: Dataset,
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
    groups: np.ndarray | None = None,
    metric_workers: int = 1,
) -> dict[str, dict]:
    weights = _validated_metric_weight(
        dataset.analysis_weight, len(dataset.labels)
    )
    if groups is None:
        groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    output: dict[str, dict] = {}
    if metric_workers > 1 and len(predictions) > 1:
        with ProcessPoolExecutor(
            max_workers=min(metric_workers, len(predictions))
        ) as executor:
            futures = {
                executor.submit(
                    _evaluate_one_model,
                    name,
                    dataset.labels,
                    probabilities,
                    groups,
                    weights,
                    replicates,
                    seed,
                ): name
                for name, probabilities in predictions.items()
            }
            for future in as_completed(futures):
                name, result = future.result()
                output[name] = result
                print(f"[metrics] {dataset.name} {name}", flush=True)
    else:
        for name, probabilities in predictions.items():
            print(f"[metrics] {dataset.name} {name}", flush=True)
            output[name] = _evaluate_one_model(
                name,
                dataset.labels,
                probabilities,
                groups,
                weights,
                replicates,
                seed,
            )[1]
    if "dggru_tr" in predictions:
        output["comparisons_vs_primary"] = {
            name: comparison_metrics(
                dataset.labels,
                predictions["dggru_tr"],
                probabilities,
                sample_weight=weights,
            )
            for name, probabilities in predictions.items()
            if name != "dggru_tr"
        }
        add_holm_comparisons(output)
        output["paired_differences_vs_dggru_tr"] = {}
        paired_validity = {}
        for name, probabilities in predictions.items():
            if name == "dggru_tr":
                continue
            validity = {}
            output["paired_differences_vs_dggru_tr"][name] = (
                paired_auc_difference(
                dataset.labels,
                predictions["dggru_tr"],
                probabilities,
                groups,
                replicates=replicates,
                seed=seed + 1,
                sample_weight=weights,
                validity_out=validity,
                )
            )
            paired_validity[name] = validity
        output["paired_differences_bootstrap_validity_vs_dggru_tr"] = (
            paired_validity
        )
    return output


def _evaluate_one_model(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray | None,
    replicates: int,
    seed: int,
) -> tuple[str, dict]:
    ci_validity: dict = {}
    intervals = group_bootstrap(
        labels,
        probabilities,
        groups,
        replicates=replicates,
        seed=seed,
        sample_weight=sample_weight,
        validity_out=ci_validity,
    )
    return name, {
        "point": point_metrics(
            labels, probabilities, sample_weight=sample_weight
        ),
        "ci": intervals,
        "ci_validity": ci_validity,
    }


def save_predictions(
    dataset: Dataset,
    predictions: dict[str, np.ndarray],
    prefix: str,
) -> None:
    path = RESULTS / "predictions" / f"{prefix}.npz"
    np.savez_compressed(
        path,
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        subject_key=dataset.cohort["subject_key"].astype(str).to_numpy(),
        **predictions,
    )


def _fit_transport_member(
    development: Dataset,
    external: Dataset,
    name: str,
    *,
    seed: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    lambda_smooth_ratio: float,
    model_seeds: tuple[int, ...],
    cv_best_epochs: dict[str, list[int]] | None,
    cv_selected_lambdas: dict[str, list[float]] | None,
    transport_hyperparameters: dict[str, dict] | None,
) -> np.ndarray:
    """Fit one member on the full development cohort and predict external data."""
    meta = MODEL_META[name]
    if meta["kind"] == "stacking":
        raise ValueError("A stacking model cannot be transported as a member")
    if meta["kind"] in {"xgboost", "slope_logistic"}:
        frame = tabular_frame(development, meta["features"])
        ext_frame = tabular_frame(external, meta["features"]).reindex(
            columns=frame.columns
        )
        dev_x, ext_x = impute_tabular(frame, ext_frame)
        dev_weight = (
            np.ones(len(development.labels), dtype=np.float64)
            if development.analysis_weight is None
            else np.asarray(development.analysis_weight, dtype=np.float64)
        )
        parameters = (
            copy.deepcopy(
                (transport_hyperparameters or {}).get(
                    name,
                    XGBOOST_DEFAULTS,
                )
            )
            if meta["kind"] == "xgboost"
            else None
        )
        fitter = fit_xgboost if meta["kind"] == "xgboost" else fit_slope_logistic
        probabilities = np.mean(
            [
                fitter(
                    dev_x,
                    development.labels,
                    ext_x,
                    seed + 1000 * offset,
                    dev_weight,
                    **(
                        {"parameters": parameters}
                        if meta["kind"] == "xgboost"
                        else {}
                    ),
                )
                for offset in range(len(model_seeds))
            ],
            axis=0,
        )
        return _normalized_probabilities(
            probabilities,
            len(external.labels),
            context=f"{name} external probabilities",
        )

    best_epochs = (cv_best_epochs or {}).get(name, [])
    transport_epochs = (
        int(round(float(np.median(best_epochs))))
        if best_epochs
        else epochs
    )
    selected_weights = (cv_selected_lambdas or {}).get(name, [])
    transport_lambda = (
        float(np.median(selected_weights))
        if selected_weights
        else float(lambda_monotone)
    )
    transport_smooth = (
        float(lambda_smooth_ratio) * transport_lambda
        if selected_weights
        else float(lambda_smooth)
    )
    parameters = copy.deepcopy(SEQUENCE_DEFAULTS)
    parameters.update(
        (transport_hyperparameters or {}).get(name, SEQUENCE_DEFAULTS)
    )
    transport_epochs = min(int(transport_epochs), int(parameters["max_epochs"]))

    dev_static_frame = model_static_frame(
        development,
        str(meta.get("static_mode", "all")),
    )
    ext_static_frame = model_static_frame(
        external,
        str(meta.get("static_mode", "all")),
    ).reindex(columns=dev_static_frame.columns)
    dev_static, ext_static = impute_tabular(dev_static_frame, ext_static_frame)
    dev_raw_sequences, dev_raw_masks = model_sequence_inputs(
        development,
        str(meta.get("sequence_mode", "observed")),
    )
    ext_raw_sequences, ext_raw_masks = model_sequence_inputs(
        external,
        str(meta.get("sequence_mode", "observed")),
    )
    requested_time_mode = str(meta.get("time_mode", "forward"))
    effective_time_mode = requested_time_mode
    if requested_time_mode == "permute":
        dev_raw_sequences, dev_raw_masks = permute_time_order_by_subject(
            dev_raw_sequences,
            dev_raw_masks,
            development.cohort["subject_key"].astype(str).to_numpy(),
            seed=TIME_PERMUTATION_SEED,
        )
        ext_raw_sequences, ext_raw_masks = permute_time_order_by_subject(
            ext_raw_sequences,
            ext_raw_masks,
            external.cohort["subject_key"].astype(str).to_numpy(),
            seed=TIME_PERMUTATION_SEED,
        )
        effective_time_mode = "forward"
    preprocessor = SequencePreprocessor.fit(
        dev_raw_sequences,
        dev_raw_masks,
        dev_static,
    )
    mask_mode = str(meta.get("mask_mode", "any_channel"))
    dev_sequences, dev_masks = preprocessor.transform_sequence(
        dev_raw_sequences,
        dev_raw_masks,
        mask_mode=mask_mode,
    )
    ext_sequences, ext_masks = preprocessor.transform_sequence(
        ext_raw_sequences,
        ext_raw_masks,
        mask_mode=mask_mode,
    )
    dev_deep = preprocessor.transform_static(dev_static)
    ext_deep = preprocessor.transform_static(ext_static)
    probability_parts = []
    for offset in range(len(model_seeds)):
        model = fit_full_temporal_model(
            dev_sequences,
            dev_masks,
            dev_deep,
            development.labels,
            use_graph=bool(meta.get("graph", False)),
            use_regularization=bool(meta.get("regularized", False)),
            architecture=str(meta.get("architecture", "dggru")),
            regularization_direction=meta.get("direction", "increasing"),
            lambda_monotone=(
                0.0
                if meta.get("laplace_only", False)
                else transport_lambda
            ),
            lambda_smooth=transport_smooth,
            seed=seed + 1000 * offset,
            epochs=transport_epochs,
            batch_size=int(parameters["batch_size"]),
            learning_rate=float(parameters["learning_rate"]),
            weight_decay=float(parameters["weight_decay"]),
            hidden=int(parameters["hidden_size"]),
            dropout=float(parameters["dropout"]),
            time_mode=effective_time_mode,
            sample_weight=(
                None
                if development.analysis_weight is None
                else np.asarray(development.analysis_weight, dtype=np.float64)
            ),
            laplace_only=bool(meta.get("laplace_only", False)),
            laplace_input_scale=LAPLACE_INPUT_SCALE,
        )
        probabilities, _ = predict_temporal_model(
            model,
            ext_sequences,
            ext_masks,
            ext_deep,
            time_mode=effective_time_mode,
        )
        probability_parts.append(probabilities)
    return _normalized_probabilities(
        np.mean(probability_parts, axis=0),
        len(external.labels),
        context=f"{name} external probabilities",
    )


def train_full_and_external(
    development: Dataset,
    external_datasets: dict[str, Dataset],
    models: tuple[str, ...],
    epochs: int,
    seed: int,
    lambda_monotone: float,
    lambda_smooth: float,
    cv_best_epochs: dict[str, list[int]] | None = None,
    cv_selected_lambdas: dict[str, list[float]] | None = None,
    lambda_smooth_ratio: float = 0.4,
    model_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    metric_workers: int = 1,
    replicates: int = 2000,
    lambda_grid: tuple[float, ...] = (),
    nested_inner_folds: int = STACKING_INNER_FOLDS,
    stacking_c: float = STACKING_C,
    stacking_members: tuple[str, ...] = STACKING_MEMBERS,
    xgb_candidates: tuple[dict, ...] = (),
    sequence_hyperparameters: tuple[dict, ...] = (),
    cv_selected_hyperparameters: dict[str, list[dict]] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
    external_predictions: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in external_datasets
    }
    external_metrics: dict[str, dict] = {}
    effective_selected_lambdas = {
        name: list(values)
        for name, values in (cv_selected_lambdas or {}).items()
    }
    primary_lambdas = effective_selected_lambdas.get("dggru_tr", [])
    if primary_lambdas:
        matched_control_lambda = float(np.median(primary_lambdas))
        for control in (
            "dggru_tr_treverse",
            "dggru_tr_tpermute",
            "dggru_laplace",
            "dggru_tr_gcsmask",
            "dggru_tr_noseq",
            "dggru_tr_static",
            "dggru_tr_traj",
        ):
            if control in models:
                effective_selected_lambdas[control] = [matched_control_lambda]
    sequence_records = [
        record
        for records in (cv_selected_hyperparameters or {}).values()
        for record in records
        if record.get("family") == "sequence_models"
    ]
    xgboost_records = [
        record
        for records in (cv_selected_hyperparameters or {}).values()
        for record in records
        if record.get("family") == "xgboost"
    ]
    transport_hyperparameters: dict[str, dict] = {}
    for name, meta in MODEL_META.items():
        if meta["kind"] == "stacking":
            continue
        records = (cv_selected_hyperparameters or {}).get(name, [])
        if meta["kind"] == "xgboost":
            fallback = modal_hyperparameters(
                xgboost_records,
                default=XGBOOST_DEFAULTS,
            )
            transport_hyperparameters[name] = modal_hyperparameters(
                records,
                default=fallback,
            )
        elif meta["kind"] == "temporal":
            fallback_records = records
            if not fallback_records and name.startswith("dggru"):
                fallback_records = (
                    cv_selected_hyperparameters or {}
                ).get("dggru_tr", [])
            if not fallback_records:
                fallback_records = sequence_records
            transport_hyperparameters[name] = modal_hyperparameters(
                fallback_records,
                default=SEQUENCE_DEFAULTS,
            )
    for external_name, external in external_datasets.items():
        print(f"[external] fitting transport models for {external_name}", flush=True)
        stack_members = (
            set(stacking_members) if "nested_stack" in models else set()
        )
        transport_names = list(models)
        transport_names.extend(
            member
            for member in stacking_members
            if member not in transport_names
        )
        transported_members: dict[str, np.ndarray] = {}
        for name in transport_names:
            if MODEL_META[name]["kind"] == "stacking":
                continue
            probabilities = _fit_transport_member(
                development,
                external,
                name,
                seed=seed,
                epochs=epochs,
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                lambda_smooth_ratio=lambda_smooth_ratio,
                model_seeds=model_seeds,
                cv_best_epochs=cv_best_epochs,
                cv_selected_lambdas=effective_selected_lambdas,
                transport_hyperparameters=transport_hyperparameters,
            )
            if name in models:
                external_predictions[external_name][name] = probabilities
            if name in stack_members:
                transported_members[name] = probabilities
            print(f"  transported {name}", flush=True)

        if "nested_stack" in models:
            development_index = np.arange(len(development.labels), dtype=int)
            stack_parameter_overrides = {
                member: copy.deepcopy(transport_hyperparameters[member])
                for member in stacking_members
                if member in transport_hyperparameters
            }
            stack_lambda_overrides: dict[str, float] = {}
            stack_smooth_overrides: dict[str, float] = {}
            stack_epoch_overrides: dict[str, int] = {}
            for member in stacking_members:
                selected_weights = effective_selected_lambdas.get(member, [])
                member_lambda = (
                    float(np.median(selected_weights))
                    if selected_weights
                    else float(lambda_monotone)
                )
                stack_lambda_overrides[member] = member_lambda
                stack_smooth_overrides[member] = (
                    float(lambda_smooth_ratio) * member_lambda
                    if selected_weights
                    else float(lambda_smooth)
                )
                member_epochs = (cv_best_epochs or {}).get(member, [])
                if member_epochs:
                    parameters = stack_parameter_overrides.get(
                        member,
                        SEQUENCE_DEFAULTS,
                    )
                    stack_epoch_overrides[member] = min(
                        int(round(float(np.median(member_epochs)))),
                        int(parameters.get("max_epochs", epochs)),
                    )
            oof_probabilities, _ = _stacking_oof_probabilities(
                development,
                development_index,
                seed=seed + 60_000,
                fold=900,
                epochs=epochs,
                lambda_monotone=lambda_monotone,
                lambda_smooth=lambda_smooth,
                lambda_grid=lambda_grid,
                lambda_smooth_ratio=lambda_smooth_ratio,
                model_seeds=model_seeds,
                members=stacking_members,
                inner_folds=nested_inner_folds,
                xgb_candidates=xgb_candidates,
                sequence_hyperparameters=sequence_hyperparameters,
                hyperparameter_overrides=stack_parameter_overrides,
                lambda_overrides=stack_lambda_overrides,
                smooth_overrides=stack_smooth_overrides,
                epoch_overrides=stack_epoch_overrides,
            )
            meta_model = fit_stacking_meta_model(
                oof_probabilities,
                development.labels,
                members=stacking_members,
                c=stacking_c,
                seed=seed + 901,
                sample_weight=(
                    None
                    if development.analysis_weight is None
                    else np.asarray(
                        development.analysis_weight, dtype=np.float64
                    )
                ),
            )
            missing_members = set(stacking_members).difference(
                transported_members
            )
            if missing_members:
                raise RuntimeError(
                    "Stacking members were not transported: "
                    + ", ".join(sorted(missing_members))
                )
            external_predictions[external_name]["nested_stack"] = (
                predict_stacking_meta_model(
                    meta_model,
                    transported_members,
                    members=stacking_members,
                )
            )
            print("  transported nested_stack", flush=True)
        if (
            "hospital_key" in external.cohort
            and external.cohort["hospital_key"].nunique(dropna=True) > 1
        ):
            validation_groups = external.cohort["hospital_key"].astype(str).to_numpy()
        else:
            validation_groups = external.cohort["subject_key"].astype(str).to_numpy()
        external_metrics[external_name] = evaluate_predictions(
            external,
            external_predictions[external_name],
            replicates=replicates,
            seed=seed,
            groups=validation_groups,
            metric_workers=metric_workers,
        )
    return external_predictions, external_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--lambda-monotone", type=float, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument(
        "--fixed-lambda",
        action="store_true",
        help="Evaluate the supplied lambda pair without inner lambda selection.",
    )
    parser.add_argument(
        "--freeze-from",
        default=None,
        help=(
            "Path to a completed run_manifest_<tag>.json. Freeze the selected "
            "dggru_tr lambda and the matched-control epochs from that run."
        ),
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-threads", type=int, default=None)
    parser.add_argument("--metric-workers", type=int, default=None)
    args = parser.parse_args()
    source_sha256_at_start = training_source_snapshot()
    snapshot_started_at_utc = datetime.now(timezone.utc).isoformat()
    (
        config,
        config_sha256_at_start,
        data_sha256_at_start,
    ) = snapshot_training_inputs()
    ensure_dirs()
    args.epochs = (
        int(args.epochs)
        if args.epochs is not None
        else max(
            int(candidate["max_epochs"])
            for candidate in config["hyperparameters"]["sequence_models"][
                "candidates"
            ]
        )
    )
    args.folds = (
        int(args.folds)
        if args.folds is not None
        else int(config["cv_folds"])
    )
    args.replicates = (
        int(args.replicates)
        if args.replicates is not None
        else int(config["bootstrap_replicates"])
    )
    args.workers = (
        int(args.workers)
        if args.workers is not None
        else int(config.get("parallel_fold_workers", 1))
    )
    args.worker_threads = (
        int(args.worker_threads)
        if args.worker_threads is not None
        else int(config.get("worker_threads", 1))
    )
    args.metric_workers = (
        int(args.metric_workers)
        if args.metric_workers is not None
        else int(config.get("parallel_metric_workers", 1))
    )
    if min(
        args.epochs,
        args.folds,
        args.replicates,
        args.workers,
        args.worker_threads,
        args.metric_workers,
    ) < 1:
        raise ValueError("Training, CV, bootstrap, and worker counts must be positive")
    seed = int(config["random_seed"])
    model_seeds = tuple(
        range(int(config.get("model_seeds", len(MODEL_SEEDS))))
    )
    cv_repeats = (
        int(config.get("cv_repeats", 1))
        if args.cv_repeats is None
        else int(args.cv_repeats)
    )
    development_raw = load_dataset("mimic4")
    feature_columns = list(development_raw.static_all.columns)
    development = load_dataset("mimic4", feature_columns=feature_columns)
    external = {
        name: load_dataset(name, feature_columns=feature_columns)
        for name in ("mimic3", "eicu")
    }
    selected_models = (
        ALL_MODEL_ORDER
        if args.models == "all"
        else tuple(part.strip() for part in args.models.split(","))
    )
    unknown_models = [name for name in selected_models if name not in MODEL_META]
    if unknown_models:
        raise ValueError("Unknown models: " + ", ".join(unknown_models))
    lambda_smooth_ratio = float(config.get("lambda_smooth_ratio", 0.4))
    frozen_manifest_path: Path | None = None
    frozen_manifest: dict | None = None
    frozen_lambda: float | None = None
    frozen_epoch: int | None = None
    if args.freeze_from:
        frozen_manifest_path = Path(args.freeze_from).resolve()
        if not frozen_manifest_path.exists():
            raise FileNotFoundError(
                f"--freeze-from manifest does not exist: {frozen_manifest_path}"
            )
        with open(frozen_manifest_path, encoding="utf-8") as fh:
            frozen_manifest = json.load(fh)
        primary_lambda = (
            frozen_manifest.get("cv_selected_lambdas", {})
            .get("dggru_tr", {})
            .get("median")
        )
        primary_epochs = (
            frozen_manifest.get("cv_best_epochs", {})
            .get("dggru_tr", {})
            .get("values", [])
        )
        if primary_lambda is None or not np.isfinite(float(primary_lambda)):
            raise ValueError(
                "--freeze-from manifest lacks a finite dggru_tr selected lambda"
            )
        if not primary_epochs:
            raise ValueError(
                "--freeze-from manifest lacks dggru_tr selected epochs"
            )
        frozen_lambda = float(primary_lambda)
        frozen_epoch = int(round(float(np.median(primary_epochs))))
        if args.lambda_monotone is None and args.lambda_smooth is None:
            args.fixed_lambda = True
    freezeable_temporal = {
        "dggru_tr",
        "dggru_tr_treverse",
        "dggru_tr_tpermute",
        "dggru_laplace",
        *SECONDARY_MODEL_ORDER,
    }
    epoch_overrides: dict[str, int] = {}
    lambda_overrides: dict[str, float] = {}
    smooth_overrides: dict[str, float] = {}
    if frozen_lambda is not None and frozen_epoch is not None:
        for name in selected_models:
            if (
                MODEL_META[name]["kind"] == "temporal"
                and name in freezeable_temporal
            ):
                epoch_overrides[name] = frozen_epoch
                lambda_overrides[name] = frozen_lambda
                smooth_overrides[name] = lambda_smooth_ratio * frozen_lambda
    lambda_monotone = (
        float(frozen_lambda)
        if frozen_lambda is not None and args.lambda_monotone is None
        else (
            float(config["lambda_monotone"])
            if args.lambda_monotone is None
            else float(args.lambda_monotone)
        )
    )
    lambda_smooth = (
        float(lambda_smooth_ratio * frozen_lambda)
        if frozen_lambda is not None and args.lambda_smooth is None
        else (
            float(config["lambda_smooth"])
            if args.lambda_smooth is None
            else float(args.lambda_smooth)
        )
    )
    xgb_candidates = configured_candidates(
        config,
        "xgboost",
        set(XGBOOST_DEFAULTS),
    )
    sequence_candidates = configured_candidates(
        config,
        "sequence_models",
        set(SEQUENCE_DEFAULTS),
    )
    tag_suffix = f"_{args.tag}" if args.tag else ""
    lambda_grid = (
        tuple()
        if args.fixed_lambda or frozen_lambda is not None
        else tuple(float(value) for value in config.get("lambda_grid", ()))
    )
    if lambda_grid:
        if len(lambda_grid) != 18:
            raise ValueError(
                "The prespecified temporal-regularization grid must contain "
                f"exactly 18 values; got {len(lambda_grid)}."
            )
        if len(set(lambda_grid)) != len(lambda_grid):
            raise ValueError("The temporal-regularization grid contains duplicates.")
        if not all(np.isfinite(lambda_grid)) or min(lambda_grid) < 0:
            raise ValueError(
                "The temporal-regularization grid must contain finite "
                "non-negative values."
            )
    nested_inner_folds = int(
        config.get("nested_inner_folds", STACKING_INNER_FOLDS)
    )
    stacking_c = float(config.get("stacking_c", STACKING_C))
    stacking_members = tuple(
        config.get("stacking_members", STACKING_MEMBERS)
    )
    (
        predictions,
        splits,
        cv_best_epochs,
        cv_selected_lambdas,
        cv_selected_hyperparameters,
    ) = run_cv(
        development,
        folds=args.folds,
        epochs=args.epochs,
        seed=seed,
        models=selected_models,
        lambda_monotone=lambda_monotone,
        lambda_smooth=lambda_smooth,
        lambda_grid=lambda_grid,
        lambda_smooth_ratio=lambda_smooth_ratio,
        workers=args.workers,
        worker_threads=args.worker_threads,
        model_seeds=model_seeds,
        repeats=cv_repeats,
        repeat_prefix=f"mimic4_cv{tag_suffix}",
        nested_inner_folds=nested_inner_folds,
        stacking_c=stacking_c,
        stacking_members=stacking_members,
        xgb_candidates=xgb_candidates,
        sequence_hyperparameters=sequence_candidates,
        epoch_overrides=epoch_overrides,
        lambda_overrides=lambda_overrides,
        smooth_overrides=smooth_overrides,
    )
    save_predictions(development, predictions, f"mimic4_cv{tag_suffix}")
    repeat_path = RESULTS / "predictions" / f"mimic4_cv{tag_suffix}_repeats.npz"
    metrics = {}
    if cv_repeats > 1:
        development_weights = _validated_metric_weight(
            development.analysis_weight, len(development.labels)
        )
        metrics["development"] = evaluate_repeated_cv(
            development,
            repeat_path,
            selected_models,
            cv_repeats,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )
        metrics["development"]["cv_repeat_variability"] = repeated_cv_variability(
            development,
            repeat_path,
            selected_models,
            cv_repeats,
        )
        metrics["development_probability_ensemble"] = {
            "description": (
                "Secondary analysis based on averaging OOF probabilities "
                "across repeated CV partitions."
            ),
            "point": {
                name: point_metrics(
                    development.labels,
                    values,
                    sample_weight=development_weights,
                )
                for name, values in predictions.items()
            },
        }
    else:
        metrics["development"] = evaluate_predictions(
            development,
            predictions,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )
    fold_rows = []
    for index, (_, test_index) in enumerate(splits):
        repeat, fold = divmod(index, args.folds)
        fold_rows.append(
            pd.DataFrame(
                {
                    "stay_id": development.cohort["stay_id"].iloc[test_index].to_numpy(),
                    "subject_key": development.cohort["subject_key"].iloc[test_index].to_numpy(),
                    "outcome": development.labels[test_index],
                    "repeat": repeat,
                    "fold": fold,
                }
            )
        )
    pd.DataFrame(
        pd.concat(fold_rows, ignore_index=True)
    ).to_csv(RESULTS / f"mimic4_folds{tag_suffix}.csv", index=False)
    folds_path = RESULTS / f"mimic4_folds{tag_suffix}.csv"
    if not args.skip_external:
        external_models = tuple(
            name for name in selected_models if name in OUTLINE_MODEL_ORDER
        )
        external_predictions, external_metrics = train_full_and_external(
            development,
            external,
            models=external_models,
            epochs=args.epochs,
            seed=seed + 5000,
            lambda_monotone=lambda_monotone,
            lambda_smooth=lambda_smooth,
            cv_best_epochs=cv_best_epochs,
            cv_selected_lambdas=cv_selected_lambdas,
            lambda_smooth_ratio=lambda_smooth_ratio,
            model_seeds=model_seeds,
            metric_workers=args.metric_workers,
            replicates=args.replicates,
            lambda_grid=lambda_grid,
            nested_inner_folds=nested_inner_folds,
            stacking_c=stacking_c,
            stacking_members=stacking_members,
            xgb_candidates=xgb_candidates,
            sequence_hyperparameters=sequence_candidates,
            cv_selected_hyperparameters=cv_selected_hyperparameters,
        )
        metrics["external"] = external_metrics
        for name, preds in external_predictions.items():
            save_predictions(
                external[name], preds, f"{name}_external{tag_suffix}"
            )
    require_inputs_unchanged(
        config_sha256_at_start,
        data_sha256_at_start,
    )
    source_sha256_at_end = training_source_snapshot()
    if source_sha256_at_end != source_sha256_at_start:
        changed_sources = sorted(
            name
            for name in set(source_sha256_at_start) | set(source_sha256_at_end)
            if source_sha256_at_start.get(name)
            != source_sha256_at_end.get(name)
        )
        raise RuntimeError(
            "Model-fitting source changed during the formal run: "
            + ", ".join(changed_sources)
        )
    snapshot_verified_at_utc = datetime.now(timezone.utc).isoformat()
    with open(RESULTS / f"metrics{tag_suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, allow_nan=True)
    sequence_records = [
        record
        for records in cv_selected_hyperparameters.values()
        for record in records
        if record.get("family") == "sequence_models"
    ]
    xgboost_records = [
        record
        for records in cv_selected_hyperparameters.values()
        for record in records
        if record.get("family") == "xgboost"
    ]
    modal_parameters: dict[str, dict] = {}
    for name, meta in MODEL_META.items():
        if meta["kind"] == "stacking":
            continue
        records = cv_selected_hyperparameters.get(name, [])
        if meta["kind"] == "xgboost":
            fallback = modal_hyperparameters(
                xgboost_records,
                default=XGBOOST_DEFAULTS,
            )
            modal_parameters[name] = modal_hyperparameters(
                records,
                default=fallback,
            )
        elif meta["kind"] == "temporal":
            fallback_records = records
            if not fallback_records and name.startswith("dggru"):
                fallback_records = cv_selected_hyperparameters.get(
                    "dggru_tr", []
                )
            if not fallback_records:
                fallback_records = sequence_records
            modal_parameters[name] = modal_hyperparameters(
                fallback_records,
                default=SEQUENCE_DEFAULTS,
            )
    manifest = {
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "development_n": int(len(development.labels)),
        "development_outcomes": {
            str(k): int(v)
            for k, v in development.cohort["outcome"].value_counts().sort_index().items()
        },
        "features": feature_columns,
        "feature_selection": {
            "method": (
                "fixed_source_schema_numeric_columns_after_explicit_"
                "landmark_boundary_blocklist"
            ),
            "outcome_or_missingness_screening_used": False,
            "post_landmark_columns": sorted(POST_LANDMARK_COLUMNS),
            "post_landmark_prefixes": list(POST_LANDMARK_PREFIXES),
        },
        "temporal_static_features": [
            column for column in CORE_STATIC if column in feature_columns
        ],
        "temporal_static_feature_count": len(
            [column for column in CORE_STATIC if column in feature_columns]
        ),
        "model_feature_contract": model_feature_contract(development),
        "deep_static_reference_subset": [
            column for column in CORE_STATIC if column in feature_columns
        ],
        "core_static_features": [
            column for column in CORE_STATIC if column in feature_columns
        ],
        "models": list(selected_models),
        "model_cv_repeats": {
            name: int(cv_repeats) for name in selected_models
        },
        "outline_models": list(OUTLINE_MODEL_ORDER),
        "secondary_models": list(SECONDARY_MODEL_ORDER),
        "epochs": args.epochs,
        "bootstrap_replicates": args.replicates,
        "cv_best_epochs": {
            model: {
                "values": values,
                "external_transport_epochs": int(round(float(np.median(values)))),
            }
            for model, values in cv_best_epochs.items()
            if values
        },
        "lambda_grid": list(lambda_grid),
        "lambda_smooth_ratio": lambda_smooth_ratio,
        "lambda_policy": {
            "dggru_tr": (
                "frozen_from_manifest"
                if frozen_lambda is not None
                else (
                    "fixed_supplied_value"
                    if args.fixed_lambda
                    else "inner_cv_selection"
                )
            ),
            "matched_controls": (
                "frozen_to_primary"
                if frozen_lambda is not None
                else "matched_to_primary_fit"
            ),
            "external_lambda_source": "development_cv_only",
            "frozen_lambda": frozen_lambda,
        },
        "epoch_policy": {
            "matched_controls": (
                "frozen_to_primary_median_epoch"
                if frozen_epoch is not None
                else "matched_to_primary_fold_fit"
            ),
            "frozen_epoch": frozen_epoch,
        },
        "stacking_inner_oof_seeds": (
            1 if "nested_stack" in selected_models else 0
        ),
        "freeze_from_manifest": (
            {
                "path": str(frozen_manifest_path),
                "sha256": sha256_file(frozen_manifest_path),
            }
            if frozen_manifest_path is not None
            else None
        ),
        "laplace_control": {
            "implementation": "generic_laplace_noise_on_observed_standardised_inputs",
            "scale": LAPLACE_INPUT_SCALE,
            "applies_to": "dggru_laplace",
        },
        "hyperparameter_search": {
            "xgboost": {
                "strategy": "explicit_prespecified_grid",
                "candidates": list(xgb_candidates),
            },
            "sequence_models": {
                "strategy": "explicit_prespecified_grid",
                "candidates": list(sequence_candidates),
            },
            "selection_metric": "inner_patient_grouped_macro_auc",
            "selection_estimator": (
                "fixed_single_seed_selection_then_five_seed_final_ensemble"
            ),
            "selection_scope": "outer_training_data_only",
        },
        "statistical_inference": {
            "primary": (
                "conditional_paired_group_bootstrap_of_mean_repeated_cv_"
                "macro_auc"
            ),
            "bootstrap_estimability": (
                "rejection_sample_clusters_until_all_three_ordinal_grades_"
                "are_present"
            ),
            "requested_bootstrap_replicates": args.replicates,
            "bootstrap_cluster": {
                "development": "subject_key",
                "mimic3": "subject_key",
                "eicu": "hospital_key_then_subject_key_fallback",
            },
            "secondary_delong": {
                "status": (
                    "unclustered_first_partition_secondary_not_primary"
                ),
                "holm_family": "fixed_five_outline_baselines_x_any_severe",
                "comparators": list(OUTLINE_HOLM_COMPARATORS),
            },
        },
        "cv_selected_hyperparameters": cv_selected_hyperparameters,
        "development_modal_hyperparameters": modal_parameters,
        "cv_selected_lambdas": {
            model: {
                "values": values,
                "median": float(np.median(values)),
                "fit_records": len(values),
                "independent_values": (
                    values[:: max(1, len(model_seeds))]
                    if model == "dggru_tr" and bool(lambda_grid)
                    else []
                ),
                "independent_selection_records": (
                    args.folds * cv_repeats
                    if model == "dggru_tr" and bool(lambda_grid)
                    else 0
                ),
            }
            for model, values in cv_selected_lambdas.items()
            if values
        },
        "folds": args.folds,
        "cv_repeats": cv_repeats,
        "parallel_fold_workers": args.workers,
        "worker_threads": args.worker_threads,
        "metric_workers": args.metric_workers,
        "seed": seed,
        "model_seeds": list(model_seeds),
        "lambda_monotone": lambda_monotone,
        "lambda_smooth": lambda_smooth,
        "config": config,
        "config_sha256": config_sha256_at_start,
        "data_sha256": data_sha256_at_start,
        "input_snapshot": {
            "captured_at_utc": snapshot_started_at_utc,
            "verified_unchanged_at_utc": snapshot_verified_at_utc,
            "capture_phase": "before_data_loading_or_model_fitting",
            "verification_phase": "after_all_model_fitting_before_metrics_write",
            "config_sha256": config_sha256_at_start,
            "data_file_count": len(data_sha256_at_start),
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "xgboost": xgb.__version__,
            "scikit-learn": installed_version("scikit-learn"),
            "scipy": installed_version("scipy"),
            "duckdb": installed_version("duckdb"),
            "python-docx": installed_version("python-docx"),
            "docx2pdf": installed_version("docx2pdf"),
        },
        "source_sha256": {
            **source_sha256_at_start,
        },
        "source_sha256_at_end": source_sha256_at_end,
        "source_hash_snapshot": (
            "captured_before_data_loading_or_model_fitting_and_verified_equal_"
            "after_all_model_fitting"
        ),
        "fold_artifact": {
            "path": str(folds_path),
            "sha256": sha256_file(folds_path),
        },
        "result_sha256": {
            "metrics": sha256_file(RESULTS / f"metrics{tag_suffix}.json"),
            "development_repeated_predictions": sha256_file(repeat_path),
            "development_probability_ensemble": sha256_file(
                RESULTS / "predictions" / f"mimic4_cv{tag_suffix}.npz"
            ),
            **{
                f"{name}_external_predictions": sha256_file(
                    RESULTS
                    / "predictions"
                    / f"{name}_external{tag_suffix}.npz"
                )
                for name in external
                if (
                    RESULTS
                    / "predictions"
                    / f"{name}_external{tag_suffix}.npz"
                ).exists()
            },
        },
    }
    with open(
        RESULTS / f"run_manifest{tag_suffix}.json", "w", encoding="utf-8"
    ) as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved results/metrics{tag_suffix}.json", flush=True)


if __name__ == "__main__":
    main()
