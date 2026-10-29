"""Stratified external transport analysis by GCS documentation intensity.

The documentation strata are fixed from the MIMIC-IV development cohort.
MIMIC-III and eICU use those absolute cutpoints and are never requantilized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 3
STRATA = ("low", "middle", "high")
CUTPOINT_SCHEMA_VERSION = 1
CUTPOINT_METHOD = "MIMIC-IV development monitoring_intensity_24h tertiles"
QUANTILE_METHOD = "linear"
QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_MODEL = "dggru_tr"
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_SEED = 20261024
ECE_BINS = 10
CALIBRATION_CLIP = 1e-5
EVENT_RATE_CI_Z = 1.959963984540054
DOCUMENTATION_GRADIENT_CAVEAT = (
    "These strata describe documentation-associated transport gradients. "
    "They do not identify or explain differences attributable to calendar "
    "period, case-mix, treatment, or care intensity."
)


def _canonical_id(value: Any) -> str:
    """Return a stable identifier without changing its numeric value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Identifier is missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Identifier is empty")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else "NA"
    if isinstance(value, Path):
        return str(value)
    return value


def _finite_or_na(value: float) -> float | str:
    return float(value) if np.isfinite(value) else "NA"


def _validate_outcomes(values: Iterable[Any], source: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source}: outcome contains missing or nonnumeric values")
    rounded = np.rint(numeric).astype(int)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{source}: outcome must contain integer grades")
    if not set(np.unique(rounded)).issubset(set(range(N_CLASSES))):
        raise ValueError(f"{source}: outcome must be restricted to grades 0, 1, and 2")
    return rounded


def load_cohort(path: Path, cluster_column: str) -> pd.DataFrame:
    """Load and validate a cohort used by the transport analysis."""
    frame = pd.read_csv(path)
    required = {
        "stay_id",
        "subject_key",
        "outcome",
        "monitoring_intensity_24h",
        cluster_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: cohort is empty")

    result = frame.copy()
    result["_stay_key"] = [_canonical_id(value) for value in result["stay_id"]]
    result["_subject_key"] = [
        _canonical_id(value) for value in result["subject_key"]
    ]
    result["_cluster_key"] = [
        _canonical_id(value) for value in result[cluster_column]
    ]
    if result["_stay_key"].duplicated().any():
        raise ValueError(f"{path}: duplicate stay_id values")
    if result["_cluster_key"].isna().any():
        raise ValueError(f"{path}: cluster column {cluster_column!r} has missing values")

    intensity = pd.to_numeric(
        result["monitoring_intensity_24h"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(intensity).all():
        raise ValueError(
            f"{path}: monitoring_intensity_24h has missing or nonnumeric values"
        )
    if (intensity < 0).any():
        raise ValueError(f"{path}: monitoring_intensity_24h contains negative values")
    result["_intensity"] = intensity
    result["_outcome"] = _validate_outcomes(result["outcome"], str(path))
    return result


def development_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint the development intensity values used to fix cutpoints."""
    if "_stay_key" not in frame or "_intensity" not in frame:
        raise ValueError("Development cohort has not passed load_cohort validation")
    ordered = frame.sort_values("_stay_key")
    payload = "\n".join(
        f"{stay}\t{float(intensity):.17g}"
        for stay, intensity in zip(ordered["_stay_key"], ordered["_intensity"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_development_cutpoints(frame: pd.DataFrame) -> tuple[float, float]:
    """Compute the two prespecified MIMIC-IV intensity tertile cutpoints."""
    if "_intensity" not in frame:
        raise ValueError("Development cohort has not passed load_cohort validation")
    values = frame["_intensity"].to_numpy(dtype=float)
    cutpoints = np.quantile(values, QUANTILES, method=QUANTILE_METHOD)
    low, high = float(cutpoints[0]), float(cutpoints[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError("Development intensity values do not define distinct tertiles")
    return low, high


def load_or_create_cutpoints(
    development: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    """Load fixed cutpoints or create them once from the development cohort."""
    fingerprint = development_fingerprint(development)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CUTPOINT_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported cutpoint schema")
        if payload.get("method") != CUTPOINT_METHOD:
            raise ValueError(f"{path}: unexpected cutpoint method")
        if payload.get("quantile_method") != QUANTILE_METHOD:
            raise ValueError(f"{path}: unexpected quantile method")
        if payload.get("development_fingerprint") != fingerprint:
            raise ValueError(
                f"{path}: cutpoints were created from a different development cohort"
            )
        cutpoints = payload.get("cutpoints")
        if (
            not isinstance(cutpoints, list)
            or len(cutpoints) != 2
            or not all(isinstance(value, (int, float)) for value in cutpoints)
            or not float(cutpoints[0]) < float(cutpoints[1])
        ):
            raise ValueError(f"{path}: invalid fixed cutpoints")
        return payload

    low, high = compute_development_cutpoints(development)
    payload = {
        "schema_version": CUTPOINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": CUTPOINT_METHOD,
        "quantile_method": QUANTILE_METHOD,
        "quantiles": list(QUANTILES),
        "cutpoints": [low, high],
        "development_fingerprint": fingerprint,
        "development_n": int(len(development)),
        "external_requantilization_permitted": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assign_documentation_strata(
    intensity: Sequence[float],
    cutpoints: Sequence[float],
) -> np.ndarray:
    """Assign fixed absolute strata: <= q1, (q1, q2], and > q2."""
    values = np.asarray(intensity, dtype=float)
    if values.ndim != 1:
        raise ValueError("Documentation intensity must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("Documentation intensity contains missing values")
    if (values < 0).any():
        raise ValueError("Documentation intensity contains negative values")
    if len(cutpoints) != 2:
        raise ValueError("Exactly two documentation cutpoints are required")
    low, high = float(cutpoints[0]), float(cutpoints[1])
    if not low < high:
        raise ValueError("Documentation cutpoints must be strictly increasing")
    output = np.full(len(values), "high", dtype=object)
    output[values <= low] = "low"
    output[(values > low) & (values <= high)] = "middle"
    return output.astype(str)


def load_prediction_file(
    path: Path,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load IDs, labels, and one model's three-class probabilities."""
    with np.load(path, allow_pickle=True) as payload:
        required = {"labels", "stay_id", "subject_key", model}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path}: missing prediction arrays {sorted(missing)}")
        labels = _validate_outcomes(payload["labels"], f"{path}: labels")
        stay_id = np.asarray(payload["stay_id"])
        subject_key = np.asarray(payload["subject_key"])
        probabilities = np.asarray(payload[model], dtype=float)

    if labels.shape != stay_id.shape or labels.shape != subject_key.shape:
        raise ValueError(f"{path}: prediction IDs and labels have inconsistent lengths")
    if probabilities.shape != (len(labels), N_CLASSES):
        raise ValueError(
            f"{path}: probability array for {model!r} must have shape (n, 3)"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain nonfinite values")
    if (probabilities < 0).any():
        raise ValueError(f"{path}: probabilities contain negative values")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(
            f"{path}: probabilities do not sum to one within tolerance 1e-5"
        )
    probabilities = probabilities / row_sums[:, None]
    return labels, stay_id, subject_key, probabilities


def align_predictions(
    cohort: pd.DataFrame,
    prediction_path: Path,
    model: str,
) -> pd.DataFrame:
    """Strictly align an external prediction file with its cohort."""
    labels, stay_ids, subject_keys, probabilities = load_prediction_file(
        prediction_path, model
    )
    prediction_keys = np.asarray(
        [_canonical_id(value) for value in stay_ids], dtype=object
    )
    if pd.Series(prediction_keys).duplicated().any():
        raise ValueError(f"{prediction_path}: duplicate prediction stay_id values")

    cohort_by_key = cohort.set_index("_stay_key", drop=False)
    prediction_ids = set(prediction_keys)
    cohort_ids = set(cohort["_stay_key"])
    missing_from_predictions = sorted(cohort_ids.difference(prediction_ids))
    absent_from_cohort = sorted(prediction_ids.difference(cohort_ids))
    if missing_from_predictions or absent_from_cohort:
        missing_example = ", ".join(missing_from_predictions[:5]) or "none"
        extra_example = ", ".join(absent_from_cohort[:5]) or "none"
        raise ValueError(
            f"{prediction_path}: predictions and cohort do not cover the same "
            f"stays (cohort missing={len(missing_from_predictions)} "
            f"[{missing_example}], prediction extra={len(absent_from_cohort)} "
            f"[{extra_example}])"
        )

    aligned = cohort_by_key.loc[list(prediction_keys)].reset_index(drop=True)
    if not np.array_equal(aligned["_outcome"].to_numpy(), labels):
        mismatch = int(np.sum(aligned["_outcome"].to_numpy() != labels))
        raise ValueError(
            f"{prediction_path}: labels disagree with cohort outcomes for "
            f"{mismatch} stays"
        )
    prediction_subjects = [_canonical_id(value) for value in subject_keys]
    if aligned["_subject_key"].tolist() != prediction_subjects:
        raise ValueError(f"{prediction_path}: subject_key disagrees with cohort")

    aligned["probability_0"] = probabilities[:, 0]
    aligned["probability_1"] = probabilities[:, 1]
    aligned["probability_2"] = probabilities[:, 2]
    return aligned


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Macro one-vs-rest AUC, undefined unless all three grades are present."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(labels).size < N_CLASSES:
        return float("nan")
    values = [
        roc_auc_score((labels == grade).astype(int), probabilities[:, grade])
        for grade in range(N_CLASSES)
    ]
    return float(np.mean(values))


def wilson_score_interval(
    events: int,
    total: int,
    *,
    z: float = EVENT_RATE_CI_Z,
) -> tuple[float, float]:
    """Return a reproducible unclustered Wilson score interval for a rate."""
    if isinstance(events, bool) or isinstance(total, bool):
        raise ValueError("Event counts must be integers")
    if int(events) != events or int(total) != total:
        raise ValueError("Event counts must be integers")
    events = int(events)
    total = int(total)
    if total <= 0:
        raise ValueError("Event-rate intervals require a nonempty stratum")
    if events < 0 or events > total:
        raise ValueError("Event count must lie between zero and the stratum size")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Wilson interval z must be positive and finite")
    proportion = events / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (
        float(max(0.0, center - half_width)),
        float(min(1.0, center + half_width)),
    )


def binary_expected_calibration_error(
    target: np.ndarray,
    predicted: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    """Equal-width binary ECE over the fixed interval [0, 1]."""
    target = np.asarray(target, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if target.shape != predicted.shape:
        raise ValueError("Calibration target and prediction shapes differ")
    if not len(target):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted >= low) & (
            predicted <= high if high == 1.0 else predicted < high
        )
        if mask.any():
            result += mask.mean() * abs(
                float(target[mask].mean() - predicted[mask].mean())
            )
    return float(result)


def calibration_slope_intercept(
    target: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """Fit observed outcome on the logit of the cumulative prediction."""
    target = np.asarray(target, dtype=int)
    predicted = np.asarray(predicted, dtype=float)
    if target.shape != predicted.shape:
        raise ValueError("Calibration target and prediction shapes differ")
    if len(target) < 2 or np.unique(target).size < 2:
        return float("nan"), float("nan")
    predicted = np.clip(predicted, CALIBRATION_CLIP, 1.0 - CALIBRATION_CLIP)
    logit = np.log(predicted / (1.0 - predicted)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-8,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(logit, target)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return float("nan"), float("nan")
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan")
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return float("nan"), float("nan")
    return slope, intercept


def cumulative_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    target = (labels >= threshold).astype(int)
    predicted = probabilities[:, threshold:].sum(axis=1)
    slope, intercept = calibration_slope_intercept(target, predicted)
    return {
        "events": int(target.sum()),
        "slope": slope,
        "intercept": intercept,
        "ece": binary_expected_calibration_error(target, predicted),
    }


def _bootstrap_seed(
    base_seed: int,
    dataset: str,
    model: str,
    stratum: str,
) -> int:
    material = f"{base_seed}|{dataset}|{model}|{stratum}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def cluster_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clusters: Sequence[Any],
    replicates: int,
    seed: int,
) -> tuple[float, float, int, int]:
    """Return percentile CI, valid replicates, and available clusters."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    cluster_values = np.asarray(
        [_canonical_id(value) for value in clusters], dtype=object
    )
    unique_clusters = np.unique(cluster_values)
    if (
        len(unique_clusters) < 2
        or replicates <= 0
        or not np.isfinite(macro_auc(labels, probabilities))
    ):
        return float("nan"), float("nan"), 0, int(len(unique_clusters))

    indices_by_cluster = {
        cluster: np.flatnonzero(cluster_values == cluster)
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_clusters = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        index = np.concatenate(
            [indices_by_cluster[cluster] for cluster in sampled_clusters]
        )
        estimate = macro_auc(labels[index], probabilities[index])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan"), 0, int(len(unique_clusters))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(estimates), int(len(unique_clusters))


def _record_for_stratum(
    dataset: str,
    model: str,
    stratum: str,
    frame: pd.DataFrame,
    cutpoints: Sequence[float],
    cluster_column: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["_outcome"].to_numpy(dtype=int)
    probabilities = frame[
        ["probability_0", "probability_1", "probability_2"]
    ].to_numpy(dtype=float)
    point_auc = macro_auc(labels, probabilities)
    ci_low, ci_high, valid, cluster_n = cluster_bootstrap_auc(
        labels,
        probabilities,
        frame["_cluster_key"].to_numpy(),
        replicates=bootstrap_replicates,
        seed=_bootstrap_seed(seed, dataset, model, stratum),
    )
    any_calibration = cumulative_calibration(labels, probabilities, threshold=1)
    severe_calibration = cumulative_calibration(
        labels, probabilities, threshold=2
    )
    any_events = int(np.sum(labels >= 1))
    severe_events = int(np.sum(labels == 2))
    any_rate_low, any_rate_high = wilson_score_interval(any_events, len(frame))
    severe_rate_low, severe_rate_high = wilson_score_interval(
        severe_events,
        len(frame),
    )
    return {
        "dataset": dataset,
        "model": model,
        "documentation_stratum": stratum,
        "q1_cutpoint": float(cutpoints[0]),
        "q2_cutpoint": float(cutpoints[1]),
        "n": int(len(frame)),
        "grade_0": int(np.sum(labels == 0)),
        "grade_1": int(np.sum(labels == 1)),
        "grade_2": int(np.sum(labels == 2)),
        "events": any_events,
        "event_rate": float(any_events / len(frame)),
        "event_rate_ci_low": any_rate_low,
        "event_rate_ci_high": any_rate_high,
        "any_events": any_events,
        "any_event_rate": float(any_events / len(frame)),
        "any_event_rate_ci_low": any_rate_low,
        "any_event_rate_ci_high": any_rate_high,
        "severe_events": severe_events,
        "severe_event_rate": float(severe_events / len(frame)),
        "severe_event_rate_ci_low": severe_rate_low,
        "severe_event_rate_ci_high": severe_rate_high,
        "event_rate_ci_method": "wilson_score_95_unclustered_deterministic",
        "macro_auc": point_auc,
        "macro_auc_ci_low": ci_low,
        "macro_auc_ci_high": ci_high,
        "bootstrap_clusters": cluster_n,
        "bootstrap_valid_replicates": valid,
        "cluster_column": cluster_column,
        "bootstrap_seed": _bootstrap_seed(seed, dataset, model, stratum),
        "cumulative_any": any_calibration,
        "cumulative_severe": severe_calibration,
    }


def analyze_dataset(
    dataset: str,
    cohort: pd.DataFrame,
    prediction_path: Path,
    model: str,
    cutpoints: Sequence[float],
    cluster_column: str,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    aligned = align_predictions(cohort, prediction_path, model)
    aligned["_documentation_stratum"] = assign_documentation_strata(
        aligned["_intensity"].to_numpy(dtype=float),
        cutpoints,
    )
    records = []
    for stratum in STRATA:
        selected = aligned.loc[
            aligned["_documentation_stratum"].eq(stratum)
        ].copy()
        if selected.empty:
            continue
        records.append(
            _record_for_stratum(
                dataset=dataset,
                model=model,
                stratum=stratum,
                frame=selected,
                cutpoints=cutpoints,
                cluster_column=cluster_column,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
        )
    return records


def validate_external_strata(
    records: Sequence[dict[str, Any]],
    datasets: Sequence[str],
    models: Sequence[str],
) -> dict[str, Any]:
    """Fail if any external dataset/model is missing an expected stratum."""
    dataset_order = list(dict.fromkeys(datasets))
    model_order = list(dict.fromkeys(models))
    expected = [
        (dataset, model, stratum)
        for dataset in dataset_order
        for model in model_order
        for stratum in STRATA
    ]
    actual: list[tuple[str, str, str]] = []
    event_rate_fields = (
        "events",
        "event_rate",
        "event_rate_ci_low",
        "event_rate_ci_high",
        "any_events",
        "any_event_rate",
        "any_event_rate_ci_low",
        "any_event_rate_ci_high",
        "severe_events",
        "severe_event_rate",
        "severe_event_rate_ci_low",
        "severe_event_rate_ci_high",
    )
    invalid_groups: list[str] = []
    for index, record in enumerate(records):
        key = (
            str(record.get("dataset", "")),
            str(record.get("model", "")),
            str(record.get("documentation_stratum", "")),
        )
        actual.append(key)
        missing_fields = []
        for field in event_rate_fields:
            if field not in record or record[field] is None:
                missing_fields.append(field)
                continue
            try:
                numeric = float(record[field])
            except (TypeError, ValueError):
                missing_fields.append(field)
                continue
            if not np.isfinite(numeric):
                missing_fields.append(field)
        if missing_fields:
            invalid_groups.append(
                f"record {index} is missing valid event-rate fields "
                f"{missing_fields}"
            )
        if int(record.get("n", 0)) <= 0:
            invalid_groups.append(f"record {index} has an empty stratum")
        elif not missing_fields and not (
            0 <= float(record["events"]) <= int(record["n"])
            and 0 <= float(record["severe_events"]) <= int(record["n"])
            and 0.0 <= float(record["event_rate"]) <= 1.0
            and 0.0 <= float(record["severe_event_rate"]) <= 1.0
        ):
            invalid_groups.append(
                f"record {index} has an invalid event rate or event count"
            )

    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set.difference(actual_set))
    unexpected = sorted(actual_set.difference(expected_set))
    duplicates = sorted(
        key for key in actual_set if actual.count(key) > 1
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing groups={missing}")
    if unexpected:
        problems.append(f"unexpected groups={unexpected}")
    if duplicates:
        problems.append(f"duplicate groups={duplicates}")
    if invalid_groups:
        problems.append("; ".join(invalid_groups))
    if problems:
        raise ValueError(
            "External documentation-stratum completeness failed: "
            + " | ".join(problems)
        )
    return {
        "status": "pass",
        "expected_groups": [
            {
                "dataset": dataset,
                "model": model,
                "documentation_stratum": stratum,
            }
            for dataset, model, stratum in expected
        ],
        "observed_group_count": len(actual),
        "datasets": dataset_order,
        "models": model_order,
        "strata": list(STRATA),
        "empty_groups_permitted": False,
    }


def records_to_table(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        any_calibration = record["cumulative_any"]
        severe_calibration = record["cumulative_severe"]
        rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "documentation_stratum": record["documentation_stratum"],
                "q1_cutpoint": record["q1_cutpoint"],
                "q2_cutpoint": record["q2_cutpoint"],
                "n": record["n"],
                "grade_0": record["grade_0"],
                "grade_1": record["grade_1"],
                "grade_2": record["grade_2"],
                "events": record["events"],
                "event_rate": _finite_or_na(record["event_rate"]),
                "event_rate_ci_low": _finite_or_na(
                    record["event_rate_ci_low"]
                ),
                "event_rate_ci_high": _finite_or_na(
                    record["event_rate_ci_high"]
                ),
                "any_events": record["any_events"],
                "any_event_rate": _finite_or_na(record["any_event_rate"]),
                "any_event_rate_ci_low": _finite_or_na(
                    record["any_event_rate_ci_low"]
                ),
                "any_event_rate_ci_high": _finite_or_na(
                    record["any_event_rate_ci_high"]
                ),
                "severe_events": record["severe_events"],
                "severe_event_rate": _finite_or_na(
                    record["severe_event_rate"]
                ),
                "severe_event_rate_ci_low": _finite_or_na(
                    record["severe_event_rate_ci_low"]
                ),
                "severe_event_rate_ci_high": _finite_or_na(
                    record["severe_event_rate_ci_high"]
                ),
                "event_rate_ci_method": record["event_rate_ci_method"],
                "macro_auc": _finite_or_na(record["macro_auc"]),
                "macro_auc_ci_low": _finite_or_na(record["macro_auc_ci_low"]),
                "macro_auc_ci_high": _finite_or_na(record["macro_auc_ci_high"]),
                "bootstrap_clusters": record["bootstrap_clusters"],
                "bootstrap_valid_replicates": record[
                    "bootstrap_valid_replicates"
                ],
                "cluster_column": record["cluster_column"],
                "cumulative_any_events": any_calibration["events"],
                "calibration_slope_any": _finite_or_na(
                    any_calibration["slope"]
                ),
                "calibration_intercept_any": _finite_or_na(
                    any_calibration["intercept"]
                ),
                "ece_any": _finite_or_na(any_calibration["ece"]),
                "cumulative_severe_events": severe_calibration["events"],
                "calibration_slope_severe": _finite_or_na(
                    severe_calibration["slope"]
                ),
                "calibration_intercept_severe": _finite_or_na(
                    severe_calibration["intercept"]
                ),
                "ece_severe": _finite_or_na(severe_calibration["ece"]),
                "interpretation_scope": DOCUMENTATION_GRADIENT_CAVEAT,
            }
        )
    return pd.DataFrame(rows)


def build_analysis(
    development_path: Path,
    external_inputs: Sequence[
        tuple[str, Path, Path, str]
    ],
    models: Sequence[str],
    cutpoint_path: Path,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the prespecified stratified analysis without writing files."""
    development = load_cohort(development_path, cluster_column="subject_key")
    cutpoint_payload = load_or_create_cutpoints(development, cutpoint_path)
    cutpoints = [float(value) for value in cutpoint_payload["cutpoints"]]

    all_records: list[dict[str, Any]] = []
    dataset_audits: list[dict[str, Any]] = []
    external_dataset_names: list[str] = []
    for dataset, cohort_path, prediction_path, cluster_column in external_inputs:
        external_dataset_names.append(dataset)
        cohort = load_cohort(cohort_path, cluster_column=cluster_column)
        for model in models:
            records = analyze_dataset(
                dataset=dataset,
                cohort=cohort,
                prediction_path=prediction_path,
                model=model,
                cutpoints=cutpoints,
                cluster_column=cluster_column,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            )
            all_records.extend(records)
        dataset_audits.append(
            {
                "dataset": dataset,
                "cohort": str(cohort_path),
                "cohort_sha256": _sha256_file(cohort_path),
                "predictions": str(prediction_path),
                "predictions_sha256": _sha256_file(prediction_path),
                "n": int(len(cohort)),
                "grade_0": int(np.sum(cohort["_outcome"] == 0)),
                "grade_1": int(np.sum(cohort["_outcome"] == 1)),
                "grade_2": int(np.sum(cohort["_outcome"] == 2)),
                "cluster_column": cluster_column,
                "cluster_n": int(cohort["_cluster_key"].nunique()),
                "models": list(models),
            }
        )

    completeness = validate_external_strata(
        all_records,
        external_dataset_names,
        models,
    )
    audit = {
        "analysis": "GCS documentation intensity external transport",
        "outline_section": "Results 3.6",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stratification": {
            "method": CUTPOINT_METHOD,
            "development_cohort": str(development_path),
            "development_cohort_sha256": _sha256_file(development_path),
            "development_fingerprint": development_fingerprint(development),
            "development_n": int(len(development)),
            "quantiles": list(QUANTILES),
            "quantile_method": QUANTILE_METHOD,
            "cutpoints": cutpoints,
            "cutpoint_artifact": str(cutpoint_path),
            "stratum_rule": (
                "low: intensity <= q1; middle: q1 < intensity <= q2; "
                "high: intensity > q2"
            ),
            "external_requantilization": False,
        },
        "metrics": {
            "primary_discrimination": "macro one-vs-rest AUROC",
            "auc_missing_class_rule": (
                "NA unless all three outcome grades are present in the stratum"
            ),
            "calibration_targets": {
                "any": "grade >= 1",
                "severe": "grade == 2",
            },
            "calibration_model": (
                "logistic regression of observed outcome on logit(predicted "
                f"cumulative risk), clipping={CALIBRATION_CLIP}"
            ),
            "ece_bins": ECE_BINS,
            "ece_bin_rule": "fixed equal-width bins over [0, 1]",
            "event_rate_definition": {
                "any": "grade >= 1",
                "severe": "grade == 2",
            },
            "event_rate_interval_method": (
                "Wilson score 95% interval, unclustered and deterministic"
            ),
        },
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "base_seed": int(seed),
            "percentile_interval": [2.5, 97.5],
            "within_stratum": True,
            "resampling_unit": "cluster column listed per external dataset",
            "invalid_replicate_rule": (
                "replicates without all three grades are omitted from the CI"
            ),
        },
        "completeness": completeness,
        "interpretation": {
            "scope": DOCUMENTATION_GRADIENT_CAVEAT,
            "documentation_associated_gradient": True,
            "calendar_period_explained": False,
            "case_mix_explained": False,
            "care_intensity_explained": False,
            "causal_effect_identified": False,
        },
        "external_datasets": dataset_audits,
        "results": all_records,
    }
    return all_records, _json_safe(audit)


def write_analysis_outputs(
    records: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    table_path: Path,
    audit_path: Path,
) -> pd.DataFrame:
    table = records_to_table(records)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False, na_rep="NA")
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return table


def _default_external_inputs(
    data_root: Path,
    prediction_root: Path,
) -> list[tuple[str, Path, Path, str]]:
    return [
        (
            "mimic3",
            data_root / "mimic3" / "cohort.csv",
            prediction_root / "mimic3_external.npz",
            "subject_key",
        ),
        (
            "eicu",
            data_root / "eicu" / "cohort.csv",
            prediction_root / "eicu_external.npz",
            "hospital_key",
        ),
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stratify external transport performance using MIMIC-IV-fixed "
            "GCS documentation intensity tertiles."
        )
    )
    parser.add_argument(
        "--development-cohort",
        type=Path,
        default=ROOT / "data" / "processed" / "mimic4" / "cohort.csv",
    )
    parser.add_argument(
        "--mimic3-cohort",
        type=Path,
        default=ROOT / "data" / "processed" / "mimic3" / "cohort.csv",
    )
    parser.add_argument(
        "--mimic3-predictions",
        type=Path,
        default=ROOT / "results" / "predictions" / "mimic3_external.npz",
    )
    parser.add_argument(
        "--eicu-cohort",
        type=Path,
        default=ROOT / "data" / "processed" / "eicu" / "cohort.csv",
    )
    parser.add_argument(
        "--eicu-predictions",
        type=Path,
        default=ROOT / "results" / "predictions" / "eicu_external.npz",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="NPZ prediction arrays to analyze (default: dggru_tr)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cutpoint-output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "audits"
            / "transport_documentation_cutpoints.json"
        ),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "tables"
            / "table_s_transport_documentation.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "audits"
            / "transport_documentation.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    external_inputs = [
        (
            "mimic3",
            args.mimic3_cohort,
            args.mimic3_predictions,
            "subject_key",
        ),
        (
            "eicu",
            args.eicu_cohort,
            args.eicu_predictions,
            "hospital_key",
        ),
    ]
    records, audit = build_analysis(
        development_path=args.development_cohort,
        external_inputs=external_inputs,
        models=args.models,
        cutpoint_path=args.cutpoint_output,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    table = write_analysis_outputs(
        records,
        audit,
        table_path=args.table_output,
        audit_path=args.audit_output,
    )
    print(f"Wrote {len(table)} stratified rows to {args.table_output}")
    print(f"Wrote audit to {args.audit_output}")
    print(f"Fixed cutpoints: {audit['stratification']['cutpoints']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
