"""Post hoc explanation using manifest-selected complete-development refits."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from work.audit_source_drift import training_runtime_sources

from .common import PROJECT_ROOT, RESULTS, ensure_dirs
from .models import SequencePreprocessor, fit_full_temporal_model
from .train_evaluate import impute_tabular, load_dataset, tabular_frame


FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
AUDITS = RESULTS / "audits"
FORMAL_MANIFEST = RESULTS / "run_manifest.json"
SOURCE_ROOT = PROJECT_ROOT / "src" / "ich_dynamic"
XGB_MODEL = "xgb_summary"
TEMPORAL_MODEL = "dggru_tr"

XGB_PARAMETER_FIELDS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }
)
SEQUENCE_PARAMETER_FIELDS = frozenset(
    {
        "hidden_size",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    }
)


class ExplanationManifestError(RuntimeError):
    """Raised when explanations cannot be tied to a valid formal manifest."""


@dataclass(frozen=True)
class ExplanationSelection:
    manifest_path: Path
    manifest_sha256: str
    features: tuple[str, ...]
    channels: tuple[str, ...]
    sequence_hours: int
    seed: int
    xgb_parameters: dict[str, Any]
    temporal_parameters: dict[str, Any]
    temporal_epochs: int
    lambda_monotone: float
    lambda_smooth: float
    source_hashes: dict[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ExplanationManifestError(f"{context}: missing object {key!r}")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ExplanationManifestError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExplanationManifestError(f"{context} must be numeric") from exc
    if not np.isfinite(result):
        raise ExplanationManifestError(f"{context} must be finite")
    return result


def _positive_int(value: Any, context: str) -> int:
    result = _finite_float(value, context)
    if result <= 0 or not result.is_integer():
        raise ExplanationManifestError(f"{context} must be a positive integer")
    return int(result)


def _validate_parameters(
    parameters: Mapping[str, Any],
    required: frozenset[str],
    context: str,
) -> dict[str, Any]:
    missing = sorted(required.difference(parameters))
    if missing:
        raise ExplanationManifestError(
            f"{context}: missing selected parameter fields {missing}"
        )
    result = dict(parameters)
    for key in required:
        _finite_float(result[key], f"{context}.{key}")
    return result


def _selected_parameter_records(
    manifest: Mapping[str, Any],
    model: str,
    family: str,
    required_fields: frozenset[str],
) -> list[dict[str, Any]]:
    all_records = manifest.get("cv_selected_hyperparameters")
    if not isinstance(all_records, Mapping):
        raise ExplanationManifestError(
            "Formal manifest is missing cv_selected_hyperparameters"
        )
    raw_records = all_records.get(model)
    if not isinstance(raw_records, list) or not raw_records:
        raise ExplanationManifestError(
            f"Formal manifest has no selected hyperparameter records for {model}"
        )
    records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        context = f"cv_selected_hyperparameters.{model}[{index}]"
        if not isinstance(record, Mapping):
            raise ExplanationManifestError(f"{context} must be an object")
        if record.get("family") != family:
            raise ExplanationManifestError(
                f"{context}.family must be {family!r}"
            )
        parameters = record.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ExplanationManifestError(
                f"{context}.parameters must be an object"
            )
        records.append(
            {
                "parameters": _validate_parameters(
                    parameters,
                    required_fields,
                    f"{context}.parameters",
                )
            }
        )
    return records


def _modal_parameters(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, tuple[int, dict[str, Any]]] = {}
    for record in records:
        parameters = dict(record["parameters"])
        canonical = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
        )
        count = counts.get(canonical, (0, parameters))[0] + 1
        counts[canonical] = (count, parameters)
    return dict(max(counts.values(), key=lambda item: item[0])[1])


def _require_recorded_modal_parameters(
    manifest: Mapping[str, Any],
    model: str,
    record_modal: Mapping[str, Any],
    required_fields: frozenset[str],
) -> dict[str, Any]:
    modal = _require_mapping(
        manifest,
        "development_modal_hyperparameters",
        "Formal manifest",
    )
    raw = modal.get(model)
    if not isinstance(raw, Mapping):
        raise ExplanationManifestError(
            "Formal manifest is missing development_modal_hyperparameters."
            f"{model}"
        )
    selected = _validate_parameters(
        raw,
        required_fields,
        f"development_modal_hyperparameters.{model}",
    )
    if json.dumps(selected, sort_keys=True) != json.dumps(
        dict(record_modal), sort_keys=True
    ):
        raise ExplanationManifestError(
            f"development_modal_hyperparameters.{model} does not match the "
            "modal selected CV hyperparameters"
        )
    return selected


def _check_runtime_sources(
    manifest: Mapping[str, Any],
    source_root: Path,
    runtime_sources: Sequence[str],
) -> dict[str, str]:
    recorded = manifest.get("source_sha256")
    if not isinstance(recorded, Mapping):
        raise ExplanationManifestError(
            "Formal manifest is missing source_sha256"
        )
    missing = sorted(set(runtime_sources).difference(recorded))
    if missing:
        raise ExplanationManifestError(
            f"Formal manifest does not record explanation runtime sources {missing}"
        )
    current_hashes: dict[str, str] = {}
    mismatches: list[str] = []
    for filename in runtime_sources:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExplanationManifestError(
                f"Invalid runtime source path in manifest: {filename!r}"
            )
        path = source_root / relative
        if not path.is_file():
            raise ExplanationManifestError(
                f"Explanation runtime source is missing: {path}"
            )
        current = _sha256_file(path)
        current_hashes[filename] = current
        if current != str(recorded[filename]):
            mismatches.append(filename)
    if mismatches:
        raise ExplanationManifestError(
            "Formal manifest runtime source hashes are stale for "
            + ", ".join(mismatches)
        )
    return current_hashes


def load_explanation_manifest(
    manifest_path: Path = FORMAL_MANIFEST,
    *,
    source_root: Path = SOURCE_ROOT,
    runtime_sources: Sequence[str] | None = None,
) -> ExplanationSelection:
    """Load and validate the exact configuration used by formal results."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ExplanationManifestError(
            f"Formal explanation manifest is missing: {path}"
        )
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExplanationManifestError(
            f"Formal explanation manifest is not valid JSON: {path}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ExplanationManifestError("Formal explanation manifest must be an object")

    sources = tuple(
        dict.fromkeys(
            [
                *(runtime_sources or training_runtime_sources()),
                "explain.py",
            ]
        )
    )
    current_sources = _check_runtime_sources(
        manifest,
        Path(source_root),
        sources,
    )

    xgb_records = _selected_parameter_records(
        manifest,
        XGB_MODEL,
        "xgboost",
        XGB_PARAMETER_FIELDS,
    )
    temporal_records = _selected_parameter_records(
        manifest,
        TEMPORAL_MODEL,
        "sequence_models",
        SEQUENCE_PARAMETER_FIELDS,
    )
    xgb_parameters = _require_recorded_modal_parameters(
        manifest,
        XGB_MODEL,
        _modal_parameters(xgb_records),
        XGB_PARAMETER_FIELDS,
    )
    temporal_parameters = _require_recorded_modal_parameters(
        manifest,
        TEMPORAL_MODEL,
        _modal_parameters(temporal_records),
        SEQUENCE_PARAMETER_FIELDS,
    )
    for key in ("n_estimators", "max_depth"):
        _positive_int(xgb_parameters[key], f"{XGB_MODEL}.{key}")
    for key in ("hidden_size", "batch_size", "max_epochs", "patience"):
        _positive_int(temporal_parameters[key], f"{TEMPORAL_MODEL}.{key}")
    dropout = _finite_float(
        temporal_parameters["dropout"],
        f"{TEMPORAL_MODEL}.dropout",
    )
    if not 0.0 <= dropout < 1.0:
        raise ExplanationManifestError(
            "development_modal_hyperparameters.dggru_tr.dropout must be in [0, 1)"
        )
    for key in ("learning_rate", "weight_decay"):
        if _finite_float(
            temporal_parameters[key],
            f"{TEMPORAL_MODEL}.{key}",
        ) < 0:
            raise ExplanationManifestError(
                f"development_modal_hyperparameters.dggru_tr.{key} "
                "must be nonnegative"
            )

    epochs = _require_mapping(manifest, "cv_best_epochs", "Formal manifest")
    epoch_record = epochs.get(TEMPORAL_MODEL)
    if not isinstance(epoch_record, Mapping):
        raise ExplanationManifestError(
            f"Formal manifest is missing cv_best_epochs.{TEMPORAL_MODEL}"
        )
    epoch_values = epoch_record.get("values")
    if not isinstance(epoch_values, list) or not epoch_values:
        raise ExplanationManifestError(
            f"cv_best_epochs.{TEMPORAL_MODEL}.values must be nonempty"
        )
    parsed_epochs = [
        _positive_int(value, f"cv_best_epochs.{TEMPORAL_MODEL}.values")
        for value in epoch_values
    ]
    temporal_epochs = _positive_int(
        epoch_record.get("external_transport_epochs"),
        f"cv_best_epochs.{TEMPORAL_MODEL}.external_transport_epochs",
    )
    if temporal_epochs > int(temporal_parameters["max_epochs"]):
        raise ExplanationManifestError(
            "Selected temporal transport epoch exceeds selected max_epochs"
        )
    expected_epoch = int(round(float(np.median(parsed_epochs))))
    if temporal_epochs != expected_epoch:
        raise ExplanationManifestError(
            "cv_best_epochs.dggru_tr.external_transport_epochs does not match "
            "the recorded median selected epoch"
        )

    lambdas = _require_mapping(
        manifest,
        "cv_selected_lambdas",
        "Formal manifest",
    )
    lambda_record = lambdas.get(TEMPORAL_MODEL)
    if not isinstance(lambda_record, Mapping):
        raise ExplanationManifestError(
            f"Formal manifest is missing cv_selected_lambdas.{TEMPORAL_MODEL}"
        )
    lambda_values = lambda_record.get("values")
    if not isinstance(lambda_values, list) or not lambda_values:
        raise ExplanationManifestError(
            f"cv_selected_lambdas.{TEMPORAL_MODEL}.values must be nonempty"
        )
    parsed_lambdas = [
        _finite_float(
            value,
            f"cv_selected_lambdas.{TEMPORAL_MODEL}.values",
        )
        for value in lambda_values
    ]
    if any(value < 0 for value in parsed_lambdas):
        raise ExplanationManifestError(
            "Selected temporal lambda values must be nonnegative"
        )
    lambda_monotone = _finite_float(
        lambda_record.get("median"),
        f"cv_selected_lambdas.{TEMPORAL_MODEL}.median",
    )
    if lambda_monotone < 0:
        raise ExplanationManifestError("Selected lambda median must be nonnegative")
    if not np.isclose(
        lambda_monotone,
        float(np.median(parsed_lambdas)),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ExplanationManifestError(
            "cv_selected_lambdas.dggru_tr.median does not match its values"
        )
    smooth_ratio = _finite_float(
        manifest.get("lambda_smooth_ratio"),
        "lambda_smooth_ratio",
    )
    if smooth_ratio < 0:
        raise ExplanationManifestError("lambda_smooth_ratio must be nonnegative")
    lambda_smooth = smooth_ratio * lambda_monotone

    features = manifest.get("features")
    if not isinstance(features, list) or not features or not all(
        isinstance(value, str) and value for value in features
    ):
        raise ExplanationManifestError(
            "Formal manifest must contain a nonempty string feature list"
        )
    config = _require_mapping(manifest, "config", "Formal manifest")
    channels = config.get("channels")
    if not isinstance(channels, list) or not channels or not all(
        isinstance(value, str) and value for value in channels
    ):
        raise ExplanationManifestError(
            "Formal manifest config.channels must be a nonempty string list"
        )
    sequence_hours = _positive_int(
        config.get("sequence_hours"),
        "config.sequence_hours",
    )
    seed = _positive_int(manifest.get("seed"), "seed")

    if path.read_bytes() != raw:
        raise ExplanationManifestError(
            "Formal explanation manifest changed while it was being validated"
        )
    return ExplanationSelection(
        manifest_path=path.resolve(),
        manifest_sha256=_sha256_bytes(raw),
        features=tuple(features),
        channels=tuple(channels),
        sequence_hours=sequence_hours,
        seed=seed,
        xgb_parameters=xgb_parameters,
        temporal_parameters=temporal_parameters,
        temporal_epochs=temporal_epochs,
        lambda_monotone=lambda_monotone,
        lambda_smooth=lambda_smooth,
        source_hashes=current_sources,
    )


def _verify_manifest_unchanged(selection: ExplanationSelection) -> None:
    if _sha256_file(selection.manifest_path) != selection.manifest_sha256:
        raise ExplanationManifestError(
            "Formal explanation manifest changed after selection was loaded"
        )


def _fit_tree_model(
    train_x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    parameters: Mapping[str, Any],
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        **dict(parameters),
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(train_x, labels, verbose=False)
    return model


def tree_shap_top15(selection: ExplanationSelection) -> pd.DataFrame:
    """Refit the manifest-selected Summary XGBoost and compute TreeSHAP."""
    dataset = load_dataset("mimic4", feature_columns=list(selection.features))
    frame = tabular_frame(dataset, "all")
    values, _ = impute_tabular(frame, frame)
    labels = dataset.labels.astype(int)
    model = _fit_tree_model(
        values,
        labels,
        selection.seed,
        selection.xgb_parameters,
    )

    contributor = model.get_booster().predict(
        xgb.DMatrix(values),
        pred_contribs=True,
    )
    expected = len(frame.columns) + 1
    if contributor.shape[1] != 3 * expected:
        raise RuntimeError(
            "Unexpected multiclass TreeSHAP shape: "
            f"{contributor.shape}; expected {3 * expected} columns"
        )
    contributions = contributor.reshape(len(values), 3, expected)[:, :, :-1]
    importance = np.mean(np.abs(contributions), axis=(0, 1))
    result = (
        pd.DataFrame(
            {
                "feature": frame.columns.astype(str),
                "mean_abs_tree_shap": importance,
                "manifest_sha256": selection.manifest_sha256,
            }
        )
        .sort_values("mean_abs_tree_shap", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    result.to_csv(TABLES / "table_s_shap_top15.csv", index=False)
    return result


def temporal_gradient_importance(
    selection: ExplanationSelection,
) -> pd.DataFrame:
    """Refit manifest-selected DG-GRU+TR and aggregate absolute gradients."""
    dataset = load_dataset("mimic4", feature_columns=list(selection.features))
    preprocessor = SequencePreprocessor.fit(
        dataset.sequences,
        dataset.masks,
        dataset.static_all.to_numpy(dtype=np.float32),
    )
    sequences, sequence_masks = preprocessor.transform_sequence(
        dataset.sequences,
        dataset.masks,
    )
    static = preprocessor.transform_static(
        dataset.static_all.to_numpy(dtype=np.float32)
    )
    parameters = selection.temporal_parameters
    model = fit_full_temporal_model(
        sequences,
        sequence_masks,
        static,
        dataset.labels,
        use_graph=True,
        use_regularization=True,
        architecture="dggru",
        seed=selection.seed,
        lambda_monotone=selection.lambda_monotone,
        lambda_smooth=selection.lambda_smooth,
        epochs=selection.temporal_epochs,
        batch_size=int(parameters["batch_size"]),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        hidden=int(parameters["hidden_size"]),
        dropout=float(parameters["dropout"]),
    )
    model.eval()
    n_values = len(selection.channels)
    total = np.zeros((selection.sequence_hours, n_values), dtype=float)
    batch_size = 128
    for start in range(0, len(sequences), batch_size):
        stop = min(start + batch_size, len(sequences))
        sequence = torch.from_numpy(sequences[start:stop].copy()).requires_grad_(True)
        mask = torch.from_numpy(sequence_masks[start:stop].copy())
        static_batch = torch.from_numpy(static[start:stop].copy())
        probabilities, _, _ = model(sequence, mask, static_batch)
        any_probability = probabilities[:, 1:].sum(dim=1).clamp(1e-6, 1 - 1e-6)
        objective = torch.special.logit(any_probability).sum()
        gradient, = torch.autograd.grad(objective, sequence)
        values_gradient = gradient[:, :, :n_values].abs().numpy()
        total += values_gradient.sum(axis=0)
    importance = total / max(len(sequences), 1)
    long = pd.DataFrame(
        [
            {
                "hour": hour,
                "channel": channel,
                "mean_abs_gradient": float(importance[hour, index]),
                "manifest_sha256": selection.manifest_sha256,
            }
            for hour in range(importance.shape[0])
            for index, channel in enumerate(selection.channels)
        ]
    )
    long.to_csv(TABLES / "table_s_time_dependent_importance.csv", index=False)
    return long


def plot_interpretation(
    shap_table: pd.DataFrame,
    temporal: pd.DataFrame,
    selection: ExplanationSelection,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    ordered = shap_table.iloc[::-1]
    axes[0].barh(
        np.arange(len(ordered)),
        ordered["mean_abs_tree_shap"],
        color="#D55E00",
    )
    axes[0].set_yticks(np.arange(len(ordered)))
    axes[0].set_yticklabels(ordered["feature"], fontsize=8)
    axes[0].set_xlabel("Mean absolute TreeSHAP value")
    axes[0].set_title("Top 15 predictors (Summary XGBoost)", fontsize=10)
    axes[0].grid(axis="x", alpha=0.2)

    matrix = (
        temporal.pivot(index="channel", columns="hour", values="mean_abs_gradient")
        .reindex(selection.channels)
    )
    image = axes[1].imshow(matrix, aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(len(matrix.index)))
    axes[1].set_yticklabels(matrix.index, fontsize=8)
    axes[1].set_xticks(np.arange(0, matrix.shape[1], 4))
    axes[1].set_xticklabels(np.arange(0, matrix.shape[1], 4))
    axes[1].set_xlabel("ICU hour")
    axes[1].set_title("DG-GRU+TR time-dependent gradient importance", fontsize=10)
    fig.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure6_interpretation.png", dpi=300)
    fig.savefig(FIGURES / "figure6_interpretation.pdf")
    plt.close(fig)


def _output_binding(
    selection: ExplanationSelection,
    output_paths: Sequence[Path],
) -> dict[str, Any]:
    _verify_manifest_unchanged(selection)
    outputs = {str(path): _sha256_file(path) for path in output_paths}
    return {
        "status": "pass",
        "manifest": str(selection.manifest_path),
        "manifest_sha256": selection.manifest_sha256,
        "runtime_sources_current": selection.source_hashes,
        "selected_models": {
            XGB_MODEL: {
                "parameters": selection.xgb_parameters,
            },
            TEMPORAL_MODEL: {
                "parameters": selection.temporal_parameters,
                "epochs": selection.temporal_epochs,
                "lambda_monotone": selection.lambda_monotone,
                "lambda_smooth": selection.lambda_smooth,
            },
        },
        "outputs": outputs,
    }


def write_manifest_binding(
    selection: ExplanationSelection,
    output_paths: Sequence[Path],
    path: Path = AUDITS / "explain_manifest_binding.json",
) -> dict[str, Any]:
    binding = _output_binding(selection, output_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(binding, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _verify_manifest_unchanged(selection)
    return binding


def main() -> None:
    ensure_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    selection = load_explanation_manifest()
    shap_table = tree_shap_top15(selection)
    temporal = temporal_gradient_importance(selection)
    plot_interpretation(shap_table, temporal, selection)
    output_paths = [
        TABLES / "table_s_shap_top15.csv",
        TABLES / "table_s_time_dependent_importance.csv",
        FIGURES / "figure6_interpretation.png",
        FIGURES / "figure6_interpretation.pdf",
    ]
    write_manifest_binding(selection, output_paths)
    print(
        "[explain] saved manifest-bound TreeSHAP, time-dependent importance, "
        f"Figure 6, and binding for {selection.manifest_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
