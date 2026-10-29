"""Assemble the split formal run into the canonical result artifacts.

The formal analysis is executed as disjoint jobs:

* ``core``      - xgb_summary, gru, dggru, dggru_tr (5 CV repeats)
* ``baselines`` - xgb_static, gcs_slope, grud, transformer (5 CV repeats)
* ``stack``     - nested_stack (5 CV repeats, one inner-OOF seed)
* ``ablation``  - negative controls and secondary ablations (1 CV repeat,
                  frozen primary lambda and epoch)

This script merges predictions, recomputes every comparison and paired
interval from the merged arrays, and writes ``metrics.json`` and
``run_manifest.json``. It does not refit any model.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic.common import PROJECT_ROOT, RESULTS
from src.ich_dynamic.train_evaluate import (
    ALL_MODEL_ORDER,
    MODEL_META,
    OUTLINE_MODEL_ORDER,
    SECONDARY_MODEL_ORDER,
    _evaluate_repeated_model,
    _validated_metric_weight,
    add_holm_comparisons,
    comparison_metrics,
    comparison_metrics_repeated,
    load_dataset,
    paired_auc_difference,
    paired_auc_difference_repeated,
    point_metrics,
    repeated_cv_variability,
    sha256_file,
)

COMPONENT_MODELS = {
    "core": ("xgb_summary", "gru", "dggru", "dggru_tr"),
    "baselines": ("xgb_static", "gcs_slope", "grud", "transformer"),
    "stack": ("nested_stack",),
    "ablation": tuple(SECONDARY_MODEL_ORDER)
    + ("dggru_tr_treverse", "dggru_tr_tpermute", "dggru_laplace"),
}
COMPONENT_REPEATS = {
    "core": 5,
    "baselines": 5,
    "stack": 5,
    "ablation": 1,
}
CONTROL_MODELS = (
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _component(tag: str) -> dict:
    suffix = f"_{tag}"
    manifest = _load_json(RESULTS / f"run_manifest{suffix}.json")
    metrics = _load_json(RESULTS / f"metrics{suffix}.json")
    predictions = np.load(
        RESULTS / "predictions" / f"mimic4_cv{suffix}.npz",
        allow_pickle=True,
    )
    repeats = np.load(
        RESULTS / "predictions" / f"mimic4_cv{suffix}_repeats.npz",
        allow_pickle=True,
    )
    external = {
        name: np.load(
            RESULTS / "predictions" / f"{name}_external{suffix}.npz",
            allow_pickle=True,
        )
        for name in ("mimic3", "eicu")
    }
    folds = pd.read_csv(RESULTS / f"mimic4_folds{suffix}.csv")
    return {
        "tag": tag,
        "suffix": suffix,
        "manifest": manifest,
        "metrics": metrics,
        "predictions": predictions,
        "repeats": repeats,
        "external": external,
        "folds": folds,
        "manifest_path": RESULTS / f"run_manifest{suffix}.json",
        "metrics_path": RESULTS / f"metrics{suffix}.json",
    }


def _ids(archive) -> np.ndarray:
    return archive["stay_id"].astype(np.int64)


def _labels(archive) -> np.ndarray:
    return archive["labels"].astype(int)


def _check_components(components: dict[str, dict]) -> None:
    reference = components["core"]
    ref_ids = _ids(reference["predictions"])
    ref_subjects = reference["predictions"]["subject_key"].astype(str)
    ref_labels = _labels(reference["predictions"])
    for tag, component in components.items():
        ids = _ids(component["predictions"])
        subjects = component["predictions"]["subject_key"].astype(str)
        labels = _labels(component["predictions"])
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"{tag} development stay_id order differs from core")
        if not np.array_equal(subjects, ref_subjects):
            raise ValueError(
                f"{tag} development subject_key order differs from core"
            )
        if not np.array_equal(labels, ref_labels):
            raise ValueError(f"{tag} development labels differ from core")
        expected_models = set(COMPONENT_MODELS[tag])
        observed_models = {
            key
            for key in component["predictions"].files
            if key not in {"labels", "stay_id", "subject_key"}
        }
        if observed_models != expected_models:
            raise ValueError(
                f"{tag} prediction keys differ from the declared component: "
                f"missing={sorted(expected_models - observed_models)}, "
                f"extra={sorted(observed_models - expected_models)}"
            )
        if int(component["manifest"]["cv_repeats"]) != COMPONENT_REPEATS[tag]:
            raise ValueError(
                f"{tag} reports {component['manifest']['cv_repeats']} repeats, "
                f"expected {COMPONENT_REPEATS[tag]}"
            )
        for model in COMPONENT_MODELS[tag]:
            if model not in component["metrics"]["development"]:
                raise ValueError(f"{tag} metrics lack development block for {model}")
        if tag != "core":
            core_folds = reference["folds"]
            other_folds = component["folds"]
            shared = other_folds["repeat"].astype(int) == 0
            core_shared = core_folds["repeat"].astype(int) == 0
            if not np.array_equal(
                other_folds.loc[shared, "stay_id"].astype(np.int64).to_numpy(),
                core_folds.loc[core_shared, "stay_id"]
                .astype(np.int64)
                .to_numpy(),
            ):
                raise ValueError(f"{tag} repeat-0 fold assignments differ from core")


def _merge_development(components: dict[str, dict]) -> dict[str, np.ndarray]:
    reference = components["core"]["predictions"]
    payload: dict[str, np.ndarray] = {
        "labels": _labels(reference),
        "stay_id": _ids(reference),
        "subject_key": reference["subject_key"].astype(str),
    }
    for tag, component in components.items():
        for model in COMPONENT_MODELS[tag]:
            payload[model] = component["predictions"][model].astype(np.float64)
    return payload


def _merge_repeats(components: dict[str, dict]) -> dict[str, np.ndarray]:
    reference = components["core"]["repeats"]
    payload: dict[str, np.ndarray] = {
        "labels": _labels(reference),
        "stay_id": _ids(reference),
        "subject_key": reference["subject_key"].astype(str),
    }
    max_repeats = max(COMPONENT_REPEATS.values())
    for repeat in range(max_repeats):
        payload[f"repeat{repeat}_analysis_weight"] = reference[
            f"repeat{repeat}_analysis_weight"
        ].astype(np.float64)
    for tag, component in components.items():
        archive = component["repeats"]
        for model in COMPONENT_MODELS[tag]:
            for repeat in range(COMPONENT_REPEATS[tag]):
                key = f"repeat{repeat}_{model}"
                if key not in archive:
                    raise ValueError(f"{tag} repeated predictions lack {key}")
                payload[key] = archive[key].astype(np.float64)
    return payload


def _merge_external(
    components: dict[str, dict],
    dataset: str,
) -> dict[str, np.ndarray]:
    reference = components["core"]["external"][dataset]
    payload: dict[str, np.ndarray] = {
        "labels": _labels(reference),
        "stay_id": _ids(reference),
        "subject_key": reference["subject_key"].astype(str),
    }
    for tag, component in components.items():
        archive = component["external"][dataset]
        if not np.array_equal(_ids(archive), _ids(reference)):
            raise ValueError(f"{tag} {dataset} stay_id order differs from core")
        if not np.array_equal(_labels(archive), _labels(reference)):
            raise ValueError(f"{tag} {dataset} labels differ from core")
        for model in COMPONENT_MODELS[tag]:
            if model in OUTLINE_MODEL_ORDER:
                payload[model] = archive[model].astype(np.float64)
    missing = set(OUTLINE_MODEL_ORDER) - set(payload)
    if missing:
        raise ValueError(
            f"{dataset} merged predictions lack outline models: "
            + ", ".join(sorted(missing))
        )
    return payload


def _development_metrics(
    dataset,
    components: dict[str, dict],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict, dict[str, list[np.ndarray]], list[np.ndarray | None]]:
    labels = dataset.labels.astype(int)
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    reference = components["core"]["repeats"]
    max_repeats = max(COMPONENT_REPEATS.values())
    weights: list[np.ndarray | None] = []
    for repeat in range(max_repeats):
        key = f"repeat{repeat}_analysis_weight"
        weights.append(
            _validated_metric_weight(
                reference[key].astype(np.float64),
                len(labels),
            )
        )
    repeated: dict[str, list[np.ndarray]] = {}
    output: dict = {}
    for tag, component in components.items():
        archive = component["repeats"]
        for model in COMPONENT_MODELS[tag]:
            values = [
                archive[f"repeat{repeat}_{model}"].astype(np.float64)
                for repeat in range(COMPONENT_REPEATS[tag])
            ]
            repeated[model] = values
            model_weights = weights[: COMPONENT_REPEATS[tag]]
            name, block = _evaluate_repeated_model(
                model,
                labels,
                values,
                groups,
                model_weights,
                replicates,
                seed,
            )
            block["cv_partition_count"] = COMPONENT_REPEATS[tag]
            output[name] = block
    return output, repeated, weights


def _comparisons(
    labels: np.ndarray,
    repeated: dict[str, list[np.ndarray]],
    weights: list[np.ndarray | None],
    *,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    primary = repeated["dggru_tr"]
    comparisons: dict[str, dict] = {}
    paired: dict[str, list[float]] = {}
    paired_validity: dict[str, dict] = {}
    for model, values in repeated.items():
        if model == "dggru_tr":
            continue
        if len(values) == len(primary):
            comparisons[model] = comparison_metrics_repeated(
                labels,
                primary,
                values,
                sample_weight=weights[: len(values)],
            )
            validity: dict = {}
            paired[model] = paired_auc_difference_repeated(
                labels,
                primary,
                values,
                groups,
                replicates=replicates,
                seed=seed + 1,
                sample_weight=weights[: len(values)],
                validity_out=validity,
            )
        else:
            comparisons[model] = comparison_metrics(
                labels,
                primary[0],
                values[0],
                sample_weight=weights[0],
            )
            validity = {}
            paired[model] = paired_auc_difference(
                labels,
                primary[0],
                values[0],
                groups,
                replicates=replicates,
                seed=seed + 1,
                sample_weight=weights[0],
                validity_out=validity,
            )
        paired_validity[model] = validity
    return comparisons, paired, paired_validity


def _external_metrics(
    dataset,
    predictions: dict[str, np.ndarray],
    components: dict[str, dict],
    *,
    replicates: int,
    seed: int,
) -> dict:
    labels = dataset.labels.astype(int)
    if (
        "hospital_key" in dataset.cohort
        and dataset.cohort["hospital_key"].nunique(dropna=True) > 1
    ):
        groups = dataset.cohort["hospital_key"].astype(str).to_numpy()
    else:
        groups = dataset.cohort["subject_key"].astype(str).to_numpy()
    weights = _validated_metric_weight(
        dataset.analysis_weight,
        len(labels),
    )
    output: dict = {}
    for model in OUTLINE_MODEL_ORDER:
        source = None
        for tag, component in components.items():
            if model in COMPONENT_MODELS[tag]:
                source = component["metrics"]["external"][dataset.name][model]
                break
        if source is None:
            raise ValueError(f"No component metric block found for external {model}")
        output[model] = copy.deepcopy(source)
    primary = predictions["dggru_tr"]
    comparisons: dict[str, dict] = {}
    paired: dict[str, list[float]] = {}
    paired_validity: dict[str, dict] = {}
    for model in OUTLINE_MODEL_ORDER:
        if model == "dggru_tr":
            continue
        comparisons[model] = comparison_metrics(
            labels,
            primary,
            predictions[model],
            sample_weight=weights,
        )
        validity: dict = {}
        paired[model] = paired_auc_difference(
            labels,
            primary,
            predictions[model],
            groups,
            replicates=replicates,
            seed=seed + 1,
            sample_weight=weights,
            validity_out=validity,
        )
        paired_validity[model] = validity
    output["comparisons_vs_primary"] = comparisons
    add_holm_comparisons(output)
    output["paired_differences_vs_dggru_tr"] = paired
    output["paired_differences_bootstrap_validity_vs_dggru_tr"] = paired_validity
    return output


def _merge_manifest(
    components: dict[str, dict],
    *,
    model_repeats: dict[str, int],
    merged_repeat_path: Path,
    merged_predictions_path: Path,
    folds_path: Path,
) -> dict:
    base = copy.deepcopy(components["core"]["manifest"])
    base["command"] = [
        "python",
        "work/assemble_split_run.py",
        "--components",
        ",".join(components),
    ]
    base["models"] = list(ALL_MODEL_ORDER)
    base["model_cv_repeats"] = dict(model_repeats)
    base["cv_repeats"] = max(model_repeats.values())
    base["folds"] = 5
    base["component_runs"] = {
        tag: {
            "tag": tag,
            "metrics": str(component["metrics_path"]),
            "metrics_sha256": sha256_file(component["metrics_path"]),
            "manifest": str(component["manifest_path"]),
            "manifest_sha256": sha256_file(component["manifest_path"]),
            "models": list(COMPONENT_MODELS[tag]),
            "cv_repeats": COMPONENT_REPEATS[tag],
        }
        for tag, component in components.items()
    }
    base["lambda_policy"] = {
        "dggru_tr": "frozen_from_lambda_grid",
        "matched_controls": "frozen_to_primary",
        "external_lambda_source": "development_cv_only",
        "selection_artifact": "results/lambda_grid_selection.json",
    }
    base["epoch_policy"] = {
        "dggru_tr": "selected_within_core_outer_folds",
        "matched_controls": "frozen_to_primary_median_epoch",
        "secondary_ablations": "frozen_to_primary_median_epoch",
    }
    base["stacking_inner_oof_seeds"] = 1
    base["lambda_grid"] = list(
        base.get("config", {}).get("lambda_grid", [])
    )
    base["cv_scope"] = {
        "primary_models": 5,
        "nested_stack": 5,
        "negative_controls_and_secondary_ablations": 1,
        "single_partition_comparators": list(CONTROL_MODELS)
        + list(SECONDARY_MODEL_ORDER),
    }
    base["secondary_models"] = list(SECONDARY_MODEL_ORDER)
    base["outline_models"] = list(OUTLINE_MODEL_ORDER)
    base["cv_best_epochs"] = {}
    base["cv_selected_hyperparameters"] = {}
    base["cv_selected_lambdas"] = {}
    base["development_modal_hyperparameters"] = {}
    for tag, component in components.items():
        manifest = component["manifest"]
        for model in COMPONENT_MODELS[tag]:
            if model in manifest.get("cv_best_epochs", {}):
                base["cv_best_epochs"][model] = copy.deepcopy(
                    manifest["cv_best_epochs"][model]
                )
            if model in manifest.get("cv_selected_lambdas", {}):
                base["cv_selected_lambdas"][model] = copy.deepcopy(
                    manifest["cv_selected_lambdas"][model]
                )
            records = manifest.get("cv_selected_hyperparameters", {}).get(model)
            if records:
                base["cv_selected_hyperparameters"][model] = copy.deepcopy(records)
            modal = manifest.get(
                "development_modal_hyperparameters", {}
            ).get(model)
            if modal:
                base["development_modal_hyperparameters"][model] = copy.deepcopy(
                    modal
                )
    base["freeze_from_manifest"] = {
        "path": str(components["core"]["manifest_path"]),
        "sha256": sha256_file(components["core"]["manifest_path"]),
    }
    base["assembled_at_utc"] = datetime.now(timezone.utc).isoformat()
    base["fold_artifact"] = {
        "path": str(folds_path),
        "sha256": sha256_file(folds_path),
    }
    base["result_sha256"] = {
        "metrics": "",
        "development_repeated_predictions": sha256_file(merged_repeat_path),
        "development_probability_ensemble": sha256_file(
            merged_predictions_path
        ),
        "mimic3_external_predictions": sha256_file(
            RESULTS / "predictions" / "mimic3_external.npz"
        ),
        "eicu_external_predictions": sha256_file(
            RESULTS / "predictions" / "eicu_external.npz"
        ),
    }
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        default="core,baselines,stack,ablation",
    )
    args = parser.parse_args()
    tags = tuple(part.strip() for part in args.components.split(","))
    components = {tag: _component(tag) for tag in tags}
    missing_tags = set(COMPONENT_MODELS) - set(components)
    if missing_tags:
        raise ValueError(
            "Assembly requires all four components; missing "
            + ", ".join(sorted(missing_tags))
        )
    _check_components(components)

    model_repeats = {
        model: COMPONENT_REPEATS[tag]
        for tag, models in COMPONENT_MODELS.items()
        for model in models
    }
    if set(model_repeats) != set(ALL_MODEL_ORDER):
        raise ValueError("Component model sets do not cover ALL_MODEL_ORDER")

    repeats_payload = _merge_repeats(components)
    predictions_payload = _merge_development(components)
    repeat_path = RESULTS / "predictions" / "mimic4_cv_repeats.npz"
    predictions_path = RESULTS / "predictions" / "mimic4_cv.npz"
    np.savez_compressed(repeat_path, **repeats_payload)
    np.savez_compressed(predictions_path, **predictions_payload)

    external_payload = {}
    for dataset in ("mimic3", "eicu"):
        external_payload[dataset] = _merge_external(components, dataset)
        np.savez_compressed(
            RESULTS / "predictions" / f"{dataset}_external.npz",
            **external_payload[dataset],
        )

    folds = components["core"]["folds"].copy()
    folds_path = RESULTS / "mimic4_folds.csv"
    folds.to_csv(folds_path, index=False)

    development = load_dataset("mimic4")
    external = {
        name: load_dataset(name) for name in ("mimic3", "eicu")
    }
    seed = int(components["core"]["manifest"]["seed"])
    replicates = int(components["core"]["manifest"]["bootstrap_replicates"])
    development_block, repeated, weights = _development_metrics(
        development,
        components,
        replicates=replicates,
        seed=seed,
    )
    comparisons, paired, paired_validity = _comparisons(
        development.labels.astype(int),
        repeated,
        weights,
        groups=development.cohort["subject_key"].astype(str).to_numpy(),
        replicates=replicates,
        seed=seed,
    )
    development_block["comparisons_vs_primary"] = comparisons
    add_holm_comparisons(development_block)
    development_block["paired_differences_vs_dggru_tr"] = paired
    development_block[
        "paired_differences_bootstrap_validity_vs_dggru_tr"
    ] = paired_validity
    development_block["cv_scope"] = {
        "primary_cv_repeats": max(model_repeats.values()),
        "single_partition_comparators": list(CONTROL_MODELS)
        + list(SECONDARY_MODEL_ORDER),
        "stacking_inner_oof_seeds": 1,
    }

    five_repeat_models = tuple(
        model
        for model, count in model_repeats.items()
        if count == max(model_repeats.values())
    )
    variability = repeated_cv_variability(
        development,
        repeat_path,
        five_repeat_models,
        max(model_repeats.values()),
    )
    for model, count in model_repeats.items():
        if count != 1:
            continue
        values = repeated[model]
        row = point_metrics(
            development.labels,
            values[0],
            sample_weight=weights[0],
        )
        variability[model] = {
            "macro_auc": {
                "values": [float(row["macro_auc"])],
                "mean": float(row["macro_auc"]),
                "sd": 0.0,
                "min": float(row["macro_auc"]),
                "max": float(row["macro_auc"]),
                "partition_count": 1,
            },
            "auc_any_deterioration": {
                "values": [float(row["auc_any_deterioration"])],
                "mean": float(row["auc_any_deterioration"]),
                "sd": 0.0,
                "min": float(row["auc_any_deterioration"]),
                "max": float(row["auc_any_deterioration"]),
                "partition_count": 1,
            },
        }
    development_block["cv_repeat_variability"] = variability

    development_weights = _validated_metric_weight(
        development.analysis_weight,
        len(development.labels),
    )
    ensemble = {
        model: point_metrics(
            development.labels,
            predictions_payload[model],
            sample_weight=development_weights,
        )
        for model in ALL_MODEL_ORDER
    }

    metrics = {
        "development": development_block,
        "external": {
            dataset: _external_metrics(
                external[dataset],
                external_payload[dataset],
                components,
                replicates=replicates,
                seed=seed + 5000,
            )
            for dataset in ("mimic3", "eicu")
        },
        "development_probability_ensemble": {
            "description": (
                "Secondary analysis based on averaging out-of-fold "
                "probabilities across the available CV partitions per model."
            ),
            "point": ensemble,
        },
    }
    metrics_path = RESULTS / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)

    manifest = _merge_manifest(
        components,
        model_repeats=model_repeats,
        merged_repeat_path=repeat_path,
        merged_predictions_path=predictions_path,
        folds_path=folds_path,
    )
    manifest["result_sha256"]["metrics"] = sha256_file(metrics_path)
    with (RESULTS / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Wrote results/metrics.json and results/run_manifest.json")


if __name__ == "__main__":
    main()
