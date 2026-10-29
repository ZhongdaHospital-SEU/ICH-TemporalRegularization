"""Run one outline sensitivity scenario with the primary lambda frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ich_dynamic import sensitivity_analysis as sensitivity
from src.ich_dynamic.common import CONFIG_PATH, PROCESSED, RESULTS


RUNTIME_SOURCE_PATHS = (
    "src/ich_dynamic/sensitivity_analysis.py",
    "src/ich_dynamic/models.py",
    "src/ich_dynamic/train_evaluate.py",
    "src/ich_dynamic/common.py",
    "work/run_sensitivity_frozen.py",
)
DATA_LINEAGE_FILES = (
    "cohort_pre_filter.csv",
    "gcs_events.csv",
    "rass_events.csv",
)
DATASET_NAMES = ("mimic4", "mimic3", "eicu")
SUPPORT_PLAN_SCHEMA_VERSION = 2
MAX_SUPPORT_PLAN_AGE_HOURS = 24
MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Required support-plan input is missing: {path}") from exc


def _file_binding(path: Path) -> dict[str, str]:
    resolved = _resolved(path)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def support_plan_path(results_root: Path | None = None) -> Path:
    return (
        Path(results_root if results_root is not None else RESULTS)
        / "sensitivity"
        / "sensitivity_support_plan.json"
    )


def _lineage_binding(
    *,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    project_root = Path(project_root if project_root is not None else PROJECT_ROOT)
    processed_root = Path(
        processed_root if processed_root is not None else PROCESSED
    )
    config_path = Path(config_path if config_path is not None else CONFIG_PATH)
    runtime_sources = {
        relative: _file_binding(project_root / relative)
        for relative in RUNTIME_SOURCE_PATHS
    }
    data = {
        f"{dataset}/{filename}": _file_binding(
            processed_root / dataset / filename
        )
        for dataset in DATASET_NAMES
        for filename in DATA_LINEAGE_FILES
    }
    return {
        "runtime_sources": runtime_sources,
        "config": _file_binding(config_path),
        "data": data,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("mimic4", "mimic3", "eicu"),
        default=None,
    )
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--list-supported",
        action="store_true",
        help="Print the prespecified dataset-by-scenario support plan as JSON.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Source run manifest used to freeze the primary lambda and model "
            "hyperparameters. Defaults to results/run_manifest.json."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def frozen_primary_config(
    manifest_path: Path | None = None,
) -> tuple[dict, dict, Path]:
    manifest_path = Path(
        manifest_path if manifest_path is not None else RESULTS / "run_manifest.json"
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "The canonical formal run manifest is missing. Complete, validate, "
            f"and promote the primary run before sensitivity fitting: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("cv_repeats", -1)) != 5:
        raise ValueError("The canonical primary run must use 5x5 repeated CV.")
    selection = (
        manifest.get("cv_selected_lambdas", {})
        .get("dggru_tr", {})
        .get("median")
    )
    if selection is None:
        raise ValueError("The canonical manifest has no selected DG-GRU + TR lambda.")
    config = sensitivity.load_config()
    config["lambda_monotone"] = float(selection)
    config["lambda_smooth"] = (
        float(selection) * float(config["lambda_smooth_ratio"])
    )
    # Sensitivity analyses inherit the primary tuning result. They do not
    # reselect lambda on the sensitivity endpoint.
    config["lambda_grid"] = []
    modal = manifest.get("development_modal_hyperparameters", {})
    xgb_parameters = modal.get("xgb_summary")
    sequence_parameters = modal.get("dggru_tr")
    if not isinstance(xgb_parameters, dict) or not isinstance(
        sequence_parameters, dict
    ):
        raise ValueError(
            "The canonical manifest lacks development-derived hyperparameters."
        )
    config["hyperparameters"]["xgboost"]["candidates"] = [xgb_parameters]
    config["hyperparameters"]["sequence_models"]["candidates"] = [
        sequence_parameters
    ]
    return config, manifest, manifest_path


def _support_inputs(
    dataset: str,
    processed_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    import pandas as pd

    root = Path(processed_root if processed_root is not None else PROCESSED)
    prefilter = pd.read_csv(root / dataset / "cohort_pre_filter.csv")
    gcs = pd.read_csv(root / dataset / "gcs_events.csv")
    rass_path = root / dataset / "rass_events.csv"
    rass = pd.read_csv(rass_path) if rass_path.is_file() else None
    return prefilter, gcs, rass


def support_plan(
    *,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    datasets: dict[str, list[dict]] = {}
    for dataset in DATASET_NAMES:
        prefilter, gcs, rass = _support_inputs(dataset, processed_root)
        datasets[dataset] = []
        for spec in sensitivity.SCENARIOS:
            status = sensitivity.assess_scenario_support(
                spec,
                dataset,
                prefilter=prefilter,
                gcs=gcs,
                rass_events=rass,
            )
            datasets[dataset].append(
                {
                    "scenario": asdict(spec),
                    "status": asdict(status),
                }
            )
    return {
        "schema_version": SUPPORT_PLAN_SCHEMA_VERSION,
        "outline_section": "2.8",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "binding": _lineage_binding(
            project_root=project_root,
            processed_root=processed_root,
            config_path=config_path,
        ),
        "datasets": datasets,
    }


def _parse_generated_at(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Sensitivity support plan has no generation timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Sensitivity support plan generation timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(
            "Sensitivity support plan generation timestamp must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _validate_binding_section(
    recorded: object,
    expected: dict,
    *,
    label: str,
) -> None:
    if not isinstance(recorded, dict):
        raise ValueError(f"Sensitivity support plan {label} binding is missing")
    if set(recorded) != set(expected):
        missing = sorted(set(expected) - set(recorded))
        unexpected = sorted(set(recorded) - set(expected))
        raise ValueError(
            f"Sensitivity support plan {label} binding mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, current in expected.items():
        value = recorded[key]
        if not isinstance(value, dict):
            raise ValueError(
                f"Sensitivity support plan {label} binding {key!r} is invalid"
            )
        if set(value) != {"path", "sha256"}:
            raise ValueError(
                f"Sensitivity support plan {label} binding {key!r} "
                "must contain exactly path and sha256"
            )
        recorded_path = value["path"]
        if not isinstance(recorded_path, str) or not Path(recorded_path).is_absolute():
            raise ValueError(
                f"Sensitivity support plan {label} binding {key!r} "
                "has no absolute path"
            )
        if Path(recorded_path).resolve() != Path(current["path"]).resolve():
            raise ValueError(
                f"Sensitivity support plan {label} binding {key!r} "
                "points to another path"
            )
        recorded_hash = value["sha256"]
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise ValueError(
                f"Sensitivity support plan {label} binding {key!r} "
                "has no valid SHA256"
            )
        if recorded_hash.lower() != current["sha256"]:
            raise ValueError(
                f"Sensitivity support plan is stale: {label} hash changed "
                f"for {key}"
            )


def validate_support_plan(
    plan: dict,
    *,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Sensitivity support plan must be a JSON object")
    if plan.get("schema_version") != SUPPORT_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "Sensitivity support plan schema is obsolete; regenerate it"
        )
    if plan.get("outline_section") != "2.8":
        raise ValueError("Sensitivity support plan outline section must be 2.8")
    generated_at = _parse_generated_at(plan.get("generated_at_utc"))
    current_time = now if now is not None else datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("Support-plan validation time must be timezone-aware")
    current_time = current_time.astimezone(timezone.utc)
    if generated_at - current_time > MAX_FUTURE_CLOCK_SKEW:
        raise ValueError(
            "Sensitivity support plan generation timestamp is in the future"
        )
    if current_time - generated_at > timedelta(
        hours=MAX_SUPPORT_PLAN_AGE_HOURS
    ):
        raise ValueError(
            "Sensitivity support plan is older than "
            f"{MAX_SUPPORT_PLAN_AGE_HOURS} hours; regenerate it"
        )

    binding = plan.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Sensitivity support plan lineage binding is missing")
    expected_binding = _lineage_binding(
        project_root=project_root,
        processed_root=processed_root,
        config_path=config_path,
    )
    _validate_binding_section(
        binding.get("runtime_sources"),
        expected_binding["runtime_sources"],
        label="runtime source",
    )
    _validate_binding_section(
        {"config": binding.get("config")},
        {"config": expected_binding["config"]},
        label="configuration",
    )
    _validate_binding_section(
        binding.get("data"),
        expected_binding["data"],
        label="data",
    )

    datasets = plan.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_NAMES):
        raise ValueError(
            "Sensitivity support plan must contain exactly the three datasets"
        )
    scenario_ids = {spec.scenario_id for spec in sensitivity.SCENARIOS}
    for dataset in DATASET_NAMES:
        records = datasets[dataset]
        if not isinstance(records, list):
            raise ValueError(
                f"Sensitivity support plan dataset {dataset!r} is invalid"
            )
        recorded_ids = [
            item.get("scenario", {}).get("scenario_id")
            for item in records
            if isinstance(item, dict)
        ]
        if (
            len(recorded_ids) != len(records)
            or set(recorded_ids) != scenario_ids
            or len(recorded_ids) != len(set(recorded_ids))
        ):
            raise ValueError(
                f"Sensitivity support plan scenarios are incomplete or duplicated "
                f"for {dataset}"
            )


def write_support_plan(
    *,
    path: Path | None = None,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
    results_root: Path | None = None,
) -> tuple[dict, Path]:
    plan = support_plan(
        project_root=project_root,
        processed_root=processed_root,
        config_path=config_path,
    )
    validate_support_plan(
        plan,
        project_root=project_root,
        processed_root=processed_root,
        config_path=config_path,
    )
    output = (
        Path(path)
        if path is not None
        else support_plan_path(results_root)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return plan, output


def load_support_plan(
    *,
    path: Path | None = None,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    output = (
        Path(path)
        if path is not None
        else support_plan_path(results_root)
    )
    if not output.is_file():
        raise FileNotFoundError(
            "Sensitivity support plan is missing. Generate it with "
            f"--list-supported first: {output}"
        )
    try:
        plan = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read sensitivity support plan: {output}"
        ) from exc
    validate_support_plan(
        plan,
        project_root=project_root,
        processed_root=processed_root,
        config_path=config_path,
        now=now,
    )
    return plan


def require_supported(
    dataset: str,
    scenario: str,
    *,
    plan_path: Path | None = None,
    project_root: Path | None = None,
    processed_root: Path | None = None,
    config_path: Path | None = None,
    results_root: Path | None = None,
) -> sensitivity.ScenarioStatus:
    plan = load_support_plan(
        path=plan_path,
        project_root=project_root,
        processed_root=processed_root,
        config_path=config_path,
        results_root=results_root,
    )
    matching = [
        item
        for item in plan["datasets"][dataset]
        if item["scenario"]["scenario_id"] == scenario
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Sensitivity support plan has no unique record for {dataset}/{scenario}"
        )
    recorded_status = matching[0].get("status")
    if not isinstance(recorded_status, dict):
        raise ValueError(
            f"Sensitivity support plan status is invalid for {dataset}/{scenario}"
        )
    prefilter, gcs, rass = _support_inputs(dataset, processed_root)
    status = sensitivity.assess_scenario_support(
        scenario,
        dataset,
        prefilter=prefilter,
        gcs=gcs,
        rass_events=rass,
    )
    if asdict(status) != recorded_status:
        raise ValueError(
            "Sensitivity support plan status is stale for "
            f"{dataset}/{scenario}; regenerate it"
        )
    if not status.model_run_allowed:
        raise ValueError(
            f"{dataset}/{scenario} must not be fitted: "
            f"{status.support}: {status.reason}"
        )
    return status


def record_path(dataset: str, scenario_id: str) -> Path:
    return (
        RESULTS
        / "sensitivity"
        / dataset
        / "records"
        / f"{scenario_id}.json"
    )


def main() -> None:
    args = parse_args()
    if args.list_supported:
        plan, _ = write_support_plan()
        print(json.dumps(plan, indent=2), flush=True)
        return
    if args.dataset is None or args.scenario is None:
        raise ValueError("--dataset and --scenario are required for model fitting.")
    require_supported(args.dataset, args.scenario)
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else RESULTS / "run_manifest.json"
    )
    config, manifest, manifest_path = frozen_primary_config(manifest_path)
    signature = hashlib.sha256(
        json.dumps(
            {
                "formal_manifest_sha256": sha256_file(manifest_path),
                "dataset": args.dataset,
                "scenario": args.scenario,
                "lambda_monotone": float(config["lambda_monotone"]),
                "lambda_smooth": float(config["lambda_smooth"]),
                "xgboost_hyperparameters": config["hyperparameters"][
                    "xgboost"
                ]["candidates"][0],
                "sequence_hyperparameters": config["hyperparameters"][
                    "sequence_models"
                ]["candidates"][0],
                "epochs": args.epochs,
                "folds": args.folds,
                "cv_repeats": args.cv_repeats,
                "replicates": args.replicates,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    output = record_path(args.dataset, args.scenario)
    if output.is_file() and not args.force:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("run_signature") == signature:
            artifact = previous.get("result", {}).get("artifact")
            if artifact is None or Path(artifact).is_file():
                print(f"reuse {args.dataset}/{args.scenario}", flush=True)
                return

    original_load_config = sensitivity.load_config
    sensitivity.load_config = lambda: config
    try:
        result = sensitivity.run_scenario(
            args.scenario,
            dataset=args.dataset,
            epochs=args.epochs,
            folds=args.folds,
            repeats=args.cv_repeats,
            replicates=args.replicates,
            workers=1,
            worker_threads=args.worker_threads,
        )
    finally:
        sensitivity.load_config = original_load_config

    payload = {
        "run_signature": signature,
        "dataset": args.dataset,
        "scenario": args.scenario,
        "formal_manifest_sha256": sha256_file(manifest_path),
        "formal_model_seeds": manifest.get("model_seeds"),
        "fitting_protocol": {
            "lambda_selection": "frozen_primary_median",
            "lambda_monotone": float(config["lambda_monotone"]),
            "lambda_smooth": float(config["lambda_smooth"]),
            "xgboost_hyperparameters": config["hyperparameters"]["xgboost"][
                "candidates"
            ][0],
            "sequence_hyperparameters": config["hyperparameters"][
                "sequence_models"
            ]["candidates"][0],
            "epochs": args.epochs,
            "folds": args.folds,
            "cv_repeats": args.cv_repeats,
            "replicates": args.replicates,
            "workers": 1,
            "worker_threads": args.worker_threads,
        },
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"completed {args.dataset}/{args.scenario}", flush=True)


if __name__ == "__main__":
    main()
