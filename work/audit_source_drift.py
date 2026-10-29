"""Audit source drift without conflating training code and reporting code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "ich_dynamic"
RESULTS = PROJECT_ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_runtime_sources(entrypoint: str = "train_evaluate.py") -> list[str]:
    """Resolve relative imports reachable from the numerical entry point."""
    pending = ["__init__.py", entrypoint]
    visited: set[str] = set()
    while pending:
        filename = pending.pop()
        if filename in visited:
            continue
        path = SOURCE_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Training source is missing: {path}")
        visited.add(filename)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module:
                dependency = f"{node.module.split('.')[0]}.py"
                if (SOURCE_ROOT / dependency).is_file():
                    pending.append(dependency)
                continue
            for alias in node.names:
                dependency = f"{alias.name}.py"
                if (SOURCE_ROOT / dependency).is_file():
                    pending.append(dependency)
    return sorted(visited)


def audit(tag: str) -> dict[str, object]:
    suffix = f"_{tag}" if tag else ""
    manifest_path = RESULTS / f"run_manifest{suffix}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    runtime_files = training_runtime_sources()
    recorded_sources = {
        str(name): str(digest)
        for name, digest in manifest.get("source_sha256", {}).items()
    }
    missing_runtime_sources = sorted(
        set(runtime_files).difference(recorded_sources)
    )
    source_records = []
    for filename, recorded in sorted(recorded_sources.items()):
        path = SOURCE_ROOT / filename
        current = sha256_file(path) if path.is_file() else None
        source_records.append(
            {
                "file": filename,
                "recorded_sha256": recorded,
                "current_sha256": current,
                "matches": current == recorded,
                "model_fit_runtime": filename in runtime_files,
            }
        )

    data_mismatches = []
    for relative, recorded in manifest.get("data_sha256", {}).items():
        path = PROJECT_ROOT / "data" / "processed" / relative
        current = sha256_file(path) if path.is_file() else None
        if current != recorded:
            data_mismatches.append(
                {
                    "file": relative,
                    "recorded_sha256": recorded,
                    "current_sha256": current,
                }
            )

    config_path = PROJECT_ROOT / "configs" / "study.json"
    current_config_hash = sha256_file(config_path)
    config_matches = current_config_hash == manifest.get("config_sha256")
    runtime_mismatches = [
        row["file"]
        for row in source_records
        if row["model_fit_runtime"] and not row["matches"]
    ]
    nonruntime_mismatches = [
        row["file"]
        for row in source_records
        if not row["model_fit_runtime"] and not row["matches"]
    ]
    passed = (
        not missing_runtime_sources
        and not runtime_mismatches
        and not data_mismatches
        and config_matches
    )
    return {
        "status": "pass" if passed else "fail",
        "tag": tag,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_fit_entrypoint": "train_evaluate.py",
        "model_fit_runtime_sources": runtime_files,
        "missing_recorded_runtime_sources": missing_runtime_sources,
        "training_runtime_mismatches": runtime_mismatches,
        "nonruntime_source_mismatches": nonruntime_mismatches,
        "data_hash_mismatches": data_mismatches,
        "config_hash_matches": config_matches,
        "current_config_sha256": current_config_hash,
        "recorded_config_sha256": manifest.get("config_sha256"),
        "source_records": source_records,
        "interpretation": (
            "Numerical results are tied to the recorded model-fitting runtime, "
            "configuration, and hashed processed inputs. Changes outside that "
            "runtime affect reporting only and are listed explicitly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = RESULTS / "audits" / f"source_drift_{args.tag}.json"
    payload = audit(args.tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "training_runtime_mismatches": payload[
                    "training_runtime_mismatches"
                ],
                "missing_recorded_runtime_sources": payload[
                    "missing_recorded_runtime_sources"
                ],
                "nonruntime_source_mismatches": payload[
                    "nonruntime_source_mismatches"
                ],
                "data_hash_mismatches": payload["data_hash_mismatches"],
                "config_hash_matches": payload["config_hash_matches"],
            },
            indent=2,
        )
    )
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
