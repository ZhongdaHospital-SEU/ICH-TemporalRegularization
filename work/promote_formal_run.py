"""Promote one validated tagged formal run to canonical result filenames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.ich_dynamic.common import RESULTS
from work.validate_formal_run import validate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, context: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{context} hash mismatch: expected {expected}, observed {observed}"
        )


def _copy_atomic(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_file():
        raise FileNotFoundError(f"Tagged artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.promote.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": sha256_file(destination),
    }


@contextmanager
def _promotion_lock():
    """Serialize validation, staging, commit, and rollback."""
    lock_path = RESULTS / "audits" / ".formal_run_promotion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            age_seconds = datetime.now().timestamp() - lock_path.stat().st_mtime
            if age_seconds < 24 * 60 * 60:
                raise RuntimeError(
                    f"Another formal promotion is active: {lock_path}"
                )
            lock_path.unlink()
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        os.write(
            descriptor,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            ).encode("utf-8"),
        )
        yield lock_path
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_path.exists():
            lock_path.unlink()


def _validation_gate(tag: str) -> Path:
    report_path = RESULTS / "audits" / f"formal_run_validation_{tag}.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Validation report is missing; run validator first: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError(f"Validation report did not pass: {report_path}")
    if report.get("tag") != tag:
        raise ValueError(
            f"Validation report tag {report.get('tag')!r} does not match {tag!r}"
        )
    # Re-run the fail-closed checks immediately before promotion so a stale
    # approval report cannot authorise changed artifacts.
    validate(tag)
    return report_path


def promote(tag: str) -> dict[str, object]:
    if not tag:
        raise ValueError("A non-empty formal run tag is required")
    with _promotion_lock() as lock_path:
        validation_report = _validation_gate(tag)
        suffix = f"_{tag}"
        sources = {
            "metrics": RESULTS / f"metrics{suffix}.json",
            "manifest": RESULTS / f"run_manifest{suffix}.json",
            "development_folds": RESULTS / f"mimic4_folds{suffix}.csv",
            "development_predictions": (
                RESULTS / "predictions" / f"mimic4_cv{suffix}.npz"
            ),
            "development_repeated_predictions": (
                RESULTS / "predictions" / f"mimic4_cv{suffix}_repeats.npz"
            ),
            "mimic3_external": (
                RESULTS / "predictions" / f"mimic3_external{suffix}.npz"
            ),
            "eicu_external": (
                RESULTS / "predictions" / f"eicu_external{suffix}.npz"
            ),
        }
        destinations = {
            "metrics": RESULTS / "metrics.json",
            "manifest": RESULTS / "run_manifest.json",
            "development_folds": RESULTS / "mimic4_folds.csv",
            "development_predictions": (
                RESULTS / "predictions" / "mimic4_cv.npz"
            ),
            "development_repeated_predictions": (
                RESULTS / "predictions" / "mimic4_cv_repeats.npz"
            ),
            "mimic3_external": (
                RESULTS / "predictions" / "mimic3_external.npz"
            ),
            "eicu_external": (
                RESULTS / "predictions" / "eicu_external.npz"
            ),
        }
        missing = [str(path) for path in sources.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Formal promotion preflight failed; missing artifacts: "
                + ", ".join(missing)
            )
        source_hashes = {
            name: sha256_file(path) for name, path in sources.items()
        }
        staging = RESULTS / "audits" / f".staging_{tag}"
        backup = RESULTS / "audits" / f".backup_{tag}"
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists():
            shutil.rmtree(backup)
        staging.mkdir(parents=True)
        backup.mkdir(parents=True)
        staged: dict[str, Path] = {}
        try:
            for name, source in sources.items():
                target = staging / source.name
                shutil.copy2(source, target)
                _require_hash(target, source_hashes[name], f"staged {name}")
                staged[name] = target
            manifest = json.loads(
                staged["manifest"].read_text(encoding="utf-8")
            )
            manifest["fold_artifact"]["path"] = str(
                destinations["development_folds"]
            )
            staged["manifest"].write_text(
                json.dumps(manifest, indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            staged_hashes = {
                name: sha256_file(path) for name, path in staged.items()
            }
            for name, source in sources.items():
                if name == "manifest":
                    continue
                if sha256_file(source) != source_hashes[name]:
                    raise RuntimeError(
                        f"Tagged artifact changed during promotion staging: {source}"
                    )
            existing = {
                name: path.exists() for name, path in destinations.items()
            }
            for name, destination in destinations.items():
                if existing[name]:
                    shutil.copy2(destination, backup / destination.name)
            promoted = {}
            try:
                for name, destination in destinations.items():
                    promoted[name] = _copy_atomic(
                        staged[name],
                        destination,
                    )
                    _require_hash(
                        destination,
                        staged_hashes[name],
                        f"promoted {name}",
                    )
                validate("")
            except Exception:
                for name, destination in destinations.items():
                    prior = backup / destination.name
                    if existing[name] and prior.is_file():
                        _copy_atomic(prior, destination)
                    elif not existing[name] and destination.exists():
                        destination.unlink()
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists():
                shutil.rmtree(backup)

        record = {
            "tag": tag,
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "promotion_lock": str(lock_path),
            "validation_report": {
                "path": str(validation_report),
                "sha256": sha256_file(validation_report),
            },
            "source_sha256": source_hashes,
            "canonical_sha256": staged_hashes,
            "transaction": {
                "staged_before_replace": True,
                "backup_created": True,
                "canonical_revalidation": "passed",
            },
            "artifacts": promoted,
        }
        output = RESULTS / "audits" / "formal_run_promotion.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp")
        temporary.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output)
        return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    record = promote(args.tag)
    print(
        f"Promoted {len(record['artifacts'])} canonical artifacts",
        flush=True,
    )


if __name__ == "__main__":
    main()
