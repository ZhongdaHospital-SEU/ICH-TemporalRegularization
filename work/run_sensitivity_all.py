"""Run all supported sensitivity scenarios with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUPPORT_PLAN = RESULTS / "sensitivity" / "sensitivity_support_plan.json"
LOG_DIR = ROOT / "work" / "logs"
PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class Job:
    dataset: str
    scenario: str
    analysis_id: str
    title: str


def _load_jobs(
    datasets: set[str] | None,
) -> tuple[list[Job], list[dict]]:
    plan = json.loads(SUPPORT_PLAN.read_text(encoding="utf-8"))
    jobs: list[Job] = []
    unsupported: list[dict] = []
    for dataset, records in plan["datasets"].items():
        if datasets is not None and dataset not in datasets:
            continue
        for record in records:
            scenario = record["scenario"]
            status = record["status"]
            if not status.get("model_run_allowed"):
                unsupported.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario["scenario_id"],
                        "support": status.get("support"),
                        "reason": status.get("reason"),
                    }
                )
                continue
            jobs.append(
                Job(
                    dataset=dataset,
                    scenario=scenario["scenario_id"],
                    analysis_id=str(scenario["analysis_id"]),
                    title=str(scenario["analysis_title"]),
                )
            )
    return jobs, unsupported


def _run_job(
    job: Job,
    *,
    epochs: int,
    folds: int,
    cv_repeats: int,
    replicates: int,
    worker_threads: int,
    manifest: Path,
) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"sensitivity_{job.dataset}_{job.scenario}.out.log"
    stderr_path = LOG_DIR / f"sensitivity_{job.dataset}_{job.scenario}.err.log"
    command = [
        str(PYTHON),
        "-u",
        str(ROOT / "work" / "run_sensitivity_frozen.py"),
        "--dataset",
        job.dataset,
        "--scenario",
        job.scenario,
        "--epochs",
        str(epochs),
        "--folds",
        str(folds),
        "--cv-repeats",
        str(cv_repeats),
        "--replicates",
        str(replicates),
        "--worker-threads",
        str(worker_threads),
        "--manifest",
        str(manifest),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ICH_TORCH_THREADS"] = str(worker_threads)
    environment["ICH_XGB_THREADS"] = str(worker_threads)
    environment.setdefault("OMP_NUM_THREADS", str(worker_threads))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "dataset": job.dataset,
        "scenario": job.scenario,
        "analysis_id": job.analysis_id,
        "title": job.title,
        "returncode": int(completed.returncode),
        "elapsed_seconds": round(time.time() - started, 1),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RESULTS / "run_manifest_core.json",
    )
    parser.add_argument(
        "--datasets",
        default="mimic4,mimic3,eicu",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    datasets = {
        part.strip() for part in args.datasets.split(",") if part.strip()
    }
    jobs, unsupported = _load_jobs(datasets)
    output = {
        "settings": {
            "epochs": args.epochs,
            "folds": args.folds,
            "cv_repeats": args.cv_repeats,
            "replicates": args.replicates,
            "worker_threads": args.worker_threads,
            "workers": args.workers,
            "manifest": str(manifest),
            "datasets": sorted(datasets),
        },
        "scheduled_jobs": len(jobs),
        "unsupported": unsupported,
        "completed": [],
    }
    print(
        json.dumps(
            {
                "scheduled_jobs": len(jobs),
                "unsupported": len(unsupported),
                "datasets": sorted(datasets),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for job in jobs:
            print(f"{job.dataset}\t{job.scenario}\t{job.title}", flush=True)
        return
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                epochs=args.epochs,
                folds=args.folds,
                cv_repeats=args.cv_repeats,
                replicates=args.replicates,
                worker_threads=args.worker_threads,
                manifest=manifest,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            output["completed"].append(result)
            state = "ok" if result["returncode"] == 0 else "FAILED"
            print(
                f"[{state}] {result['dataset']}/{result['scenario']} "
                f"{result['elapsed_seconds']}s",
                flush=True,
            )
            if result["returncode"] != 0:
                failures += 1
    output_path = RESULTS / "sensitivity" / "run_summary.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}", flush=True)
    if failures:
        raise SystemExit(f"{failures} sensitivity jobs failed")


if __name__ == "__main__":
    main()
