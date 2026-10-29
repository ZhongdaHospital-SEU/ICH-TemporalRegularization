"""Re-run the endpoint-redefinition sensitivity scenarios after the label fix.

Only scenarios 1/2/3 change how the endpoint is reconstructed. Earlier saved
results for these scenarios used the primary label and are therefore invalid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "work" / "logs"
PYTHON = Path(sys.executable)

JOBS = [
    ("mimic4", "s1_baseline_last"),
    ("mimic4", "s2_persistence_1h"),
    ("mimic4", "s3_exclude_continuous_sedation"),
    ("mimic3", "s1_baseline_last"),
    ("mimic3", "s2_persistence_1h"),
    ("eicu", "s1_baseline_last"),
    ("eicu", "s2_persistence_1h"),
]

EPOCHS = 25
FOLDS = 5
CV_REPEATS = 1
REPLICATES = 500
WORKER_THREADS = 2
MANIFEST = RESULTS / "run_manifest_core.json"


def run_job(dataset: str, scenario: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"sensitivity_rerun_{dataset}_{scenario}.out.log"
    stderr_path = LOG_DIR / f"sensitivity_rerun_{dataset}_{scenario}.err.log"
    command = [
        str(PYTHON),
        "-u",
        str(ROOT / "work" / "run_sensitivity_frozen.py"),
        "--dataset",
        dataset,
        "--scenario",
        scenario,
        "--epochs",
        str(EPOCHS),
        "--folds",
        str(FOLDS),
        "--cv-repeats",
        str(CV_REPEATS),
        "--replicates",
        str(REPLICATES),
        "--worker-threads",
        str(WORKER_THREADS),
        "--manifest",
        str(MANIFEST),
        "--force",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ICH_TORCH_THREADS"] = str(WORKER_THREADS)
    environment["ICH_XGB_THREADS"] = str(WORKER_THREADS)
    environment.setdefault("OMP_NUM_THREADS", str(WORKER_THREADS))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    return {
        "dataset": dataset,
        "scenario": scenario,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 1),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> None:
    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(JOBS)) as executor:
        futures = {
            executor.submit(run_job, dataset, scenario): (dataset, scenario)
            for dataset, scenario in JOBS
        }
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            print(json.dumps(record), flush=True)
    summary = {"jobs": completed}
    path = RESULTS / "sensitivity" / "rerun_fix_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
