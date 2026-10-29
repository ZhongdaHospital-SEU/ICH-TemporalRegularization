"""Fail-fast checks tying the manuscript, predictions, and manifest together."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from work.audit_source_drift import training_runtime_sources


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_manifest() -> dict:
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    runtime_sources = set(training_runtime_sources())
    for relative in sorted(runtime_sources):
        expected = manifest["source_sha256"].get(relative)
        require(expected is not None, f"Runtime source not hashed in manifest: {relative}")
        actual = sha256_file(ROOT / "src" / "ich_dynamic" / relative)
        require(actual == expected, f"Training-runtime hash mismatch: {relative}")
    require(
        sha256_file(ROOT / "configs" / "study.json")
        == manifest["config_sha256"],
        "Configuration hash mismatch",
    )
    for relative, expected in manifest["data_sha256"].items():
        actual = sha256_file(PROCESSED / relative)
        require(actual == expected, f"Input data hash mismatch: {relative}")
    result_paths = {
        "metrics": RESULTS / "metrics.json",
        "development_probability_ensemble": (
            RESULTS / "predictions" / "mimic4_cv.npz"
        ),
        "development_repeated_predictions": (
            RESULTS / "predictions" / "mimic4_cv_repeats.npz"
        ),
        "mimic3_external_predictions": (
            RESULTS / "predictions" / "mimic3_external.npz"
        ),
        "eicu_external_predictions": (
            RESULTS / "predictions" / "eicu_external.npz"
        ),
    }
    for relative, expected in manifest["result_sha256"].items():
        path = result_paths.get(relative)
        if path is None:
            require(
                False,
                f"Unrecognized manifest result artifact: {relative}",
            )
        require(sha256_file(path) == expected, f"Result hash mismatch: {relative}")
    require(manifest["cv_repeats"] == 5, "Primary analysis is not five-repeat CV")
    expected_seeds = list(range(int(config["model_seeds"])))
    require(
        manifest["model_seeds"] == expected_seeds,
        "Manifest model seeds differ from configs/study.json",
    )
    return manifest


def check_prediction_alignment() -> None:
    cases = {
        "mimic4_cv": "mimic4",
        "mimic4_cv_repeats": "mimic4",
        "mimic3_external": "mimic3",
        "eicu_external": "eicu",
    }
    for prefix, cohort_name in cases.items():
        path = RESULTS / "predictions" / f"{prefix}.npz"
        cohort = pd.read_csv(PROCESSED / cohort_name / "cohort.csv")
        with np.load(path, allow_pickle=True) as archive:
            require(
                np.array_equal(
                    archive["stay_id"].astype(np.int64),
                    cohort["stay_id"].to_numpy(dtype=np.int64),
                ),
                f"{prefix}: stay_id order mismatch",
            )
            require(
                np.array_equal(
                    archive["labels"].astype(int),
                    cohort["outcome"].astype(int).to_numpy(),
                ),
                f"{prefix}: labels mismatch",
            )


def check_manuscript(manifest: dict) -> None:
    text = (OUTPUTS / "manuscript.md").read_text(encoding="utf-8")
    forbidden = (
        "prespecified",
        "116-predictor",
        "119-predictor",
        "119-variable",
        "29 hospitals",
        "MIMIC-III temporal external",
        "standardized temporal validation",
        "transport variation",
        "The strongest internal model",
        "deterministic reruns",
        "derived aggregate tables, predictions, model manifests",
    )
    for phrase in forbidden:
        require(phrase.lower() not in text.lower(), f"Stale wording remains: {phrase}")
    for phrase in (
        "same-institution external cohort from a different database generation",
        "conditional patient-cluster bootstrap interval",
        "combined CV-partition and training stochasticity",
        "availability and variance screening",
        "observed RASS",
    ):
        require(phrase.lower() in text.lower(), f"Required wording absent: {phrase}")
    development = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    for count in (
        len(development),
        int((development["outcome"] == 0).sum()),
        int((development["outcome"] == 1).sum()),
        int((development["outcome"] == 2).sum()),
    ):
        require(f"{count:,}" in text, f"Current cohort count absent: {count}")
    require(
        manifest.get("temporal_static_feature_count") == len(manifest["features"]),
        "The neural static-feature count does not match the manifest feature list",
    )
    age_audit = json.loads(
        (RESULTS / "audits" / "mimic3_age_sentinel.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        age_audit["analysis_cohort"]["sentinel_stays"]
        == age_audit["analysis_cohort"]["age_gt89_stays"],
        "MIMIC-III age-sentinel indicator count does not match sentinel stays",
    )


def check_artifacts() -> None:
    required = (
        OUTPUTS / "manuscript.docx",
        OUTPUTS / "manuscript.pdf",
        OUTPUTS / "Supplementary_Material.docx",
        OUTPUTS / "Supplementary_Material.pdf",
        OUTPUTS / "TRIPOD_AI_checklist.docx",
        OUTPUTS / "Code_and_Reproduction.zip",
        RESULTS / "tables" / "table3_model_performance.csv",
        RESULTS / "tables" / "table_s_transport_documentation.csv",
        RESULTS / "audits" / "transport_documentation.json",
        RESULTS / "figures" / "figure5_temporal_structure.png",
        RESULTS / "audits" / "mimic3_age_sentinel.json",
    )
    for path in required:
        require(path.exists() and path.stat().st_size > 0, f"Missing artifact: {path}")
    with zipfile.ZipFile(OUTPUTS / "Code_and_Reproduction.zip") as archive:
        names = archive.namelist()
    forbidden = (
        "/predictions/",
        "_predictions.npz",
        "mimic4_folds",
        "/archive/",
    )
    for name in names:
        require(
            not any(token in name.lower() for token in forbidden),
            f"Patient-level artifact in release archive: {name}",
        )


def main() -> None:
    manifest = check_manifest()
    check_prediction_alignment()
    check_manuscript(manifest)
    check_artifacts()
    print(
        "Integrity audit passed: hashes, prediction alignment, manuscript "
        "wording, artifacts, and release archive"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Integrity audit failed: {exc}", file=sys.stderr)
        raise
