"""Summarize the site-calibration, sensitivity, and landmark analyses.

The script deliberately separates smoke or abbreviated sensitivity runs from
the prespecified sensitivity protocol. It reads only saved artifacts and does
not fit or alter any model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_ORDER = ("xgb_summary", "gru", "dggru", "dggru_tr")
SENSITIVITY_MODELS = ("xgb_summary", "dggru_tr")
MODEL_LABELS = {
    "xgb_summary": "Summary XGBoost",
    "gru": "GRU",
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU + TR",
}
LANDMARK_FILES = {
    6: RESULTS / "metrics_landmark6.json",
    24: RESULTS / "metrics_core.json",
    48: RESULTS / "metrics_landmark48.json",
}
LANDMARK_MANIFESTS = {
    6: RESULTS / "run_manifest_landmark6.json",
    24: RESULTS / "run_manifest_core.json",
    48: RESULTS / "run_manifest_landmark48.json",
}
EXPECTED_SENSITIVITY_PROTOCOL = {
    "epochs": 25,
    "folds": 5,
    "cv_repeats": 1,
    "replicates": 500,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "" if number is None else f"{number:.{digits}f}"


def fmt_ci(point: Any, interval: Any, digits: int = 3) -> str:
    value = finite(point)
    if value is None:
        return ""
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return f"{value:.{digits}f}"
    low = finite(interval[0])
    high = finite(interval[1])
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def model_point(metrics: dict[str, Any], model: str) -> dict[str, Any]:
    block = metrics.get(model)
    if not isinstance(block, dict):
        return {}
    point = block.get("point")
    return point if isinstance(point, dict) else {}


def model_ci(metrics: dict[str, Any], model: str, key: str) -> list[Any]:
    block = metrics.get(model)
    if not isinstance(block, dict):
        return []
    ci = block.get("ci")
    if not isinstance(ci, dict):
        return []
    value = ci.get(key)
    return value if isinstance(value, list) else []


def outcome_counts(manifest: dict[str, Any]) -> tuple[int | None, list[int] | None]:
    n = finite(manifest.get("development_n"))
    outcomes = manifest.get("development_outcomes")
    if not isinstance(outcomes, dict):
        return (int(n) if n is not None else None, None)
    try:
        counts = [
            int(float(outcomes.get(str(grade), outcomes.get(grade))))
            for grade in (0, 1, 2)
        ]
    except (TypeError, ValueError):
        return (int(n) if n is not None else None, None)
    return (int(n) if n is not None else sum(counts), counts)


def landmark_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for hours in (6, 24, 48):
        metrics_path = LANDMARK_FILES[hours]
        manifest_path = LANDMARK_MANIFESTS[hours]
        if not metrics_path.is_file():
            notes.append(f"{hours} h metrics missing: {metrics_path.name}")
            continue
        metrics = read_json(metrics_path)
        development = metrics.get("development")
        if not isinstance(development, dict):
            notes.append(f"{hours} h metrics have no development block")
            continue
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if not manifest:
            notes.append(f"{hours} h run manifest missing: {manifest_path.name}")
        else:
            expected = {
                "folds": 5,
                "cv_repeats": 5,
                "bootstrap_replicates": 2000,
            }
            invalid = {
                key: manifest.get(key)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            models = manifest.get("models")
            if not isinstance(models, list) or not set(MODEL_ORDER).issubset(
                set(models)
            ):
                invalid["models"] = models
            if invalid:
                notes.append(
                    f"{hours} h formal protocol mismatch: {invalid}"
                )
                continue
        n, counts = outcome_counts(manifest)
        for model in MODEL_ORDER:
            point = model_point(development, model)
            if not point:
                continue
            rows.append(
                {
                    "landmark_h": hours,
                    "n": n if n is not None else int(point.get("n", 0)),
                    "grade_0": "" if counts is None else counts[0],
                    "grade_1": "" if counts is None else counts[1],
                    "grade_2": "" if counts is None else counts[2],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "macro_auc": finite(point.get("macro_auc")),
                    "macro_auc_ci_low": (
                        finite(model_ci(development, model, "macro_auc")[0])
                        if len(model_ci(development, model, "macro_auc")) == 2
                        else None
                    ),
                    "macro_auc_ci_high": (
                        finite(model_ci(development, model, "macro_auc")[1])
                        if len(model_ci(development, model, "macro_auc")) == 2
                        else None
                    ),
                    "grade_1_auc": finite(point.get("auc_grade_1")),
                    "calibration_slope_any": finite(
                        point.get("calibration_slope_any")
                    ),
                    "calibration_intercept_any": finite(
                        point.get("calibration_intercept_any")
                    ),
                    "ece_any": finite(point.get("ece_any_deterioration")),
                    "brier_any": finite(point.get("brier_any_deterioration")),
                }
            )
    return rows, notes


def protocol_valid(record: dict[str, Any]) -> tuple[bool, str]:
    protocol = record.get("fitting_protocol")
    if not isinstance(protocol, dict):
        return False, "missing fitting_protocol"
    for key, expected in EXPECTED_SENSITIVITY_PROTOCOL.items():
        actual = protocol.get(key)
        if actual is None or int(actual) != expected:
            return False, f"{key}={actual!r}, expected {expected}"
    if protocol.get("lambda_selection") != "frozen_primary_median":
        return False, "lambda was not frozen at the primary median"
    return True, "formal sensitivity protocol"


def sensitivity_support_rows() -> list[dict[str, Any]]:
    plan_path = RESULTS / "sensitivity" / "sensitivity_support_plan.json"
    if not plan_path.is_file():
        return []
    plan = read_json(plan_path)
    rows: list[dict[str, Any]] = []
    datasets = plan.get("datasets", {})
    if not isinstance(datasets, dict):
        return rows
    for dataset in ("mimic4", "mimic3", "eicu"):
        for item in datasets.get(dataset, []):
            scenario = item.get("scenario", {})
            status = item.get("status", {})
            for model in SENSITIVITY_MODELS:
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario.get("scenario_id"),
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "analysis_id": scenario.get("analysis_id"),
                        "analysis_title": scenario.get("analysis_title"),
                        "variant_title": scenario.get("variant_title"),
                        "support": status.get("support"),
                        "support_reason": status.get("reason"),
                        "eligible": status.get("model_run_allowed"),
                        "record_file": "",
                        "protocol_status": "",
                        "fit_status": "",
                        "n": "",
                        "grade_0": "",
                        "grade_1": "",
                        "grade_2": "",
                        "macro_auc": "",
                        "grade_1_auc": "",
                        "calibration_slope_any": "",
                        "ece_any": "",
                    }
                )
    return rows


def sensitivity_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows = sensitivity_support_rows()
    by_key = {
        (row["dataset"], row["scenario"], row["model"]): row for row in rows
    }
    notes: list[str] = []
    for dataset in ("mimic4", "mimic3", "eicu"):
        record_dir = RESULTS / "sensitivity" / dataset / "records"
        if not record_dir.is_dir():
            continue
        for record_path in sorted(record_dir.glob("*.json")):
            try:
                record = read_json(record_path)
            except (OSError, json.JSONDecodeError) as exc:
                notes.append(f"unreadable sensitivity record {record_path}: {exc}")
                continue
            scenario = record.get("scenario")
            valid, reason = protocol_valid(record)
            result = record.get("result")
            if not isinstance(result, dict):
                notes.append(f"{record_path.name} has no result block")
                continue
            metric_result = result
            artifact = result.get("artifact")
            if isinstance(artifact, (str, Path)):
                artifact_path = Path(str(artifact))
                if artifact_path.is_file():
                    try:
                        loaded = read_json(artifact_path)
                    except (OSError, json.JSONDecodeError) as exc:
                        notes.append(
                            f"unreadable sensitivity artifact {artifact_path}: {exc}"
                        )
                    else:
                        if isinstance(loaded, dict):
                            metric_result = loaded
            scenario_block = result.get("scenario")
            n = finite(result.get("n"))
            outcomes = result.get("outcomes")
            if isinstance(outcomes, dict):
                try:
                    counts = [
                        int(float(outcomes.get(str(g), outcomes.get(g))))
                        for g in (0, 1, 2)
                    ]
                except (TypeError, ValueError):
                    counts = None
            else:
                counts = None
            for model in SENSITIVITY_MODELS:
                key = (dataset, scenario, model)
                row = by_key.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "analysis_id": "",
                        "analysis_title": "",
                        "variant_title": "",
                        "support": "",
                        "support_reason": "",
                        "eligible": "",
                        "record_file": "",
                        "protocol_status": "",
                        "fit_status": "",
                        "n": "",
                        "grade_0": "",
                        "grade_1": "",
                        "grade_2": "",
                        "macro_auc": "",
                        "grade_1_auc": "",
                        "calibration_slope_any": "",
                        "ece_any": "",
                    },
                )
                row["record_file"] = str(record_path.relative_to(ROOT))
                row["protocol_status"] = (
                    "valid" if valid else f"excluded: {reason}"
                )
                if isinstance(scenario_block, dict):
                    row["analysis_id"] = scenario_block.get("analysis_id", "")
                    row["analysis_title"] = scenario_block.get(
                        "analysis_title", ""
                    )
                    row["variant_title"] = scenario_block.get(
                        "variant_title", ""
                    )
                if not valid:
                    row["fit_status"] = "excluded"
                    continue
                point = model_point(metric_result, model)
                if not point:
                    row["fit_status"] = "no model result"
                    continue
                row["fit_status"] = "saved"
                row["n"] = int(n) if n is not None else point.get("n", "")
                row["grade_0"] = "" if counts is None else counts[0]
                row["grade_1"] = "" if counts is None else counts[1]
                row["grade_2"] = "" if counts is None else counts[2]
                row["macro_auc"] = finite(point.get("macro_auc"))
                row["grade_1_auc"] = finite(point.get("auc_grade_1"))
                row["calibration_slope_any"] = finite(
                    point.get("calibration_slope_any")
                )
                row["ece_any"] = finite(point.get("ece_any_deterioration"))
    run_summary_path = RESULTS / "sensitivity" / "run_summary.json"
    if run_summary_path.is_file():
        run_summary = read_json(run_summary_path)
        completed = run_summary.get("completed", [])
        scheduled = int(run_summary.get("scheduled_jobs", -1))
        if isinstance(completed, list) and len(completed) == scheduled:
            failures = {
                (item.get("dataset"), item.get("scenario")): item
                for item in completed
                if int(item.get("returncode", 0)) != 0
            }
            for row in rows:
                key = (row.get("dataset"), row.get("scenario"))
                failure = failures.get(key)
                if failure is None or row.get("record_file"):
                    continue
                if str(row.get("scenario", "")).startswith("s6_landmark"):
                    row["support"] = "separate analysis"
                    row["support_reason"] = (
                        "Landmark-specific cohorts are reported in the 6/24/48 h "
                        "landmark analysis, not in the fixed 24 h sensitivity table."
                    )
                    row["protocol_status"] = "reported_elsewhere"
                    row["fit_status"] = "separate analysis"
                    continue
                reason = f"Model fitting failed (return code {failure['returncode']})."
                stderr = Path(str(failure.get("stderr", "")))
                if stderr.is_file():
                    lines = [
                        line.strip()
                        for line in stderr.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).splitlines()
                        if line.strip()
                    ]
                    if lines:
                        reason += " " + lines[-1]
                row["support"] = "unsupported"
                row["support_reason"] = reason
                row["fit_status"] = "failed"
    return rows, notes


def site_calibration_rows() -> list[dict[str, str]]:
    path = RESULTS / "tables" / "eicu_site_calibration_methods.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def site_calibration_pairwise() -> list[dict[str, Any]]:
    path = RESULTS / "tables" / "eicu_site_calibration_sites.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("endpoint", ""),
            row.get("model", ""),
            row.get("hospitalid", ""),
        )
        by_key.setdefault(key, {})[row.get("method", "")] = row
    summaries: list[dict[str, Any]] = []
    for endpoint in ("any_deterioration", "severe_deterioration"):
        for model in ("xgb_summary", "dggru_tr"):
            site_maps = [
                methods
                for (ep, name, _), methods in by_key.items()
                if ep == endpoint and name == model and "uncalibrated" in methods
            ]
            for method in (
                "intercept_only",
                "slope_intercept",
                "logistic_recalibration",
                "empirical_bayes",
            ):
                deltas: list[float] = []
                slopes: list[float] = []
                for methods in site_maps:
                    base = finite(methods["uncalibrated"].get("ece"))
                    updated_row = methods.get(method)
                    updated = (
                        finite(updated_row.get("ece"))
                        if isinstance(updated_row, dict)
                        else None
                    )
                    if base is None or updated is None:
                        continue
                    deltas.append(updated - base)
                    slope = (
                        finite(updated_row.get("calibration_slope"))
                        if isinstance(updated_row, dict)
                        else None
                    )
                    if slope is not None:
                        slopes.append(slope)
                if not deltas:
                    continue
                deltas_sorted = sorted(deltas)
                slopes_sorted = sorted(slopes)
                midpoint = len(deltas_sorted) // 2
                median_delta = (
                    deltas_sorted[midpoint]
                    if len(deltas_sorted) % 2
                    else 0.5
                    * (
                        deltas_sorted[midpoint - 1]
                        + deltas_sorted[midpoint]
                    )
                )
                summaries.append(
                    {
                        "endpoint": endpoint,
                        "model": model,
                        "method": method,
                        "sites": len(deltas),
                        "sites_with_lower_ece": sum(delta < 0 for delta in deltas),
                        "median_site_ece_change": median_delta,
                        "median_updated_slope": (
                            slopes_sorted[len(slopes_sorted) // 2]
                            if slopes_sorted
                            else None
                        ),
                    }
                )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    landmark: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    site: list[dict[str, str]],
    notes: list[str],
) -> None:
    lines = ["# New analyses: saved-result report", ""]
    if notes:
        lines.extend(["## Audit notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    lines.extend(["## eICU site-level calibration", ""])
    lines.append(
        "- These are binary-endpoint ECE values and are not numerically "
        "comparable with the three-class ECE in the primary results."
    )
    lines.append("")
    if site:
        lines.extend(
            [
                "| Endpoint | Model | Method | n | Events | AUROC | ECE | Slope |",
                "|---|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in site:
            lines.append(
                "| {endpoint} | {model} | {method_label} | {n} | {events} | "
                "{auroc} | {ece} | {slope} |".format(
                    endpoint=row.get("endpoint", ""),
                    model=row.get("model", ""),
                    method_label=row.get("method_label", ""),
                    n=row.get("n", ""),
                    events=row.get("events", ""),
                    auroc=fmt(row.get("auroc")),
                    ece=fmt(row.get("ece")),
                    slope=fmt(row.get("calibration_slope")),
                )
            )
    else:
        lines.append("No saved site-calibration table.")

    lines.extend(["", "## 6/24/48 h landmark models", ""])
    if landmark:
        lines.extend(
            [
                "| Landmark | n | Grade 0/1/2 | Model | Macro-AUC | Grade 1 AUC | "
                "Calibration slope | ECE any |",
                "|---:|---:|---|---|---|---:|---:|---:|",
            ]
        )
        for row in landmark:
            lines.append(
                "| {hours} h | {n} | {g0}/{g1}/{g2} | {model} | {auc} | "
                "{g1auc} | {slope} | {ece} |".format(
                    hours=row["landmark_h"],
                    n=row["n"],
                    g0=row["grade_0"],
                    g1=row["grade_1"],
                    g2=row["grade_2"],
                    model=row["model_label"],
                    auc=fmt_ci(row["macro_auc"], [
                        row["macro_auc_ci_low"],
                        row["macro_auc_ci_high"],
                    ]),
                    g1auc=fmt(row["grade_1_auc"]),
                    slope=fmt(row["calibration_slope_any"]),
                    ece=fmt(row["ece_any"]),
                )
            )
    else:
        lines.append("No landmark results have been saved.")

    lines.extend(["", "## Sensitivity analyses", ""])
    if sensitivity:
        lines.extend(
            [
                "| Dataset | Scenario | Support | Protocol | n | Grade 0/1/2 | "
                "Model | Macro-AUC | Grade 1 AUC | Slope | ECE any |",
                "|---|---|---|---|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sensitivity:
            if not row.get("record_file") and not row.get("support"):
                continue
            counts = ""
            if any(
                str(row.get(field, "")).strip()
                for field in ("grade_0", "grade_1", "grade_2")
            ):
                counts = "{}/{}/{}".format(
                    row.get("grade_0", ""),
                    row.get("grade_1", ""),
                    row.get("grade_2", ""),
                )
            lines.append(
                "| {dataset} | {scenario} | {support} | {protocol} | {n} | "
                "{counts} | {model} | {auc} | {g1auc} | {slope} | {ece} |".format(
                    dataset=row.get("dataset", ""),
                    scenario=row.get("variant_title") or row.get("scenario", ""),
                    support=row.get("support", ""),
                    protocol=row.get("protocol_status", ""),
                    n=row.get("n", ""),
                    counts=counts,
                    model=row.get("model_label", ""),
                    auc=fmt(row.get("macro_auc")),
                    g1auc=fmt(row.get("grade_1_auc")),
                    slope=fmt(row.get("calibration_slope_any")),
                    ece=fmt(row.get("ece_any")),
                )
            )
    else:
        lines.append("No sensitivity records have been saved.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=RESULTS / "tables" / "new_analyses_report.md",
    )
    args = parser.parse_args()

    landmark, landmark_notes = landmark_rows()
    sensitivity, sensitivity_notes = sensitivity_rows()
    site = site_calibration_rows()
    site_pairwise = site_calibration_pairwise()

    write_csv(
        RESULTS / "tables" / "multilandmark_summary.csv",
        landmark,
        [
            "landmark_h",
            "n",
            "grade_0",
            "grade_1",
            "grade_2",
            "model",
            "model_label",
            "macro_auc",
            "macro_auc_ci_low",
            "macro_auc_ci_high",
            "grade_1_auc",
            "calibration_slope_any",
            "calibration_intercept_any",
            "ece_any",
            "brier_any",
        ],
    )
    write_csv(
        RESULTS / "tables" / "sensitivity_summary.csv",
        sensitivity,
        [
            "dataset",
            "scenario",
            "model",
            "model_label",
            "analysis_id",
            "analysis_title",
            "variant_title",
            "support",
            "support_reason",
            "eligible",
            "record_file",
            "protocol_status",
            "fit_status",
            "n",
            "grade_0",
            "grade_1",
            "grade_2",
            "macro_auc",
            "grade_1_auc",
            "calibration_slope_any",
            "ece_any",
        ],
    )
    write_csv(
        RESULTS / "tables" / "eicu_site_calibration_pairwise.csv",
        site_pairwise,
        [
            "endpoint",
            "model",
            "method",
            "sites",
            "sites_with_lower_ece",
            "median_site_ece_change",
            "median_updated_slope",
        ],
    )
    write_report(
        args.report,
        landmark,
        sensitivity,
        site,
        landmark_notes + sensitivity_notes,
    )
    print(f"Wrote {args.report}")
    print(f"Landmark rows: {len(landmark)}")
    print(f"Sensitivity rows: {len(sensitivity)}")
    print(f"Site-calibration rows: {len(site)}")
    print(f"Site-calibration pairwise rows: {len(site_pairwise)}")


if __name__ == "__main__":
    main()
