"""Build landmark-available, section-aware ICH imaging-report phenotypes."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"D:\TT paper\0811Temporal Pathway")
HOSPITAL = SOURCE_ROOT / "mimic-iv-3.1" / "hosp"
NOTE_ROOT = SOURCE_ROOT / "mimic-iv-3.1" / "mimic iv note 2.2" / "note"
RADIOLOGY = NOTE_ROOT / "radiology.csv.gz"
DETAIL = NOTE_ROOT / "radiology_detail.csv.gz"


def load_index_reports(cohort_path: Path) -> pd.DataFrame:
    """Select one latest available head CT/CTA report before ICU hour 24."""
    connection = duckdb.connect()
    query = """
        WITH cohort AS (
            SELECT
                CAST(stay_id AS BIGINT) AS stay_id,
                CAST(hadm_id AS BIGINT) AS hadm_id,
                CAST(intime AS TIMESTAMP) AS intime
            FROM read_csv_auto(?)
        ),
        head_exam AS (
            SELECT DISTINCT note_id, field_value AS exam_name
            FROM read_csv_auto(?)
            WHERE field_name = 'exam_name'
              AND (
                lower(field_value) LIKE '%ct head%'
                OR lower(field_value) LIKE '%cta head%'
                OR lower(field_value) LIKE '%stroke protocol%'
                OR lower(field_value) LIKE '%brain w/o%'
              )
              AND lower(field_value) NOT LIKE '%mr %'
              AND lower(field_value) NOT LIKE '%mra %'
              AND lower(field_value) NOT LIKE '%mrv %'
        ),
        candidate AS (
            SELECT
                c.stay_id,
                c.hadm_id,
                r.note_id,
                r.note_type,
                r.note_seq,
                CAST(r.charttime AS TIMESTAMP) AS charttime,
                CAST(r.storetime AS TIMESTAMP) AS storetime,
                r.text,
                h.exam_name,
                row_number() OVER (
                    PARTITION BY c.stay_id
                    ORDER BY r.charttime DESC, r.storetime DESC, r.note_id DESC
                ) AS report_rank
            FROM cohort c
            JOIN read_csv_auto(?) r ON c.hadm_id = r.hadm_id
            JOIN head_exam h ON r.note_id = h.note_id
            WHERE r.note_type = 'RR'
              AND r.charttime >= c.intime - INTERVAL 6 HOUR
              AND r.charttime < c.intime + INTERVAL 24 HOUR
              AND r.storetime < c.intime + INTERVAL 24 HOUR
        )
        SELECT * EXCLUDE (report_rank)
        FROM candidate
        WHERE report_rank = 1
    """
    return connection.execute(
        query,
        [str(cohort_path), str(DETAIL), str(RADIOLOGY)],
    ).fetchdf()


def extract_relevant_sections(text: str) -> tuple[str, int]:
    text = str(text).replace("_", " ")
    patterns = (
        r"(?is)\bFINDINGS\s*:\s*(.*?)(?=\n\s*(?:IMPRESSION|TECHNIQUE|"
        r"COMPARISON|NOTES?|RECOMMENDATION)\s*:|$)",
        r"(?is)\bIMPRESSION\s*:\s*(.*?)(?=\n\s*(?:FINDINGS|TECHNIQUE|"
        r"COMPARISON|NOTES?|RECOMMENDATION)\s*:|$)",
    )
    parts = []
    for pattern in patterns:
        parts.extend(
            match.group(1).strip()
            for match in re.finditer(pattern, text)
            if match.group(1).strip()
        )
    if parts:
        return "\n".join(parts), 1
    return text, 0


def _sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.;])\s+|\n+", text)
        if value.strip()
    ]


def _status(pattern: re.Pattern[str], text: str) -> int | None:
    positive = False
    negative = False
    for sentence in _sentences(text):
        lower = sentence.lower()
        for match in pattern.finditer(sentence):
            prefix = lower[max(0, match.start() - 90) : match.start()]
            suffix = lower[match.end() : match.end() + 70]
            if re.search(
                r"\b(no|without|negative for|absence of|free of|"
                r"not|evaluate for|question of|rule out)\b",
                prefix,
            ) or re.search(
                r"\b(not present|not seen|has resolved|is resolved|"
                r"have resolved|was resolved|were resolved)\b",
                suffix,
            ):
                negative = True
            else:
                positive = True
    if positive:
        return 1
    if negative:
        return 0
    return None


HEME = re.compile(
    r"\b(hemorrhage|hematoma|intraparenchymal bleed|parenchymal bleed|"
    r"intracerebral hemorrhage|ICH)\b",
    flags=re.IGNORECASE,
)
IVH = re.compile(
    r"\b(intraventricular\s+(hemorrhage|blood|extension)|"
    r"hemorrhage[^.;\n]{0,35}intraventricular|"
    r"blood[^.;\n]{0,25}ventricl)",
    flags=re.IGNORECASE,
)
HYDROCEPHALUS = re.compile(r"\bhydrocephalus\b", flags=re.IGNORECASE)
MIDLINE = re.compile(
    r"\b(midline shift|shift of[^.;\n]{0,30}midline|"
    r"midline structures?[^.;\n]{0,30}shift)\b",
    flags=re.IGNORECASE,
)
HERNIATION = re.compile(
    r"\b(herniation|uncal|subfalcine|transtentorial)\b",
    flags=re.IGNORECASE,
)
LOBAR = re.compile(
    r"\b(lobar|frontal lobe|lobar frontal|parietal lobe|temporal lobe|"
    r"occipital lobe)\b",
    flags=re.IGNORECASE,
)
DEEP = re.compile(
    r"\b(basal ganglia|thalamus|thalamic|putamen|globus pallidus|"
    r"internal capsule|caudate)\b",
    flags=re.IGNORECASE,
)
BRAINSTEM = re.compile(
    r"\b(brainstem|brain stem|pons|pontine|midbrain|medulla)\b",
    flags=re.IGNORECASE,
)
CEREBELLAR = re.compile(r"\b(cerebellum|cerebellar)\b", flags=re.IGNORECASE)


def _positive_hemorrhage_location(
    pattern: re.Pattern[str],
    text: str,
) -> int | None:
    positive = False
    negative = False
    for sentence in _sentences(text):
        if not HEME.search(sentence):
            continue
        lower = sentence.lower()
        if re.search(r"\b(microhemorrhage|chronic|old|trace|petechial)\b", lower):
            continue
        for match in pattern.finditer(sentence):
            prefix = lower[max(0, match.start() - 80) : match.start()]
            if re.search(
                r"\b(no|without|absence of|not|evaluate for|question of)\b",
                prefix,
            ):
                negative = True
            else:
                positive = True
    if positive:
        return 1
    if negative:
        return 0
    return None


def extract_report_row(report: pd.Series) -> dict[str, object]:
    sections, section_found = extract_relevant_sections(report["text"])
    return {
        "stay_id": int(report["stay_id"]),
        "imaging_available": 1,
        "imaging_note_id": str(report["note_id"]),
        "imaging_charttime": report["charttime"],
        "imaging_storetime": report["storetime"],
        "imaging_exam_name": str(report["exam_name"]),
        "imaging_section_found": section_found,
        "imaging_ivh": _status(IVH, sections),
        "imaging_hydrocephalus": _status(HYDROCEPHALUS, sections),
        "imaging_midline_shift": _status(MIDLINE, sections),
        "imaging_herniation": _status(HERNIATION, sections),
        "imaging_lobar": _positive_hemorrhage_location(LOBAR, sections),
        "imaging_deep": _positive_hemorrhage_location(DEEP, sections),
        "imaging_brainstem": _positive_hemorrhage_location(BRAINSTEM, sections),
        "imaging_cerebellar": _positive_hemorrhage_location(CEREBELLAR, sections),
        "imaging_posterior_fossa": (
            1
            if any(
                value == 1
                for value in (
                    _positive_hemorrhage_location(BRAINSTEM, sections),
                    _positive_hemorrhage_location(CEREBELLAR, sections),
                )
            )
            else (
                0
                if any(
                    value == 0
                    for value in (
                        _positive_hemorrhage_location(BRAINSTEM, sections),
                        _positive_hemorrhage_location(CEREBELLAR, sections),
                    )
                )
                else None
            )
        ),
        "imaging_noncontrast_ct": int(
            bool(re.search(r"ct head w/o", str(report["exam_name"]), re.IGNORECASE))
            and not bool(
                re.search(
                    r"\bcta\b|\bw/\s*(?:&|and|contrast)",
                    str(report["exam_name"]),
                    re.IGNORECASE,
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "mimic4" / "imaging_features_v2.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT
        / "results"
        / "experiments"
        / "imaging_index_report_audit.csv",
    )
    args = parser.parse_args()

    cohort_path = ROOT / "data" / "processed" / "mimic4" / "cohort.csv"
    cohort = pd.read_csv(cohort_path)
    reports = load_index_reports(cohort_path)
    rows = [
        extract_report_row(report)
        for _, report in reports.iterrows()
    ]
    features = pd.DataFrame(rows)
    output = cohort[["stay_id"]].merge(features, on="stay_id", how="left")
    binary_columns = [
        "imaging_ivh",
        "imaging_hydrocephalus",
        "imaging_midline_shift",
        "imaging_herniation",
        "imaging_lobar",
        "imaging_deep",
        "imaging_brainstem",
        "imaging_cerebellar",
        "imaging_posterior_fossa",
    ]
    for column in binary_columns:
        output[column] = output[column].astype("Int64")
    output["imaging_available"] = output["imaging_available"].fillna(0).astype(int)
    output["imaging_noncontrast_ct"] = (
        output["imaging_noncontrast_ct"].fillna(0).astype(int)
    )
    output["imaging_section_found"] = (
        output["imaging_section_found"].fillna(0).astype(int)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    audit = reports.copy()
    audit["parser_excerpt"] = audit["text"].fillna("").astype(str).str.slice(0, 1200)
    audit = audit[
        [
            "stay_id",
            "note_id",
            "exam_name",
            "charttime",
            "storetime",
            "parser_excerpt",
        ]
    ]
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output, index=False)

    summary = {
        "cohort_n": int(len(output)),
        "available_n": int(output["imaging_available"].sum()),
        "noncontrast_ct_n": int(output["imaging_noncontrast_ct"].sum()),
        "section_found_n": int(output["imaging_section_found"].sum()),
        "feature_positive_n": {
            column: int((output[column] == 1).sum())
            for column in binary_columns
        },
        "feature_explicit_negative_n": {
            column: int((output[column] == 0).sum())
            for column in binary_columns
        },
        "feature_unmentioned_n": {
            column: int(output[column].isna().sum())
            for column in binary_columns
        },
    }
    print(pd.Series(summary, dtype=object).to_string())
    print(f"Saved {args.output}")
    print(f"Saved {args.audit_output}")


if __name__ == "__main__":
    main()
