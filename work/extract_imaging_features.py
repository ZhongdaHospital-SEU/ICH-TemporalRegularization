"""Extract pre-landmark ICH imaging phenotypes from MIMIC-IV radiology notes."""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic.common import PROCESSED, load_config


SOURCE_ROOT = Path(
    r"D:\TT paper\0811Temporal Pathway"
)
HOSPITAL = SOURCE_ROOT / "mimic-iv-3.1" / "hosp"
RADIOLOGY = (
    SOURCE_ROOT
    / "mimic-iv-3.1"
    / "mimic iv note 2.2"
    / "note"
    / "radiology.csv.gz"
)

NEGATION = re.compile(
    r"\b(no|without|absence of|negative for|free of|resolved)\b"
    r"[^.;\n]{0,60}$",
    flags=re.IGNORECASE,
)
HEME = re.compile(r"\b(hemorrhage|hematoma|blood)\b", flags=re.IGNORECASE)
IVH = re.compile(
    r"\b(intraventricular\s+(hemorrhage|blood|extension)|"
    r"hemorrhage[^.;\n]{0,30}intraventricular)\b",
    flags=re.IGNORECASE,
)
HYDRO = re.compile(r"\bhydrocephalus\b", flags=re.IGNORECASE)
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
    r"\b(lobar|frontal lobe|parietal lobe|temporal lobe|occipital lobe|"
    r"frontal|parietal|temporal|occipital)\b",
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

LENGTH_UNIT = r"(mm|millimeters?|cm|centimeters?)"
DIMENSIONS = re.compile(
    rf"(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)"
    rf"(?:\s*(?:x|by)\s*(\d+(?:\.\d+)?))?\s*{LENGTH_UNIT}",
    flags=re.IGNORECASE,
)
MIDLINE_MM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm)[^.;\n]{0,50}midline|"
    r"midline[^.;\n]{0,50}?(\d+(?:\.\d+)?)\s*(mm|cm)",
    flags=re.IGNORECASE,
)


def load_notes(cohort_path: Path, radiology_path: Path) -> pd.DataFrame:
    admission_path = HOSPITAL / "admissions.csv.gz"
    connection = duckdb.connect()
    query = """
        WITH cohort AS (
            SELECT
                CAST(stay_id AS BIGINT) AS stay_id,
                CAST(hadm_id AS BIGINT) AS hadm_id,
                CAST(subject_id AS BIGINT) AS subject_id,
                CAST(intime AS TIMESTAMP) AS intime
            FROM read_csv_auto(?)
        ),
        admission AS (
            SELECT
                CAST(hadm_id AS BIGINT) AS hadm_id,
                min(CAST(admittime AS TIMESTAMP)) AS admittime
            FROM read_csv_auto(?)
            GROUP BY 1
        )
        SELECT
            c.stay_id,
            c.hadm_id,
            CAST(r.charttime AS TIMESTAMP) AS charttime,
            r.text
        FROM cohort c
        JOIN admission a ON c.hadm_id = a.hadm_id
        JOIN read_csv_auto(?) r ON c.hadm_id = r.hadm_id
        WHERE r.charttime >= a.admittime - INTERVAL 6 HOUR
          AND r.charttime < c.intime + INTERVAL 24 HOUR
          AND (
              lower(r.text) LIKE '%head%'
              OR lower(r.text) LIKE '%brain%'
          )
          AND (
              lower(r.text) LIKE '%hemorrh%'
              OR lower(r.text) LIKE '%hematoma%'
              OR lower(r.text) LIKE '%hydrocephalus%'
              OR lower(r.text) LIKE '%midline%'
          )
    """
    return connection.execute(
        query,
        [str(cohort_path), str(admission_path), str(radiology_path)],
    ).fetchdf()


def sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.;])\s+|\n+", text)
        if part.strip()
    ]


def negated(text: str, match: re.Match[str]) -> bool:
    prefix = text[max(0, match.start() - 90) : match.start()]
    return bool(NEGATION.search(prefix))


def positive(pattern: re.Pattern[str], text: str) -> bool:
    return any(not negated(text, match) for match in pattern.finditer(text))


def positive_heme_sentence(pattern: re.Pattern[str], text: str) -> bool:
    for sentence in sentences(text):
        if not HEME.search(sentence):
            continue
        for match in pattern.finditer(sentence):
            if not negated(sentence, match):
                return True
    return False


def to_mm(value: float, unit: str) -> float:
    return value * 10.0 if unit.lower().startswith("cm") else value


def parse_midline(text: str) -> float | None:
    values: list[float] = []
    for match in MIDLINE_MM.finditer(text):
        number = match.group(1) or match.group(3)
        unit = match.group(2) or match.group(4)
        if number and unit:
            values.append(to_mm(float(number), unit))
    return max(values) if values else None


def parse_dimensions(text: str) -> tuple[float | None, float | None]:
    volumes: list[float] = []
    diameters: list[float] = []
    for sentence in sentences(text):
        if not HEME.search(sentence):
            continue
        for match in DIMENSIONS.finditer(sentence):
            dimensions = [
                float(value)
                for value in match.groups()[:3]
                if value is not None
            ]
            unit = match.group(4)
            dimensions_cm = [
                to_mm(value, unit) / 10.0 for value in dimensions
            ]
            diameters.extend(dimensions_cm)
            if len(dimensions_cm) == 3:
                a, b, c = dimensions_cm
                volumes.append(0.5 * a * b * c)
    volume = max(volumes) if volumes else None
    diameter = max(diameters) if diameters else None
    return volume, diameter


def extract_row(group: pd.DataFrame) -> dict[str, float | int]:
    texts = group["text"].fillna("").astype(str).tolist()
    joined = "\n".join(texts)
    volumes: list[float] = []
    diameters: list[float] = []
    shifts: list[float] = []
    for text in texts:
        volume, diameter = parse_dimensions(text)
        if volume is not None:
            volumes.append(volume)
        if diameter is not None:
            diameters.append(diameter)
        shift = parse_midline(text)
        if shift is not None:
            shifts.append(shift)
    return {
        "imaging_report_n": len(texts),
        "imaging_available": 1,
        "imaging_ivh": int(positive(IVH, joined)),
        "imaging_hydrocephalus": int(positive(HYDRO, joined)),
        "imaging_midline_shift": int(positive(MIDLINE, joined)),
        "imaging_midline_shift_mm": max(shifts) if shifts else math.nan,
        "imaging_herniation": int(positive(HERNIATION, joined)),
        "imaging_lobar": int(positive_heme_sentence(LOBAR, joined)),
        "imaging_deep": int(positive_heme_sentence(DEEP, joined)),
        "imaging_brainstem": int(positive_heme_sentence(BRAINSTEM, joined)),
        "imaging_cerebellar": int(
            positive_heme_sentence(CEREBELLAR, joined)
        ),
        "imaging_volume_cm3": max(volumes) if volumes else math.nan,
        "imaging_max_diameter_cm": (
            max(diameters) if diameters else math.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED / "mimic4" / "imaging_features.csv",
    )
    args = parser.parse_args()

    cohort_path = PROCESSED / "mimic4" / "cohort.csv"
    notes = load_notes(cohort_path, RADIOLOGY)
    cohort_ids = pd.read_csv(cohort_path, usecols=["stay_id"])
    rows = [
        {"stay_id": int(stay_id), **extract_row(group)}
        for stay_id, group in notes.groupby("stay_id", sort=False)
    ]
    features = pd.DataFrame(rows)
    output = cohort_ids.merge(features, on="stay_id", how="left")
    feature_columns = [column for column in output if column != "stay_id"]
    output[feature_columns] = output[feature_columns].fillna(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    available = int(output["imaging_available"].sum())
    print(f"cohort={len(output)} notes_stays={len(features)} available={available}")
    print(output[feature_columns].sum(numeric_only=True).to_string())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
