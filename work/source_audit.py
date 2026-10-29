"""Audit ICH cohort size and key neuro-monitoring fields in MIMIC-IV, MIMIC-III, and eICU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb


SOURCE_ROOT = Path(r"D:\TT paper\0811Temporal Pathway")

MIMIC_IV = {
    "diagnoses": SOURCE_ROOT / "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz",
    "admissions": SOURCE_ROOT / "mimic-iv-3.1/hosp/admissions.csv.gz",
    "patients": SOURCE_ROOT / "mimic-iv-3.1/hosp/patients.csv.gz",
    "icustays": SOURCE_ROOT / "mimic-iv-3.1/icu/icustays.csv.gz",
    "chartevents": SOURCE_ROOT / "mimic-iv-3.1/icu/chartevents.csv.gz",
    "d_items": SOURCE_ROOT / "mimic-iv-3.1/icu/d_items.csv.gz",
}

MIMIC_III = {
    "diagnoses": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz",
    "admissions": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz",
    "patients": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/PATIENTS.csv.gz",
    "icustays": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/ICUSTAYS.csv.gz",
    "chartevents": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz",
    "d_items": SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1/mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz",
}

EICU = {
    "diagnosis": SOURCE_ROOT / "eicu/diagnosis.csv.gz",
    "patient": SOURCE_ROOT / "eicu/patient.csv.gz",
    "nursecharting": SOURCE_ROOT / "eicu/nurseCharting.csv.gz",
}


def view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    con.execute(
        f"CREATE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true)"
    )


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def audit_mimic4() -> dict[str, object]:
    con = duckdb.connect()
    for name, path in MIMIC_IV.items():
        view(con, name, path)
    result: dict[str, object] = {}
    result["dataset"] = "MIMIC-IV v3.1"
    result["ich_i61_admissions"] = scalar(
        con,
        "SELECT COUNT(DISTINCT hadm_id) FROM diagnoses "
        "WHERE icd_version = '10' AND icd_code LIKE 'I61%'",
    )
    result["ich_i61_i62_admissions"] = scalar(
        con,
        "SELECT COUNT(DISTINCT hadm_id) FROM diagnoses "
        "WHERE icd_version = '10' AND (icd_code LIKE 'I61%' OR icd_code LIKE 'I62%')",
    )
    result["ich_i61_icu_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT i.stay_id)
        FROM icustays i
        WHERE i.hadm_id IN (
            SELECT DISTINCT hadm_id FROM diagnoses
            WHERE icd_version = '10' AND icd_code LIKE 'I61%'
        )
        """,
    )
    result["ich_i61_patients"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT a.subject_id)
        FROM admissions a
        WHERE a.hadm_id IN (
            SELECT DISTINCT hadm_id FROM diagnoses
            WHERE icd_version = '10' AND icd_code LIKE 'I61%'
        )
        """,
    )
    result["gcs_rows"] = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM chartevents ce
        WHERE ce.itemid IN (220739, 223900, 223901)
          AND ce.stay_id IN (
              SELECT DISTINCT i.stay_id
              FROM icustays i
              WHERE i.hadm_id IN (
                  SELECT DISTINCT hadm_id FROM diagnoses
                  WHERE icd_version = '10' AND icd_code LIKE 'I61%'
              )
          )
        """,
    )
    result["gcs_patients"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT ce.stay_id)
        FROM chartevents ce
        WHERE ce.itemid IN (220739, 223900, 223901)
          AND ce.stay_id IN (
              SELECT DISTINCT i.stay_id
              FROM icustays i
              WHERE i.hadm_id IN (
                  SELECT DISTINCT hadm_id FROM diagnoses
                  WHERE icd_version = '10' AND icd_code LIKE 'I61%'
              )
          )
        """,
    )
    result["gcs_items"] = con.execute(
        """
        SELECT itemid, label, abbreviation, category, unitname
        FROM d_items
        WHERE itemid IN (220739, 223900, 223901, 226755, 228112, 223900)
        ORDER BY itemid
        """
    ).fetchdf().to_dict(orient="records")
    result["sedation_items"] = con.execute(
        """
        SELECT itemid, label, abbreviation, category
        FROM d_items
        WHERE lower(label) LIKE '%richmond%'
           OR lower(label) LIKE '%rass%'
           OR lower(label) LIKE '%sedation%'
        ORDER BY itemid
        """
    ).fetchdf().to_dict(orient="records")
    return result


def audit_mimic3() -> dict[str, object]:
    con = duckdb.connect()
    for name, path in MIMIC_III.items():
        view(con, name, path)
    result: dict[str, object] = {}
    result["dataset"] = "MIMIC-III v1.4"
    result["ich_431_admissions"] = scalar(
        con,
        "SELECT COUNT(DISTINCT hadm_id) FROM diagnoses WHERE icd9_code LIKE '431%'",
    )
    result["ich_431_432_admissions"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT hadm_id) FROM diagnoses
        WHERE icd9_code LIKE '431%' OR icd9_code LIKE '432%'
        """,
    )
    result["ich_431_icu_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT i.icustay_id)
        FROM icustays i
        WHERE i.hadm_id IN (
            SELECT DISTINCT hadm_id FROM diagnoses WHERE icd9_code LIKE '431%'
        )
        """,
    )
    result["ich_431_patients"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT a.subject_id)
        FROM admissions a
        WHERE a.hadm_id IN (
            SELECT DISTINCT hadm_id FROM diagnoses WHERE icd9_code LIKE '431%'
        )
        """,
    )
    result["gcs_rows"] = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM chartevents ce
        WHERE ce.itemid IN (198, 220739, 223900, 223901)
          AND ce.icustay_id IN (
              SELECT DISTINCT i.icustay_id
              FROM icustays i
              WHERE i.hadm_id IN (
                  SELECT DISTINCT hadm_id FROM diagnoses WHERE icd9_code LIKE '431%'
              )
          )
        """,
    )
    result["gcs_items"] = con.execute(
        """
        SELECT itemid, label, abbreviation, category, unitname
        FROM d_items
        WHERE itemid IN (198, 220739, 223900, 223901)
        ORDER BY itemid
        """
    ).fetchdf().to_dict(orient="records")
    result["sedation_items"] = con.execute(
        """
        SELECT itemid, label, abbreviation, category
        FROM d_items
        WHERE lower(label) LIKE '%richmond%'
           OR lower(label) LIKE '%rass%'
           OR lower(label) LIKE '%sedation%'
        ORDER BY itemid
        """
    ).fetchdf().to_dict(orient="records")
    return result


def audit_eicu() -> dict[str, object]:
    con = duckdb.connect()
    for name, path in EICU.items():
        view(con, name, path)
    result: dict[str, object] = {}
    result["dataset"] = "eICU-CRD v2.0"
    result["ich_text_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM diagnosis
        WHERE lower(diagnosisstring) LIKE '%intracerebral hemorrhage%'
           OR lower(diagnosisstring) LIKE '%intraparenchymal hemorrhage%'
        """,
    )
    result["ich_icd9_431_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM diagnosis
        WHERE icd9code LIKE '431%'
        """,
    )
    result["ich_text_or_code_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM diagnosis
        WHERE lower(diagnosisstring) LIKE '%intracerebral hemorrhage%'
           OR lower(diagnosisstring) LIKE '%intraparenchymal hemorrhage%'
           OR icd9code LIKE '431%'
        """,
    )
    result["gcs_rows"] = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM nursecharting n
        WHERE n.nursingchartcelltypevalname = 'GCS Total'
          AND n.patientunitstayid IN (
              SELECT DISTINCT patientunitstayid FROM diagnosis
              WHERE lower(diagnosisstring) LIKE '%intracerebral hemorrhage%'
                 OR lower(diagnosisstring) LIKE '%intraparenchymal hemorrhage%'
                 OR icd9code LIKE '431%'
          )
        """,
    )
    result["gcs_stays"] = scalar(
        con,
        """
        SELECT COUNT(DISTINCT n.patientunitstayid)
        FROM nursecharting n
        WHERE n.nursingchartcelltypevalname = 'GCS Total'
          AND n.patientunitstayid IN (
              SELECT DISTINCT patientunitstayid FROM diagnosis
              WHERE lower(diagnosisstring) LIKE '%intracerebral hemorrhage%'
                 OR lower(diagnosisstring) LIKE '%intraparenchymal hemorrhage%'
                 OR icd9code LIKE '431%'
          )
        """,
    )
    result["diagnosis_examples"] = con.execute(
        """
        SELECT icd9code, diagnosisstring, COUNT(*) AS n
        FROM diagnosis
        WHERE lower(diagnosisstring) LIKE '%intracerebral hemorrhage%'
           OR lower(diagnosisstring) LIKE '%intraparenchymal hemorrhage%'
           OR icd9code LIKE '431%'
        GROUP BY 1, 2
        ORDER BY n DESC
        LIMIT 30
        """
    ).fetchdf().to_dict(orient="records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mimic4", "mimic3", "eicu", "all"], default="all")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "work" / "source_audit.json",
    )
    args = parser.parse_args()
    audits: list[dict[str, object]] = []
    if args.source in ("mimic4", "all"):
        audits.append(audit_mimic4())
    if args.source in ("mimic3", "all"):
        audits.append(audit_mimic3())
    if args.source in ("eicu", "all"):
        audits.append(audit_eicu())
    import json

    os.makedirs(args.out.parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(audits, fh, indent=2, ensure_ascii=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
