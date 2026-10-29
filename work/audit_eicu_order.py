"""Check how eICU first-stay selection changes under visit-aware ordering."""
from __future__ import annotations

from pathlib import Path

import duckdb


SOURCE = Path(r"D:\TT paper\0811Temporal Pathway\eicu")


def main() -> None:
    con = duckdb.connect()
    diagnosis = (SOURCE / "diagnosis.csv.gz").as_posix()
    patient = (SOURCE / "patient.csv.gz").as_posix()
    query = f"""
        WITH diagnosis_flags AS (
            SELECT patientunitstayid,
                   MAX(
                       CASE
                           WHEN regexp_matches(COALESCE(icd9code, ''), '(^|[, ])431([., ]|$)')
                                OR regexp_matches(COALESCE(icd9code, ''), '(^|[, ])I61(\\.|[, ]|$)')
                           THEN 1 ELSE 0
                       END
                   ) AS has_ich,
                   MAX(
                       CASE
                           WHEN lower(COALESCE(diagnosisstring, '')) LIKE '%trauma%'
                                OR lower(COALESCE(diagnosisstring, '')) LIKE '%subarachnoid%'
                                OR lower(COALESCE(diagnosisstring, '')) LIKE '%subdural%'
                                OR regexp_matches(
                                    COALESCE(icd9code, ''),
                                    '(^|[, ])(80[0-4]|85[1-4]|430|432)([., ]|$)'
                                )
                                OR lower(COALESCE(diagnosisstring, '')) LIKE '%toxicology%'
                           THEN 1 ELSE 0
                       END
                   ) AS has_exclusion
            FROM read_csv_auto('{diagnosis}')
            GROUP BY patientunitstayid
        ),
        eligible AS (
            SELECT p.patientunitstayid, p.uniquepid, p.hospitalid,
                   p.patienthealthsystemstayid, p.unitvisitnumber,
                   p.hospitaldischargeyear, p.unitadmittime24,
                   p.hospitaladmitoffset, p.hospitaldischargeoffset
            FROM read_csv_auto('{patient}') p
            JOIN diagnosis_flags d
              ON p.patientunitstayid = d.patientunitstayid
            WHERE d.has_ich = 1 AND d.has_exclusion = 0
              AND p.unitdischargeoffset > 1440
              AND NOT (
                  lower(COALESCE(p.unitdischargestatus, '')) = 'expired'
                  AND p.unitdischargeoffset <= 1440
              )
              AND NOT (
                  lower(COALESCE(p.hospitaldischargestatus, '')) = 'expired'
                  AND p.hospitaldischargeoffset <= 1440
              )
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY uniquepid
                       ORDER BY unitadmittime24, patientunitstayid
                   ) AS current_rank,
                   ROW_NUMBER() OVER (
                       PARTITION BY uniquepid
                       ORDER BY hospitaldischargeyear,
                                patienthealthsystemstayid,
                                unitvisitnumber,
                                patientunitstayid
                   ) AS visit_rank
            FROM eligible
        ),
        current AS (
            SELECT uniquepid, patientunitstayid AS current_id,
                   patienthealthsystemstayid AS current_system_stay,
                   unitvisitnumber AS current_visit
            FROM ranked WHERE current_rank = 1
        ),
        visit_aware AS (
            SELECT uniquepid, patientunitstayid AS visit_id,
                   patienthealthsystemstayid AS visit_system_stay,
                   unitvisitnumber AS visit_number
            FROM ranked WHERE visit_rank = 1
        )
        SELECT uniquepid, current_id, visit_id, current_system_stay,
               visit_system_stay, current_visit, visit_number
        FROM current JOIN visit_aware USING (uniquepid)
    """
    differences = con.execute(query).fetchdf()
    summary = {
        "eligible_patients": len(differences),
        "selection_differences": int(
            (differences["current_id"] != differences["visit_id"]).sum()
        ),
        "current_not_first_unit_visit": int(
            (differences["current_visit"] > differences["visit_number"]).sum()
        ),
    }
    print(summary)
    print(
        differences[differences["current_id"] != differences["visit_id"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
