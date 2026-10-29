"""Document the MIMIC-III >89-year date-shift sentinel from source records."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path.cwd()))

from src.ich_dynamic.build_cohorts import MIMIC_III_ROOT
from src.ich_dynamic.common import PROCESSED, PROJECT_ROOT, RESULTS


DOCUMENTATION = "https://mimic.mit.edu/docs/iii/tables/patients.html"
MIMIC4_DOCUMENTATION = "https://mimic.mit.edu/docs/iv/modules/hosp/patients.html"
RELEASE_NOTE = "https://physionet.org/content/mimiciii/1.4/"


def query_one(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, object]:
    frame = con.execute(sql).fetchdf()
    return {
        column: (value.item() if hasattr(value, "item") else value)
        for column, value in frame.iloc[0].items()
    }


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET memory_limit='2GB'")
    admissions = (MIMIC_III_ROOT / "ADMISSIONS.csv.gz").as_posix()
    patients = (MIMIC_III_ROOT / "PATIENTS.csv.gz").as_posix()
    con.execute(
        f"CREATE VIEW admissions AS "
        f"SELECT * FROM read_csv_auto('{admissions}')"
    )
    con.execute(
        f"CREATE VIEW patients AS "
        f"SELECT * FROM read_csv_auto('{patients}')"
    )
    full = query_one(
        con,
        """
        WITH ages AS (
            SELECT
                a.subject_id,
                a.hadm_id,
                a.admittime,
                p.dob,
                date_diff('year', p.dob, a.admittime) AS age_years,
                date_diff('day', p.dob, a.admittime) / 365.2425 AS age_exact
            FROM admissions a
            JOIN patients p USING (subject_id)
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id ORDER BY admittime, hadm_id
                ) AS admission_sequence
            FROM ages
            WHERE age_years >= 300
        )
        SELECT
            COUNT(*) AS sentinel_admissions,
            COUNT(DISTINCT subject_id) AS sentinel_subjects,
            MIN(age_years) AS min_sentinel_year,
            MAX(age_years) AS max_sentinel_year,
            MIN(age_exact) AS min_sentinel_exact_years,
            MAX(age_exact) AS max_sentinel_exact_years,
            SUM(CASE WHEN admission_sequence = 1 AND age_years = 300 THEN 1 ELSE 0 END)
                AS first_admissions_with_age_300,
            SUM(CASE WHEN admission_sequence = 1 THEN 1 ELSE 0 END)
                AS first_admissions_in_sentinel_subjects,
            SUM(
                CASE
                    WHEN admission_sequence = 1
                     AND date_part('month', dob) = date_part('month', admittime)
                     AND date_part('day', dob) = date_part('day', admittime)
                    THEN 1 ELSE 0
                END
            ) AS first_admissions_with_same_month_and_day
        FROM ranked
        """,
    )
    cohort_path = PROCESSED / "mimic3" / "cohort.csv"
    cohort = None
    if cohort_path.exists():
        cohort = query_one(
            con,
            f"""
            SELECT
                COUNT(*) AS n,
                SUM(CASE WHEN age_raw >= 300 THEN 1 ELSE 0 END)
                    AS sentinel_stays,
                SUM(age_gt89) AS age_gt89_stays,
                MIN(age_raw) AS min_age_raw,
                MAX(age_raw) AS max_age_raw,
                MIN(age) AS min_harmonized_age,
                MAX(age) AS max_harmonized_age,
                MIN(CASE WHEN age_raw >= 300 THEN age END)
                    AS min_harmonized_sentinel_age,
                MAX(CASE WHEN age_raw >= 300 THEN age END)
                    AS max_harmonized_sentinel_age
            FROM read_csv_auto('{cohort_path.as_posix()}')
            """
        )
    payload = {
        "documentation": DOCUMENTATION,
        "mimic4_documentation": MIMIC4_DOCUMENTATION,
        "release_note": RELEASE_NOTE,
        "documented_rule": (
            "For patients older than 89 years, DOB is set to exactly 300 "
            "years before the first admission. The resulting age is a "
            "privacy sentinel, not a recorded age."
        ),
        "analysis_recoding": {
            "sentinel": "age_raw >= 300",
            "continuous_age": 91.0,
            "indicator": "age_gt89",
            "rationale": (
                "91 matches the MIMIC-IV anchor_age convention, which sets "
                "anchor_age to 91 when it is over 89; the indicator retains "
                "interval-censoring information."
            ),
        },
        "full_database": full,
        "analysis_cohort": cohort,
    }
    output = RESULTS / "audits" / "mimic3_age_sentinel.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = RESULTS / "audits" / "mimic3_age_sentinel.md"
    markdown.write_text(
        "# MIMIC-III age sentinel audit\n\n"
        f"- Official rule: {payload['documented_rule']}\n"
        f"- Source: {DOCUMENTATION}\n"
        f"- Release note: {RELEASE_NOTE}\n"
        f"- MIMIC-IV comparison rule: {MIMIC4_DOCUMENTATION}\n"
        f"- Full database: {int(full['sentinel_admissions']):,} sentinel "
        f"admissions among {int(full['sentinel_subjects']):,} patients.\n"
        f"- First admissions: {int(full['first_admissions_with_age_300']):,}/"
        f"{int(full['first_admissions_in_sentinel_subjects']):,} have age 300; "
        f"{int(full['first_admissions_with_same_month_and_day']):,} also have "
        "the same month and day in DOB and admission time.\n"
        f"- Sentinel ages range from {full['min_sentinel_exact_years']:.6f} "
        f"to {full['max_sentinel_exact_years']:.6f} years.\n"
        "- Recoding: `age_raw >= 300` to 91 years plus `age_gt89=1`. "
        "The eICU `> 89` category uses the same rule.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"Saved {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
