"""Audit whether post-landmark unplanned intubation is identifiable in each source."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\TT paper\0811Temporal Pathway")
PROCESSED = PROJECT / "data" / "processed"
OUTPUT = PROJECT / "results" / "audits" / "intubation_source_audit.json"

MIMIC4_ROOT = SOURCE / "mimic-iv-3.1" / "icu"
MIMIC3_ROOT = (
    SOURCE
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
EICU_ROOT = SOURCE / "eicu"


def _read_frame(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    columns: str = "*",
    limit: int | None = None,
) -> pd.DataFrame:
    suffix = f" LIMIT {limit}" if limit is not None else ""
    return connection.execute(
        f"SELECT {columns} FROM read_csv_auto(?, header=true){suffix}",
        [str(path)],
    ).fetchdf()


def _mimic_item_catalog(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[dict[str, object]]:
    frame = connection.execute(
        """
        SELECT itemid, label, linksto, category
        FROM read_csv_auto(?, header=true)
        WHERE lower(label) LIKE '%intub%'
           OR lower(label) LIKE '%endotracheal%'
           OR lower(label) LIKE '%airway%'
           OR lower(label) LIKE '%ventil%'
        ORDER BY itemid
        """,
        [str(path)],
    ).fetchdf()
    return frame.to_dict(orient="records")


def _mimic_event_count(
    connection: duckdb.DuckDBPyConnection,
    dataset_root: Path,
    procedure_path: Path,
    stay_column: str,
    itemids: list[int],
) -> list[dict[str, object]]:
    cohort = pd.read_csv(dataset_root / "cohort_pre_filter.csv")
    connection.register("selected_cohort", cohort[["stay_id", "intime"]].copy())
    item_list = ",".join(str(int(item)) for item in itemids)
    query = f"""
        SELECT p.itemid, count(*) AS events,
               count(DISTINCT c.stay_id) AS stays
        FROM read_csv_auto(?, header=true) p
        JOIN selected_cohort c
          ON TRY_CAST(p.{stay_column} AS BIGINT) = TRY_CAST(c.stay_id AS BIGINT)
        WHERE TRY_CAST(p.itemid AS BIGINT) IN ({item_list})
          AND TRY_CAST(p.starttime AS TIMESTAMP)
              >= TRY_CAST(c.intime AS TIMESTAMP) + INTERVAL 24 HOUR
          AND TRY_CAST(p.starttime AS TIMESTAMP)
              <= TRY_CAST(c.intime AS TIMESTAMP) + INTERVAL 72 HOUR
        GROUP BY p.itemid
        ORDER BY p.itemid
    """
    frame = connection.execute(query, [str(procedure_path)]).fetchdf()
    return frame.to_dict(orient="records")


def _eicu_nursing_catalog(
    connection: duckdb.DuckDBPyConnection,
    nurse_path: Path,
) -> list[dict[str, object]]:
    frame = connection.execute(
        """
        SELECT nursingchartcelltypevallabel AS label,
               nursingchartcelltypevalname AS name,
               count(*) AS events,
               count(DISTINCT patientunitstayid) AS stays
        FROM read_csv_auto(?, header=true)
        WHERE lower(coalesce(nursingchartcelltypevallabel, ''))
                  LIKE '%intub%'
           OR lower(coalesce(nursingchartcelltypevallabel, ''))
                  LIKE '%airway%'
           OR lower(coalesce(nursingchartcelltypevallabel, ''))
                  LIKE '%ventil%'
           OR lower(coalesce(nursingchartcelltypevallabel, ''))
                  LIKE '%tube%'
           OR lower(coalesce(nursingchartcelltypevalname, ''))
                  LIKE '%intub%'
           OR lower(coalesce(nursingchartcelltypevalname, ''))
                  LIKE '%airway%'
           OR lower(coalesce(nursingchartcelltypevalname, ''))
                  LIKE '%ventil%'
           OR lower(coalesce(nursingchartcelltypevalname, ''))
                  LIKE '%tube%'
        GROUP BY 1, 2
        ORDER BY events DESC
        """,
        [str(nurse_path)],
    ).fetchdf()
    return frame.to_dict(orient="records")


def main() -> None:
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    audit: dict[str, object] = {
        "purpose": (
            "Determine whether non-planned intubation after the 24-hour landmark "
            "can be constructed from the licensed source files without treating a "
            "generic ventilation or airway indicator as an unplanned intubation."
        ),
        "outline_endpoint": (
            "Grade 2 includes a GCS decline >=4, unplanned intubation after the "
            "24-hour landmark, or death by 72 hours."
        ),
    }

    mimic4_items = MIMIC4_ROOT / "d_items.csv.gz"
    mimic4_procedures = MIMIC4_ROOT / "procedureevents.csv.gz"
    mimic3_items = MIMIC3_ROOT / "D_ITEMS.csv.gz"
    mimic3_procedures = MIMIC3_ROOT / "PROCEDUREEVENTS_MV.csv.gz"
    eicu_nurse = EICU_ROOT / "nurseCharting.csv.gz"
    eicu_treatment = EICU_ROOT / "treatment.csv.gz"

    audit["mimic4"] = {
        "catalog": _mimic_item_catalog(connection, mimic4_items),
        "post_landmark_events": _mimic_event_count(
            connection,
            PROCESSED / "mimic4",
            mimic4_procedures,
            "stay_id",
            [224385],
        ),
        "source_files": {
            "procedureevents": mimic4_procedures.exists(),
            "d_items": mimic4_items.exists(),
        },
    }
    audit["mimic3"] = {
        "catalog": _mimic_item_catalog(connection, mimic3_items),
        "post_landmark_events": _mimic_event_count(
            connection,
            PROCESSED / "mimic3",
            mimic3_procedures,
            "icustay_id",
            [224385],
        ),
        "source_files": {
            "procedureevents_mv": mimic3_procedures.exists(),
            "d_items": mimic3_items.exists(),
        },
    }
    audit["eicu"] = {
        "treatment_file_present": eicu_treatment.exists(),
        "nurse_catalog": _eicu_nursing_catalog(connection, eicu_nurse),
        "interpretation": (
            "Nurse-charting airway and ventilation fields establish device state "
            "or respiratory support, not whether a post-landmark intubation was "
            "unplanned. The original eICU treatment table is absent from this "
            "licensed extract."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    for dataset in ("mimic4", "mimic3", "eicu"):
        print(f"\n## {dataset}")
        section = audit[dataset]
        print(json.dumps(section, indent=2, default=str)[:12000])


if __name__ == "__main__":
    main()
