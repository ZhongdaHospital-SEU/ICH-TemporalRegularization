"""Inspect MIMIC-IV radiology report metadata used by the imaging extractor."""
from __future__ import annotations

from pathlib import Path

import duckdb

NOTE_ROOT = (
    Path(r"D:\TT paper\0811Temporal Pathway")
    / "mimic-iv-3.1"
    / "mimic iv note 2.2"
    / "note"
)


def main() -> None:
    connection = duckdb.connect()
    detail_path = str(NOTE_ROOT / "radiology_detail.csv.gz")
    radiology_path = str(NOTE_ROOT / "radiology.csv.gz")
    print(connection.execute("SELECT * FROM read_csv_auto(?) LIMIT 8", [detail_path]).fetchdf())
    print(
        connection.execute(
            """
            SELECT field_name, COUNT(*) AS n
            FROM read_csv_auto(?)
            GROUP BY field_name
            ORDER BY n DESC
            """,
            [detail_path],
        ).fetchdf()
    )
    print(
        connection.execute(
            """
            SELECT d.field_value AS exam_name, COUNT(*) AS n
            FROM read_csv_auto(?) d
            WHERE d.field_name = 'exam_name'
              AND (
                lower(d.field_value) LIKE '%head%'
                OR lower(d.field_value) LIKE '%brain%'
              )
            GROUP BY 1
            ORDER BY n DESC
            """,
            [detail_path],
        ).fetchdf().to_string(index=False)
    )
    print(
        connection.execute(
            """
            SELECT r.note_type, r.note_seq, COUNT(*) AS n
            FROM read_csv_auto(?) r
            GROUP BY 1, 2
            ORDER BY n DESC
            LIMIT 20
            """,
            [radiology_path],
        ).fetchdf()
    )


if __name__ == "__main__":
    main()
