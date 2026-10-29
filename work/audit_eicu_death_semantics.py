"""Check eICU hospital offset semantics before changing death timing."""
from __future__ import annotations

from pathlib import Path

import duckdb


SOURCE = Path(r"D:\TT paper\0811Temporal Pathway\eicu")


def main() -> None:
    con = duckdb.connect()
    patient = (SOURCE / "patient.csv.gz").as_posix()
    frame = con.execute(
        f"""
        SELECT patientunitstayid AS stay_id, uniquepid, hospitalid,
               unitvisitnumber, unitadmittime24, unitdischargeoffset,
               unitdischargestatus, hospitaldischargeoffset,
               hospitaladmitoffset, hospitaldischargestatus
        FROM read_csv_auto('{patient}')
        """
    ).fetchdf()
    expired = (
        frame["hospitaldischargestatus"].fillna("").astype(str).str.lower()
        .eq("expired")
    )
    unit_expired = (
        frame["unitdischargestatus"].fillna("").astype(str).str.lower()
        .eq("expired")
    )
    print("rows", len(frame))
    print("hospital admit offset quantiles (minutes)")
    print(frame["hospitaladmitoffset"].quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]))
    print("hospital admit offset > 0", int((frame["hospitaladmitoffset"] > 0).sum()))
    print("hospital discharge offset < 0", int((frame["hospitaldischargeoffset"] < 0).sum()))
    both = frame[expired & unit_expired].copy()
    both["direct_hour"] = both["hospitaldischargeoffset"] / 60.0
    both["difference_hour"] = (
        both["hospitaldischargeoffset"] - both["hospitaladmitoffset"]
    ) / 60.0
    both["unit_hour"] = both["unitdischargeoffset"] / 60.0
    print("both expired rows", len(both))
    print("direct hospital death before unit death", int((both["direct_hour"] < both["unit_hour"]).sum()))
    print("difference from admit before unit death", int((both["difference_hour"] < both["unit_hour"]).sum()))
    print("direct within 24h", int(both["direct_hour"].between(0, 24).sum()))
    print("difference within 24h", int(both["difference_hour"].between(0, 24).sum()))
    print("exact requested row")
    print(
        frame.loc[frame["stay_id"].eq(2896176)].to_string(index=False)
    )
    print("both expired rows with direct offset before 24h")
    print(
        both.loc[
            both["direct_hour"].between(0, 24),
            [
                "stay_id",
                "hospitaladmitoffset",
                "hospitaldischargeoffset",
                "unitdischargeoffset",
                "direct_hour",
                "difference_hour",
            ],
        ].sort_values("direct_hour").to_string(index=False)
    )


if __name__ == "__main__":
    main()
