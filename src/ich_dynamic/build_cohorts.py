"""Build harmonized ICH cohorts and 24-hour sequences from three ICU databases."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import (
    CHANNELS,
    DATASET_NAMES,
    EICU_LABS,
    LAB_ITEMS,
    LAB_RANGES,
    P020_COMPLETE,
    P020_STATUS,
    PROCESSED,
    SOURCE_ROOT,
    add_outcome,
    clamp_series,
    combine_gcs_components,
    complete_icu_followup,
    death_in_icu_within_window,
    ensure_dirs,
    hourly_medians,
    lab_summary,
    load_config,
    recorded_intubation_sensitivity_proxy,
    save_dataset,
    sequence_summary,
)


MIMIC_IV_ROOT = SOURCE_ROOT / "mimic-iv-3.1"
MIMIC_III_ROOT = (
    SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
EICU_ROOT = SOURCE_ROOT / "eicu"

PHYSIOLOGIC_RANGES = {
    "gcs_eye": (1.0, 4.0),
    "gcs_verbal": (1.0, 5.0),
    "gcs_motor": (1.0, 6.0),
    "gcs_total": (3.0, 15.0),
    "hr": (20.0, 250.0),
    "rr": (0.0, 80.0),
    "spo2": (50.0, 100.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "mbp": (20.0, 250.0),
    "temp": (25.0, 45.0),
    "urine_output": (0.0, 3000.0),
    "fio2": (0.21, 1.0),
    "rass": (-5.0, 4.0),
}

NEUROSURGERY_PROCEDURE_ITEMS = (
    225456,
    226124,
    226474,
    226475,
    229518,
    229521,
    229522,
    229523,
)
NEUROSURGERY_ICD9_PREFIXES = ("01", "02")
NEUROSURGERY_ICD10_PREFIXES = ("00", "0W", "BW")
ANTICOAGULANT_DRUG_PATTERNS = (
    "%warfarin%",
    "%apixaban%",
    "%rivaroxaban%",
    "%edoxaban%",
    "%dabigatran%",
    "%enoxaparin%",
    "%fondaparinux%",
    "%argatroban%",
    "%bivalirudin%",
    "%heparin%",
)
SEDATIVE_ITEM_IDS = (
    221668,  # midazolam
    221744,  # fentanyl
    222168,  # propofol
    225150,  # dexmedetomidine
    225154,  # morphine
    225942,  # fentanyl concentrate
    225972,  # fentanyl push
    229420,  # dexmedetomidine
    30118,
    30124,
    30126,
    30131,
    30149,
    30150,
    30308,
    43136,
    43387,
    45520,
    46301,
)


def is_ich_diagnosis(code: object, version: object) -> bool:
    """Return whether an ICD code belongs to the outline ICH definition."""
    if code is None:
        return False
    normalized = str(code).strip().upper().replace(".", "")
    if not normalized or normalized == "NAN":
        return False
    version_text = str(version).strip()
    if version_text in {"9", "9.0"}:
        return normalized.startswith("431") or normalized.startswith("432")
    if version_text in {"10", "10.0"}:
        return normalized.startswith("I61") or normalized.startswith("I62")
    return False


def derive_comorbidity_flags(
    stay_id: object,
    codes: list[tuple[object, object]],
) -> dict[str, object]:
    """Aggregate prespecified diagnosis flags from ICD-9/10 code pairs."""
    normalized = [
        str(code).strip().upper().replace(".", "")
        for code, _ in codes
        if code is not None
    ]

    def any_code(patterns: tuple[str, ...]) -> float:
        return float(any(code.startswith(patterns) for code in normalized))

    return {
        "stay_id": stay_id,
        "comorb_hypertension": any_code(("401", "402", "403", "404", "405", "I10", "I11", "I12", "I13", "I15")),
        "comorb_diabetes": any_code(("250", "E10", "E11", "E12", "E13", "E14")),
        "comorb_ckd": any_code(("585", "N18")),
        "comorb_liver_disease": any_code(("570", "571", "572", "573", "K70", "K71", "K72", "K73", "K74", "K75", "K76")),
        "comorb_atrial_fibrillation": any_code(("42731", "I48")),
        "comorb_coronary_disease": any_code(("410", "411", "412", "413", "414", "I20", "I21", "I22", "I23", "I24", "I25")),
        "comorb_heart_failure": any_code(("428", "I50")),
        "comorb_copd": any_code(("490", "491", "492", "493", "494", "495", "496", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47")),
        "coagulopathy_diagnosis": any_code(("286", "D65", "D66", "D67", "D68", "D69")),
    }


def is_neurosurgery_code(code: object, version: object) -> bool:
    """Identify broad cranial/neurosurgical procedure codes for cohort auditing."""
    if code is None:
        return False
    normalized = str(code).strip().upper().replace(".", "")
    version_text = str(version).strip()
    if version_text in {"9", "9.0"}:
        return normalized.startswith(NEUROSURGERY_ICD9_PREFIXES)
    if version_text in {"10", "10.0"}:
        return normalized.startswith(NEUROSURGERY_ICD10_PREFIXES)
    return False


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    """Open a bounded DuckDB connection so cohort builds can run in parallel."""
    con = duckdb.connect()
    threads = int(os.environ.get("ICH_DUCKDB_THREADS", "5"))
    memory_limit = os.environ.get("ICH_DUCKDB_MEMORY", "4GB")
    con.execute(f"SET threads={max(1, threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    return con


def create_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    path: Path,
    all_varchar: bool = False,
) -> None:
    options = ", header=true, all_varchar=true" if all_varchar else ", header=true"
    con.execute(
        f"CREATE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}'{options})"
    )


def build_vitals_long(
    frame: pd.DataFrame,
    value_map: dict[str, float],
    item_col: str = "itemid",
) -> pd.DataFrame:
    """Convert a long event table to the common (stay_id, hour, channel, value) form."""
    work = frame[frame[item_col].isin(value_map)].copy()
    work["channel"] = work[item_col].map(value_map)
    work["value"] = pd.to_numeric(work["valuenum"], errors="coerce")
    work["time_hour"] = pd.to_numeric(work["time_hour"], errors="coerce")
    work["hour"] = np.floor(work["time_hour"]).astype("Int64")
    work = work[work["hour"].between(0, 71)]
    work = work.dropna(subset=["value", "hour"])
    return (
        work.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )


def assemble_hourly(
    vitals: pd.DataFrame,
    gcs: pd.DataFrame,
) -> pd.DataFrame:
    gcs_long = gcs.copy()
    if not gcs_long.empty:
        gcs_long["time_hour"] = pd.to_numeric(gcs_long["time_hour"], errors="coerce")
        gcs_long["hour"] = np.floor(gcs_long["time_hour"]).astype("Int64")
        gcs_long = gcs_long[gcs_long["hour"].between(0, 71)].dropna(
            subset=["gcs", "hour"]
        )
        gcs_hourly = (
            gcs_long.groupby(["stay_id", "hour"], as_index=False)["gcs"]
            .median()
            .rename(columns={"gcs": "value"})
        )
        gcs_hourly["channel"] = "gcs_total"
    else:
        gcs_hourly = pd.DataFrame(columns=["stay_id", "hour", "value", "channel"])
    out = pd.concat([vitals, gcs_hourly], ignore_index=True)
    if out.empty:
        return out
    out = (
        out.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
        .reset_index(drop=True)
    )
    return out


def clamp_hourly_values(hourly: pd.DataFrame) -> pd.DataFrame:
    """Remove physiologically implausible values before model preprocessing."""
    out = hourly.copy()
    keep = pd.Series(False, index=out.index)
    for channel, (low, high) in PHYSIOLOGIC_RANGES.items():
        mask = out["channel"] == channel
        out.loc[mask, "value"] = clamp_series(
            out.loc[mask, "value"], low, high
        )
        keep |= mask
    return out[keep].dropna(subset=["value"]).reset_index(drop=True)


def prepare_rass_events(
    events: pd.DataFrame,
    value_column: str = "value",
    start_hour: float = 0.0,
    end_hour: float = 72.0,
) -> pd.DataFrame:
    """Normalize event-level RASS without aggregation, fill, or zero coding."""
    columns = ["stay_id", "time_hour", "rass"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    if "stay_id" not in events or "time_hour" not in events:
        raise ValueError("RASS events require stay_id and time_hour columns.")
    if value_column not in events and value_column == "value" and "rass" in events:
        value_column = "rass"
    if value_column not in events:
        raise ValueError(f"RASS events are missing value column {value_column!r}.")
    out = events.copy()
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce")
    out["time_hour"] = pd.to_numeric(out["time_hour"], errors="coerce")
    out["rass"] = pd.to_numeric(out[value_column], errors="coerce")
    keep = (
        out["stay_id"].notna()
        & out["time_hour"].between(start_hour, end_hour, inclusive="both")
        & out["rass"].between(-5.0, 4.0)
    )
    out = out.loc[keep, columns].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["stay_id"] = out["stay_id"].astype(np.int64)
    return out.sort_values(["stay_id", "time_hour"]).reset_index(drop=True)


def eicu_rass_events_from_nurse(nurse: pd.DataFrame) -> pd.DataFrame:
    """Extract eICU RASS values without mixing sedation scales.

    eICU stores some RASS observations as two rows at the same chart offset:
    a ``Sedation Scale`` row whose value is ``RASS`` and a ``Sedation Score``
    row with the numeric score. Those paired rows are accepted only when the
    scale row explicitly identifies RASS. Native RASS/SEDATION SCORE rows are
    accepted directly.
    """
    required = {
        "stay_id",
        "time_hour",
        "nursingchartoffset",
        "label",
        "item",
        "value",
    }
    missing = required.difference(nurse.columns)
    if missing:
        raise ValueError(
            "eICU nurse charting lacks columns: " + ", ".join(sorted(missing))
        )
    work = nurse.copy()
    label = (
        work["label"]
        .fillna("")
        .astype(str)
        .str.lower()
    )
    item = (
        work["item"]
        .fillna("")
        .astype(str)
        .str.lower()
    )
    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    explicit = work[
        label.isin(["rass", "sedation score"]) & item.eq("value")
    ].copy()
    generic = label.eq("sedation scale/score/goal")
    raw_value = (
        work["nursingchartvalue"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        if "nursingchartvalue" in work
        else pd.Series("", index=work.index, dtype=object)
    )
    rass_offsets = work.loc[
        generic & item.eq("sedation scale") & raw_value.eq("RASS"),
        ["stay_id", "nursingchartoffset"],
    ].drop_duplicates()
    paired = work.loc[
        generic
        & item.eq("sedation score")
        & work["value"].between(-5.0, 4.0)
    ].merge(
        rass_offsets,
        on=["stay_id", "nursingchartoffset"],
        how="inner",
        validate="many_to_one",
    )
    combined = pd.concat([explicit, paired], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["stay_id", "nursingchartoffset", "value"]
    )
    return prepare_rass_events(combined)


def write_rass_events(path: Path, events: pd.DataFrame) -> Path:
    """Atomically write the complete event-level RASS table."""
    normalized = prepare_rass_events(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    normalized.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def save_rass_events(dataset: str, events: pd.DataFrame) -> Path:
    normalized = prepare_rass_events(events)
    path = write_rass_events(
        PROCESSED / dataset / "rass_events.csv",
        normalized,
    )
    print(
        f"[{dataset}] saved event-level RASS rows={len(normalized)}",
        flush=True,
    )
    return path


def summarize_deep_sedation(
    events: pd.DataFrame,
    start_hour: float = 0.0,
    end_hour: float = 24.0,
) -> pd.DataFrame:
    """Return explicit deep-sedation state while preserving non-measurement."""
    columns = [
        "stay_id",
        "min_rass_24h",
        "any_rass_le_minus2_24h",
        "any_deep_sedation_24h",
        "deep_sedation_observed_24h",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    work = events.copy()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["time_hour"] = pd.to_numeric(work["time_hour"], errors="coerce")
    work = work[
        work["time_hour"].between(start_hour, end_hour, inclusive="left")
        & work["value"].between(-5, 4)
        & work["stay_id"].notna()
    ]
    if work.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        work.groupby("stay_id", as_index=False)
        .agg(min_rass=("value", "min"), n_rass=("value", "size"))
        .assign(
            min_rass_24h=lambda frame: frame["min_rass"],
            any_rass_le_minus2_24h=lambda frame: frame["min_rass"].le(-2).astype(float),
            any_deep_sedation_24h=lambda frame: frame["min_rass"].le(-3).astype(float),
            deep_sedation_observed_24h=1.0,
        )
    )
    summary["stay_id"] = summary["stay_id"].astype(np.int64)
    return summary[columns]


def attach_gcs_time_hour(
    gcs: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Join the ICU anchor and calculate elapsed hours without index alignment."""
    out = gcs.copy()
    out["charttime"] = pd.to_datetime(out["charttime"], errors="coerce")
    anchors = cohort[["stay_id", "intime"]].copy()
    anchors["stay_id"] = pd.to_numeric(anchors["stay_id"], errors="coerce")
    anchors["intime"] = pd.to_datetime(anchors["intime"], errors="coerce")
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce")
    out = out.merge(anchors, on="stay_id", how="left", validate="many_to_one")
    out["time_hour"] = (
        out["charttime"] - out["intime"]
    ).dt.total_seconds() / 3600.0
    return out.drop(columns=["intime"])


def attach_mimic_contextual_flags(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    chart: pd.DataFrame,
) -> pd.DataFrame:
    """Attach pre-landmark ventilation, sedation, anticoagulation, and diagnoses."""
    out = cohort.copy()
    diagnoses = con.execute(
        """
        SELECT c.stay_id, d.icd_code, d.icd_version
        FROM context_diagnoses d
        JOIN (SELECT DISTINCT stay_id, hadm_id FROM cohort_selected) c
          ON d.hadm_id = c.hadm_id
        """
    ).fetchdf()
    diagnosis_flags = pd.DataFrame(
        [
            derive_comorbidity_flags(
                stay_id,
                list(zip(group["icd_code"], group["icd_version"])),
            )
            for stay_id, group in diagnoses.groupby("stay_id")
        ]
    )
    if diagnosis_flags.empty:
        diagnosis_flags = pd.DataFrame(
            columns=[
                "stay_id",
                "comorb_hypertension",
                "comorb_diabetes",
                "comorb_ckd",
                "comorb_liver_disease",
                "comorb_atrial_fibrillation",
                "comorb_coronary_disease",
                "comorb_heart_failure",
                "comorb_copd",
                "coagulopathy_diagnosis",
            ]
        )

    procedures = con.execute(
        """
        SELECT c.stay_id, p.icd_code, p.icd_version
        FROM context_procedures p
        JOIN (SELECT DISTINCT stay_id, hadm_id FROM cohort_selected) c
          ON p.hadm_id = c.hadm_id
        """
    ).fetchdf()
    neurosurgery_admission = (
        procedures.assign(
            flag=[
                is_neurosurgery_code(code, version)
                for code, version in zip(
                    procedures["icd_code"], procedures["icd_version"]
                )
            ]
        )
        .groupby("stay_id", as_index=False)["flag"]
        .max()
        .rename(columns={"flag": "neurosurgery_admission"})
        if not procedures.empty
        else pd.DataFrame(columns=["stay_id", "neurosurgery_admission"])
    )

    event_flags = con.execute(
        f"""
        SELECT
            c.stay_id,
            MAX(
                CASE WHEN p.itemid = 225792
                          AND p.starttime < c.intime + INTERVAL 24 HOUR
                          AND COALESCE(p.endtime, p.starttime) >= c.intime
                          AND lower(COALESCE(p.statusdescription, '')) NOT LIKE '%cancel%'
                     THEN 1 ELSE 0 END
            ) AS mechanical_ventilation_24h,
            MAX(
                CASE WHEN p.itemid IN ({','.join(str(x) for x in NEUROSURGERY_PROCEDURE_ITEMS)})
                          AND p.starttime < c.intime + INTERVAL 24 HOUR
                          AND COALESCE(p.endtime, p.starttime) >= c.intime
                          AND lower(COALESCE(p.statusdescription, '')) NOT LIKE '%cancel%'
                     THEN 1 ELSE 0 END
            ) AS neurosurgery_24h
        FROM context_procedure_events p
        JOIN cohort_selected c ON p.stay_id = c.stay_id
        GROUP BY c.stay_id
        """
    ).fetchdf()

    sedation = con.execute(
        f"""
        SELECT c.stay_id
        FROM context_inputevents i
        JOIN cohort_selected c ON i.stay_id = c.stay_id
        WHERE i.itemid IN ({','.join(str(x) for x in SEDATIVE_ITEM_IDS)})
          AND i.starttime < c.intime + INTERVAL 24 HOUR
          AND COALESCE(i.endtime, i.starttime) >= c.intime
          AND lower(COALESCE(i.statusdescription, '')) NOT LIKE '%cancel%'
        GROUP BY c.stay_id
        """
    ).fetchdf()
    sedation["sedation_exposure_24h"] = 1.0

    anticoag_pattern = " OR ".join(
        f"lower(r.drug) LIKE '{pattern}'"
        for pattern in ANTICOAGULANT_DRUG_PATTERNS
    )
    anticoag = con.execute(
        f"""
        SELECT c.stay_id
        FROM context_prescriptions r
        JOIN (SELECT DISTINCT stay_id, hadm_id, intime FROM cohort_selected) c
          ON r.hadm_id = c.hadm_id
        WHERE ({anticoag_pattern})
          AND lower(COALESCE(r.drug, '')) NOT LIKE '%flush%'
          AND r.starttime < c.intime + INTERVAL 24 HOUR
          AND COALESCE(r.stoptime, r.starttime) >= c.intime
        GROUP BY c.stay_id
        """
    ).fetchdf()
    anticoag["anticoagulant_use_24h"] = 1.0

    chart_vent = chart[
        chart["itemid"].eq(226260)
        & chart["time_hour"].between(0.0, 24.0, inclusive="left")
        & (
            pd.to_numeric(chart["valuenum"], errors="coerce").eq(1)
            | chart["value"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(r"\byes\b|true", regex=True)
        )
    ]
    chart_vent = (
        chart_vent[["stay_id"]]
        .drop_duplicates()
        .assign(mechanical_ventilation_chart_24h=1.0)
    )

    for frame, value_columns in (
        (diagnosis_flags, None),
        (neurosurgery_admission, None),
        (event_flags, None),
        (sedation, None),
        (anticoag, None),
        (chart_vent, None),
    ):
        if value_columns is None and not frame.empty:
            out = out.merge(frame, on="stay_id", how="left")
    for column in (
        "comorb_hypertension",
        "comorb_diabetes",
        "comorb_ckd",
        "comorb_liver_disease",
        "comorb_atrial_fibrillation",
        "comorb_coronary_disease",
        "comorb_heart_failure",
        "comorb_copd",
        "coagulopathy_diagnosis",
        "neurosurgery_admission",
        "mechanical_ventilation_24h",
        "neurosurgery_24h",
        "sedation_exposure_24h",
        "anticoagulant_use_24h",
        "mechanical_ventilation_chart_24h",
    ):
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["mechanical_ventilation_24h"] = out[
        ["mechanical_ventilation_24h", "mechanical_ventilation_chart_24h"]
    ].max(axis=1)
    out["anticoagulation_related_hemorrhage"] = out[
        ["anticoagulant_use_24h", "coagulopathy_diagnosis"]
    ].max(axis=1)
    return out.drop(columns=["mechanical_ventilation_chart_24h"])


def attach_eicu_contextual_flags(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Attach eICU diagnosis and Day-1 ventilation status without imputing sedation."""
    out = cohort.copy()
    diagnoses = con.execute(
        """
        SELECT c.stay_id, d.icd9code,
               lower(COALESCE(d.diagnosisstring, '')) AS diagnosisstring
        FROM diagnosis d
        JOIN (SELECT DISTINCT stay_id FROM cohort_selected) c
          ON d.patientunitstayid = c.stay_id
        """
    ).fetchdf()
    rows = []
    for stay_id, group in diagnoses.groupby("stay_id"):
        codes = []
        for code_field in group["icd9code"].dropna():
            for code in str(code_field).split(","):
                normalized = code.strip().upper()
                if not normalized:
                    continue
                version = "10" if normalized.startswith("I") else "9"
                codes.append((normalized, version))
        flags = derive_comorbidity_flags(stay_id, codes)
        text = " ".join(group["diagnosisstring"].fillna("").astype(str))
        flags["neurosurgery_admission"] = float(
            any(
                token in text
                for token in (
                    "craniotomy",
                    "craniectomy",
                    "evacuation",
                    "ventriculostomy",
                    "burr hole",
                    "icp monitor",
                )
            )
        )
        flags["anticoagulation_related_hemorrhage"] = max(
            float(flags["coagulopathy_diagnosis"]),
            float("anticoagul" in text),
        )
        rows.append(flags)
    diagnosis_flags = pd.DataFrame(rows)
    if not diagnosis_flags.empty:
        diagnosis_flags["neurosurgery_24h"] = diagnosis_flags[
            "neurosurgery_admission"
        ]
        out = out.merge(diagnosis_flags, on="stay_id", how="left")
    for column in (
        "comorb_hypertension",
        "comorb_diabetes",
        "comorb_ckd",
        "comorb_liver_disease",
        "comorb_atrial_fibrillation",
        "comorb_coronary_disease",
        "comorb_heart_failure",
        "comorb_copd",
        "coagulopathy_diagnosis",
        "neurosurgery_admission",
        "neurosurgery_24h",
        "anticoagulation_related_hemorrhage",
    ):
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["sedation_exposure_24h"] = np.nan
    out["anticoagulant_use_24h"] = np.nan

    ventilation = con.execute(
        """
        SELECT c.stay_id,
               MAX(CASE WHEN TRY_CAST(a.vent AS DOUBLE) > 0
                        THEN 1 ELSE 0 END) AS mechanical_ventilation_24h
        FROM apacheapsvar a
        JOIN (SELECT DISTINCT stay_id FROM cohort_selected) c
          ON a.patientunitstayid = c.stay_id
        GROUP BY c.stay_id
        """
    ).fetchdf()
    if not ventilation.empty:
        out = out.merge(ventilation, on="stay_id", how="left")
    if "mechanical_ventilation_24h" not in out:
        out["mechanical_ventilation_24h"] = np.nan
    out["mechanical_ventilation_24h"] = pd.to_numeric(
        out["mechanical_ventilation_24h"], errors="coerce"
    )
    return out


def harmonize_mimic_death(cohort: pd.DataFrame) -> pd.DataFrame:
    """Recompute death timing without applying a 24-hour landmark filter."""
    out = cohort.copy()
    out["intime"] = pd.to_datetime(out["intime"], errors="coerce")
    out["deathtime"] = pd.to_datetime(out["deathtime"], errors="coerce")
    out["death_hour_from_icu"] = (
        out["deathtime"] - out["intime"]
    ).dt.total_seconds() / 3600.0
    out["death_within_72h"] = out["death_hour_from_icu"].between(0, 72).fillna(False)
    return out


def harmonize_eicu_death(cohort: pd.DataFrame) -> pd.DataFrame:
    """Derive death time relative to ICU admission.

    eICU encodes both unitdischargeoffset and hospitaldischargeoffset as minutes
    from ICU admission. hospitaladmitoffset is separately relative to ICU
    admission and must not be subtracted from hospitaldischargeoffset.
    """
    out = cohort.copy()
    unit_death = (
        out["unitdischargestatus"].fillna("").astype(str).str.lower().eq("expired")
    )
    hospital_death = (
        out["hospitaldischargestatus"].fillna("").astype(str).str.lower().eq("expired")
    )
    unit_hour = pd.to_numeric(out["unitdischargeoffset"], errors="coerce") / 60.0
    hospital_hour = (
        pd.to_numeric(out["hospitaldischargeoffset"], errors="coerce") / 60.0
    )
    death_hour = pd.Series(np.nan, index=out.index, dtype=float)
    death_hour = death_hour.where(~unit_death, unit_hour)
    death_hour = np.fmin(
        death_hour.to_numpy(dtype=float),
        np.where(hospital_death, hospital_hour, np.nan),
    )
    out["death_hour_from_icu"] = death_hour
    out["death_within_72h"] = out["death_hour_from_icu"].between(0, 72).fillna(False)
    return out


def harmonize_mimic_age(age: pd.Series) -> pd.Series:
    """Recode the MIMIC-III date-shift sentinel without discarding old age.

    For patients older than 89 years, MIMIC-III sets DOB exactly 300 years
    before the first admission, so age is 300 at that admission and can be
    301 or greater at later admissions. This is a privacy mechanism, not a
    recorded age. Recode it to 91 years to match the MIMIC-IV anchor_age
    convention; callers retain an ``age_gt89`` indicator separately.
    """
    numeric = pd.to_numeric(age, errors="coerce")
    return numeric.mask(numeric >= 300, 91.0)


def build_mimic4(config: dict) -> None:
    name = "mimic4"
    con = connect_duckdb()
    for view, rel in {
        "admissions": "hosp/admissions.csv.gz",
        "patients": "hosp/patients.csv.gz",
        "diagnoses": "hosp/diagnoses_icd.csv.gz",
        "icustays": "icu/icustays.csv.gz",
        "chartevents": "icu/chartevents.csv.gz",
        "outputevents": "icu/outputevents.csv.gz",
        "labevents": "hosp/labevents.csv.gz",
        "procedureevents": "icu/procedureevents.csv.gz",
        "inputevents": "icu/inputevents.csv.gz",
        "prescriptions": "hosp/prescriptions.csv.gz",
        "procedures_icd": "hosp/procedures_icd.csv.gz",
    }.items():
        create_view(con, view, MIMIC_IV_ROOT / rel)
    con.execute(
        """
        CREATE VIEW context_diagnoses AS
        SELECT hadm_id, icd_code, icd_version FROM diagnoses
        """
    )
    con.execute(
        """
        CREATE VIEW context_procedures AS
        SELECT hadm_id, icd_code, icd_version FROM procedures_icd
        """
    )
    con.execute(
        """
        CREATE VIEW context_procedure_events AS
        SELECT stay_id, itemid, starttime, endtime, statusdescription
        FROM procedureevents
        """
    )
    con.execute(
        """
        CREATE VIEW context_inputevents AS
        SELECT stay_id, itemid, starttime, endtime, statusdescription
        FROM inputevents
        """
    )
    con.execute(
        """
        CREATE VIEW context_prescriptions AS
        SELECT hadm_id, starttime, stoptime, drug FROM prescriptions
        """
    )

    cohort = con.execute(
        """
        WITH ich AS (
            SELECT hadm_id
            FROM diagnoses
            GROUP BY hadm_id
            HAVING MAX(
                       CASE WHEN (
                                      icd_version = '10'
                                      AND (
                                          icd_code LIKE 'I61%'
                                          OR icd_code LIKE 'I62%'
                                      )
                                  )
                                  OR (
                                      icd_version = '9'
                                      AND (
                                          icd_code LIKE '431%'
                                          OR icd_code LIKE '432%'
                                      )
                                  )
                            THEN 1 ELSE 0 END
                   ) = 1
        ),
        candidates AS (
            SELECT
                i.stay_id,
                i.hadm_id,
                i.subject_id,
                i.intime,
                i.outtime,
                i.first_careunit,
                a.admission_type,
                a.race,
                a.deathtime,
                a.hospital_expire_flag,
                p.gender,
                p.anchor_age + date_part('year', i.intime) - p.anchor_year AS age,
                CASE WHEN ich.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS has_ich,
                date_diff('minute', i.intime, i.outtime) / 60.0 AS icu_out_hour,
                CASE WHEN a.deathtime IS NOT NULL
                       AND a.deathtime <= i.intime + INTERVAL 72 HOUR
                     THEN TRUE ELSE FALSE END AS death_within_72h
            FROM icustays i
            LEFT JOIN ich ON i.hadm_id = ich.hadm_id
            JOIN admissions a ON i.hadm_id = a.hadm_id
            JOIN patients p ON i.subject_id = p.subject_id
        ),
        ranked AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY subject_id ORDER BY intime, stay_id
                    ) AS patient_sequence
                FROM candidates
            )
            WHERE patient_sequence = 1
        ),
        eligible AS (
            SELECT *
            FROM ranked
            WHERE patient_sequence = 1
              AND has_ich = 1
              AND age >= 18
        )
        SELECT *
        FROM eligible
        """
    ).fetchdf()
    cohort.columns = [column.lower() for column in cohort.columns]
    cohort = harmonize_mimic_death(cohort)
    cohort["dataset"] = name
    cohort["subject_key"] = "mimic4_" + cohort["subject_id"].astype(str)
    cohort["hospital_key"] = "mimic4"
    cohort["age"] = harmonize_mimic_age(cohort["age"])
    cohort["age_gt89"] = cohort["age"].ge(90).astype(float)
    cohort["sex_male"] = (cohort["gender"].astype(str) == "M").astype(float)
    cohort["race_white"] = (
        cohort["race"].astype(str).str.upper().str.contains("WHITE")
    ).astype(float)
    cohort["emergency_admission"] = (
        cohort["admission_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains("EMER|URGENT|DIRECT EMER")
    ).astype(float)
    con.register(
        "cohort_selected",
        cohort[["stay_id", "hadm_id", "intime"]].copy(),
    )
    raw_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT i.stay_id)
            FROM icustays i
            WHERE EXISTS (
                    SELECT 1
                    FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id
                      AND (
                          (
                              d.icd_version = '10'
                              AND (
                                  d.icd_code LIKE 'I61%'
                                  OR d.icd_code LIKE 'I62%'
                              )
                          )
                          OR (
                              d.icd_version = '9'
                              AND (
                                  d.icd_code LIKE '431%'
                                  OR d.icd_code LIKE '432%'
                              )
                          )
                      )
                )
            """
        ).fetchone()[0]
    )

    chart = con.execute(
        """
        SELECT ce.stay_id, ce.charttime, ce.itemid, ce.valuenum, ce.value,
               date_diff('second', c.intime, ce.charttime) / 3600.0 AS time_hour
        FROM chartevents ce
        JOIN cohort_selected c ON ce.stay_id = c.stay_id
        WHERE ce.itemid IN (
            220045, 220210, 220277, 220050, 220051, 220052,
            220179, 220180, 220181,
            223761, 223762, 220739, 223900, 223901, 228096, 223835,
            226260
        )
          AND ce.charttime >= c.intime
          AND ce.charttime <= c.intime + INTERVAL 72 HOUR
        """
    ).fetchdf()

    gcs = combine_gcs_components(
        chart[chart["itemid"].isin([220739, 223900, 223901])]
    )
    gcs = attach_gcs_time_hour(gcs, cohort)

    vital_map = {
        220045: "hr",
        220210: "rr",
        220277: "spo2",
        220050: "sbp",
        220051: "dbp",
        220052: "mbp",
        220179: "sbp",
        220180: "dbp",
        220181: "mbp",
        220739: "gcs_eye",
        223900: "gcs_verbal",
        223901: "gcs_motor",
        228096: "rass",
        223835: "fio2",
    }
    vitals = build_vitals_long(chart, vital_map)
    fio2 = vitals["channel"].eq("fio2")
    vitals.loc[fio2, "value"] = vitals.loc[fio2, "value"].where(
        vitals.loc[fio2, "value"].between(0.21, 1.0)
    ).fillna(
        vitals.loc[fio2, "value"].where(
            vitals.loc[fio2, "value"].between(21.0, 100.0)
        )
        / 100.0
    )
    temp_c = chart[chart["itemid"] == 223762].copy()
    temp_c["channel"] = "temp"
    temp_c["value"] = pd.to_numeric(temp_c["valuenum"], errors="coerce")
    temp_c["hour"] = np.floor(temp_c["time_hour"]).astype("Int64")
    temp_f = chart[chart["itemid"] == 223761].copy()
    temp_f["channel"] = "temp"
    temp_f["value"] = (
        pd.to_numeric(temp_f["valuenum"], errors="coerce") - 32.0
    ) * 5.0 / 9.0
    temp_f["hour"] = np.floor(temp_f["time_hour"]).astype("Int64")
    temp = pd.concat([temp_c, temp_f], ignore_index=True)
    temp = temp[temp["hour"].between(0, 71)].dropna(subset=["value", "hour"])
    temp = (
        temp.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    vitals = pd.concat([vitals, temp], ignore_index=True)
    urine = con.execute(
        """
        SELECT o.stay_id, o.charttime,
               date_diff('second', c.intime, o.charttime) / 3600.0 AS time_hour,
               o.value
        FROM outputevents o
        JOIN cohort_selected c ON o.stay_id = c.stay_id
        WHERE o.itemid IN (
            226557, 226558, 226559, 226560, 226561, 226563,
            226564, 226565, 226566, 226567, 226584, 226627,
            226631, 227488, 227489
        )
          AND o.charttime >= c.intime
          AND o.charttime < c.intime + INTERVAL 72 HOUR
          AND o.value IS NOT NULL
        """
    ).fetchdf()
    urine.columns = [column.lower() for column in urine.columns]
    urine["value"] = pd.to_numeric(urine["value"], errors="coerce")
    urine["hour"] = np.floor(
        pd.to_numeric(urine["time_hour"], errors="coerce")
    ).astype("Int64")
    urine["channel"] = "urine_output"
    urine = urine[urine["hour"].between(0, 71)].dropna(subset=["value", "hour"])
    urine = (
        urine.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .sum()
        .sort_values(["stay_id", "hour", "channel"])
    )
    vitals = pd.concat([vitals, urine], ignore_index=True)
    hourly = assemble_hourly(vitals, gcs)
    hourly = clamp_hourly_values(hourly)

    rass = chart[chart["itemid"] == 228096].copy()
    rass["value"] = pd.to_numeric(rass["valuenum"], errors="coerce")
    rass_events = prepare_rass_events(rass)
    cohort = cohort.merge(
        summarize_deep_sedation(
            rass_events.rename(columns={"rass": "value"})
        ),
        on="stay_id",
        how="left",
    )
    cohort["any_deep_sedation_24h"] = pd.to_numeric(
        cohort["any_deep_sedation_24h"], errors="coerce"
    )
    cohort["min_rass_24h"] = pd.to_numeric(
        cohort["min_rass_24h"], errors="coerce"
    )
    cohort["any_rass_le_minus2_24h"] = pd.to_numeric(
        cohort["any_rass_le_minus2_24h"], errors="coerce"
    )
    cohort["deep_sedation_observed_24h"] = pd.to_numeric(
        cohort["deep_sedation_observed_24h"], errors="coerce"
    ).fillna(0.0)

    labs = extract_mimic_labs(con, cohort, "mimic4")
    recorded_intubation = con.execute(
        """
        SELECT p.stay_id, MIN(p.starttime) AS intubation_time
        FROM procedureevents p
        JOIN cohort_selected c ON p.stay_id = c.stay_id
        WHERE p.itemid = 224385
          AND p.starttime >= c.intime + INTERVAL 24 HOUR
          AND p.starttime < c.intime + INTERVAL 72 HOUR
        GROUP BY p.stay_id
        """
    ).fetchdf()
    cohort["recorded_intubation_24_72h"] = cohort["stay_id"].isin(
        recorded_intubation["stay_id"]
    ).astype(float)
    cohort = attach_mimic_contextual_flags(con, cohort, chart)

    cohort = add_outcome(cohort, gcs, include_intubation=False)
    cohort = finish_cohort(cohort, hourly, labs, config, name=name)
    prefilter_n = len(pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv"))
    flow = make_flow(
        name,
        raw_n,
        cohort,
        prefilter_n,
        bool(config.get("primary_include_early_live_transfer", True)),
    )
    save_rass_events(name, rass_events)
    save_dataset(name, cohort, gcs, hourly, flow, config)


def build_mimic3(config: dict) -> None:
    name = "mimic3"
    con = connect_duckdb()
    for view, rel in {
        "admissions": "ADMISSIONS.csv.gz",
        "patients": "PATIENTS.csv.gz",
        "diagnoses": "DIAGNOSES_ICD.csv.gz",
        "icustays": "ICUSTAYS.csv.gz",
        "chartevents": "CHARTEVENTS.csv.gz",
        "outputevents": "OUTPUTEVENTS.csv.gz",
        "labevents": "LABEVENTS.csv.gz",
        "procedureevents": "PROCEDUREEVENTS_MV.csv.gz",
        "inputevents_mv": "INPUTEVENTS_MV.csv.gz",
        "inputevents_cv": "INPUTEVENTS_CV.csv.gz",
        "prescriptions": "PRESCRIPTIONS.csv.gz",
        "procedures_icd": "PROCEDURES_ICD.csv.gz",
    }.items():
        create_view(
            con,
            view,
            MIMIC_III_ROOT / rel,
            all_varchar=(view == "chartevents"),
        )
    con.execute(
        """
        CREATE VIEW context_diagnoses AS
        SELECT hadm_id, icd9_code AS icd_code, '9' AS icd_version
        FROM diagnoses
        """
    )
    con.execute(
        """
        CREATE VIEW context_procedures AS
        SELECT hadm_id, icd9_code AS icd_code, '9' AS icd_version
        FROM procedures_icd
        """
    )
    con.execute(
        """
        CREATE VIEW context_procedure_events AS
        SELECT icustay_id AS stay_id, itemid, starttime, endtime,
               statusdescription
        FROM procedureevents
        """
    )
    con.execute(
        """
        CREATE VIEW context_inputevents AS
        SELECT icustay_id AS stay_id, itemid, starttime, endtime,
               statusdescription
        FROM inputevents_mv
        UNION ALL
        SELECT icustay_id AS stay_id, itemid, charttime AS starttime,
               charttime AS endtime, stopped AS statusdescription
        FROM inputevents_cv
        """
    )
    con.execute(
        """
        CREATE VIEW context_prescriptions AS
        SELECT hadm_id, startdate AS starttime, enddate AS stoptime, drug
        FROM prescriptions
        """
    )

    cohort = con.execute(
        """
        WITH ich AS (
            SELECT hadm_id
            FROM diagnoses
            GROUP BY hadm_id
            HAVING MAX(
                       CASE WHEN icd9_code LIKE '431%'
                                  OR icd9_code LIKE '432%'
                            THEN 1 ELSE 0 END
                   ) = 1
        ),
        candidates AS (
            SELECT
                i.icustay_id AS stay_id,
                i.hadm_id,
                i.subject_id,
                i.intime,
                i.outtime,
                i.first_careunit,
                a.admission_type,
                a.ethnicity AS race,
                a.deathtime,
                a.hospital_expire_flag,
                p.gender,
                CASE WHEN ich.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS has_ich,
                date_diff('year', p.dob, a.admittime) AS age,
                date_diff('minute', i.intime, i.outtime) / 60.0 AS icu_out_hour,
                CASE WHEN a.deathtime IS NOT NULL
                       AND a.deathtime <= i.intime + INTERVAL 72 HOUR
                     THEN TRUE ELSE FALSE END AS death_within_72h
            FROM icustays i
            LEFT JOIN ich ON i.hadm_id = ich.hadm_id
            JOIN admissions a ON i.hadm_id = a.hadm_id
            JOIN patients p ON i.subject_id = p.subject_id
        ),
        ranked AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY subject_id ORDER BY intime, stay_id
                    ) AS patient_sequence
                FROM candidates
            )
            WHERE patient_sequence = 1
        ),
        eligible AS (
            SELECT *
            FROM ranked
            WHERE patient_sequence = 1
              AND has_ich = 1
              AND age >= 18
        )
        SELECT *
        FROM eligible
        """
    ).fetchdf()
    cohort.columns = [column.lower() for column in cohort.columns]
    cohort = harmonize_mimic_death(cohort)
    cohort["dataset"] = name
    cohort["subject_key"] = "mimic3_" + cohort["subject_id"].astype(str)
    cohort["hospital_key"] = "mimic3"
    cohort["age_raw"] = pd.to_numeric(cohort["age"], errors="coerce")
    cohort["age"] = harmonize_mimic_age(cohort["age_raw"])
    cohort["age_gt89"] = cohort["age_raw"].ge(300).astype(float)
    cohort["sex_male"] = (cohort["gender"].astype(str) == "M").astype(float)
    cohort["race_white"] = (
        cohort["race"].astype(str).str.upper().str.contains("WHITE")
    ).astype(float)
    cohort["emergency_admission"] = (
        cohort["admission_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains("EMER|URGENT")
    ).astype(float)
    con.register(
        "cohort_selected",
        cohort[["stay_id", "hadm_id", "intime"]].copy(),
    )
    raw_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT i.icustay_id)
            FROM icustays i
            WHERE EXISTS (
                    SELECT 1
                    FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id
                      AND (
                          d.icd9_code LIKE '431%'
                          OR d.icd9_code LIKE '432%'
                      )
                )
            """
        ).fetchone()[0]
    )

    chart = con.execute(
        """
        SELECT ce.icustay_id AS stay_id, ce.charttime, ce.itemid,
               ce.valuenum, ce.value,
               date_diff(
                   'second', c.intime, CAST(ce.charttime AS TIMESTAMP)
               ) / 3600.0 AS time_hour
        FROM chartevents ce
        JOIN cohort_selected c
          ON ce.icustay_id = c.stay_id
        WHERE TRY_CAST(ce.itemid AS INTEGER) IN (
            211, 220045, 618, 220210, 646, 220277,
            51, 455, 220050, 220179, 8368, 8440, 8441,
            220051, 220180, 52, 456, 220052, 220181, 676, 677, 678, 679,
            223761, 223762,
            198, 220739, 223900, 223901, 228096, 223835,
            628, 1090, 3611, 226260
        )
          AND CAST(ce.charttime AS TIMESTAMP) >= c.intime
          AND CAST(ce.charttime AS TIMESTAMP) <= c.intime + INTERVAL 72 HOUR
        """
    ).fetchdf()
    chart.columns = [column.lower() for column in chart.columns]
    chart["itemid"] = pd.to_numeric(chart["itemid"], errors="coerce")
    chart["valuenum"] = pd.to_numeric(chart["valuenum"], errors="coerce")
    gcs = combine_gcs_components(
        chart[chart["itemid"].isin([198, 220739, 223900, 223901])]
    )
    gcs = attach_gcs_time_hour(gcs, cohort)

    vital_map = {
        211: "hr",
        220045: "hr",
        618: "rr",
        220210: "rr",
        646: "spo2",
        220277: "spo2",
        51: "sbp",
        455: "sbp",
        220050: "sbp",
        220179: "sbp",
        8368: "dbp",
        8440: "dbp",
        8441: "dbp",
        220051: "dbp",
        220180: "dbp",
        52: "mbp",
        456: "mbp",
        220052: "mbp",
        220181: "mbp",
        220739: "gcs_eye",
        223900: "gcs_verbal",
        223901: "gcs_motor",
        228096: "rass",
        223835: "fio2",
    }
    vitals = build_vitals_long(chart, vital_map)
    fio2 = vitals["channel"].eq("fio2")
    vitals.loc[fio2, "value"] = vitals.loc[fio2, "value"].where(
        vitals.loc[fio2, "value"].between(0.21, 1.0)
    ).fillna(
        vitals.loc[fio2, "value"].where(
            vitals.loc[fio2, "value"].between(21.0, 100.0)
        )
        / 100.0
    )
    temp_c = chart[chart["itemid"].isin([676, 677, 223762])].copy()
    temp_c["channel"] = "temp"
    temp_c["value"] = pd.to_numeric(temp_c["valuenum"], errors="coerce")
    temp_c["hour"] = np.floor(temp_c["time_hour"]).astype("Int64")
    temp_f = chart[chart["itemid"].isin([678, 679, 223761])].copy()
    temp_f["channel"] = "temp"
    temp_f["value"] = (
        pd.to_numeric(temp_f["valuenum"], errors="coerce") - 32.0
    ) * 5.0 / 9.0
    temp_f["hour"] = np.floor(temp_f["time_hour"]).astype("Int64")
    temp = pd.concat([temp_c, temp_f], ignore_index=True)
    temp = temp[temp["hour"].between(0, 71)].dropna(subset=["value", "hour"])
    temp = (
        temp.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    urine = con.execute(
        """
        SELECT o.icustay_id AS stay_id, o.charttime,
               date_diff(
                   'second', c.intime, CAST(o.charttime AS TIMESTAMP)
               ) / 3600.0 AS time_hour,
               o.value
        FROM outputevents o
        JOIN cohort_selected c ON o.icustay_id = c.stay_id
        WHERE TRY_CAST(o.itemid AS INTEGER) IN (
            40055, 40056, 40057, 40061, 40065, 40069, 40085,
            40094, 40096, 40288, 40405, 40428, 40473, 40651,
            40715, 41857, 42001, 42042, 42111, 42119, 42130,
            42366, 42463, 42507, 42510, 42556, 42592, 42666,
            42676, 42700, 42765, 42810, 42859, 42892, 43053,
            43057, 43093, 43462, 43519, 43537, 43931, 43966,
            43987, 44051, 44080, 44103, 44132, 44237, 44253,
            44278, 44313, 44325, 44506, 44676, 44684, 44706,
            44752, 44824, 44834, 44837, 44911, 44925, 45415,
            45804, 45841, 45927, 45991, 46177, 46180, 46532,
            46578, 46658, 46727, 46732, 46748, 46804,
            226557, 226558, 226559, 226560, 226561, 226563,
            226564, 226565, 226567, 226584, 226627, 226631
        )
          AND CAST(o.charttime AS TIMESTAMP) >= c.intime
          AND CAST(o.charttime AS TIMESTAMP) < c.intime + INTERVAL 72 HOUR
          AND o.value IS NOT NULL
        """
    ).fetchdf()
    urine.columns = [column.lower() for column in urine.columns]
    urine["value"] = pd.to_numeric(urine["value"], errors="coerce")
    urine["hour"] = np.floor(
        pd.to_numeric(urine["time_hour"], errors="coerce")
    ).astype("Int64")
    urine["channel"] = "urine_output"
    urine = urine[urine["hour"].between(0, 71)].dropna(subset=["value", "hour"])
    urine = (
        urine.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .sum()
        .sort_values(["stay_id", "hour", "channel"])
    )
    hourly = assemble_hourly(
        pd.concat([vitals, temp, urine], ignore_index=True),
        gcs,
    )
    hourly = clamp_hourly_values(hourly)

    labs = extract_mimic_labs(con, cohort, "mimic3")
    recorded_intubation = con.execute(
        """
        SELECT p.icustay_id AS stay_id, MIN(p.starttime) AS intubation_time
        FROM procedureevents p
        JOIN cohort_selected c
          ON p.icustay_id = c.stay_id
        WHERE p.itemid = 224385
          AND p.starttime >= c.intime + INTERVAL 24 HOUR
          AND p.starttime < c.intime + INTERVAL 72 HOUR
        GROUP BY p.icustay_id
        """
    ).fetchdf()
    cohort["recorded_intubation_24_72h"] = cohort["stay_id"].isin(
        recorded_intubation["stay_id"]
    ).astype(float)
    cohort = attach_mimic_contextual_flags(con, cohort, chart)
    rass = chart[chart["itemid"] == 228096].copy()
    rass["value"] = pd.to_numeric(rass["valuenum"], errors="coerce")
    rass_events = prepare_rass_events(rass)
    cohort = cohort.merge(
        summarize_deep_sedation(
            rass_events.rename(columns={"rass": "value"})
        ),
        on="stay_id",
        how="left",
    )
    cohort["any_deep_sedation_24h"] = pd.to_numeric(
        cohort["any_deep_sedation_24h"], errors="coerce"
    )
    cohort["min_rass_24h"] = pd.to_numeric(
        cohort["min_rass_24h"], errors="coerce"
    )
    cohort["any_rass_le_minus2_24h"] = pd.to_numeric(
        cohort["any_rass_le_minus2_24h"], errors="coerce"
    )
    cohort["deep_sedation_observed_24h"] = pd.to_numeric(
        cohort["deep_sedation_observed_24h"], errors="coerce"
    ).fillna(0.0)

    cohort = add_outcome(cohort, gcs, include_intubation=False)
    cohort = finish_cohort(cohort, hourly, labs, config, name=name)
    prefilter_n = len(pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv"))
    flow = make_flow(
        name,
        raw_n,
        cohort,
        prefilter_n,
        bool(config.get("primary_include_early_live_transfer", True)),
    )
    save_rass_events(name, rass_events)
    save_dataset(name, cohort, gcs, hourly, flow, config)


def extract_mimic_labs(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    item_to_lab = {
        int(item): lab for lab, items in LAB_ITEMS.items() for item in items
    }
    item_list = ",".join(str(x) for x in item_to_lab)
    frame = con.execute(
        f"""
        SELECT l.hadm_id, l.itemid, l.charttime,
               date_diff('second', c.intime, l.charttime) / 3600.0 AS time_hour,
               l.valuenum
        FROM labevents l
        JOIN cohort_selected c ON l.hadm_id = c.hadm_id
        WHERE l.itemid IN ({item_list})
          AND l.charttime >= c.intime
          AND l.charttime < c.intime + INTERVAL 24 HOUR
          AND l.valuenum IS NOT NULL
        """
    ).fetchdf()
    frame.columns = [column.lower() for column in frame.columns]
    frame["itemid"] = pd.to_numeric(frame["itemid"], errors="coerce")
    frame["lab"] = frame["itemid"].map(item_to_lab)
    frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    for lab, (low, high) in LAB_RANGES.items():
        mask = frame["lab"] == lab
        frame.loc[mask, "value"] = frame.loc[mask, "value"].where(
            frame.loc[mask, "value"].between(low, high)
        )
    frame = frame.dropna(subset=["value"]).copy()
    stay_map = cohort[["hadm_id", "stay_id"]].drop_duplicates()
    frame = frame.merge(stay_map, on="hadm_id", how="left")
    frame = frame.sort_values(["stay_id", "itemid", "charttime"])
    return lab_summary(frame[["stay_id", "lab", "value"]])


def build_eicu(config: dict) -> None:
    name = "eicu"
    con = connect_duckdb()
    for view, rel in {
        "diagnosis": "diagnosis.csv.gz",
        "patient": "patient.csv.gz",
        "nursecharting": "nurseCharting.csv.gz",
        "vitalperiodic": "vitalPeriodic.csv.gz",
        "vitalaperiodic": "vitalAperiodic.csv.gz",
        "lab": "lab.csv.gz",
        "apachepatientresult": "apachePatientResult.csv.gz",
        "apacheapsvar": "apacheApsVar.csv.gz",
    }.items():
        create_view(con, view, EICU_ROOT / rel)

    cohort = con.execute(
        """
        WITH diagnosis_flags AS (
            SELECT patientunitstayid,
                   MAX(
                       CASE WHEN regexp_matches(
                                      COALESCE(icd9code, ''),
                                      '(^|[, ])(431|432)([., ]|$)'
                                  )
                                  OR regexp_matches(
                                      COALESCE(icd9code, ''),
                                      '(^|[, ])I6[12](\\.|[, ]|$)'
                                  )
                            THEN 1 ELSE 0 END
                   ) AS has_ich
            FROM diagnosis
            GROUP BY patientunitstayid
        ),
        candidates AS (
            SELECT
                p.patientunitstayid AS stay_id,
                p.uniquepid AS subject_id,
                p.hospitalid,
                p.unitvisitnumber,
                p.gender,
                p.age AS age_raw,
                p.ethnicity AS race,
                p.unitadmittime24 AS intime,
                p.unitdischargeoffset / 60.0 AS icu_out_hour,
                p.unitdischargeoffset,
                p.unitdischargestatus,
                p.hospitaldischargestatus,
                p.hospitaldischargeoffset,
                p.hospitaladmitoffset,
                COALESCE(d.has_ich, 0) AS has_ich,
                CASE
                    WHEN lower(COALESCE(p.unitdischargestatus, '')) = 'expired'
                         AND p.unitdischargeoffset <= 4320
                    THEN TRUE
                    WHEN lower(COALESCE(p.hospitaldischargestatus, '')) = 'expired'
                         AND p.hospitaldischargeoffset <= 4320
                    THEN TRUE
                    ELSE FALSE
                END AS death_within_72h
            FROM patient p
            LEFT JOIN diagnosis_flags d
              ON p.patientunitstayid = d.patientunitstayid
        ),
        ranked AS (
            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY subject_id
                        ORDER BY unitvisitnumber, intime, stay_id
                    ) AS patient_sequence
                FROM candidates
            )
            WHERE patient_sequence = 1
        ),
        eligible AS (
            SELECT *
            FROM ranked
            WHERE patient_sequence = 1
              AND has_ich = 1
              AND TRY_CAST(
                      TRIM(REPLACE(TRIM(age_raw), '>', '')) AS DOUBLE
                  ) >= 18
        )
        SELECT *
        FROM eligible
        """
    ).fetchdf()
    cohort.columns = [column.lower() for column in cohort.columns]
    cohort = harmonize_eicu_death(cohort)
    cohort["age"] = cohort["age_raw"].map(parse_eicu_age)
    cohort["age_gt89"] = pd.to_numeric(
        cohort["age"], errors="coerce"
    ).ge(90).astype(float)
    cohort = cohort[pd.to_numeric(cohort["age"], errors="coerce") >= 18].copy()
    cohort["dataset"] = name
    cohort["subject_key"] = "eicu_" + cohort["subject_id"].astype(str)
    cohort["hospital_key"] = "eicu_" + cohort["hospitalid"].astype(str)
    cohort["sex_male"] = (cohort["gender"].astype(str).str.lower() == "male").astype(float)
    cohort["race_white"] = (
        cohort["race"].astype(str).str.lower().eq("caucasian")
    ).astype(float)
    cohort["emergency_admission"] = np.nan
    con.register("cohort_selected", cohort[["stay_id"]].copy())
    raw_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT p.patientunitstayid)
            FROM patient p
            WHERE EXISTS (
                    SELECT 1
                    FROM diagnosis d
                    WHERE d.patientunitstayid = p.patientunitstayid
                      AND (
                          regexp_matches(
                              COALESCE(d.icd9code, ''),
                              '(^|[, ])(431|432)([., ]|$)'
                          )
                          OR regexp_matches(
                              COALESCE(d.icd9code, ''),
                              '(^|[, ])I6[12](\\.|[, ]|$)'
                          )
                      )
                )
            """
        ).fetchone()[0]
    )

    ids = ",".join(str(int(x)) for x in cohort["stay_id"])
    vital = con.execute(
        f"""
        SELECT patientunitstayid AS stay_id, observationoffset / 60.0 AS time_hour,
               heartrate, respiration, sao2, systemicsystolic,
               systemicdiastolic, systemicmean, temperature
        FROM vitalperiodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset >= 0 AND observationoffset < 4320
        """
    ).fetchdf()
    vital_long = vital.melt(
        id_vars=["stay_id", "time_hour"],
        value_vars=[
            "heartrate",
            "respiration",
            "sao2",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
            "temperature",
        ],
        var_name="channel",
        value_name="value",
    )
    vital_long["channel"] = vital_long["channel"].map(
        {
            "heartrate": "hr",
            "respiration": "rr",
            "sao2": "spo2",
            "systemicsystolic": "sbp",
            "systemicdiastolic": "dbp",
            "systemicmean": "mbp",
            "temperature": "temp",
        }
    )
    vital_long["value"] = pd.to_numeric(vital_long["value"], errors="coerce")
    vital_long["hour"] = np.floor(vital_long["time_hour"]).astype("Int64")
    vital_long = vital_long[vital_long["hour"].between(0, 71)].dropna(
        subset=["value", "hour"]
    )
    vitals = (
        vital_long.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    aperiodic = con.execute(
        f"""
        SELECT patientunitstayid AS stay_id, observationoffset / 60.0 AS time_hour,
               noninvasivesystolic, noninvasivediastolic, noninvasivemean
        FROM vitalaperiodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset >= 0 AND observationoffset < 4320
        """
    ).fetchdf()
    aperiodic_long = aperiodic.melt(
        id_vars=["stay_id", "time_hour"],
        value_vars=["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"],
        var_name="channel",
        value_name="value",
    )
    aperiodic_long["channel"] = aperiodic_long["channel"].map(
        {
            "noninvasivesystolic": "sbp",
            "noninvasivediastolic": "dbp",
            "noninvasivemean": "mbp",
        }
    )
    aperiodic_long["value"] = pd.to_numeric(
        aperiodic_long["value"], errors="coerce"
    )
    aperiodic_long["hour"] = np.floor(aperiodic_long["time_hour"]).astype("Int64")
    aperiodic_long = aperiodic_long[aperiodic_long["hour"].between(0, 71)].dropna(
        subset=["value", "hour"]
    )
    aperiodic_hourly = (
        aperiodic_long.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    vitals = pd.concat([vitals, aperiodic_hourly], ignore_index=True)

    nurse = con.execute(
        f"""
        SELECT patientunitstayid AS stay_id,
               nursingchartoffset / 60.0 AS time_hour,
               nursingchartoffset,
               nursingchartcelltypevallabel AS label,
               nursingchartcelltypevalname AS item,
               nursingchartvalue,
               TRY_CAST(TRIM(nursingchartvalue) AS DOUBLE) AS value
        FROM nursecharting
        WHERE patientunitstayid IN ({ids})
          AND nursingchartoffset >= 0 AND nursingchartoffset < 4320
          AND (
              lower(nursingchartcelltypevalname) IN (
                  'gcs total', 'eyes', 'motor', 'verbal'
              )
              OR (
                  lower(nursingchartcelltypevallabel) IN ('rass', 'sedation score')
                  AND lower(nursingchartcelltypevalname) = 'value'
              )
              OR (
                  lower(nursingchartcelltypevallabel)
                      = 'sedation scale/score/goal'
                  AND lower(nursingchartcelltypevalname)
                      IN ('sedation score', 'sedation scale')
              )
              OR lower(nursingchartcelltypevalname) IN (
                  'temperature (c)', 'temperature (f)'
              )
          )
        """
    ).fetchdf()
    gcs = nurse[
        nurse["item"].str.lower().eq("gcs total")
        & nurse["label"].str.lower().eq("glasgow coma score")
    ].copy()
    gcs["value"] = gcs["value"].where(gcs["value"].between(3, 15))
    gcs = gcs.dropna(subset=["value"]).rename(columns={"value": "gcs"})
    gcs["ett_verbal"] = 0.0
    component_rows = nurse[
        nurse["label"].str.lower().eq("glasgow coma score")
        & nurse["item"].str.lower().isin(["eyes", "motor", "verbal"])
    ].copy()
    component_rows["channel"] = component_rows["item"].str.lower().map(
        {
            "eyes": "gcs_eye",
            "verbal": "gcs_verbal",
            "motor": "gcs_motor",
        }
    )
    component_rows["hour"] = np.floor(
        pd.to_numeric(component_rows["time_hour"], errors="coerce")
    ).astype("Int64")
    component_rows = component_rows[
        component_rows["hour"].between(0, 71)
    ].dropna(subset=["value", "hour"])
    component_hourly = (
        component_rows.groupby(
            ["stay_id", "hour", "channel"], as_index=False
        )["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    rass_events = eicu_rass_events_from_nurse(nurse)
    rass_hourly = rass_events.rename(columns={"rass": "value"}).copy()
    rass_hourly["channel"] = "rass"
    rass_hourly["hour"] = np.floor(
        pd.to_numeric(rass_hourly["time_hour"], errors="coerce")
    ).astype("Int64")
    rass_hourly = rass_hourly[
        rass_hourly["hour"].between(0, 71)
    ].dropna(subset=["value", "hour"])
    rass_hourly = (
        rass_hourly.groupby(
            ["stay_id", "hour", "channel"], as_index=False
        )["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    cohort = cohort.merge(
        summarize_deep_sedation(
            rass_events.rename(columns={"rass": "value"})
        ),
        on="stay_id",
        how="left",
    )
    cohort["any_deep_sedation_24h"] = pd.to_numeric(
        cohort["any_deep_sedation_24h"], errors="coerce"
    )
    cohort["min_rass_24h"] = pd.to_numeric(
        cohort["min_rass_24h"], errors="coerce"
    )
    cohort["any_rass_le_minus2_24h"] = pd.to_numeric(
        cohort["any_rass_le_minus2_24h"], errors="coerce"
    )
    cohort["deep_sedation_observed_24h"] = pd.to_numeric(
        cohort["deep_sedation_observed_24h"], errors="coerce"
    ).fillna(0.0)
    temp_c = nurse[nurse["item"].str.lower() == "temperature (c)"].copy()
    temp_f = nurse[nurse["item"].str.lower() == "temperature (f)"].copy()
    temp_f["value"] = (pd.to_numeric(temp_f["value"], errors="coerce") - 32.0) * 5.0 / 9.0
    temp = pd.concat([temp_c, temp_f], ignore_index=True)
    temp["channel"] = "temp"
    temp["hour"] = np.floor(temp["time_hour"]).astype("Int64")
    temp = temp[temp["hour"].between(0, 71)].dropna(subset=["value", "hour"])
    temp_hourly = (
        temp.groupby(["stay_id", "hour", "channel"], as_index=False)["value"]
        .median()
        .sort_values(["stay_id", "hour", "channel"])
    )
    vitals = pd.concat([vitals, temp_hourly], ignore_index=True)
    hourly = assemble_hourly(
        pd.concat([vitals, component_hourly, rass_hourly], ignore_index=True),
        gcs,
    )
    hourly = clamp_hourly_values(hourly)
    labs = extract_eicu_labs(con, ids)
    cohort["recorded_intubation_24_72h"] = np.nan
    cohort = attach_eicu_contextual_flags(con, cohort)

    cohort = add_outcome(cohort, gcs, include_intubation=False)
    cohort = finish_cohort(cohort, hourly, labs, config, name=name)
    prefilter_n = len(pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv"))
    flow = make_flow(
        name,
        raw_n,
        cohort,
        prefilter_n,
        bool(config.get("primary_include_early_live_transfer", True)),
    )
    save_rass_events(name, rass_events)
    save_dataset(name, cohort, gcs, hourly, flow, config)


def parse_eicu_age(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text.startswith(">"):
        return 91.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def extract_eicu_labs(
    con: duckdb.DuckDBPyConnection,
    stay_ids: str,
) -> pd.DataFrame:
    names = [name for values in EICU_LABS.values() for name in values]
    escaped = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    frame = con.execute(
        f"""
        SELECT patientunitstayid AS stay_id, labname, labresultoffset / 60.0 AS time_hour,
               TRY_CAST(labresult AS DOUBLE) AS value
        FROM lab
        WHERE patientunitstayid IN ({stay_ids})
          AND labresultoffset >= 0 AND labresultoffset < 1440
          AND labname IN ({escaped})
        """
    ).fetchdf()
    name_map = {raw: lab for lab, values in EICU_LABS.items() for raw in values}
    frame["lab"] = frame["labname"].map(name_map)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for lab, (low, high) in LAB_RANGES.items():
        mask = frame["lab"] == lab
        frame.loc[mask, "value"] = frame.loc[mask, "value"].where(
            frame.loc[mask, "value"].between(low, high)
        )
    frame = frame.dropna(subset=["lab", "value"]).sort_values(
        ["stay_id", "lab", "time_hour"]
    )
    return lab_summary(frame[["stay_id", "lab", "value"]])


def finish_cohort(
    cohort: pd.DataFrame,
    hourly: pd.DataFrame,
    labs: pd.DataFrame,
    config: dict,
    name: str,
) -> pd.DataFrame:
    """Attach sequence and laboratory summaries and enforce primary follow-up."""
    cohort = cohort.drop_duplicates("stay_id").copy()
    legacy_columns = [
        column
        for column in cohort.columns
        if column.startswith("outcome_with_")
        or (
            column.endswith("_intubation_24_72h")
            and column != "recorded_intubation_24_72h"
        )
    ]
    cohort = cohort.drop(columns=legacy_columns, errors="ignore")
    values, masks, ids = sequence_from_hourly(hourly, cohort["stay_id"], config)
    summaries = sequence_summary(values, masks, ids, config["channels"])
    cohort = cohort.merge(summaries, on="stay_id", how="left")
    if not labs.empty:
        cohort = cohort.merge(labs, on="stay_id", how="left")
    gcs_count = cohort.get("gcs_total_n")
    if gcs_count is None:
        gcs_count = pd.Series(0.0, index=cohort.index)
    cohort["monitoring_intensity_24h"] = (
        pd.to_numeric(gcs_count, errors="coerce").fillna(0.0) / 24.0
    )
    if "recorded_intubation_24_72h" not in cohort:
        cohort["recorded_intubation_24_72h"] = np.nan
    recorded_intubation = pd.to_numeric(
        cohort["recorded_intubation_24_72h"], errors="coerce"
    )
    cohort["outcome_recorded_intubation_composite"] = (
        recorded_intubation_sensitivity_proxy(
            cohort["outcome"],
            recorded_intubation,
        )
    )
    cohort["primary_outcome_includes_intubation"] = False
    cohort["p020_status"] = P020_STATUS
    cohort["p020_complete"] = P020_COMPLETE
    cohort["recorded_intubation_composite_role"] = "sensitivity_proxy_only"
    cohort["recorded_intubation_composite_status"] = np.where(
        recorded_intubation.notna(),
        "ordinary_recorded_intubation_available",
        "unavailable_not_verified_unplanned",
    )
    cohort["recorded_intubation_composite_note"] = (
        "Ordinary recorded intubation; planned versus unplanned cannot be "
        "verified. This is not a P020-complete primary endpoint."
    )
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    death_hour = pd.to_numeric(
        cohort.get(
            "death_hour_from_icu",
            pd.Series(np.nan, index=cohort.index, dtype=float),
        ),
        errors="coerce",
    )
    icu_out_hour = pd.to_numeric(
        cohort.get(
            "icu_out_hour",
            pd.Series(np.nan, index=cohort.index, dtype=float),
        ),
        errors="coerce",
    )
    alive_at_primary_landmark = (
        ~cohort["death_within_72h"].astype(bool) | death_hour.gt(24)
    )
    in_icu_at_primary_landmark = icu_out_hour.ge(24)
    cohort["alive_at_primary_landmark"] = alive_at_primary_landmark
    cohort["in_icu_at_primary_landmark"] = in_icu_at_primary_landmark
    cohort["eligible_at_primary_landmark"] = (
        in_icu_at_primary_landmark & alive_at_primary_landmark
    )
    cohort["death_in_icu_within_72h"] = death_in_icu_within_window(
        cohort["death_within_72h"],
        death_hour,
        icu_out_hour,
    )
    cohort["complete_followup_only"] = (
        cohort["eligible_at_primary_landmark"]
        & complete_icu_followup(
            cohort["outcome"],
            icu_out_hour,
            cohort["death_in_icu_within_72h"],
        )
    )
    # Retained under the historical name because the feature-exclusion lists
    # in train_evaluate and make_manuscript refer to it.
    cohort["primary_followup_complete"] = cohort["complete_followup_only"]
    cohort["early_live_icu_transfer_before_72h"] = (
        cohort["eligible_at_primary_landmark"]
        & icu_out_hour.lt(72)
        & ~cohort["death_in_icu_within_72h"]
    )
    # The model is applied at ICU hour 24, so eligibility must be decidable at
    # that time. Requiring survival to hour 72 selects on future information.
    # A live ICU discharge before hour 72 is therefore retained as grade 0 when
    # the observed window yielded a classifiable outcome; the complete-follow-up
    # subset remains available as a prespecified sensitivity analysis.
    include_early_transfer = bool(
        config.get("primary_include_early_live_transfer", True)
    )
    cohort["primary_included"] = (
        cohort["eligible_at_primary_landmark"]
        & cohort["outcome"].notna()
        & (
            pd.Series(True, index=cohort.index)
            if include_early_transfer
            else cohort["complete_followup_only"]
        )
    )
    outdir = PROCESSED / name
    outdir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(outdir / "cohort_pre_filter.csv", index=False)
    cohort = cohort[cohort["primary_included"]].copy()
    cohort["outcome"] = cohort["outcome"].astype(int)
    for column in (
        "recorded_intubation_24_72h",
        "min_rass_24h",
        "any_rass_le_minus2_24h",
        "any_deep_sedation_24h",
        "deep_sedation_observed_24h",
    ):
        if column not in cohort:
            cohort[column] = np.nan
    return cohort.reset_index(drop=True)


def sequence_from_hourly(
    hourly: pd.DataFrame,
    stay_ids: pd.Series,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from .common import build_sequence_tensor

    return build_sequence_tensor(
        hourly,
        stay_ids,
        channels=config["channels"],
        max_hour=config["sequence_hours"],
    )


def make_flow(
    name: str,
    raw_n: int,
    included: pd.DataFrame,
    prefilter_n: int,
    include_early_transfer: bool = True,
) -> pd.DataFrame:
    prefilter = pd.read_csv(PROCESSED / name / "cohort_pre_filter.csv")
    if len(prefilter) != int(prefilter_n):
        raise ValueError(
            f"{name}: prefilter_n={prefilter_n} does not match "
            f"cohort_pre_filter.csv rows={len(prefilter)}"
        )
    icu_out_hour = pd.to_numeric(prefilter["icu_out_hour"], errors="coerce")
    death_hour = pd.to_numeric(
        prefilter["death_hour_from_icu"], errors="coerce"
    )
    death = prefilter["death_within_72h"].fillna(False).astype(bool)
    in_icu = prefilter["in_icu_at_primary_landmark"].fillna(False).astype(bool)
    alive = prefilter["alive_at_primary_landmark"].fillna(False).astype(bool)
    baseline = pd.to_numeric(
        prefilter["baseline_gcs_max"], errors="coerce"
    ).notna()
    outcome = prefilter["outcome"].notna()
    death_in_icu = prefilter["death_in_icu_within_72h"].fillna(False).astype(bool)
    followup = icu_out_hour.ge(72) | death_in_icu
    early_death = death & death_hour.le(24)
    not_in_icu = ~in_icu
    not_alive = ~alive
    missing_baseline = ~baseline
    if include_early_transfer:
        # Early live transfers with a classifiable outcome are retained under
        # the grade observed in the available post-landmark window; only those
        # without an observed outcome are excluded.
        early_transfer = icu_out_hour.lt(72) & ~death_in_icu & ~outcome
        incomplete = ~outcome
        early_transfer_label = (
            "Excluded: live ICU discharge before 72 hours without a "
            "classifiable outcome"
        )
    else:
        early_transfer = icu_out_hour.lt(72) & ~death_in_icu
        incomplete = ~(outcome & followup)
        early_transfer_label = "Excluded: live ICU discharge before 72 hours"

    # Assign one reason per stay in a fixed clinical priority. Every later
    # mask is restricted to stays still pending, so Figure 1 cannot double-count
    # a patient across exclusion categories.
    pending = pd.Series(True, index=prefilter.index)

    def assign_reason(mask: pd.Series, code: str) -> int:
        nonlocal pending
        selected = pending & mask.fillna(False).astype(bool)
        pending = pending & ~selected
        return int(selected.sum())

    early_death_n = assign_reason(
        early_death, "early_death_at_or_before_24h"
    )
    not_in_icu_n = assign_reason(not_in_icu, "not_in_icu_at_24h")
    not_alive_n = assign_reason(not_alive, "not_alive_at_24h")
    missing_baseline_n = assign_reason(
        missing_baseline, "missing_pre_landmark_gcs"
    )
    early_transfer_n = assign_reason(
        early_transfer, "live_icu_discharge_before_72h"
    )
    incomplete_n = assign_reason(
        incomplete, "insufficient_post_landmark_gcs_or_unresolved_outcome"
    )
    exclusion_n = (
        early_death_n
        + not_in_icu_n
        + not_alive_n
        + missing_baseline_n
        + early_transfer_n
        + incomplete_n
    )
    expected_exclusion_n = int(prefilter_n) - len(included)
    if exclusion_n != expected_exclusion_n:
        raise ValueError(
            f"{name}: mutually exclusive exclusions sum to {exclusion_n}, "
            f"but source minus primary is {expected_exclusion_n}"
        )
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "step": "ICH ICU stays",
                "n": raw_n,
                "reason_code": "source_ich_icu_stays",
            },
            {
                "dataset": name,
                "step": (
                    "First ICH ICU stay per patient and adult "
                    "(common landmark candidate source)"
                ),
                "n": prefilter_n,
                "reason_code": "source_common_landmark_candidates",
            },
            {
                "dataset": name,
                "step": "Excluded: death at or before the 24-hour landmark",
                "n": early_death_n,
                "reason_code": "early_death_at_or_before_24h",
            },
            {
                "dataset": name,
                "step": "Excluded: not in ICU at the 24-hour landmark",
                "n": not_in_icu_n,
                "reason_code": "not_in_icu_at_24h",
            },
            {
                "dataset": name,
                "step": "Excluded: not alive at the 24-hour landmark",
                "n": not_alive_n,
                "reason_code": "not_alive_at_24h",
            },
            {
                "dataset": name,
                "step": "Excluded: no valid pre-landmark GCS",
                "n": missing_baseline_n,
                "reason_code": "missing_pre_landmark_gcs",
            },
            {
                "dataset": name,
                "step": early_transfer_label,
                "n": early_transfer_n,
                "reason_code": "live_icu_discharge_before_72h",
            },
            {
                "dataset": name,
                "step": (
                    "Excluded: in-ICU follow-up complete but fewer than two "
                    "post-landmark GCS values or unresolved outcome"
                ),
                "n": incomplete_n,
                "reason_code": (
                    "insufficient_post_landmark_gcs_or_unresolved_outcome"
                ),
            },
            {
                "dataset": name,
                "step": (
                    "Primary analytic cohort (observed to hour 72, died in "
                    "ICU by hour 72, or live ICU discharge with a classifiable "
                    "outcome)"
                ),
                "n": len(included),
                "reason_code": "primary_analytic_cohort",
            },
            {
                "dataset": name,
                "step": "Outcome 0",
                "n": int((included["outcome"] == 0).sum()),
                "reason_code": "outcome_0",
            },
            {
                "dataset": name,
                "step": "Outcome 1",
                "n": int((included["outcome"] == 1).sum()),
                "reason_code": "outcome_1",
            },
            {
                "dataset": name,
                "step": "Outcome 2",
                "n": int((included["outcome"] == 2).sum()),
                "reason_code": "outcome_2",
            },
        ]
    )


def load_rass_export_anchors(dataset: str) -> pd.DataFrame:
    """Load the prefilter stays used to restrict a lightweight RASS export."""
    candidates = (
        PROCESSED / dataset / "cohort_pre_filter.csv",
        PROCESSED / dataset / "cohort.csv",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No cohort file is available to anchor the {dataset} RASS export."
        )
    frame = pd.read_csv(path)
    if "stay_id" not in frame:
        raise ValueError(f"{path} has no stay_id column.")
    columns = ["stay_id"]
    if "intime" in frame:
        columns.append("intime")
    anchors = frame[columns].copy()
    anchors["stay_id"] = pd.to_numeric(anchors["stay_id"], errors="coerce")
    anchors = anchors.dropna(subset=["stay_id"]).copy()
    anchors["stay_id"] = anchors["stay_id"].astype(np.int64)
    return anchors.drop_duplicates("stay_id").reset_index(drop=True)


def extract_rass_events_for_export(dataset: str) -> pd.DataFrame:
    """Query event-level RASS only, using the already selected prefilter stays."""
    anchors = load_rass_export_anchors(dataset)
    con = connect_duckdb()
    if dataset in {"mimic4", "mimic3"}:
        if "intime" not in anchors:
            raise ValueError(f"{dataset} cohort anchors require intime.")
        anchors["intime"] = pd.to_datetime(anchors["intime"], errors="coerce")
        con.register("rass_export_anchors", anchors)
        if dataset == "mimic4":
            create_view(
                con,
                "rass_chartevents",
                MIMIC_IV_ROOT / "icu/chartevents.csv.gz",
            )
            events = con.execute(
                """
                SELECT
                    ce.stay_id,
                    date_diff(
                        'second', c.intime, ce.charttime
                    ) / 3600.0 AS time_hour,
                    ce.valuenum AS value
                FROM rass_chartevents ce
                JOIN rass_export_anchors c ON ce.stay_id = c.stay_id
                WHERE ce.itemid = 228096
                  AND ce.charttime >= c.intime
                  AND ce.charttime <= c.intime + INTERVAL 72 HOUR
                """
            ).fetchdf()
        else:
            create_view(
                con,
                "rass_chartevents",
                MIMIC_III_ROOT / "CHARTEVENTS.csv.gz",
                all_varchar=True,
            )
            events = con.execute(
                """
                SELECT
                    TRY_CAST(ce.icustay_id AS BIGINT) AS stay_id,
                    date_diff(
                        'second',
                        c.intime,
                        TRY_CAST(ce.charttime AS TIMESTAMP)
                    ) / 3600.0 AS time_hour,
                    TRY_CAST(ce.valuenum AS DOUBLE) AS value
                FROM rass_chartevents ce
                JOIN rass_export_anchors c
                  ON TRY_CAST(ce.icustay_id AS BIGINT) = c.stay_id
                WHERE TRY_CAST(ce.itemid AS INTEGER) = 228096
                  AND TRY_CAST(ce.charttime AS TIMESTAMP) >= c.intime
                  AND TRY_CAST(ce.charttime AS TIMESTAMP)
                      <= c.intime + INTERVAL 72 HOUR
                """
            ).fetchdf()
    elif dataset == "eicu":
        create_view(
            con,
            "rass_nursecharting",
            EICU_ROOT / "nurseCharting.csv.gz",
        )
        con.register("rass_export_anchors", anchors[["stay_id"]])
        events = con.execute(
            """
            SELECT
                n.patientunitstayid AS stay_id,
                n.nursingchartoffset / 60.0 AS time_hour,
                TRY_CAST(TRIM(n.nursingchartvalue) AS DOUBLE) AS value
            FROM rass_nursecharting n
            JOIN rass_export_anchors c
              ON n.patientunitstayid = c.stay_id
            WHERE n.nursingchartoffset >= 0
              AND n.nursingchartoffset <= 4320
              AND lower(n.nursingchartcelltypevallabel)
                  IN ('rass', 'sedation score')
              AND lower(n.nursingchartcelltypevalname) = 'value'
            """
        ).fetchdf()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    con.close()
    return prepare_rass_events(events)


def export_rass_events_only(dataset: str) -> list[Path]:
    """Export RASS for one database without rebuilding cohorts or sequences."""
    datasets = DATASET_NAMES if dataset == "all" else (dataset,)
    paths = []
    for name in datasets:
        events = extract_rass_events_for_export(name)
        paths.append(save_rass_events(name, events))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["mimic4", "mimic3", "eicu", "all"], default="all"
    )
    parser.add_argument(
        "--export-rass-only",
        action="store_true",
        help="Export event-level RASS only; do not rebuild cohorts or sequences.",
    )
    args = parser.parse_args()
    ensure_dirs()
    if args.export_rass_only:
        export_rass_events_only(args.dataset)
        return
    config = load_config()
    if args.dataset in ("mimic4", "all"):
        build_mimic4(config)
    if args.dataset in ("mimic3", "all"):
        build_mimic3(config)
    if args.dataset in ("eicu", "all"):
        build_eicu(config)


if __name__ == "__main__":
    main()
