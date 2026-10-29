"""Audit data and code against the frozen study outline.

This script is read-only with respect to study inputs. It produces a JSON
evidence record for the strict outline-compliance review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ich_dynamic import sensitivity_analysis as sensitivity
from src.ich_dynamic.common import (
    PROCESSED,
    RESULTS,
    add_outcome,
    death_in_icu_within_window,
    load_config,
)
from src.ich_dynamic.train_evaluate import (
    DEEP_STATIC,
    MODEL_META,
    SEQUENCE_DEFAULTS,
    STATIC_FEATURES,
    XGBOOST_DEFAULTS,
    Dataset,
    configured_candidates,
    load_dataset,
    model_feature_contract,
    tabular_frame,
)


DATASETS = ("mimic4", "mimic3", "eicu")
LANDMARKS = (6, 24, 48)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outcome_counts(frame: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(frame["outcome"], errors="coerce")
    counts = values.value_counts(dropna=False).sort_index()
    return {
        "n": int(len(frame)),
        "grade_counts": {
            str(key): int(value) for key, value in counts.items()
        },
        "any_event_n": int((values > 0).sum()),
        "any_event_rate": float((values > 0).mean()),
        "severe_n": int((values == 2).sum()),
        "severe_rate": float((values == 2).mean()),
    }


def _sequence_coverage(dataset: str, channels: list[str]) -> dict:
    with np.load(
        PROCESSED / dataset / "sequences.npz",
        allow_pickle=True,
    ) as arrays:
        values = arrays["values"]
        masks = arrays["masks"]
        stored_channels = [str(value) for value in arrays["channels"]]
    observed = masks > 0
    stay_coverage = (observed.sum(axis=1) > 0).mean(axis=0)
    hour_coverage = observed.mean(axis=(0, 1))
    return {
        "shape": list(values.shape),
        "channels_match_config": stored_channels == channels,
        "finite_values_where_observed": bool(
            np.isfinite(values[observed]).all()
        ),
        "stay_observation_proportion": {
            channel: float(value)
            for channel, value in zip(channels, stay_coverage, strict=True)
        },
        "hour_cell_observation_proportion": {
            channel: float(value)
            for channel, value in zip(channels, hour_coverage, strict=True)
        },
        "all_missing_channels": [
            channel
            for channel, value in zip(channels, stay_coverage, strict=True)
            if value <= 0.0
        ],
    }


def _feature_contract(dataset: Dataset) -> dict:
    all_columns = list(dataset.static_all.columns)
    static_columns = list(tabular_frame(dataset, "static").columns)
    summary_columns = list(tabular_frame(dataset, "all").columns)
    dynamic_prefixes = tuple(
        f"{channel}_" for channel in load_config()["channels"]
    )
    dynamic_columns = [
        column
        for column in all_columns
        if column.startswith(dynamic_prefixes) or column == "monitoring_intensity_24h"
    ]
    deep_available = [
        column for column in DEEP_STATIC if column in all_columns
    ]
    by_model = model_feature_contract(dataset)
    return {
        "temporal_static_input_count": len(all_columns),
        "temporal_static_inputs": all_columns,
        "xgb_static_count": len(static_columns),
        "xgb_static_excludes_dynamic_summaries": not set(
            dynamic_columns
        ).intersection(static_columns),
        "xgb_summary_count": len(summary_columns),
        "xgb_summary_includes_dynamic_summaries": set(
            dynamic_columns
        ).issubset(summary_columns),
        "dynamic_summary_count": len(dynamic_columns),
        "dynamic_summary_columns": dynamic_columns,
        "temporal_models_receive_dynamic_summaries": bool(
            set(dynamic_columns).intersection(all_columns)
        ),
        "model_feature_contract": by_model,
        "matched_sequence_ablations": [
            name
            for name in (
                "dggru_tr_noseq",
                "dggru_tr_static",
                "dggru_tr_traj",
            )
            if name in by_model
        ],
        "deep_static_available_count": len(deep_available),
        "deep_static_available": deep_available,
        "outline_named_static_core": [
            column for column in STATIC_FEATURES if column in all_columns
        ],
    }


def _endpoint_variants(dataset: str) -> dict:
    prefilter = pd.read_csv(
        PROCESSED / dataset / "cohort_pre_filter.csv"
    )
    gcs = pd.read_csv(PROCESSED / dataset / "gcs_events.csv")
    variants: dict[str, dict] = {}
    for label, baseline_mode, persistence, include_intubation in (
        ("maximum_gcs_2h_no_intubation", "max", 2, False),
        ("last_gcs_2h_no_intubation", "last", 2, False),
        ("maximum_gcs_1h_no_intubation", "max", 1, False),
        ("maximum_gcs_2h_with_intubation", "max", 2, True),
    ):
        reconstructed = add_outcome(
            prefilter,
            gcs,
            baseline_mode=baseline_mode,
            confirmation_consecutive_hours=persistence,
            include_death=True,
            include_intubation=include_intubation,
        )
        eligible = reconstructed[
            reconstructed["primary_followup_complete"].fillna(False).astype(bool)
        ].copy()
        variants[label] = _outcome_counts(eligible)
    return {
        "variants": variants,
        "baseline_mode_any_event_rate_delta": (
            variants["last_gcs_2h_no_intubation"]["any_event_rate"]
            - variants["maximum_gcs_2h_no_intubation"]["any_event_rate"]
        ),
        "intubation_grade2_reclassification_n": (
            variants["maximum_gcs_2h_with_intubation"]["severe_n"]
            - variants["maximum_gcs_2h_no_intubation"]["severe_n"]
        ),
    }


def _cohort_evidence(dataset: str) -> dict:
    cohort_path = PROCESSED / dataset / "cohort.csv"
    prefilter_path = PROCESSED / dataset / "cohort_pre_filter.csv"
    flow_path = PROCESSED / dataset / "cohort_flow.csv"
    cohort = pd.read_csv(cohort_path)
    prefilter = pd.read_csv(prefilter_path)
    flow = pd.read_csv(flow_path)
    icu_out_hour = pd.to_numeric(
        prefilter["icu_out_hour"], errors="coerce"
    )
    death_hour = pd.to_numeric(
        prefilter["death_hour_from_icu"], errors="coerce"
    )
    death_in_icu = death_in_icu_within_window(
        prefilter["death_within_72h"],
        death_hour,
        icu_out_hour,
    )
    evidence = {
        "cohort": _outcome_counts(cohort),
        "prefilter_n": int(len(prefilter)),
        "primary_followup_complete_n": int(
            prefilter["primary_followup_complete"].fillna(False).sum()
        ),
        "alive_at_24h_n": int(
            prefilter["eligible_at_primary_landmark"].fillna(False).sum()
        ),
        "not_alive_at_24h_n": int(
            (~prefilter["eligible_at_primary_landmark"].fillna(False)).sum()
        ),
        "early_live_transfer_n": int(
            (
                prefilter["eligible_at_primary_landmark"].fillna(False)
                & icu_out_hour.lt(72)
                & ~death_in_icu
            ).sum()
        ),
        "death_in_icu_within_72h_n": int(death_in_icu.sum()),
        "death_within_72h_n": int(
            prefilter["death_within_72h"].fillna(False).sum()
        ),
        "flow": flow.to_dict(orient="records"),
        "cohort_sha256": _sha256(cohort_path),
        "prefilter_sha256": _sha256(prefilter_path),
    }
    if "age_raw" in cohort:
        raw_age = pd.to_numeric(cohort["age_raw"], errors="coerce")
        sentinel = raw_age >= 300
        evidence["age_sentinel"] = {
            "raw_age_ge_300_n": int(sentinel.sum()),
            "raw_age_ge_300_proportion": float(sentinel.mean()),
            "harmonized_age_values": sorted(
                pd.to_numeric(
                    cohort.loc[sentinel, "age"], errors="raise"
                ).unique().tolist()
            ),
            "age_gt89_n": int(
                pd.to_numeric(
                    cohort["age_gt89"], errors="coerce"
                ).eq(1).sum()
            ),
        }
    if "recorded_intubation_24_72h" in prefilter:
        recorded = pd.to_numeric(
            prefilter["recorded_intubation_24_72h"], errors="coerce"
        )
        evidence["recorded_intubation_source"] = {
            "nonmissing_n": int(recorded.notna().sum()),
            "positive_n": int(recorded.eq(1).sum()),
            "missing_n": int(recorded.isna().sum()),
            "verified_unplanned": False,
        }
    if "verified_unplanned_intubation_24_72h" in prefilter:
        verified = pd.to_numeric(
            prefilter["verified_unplanned_intubation_24_72h"],
            errors="coerce",
        )
        evidence["verified_unplanned_intubation_source"] = {
            "nonmissing_n": int(verified.notna().sum()),
            "positive_n": int(verified.eq(1).sum()),
            "missing_n": int(verified.isna().sum()),
        }
    if "outcome_recorded_intubation_composite" in prefilter:
        proxy = pd.to_numeric(
            prefilter["outcome_recorded_intubation_composite"],
            errors="coerce",
        )
        evidence["recorded_intubation_composite"] = {
            "role": "sensitivity_proxy_only",
            "nonmissing_n": int(proxy.notna().sum()),
            "grade_2_n": int(proxy.eq(2).sum()),
            "status": (
                sorted(
                    prefilter["recorded_intubation_composite_status"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if "recorded_intubation_composite_status" in prefilter
                else []
            ),
        }
    p020_status = (
        sorted(prefilter["p020_status"].dropna().astype(str).unique().tolist())
        if "p020_status" in prefilter
        else []
    )
    p020_complete = (
        bool(prefilter["p020_complete"].fillna(False).astype(bool).any())
        if "p020_complete" in prefilter
        else None
    )
    evidence["p020_endpoint_status"] = {
        "status_values": p020_status,
        "complete_any": p020_complete,
    }
    return evidence


def _sensitivity_support() -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for dataset in DATASETS:
        prefilter = pd.read_csv(
            PROCESSED / dataset / "cohort_pre_filter.csv"
        )
        gcs = pd.read_csv(PROCESSED / dataset / "gcs_events.csv")
        rass_path = PROCESSED / dataset / "rass_events.csv"
        rass = pd.read_csv(rass_path) if rass_path.is_file() else None
        records = []
        for spec in sensitivity.SCENARIOS:
            status = sensitivity.assess_scenario_support(
                spec,
                dataset,
                prefilter=prefilter,
                gcs=gcs,
                rass_events=rass,
            )
            records.append(
                {
                    "analysis_id": spec.analysis_id,
                    "scenario_id": spec.scenario_id,
                    "support": status.support,
                    "model_run_allowed": status.model_run_allowed,
                    "reason": status.reason,
                }
            )
        output[dataset] = records
    return output


def _landmark_support() -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for landmark in LANDMARKS:
            valid, message = sensitivity.validate_landmark_artifacts(
                dataset, landmark
            )
            output[dataset][f"{landmark}h"] = {
                "valid": bool(valid),
                "message": message,
            }
    return output


def _search_space(config: dict) -> dict:
    xgboost = configured_candidates(
        config,
        "xgboost",
        set(XGBOOST_DEFAULTS),
    )
    sequence = configured_candidates(
        config,
        "sequence_models",
        set(SEQUENCE_DEFAULTS),
    )
    return {
        "lambda_grid_count": len(config.get("lambda_grid", [])),
        "lambda_grid": config.get("lambda_grid", []),
        "xgboost_strategy": config["hyperparameters"]["xgboost"][
            "search_strategy"
        ],
        "xgboost_candidate_count": len(xgboost),
        "xgboost_candidates": list(xgboost),
        "sequence_strategy": config["hyperparameters"]["sequence_models"][
            "search_strategy"
        ],
        "sequence_candidate_count": len(sequence),
        "sequence_candidates": list(sequence),
        "selection_metric": "inner_patient_grouped_macro_auc",
        "outside_outer_training_outcomes_used_for_selection": False,
        "hyperparameter_search_implemented": (
            len(xgboost) > 1 and len(sequence) > 1
        ),
    }


def main() -> None:
    config = load_config()
    development = load_dataset("mimic4")
    report = {
        "outline_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "outline.docx"
        ),
        "config_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "configs" / "study.json"
        ),
        "config": {
            "channels": config["channels"],
            "channel_count": len(config["channels"]),
            "cv_folds": config["cv_folds"],
            "cv_repeats": config["cv_repeats"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "model_seeds": config["model_seeds"],
            "landmark_windows": config["landmark_windows"],
        },
        "models": {
            "registered": list(MODEL_META),
            "count": len(MODEL_META),
            "stacking_members": config["stacking_members"],
        },
        "search_space": _search_space(config),
        "feature_contract": _feature_contract(development),
        "datasets": {},
        "landmark_artifacts": _landmark_support(),
        "sensitivity_support": _sensitivity_support(),
    }
    for dataset in DATASETS:
        row = _cohort_evidence(dataset)
        row["sequence_coverage"] = _sequence_coverage(
            dataset, config["channels"]
        )
        row["endpoint_variants"] = _endpoint_variants(dataset)
        report["datasets"][dataset] = row
    output = RESULTS / "audits" / "outline_strict_contract_round1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
