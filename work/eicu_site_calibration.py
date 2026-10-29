"""Site-level calibration and calibration-update analysis for eICU.

All calibration parameters are estimated inside each site with repeated
stratified cross-validation.  A held-out fold is never used to estimate its
own calibration intercept, slope, or empirical-Bayes shrinkage parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "predictions" / "eicu_external_core.npz"
COHORT = ROOT / "data" / "processed" / "eicu" / "cohort.csv"
OUT_JSON = ROOT / "results" / "audits" / "eicu_site_calibration.json"
OUT_SUPPORT = ROOT / "results" / "tables" / "eicu_site_calibration_support.csv"
OUT_SITES = ROOT / "results" / "tables" / "eicu_site_calibration_sites.csv"
OUT_METHODS = ROOT / "results" / "tables" / "eicu_site_calibration_methods.csv"

MODELS = ("xgb_summary", "dggru_tr")
MIN_SITE_N = 30
MIN_EVENTS = 10
MIN_NONEVENTS = 10
N_FOLDS = 5
N_REPEATS = 20
ECE_BINS = 10
RIDGE = 1.0


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    columns: int


METHODS = (
    Method("uncalibrated", "No calibration", 0),
    Method("intercept_only", "Intercept update", 1),
    Method("slope_intercept", "Slope and intercept", 2),
    Method(
        "logistic_recalibration",
        "Flexible logistic recalibration",
        3,
    ),
    Method("empirical_bayes", "Empirical-Bayes shrinkage", 2),
)


def _clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probability(p)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _fit_logistic(
    y: np.ndarray,
    x: np.ndarray,
    *,
    design: np.ndarray | None = None,
    ridge: float = RIDGE,
) -> np.ndarray:
    """Fit a logistic model with a ridge penalty."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if design is None:
        design = np.column_stack([np.ones(len(y)), x])
    design = np.asarray(design, dtype=float)
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag(
        [1e-8] + [float(ridge)] * max(0, design.shape[1] - 1)
    )
    for _ in range(100):
        eta = design @ beta
        mu = _expit(eta)
        weight = np.clip(mu * (1.0 - mu), 1e-8, None)
        gradient = design.T @ (y - mu) - penalty @ beta
        hessian = design.T @ (design * weight[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _fit_with_offset(
    y: np.ndarray,
    offset: np.ndarray,
    *,
    cols: int,
    ridge: float = RIDGE,
) -> np.ndarray:
    """Fit intercept/slope terms while the original logit is an offset."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    design = np.column_stack(
        [np.ones(len(y))]
        + [offset ** power for power in range(1, cols)]
    )
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag(
        [1e-8] + [float(ridge)] * max(0, design.shape[1] - 1)
    )
    for _ in range(100):
        eta = offset + design @ beta
        mu = _expit(eta)
        weight = np.clip(mu * (1.0 - mu), 1e-8, None)
        gradient = design.T @ (y - mu) - penalty @ beta
        hessian = design.T @ (design * weight[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _apply_with_offset(
    offset: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    design = np.column_stack(
        [np.ones(len(offset))]
        + [offset ** power for power in range(1, len(beta))]
    )
    return _expit(offset + design @ beta)


def _calibration_slope_intercept(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:
    x = _logit(p)
    beta = _fit_logistic(
        y,
        x,
        design=np.column_stack([np.ones(len(y)), x]),
        ridge=1e-6,
    )
    return float(beta[1]), float(beta[0])


def _ece(
    y: np.ndarray,
    p: np.ndarray,
    *,
    bins: int = ECE_BINS,
) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip_probability(p)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    total = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        total += (
            len(chunk)
            / len(p)
            * abs(float(np.mean(p[chunk])) - float(np.mean(y[chunk])))
        )
    return float(total)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _method_predictions(
    model: str,
    method: Method,
    y_train: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if method.key == "uncalibrated":
        return _expit(x_test)
    if method.key == "empirical_bayes":
        raise ValueError("Empirical Bayes is fitted at the site-set level")
    beta = _fit_with_offset(
        y_train,
        x_train,
        cols=method.columns,
    )
    return _apply_with_offset(x_test, beta)


def _empirical_bayes_fit(
    site_data: dict[str, tuple[np.ndarray, np.ndarray]],
    method: Method,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    """Shrink site intercept/slope estimates toward a training-set global fit."""
    all_y = np.concatenate([value[0] for value in site_data.values()])
    all_x = np.concatenate([value[1] for value in site_data.values()])
    global_beta = _fit_with_offset(all_y, all_x, cols=2)
    site_betas: dict[str, np.ndarray] = {}
    site_variances: dict[str, np.ndarray] = {}
    for site, (y, x) in site_data.items():
        beta = _fit_with_offset(y, x, cols=2)
        design = np.column_stack([np.ones(len(y)), x])
        mu = _expit(x + design @ beta)
        weight = np.clip(mu * (1.0 - mu), 1e-8, None)
        penalty = np.diag([1e-8, RIDGE])
        hessian = design.T @ (design * weight[:, None]) + penalty
        try:
            covariance = np.linalg.inv(hessian)
            variance = np.clip(np.diag(covariance), 1e-8, None)
        except np.linalg.LinAlgError:
            variance = np.full(2, 1e6)
        site_betas[site] = beta
        site_variances[site] = variance
    stacked = np.vstack([site_betas[site] for site in site_data])
    mean_within = np.mean(
        np.vstack([site_variances[site] for site in site_data]),
        axis=0,
    )
    observed_between = np.var(stacked, axis=0, ddof=1)
    tau2 = np.clip(observed_between - mean_within, 1e-8, None)
    shrunk: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for site, beta in site_betas.items():
        weight = tau2 / (tau2 + site_variances[site])
        shrunk[site] = weight * beta + (1.0 - weight) * global_beta
        weights[site] = float(weight[1])
    return shrunk, global_beta, weights


def _summarize_pooled(
    y: np.ndarray,
    p: np.ndarray,
    sites: np.ndarray,
) -> dict[str, float]:
    slope, intercept = _calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "auroc": _safe_auc(y, p),
        "brier": float(brier_score_loss(y, _clip_probability(p))),
        "ece": _ece(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted": float(np.mean(p)),
        "observed": float(np.mean(y)),
    }


def _site_table(
    model: str,
    y: np.ndarray,
    p: np.ndarray,
    sites: np.ndarray,
) -> list[dict]:
    rows = []
    for site in sorted(np.unique(sites), key=lambda value: str(value)):
        mask = sites == site
        y_site = y[mask]
        p_site = p[mask]
        try:
            slope, intercept = _calibration_slope_intercept(y_site, p_site)
        except np.linalg.LinAlgError:
            slope = float("nan")
            intercept = float("nan")
        rows.append(
            {
                "model": model,
                "hospitalid": int(site),
                "n": int(mask.sum()),
                "events": int(y_site.sum()),
                "nonevents": int(mask.sum() - y_site.sum()),
                "auroc": _safe_auc(y_site, p_site),
                "ece": _ece(y_site, p_site),
                "brier": float(brier_score_loss(y_site, p_site)),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "mean_predicted": float(np.mean(p_site)),
                "observed": float(np.mean(y_site)),
            }
        )
    return rows


def _bootstrap_site_ci(
    y: np.ndarray,
    p: np.ndarray,
    sites: np.ndarray,
    *,
    repeats: int = 1000,
    seed: int = 20261025,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique_sites = np.unique(sites)
    values: dict[str, list[float]] = {
        key: [] for key in ("auroc", "brier", "ece", "calibration_slope")
    }
    for _ in range(repeats):
        sampled = rng.choice(unique_sites, size=len(unique_sites), replace=True)
        y_boot = np.concatenate([y[sites == site] for site in sampled])
        p_boot = np.concatenate([p[sites == site] for site in sampled])
        try:
            summary = _summarize_pooled(y_boot, p_boot, sites)
        except (np.linalg.LinAlgError, ValueError):
            continue
        for key in values:
            if np.isfinite(summary[key]):
                values[key].append(float(summary[key]))
    result = {}
    for key, sample in values.items():
        if not sample:
            result[key] = [float("nan"), float("nan")]
        else:
            result[key] = [
                float(np.quantile(sample, 0.025)),
                float(np.quantile(sample, 0.975)),
            ]
    return result


def _read_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cohort = pd.read_csv(COHORT, low_memory=False)
    required = {"stay_id", "hospitalid", "outcome"}
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"eICU cohort lacks columns: {sorted(missing)}")
    with np.load(PREDICTIONS, allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}
    for model in MODELS:
        if model not in arrays:
            raise ValueError(f"Prediction file lacks {model}")
    stay_id = np.asarray(arrays["stay_id"])
    cohort = cohort.set_index("stay_id").loc[stay_id].reset_index()
    if not np.array_equal(cohort["stay_id"].to_numpy(), stay_id):
        raise RuntimeError("Prediction and cohort stay_id order is not aligned")
    if not np.array_equal(
        cohort["outcome"].to_numpy(),
        np.asarray(arrays["labels"]),
    ):
        raise RuntimeError("Prediction labels do not match cohort outcome")
    return cohort, arrays


def _eligible_sites(
    frame: pd.DataFrame,
    *,
    endpoint: str,
) -> tuple[list[int], list[dict]]:
    rows = []
    if endpoint == "any_deterioration":
        frame = frame.assign(endpoint=(frame["outcome"] >= 1).astype(int))
    elif endpoint == "severe_deterioration":
        frame = frame.assign(endpoint=(frame["outcome"] == 2).astype(int))
    else:
        raise ValueError(endpoint)
    for site, group in frame.groupby("hospitalid", sort=True):
        n = len(group)
        events = int(group["endpoint"].sum())
        nonevents = n - events
        supported = (
            n >= MIN_SITE_N
            and events >= MIN_EVENTS
            and nonevents >= MIN_NONEVENTS
        )
        rows.append(
            {
                "endpoint": endpoint,
                "hospitalid": int(site),
                "n": n,
                "events": events,
                "nonevents": nonevents,
                "eligible": bool(supported),
                "reason": (
                    "eligible"
                    if supported
                    else (
                        f"requires n>={MIN_SITE_N}, events>={MIN_EVENTS}, "
                        f"nonevents>={MIN_NONEVENTS}"
                    )
                ),
            }
        )
    eligible = [row["hospitalid"] for row in rows if row["eligible"]]
    return eligible, rows


def _run_endpoint(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    endpoint: str,
) -> dict:
    if endpoint == "any_deterioration":
        y_all = (frame["outcome"].to_numpy() >= 1).astype(int)
    else:
        y_all = (frame["outcome"].to_numpy() == 2).astype(int)
    eligible, site_support = _eligible_sites(frame, endpoint=endpoint)
    result: dict = {
        "endpoint": endpoint,
        "site_support": site_support,
        "eligible_sites": eligible,
    }
    if len(eligible) < 2:
        result["status"] = "unsupported"
        result["reason"] = (
            "Fewer than two sites meet the prespecified sample-size and "
            "event-count thresholds."
        )
        return result
    mask = frame["hospitalid"].isin(eligible).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    sites = subset["hospitalid"].to_numpy()
    y = y_all[mask]
    result["status"] = "supported"
    result["included_n"] = int(mask.sum())
    result["included_events"] = int(y.sum())
    result["excluded_sites"] = int(len(site_support) - len(eligible))
    result["methods"] = {}
    result["models"] = {}
    for model in MODELS:
        probabilities = np.asarray(arrays[model], dtype=float)[mask]
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(
                f"{model} predictions must be a three-class probability matrix"
            )
        if endpoint == "any_deterioration":
            risk = probabilities[:, 1] + probabilities[:, 2]
        else:
            risk = probabilities[:, 2]
        x = _logit(risk)
        oof: dict[str, list[np.ndarray]] = {
            method.key: [] for method in METHODS
        }
        site_beta_rows: list[dict] = []
        for repeat in range(N_REPEATS):
            repeat_oof = {
                method.key: np.full(len(y), np.nan) for method in METHODS
            }
            fold_ids = np.full(len(y), -1, dtype=int)
            for site in eligible:
                site_index = np.flatnonzero(sites == site)
                splitter = StratifiedKFold(
                    n_splits=N_FOLDS,
                    shuffle=True,
                    random_state=20261025 + repeat,
                )
                for fold, (_, test_local) in enumerate(splitter.split(
                    np.zeros(len(site_index)),
                    y[site_index],
                )):
                    fold_ids[site_index[test_local]] = fold
            if np.any(fold_ids < 0):
                raise RuntimeError("Site fold assignment did not cover all patients")
            for fold in range(N_FOLDS):
                train_mask = fold_ids != fold
                test_mask = fold_ids == fold
                training_sites = {}
                for site in eligible:
                    site_train = (sites == site) & train_mask
                    if not np.any(site_train):
                        continue
                    if len(np.unique(y[site_train])) < 2:
                        continue
                    training_sites[int(site)] = (
                        y[site_train],
                        x[site_train],
                    )
                shrunk = None
                global_beta = None
                weights = None
                if len(training_sites) >= 2:
                    shrunk, global_beta, weights = _empirical_bayes_fit(
                        training_sites,
                        METHODS[-1],
                    )
                for site in eligible:
                    site_train = (sites == site) & train_mask
                    site_test = (sites == site) & test_mask
                    if not np.any(site_test):
                        continue
                    if len(np.unique(y[site_train])) < 2:
                        repeat_oof["empirical_bayes"][site_test] = (
                            _expit(x[site_test])
                        )
                        for method in METHODS[:4]:
                            repeat_oof[method.key][site_test] = (
                                _expit(x[site_test])
                            )
                        continue
                    for method in METHODS[:4]:
                        repeat_oof[method.key][site_test] = (
                            _method_predictions(
                                model,
                                method,
                                y[site_train],
                                x[site_train],
                                x[site_test],
                            )
                        )
                    if shrunk is None or int(site) not in shrunk:
                        repeat_oof["empirical_bayes"][site_test] = (
                            _expit(x[site_test])
                        )
                        continue
                    beta = shrunk[int(site)]
                    design = np.column_stack(
                        [np.ones(int(site_test.sum())), x[site_test]]
                    )
                    repeat_oof["empirical_bayes"][site_test] = _expit(
                        x[site_test] + design @ beta
                    )
                    if repeat == 0 and fold == 0:
                        site_beta_rows.append(
                            {
                                "model": model,
                                "hospitalid": int(site),
                                "shrink_weight_slope": weights[int(site)],
                                "site_intercept": float(beta[0]),
                                "site_slope": float(beta[1]),
                                "global_intercept": float(global_beta[0]),
                                "global_slope": float(global_beta[1]),
                            }
                        )
            for method in METHODS:
                values = repeat_oof[method.key]
                if np.isnan(values).any():
                    raise RuntimeError(
                        f"OOF calibration did not cover all patients: {method.key}"
                    )
                oof[method.key].append(values)
        method_results = {}
        oof_mean = {}
        for method in METHODS:
            stacked = np.vstack(oof[method.key])
            mean_probability = np.mean(stacked, axis=0)
            oof_mean[method.key] = mean_probability
            per_repeat = [
                _summarize_pooled(y, values, sites)
                for values in oof[method.key]
            ]
            aggregated = {
                key: float(
                    np.nanmean([row[key] for row in per_repeat])
                )
                for key in per_repeat[0]
            }
            site_rows = _site_table(model, y, mean_probability, sites)
            aggregated["site_ece_median"] = float(
                np.nanmedian([row["ece"] for row in site_rows])
            )
            aggregated["site_ece_iqr"] = [
                float(np.nanquantile([row["ece"] for row in site_rows], 0.25)),
                float(np.nanquantile([row["ece"] for row in site_rows], 0.75)),
            ]
            aggregated["site_slope_median"] = float(
                np.nanmedian([row["calibration_slope"] for row in site_rows])
            )
            aggregated["site_slope_iqr"] = [
                float(
                    np.nanquantile(
                        [row["calibration_slope"] for row in site_rows],
                        0.25,
                    )
                ),
                float(
                    np.nanquantile(
                        [row["calibration_slope"] for row in site_rows],
                        0.75,
                    )
                ),
            ]
            aggregated["bootstrap_site_ci"] = _bootstrap_site_ci(
                y,
                mean_probability,
                sites,
            )
            method_results[method.key] = {
                "label": method.label,
                "metrics": aggregated,
                "per_repeat_metrics": per_repeat,
            }
        sites_rows = []
        for method in METHODS:
            for row in _site_table(
                model,
                y,
                oof_mean[method.key],
                sites,
            ):
                row["method"] = method.key
                sites_rows.append(row)
        result["methods"][model] = method_results
        result["models"][model] = {
            "site_rows": sites_rows,
            "empirical_bayes_fold0": site_beta_rows,
        }
    return result


def main() -> None:
    frame, arrays = _read_inputs()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUPPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SITES.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "analysis": "eICU site-level calibration with held-out-fold updates",
        "inclusion": {
            "min_site_n": MIN_SITE_N,
            "min_events": MIN_EVENTS,
            "min_nonevents": MIN_NONEVENTS,
            "n_folds": N_FOLDS,
            "n_repeats": N_REPEATS,
            "ece_bins": ECE_BINS,
            "ridge": RIDGE,
            "calibration_fit": "within-site, stratified cross-validation only",
            "empirical_bayes": (
                "site intercept/slope shrunk toward a training-set global "
                "fit, with tau-squared estimated by method of moments"
            ),
        },
        "endpoints": {},
    }
    support_rows: list[dict] = []
    site_rows: list[dict] = []
    method_rows: list[dict] = []
    for endpoint in ("any_deterioration", "severe_deterioration"):
        endpoint_result = _run_endpoint(frame, arrays, endpoint=endpoint)
        results["endpoints"][endpoint] = endpoint_result
        support_rows.extend(endpoint_result.get("site_support", []))
        if endpoint_result.get("status") != "supported":
            continue
        for model, model_result in endpoint_result["models"].items():
            for row in model_result["site_rows"]:
                row["endpoint"] = endpoint
                site_rows.append(row)
            for method_key, method_result in endpoint_result["methods"][
                model
            ].items():
                metrics = method_result["metrics"]
                method_rows.append(
                    {
                        "endpoint": endpoint,
                        "model": model,
                        "method": method_key,
                        "method_label": method_result["label"],
                        "n": metrics["n"],
                        "events": metrics["events"],
                        "auroc": metrics["auroc"],
                        "auroc_ci_low": metrics["bootstrap_site_ci"]["auroc"][0],
                        "auroc_ci_high": metrics["bootstrap_site_ci"]["auroc"][1],
                        "brier": metrics["brier"],
                        "ece": metrics["ece"],
                        "ece_ci_low": metrics["bootstrap_site_ci"]["ece"][0],
                        "ece_ci_high": metrics["bootstrap_site_ci"]["ece"][1],
                        "calibration_slope": metrics["calibration_slope"],
                        "calibration_intercept": metrics[
                            "calibration_intercept"
                        ],
                        "site_ece_median": metrics["site_ece_median"],
                        "site_slope_median": metrics["site_slope_median"],
                        "site_slope_iqr_low": metrics["site_slope_iqr"][0],
                        "site_slope_iqr_high": metrics["site_slope_iqr"][1],
                    }
                )
    OUT_JSON.write_text(
        json.dumps(results, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    pd.DataFrame(support_rows).to_csv(OUT_SUPPORT, index=False)
    pd.DataFrame(site_rows).to_csv(OUT_SITES, index=False)
    pd.DataFrame(method_rows).to_csv(OUT_METHODS, index=False)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUPPORT}")
    print(f"Wrote {OUT_SITES}")
    print(f"Wrote {OUT_METHODS}")


if __name__ == "__main__":
    main()
