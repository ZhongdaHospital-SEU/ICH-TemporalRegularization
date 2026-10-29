"""Hierarchical (partially pooled) calibration update across all eICU sites.

The primary site-level analysis required at least 30 stays, 10 events and 10
non-events per site, which left 15 of 74 hospitals for any deterioration and
8 of 74 for severe deterioration.  This sensitivity analysis keeps every
hospital: site intercepts and slopes are shrunk toward a global fit, so small
sites contribute almost nothing and large sites keep their own parameters.
Updates are still estimated in training folds only and evaluated in held-out
folds, so a stay never contributes to its own calibration parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from work.eicu_site_calibration import (  # noqa: E402
    ECE_BINS,
    MODELS,
    N_FOLDS,
    N_REPEATS,
    RIDGE,
    _apply_with_offset,
    _calibration_slope_intercept,
    _clip_probability,
    _ece,
    _expit,
    _fit_with_offset,
    _logit,
    _read_inputs,
)

OUT_JSON = ROOT / "results" / "audits" / "eicu_hierarchical_calibration.json"
SHRINKAGE_MIN_N = 30
SHRINKAGE_MIN_EVENTS = 10
SHRINKAGE_MIN_NONEVENTS = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261029


def _fold_ids(sites: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Assign five folds within each site, falling back to random for tiny sites."""
    rng = np.random.default_rng(seed)
    folds = np.full(len(y), -1, dtype=int)
    for site in np.unique(sites):
        index = np.flatnonzero(sites == site)
        counts = np.bincount(y[index], minlength=2)
        if len(index) >= N_FOLDS and counts.min() >= N_FOLDS:
            splitter = StratifiedKFold(
                n_splits=N_FOLDS,
                shuffle=True,
                random_state=seed,
            )
            for fold, (_, held) in enumerate(
                splitter.split(np.zeros(len(index)), y[index])
            ):
                folds[index[held]] = fold
        else:
            folds[index] = rng.integers(0, N_FOLDS, size=len(index))
    if np.any(folds < 0):
        raise RuntimeError("fold assignment did not cover every stay")
    return folds


def _site_fit(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one site, falling back to a strong ridge when the fit is unstable."""
    for ridge in (1e-6, 1.0):
        beta = _fit_with_offset(y, x, cols=2, ridge=ridge)
        if np.all(np.isfinite(beta)) and np.max(np.abs(beta)) < 20.0:
            break
    beta = np.clip(beta, -20.0, 20.0)
    design = np.column_stack([np.ones(len(y)), x])
    mu = _expit(x + design @ beta)
    weight = np.clip(mu * (1.0 - mu), 1e-8, None)
    hessian = design.T @ (design * weight[:, None]) + np.diag([1e-8, ridge])
    try:
        variance = np.clip(np.diag(np.linalg.inv(hessian)), 1e-8, None)
    except np.linalg.LinAlgError:
        variance = np.full(2, 1e6)
    return beta, variance


def _hierarchical_fit(
    y: np.ndarray,
    x: np.ndarray,
    sites: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    """Fit a global update and shrink site updates toward it.

    Between-site variances are estimated by a method-of-moments approach on
    sites that meet the support thresholds of the primary analysis.  The
    estimate can be zero, in which case the update collapses to the global
    fit; that is a result about the data, not a failure of the fit.
    """
    global_beta = _fit_with_offset(y, x, cols=2, ridge=1e-6)
    local: dict[int, np.ndarray] = {}
    variance: dict[int, np.ndarray] = {}
    for site in np.unique(sites):
        mask = sites == site
        beta, site_variance = _site_fit(y[mask], x[mask])
        local[int(site)] = beta
        variance[int(site)] = site_variance

    anchor = [
        int(site)
        for site in np.unique(sites)
        if int(np.sum(sites == site)) >= SHRINKAGE_MIN_N
        and int(np.sum(y[sites == site])) >= SHRINKAGE_MIN_EVENTS
        and int(np.sum(1 - y[sites == site])) >= SHRINKAGE_MIN_NONEVENTS
    ]
    if len(anchor) < 3:
        anchor = [int(site) for site in np.unique(sites)]
    stacked = np.vstack([local[site] for site in anchor])
    within = np.mean(np.vstack([variance[site] for site in anchor]), axis=0)
    tau2 = np.clip(np.var(stacked, axis=0, ddof=1) - within, 1e-8, None)

    shrunk: dict[int, np.ndarray] = {}
    weights = []
    for site, beta in local.items():
        weight = tau2 / (tau2 + variance[site])
        shrunk[site] = weight * beta + (1.0 - weight) * global_beta
        weights.append(float(weight[1]))
    summary = {
        "tau_sd_intercept": float(np.sqrt(tau2[0])),
        "tau_sd_slope": float(np.sqrt(tau2[1])),
        "median_slope_weight": float(np.median(weights)),
        "global_intercept": float(global_beta[0]),
        "global_slope": float(global_beta[1]),
        "anchor_sites": len(anchor),
    }
    return global_beta, shrunk, summary


def _apply_update(
    x: np.ndarray,
    sites: np.ndarray,
    global_beta: np.ndarray,
    shrunk: dict[int, np.ndarray],
    fallback: dict[int, np.ndarray],
) -> np.ndarray:
    beta = np.empty((len(x), 2), dtype=float)
    for row, site in enumerate(sites):
        beta[row] = shrunk.get(int(site), fallback[int(site)])
    return _expit(x + beta[:, 0] + beta[:, 1] * x)


def _pooled(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    slope, intercept = _calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, _clip_probability(p))),
        "ece": _ece(y, p),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


def _cluster_bootstrap_change(
    y: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    sites: np.ndarray,
) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.unique(sites)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(sites == site) for site in chosen])
        values.append(_ece(y[index], after[index]) - _ece(y[index], before[index]))
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _run_endpoint(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    endpoint: str,
) -> dict:
    y_all = (
        (frame["outcome"].to_numpy() >= 1).astype(int)
        if endpoint == "any_deterioration"
        else (frame["outcome"].to_numpy() == 2).astype(int)
    )
    sites_all = frame["hospitalid"].to_numpy()
    result: dict = {
        "endpoint": endpoint,
        "n_stays": int(len(frame)),
        "n_sites": int(len(np.unique(sites_all))),
        "models": {},
    }
    for model in MODELS:
        probabilities = np.asarray(arrays[model], dtype=float)
        risk = (
            probabilities[:, 1] + probabilities[:, 2]
            if endpoint == "any_deterioration"
            else probabilities[:, 2]
        )
        x = _logit(risk)
        before = _clip_probability(risk)
        oof = np.zeros((N_REPEATS, len(y_all)), dtype=float)
        summaries = []
        for repeat in range(N_REPEATS):
            folds = _fold_ids(sites_all, y_all, 20261029 + repeat)
            repeat_prediction = np.full(len(y_all), np.nan)
            for fold in range(N_FOLDS):
                train = folds != fold
                test = folds == fold
                _, shrunk, summary = _hierarchical_fit(
                    y_all[train], x[train], sites_all[train]
                )
                global_beta = _fit_with_offset(
                    y_all[train], x[train], cols=2
                )
                repeat_prediction[test] = _apply_update(
                    x[test],
                    sites_all[test],
                    global_beta,
                    shrunk,
                    {site: global_beta for site in np.unique(sites_all)},
                )
                summaries.append(summary)
            oof[repeat] = repeat_prediction
        after = np.mean(oof, axis=0)
        pooled_before = _pooled(y_all, before)
        pooled_after = _pooled(y_all, after)
        result["models"][model] = {
            "pooled_before": pooled_before,
            "pooled_after": pooled_after,
            "ece_change": pooled_after["ece"] - pooled_before["ece"],
            "ece_change_ci": _cluster_bootstrap_change(
                y_all, before, after, sites_all
            ),
            "tau_sd_intercept_median": float(
                np.median([row["tau_sd_intercept"] for row in summaries])
            ),
            "tau_sd_slope_median": float(
                np.median([row["tau_sd_slope"] for row in summaries])
            ),
            "median_slope_weight": float(
                np.median([row["median_slope_weight"] for row in summaries])
            ),
        }

        site_rows = []
        for site in np.unique(sites_all):
            mask = sites_all == site
            site_rows.append(
                {
                    "hospitalid": int(site),
                    "n": int(mask.sum()),
                    "events": int(y_all[mask].sum()),
                    "ece_before": _ece(y_all[mask], before[mask]),
                    "ece_after": _ece(y_all[mask], after[mask]),
                }
            )
        large = [row for row in site_rows if row["n"] >= 10]
        result["models"][model]["sites_with_at_least_10_stays"] = {
            "n_sites": len(large),
            "lower_ece": int(
                sum(row["ece_after"] < row["ece_before"] for row in large)
            ),
            "median_change": float(
                np.median(
                    [row["ece_after"] - row["ece_before"] for row in large]
                )
            )
            if large
            else float("nan"),
        }
    return result


def main() -> None:
    frame, arrays = _read_inputs()
    results = {
        "schema_version": 1,
        "analysis": (
            "Hierarchical calibration update across all eICU hospitals, with "
            "partial pooling of site intercepts and slopes"
        ),
        "protocol": {
            "sites": "all hospitals in the analytic cohort, no size threshold",
            "n_folds": N_FOLDS,
            "n_repeats": N_REPEATS,
            "ece_bins": ECE_BINS,
            "ridge": RIDGE,
            "shrinkage": (
                "method-of-moments tau-squared estimated on sites with at "
                f"least {SHRINKAGE_MIN_N} stays, {SHRINKAGE_MIN_EVENTS} "
                f"events and {SHRINKAGE_MIN_NONEVENTS} non-events; site "
                "updates shrunk toward the training-fold global fit"
            ),
            "evaluation": "held-out folds only; a stay never informs its own update",
            "bootstrap": "hospitals resampled, 2000 replicates",
        },
        "endpoints": {},
    }
    for endpoint in ("any_deterioration", "severe_deterioration"):
        results["endpoints"][endpoint] = _run_endpoint(frame, arrays, endpoint)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUT_JSON}")
    for endpoint, payload in results["endpoints"].items():
        for model, row in payload["models"].items():
            print(
                endpoint,
                model,
                f"ECE {row['pooled_before']['ece']:.3f} -> "
                f"{row['pooled_after']['ece']:.3f}",
                f"change {row['ece_change']:+.3f} "
                f"({row['ece_change_ci'][0]:+.3f}, {row['ece_change_ci'][1]:+.3f})",
                f"tau_slope {row['tau_sd_slope_median']:.2f}",
            )


if __name__ == "__main__":
    main()
