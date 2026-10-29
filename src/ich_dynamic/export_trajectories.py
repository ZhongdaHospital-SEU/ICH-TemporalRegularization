"""Export cross-fitted latent risk trajectories for Figure 5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .common import RESULTS, ensure_dirs
from .models import (
    SequencePreprocessor,
    fit_full_temporal_model,
    predict_temporal_model,
    train_temporal_model,
)
from .train_evaluate import (
    LAPLACE_INPUT_SCALE,
    MODEL_META,
    inner_split,
    load_dataset,
)


FIGURES = RESULTS / "figures"
FIGURE5_STEM = "figure5_temporal_structure"
TRAJECTORY_MODELS = (
    "dggru",
    "dggru_tr",
    "dggru_tr_treverse",
    "dggru_tr_tpermute",
    "dggru_laplace",
)
MODELS = {name: MODEL_META[name] for name in TRAJECTORY_MODELS}
MODEL_LABELS = {
    "dggru": "DG-GRU",
    "dggru_tr": "DG-GRU + TR",
    "dggru_tr_treverse": "DG-GRU + TR, time reversed",
    "dggru_tr_tpermute": "DG-GRU + TR, time permuted",
    "dggru_laplace": "DG-GRU + generic Laplace input-noise control",
}
COLORS = {
    "dggru": "#009E73",
    "dggru_tr": "#CC79A7",
    "dggru_tr_treverse": "#4C78A8",
    "dggru_tr_tpermute": "#F58518",
    "dggru_laplace": "#666666",
}


def temporal_training_options(
    meta: dict,
    *,
    lambda_monotone: float,
    lambda_smooth: float,
) -> dict:
    """Translate current model metadata into exact temporal-training arguments."""
    laplace_only = bool(meta.get("laplace_only", False))
    return {
        "use_graph": bool(meta.get("graph", False)),
        "use_regularization": bool(meta.get("regularized", False)),
        "architecture": str(meta.get("architecture", "dggru")),
        "regularization_direction": str(
            meta.get("direction", "increasing")
        ),
        "lambda_monotone": 0.0 if laplace_only else float(lambda_monotone),
        "lambda_smooth": float(lambda_smooth),
        "time_mode": str(meta.get("time_mode", "forward")),
        "laplace_only": laplace_only,
        "laplace_input_scale": LAPLACE_INPUT_SCALE,
    }


def fit_fold(
    dataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    meta: dict,
    *,
    seed: int,
    epochs: int,
    lambda_monotone: float,
    lambda_smooth: float,
    model_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    best_epochs: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    groups = dataset.cohort["subject_key"].astype(str).to_numpy()[train_index]
    raw_static = dataset.static_all.to_numpy(dtype=np.float32)
    training_options = temporal_training_options(
        meta,
        lambda_monotone=lambda_monotone,
        lambda_smooth=lambda_smooth,
    )
    if best_epochs is not None and len(best_epochs) != len(model_seeds):
        raise ValueError(
            "best_epochs must contain one epoch for every requested model seed"
        )
    outer_preprocessor = SequencePreprocessor.fit(
        dataset.sequences[train_index],
        dataset.masks[train_index],
        raw_static[train_index],
    )
    train_sequences, train_masks = outer_preprocessor.transform_sequence(
        dataset.sequences[train_index], dataset.masks[train_index]
    )
    test_sequences, test_masks = outer_preprocessor.transform_sequence(
        dataset.sequences[test_index], dataset.masks[test_index]
    )
    train_static = outer_preprocessor.transform_static(raw_static[train_index])
    test_static = outer_preprocessor.transform_static(raw_static[test_index])
    probability_parts = []
    trajectory_parts = []
    for offset, model_seed_offset in enumerate(model_seeds):
        model_seed = seed + 1000 * offset
        if best_epochs is None:
            inner_train, inner_valid = inner_split(
                dataset.labels[train_index], groups, model_seed
            )
            inner_preprocessor = SequencePreprocessor.fit(
                dataset.sequences[train_index][inner_train],
                dataset.masks[train_index][inner_train],
                raw_static[train_index][inner_train],
            )
            inner_sequences, inner_masks = inner_preprocessor.transform_sequence(
                dataset.sequences[train_index], dataset.masks[train_index]
            )
            inner_static = inner_preprocessor.transform_static(
                raw_static[train_index]
            )
            _, _, history = train_temporal_model(
                inner_sequences[inner_train],
                inner_masks[inner_train],
                inner_static[inner_train],
                dataset.labels[train_index][inner_train],
                inner_sequences[inner_valid],
                inner_masks[inner_valid],
                inner_static[inner_valid],
                dataset.labels[train_index][inner_valid],
                epochs=epochs,
                seed=model_seed,
                **training_options,
            )
            selected_epoch = int(
                min(history, key=lambda row: row["valid_loss"])["epoch"]
            )
        else:
            selected_epoch = int(best_epochs[offset])
        model_seed = seed + 1000 * int(model_seed_offset)
        model = fit_full_temporal_model(
            train_sequences,
            train_masks,
            train_static,
            dataset.labels[train_index],
            seed=model_seed,
            epochs=selected_epoch,
            **training_options,
        )
        probabilities, trajectories = predict_temporal_model(
            model,
            test_sequences,
            test_masks,
            test_static,
            time_mode=training_options["time_mode"],
        )
        probability_parts.append(probabilities)
        trajectory_parts.append(trajectories)
    return np.mean(probability_parts, axis=0), np.mean(trajectory_parts, axis=0)


def _cell_values(
    manifest: dict,
    block_name: str,
    model: str,
    cell: int,
    n_seeds: int,
) -> list[float | int]:
    """Return recorded seed-level values for one repeat/fold cell."""
    values = list(
        manifest.get(block_name, {})
        .get(model, {})
        .get("values", [])
    )
    expected = int(manifest["folds"]) * int(manifest["cv_repeats"]) * n_seeds
    if len(values) != expected:
        raise ValueError(
            f"{block_name}[{model}] contains {len(values)} values; "
            f"expected {expected} for folds x repeats x seeds"
        )
    start = int(cell) * n_seeds
    return values[start : start + n_seeds]


def _artifact_tag_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def trajectory_statistics(
    trajectories: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    observed = masks.sum(axis=2) > 0
    for i in range(len(trajectories)):
        valid_hours = np.flatnonzero(observed[i])
        if len(valid_hours) < 2:
            last = trajectories.shape[1]
        else:
            last = int(valid_hours.max()) + 1
        risk = trajectories[i, :last]
        differences = np.diff(risk)
        rows.append(
            {
                "outcome": int(labels[i]),
                "last_hour": last,
                "final_risk": float(risk[-1]),
                "mean_risk": float(np.mean(risk)),
                "negative_step_fraction": float(np.mean(differences < -1e-6)),
                "monotone_violation": float(np.mean(np.maximum(-differences, 0))),
                "roughness": float(np.mean(np.abs(differences))),
                "total_variation": float(np.sum(np.abs(differences))),
            }
        )
    return pd.DataFrame(rows)


def landmark_metrics_frame() -> pd.DataFrame:
    """Load prespecified landmark estimates on the matched predictor contract."""
    rows = []
    for hours in (6, 24, 48):
        path = (
            RESULTS
            / "sensitivity"
            / "mimic4"
            / f"s6_landmark_{hours}h.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Landmark sensitivity metrics are missing: {path}"
            )
        block = json.loads(path.read_text(encoding="utf-8"))
        block = block.get("development", block)
        for model in TRAJECTORY_MODELS:
            if model not in block:
                raise ValueError(f"{path} is missing model {model}.")
            interval = block[model].get("ci", {}).get("macro_auc")
            rows.append(
                {
                    "landmark_hours": hours,
                    "model": model,
                    "macro_auc": float(block[model]["point"]["macro_auc"]),
                    "ci_low": (
                        float(interval[0])
                        if interval and len(interval) == 2
                        else np.nan
                    ),
                    "ci_high": (
                        float(interval[1])
                        if interval and len(interval) == 2
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def create_figure(
    trajectories: dict[str, np.ndarray],
    masks: np.ndarray,
    labels: np.ndarray,
    landmark_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    observed = masks.sum(axis=2) > 0
    summary_rows = []
    if landmark_metrics is None or landmark_metrics.empty:
        raise ValueError(
            "Figure 5 requires the 6-, 24-, and 48-hour landmark estimates."
        )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    landmark_ax = axes[0, 0]
    risk_ax = axes[0, 1]
    reversal_ax = axes[1, 0]
    scatter_ax = axes[1, 1]
    for model in TRAJECTORY_MODELS:
        subset = (
            landmark_metrics[landmark_metrics["model"].eq(model)]
            .sort_values("landmark_hours")
        )
        point = subset["macro_auc"].to_numpy(dtype=float)
        lower = point - subset["ci_low"].to_numpy(dtype=float)
        upper = subset["ci_high"].to_numpy(dtype=float) - point
        landmark_ax.errorbar(
            subset["landmark_hours"].to_numpy(dtype=float),
            point,
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=3,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    landmark_ax.set_xticks([6, 24, 48])
    landmark_ax.set_xlabel("Prediction landmark (ICU hour)")
    landmark_ax.set_ylabel("Macro AUC")
    landmark_ax.set_title(
        "Landmark discrimination, matched 92 predictors",
        fontsize=10,
    )
    landmark_ax.legend(frameon=False, fontsize=6.5)
    landmark_ax.grid(alpha=0.2)

    hours = np.arange(trajectories["dggru_tr"].shape[1])
    for grade, color in zip((0, 1, 2), ("#4C78A8", "#F58518", "#B2182B")):
        subset = labels == grade
        matrix = trajectories["dggru_tr"][subset].copy()
        matrix[~observed[subset]] = np.nan
        median = np.nanmedian(matrix, axis=0)
        low, high = np.nanpercentile(matrix, [25, 75], axis=0)
        risk_ax.plot(
            hours,
            median,
            color=color,
            label=f"Grade {grade} (n={subset.sum()})",
        )
        risk_ax.fill_between(hours, low, high, color=color, alpha=0.15)
    risk_ax.set_xlabel("ICU hour")
    risk_ax.set_ylabel("Latent deterioration risk")
    risk_ax.set_title("DG-GRU + TR risk by outcome", fontsize=10)
    risk_ax.legend(frameon=False, fontsize=8)
    risk_ax.grid(alpha=0.2)

    selected = TRAJECTORY_MODELS
    statistics_by_model = {
        model: trajectory_statistics(trajectories[model], masks, labels)
        for model in selected
    }
    positions = np.arange(len(selected))
    values = []
    for model in selected:
        statistics = statistics_by_model[model]
        values.append(statistics["negative_step_fraction"].to_numpy())
        summary_rows.append(
            {
                "model": model,
                "negative_step_fraction_median": float(
                    statistics["negative_step_fraction"].median()
                ),
                "negative_step_fraction_mean": float(
                    statistics["negative_step_fraction"].mean()
                ),
                "roughness_median": float(statistics["roughness"].median()),
                "monotone_violation_median": float(
                    statistics["monotone_violation"].median()
                ),
                "total_variation_median": float(
                    statistics["total_variation"].median()
                ),
                "final_risk_auc_any": float(
                    roc_auc_score(
                        (labels >= 1).astype(int), statistics["final_risk"]
                    )
                ),
            }
        )
    box = reversal_ax.boxplot(
        values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    for patch, model in zip(box["boxes"], selected):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.65)
    reversal_ax.set_xticks(positions)
    reversal_ax.set_xticklabels(
        [
            MODEL_LABELS[model]
            .replace(" + TR, ", "\n+ TR, ")
            .replace(" + generic ", "\n+ generic ")
            for model in selected
        ],
        fontsize=8,
    )
    reversal_ax.set_ylabel("Fraction of decreasing hourly steps")
    reversal_ax.set_title("Trajectory reversals", fontsize=10)
    reversal_ax.grid(axis="y", alpha=0.2)

    for model in selected:
        statistics = statistics_by_model[model]
        scatter_ax.scatter(
            statistics["roughness"],
            statistics["final_risk"],
            s=8,
            alpha=0.2,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    scatter_ax.set_xlabel("Mean absolute hourly change")
    scatter_ax.set_ylabel("Landmark latent risk")
    scatter_ax.set_title("Risk level and trajectory roughness", fontsize=10)
    scatter_ax.grid(alpha=0.2)
    scatter_ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Figure 5. Temporal structure and regularization effects",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / f"{FIGURE5_STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{FIGURE5_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="",
        help="Run artifact suffix without a leading underscore.",
    )
    args = parser.parse_args(argv)
    ensure_dirs()
    FIGURES.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("mimic4")
    suffix = _artifact_tag_suffix(args.tag)
    manifest_path = RESULTS / f"run_manifest{suffix}.json"
    folds_path = RESULTS / f"mimic4_folds{suffix}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not folds_path.is_file():
        raise FileNotFoundError(folds_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = pd.read_csv(folds_path)
    model_seeds = tuple(int(value) for value in manifest["model_seeds"])
    n_folds = int(manifest["folds"])
    cv_repeats = int(manifest["cv_repeats"])
    if folds["repeat"].nunique() != cv_repeats:
        raise ValueError(
            f"{folds_path} contains {folds['repeat'].nunique()} repeats; "
            f"the manifest specifies {cv_repeats}"
        )

    probabilities = {
        model: np.zeros((len(dataset.labels), 3), dtype=np.float32)
        for model in MODELS
    }
    trajectories = {
        model: np.zeros(
            (len(dataset.labels), dataset.sequences.shape[1]), dtype=np.float32
        )
        for model in MODELS
    }
    base_seed = int(manifest["seed"])
    lambda_smooth_ratio = float(manifest["lambda_smooth_ratio"])
    order = {
        int(stay_id): index
        for index, stay_id in enumerate(dataset.cohort["stay_id"])
    }
    for repeat in range(cv_repeats):
        repeat_folds = folds[folds["repeat"].eq(repeat)].copy()
        if set(repeat_folds["fold"].astype(int)) != set(range(n_folds)):
            raise ValueError(f"Repeat {repeat} does not contain every outer fold")
        repeat_folds["_row"] = repeat_folds["stay_id"].map(order)
        repeat_folds = repeat_folds.sort_values("_row").reset_index(drop=True)
        if not np.array_equal(
            repeat_folds["stay_id"].to_numpy(),
            dataset.cohort["stay_id"].to_numpy(),
        ):
            raise ValueError(f"Repeat {repeat} fold rows do not match the cohort")
        fold_values = repeat_folds["fold"].to_numpy(dtype=int)
        for fold in range(n_folds):
            train_index = np.flatnonzero(fold_values != fold)
            test_index = np.flatnonzero(fold_values == fold)
            cell = repeat * n_folds + fold
            seed = base_seed + 10_000 * repeat + 100 * (fold + 1)
            print(
                f"[trajectories] repeat {repeat + 1}/{cv_repeats} "
                f"fold {fold + 1}/{n_folds}",
                flush=True,
            )
            for model, meta in MODELS.items():
                selected_lambda = float(
                    _cell_values(
                        manifest,
                        "cv_selected_lambdas",
                        model,
                        cell,
                        len(model_seeds),
                    )[0]
                )
                selected_epochs = [
                    int(value)
                    for value in _cell_values(
                        manifest,
                        "cv_best_epochs",
                        model,
                        cell,
                        len(model_seeds),
                    )
                ]
                probability, trajectory = fit_fold(
                    dataset,
                    train_index,
                    test_index,
                    meta,
                    seed=seed,
                    epochs=int(manifest["epochs"]),
                    lambda_monotone=selected_lambda,
                    lambda_smooth=lambda_smooth_ratio * selected_lambda,
                    model_seeds=model_seeds,
                    best_epochs=selected_epochs,
                )
                probabilities[model][test_index] += probability / cv_repeats
                trajectories[model][test_index] += trajectory / cv_repeats
    np.savez_compressed(
        RESULTS / "predictions" / "mimic4_cv_trajectories.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(),
        masks=dataset.masks,
        manifest_tag=np.asarray(args.tag, dtype=str),
        model_seeds=np.asarray(model_seeds, dtype=np.int64),
        cv_repeats=np.asarray(cv_repeats, dtype=np.int64),
        **{f"prob_{key}": value for key, value in probabilities.items()},
        **{f"traj_{key}": value for key, value in trajectories.items()},
    )
    landmark_metrics = landmark_metrics_frame()
    summary = create_figure(
        trajectories,
        dataset.masks,
        dataset.labels,
        landmark_metrics,
    )
    summary.to_csv(
        RESULTS / "tables" / "table5_trajectory_regularization.csv",
        index=False,
    )
    landmark_metrics.to_csv(
        RESULTS / "tables" / "table_s6_landmark_performance.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print("Saved cross-fitted trajectories")


if __name__ == "__main__":
    main()
