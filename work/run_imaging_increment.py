"""Test prespecified imaging-report phenotypes within the supported note era."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import PROCESSED


IMAGING_FEATURES = [
    "imaging_ivh",
    "imaging_hydrocephalus",
    "imaging_midline_shift",
    "imaging_herniation",
    "imaging_lobar",
    "imaging_deep",
    "imaging_posterior_fossa",
]


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    classifier = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=260,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        eval_metric="mlogloss",
    )
    classifier.fit(train_x, train_y, verbose=False)
    return classifier.predict_proba(test_x)


def _train_fold(task: dict) -> tuple[int, int, dict[str, np.ndarray]]:
    train_index = np.asarray(task["train_index"], dtype=np.int64)
    test_index = np.asarray(task["test_index"], dtype=np.int64)
    base = np.asarray(task["base"], dtype=np.float32)
    labels = np.asarray(task["labels"], dtype=np.int64)
    imaging = np.asarray(task["imaging"], dtype=np.float32)
    added = np.concatenate([base, imaging], axis=1)

    base_median = np.nanmedian(base[train_index], axis=0)
    base_median[~np.isfinite(base_median)] = 0.0
    base_train = np.where(
        np.isfinite(base[train_index]), base[train_index], base_median
    )
    base_test = np.where(
        np.isfinite(base[test_index]), base[test_index], base_median
    )
    output = {}
    for name, train_x, test_x in (
        ("baseline", base_train, base_test),
        ("imaging_added", added[train_index], added[test_index]),
    ):
        parts = [
            _fit_predict(
                train_x,
                labels[train_index],
                test_x,
                int(task["seed"]) + offset,
            )
            for offset in range(int(task["seed_count"]))
        ]
        output[name] = np.mean(parts, axis=0)
    return int(task["repeat"]), int(task["fold"]), output


def _configure_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--worker-threads", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--metric-workers", type=int, default=4)
    args = parser.parse_args()

    te.ensure_dirs()
    config = te.load_config()
    seed = int(config["random_seed"])
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    imaging = pd.read_csv(
        PROCESSED / "mimic4" / "imaging_features_v2.csv"
    )
    patients = pd.read_csv(
        Path(r"D:\TT paper\0811Temporal Pathway")
        / "mimic-iv-3.1"
        / "hosp"
        / "patients.csv.gz",
        usecols=["subject_id", "anchor_year_group"],
    )
    frame = (
        cohort.merge(imaging, on="stay_id", how="left")
        .merge(patients, on="subject_id", how="left")
    )
    supported_groups = {
        "2008 - 2010",
        "2011 - 2013",
        "2014 - 2016",
        "2017 - 2019",
    }
    frame = frame[
        frame["anchor_year_group"].isin(supported_groups)
        & frame["imaging_available"].eq(1)
    ].copy()
    frame = frame.sort_values("stay_id").reset_index(drop=True)
    labels = frame["outcome"].astype(int).to_numpy()
    if np.unique(labels).size != 3:
        raise ValueError("Imaging analysis cohort does not contain all grades")

    manifest = json.loads(
        (te.RESULTS / "run_manifest.json").read_text(encoding="utf-8")
    )
    feature_columns = list(manifest["features"])
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing baseline features: {missing}")
    base = frame[feature_columns].to_numpy(dtype=np.float32)
    imaging_values = frame[IMAGING_FEATURES].to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    models = ("baseline", "imaging_added")
    groups = frame["subject_key"].astype(str).to_numpy()

    predictions = {
        repeat: {
            name: np.zeros((len(frame), 3), dtype=np.float64)
            for name in models
        }
        for repeat in range(args.cv_repeats)
    }
    tasks = []
    for repeat in range(args.cv_repeats):
        splitter = te.StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=seed + repeat * 10_000,
        )
        splits = list(
            splitter.split(np.zeros(len(labels)), labels, groups=groups)
        )
        for fold, (train_index, test_index) in enumerate(splits):
            tasks.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_index": train_index,
                    "test_index": test_index,
                    "base": base,
                    "imaging": imaging_values,
                    "labels": labels,
                    "seed": seed + repeat * 10_000 + fold,
                    "seed_count": args.seed_count,
                }
            )

    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_configure_worker,
            initargs=(args.worker_threads,),
        ) as executor:
            futures = [executor.submit(_train_fold, task) for task in tasks]
            for future in as_completed(futures):
                repeat, fold, fold_predictions = future.result()
                test_index = tasks[
                    repeat * args.folds + fold
                ]["test_index"]
                for name in models:
                    predictions[repeat][name][test_index] = fold_predictions[name]
                print(f"completed repeat {repeat + 1} fold {fold + 1}", flush=True)
    else:
        for task in tasks:
            repeat, fold, fold_predictions = _train_fold(task)
            predictions[repeat][models[0]][task["test_index"]] = (
                fold_predictions[models[0]]
            )
            predictions[repeat][models[1]][task["test_index"]] = (
                fold_predictions[models[1]]
            )

    repeat_path = (
        te.RESULTS
        / "predictions"
        / "experiment_imaging_increment_repeats.npz"
    )
    repeat_payload = {
        "labels": labels,
        "stay_id": frame["stay_id"].to_numpy(),
        "subject_key": groups,
    }
    for repeat in range(args.cv_repeats):
        for name in models:
            repeat_payload[f"repeat{repeat}_{name}"] = predictions[repeat][name]
    np.savez_compressed(repeat_path, **repeat_payload)

    dataset = te.Dataset(
        name="mimic4_imaging_supported",
        cohort=frame,
        sequences=np.zeros((len(frame), 1, 1), dtype=np.float32),
        masks=np.zeros((len(frame), 1, 1), dtype=np.float32),
        labels=labels,
        static_all=frame[feature_columns],
        static_deep=frame[feature_columns],
    )
    metrics = te.evaluate_repeated_cv(
        dataset,
        repeat_path,
        models,
        args.cv_repeats,
        replicates=args.replicates,
        seed=seed,
        metric_workers=args.metric_workers,
    )
    metrics["cv_repeat_variability"] = te.repeated_cv_variability(
        dataset, repeat_path, models, args.cv_repeats
    )

    output = {
        "purpose": (
            "Fixed imaging-report increment screen in the 2008-2019 "
            "MIMIC-IV note era. Baseline and imaging models were refitted on "
            "the same complete-case folds; no external outcome selected "
            "features or weights."
        ),
        "cohort_n": int(len(frame)),
        "outcomes": {
            str(grade): int((labels == grade).sum()) for grade in (0, 1, 2)
        },
        "baseline_feature_count": int(len(feature_columns)),
        "imaging_features": IMAGING_FEATURES,
        "models": list(models),
        "folds": args.folds,
        "cv_repeats": args.cv_repeats,
        "seed_count": args.seed_count,
        "bootstrap_replicates": args.replicates,
        "metrics": metrics,
    }
    output_path = te.RESULTS / "experiments" / "imaging_increment.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
