"""Independent recomputation of the primary contrast from saved OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 5
SEED = 20261024
REPLICATES = 2000


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score((labels == k).astype(int), probabilities[:, k])
                for k in range(3)
            ]
        )
    )


def cluster_bootstrap(
    labels: np.ndarray,
    primary: list[np.ndarray],
    comparator: list[np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[list[float], float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    draws: list[float] = []
    attempts = 0
    while len(draws) < replicates and attempts < 20_000:
        attempts += 1
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([indices[g] for g in sampled])
        if np.unique(labels[index]).size < 3:
            continue
        per_repeat = [
            macro_auc(labels[index], p[index]) - macro_auc(labels[index], c[index])
            for p, c in zip(primary, comparator)
        ]
        draws.append(float(np.mean(per_repeat)))
    return [
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    ], float(np.mean(draws))


def main() -> None:
    path = ROOT / "results" / "predictions" / "mimic4_cv_core_repeats.npz"
    with np.load(path, allow_pickle=True) as data:
        labels = data["labels"].astype(int)
        groups = data["subject_key"].astype(str)
        repeated = {
            name: [data[f"repeat{r}_{name}"].astype(np.float64) for r in range(REPEATS)]
            for name in ("dggru", "dggru_tr", "gru", "xgb_summary")
        }
    print(f"n={len(labels)} clusters={len(np.unique(groups))} cases={np.bincount(labels)}")
    for name, repeats in repeated.items():
        per_repeat = [macro_auc(labels, p) for p in repeats]
        print(
            f"{name:12s} mean={np.mean(per_repeat):.6f} sd={np.std(per_repeat, ddof=1):.6f} "
            f"repeats={[round(v, 6) for v in per_repeat]}"
        )
    diffs = np.array(
        [
            macro_auc(labels, p) - macro_auc(labels, c)
            for p, c in zip(repeated["dggru_tr"], repeated["dggru"])
        ]
    )
    print(f"paired diff mean={diffs.mean():.6f} min={diffs.min():.6f} max={diffs.max():.6f}")
    for comparator in ("dggru", "xgb_summary"):
        ci, boot_mean = cluster_bootstrap(
            labels,
            repeated["dggru_tr"],
            repeated[comparator],
            groups,
            REPLICATES,
            SEED + 1,
        )
        print(f"vs {comparator:12s} bootstrap mean={boot_mean:.6f} 95% CI={ci}")
    metrics = json.loads((ROOT / "results" / "metrics_core.json").read_text())
    stored = metrics["development"]
    print(
        "stored: dggru",
        round(stored["dggru"]["point"]["macro_auc"], 6),
        "dggru_tr",
        round(stored["dggru_tr"]["point"]["macro_auc"], 6),
        "paired CI",
        [round(v, 6) for v in stored["paired_differences_vs_dggru_tr"]["dggru"]],
    )
    audit_external()


def audit_external() -> None:
    metrics = json.loads((ROOT / "results" / "metrics_core.json").read_text())["external"]
    for cohort, filename in (("mimic3", "mimic3_external_core.npz"), ("eicu", "eicu_external_core.npz")):
        path = ROOT / "results" / "predictions" / filename
        with np.load(path, allow_pickle=True) as data:
            print(f"--- {cohort} keys={sorted(data.files)[:12]}")
            labels = data["labels"].astype(int)
            stay_id = data["stay_id"].astype(np.int64)
            groups = data["subject_key"].astype(str)
            primary = data["dggru_tr"].astype(np.float64)
            comparator = data["dggru"].astype(np.float64)
        if cohort == "eicu":
            frame = pd.read_csv(
                ROOT / "data" / "processed" / "landmark_inputs" / "eicu" / "landmark_24h" / "cohort.csv",
                usecols=["stay_id", "hospital_key"],
            )
            mapping = dict(zip(frame["stay_id"].astype(np.int64), frame["hospital_key"].astype(str)))
            groups = np.asarray([mapping[int(s)] for s in stay_id])
        n_clusters = len(np.unique(groups))
        ok = min(np.bincount(labels, minlength=3)[1:]) >= 5
        print(
            f"{cohort}: n={len(labels)} clusters={n_clusters} cases={np.bincount(labels)} "
            f"dggru={macro_auc(labels, comparator):.6f} dggru_tr={macro_auc(labels, primary):.6f} "
            f"diff={macro_auc(labels, primary) - macro_auc(labels, comparator):.6f}"
        )
        ci, boot_mean = cluster_bootstrap(
            labels,
            [primary],
            [comparator],
            groups,
            REPLICATES,
            SEED + 1,
        )
        print(f"{cohort}: bootstrap mean={boot_mean:.6f} 95% CI={ci} (skip={not ok})")
        payload = metrics[cohort]
        print(
            f"{cohort} stored: dggru={payload['dggru']['point']['macro_auc']:.6f} "
            f"dggru_tr={payload['dggru_tr']['point']['macro_auc']:.6f} "
            f"CI={[round(v, 6) for v in payload['paired_differences_vs_dggru_tr']['dggru']]}"
        )


if __name__ == "__main__":
    main()
