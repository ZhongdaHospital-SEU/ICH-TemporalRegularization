"""Screen an event-time ordinal hazard model aligned with the 24-72 h endpoint."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("ICH_TORCH_THREADS", "3")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ich_dynamic import models as dm
from src.ich_dynamic import train_evaluate as te
from src.ich_dynamic.common import PROCESSED


N_BINS = 8
BIN_HOURS = 6.0
EVENT_START_HOUR = 24.0
EVENT_END_HOUR = 72.0


def _first_qualifying_time(
    events: pd.DataFrame,
    baseline: float,
    minimum_drop: float,
) -> float:
    post = events[
        events["time_hour"].between(EVENT_START_HOUR, EVENT_END_HOUR, inclusive="left")
    ].dropna(subset=["gcs"]).sort_values("time_hour")
    if post.empty or not np.isfinite(baseline):
        return math.inf
    times = post["time_hour"].to_numpy(dtype=float)
    values = post["gcs"].to_numpy(dtype=float)
    first = math.inf
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            elapsed = times[j] - times[i]
            if elapsed > 2.0 + 1e-9:
                break
            if elapsed + 1e-9 < 1.0:
                continue
            drop = baseline - max(values[i], values[j])
            if drop >= minimum_drop:
                first = min(first, float(times[j]))
    return first


def derive_event_times(
    cohort: pd.DataFrame,
    gcs_events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confirmed-event bins for any and severe events.

    Bin 0 is event-free through hour 72. Bins 1-8 cover the consecutive
    six-hour intervals from hour 24 to hour 72. Death is part of both the
    any-deterioration-or-death and severe-deterioration-or-death targets.
    """
    by_stay = {
        int(stay_id): group for stay_id, group in gcs_events.groupby("stay_id")
    }
    any_bins: list[int] = []
    severe_bins: list[int] = []
    for row in cohort.itertuples(index=False):
        events = by_stay.get(int(row.stay_id), gcs_events.iloc[0:0])
        baseline = float(row.baseline_gcs_max)
        any_time = _first_qualifying_time(events, baseline, 2.0)
        severe_time = _first_qualifying_time(events, baseline, 4.0)
        death = bool(row.death_within_72h)
        death_hour = float(row.death_hour_from_icu)
        if death and np.isfinite(death_hour):
            if EVENT_START_HOUR <= death_hour < EVENT_END_HOUR:
                any_time = min(any_time, death_hour)
                severe_time = min(severe_time, death_hour)

        any_bin = (
            0
            if not np.isfinite(any_time)
            else min(
                N_BINS,
                max(1, int((any_time - EVENT_START_HOUR) // BIN_HOURS) + 1),
            )
        )
        severe_bin = (
            0
            if not np.isfinite(severe_time)
            else min(
                N_BINS,
                max(1, int((severe_time - EVENT_START_HOUR) // BIN_HOURS) + 1),
            )
        )
        any_bins.append(any_bin)
        severe_bins.append(severe_bin)
    return (
        np.asarray(any_bins, dtype=np.int64),
        np.asarray(severe_bins, dtype=np.int64),
    )


class EventTimeHazardNet(nn.Module):
    """GRU representation with monotone cumulative ordinal event risks."""

    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        hidden: int = 64,
        use_graph: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.graph = dm.DynamicFeatureGraph(n_sequence)
        self.gru = nn.GRU(n_sequence, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(n_static),
            nn.Linear(n_static, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.any_hazard_logits = nn.Linear(hidden * 2, N_BINS)
        self.severe_share_logits = nn.Linear(hidden * 2, N_BINS)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_graph:
            sequence, _ = self.graph(sequence)
        observed_hours = (
            mask.sum(dim=-1) > 0 if mask.dim() == 3 else mask > 0
        )
        has_observation = observed_hours.any(dim=1)
        last_hour = (
            observed_hours.shape[1]
            - 1
            - torch.flip(observed_hours, dims=[1]).to(torch.int64).argmax(dim=1)
        )
        last_hour = torch.where(
            has_observation, last_hour, torch.zeros_like(last_hour)
        )
        lengths = (last_hour + 1).clamp(min=1, max=sequence.shape[1])
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        hidden = F.gelu(self.projection(self.norm(output)))
        last_index = (lengths - 1).clamp(min=0).unsqueeze(1)
        final_hidden = hidden.gather(
            1, last_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        ).squeeze(1)
        static_hidden = self.static_encoder(static)
        representation = torch.cat(
            [self.dropout(final_hidden), static_hidden], dim=1
        )

        any_hazard = torch.sigmoid(self.any_hazard_logits(representation))
        severe_share = torch.sigmoid(
            self.severe_share_logits(representation)
        )
        severe_hazard = any_hazard * severe_share
        cumulative_any = 1.0 - torch.cumprod(1.0 - any_hazard, dim=1)
        cumulative_severe = 1.0 - torch.cumprod(1.0 - severe_hazard, dim=1)
        p0 = 1.0 - cumulative_any[:, -1]
        p1 = cumulative_any[:, -1] - cumulative_severe[:, -1]
        p2 = cumulative_severe[:, -1]
        probabilities = torch.stack([p0, p1, p2], dim=1).clamp_min(1e-7)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return (
            probabilities,
            cumulative_any,
            cumulative_severe,
            any_hazard,
            severe_hazard,
        )


class HazardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
        any_bins: np.ndarray,
        severe_bins: np.ndarray,
    ):
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.masks = torch.from_numpy(masks.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.any_bins = torch.from_numpy(any_bins.astype(np.int64))
        self.severe_bins = torch.from_numpy(severe_bins.astype(np.int64))

    def __len__(self) -> int:
        return len(self.any_bins)

    def __getitem__(self, index: int):
        return (
            self.sequences[index],
            self.masks[index],
            self.static[index],
            self.any_bins[index],
            self.severe_bins[index],
        )


def discrete_event_nll(hazard: torch.Tensor, event_bin: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for interval hazards with administrative censoring."""
    clipped = hazard.clamp(1e-6, 1.0 - 1e-6)
    log_survival_steps = torch.log1p(-clipped)
    cumulative_log_survival = torch.cumsum(log_survival_steps, dim=1)
    censored = event_bin <= 0
    event_index = (event_bin - 1).clamp(min=0)
    event_hazard = clipped.gather(1, event_index.unsqueeze(1)).squeeze(1)
    cumulative_before = cumulative_log_survival.gather(
        1, event_index.unsqueeze(1)
    ).squeeze(1) - log_survival_steps.gather(
        1, event_index.unsqueeze(1)
    ).squeeze(1)
    event_nll = -(cumulative_before + torch.log(event_hazard))
    censored_nll = -cumulative_log_survival[:, -1]
    return torch.where(censored, censored_nll, event_nll).mean()


def hazard_smoothness(hazard: torch.Tensor) -> torch.Tensor:
    first_difference = hazard[:, 1:] - hazard[:, :-1]
    second_difference = first_difference[:, 1:] - first_difference[:, :-1]
    return second_difference.square().mean()


def _configure_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _train_one(
    train_sequences: np.ndarray,
    train_masks: np.ndarray,
    train_static: np.ndarray,
    train_any: np.ndarray,
    train_severe: np.ndarray,
    valid_sequences: np.ndarray,
    valid_masks: np.ndarray,
    valid_static: np.ndarray,
    valid_any: np.ndarray,
    valid_severe: np.ndarray,
    *,
    use_regularization: bool,
    lambda_smooth: float,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EventTimeHazardNet(
        train_sequences.shape[2],
        train_static.shape[1],
        use_graph=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    train_loader = torch.utils.data.DataLoader(
        HazardDataset(
            train_sequences,
            train_masks,
            train_static,
            train_any,
            train_severe,
        ),
        batch_size=128,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        HazardDataset(
            valid_sequences,
            valid_masks,
            valid_static,
            valid_any,
            valid_severe,
        ),
        batch_size=256,
        shuffle=False,
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for sequence, mask, static, any_bin, severe_bin in train_loader:
            _, _, _, any_hazard, severe_hazard = model(
                sequence, mask, static
            )
            loss = discrete_event_nll(any_hazard, any_bin)
            loss = loss + discrete_event_nll(severe_hazard, severe_bin)
            if use_regularization:
                loss = loss + lambda_smooth * (
                    hazard_smoothness(any_hazard)
                    + hazard_smoothness(severe_hazard)
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for sequence, mask, static, any_bin, severe_bin in valid_loader:
                _, _, _, any_hazard, severe_hazard = model(
                    sequence, mask, static
                )
                loss = discrete_event_nll(any_hazard, any_bin)
                loss = loss + discrete_event_nll(severe_hazard, severe_bin)
                total += float(loss.item()) * len(any_bin)
        validation_loss = total / len(valid_sequences)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(valid_sequences), 256):
            stop = min(start + 256, len(valid_sequences))
            probabilities, _, _, _, _ = model(
                torch.from_numpy(valid_sequences[start:stop].astype(np.float32)),
                torch.from_numpy(valid_masks[start:stop].astype(np.float32)),
                torch.from_numpy(valid_static[start:stop].astype(np.float32)),
            )
            parts.append(probabilities.numpy())
    return np.concatenate(parts), best_epoch


def _train_fold(task: dict) -> tuple[int, dict[str, np.ndarray], dict[str, list[int]]]:
    fold = int(task["fold"])
    train_index = np.asarray(task["train_index"], dtype=np.int64)
    test_index = np.asarray(task["test_index"], dtype=np.int64)
    sequences = np.asarray(task["sequences"], dtype=np.float32)
    masks = np.asarray(task["masks"], dtype=np.float32)
    static = np.asarray(task["static"], dtype=np.float32)
    labels = np.asarray(task["labels"], dtype=np.int64)
    any_bins = np.asarray(task["any_bins"], dtype=np.int64)
    severe_bins = np.asarray(task["severe_bins"], dtype=np.int64)
    groups = np.asarray(task["groups"], dtype=str)
    inner_train, inner_valid = te.inner_split(
        labels[train_index],
        groups[train_index],
        int(task["seed"]),
    )
    preprocessor = te.SequencePreprocessor.fit(
        sequences[train_index][inner_train],
        masks[train_index][inner_train],
        static[train_index][inner_train],
    )
    fold_sequences, fold_masks = preprocessor.transform_sequence(
        sequences[train_index], masks[train_index]
    )
    fold_static = preprocessor.transform_static(static[train_index])
    test_sequences, test_masks = preprocessor.transform_sequence(
        sequences[test_index], masks[test_index]
    )
    test_static = preprocessor.transform_static(static[test_index])

    predictions = {name: [] for name in task["models"]}
    best_epochs = {name: [] for name in task["models"]}
    for name in task["models"]:
        for seed_offset in range(int(task["seed_count"])):
            probabilities, best_epoch = _train_one(
                fold_sequences[inner_train],
                fold_masks[inner_train],
                fold_static[inner_train],
                any_bins[train_index][inner_train],
                severe_bins[train_index][inner_train],
                fold_sequences[inner_valid],
                fold_masks[inner_valid],
                fold_static[inner_valid],
                any_bins[train_index][inner_valid],
                severe_bins[train_index][inner_valid],
                use_regularization=name == "hazard_tr",
                lambda_smooth=float(task["lambda_smooth"]),
                epochs=int(task["epochs"]),
                seed=int(task["seed"]) + seed_offset,
            )
            best_epochs[name].append(best_epoch)
            torch.manual_seed(int(task["seed"]) + seed_offset)
            np.random.seed(int(task["seed"]) + seed_offset)
            full_model = EventTimeHazardNet(
                fold_sequences.shape[2],
                fold_static.shape[1],
                use_graph=True,
            )
            optimizer = torch.optim.AdamW(
                full_model.parameters(), lr=1e-3, weight_decay=1e-4
            )
            train_loader = torch.utils.data.DataLoader(
                HazardDataset(
                    fold_sequences,
                    fold_masks,
                    fold_static,
                    any_bins[train_index],
                    severe_bins[train_index],
                ),
                batch_size=128,
                shuffle=True,
            )
            for _ in range(best_epoch):
                full_model.train()
                for sequence, mask, static_batch, any_bin, severe_bin in train_loader:
                    _, _, _, any_hazard, severe_hazard = full_model(
                        sequence, mask, static_batch
                    )
                    loss = discrete_event_nll(any_hazard, any_bin)
                    loss = loss + discrete_event_nll(severe_hazard, severe_bin)
                    if name == "hazard_tr":
                        loss = loss + float(task["lambda_smooth"]) * (
                            hazard_smoothness(any_hazard)
                            + hazard_smoothness(severe_hazard)
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
                    optimizer.step()
            full_model.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(test_sequences), 256):
                    stop = min(start + 256, len(test_sequences))
                    batch_probabilities, _, _, _, _ = full_model(
                        torch.from_numpy(test_sequences[start:stop].astype(np.float32)),
                        torch.from_numpy(test_masks[start:stop].astype(np.float32)),
                        torch.from_numpy(test_static[start:stop].astype(np.float32)),
                    )
                    parts.append(batch_probabilities.numpy())
            predictions[name].append(np.concatenate(parts))
    return (
        fold,
        {
            name: np.mean(values, axis=0)
            for name, values in predictions.items()
        },
        best_epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--worker-threads", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    te.ensure_dirs()
    config = te.load_config()
    seed = int(config["random_seed"])
    development = te.load_dataset("mimic4")
    gcs_events = pd.read_csv(PROCESSED / "mimic4" / "gcs_events.csv")
    any_bins, severe_bins = derive_event_times(development.cohort, gcs_events)
    if not np.array_equal(
        (any_bins > 0).astype(int), (development.labels > 0).astype(int)
    ):
        raise ValueError("Any-event bins disagree with ordinal outcome labels")
    if not np.array_equal(
        (severe_bins > 0).astype(int), (development.labels == 2).astype(int)
    ):
        raise ValueError("Severe-event bins disagree with ordinal outcome labels")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "development_n": len(development.labels),
                    "outcomes": {
                        str(grade): int((development.labels == grade).sum())
                        for grade in (0, 1, 2)
                    },
                    "any_bins": np.bincount(
                        any_bins, minlength=N_BINS + 1
                    ).tolist(),
                    "severe_bins": np.bincount(
                        severe_bins, minlength=N_BINS + 1
                    ).tolist(),
                },
                indent=2,
            )
        )
        return

    models = ("hazard", "hazard_tr")
    predictions = {
        name: np.zeros((len(development.labels), 3), dtype=np.float64)
        for name in models
    }
    repeat_predictions = {
        repeat: {
            name: np.zeros((len(development.labels), 3), dtype=np.float64)
            for name in models
        }
        for repeat in range(args.cv_repeats)
    }
    best_epochs = {name: [] for name in models}
    groups = development.cohort["subject_key"].astype(str).to_numpy()
    tasks = []
    for repeat in range(args.cv_repeats):
        splitter = te.StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=seed + repeat * 10_000,
        )
        splits = list(
            splitter.split(development.static_all, development.labels, groups=groups)
        )
        for fold, (train_index, test_index) in enumerate(splits):
            tasks.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_index": train_index,
                    "test_index": test_index,
                    "sequences": development.sequences,
                    "masks": development.masks,
                    "static": development.static_all.fillna(
                        development.static_all.median()
                    ).to_numpy(dtype=np.float32),
                    "labels": development.labels,
                    "any_bins": any_bins,
                    "severe_bins": severe_bins,
                    "groups": groups,
                    "models": models,
                    "seed": seed + repeat * 10_000 + fold,
                    "seed_count": args.seed_count,
                    "epochs": args.epochs,
                    "lambda_smooth": args.lambda_smooth,
                }
            )

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_configure_worker,
            initargs=(args.worker_threads,),
        ) as executor:
            futures = [executor.submit(_train_fold, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"completed fold {result[0] + 1}", flush=True)
    else:
        torch.set_num_threads(args.worker_threads)
        for task in tasks:
            results.append(_train_fold(task))
            print(f"completed fold {task['fold'] + 1}", flush=True)

    for fold, fold_predictions, fold_epochs in results:
        matching = next(task for task in tasks if task["fold"] == fold)
        repeat = int(matching["repeat"])
        test_index = np.asarray(matching["test_index"], dtype=np.int64)
        for name in models:
            repeat_predictions[repeat][name][test_index] = fold_predictions[name]
            best_epochs[name].extend(fold_epochs[name])
    for name in models:
        predictions[name] = np.mean(
            [repeat_predictions[repeat][name] for repeat in range(args.cv_repeats)],
            axis=0,
        )

    output_dir = te.RESULTS / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_path = te.RESULTS / "predictions" / "experiment_hazard_repeats.npz"
    repeat_payload = {
        "labels": development.labels,
        "stay_id": development.cohort["stay_id"].to_numpy(),
        "subject_key": groups,
    }
    for repeat in range(args.cv_repeats):
        for name in models:
            repeat_payload[f"repeat{repeat}_{name}"] = repeat_predictions[repeat][name]
    np.savez_compressed(repeat_path, **repeat_payload)

    if args.cv_repeats > 1:
        metrics = te.evaluate_repeated_cv(
            development,
            repeat_path,
            models,
            args.cv_repeats,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )
        metrics["cv_repeat_variability"] = te.repeated_cv_variability(
            development, repeat_path, models, args.cv_repeats
        )
    else:
        metrics = te.evaluate_predictions(
            development,
            predictions,
            replicates=args.replicates,
            seed=seed,
            metric_workers=args.metric_workers,
        )

    payload = {
        "purpose": (
            "Event-time ordinal hazard screen: predict confirmed GCS "
            "deterioration or death in six-hour intervals from hours 24-72. "
            "Cumulative risks are monotone by construction; hazard_tr adds a "
            "predefined smoothness penalty to interval hazards."
        ),
        "development_n": len(development.labels),
        "outcomes": {
            str(grade): int((development.labels == grade).sum())
            for grade in (0, 1, 2)
        },
        "event_bins": {
            "n_bins": N_BINS,
            "bin_hours": BIN_HOURS,
            "any_counts": np.bincount(any_bins, minlength=N_BINS + 1).tolist(),
            "severe_counts": np.bincount(
                severe_bins, minlength=N_BINS + 1
            ).tolist(),
        },
        "models": list(models),
        "folds": args.folds,
        "cv_repeats": args.cv_repeats,
        "epochs": args.epochs,
        "seed_count": args.seed_count,
        "lambda_smooth": args.lambda_smooth,
        "bootstrap_replicates": args.replicates,
        "cv_best_epochs": best_epochs,
        "metrics": metrics,
    }
    output_path = output_dir / "hazard_tr.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
