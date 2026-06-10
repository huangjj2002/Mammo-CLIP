"""
Train a Prototype-DST head directly on cached Mammo-CLIP embeddings.

Input bundle:
  - embeddings.npy
  - metadata.csv

Outputs:
  - dst_all_predictions.csv
  - dst_metrics.csv
  - dst_training_history.csv
  - dst_loss_curve.png
  - dst_best.pt
  - dst_manifest.json
"""

import argparse
import copy
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dst_pytorch import Dempster_Shafer_Module, DistanceActivation_layer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent


def save_training_history(history, output_dir, prefix):
    history_df = pd.DataFrame(history)
    history_path = output_dir / f"{prefix}_training_history.csv"
    history_df.to_csv(history_path, index=False)

    loss_curve_path = output_dir / f"{prefix}_loss_curve.png"
    saved_curve_path = None
    if not history_df.empty:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x_values = (
                pd.to_numeric(history_df["epoch"], errors="coerce")
                if "epoch" in history_df.columns
                else np.arange(1, len(history_df) + 1)
            )
            fig, ax_loss = plt.subplots(figsize=(8, 5))
            for col in ["train_loss", "eval_loss", "proto_loss"]:
                if col not in history_df.columns:
                    continue
                values = pd.to_numeric(history_df[col], errors="coerce")
                if values.notna().any():
                    ax_loss.plot(x_values, values, marker="o", linewidth=1.5, label=col)

            ax_loss.set_xlabel("epoch")
            ax_loss.set_ylabel("loss")
            ax_loss.grid(True, alpha=0.3)
            if ax_loss.get_legend_handles_labels()[0]:
                ax_loss.legend(loc="upper right")

            metric_cols = ["eval_auc", "eval_bacc_at_threshold", "best_metric"]
            metric_plotted = False
            ax_metric = ax_loss.twinx()
            for col in metric_cols:
                if col not in history_df.columns:
                    continue
                values = pd.to_numeric(history_df[col], errors="coerce")
                if values.notna().any():
                    ax_metric.plot(x_values, values, linestyle="--", linewidth=1.2, label=col)
                    metric_plotted = True
            if metric_plotted:
                ax_metric.set_ylabel("metric")
                ax_metric.legend(loc="lower right")
            else:
                ax_metric.remove()

            fig.tight_layout()
            fig.savefig(loss_curve_path, dpi=150)
            plt.close(fig)
            saved_curve_path = loss_curve_path
        except Exception as exc:
            print(f"[WARN] Could not save loss curve PNG: {exc}")

    return history_path, saved_curve_path


def build_parser():
    parser = argparse.ArgumentParser(description="Train Prototype-DST from cached Mammo-CLIP embeddings.")
    parser.add_argument("--embedding-dir", default="embeddings/origin_finetuned_encoder")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.npy. Overrides --embedding-dir.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.csv. Overrides --embedding-dir.")
    parser.add_argument("--output-dir", default="dst_results/origin_finetuned_dst")
    parser.add_argument("--label", default="cancer")
    parser.add_argument("--prototypes-per-class", "--prototypes_per_class", type=int, default=10)
    parser.add_argument("--prototype-topk", "--prototype_topk", type=int, default=3)
    parser.add_argument("--prototype-init", "--prototype_init", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--dst-gamma-init", "--dst_gamma_init", type=float, default=1.0)
    parser.add_argument("--dst-alpha-init", "--dst_alpha_init", type=float, default=0.0)
    parser.add_argument(
        "--temperature",
        default=None,
        help="Backward-compatible distance temperature. A positive value maps to gamma_init=1/temperature; 'auto' uses --dst-gamma-init.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization in the DST head.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=512)
    parser.add_argument(
        "--patience",
        "--early-stopping-patience",
        "--early_stopping_patience",
        type=int,
        default=10,
    )
    parser.add_argument("--class-weight-mode", "--class_weight_mode", choices=["none", "inverse", "effective"], default="inverse")
    parser.add_argument("--effective-beta", "--effective_beta", type=float, default=0.9999)
    parser.add_argument("--proto-attract-weight", "--proto_attract_weight", type=float, default=0.0)
    parser.add_argument("--proto-separation-weight", "--proto_separation_weight", type=float, default=0.0)
    parser.add_argument("--proto-diversity-weight", "--proto_diversity_weight", type=float, default=0.0)
    parser.add_argument("--proto-loss-weight", "--proto_loss_weight", type=float, default=1.0)
    parser.add_argument("--proto-margin", "--proto_margin", type=float, default=1.0)
    parser.add_argument("--proto-balance-classes", "--proto_balance_classes", choices=["y", "n"], default="y")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--best-metric", "--best_metric", choices=["bacc", "auc", "loss"], default="bacc")
    parser.add_argument("--split-mode", "--split_mode", choices=["auto", "metadata", "cohort", "fold"], default="auto")
    parser.add_argument("--split-col", "--split_col", default="split")
    parser.add_argument("--fold-col", "--fold_col", default="fold")
    parser.add_argument("--val-fold", "--val_fold", type=int, default=0)
    parser.add_argument("--cohort-col", "--cohort_col", default="cohort_num")
    parser.add_argument("--train-cohorts", "--train_cohorts", default="1-8")
    parser.add_argument("--test-cohorts", "--test_cohorts", default="9-10")
    parser.add_argument("--holdout-val-percent", "--holdout_val_percent", type=float, default=20.0)
    parser.add_argument("--group-cols", "--group_cols", default="patient_id")
    parser.add_argument("--score-agg", "--score_agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--max-samples", "--max_samples", type=int, default=None)
    parser.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", "--gpu_id", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--show-progress", "--show_progress", action="store_true")
    parser.add_argument("--n-folds", "--n_folds", type=int, default=1)
    parser.add_argument("--fold-start", "--fold_start", type=int, default=0)
    parser.add_argument("--cv-group-col", "--cv_group_col", default="patient_id")
    parser.add_argument("--run-id", "--run_id", default=None, help="Run id appended to --output-dir. Defaults to a timestamp.")
    parser.add_argument("--no-timestamp", "--no_timestamp", action="store_true", help="Use --output-dir exactly as provided.")
    return parser


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_run_id(run_id=None):
    if run_id is None or str(run_id).strip() == "":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).strip())
    return run_id.strip("_") or datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamped_output_dir(output_dir, args):
    base_dir = resolve_path(output_dir)
    if getattr(args, "no_timestamp", False):
        return base_dir, None
    run_id = resolve_run_id(getattr(args, "run_id", None))
    return base_dir.with_name(f"{base_dir.name}_run_{run_id}"), run_id


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(args.device)


def parse_int_set(spec):
    values = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(part))
    return values


def parse_group_cols(group_cols):
    cols = [col.strip() for col in str(group_cols).split(",") if col.strip()]
    return cols or ["patient_id"]


def normalize_features(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def stratified_holdout_indices(indices, labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels).astype(int)
    val_indices = []
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id].copy()
        if len(class_indices) == 0:
            continue
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_fraction))
        if len(class_indices) > 1:
            n_val = min(max(n_val, 1), len(class_indices) - 1)
        else:
            n_val = 0
        val_indices.extend(class_indices[:n_val].tolist())
    if not val_indices:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_indices = shuffled[:n_val].tolist()
    return np.asarray(val_indices, dtype=int)


def assign_splits(meta, args):
    split = pd.Series("train", index=meta.index, dtype="object")

    use_metadata = args.split_mode in {"auto", "metadata"} and args.split_col in meta.columns
    if use_metadata:
        raw = meta[args.split_col].astype(str).str.strip().str.lower()
        known = raw.isin(["train", "training", "val", "valid", "validation", "test"])
        if known.any():
            split.loc[raw.isin(["val", "valid", "validation"])] = "val"
            split.loc[raw == "test"] = "test"
            split.loc[raw.isin(["train", "training"])] = "train"

    use_fold = args.split_mode in {"auto", "fold"} and args.fold_col in meta.columns
    if use_fold:
        fold = pd.to_numeric(meta[args.fold_col], errors="coerce")
        split.loc[fold == -1] = "test"
        if (split == "val").sum() == 0:
            split.loc[(fold == int(args.val_fold)) & (split == "train")] = "val"

    use_cohort = args.split_mode in {"auto", "cohort"} and args.cohort_col in meta.columns
    if use_cohort and (split == "test").sum() == 0:
        cohort = pd.to_numeric(meta[args.cohort_col], errors="coerce")
        train_cohorts = parse_int_set(args.train_cohorts)
        test_cohorts = parse_int_set(args.test_cohorts)
        split.loc[cohort.isin(test_cohorts)] = "test"
        split.loc[cohort.isin(train_cohorts)] = "train"

    if (split == "val").sum() == 0 and args.holdout_val_percent > 0:
        train_idx = np.flatnonzero(split.values == "train")
        if len(train_idx) > 1:
            labels = meta.iloc[train_idx][args.label].astype(int).values
            val_idx = stratified_holdout_indices(
                train_idx,
                labels,
                args.holdout_val_percent / 100.0,
                args.seed,
            )
            split.iloc[val_idx] = "val"

    return split


def infer_test_mask(meta, args):
    test_mask = pd.Series(False, index=meta.index)

    if args.split_col in meta.columns:
        raw = meta[args.split_col].astype(str).str.strip().str.lower()
        test_mask |= raw == "test"

    if args.fold_col in meta.columns:
        fold = pd.to_numeric(meta[args.fold_col], errors="coerce")
        test_mask |= fold == -1

    if args.cohort_col in meta.columns:
        cohort = pd.to_numeric(meta[args.cohort_col], errors="coerce")
        test_mask |= cohort.isin(parse_int_set(args.test_cohorts))

    return test_mask


def assign_cv_splits(meta, args):
    if args.cv_group_col not in meta.columns:
        raise ValueError(f"Patient-safe CV requires group column '{args.cv_group_col}' in metadata.")

    split = pd.Series("train", index=meta.index, dtype="object")
    test_mask = infer_test_mask(meta, args)
    test_groups = meta.loc[test_mask, args.cv_group_col].drop_duplicates()
    if len(test_groups) > 0:
        test_mask = meta[args.cv_group_col].isin(test_groups)
    split.loc[test_mask] = "test"
    train_val_mask = ~test_mask

    if args.fold_col in meta.columns:
        fold = pd.to_numeric(meta[args.fold_col], errors="coerce")
        val_mask = train_val_mask & (fold == int(args.val_fold))
        if val_mask.any():
            split.loc[val_mask] = "val"
            validate_cv_group_split(meta, split, args.cv_group_col)
            return split, "metadata_fold"

        present_folds = sorted(
            int(value)
            for value in fold[train_val_mask].dropna().unique().tolist()
            if int(value) != -1
        )
        if present_folds:
            raise ValueError(
                f"No rows with {args.fold_col} == {args.val_fold}. "
                f"Available folds: {present_folds}. Try --fold_start {min(present_folds)}."
            )

    n_folds = int(getattr(args, "_cv_n_folds", args.n_folds))
    rel_fold = int(args.val_fold) - int(args.fold_start)
    if rel_fold < 0 or rel_fold >= n_folds:
        raise ValueError(f"Generated CV fold index {rel_fold} is outside [0, {n_folds - 1}].")

    pool_idx = np.flatnonzero(train_val_mask.values)
    if len(pool_idx) < n_folds:
        raise ValueError(f"Cannot generate {n_folds} folds from only {len(pool_idx)} train/val rows.")

    labels = meta[args.label].astype(int).to_numpy()
    group_labels = (
        meta.iloc[pool_idx]
        .assign(_label=labels[pool_idx])
        .groupby(args.cv_group_col, dropna=False)["_label"]
        .max()
    )
    group_values = group_labels.index.to_numpy()
    group_y = group_labels.to_numpy(dtype=int)
    if len(group_values) < n_folds:
        raise ValueError(f"Cannot generate {n_folds} folds from only {len(group_values)} {args.cv_group_col} groups.")

    rng = np.random.default_rng(args.seed)
    val_groups = []
    for class_id in np.unique(group_y):
        class_groups = group_values[group_y == class_id].copy()
        rng.shuffle(class_groups)
        folds = np.array_split(class_groups, n_folds)
        val_groups.extend(folds[rel_fold].tolist())

    if not val_groups:
        raise ValueError("Generated CV validation split is empty.")
    val_group_set = set(val_groups)
    split.loc[train_val_mask & meta[args.cv_group_col].isin(val_group_set)] = "val"
    validate_cv_group_split(meta, split, args.cv_group_col)
    return split, f"generated_group_stratified:{args.cv_group_col}"


def validate_cv_group_split(meta, split, group_col):
    grouped_splits = split.groupby(meta[group_col], dropna=False).agg(lambda values: set(values))
    leaked = grouped_splits[grouped_splits.map(len) > 1]
    if not leaked.empty:
        examples = [str(value) for value in leaked.index[:5].tolist()]
        raise ValueError(
            f"Patient leakage detected: {group_col} appears in multiple splits. "
            f"Example groups: {examples}"
        )


def stratified_sample_positions(meta, label_col, split_col, max_samples, seed):
    if max_samples is None or len(meta) <= max_samples:
        return np.arange(len(meta), dtype=int)
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    rng = np.random.default_rng(seed)
    selected = []
    total = len(meta)
    group_keys = meta[[split_col, label_col]].astype(str).agg("|".join, axis=1)
    positions = pd.Series(np.arange(total, dtype=int), index=meta.index)
    for _, group_positions in positions.groupby(group_keys, sort=False):
        group_values = group_positions.to_numpy(dtype=int)
        quota = int(round(len(group_values) * max_samples / total))
        quota = min(len(group_values), max(1, quota))
        selected.extend(rng.choice(group_values, size=quota, replace=False).tolist())

    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > max_samples:
        selected = np.sort(rng.choice(selected, size=max_samples, replace=False))
    elif len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(total, dtype=int), selected, assume_unique=False)
        fill = min(len(remaining), max_samples - len(selected))
        if fill > 0:
            selected = np.sort(np.concatenate([selected, rng.choice(remaining, size=fill, replace=False)]))
    return selected


def stratified_limit_positions(positions, labels, max_count, seed):
    positions = np.asarray(positions, dtype=int)
    if max_count is None or len(positions) <= max_count:
        return positions
    if max_count <= 0:
        raise ValueError("--max-train-samples must be positive.")

    rng = np.random.default_rng(seed)
    selected = []
    labels = np.asarray(labels).astype(int)
    for class_id in np.unique(labels[positions]):
        class_positions = positions[labels[positions] == class_id]
        quota = int(round(len(class_positions) * max_count / len(positions)))
        quota = min(len(class_positions), max(1, quota))
        selected.extend(rng.choice(class_positions, size=quota, replace=False).tolist())

    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > max_count:
        selected = np.sort(rng.choice(selected, size=max_count, replace=False))
    elif len(selected) < max_count:
        remaining = np.setdiff1d(positions, selected, assume_unique=False)
        fill = min(len(remaining), max_count - len(selected))
        if fill > 0:
            selected = np.sort(np.concatenate([selected, rng.choice(remaining, size=fill, replace=False)]))
    return selected


def simple_kmeans(x, k, seed, max_iter=100):
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError("simple_kmeans requires len(x) >= k")
    init_idx = rng.choice(len(x), size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, len(x))]
    return centers.astype(np.float32)


def classwise_prototypes(x_train, y_train, prototypes_per_class, seed):
    centers = []
    for class_id in [0, 1]:
        class_x = x_train[y_train == class_id]
        if len(class_x) == 0:
            raise ValueError(f"No training rows for class {class_id}; cannot initialize class-wise DST prototypes.")

        k = int(prototypes_per_class)
        if k <= 0:
            raise ValueError("--prototypes-per-class must be positive.")
        if k == 1:
            class_centers = class_x.mean(axis=0, keepdims=True)
        elif len(class_x) >= k:
            class_centers = simple_kmeans(class_x, k, seed + class_id)
        else:
            repeat_count = int(np.ceil(k / len(class_x)))
            class_centers = np.tile(class_x, (repeat_count, 1))[:k]
        centers.append(class_centers.astype(np.float32))
    return torch.tensor(np.stack(centers, axis=0), dtype=torch.float32)


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_numpy(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(score)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= threshold).astype(int)
    auc = roc_auc_numpy(y_true, score)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    bacc = np.nanmean([sensitivity, specificity])
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auc": auc,
        "bacc_at_threshold": float(bacc),
        "sensitivity_at_threshold": float(sensitivity),
        "specificity_at_threshold": float(specificity),
        "pred_pos_at_threshold": int(y_pred.sum()),
        "threshold": float(threshold),
    }


def grouped_frame(pred_df, label_col, score_col, group_cols, score_agg):
    missing = [col for col in group_cols if col not in pred_df.columns]
    if missing:
        return None
    return (
        pred_df.groupby(group_cols, dropna=False)
        .agg({label_col: "max", score_col: score_agg})
        .reset_index()
    )


def attach_prediction_columns(pred_df, label_col, score_col, group_cols, score_agg, threshold):
    missing = [col for col in group_cols if col not in pred_df.columns]
    if missing:
        raise KeyError(f"Missing column(s) for grouped predictions: {missing}")

    result_df = pred_df.copy()
    result_df["image_prediction_prob"] = result_df[score_col]
    result_df["patient_prediction_prob"] = result_df.groupby(group_cols, dropna=False)[
        "image_prediction_prob"
    ].transform(score_agg)
    result_df["prediction_prob"] = result_df["patient_prediction_prob"]
    result_df["prediction_label"] = (result_df["prediction_prob"] >= threshold).astype(int)
    result_df["prediction_group_cols"] = ",".join(group_cols)
    result_df["prediction_score_agg"] = score_agg
    result_df["prediction_threshold"] = float(threshold)
    result_df["prediction_image_score_col"] = "image_prediction_prob"
    return result_df


def evaluate_predictions(pred_df, args):
    rows = []
    group_cols = parse_group_cols(args.group_cols)
    for split_name in ["train", "val", "test", "all"]:
        if split_name == "all":
            split_df = pred_df
        else:
            split_df = pred_df[pred_df["dst_split"] == split_name]
        if len(split_df) == 0:
            continue

        image_metrics = binary_metrics(split_df[args.label], split_df["image_prediction_prob"], args.threshold)
        image_metrics.update({"split": split_name, "grain": "image"})
        rows.append(image_metrics)

        grouped = grouped_frame(split_df, args.label, "image_prediction_prob", group_cols, args.score_agg)
        if grouped is not None and len(grouped) > 0:
            group_metrics = binary_metrics(grouped[args.label], grouped["image_prediction_prob"], args.threshold)
            group_metrics.update({"split": split_name, "grain": ",".join(group_cols), "score_agg": args.score_agg})
            rows.append(group_metrics)

    return pd.DataFrame(rows)


class EmbeddingDataset(Dataset):
    def __init__(self, features, labels, positions):
        self.features = features
        self.labels = np.asarray(labels).astype(np.int64)
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        row = int(self.positions[index])
        feature = torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))
        label = int(self.labels[row])
        return feature, label, row


def make_loader(features, labels, positions, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EmbeddingDataset(features, labels, positions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def maybe_progress(iterable, args, desc):
    if not getattr(args, "show_progress", False) or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=len(iterable), leave=False, dynamic_ncols=True)


def pignistic(mass, num_classes):
    class_mass = mass[..., :num_classes]
    omega = mass[..., num_classes]
    prob = class_mass + omega.unsqueeze(-1) / float(num_classes)
    return prob, omega


def dst_activation_init_gamma(gamma_init):
    return np.sqrt(max(float(gamma_init), 1e-6))


def dst_activation_init_alpha(alpha_init):
    return float(alpha_init)


class PrototypeDSTHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes=2,
        prototypes_per_class=10,
        topk=3,
        gamma_init=1.0,
        alpha_init=0.0,
        normalize=True,
        dropout=0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("PrototypeDSTHead currently supports binary classification only.")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive.")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.n_prototypes = self.num_classes * self.prototypes_per_class
        self.topk = int(topk)
        self.normalize = bool(normalize)
        self.gamma_init = float(gamma_init)
        self.alpha_init = float(alpha_init)
        self.dropout = nn.Dropout(p=float(dropout))

        self.ds_module = Dempster_Shafer_Module(
            n_feature_maps=self.feature_dim,
            n_classes=self.num_classes,
            n_prototypes=self.n_prototypes,
        )
        self.ds_module.ds1_activate = DistanceActivation_layer(
            n_prototypes=self.n_prototypes,
            init_alpha=dst_activation_init_alpha(self.alpha_init),
            init_gamma=dst_activation_init_gamma(self.gamma_init),
        )
        self.reset_parameters()

    @property
    def prototypes(self):
        return self.ds_module.ds1.w.view(
            self.num_classes,
            self.prototypes_per_class,
            self.feature_dim,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ds_module.ds1.w)

    def initialize_prototypes(self, prototypes):
        expected_shape = (self.num_classes, self.prototypes_per_class, self.feature_dim)
        if tuple(prototypes.shape) != expected_shape:
            raise ValueError(f"Expected prototype tensor shape {expected_shape}, got {tuple(prototypes.shape)}.")
        flat_prototypes = prototypes.reshape(self.n_prototypes, self.feature_dim)
        with torch.no_grad():
            self.ds_module.ds1.w.copy_(
                flat_prototypes.to(device=self.ds_module.ds1.w.device, dtype=self.ds_module.ds1.w.dtype)
            )

    def _compute_distances(self, features):
        prototypes = self.ds_module.ds1.w
        if self.normalize:
            features = F.normalize(features, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        return (features[:, None, :] - prototypes[None, :, :]).pow(2).sum(dim=-1)

    def _reshape_prototypes(self, tensor):
        return tensor.view(tensor.shape[0], self.num_classes, self.prototypes_per_class)

    def forward(self, features):
        if features.dim() != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features [B, {self.feature_dim}], got {tuple(features.shape)}.")

        features = self.dropout(features.float())
        distances = self._compute_distances(features)
        ed_ac = self.ds_module.ds1_activate(distances)
        mass_prototypes = self.ds_module.ds2(ed_ac)
        mass_prototypes_omega = self.ds_module.ds2_omega(mass_prototypes)
        mass_dempster = self.ds_module.ds3_dempster(mass_prototypes_omega)
        mass = self.ds_module.ds3_normalize(mass_dempster)
        prob, uncertainty = pignistic(mass, self.num_classes)

        prototype_evidence = torch.zeros(
            features.size(0),
            self.num_classes,
            self.prototypes_per_class,
            device=features.device,
            dtype=features.dtype,
        )
        for class_idx in range(self.num_classes):
            start = class_idx * self.prototypes_per_class
            end = start + self.prototypes_per_class
            prototype_evidence[:, class_idx, :] = mass_prototypes[:, start:end, class_idx]

        distances_by_class = self._reshape_prototypes(distances)
        similarity = self._reshape_prototypes(ed_ac)
        out = {
            "prob": prob,
            "uncertainty": uncertainty,
            "dst_mass": mass,
            "prototype_distances": distances_by_class,
            "prototype_similarity": similarity,
            "prototype_evidence": prototype_evidence,
            "prototype_mass": prototype_evidence,
        }

        if self.topk > 0:
            topk = min(self.topk, self.prototypes_per_class)
            top_evidence, top_idx = torch.topk(prototype_evidence, k=topk, dim=-1)
            out.update(
                {
                    "topk_proto_idx": top_idx,
                    "topk_proto_evidence": top_evidence,
                    "topk_proto_mass": top_evidence,
                    "topk_proto_similarity": torch.gather(similarity, dim=-1, index=top_idx),
                    "topk_proto_distances": torch.gather(distances_by_class, dim=-1, index=top_idx),
                }
            )
        return out


class EmbeddingPrototypeDST(nn.Module):
    def __init__(self, feature_dim, prototypes_per_class, topk, gamma_init, alpha_init, normalize, dropout):
        super().__init__()
        self.dst_head = PrototypeDSTHead(
            feature_dim=feature_dim,
            num_classes=2,
            prototypes_per_class=prototypes_per_class,
            topk=topk,
            gamma_init=gamma_init,
            alpha_init=alpha_init,
            normalize=normalize,
            dropout=dropout,
        )

    def initialize_prototypes(self, prototypes):
        self.dst_head.initialize_prototypes(prototypes)

    def forward(self, features):
        return self.dst_head(features)


class PrototypeDSTNLLLoss(nn.Module):
    def __init__(self, class_weights=None, eps=1e-10):
        super().__init__()
        self.eps = float(eps)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, head_output, target):
        prob = head_output["prob"].clamp(min=self.eps, max=1.0)
        target_indices = target.long().to(prob.device)
        weight = None if self.class_weights is None else self.class_weights.to(device=prob.device, dtype=prob.dtype)
        return F.nll_loss(torch.log(prob), target_indices, weight=weight)


def compute_class_weights(y_train, args):
    mode = args.class_weight_mode
    info = {
        "mode": mode,
        "n_neg": int((y_train == 0).sum()),
        "n_pos": int((y_train == 1).sum()),
        "effective_beta": float(args.effective_beta),
        "class_weights": None,
    }
    if mode == "none":
        return None, info
    if info["n_neg"] <= 0 or info["n_pos"] <= 0:
        print("[WARN] Training split has a missing class; using unweighted DST NLL.")
        return None, info
    if mode == "inverse":
        weights = np.array([1.0, info["n_neg"] / max(info["n_pos"], 1)], dtype=np.float32)
    else:
        beta = float(args.effective_beta)
        if beta < 0.0 or beta >= 1.0:
            raise ValueError("--effective-beta must be in [0, 1).")
        if beta == 0.0:
            weights = np.array([1.0, 1.0], dtype=np.float32)
        else:
            effective_neg = (1.0 - np.power(beta, info["n_neg"])) / (1.0 - beta)
            effective_pos = (1.0 - np.power(beta, info["n_pos"])) / (1.0 - beta)
            weights = np.array([1.0 / effective_neg, 1.0 / effective_pos], dtype=np.float32)
            weights = weights / np.mean(weights)
    info["class_weights"] = [float(weights[0]), float(weights[1])]
    return torch.tensor(weights, dtype=torch.float32), info


def _class_balanced_mean(values, labels, num_classes, balance_classes):
    if not balance_classes:
        return values.mean()

    class_means = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.any():
            class_means.append(values[class_mask].mean())
    if not class_means:
        return values.mean()
    return torch.stack(class_means).mean()


def _prototype_diversity_loss(model, margin):
    prototypes = model.dst_head.prototypes
    if model.dst_head.normalize:
        prototypes = F.normalize(prototypes, p=2, dim=-1)

    per_class_losses = []
    for class_idx in range(prototypes.shape[0]):
        if prototypes.shape[1] < 2:
            continue
        pairwise_distance = torch.pdist(prototypes[class_idx], p=2)
        per_class_losses.append(F.relu(float(margin) - pairwise_distance).pow(2).mean())

    if not per_class_losses:
        return prototypes.new_zeros(())
    return torch.stack(per_class_losses).mean()


def prototype_regularization_loss(model, head_output, labels, args):
    distances = head_output["prototype_distances"].clamp_min(0.0)
    labels = labels.long()
    num_classes = int(distances.shape[1])
    if torch.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"Prototype DST labels must be in [0, {num_classes - 1}].")

    batch_indices = torch.arange(labels.shape[0], device=labels.device)
    margin = float(args.proto_margin)
    balance_classes = str(args.proto_balance_classes).lower() == "y"

    own_distances = distances[batch_indices, labels]
    nearest_own = own_distances.min(dim=-1).values.clamp_min(1e-12).sqrt()
    attract_loss = _class_balanced_mean(nearest_own, labels, num_classes, balance_classes)

    other_mask = F.one_hot(labels, num_classes=num_classes).bool().unsqueeze(-1)
    other_distances = distances.masked_fill(other_mask, float("inf")).flatten(start_dim=1)
    nearest_other = other_distances.min(dim=1).values.clamp_min(1e-12).sqrt()
    separation_loss = _class_balanced_mean(
        F.relu(margin - nearest_other).pow(2),
        labels,
        num_classes,
        balance_classes,
    )

    diversity_loss = _prototype_diversity_loss(model, margin)
    raw = (
        float(args.proto_attract_weight) * attract_loss
        + float(args.proto_separation_weight) * separation_loss
        + float(args.proto_diversity_weight) * diversity_loss
    )
    total = float(args.proto_loss_weight) * raw
    return total, {
        "proto_loss": float(total.detach().cpu()),
        "proto_loss_raw": float(raw.detach().cpu()),
        "proto_attract_loss": float(attract_loss.detach().cpu()),
        "proto_separation_loss": float(separation_loss.detach().cpu()),
        "proto_diversity_loss": float(diversity_loss.detach().cpu()),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_n = 0
    proto_totals = {
        "proto_loss": 0.0,
        "proto_loss_raw": 0.0,
        "proto_attract_loss": 0.0,
        "proto_separation_loss": 0.0,
        "proto_diversity_loss": 0.0,
    }
    for features, labels, _ in maybe_progress(loader, args, "train"):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        head_output = model(features)
        task_loss = criterion(head_output, labels)
        proto_loss, proto_stats = prototype_regularization_loss(model, head_output, labels, args)
        loss = task_loss + proto_loss
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_n += batch_size
        for key in proto_totals:
            proto_totals[key] += proto_stats[key] * batch_size
    stats = {key: value / max(total_n, 1) for key, value in proto_totals.items()}
    return total_loss / max(total_n, 1), stats


def predict_to_frame(meta, features, labels, positions, model, criterion, args, device):
    loader = make_loader(
        features,
        labels,
        positions,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model.eval()
    row_ids = []
    prob_rows = []
    mass_rows = []
    uncertainty_rows = []
    loss_total = 0.0
    loss_n = 0

    with torch.no_grad():
        for batch_features, batch_labels, batch_rows in maybe_progress(loader, args, "predict"):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            head_output = model(batch_features)
            loss = criterion(head_output, batch_labels)
            batch_size = int(batch_labels.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            loss_n += batch_size
            row_ids.append(batch_rows.numpy())
            prob_rows.append(head_output["prob"].detach().cpu().numpy())
            mass_rows.append(head_output["dst_mass"].detach().cpu().numpy())
            uncertainty_rows.append(head_output["uncertainty"].detach().cpu().numpy())

    if not row_ids:
        return meta.iloc[[]].copy(), float("nan")

    row_ids = np.concatenate(row_ids).astype(int)
    order = np.argsort(row_ids)
    row_ids = row_ids[order]
    probs_np = np.concatenate(prob_rows, axis=0)[order]
    mass_np = np.concatenate(mass_rows, axis=0)[order]
    uncertainty_np = np.concatenate(uncertainty_rows, axis=0)[order]

    pred_df = meta.iloc[row_ids].copy().reset_index(drop=True)
    pred_df["dst_mass_0"] = mass_np[:, 0]
    pred_df["dst_mass_1"] = mass_np[:, 1]
    pred_df["dst_mass_omega"] = mass_np[:, 2]
    pred_df["dst_probability_0"] = probs_np[:, 0]
    pred_df["dst_probability_1"] = probs_np[:, 1]
    pred_df["probability_1"] = pred_df["dst_probability_1"]
    pred_df["dst_uncertainty"] = uncertainty_np
    pred_df = attach_prediction_columns(
        pred_df,
        args.label,
        "dst_probability_1",
        parse_group_cols(args.group_cols),
        args.score_agg,
        args.threshold,
    )
    return pred_df, loss_total / max(loss_n, 1)


def select_eval_metric(metrics_df, eval_split, args, eval_loss):
    if args.best_metric == "loss":
        return -float(eval_loss), {
            "split": eval_split,
            "grain": "loss",
            "auc": float("nan"),
            "bacc_at_threshold": float("nan"),
            "pred_pos_at_threshold": float("nan"),
        }

    group_grain = ",".join(parse_group_cols(args.group_cols))
    split_rows = metrics_df[metrics_df["split"] == eval_split]
    if split_rows.empty:
        split_rows = metrics_df[metrics_df["split"] == "all"]
    if split_rows.empty:
        return float("-inf"), {}

    group_rows = split_rows[split_rows["grain"] == group_grain]
    metric_row = (group_rows if not group_rows.empty else split_rows).iloc[0].to_dict()
    key = "auc" if args.best_metric == "auc" else "bacc_at_threshold"
    value = float(metric_row.get(key, float("nan")))
    if not np.isfinite(value):
        value = float("-inf")
    return value, metric_row


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_gamma_init(args):
    if args.temperature is None or str(args.temperature).lower() == "auto":
        return float(args.dst_gamma_init)
    temperature = float(args.temperature)
    if temperature <= 0:
        raise ValueError("--temperature must be positive or 'auto'.")
    return 1.0 / temperature


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.prototypes_per_class <= 0:
        raise ValueError("--prototypes-per-class must be positive.")
    if args.prototype_topk < 0:
        raise ValueError("--prototype-topk must be non-negative.")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative.")
    if args.n_folds <= 0:
        raise ValueError("--n-folds must be positive.")
    for name in ["proto_attract_weight", "proto_separation_weight", "proto_diversity_weight", "proto_loss_weight"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def run_single(args):
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args)
    gamma_init = resolve_gamma_init(args)

    embedding_dir = resolve_path(args.embedding_dir)
    embedding_path = resolve_path(args.embeddings) if args.embeddings else embedding_dir / "embeddings.npy"
    metadata_path = resolve_path(args.metadata) if args.metadata else embedding_dir / "metadata.csv"
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embedding_path, mmap_mode="r")
    meta = pd.read_csv(metadata_path)
    if len(meta) != len(embeddings):
        raise ValueError(f"metadata rows ({len(meta)}) != embeddings rows ({len(embeddings)})")
    if args.label not in meta.columns:
        raise ValueError(f"metadata is missing label column: {args.label}")
    meta = meta.copy().reset_index(drop=True)
    if getattr(args, "_cv_mode", False):
        meta["dst_split"], cv_fold_source = assign_cv_splits(meta, args)
        args._cv_fold_source = cv_fold_source
        print(f"[CV] split source: {cv_fold_source}")
    else:
        meta["dst_split"] = assign_splits(meta, args)
        args._cv_fold_source = None
    original_num_rows = int(len(meta))
    sample_positions = stratified_sample_positions(meta, args.label, "dst_split", args.max_samples, args.seed)
    if len(sample_positions) < len(meta):
        embeddings = embeddings[sample_positions]
        meta = meta.iloc[sample_positions].copy().reset_index(drop=True)

    labels = meta[args.label].astype(int).to_numpy()
    features = np.asarray(embeddings, dtype=np.float32)
    if not args.no_normalize:
        features = normalize_features(features).astype(np.float32, copy=False)

    train_positions = np.flatnonzero(meta["dst_split"].values == "train")
    if len(train_positions) == 0:
        raise ValueError("No training rows after split assignment.")
    train_positions = stratified_limit_positions(train_positions, labels, args.max_train_samples, args.seed)
    y_train = labels[train_positions]
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both classes for Prototype-DST.")

    val_positions = np.flatnonzero(meta["dst_split"].values == "val")
    test_positions = np.flatnonzero(meta["dst_split"].values == "test")
    if len(val_positions) > 0:
        eval_positions = val_positions
        eval_split = "val"
    elif len(test_positions) > 0:
        eval_positions = test_positions
        eval_split = "test"
        print("[WARN] No validation rows; early stopping will use test rows.")
    else:
        eval_positions = train_positions
        eval_split = "train"
        print("[WARN] No validation/test rows; early stopping will use train rows.")

    feature_dim = int(features.shape[1])
    model = EmbeddingPrototypeDST(
        feature_dim=feature_dim,
        prototypes_per_class=args.prototypes_per_class,
        topk=args.prototype_topk,
        gamma_init=gamma_init,
        alpha_init=args.dst_alpha_init,
        normalize=not bool(args.no_normalize),
        dropout=args.dropout,
    ).to(device)

    prototype_initialized_from = "random"
    if args.prototype_init == "kmeans":
        print(
            "Initializing DST prototypes with KMeans "
            f"(train_rows={len(train_positions)}, k={args.prototypes_per_class} per class)..."
        )
        prototypes = classwise_prototypes(features[train_positions], labels[train_positions], args.prototypes_per_class, args.seed)
        model.initialize_prototypes(prototypes.to(device))
        prototype_initialized_from = "train_embedding_kmeans"
        print(f"Initialized DST prototypes from train embeddings: {tuple(prototypes.shape)}")

    class_weights, class_weight_info = compute_class_weights(y_train, args)
    if class_weight_info["class_weights"] is not None:
        print(
            "Class weights "
            f"({class_weight_info['mode']}): neg={class_weight_info['class_weights'][0]:.6f}, "
            f"pos={class_weight_info['class_weights'][1]:.6f}"
        )

    criterion = PrototypeDSTNLLLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(features, labels, train_positions, args.batch_size, True, args.seed, args.num_workers)

    best_path = output_dir / "dst_best.pt"
    history = []
    best_metric = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0

    print(f"[device] {device}")
    print(f"[split counts] {meta['dst_split'].value_counts().to_dict()}")
    print(f"[train rows] {len(train_positions)}  [eval rows] {len(eval_positions)} ({eval_split})")
    print(f"[DST] trainable Prototype-DST head, gamma_init={gamma_init:g}, alpha_init={args.dst_alpha_init:g}")

    for epoch in range(args.epochs):
        train_loss, train_proto_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, args)
        eval_df, eval_loss = predict_to_frame(meta, features, labels, eval_positions, model, criterion, args, device)
        eval_metrics = evaluate_predictions(eval_df, args)
        metric_value, metric_row = select_eval_metric(eval_metrics, eval_split, args, eval_loss)
        improved = metric_value > best_metric + 1e-8
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "feature_dim": feature_dim,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_metric_name": args.best_metric,
                    "eval_split": eval_split,
                    "args": vars(args),
                    "gamma_init": gamma_init,
                    "class_weight_info": class_weight_info,
                    "prototype_initialized_from": prototype_initialized_from,
                    "history": history,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "eval_split": eval_split,
            "eval_loss": float(eval_loss),
            "best_metric_name": args.best_metric,
            "eval_metric": float(metric_value),
            "best_metric": float(best_metric),
            "improved": bool(improved),
        }
        history_row.update(train_proto_stats)
        for key in ("grain", "auc", "bacc_at_threshold", "sensitivity_at_threshold", "specificity_at_threshold", "pred_pos_at_threshold"):
            if key in metric_row:
                history_row[f"eval_{key}"] = metric_row[key]
        history.append(history_row)

        auc_text = metric_row.get("auc", float("nan")) if metric_row else float("nan")
        bacc_text = metric_row.get("bacc_at_threshold", float("nan")) if metric_row else float("nan")
        pred_pos_text = metric_row.get("pred_pos_at_threshold", float("nan")) if metric_row else float("nan")
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} {eval_split}_loss={eval_loss:.4f} "
            f"AUC={auc_text:.4f} bACC@{args.threshold:g}={bacc_text:.4f} "
            f"Pred_Pos@{args.threshold:g}={pred_pos_text} "
            f"proto={train_proto_stats['proto_loss']:.4f} "
            f"best_{args.best_metric}={best_metric:.4f}"
        )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs (patience={args.patience}).")
            break

    if best_epoch < 0:
        torch.save(
            {
                "model": model.state_dict(),
                "feature_dim": feature_dim,
                "epoch": args.epochs - 1,
                "best_metric": best_metric,
                "best_metric_name": args.best_metric,
                "eval_split": eval_split,
                "args": vars(args),
                "gamma_init": gamma_init,
                "class_weight_info": class_weight_info,
                "prototype_initialized_from": prototype_initialized_from,
                "history": history,
            },
            best_path,
        )
    checkpoint = torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    all_positions = np.arange(len(meta), dtype=int)
    pred_df, _ = predict_to_frame(meta, features, labels, all_positions, model, criterion, args, device)
    metrics_df = evaluate_predictions(pred_df, args)

    pred_path = output_dir / "dst_all_predictions.csv"
    metrics_path = output_dir / "dst_metrics.csv"
    manifest_path = output_dir / "dst_manifest.json"
    pred_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    history_path, loss_curve_path = save_training_history(history, output_dir, "dst")

    split_counts = {str(k): int(v) for k, v in meta["dst_split"].value_counts().to_dict().items()}
    label_counts = {str(k): int(v) for k, v in pd.Series(labels).value_counts().to_dict().items()}
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "requested_output_dir": getattr(args, "_requested_output_dir", str(output_dir)),
        "output_dir": str(output_dir),
        "run_id": getattr(args, "_resolved_run_id", None),
        "timestamped_output_dir": not bool(getattr(args, "no_timestamp", False)),
        "label": args.label,
        "original_num_rows": original_num_rows,
        "num_rows": int(len(meta)),
        "feature_dim": feature_dim,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "effective_device": str(device),
        "split_counts": split_counts,
        "cv_mode": bool(getattr(args, "_cv_mode", False)),
        "cv_fold": int(args.val_fold) if getattr(args, "_cv_mode", False) else None,
        "cv_fold_source": getattr(args, "_cv_fold_source", None),
        "cv_group_col": args.cv_group_col,
        "label_counts": label_counts,
        "prototypes_per_class": int(args.prototypes_per_class),
        "prototype_topk": int(args.prototype_topk),
        "prototype_init": args.prototype_init,
        "prototype_initialized_from": prototype_initialized_from,
        "dst_gamma_init": float(gamma_init),
        "dst_alpha_init": float(args.dst_alpha_init),
        "class_weight_info": class_weight_info,
        "proto_attract_weight": float(args.proto_attract_weight),
        "proto_separation_weight": float(args.proto_separation_weight),
        "proto_diversity_weight": float(args.proto_diversity_weight),
        "proto_loss_weight": float(args.proto_loss_weight),
        "proto_margin": float(args.proto_margin),
        "proto_balance_classes": args.proto_balance_classes,
        "threshold": float(args.threshold),
        "normalize": not bool(args.no_normalize),
        "group_cols": parse_group_cols(args.group_cols),
        "score_agg": args.score_agg,
        "best_metric_name": args.best_metric,
        "best_metric_value": float(checkpoint.get("best_metric", best_metric)),
        "best_epoch": int(checkpoint.get("epoch", best_epoch)) + 1,
        "eval_split": eval_split,
        "history": history,
        "prediction_file": str(pred_path),
        "metrics_file": str(metrics_path),
        "history_file": str(history_path),
        "loss_curve_file": str(loss_curve_path) if loss_curve_path is not None else None,
        "checkpoint_file": str(best_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  predictions: {pred_path}")
    print(f"  metrics:     {metrics_path}")
    print(f"  history:     {history_path}")
    if loss_curve_path is not None:
        print(f"  loss curve:  {loss_curve_path}")
    print(f"  checkpoint:  {best_path}")
    print(f"  manifest:    {manifest_path}")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    return metrics_df, manifest


def write_cv_summary(base_output_dir, fold_metrics):
    base_output_dir.mkdir(parents=True, exist_ok=True)
    cv_metrics = pd.concat(fold_metrics, ignore_index=True)
    cv_metrics_path = base_output_dir / "cv_metrics.csv"
    cv_metrics.to_csv(cv_metrics_path, index=False)

    group_cols = [col for col in ["split", "grain", "score_agg"] if col in cv_metrics.columns]
    numeric_cols = [
        col
        for col in cv_metrics.select_dtypes(include=[np.number]).columns
        if col != "cv_fold"
    ]
    summary_path = base_output_dir / "cv_metrics_summary.csv"
    if group_cols and numeric_cols:
        summary = cv_metrics.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(summary_path, index=False)
    else:
        summary = pd.DataFrame()
        summary.to_csv(summary_path, index=False)

    print("\nCross-validation done.")
    print(f"  cv metrics:  {cv_metrics_path}")
    print(f"  cv summary:  {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return cv_metrics, summary


def main():
    args = build_parser().parse_args()
    args._requested_output_dir = args.output_dir
    resolved_output_dir, resolved_run_id = timestamped_output_dir(args.output_dir, args)
    args.output_dir = str(resolved_output_dir)
    args._resolved_run_id = resolved_run_id
    if args.n_folds <= 1:
        run_single(args)
        return

    base_output_dir = resolve_path(args.output_dir)
    fold_metrics = []
    for fold in range(int(args.fold_start), int(args.fold_start) + int(args.n_folds)):
        fold_args = copy.deepcopy(args)
        fold_args._cv_n_folds = args.n_folds
        fold_args.n_folds = 1
        fold_args._cv_mode = True
        fold_args.split_mode = "fold"
        fold_args.val_fold = fold
        fold_args.output_dir = str(base_output_dir / f"fold_{fold}")
        print(f"\n[CV] Fold {fold} / {args.fold_start + args.n_folds - 1}")
        metrics_df, _ = run_single(fold_args)
        metrics_df = metrics_df.copy()
        metrics_df.insert(0, "cv_fold", fold)
        fold_metrics.append(metrics_df)

    write_cv_summary(base_output_dir, fold_metrics)


if __name__ == "__main__":
    main()
