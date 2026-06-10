"""
Run a lightweight Dempster-Shafer-style prototype classifier on cached embeddings.

Input bundle:
  - embeddings.npy
  - metadata.csv

Outputs:
  - dst_all_predictions.csv
  - dst_metrics.csv
  - dst_manifest.json
  - dst_prototypes.npz
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Run prototype DST from cached Mammo-CLIP embeddings.")
    parser.add_argument("--embedding-dir", default="embeddings/origin_finetuned_encoder")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.npy. Overrides --embedding-dir.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.csv. Overrides --embedding-dir.")
    parser.add_argument("--output-dir", default="dst_results/origin_finetuned_dst")
    parser.add_argument("--label", default="cancer")
    parser.add_argument("--prototypes-per-class", type=int, default=10)
    parser.add_argument(
        "--temperature",
        default="auto",
        help="Positive float or 'auto'. Auto chooses temperature on validation BACC@threshold.",
    )
    parser.add_argument(
        "--temperature-grid",
        default="0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100",
        help="Comma-separated candidate temperatures used when --temperature auto.",
    )
    parser.add_argument("--uncertainty-strength", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization before fitting DST.")
    parser.add_argument("--split-mode", choices=["auto", "metadata", "cohort", "fold"], default="auto")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--cohort-col", default="cohort_num")
    parser.add_argument("--train-cohorts", default="1-8")
    parser.add_argument("--test-cohorts", default="9-10")
    parser.add_argument("--holdout-val-percent", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-cols", default="patient_id")
    parser.add_argument("--score-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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
        class_indices = indices[labels == class_id]
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


def classwise_prototypes(x_train, y_train, prototypes_per_class, seed):
    prototypes = []
    prototype_labels = []
    for class_id in [0, 1]:
        class_x = x_train[y_train == class_id]
        if len(class_x) == 0:
            raise ValueError(f"No training rows for class {class_id}; cannot fit class-wise DST prototypes.")

        k = int(prototypes_per_class)
        if k <= 0:
            raise ValueError("--prototypes-per-class must be positive.")
        if k == 1:
            centers = class_x.mean(axis=0, keepdims=True)
        elif len(class_x) >= k:
            centers = simple_kmeans(class_x, k, seed + class_id)
        else:
            repeat_count = int(np.ceil(k / len(class_x)))
            centers = np.tile(class_x, (repeat_count, 1))[:k]
        prototypes.append(centers.astype(np.float32))
        prototype_labels.extend([class_id] * len(centers))

    return np.vstack(prototypes).astype(np.float32), np.asarray(prototype_labels, dtype=np.int64)


def dst_predict(x, prototypes, prototype_labels, temperature, uncertainty_strength, batch_size):
    out = {
        "mass_0": np.zeros(len(x), dtype=np.float32),
        "mass_1": np.zeros(len(x), dtype=np.float32),
        "uncertainty": np.zeros(len(x), dtype=np.float32),
        "probability_1": np.zeros(len(x), dtype=np.float32),
    }

    temp = max(float(temperature), 1e-12)
    uncertainty_strength = max(float(uncertainty_strength), 1e-12)
    proto0 = prototypes[prototype_labels == 0]
    proto1 = prototypes[prototype_labels == 1]

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = np.asarray(x[start:end], dtype=np.float32)
        dist0 = ((xb[:, None, :] - proto0[None, :, :]) ** 2).sum(axis=2)
        dist1 = ((xb[:, None, :] - proto1[None, :, :]) ** 2).sum(axis=2)
        support0 = np.exp(-dist0 / temp).mean(axis=1)
        support1 = np.exp(-dist1 / temp).mean(axis=1)
        denom = support0 + support1 + uncertainty_strength
        mass0 = support0 / denom
        mass1 = support1 / denom
        uncertainty = uncertainty_strength / denom
        probability_1 = mass1 + 0.5 * uncertainty

        out["mass_0"][start:end] = mass0
        out["mass_1"][start:end] = mass1
        out["uncertainty"][start:end] = uncertainty
        out["probability_1"][start:end] = probability_1
    return out


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
    grouped = pred_df.groupby(group_cols, dropna=False).agg({label_col: "max", score_col: score_agg}).reset_index()
    return grouped


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


def tune_temperature(x_val, y_val, prototypes, prototype_labels, args):
    candidates = [float(item.strip()) for item in args.temperature_grid.split(",") if item.strip()]
    best_temperature = candidates[0]
    best_bacc = -np.inf
    best_auc = -np.inf
    for temp in candidates:
        pred = dst_predict(
            x_val,
            prototypes,
            prototype_labels,
            temp,
            args.uncertainty_strength,
            args.batch_size,
        )
        metrics = binary_metrics(y_val, pred["probability_1"], args.threshold)
        bacc = metrics["bacc_at_threshold"]
        auc = metrics["auc"] if np.isfinite(metrics["auc"]) else -np.inf
        if bacc > best_bacc or (np.isclose(bacc, best_bacc) and auc > best_auc):
            best_temperature = temp
            best_bacc = bacc
            best_auc = auc
    return best_temperature, best_bacc, best_auc


def main():
    args = build_parser().parse_args()
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

    meta = meta.copy()
    meta["dst_split"] = assign_splits(meta, args)
    label_values = meta[args.label].astype(int).values

    x_all = np.asarray(embeddings, dtype=np.float32)
    if not args.no_normalize:
        x_all = normalize_features(x_all)

    train_mask = meta["dst_split"].values == "train"
    if train_mask.sum() == 0:
        raise ValueError("No training rows after split assignment.")
    x_train = x_all[train_mask]
    y_train = label_values[train_mask]
    if args.max_train_samples is not None and len(x_train) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(len(x_train), size=args.max_train_samples, replace=False)
        x_train = x_train[sample_idx]
        y_train = y_train[sample_idx]

    prototypes, prototype_labels = classwise_prototypes(
        x_train,
        y_train,
        args.prototypes_per_class,
        args.seed,
    )

    if str(args.temperature).lower() == "auto":
        val_mask = meta["dst_split"].values == "val"
        if val_mask.sum() > 0 and len(np.unique(label_values[val_mask])) == 2:
            temperature, val_bacc, val_auc = tune_temperature(x_all[val_mask], label_values[val_mask], prototypes, prototype_labels, args)
            print(f"[temperature] auto selected {temperature:g} on val bACC={val_bacc:.4f}, AUC={val_auc:.4f}")
        else:
            temperature = 1.0
            print("[temperature] no usable validation split; using 1.0")
    else:
        temperature = float(args.temperature)

    pred = dst_predict(
        x_all,
        prototypes,
        prototype_labels,
        temperature,
        args.uncertainty_strength,
        args.batch_size,
    )

    pred_df = meta.copy()
    pred_df["dst_mass_0"] = pred["mass_0"]
    pred_df["dst_mass_1"] = pred["mass_1"]
    pred_df["dst_uncertainty"] = pred["uncertainty"]
    pred_df["dst_probability_1"] = pred["probability_1"]
    pred_df = attach_prediction_columns(
        pred_df,
        args.label,
        "dst_probability_1",
        parse_group_cols(args.group_cols),
        args.score_agg,
        args.threshold,
    )

    pred_path = output_dir / "dst_all_predictions.csv"
    metrics_path = output_dir / "dst_metrics.csv"
    prototypes_path = output_dir / "dst_prototypes.npz"
    manifest_path = output_dir / "dst_manifest.json"

    pred_df.to_csv(pred_path, index=False)
    metrics_df = evaluate_predictions(pred_df, args)
    metrics_df.to_csv(metrics_path, index=False)
    np.savez_compressed(prototypes_path, prototypes=prototypes, prototype_labels=prototype_labels)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "label": args.label,
        "num_rows": int(len(meta)),
        "feature_dim": int(embeddings.shape[1]),
        "split_counts": meta["dst_split"].value_counts().to_dict(),
        "prototypes_per_class": int(args.prototypes_per_class),
        "temperature": float(temperature),
        "uncertainty_strength": float(args.uncertainty_strength),
        "threshold": float(args.threshold),
        "normalize": not bool(args.no_normalize),
        "group_cols": parse_group_cols(args.group_cols),
        "score_agg": args.score_agg,
        "val_fold": int(args.val_fold),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  predictions: {pred_path}")
    print(f"  metrics:     {metrics_path}")
    print(f"  prototypes:  {prototypes_path}")
    print(f"  manifest:    {manifest_path}")
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
