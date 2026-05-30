"""
Prepare split CSVs for Mammo-CLIP finetuning.

Supports either:
1. Legacy split-column based partitioning
2. Cohort-driven partitioning via a CSV cohort column such as `cohort_num`
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_int_spec(spec):
    values = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid range '{chunk}': start must be <= end.")
            values.update(range(start, end + 1))
        else:
            values.add(int(chunk))
    if not values:
        raise ValueError("The cohort specification resolved to an empty set.")
    return values


def backup_existing_split(df):
    if "split" not in df.columns:
        return None

    for candidate in ("original_split", "input_split", "legacy_split"):
        if candidate not in df.columns:
            df[candidate] = df["split"]
            return candidate

    raise ValueError(
        "Input CSV already contains 'split', 'original_split', 'input_split', and 'legacy_split'. "
        "Refusing to overwrite any existing split history columns."
    )


def build_base_split(df, args):
    if args.split_mode == "cohort":
        if args.cohort_col not in df.columns:
            raise ValueError(f"Missing cohort column '{args.cohort_col}' in CSV.")

        train_cohorts = parse_int_spec(args.train_cohorts)
        test_cohorts = parse_int_spec(args.test_cohorts)
        overlap = sorted(train_cohorts & test_cohorts)
        if overlap:
            raise ValueError(f"Train/test cohort sets overlap: {overlap}")

        cohort_values = pd.to_numeric(df[args.cohort_col], errors="coerce")
        if cohort_values.isna().any():
            bad_count = int(cohort_values.isna().sum())
            raise ValueError(f"Found {bad_count} rows with non-numeric cohort values in '{args.cohort_col}'.")

        df = df.copy()
        df[args.cohort_col] = cohort_values.astype(int)
        mask_train = df[args.cohort_col].isin(train_cohorts)
        mask_test = df[args.cohort_col].isin(test_cohorts)
        mask_unassigned = ~(mask_train | mask_test)
        if mask_unassigned.any():
            bad_cohorts = sorted(df.loc[mask_unassigned, args.cohort_col].unique().tolist())
            raise ValueError(
                f"Found cohort values not covered by train/test specs: {bad_cohorts}. "
                "Update --train-cohorts/--test-cohorts before continuing."
            )

        df["split"] = "train"
        df.loc[mask_test, "split"] = "test"
        return df

    if "split" not in df.columns:
        raise ValueError("split-mode=split requires an existing 'split' column in the input CSV.")

    normalized_split = df["split"].astype(str).str.strip().str.lower()
    split_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    normalized_split = normalized_split.map(split_map)
    if normalized_split.isna().any():
        bad_values = sorted(df.loc[normalized_split.isna(), "split"].astype(str).unique().tolist())
        raise ValueError(f"Unsupported split values for split-mode=split: {bad_values}")

    df = df.copy()
    df["split"] = normalized_split
    df.loc[df["split"] == "val", "split"] = "train"
    return df


def _validate_percent(name, value):
    if value < 0 or value >= 100:
        raise ValueError(f"{name} must be in [0, 100). Got {value}.")


def assign_holdout_validation(train_df, val_percent, val_max_percent, seed, label_col):
    train_df = train_df.copy().reset_index(drop=True)
    train_df["split"] = "train"
    train_df["fold"] = 0

    if val_percent <= 0:
        return train_df

    patient_labels = (
        train_df.groupby("patient_id")
        .agg(patient_label=(label_col, "max"), sample_count=("patient_id", "size"))
        .reset_index()
    )
    n_patients = len(patient_labels)
    if n_patients < 2:
        raise ValueError("Need at least 2 unique training patients to create a holdout validation split.")

    if val_max_percent <= 0:
        val_max_percent = val_percent
    if val_max_percent < val_percent:
        raise ValueError("--holdout-val-max-percent must be 0 or >= --holdout-val-percent.")

    total_samples = len(train_df)
    target_sample_count = int(math.ceil(total_samples * val_percent / 100.0))
    target_sample_count = min(max(target_sample_count, 1), total_samples - 1)
    max_sample_count = int(math.ceil(total_samples * val_max_percent / 100.0))
    max_sample_count = min(max(max_sample_count, target_sample_count), total_samples - 1)

    shuffled = patient_labels.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_patient_ids = []
    val_sample_count = 0
    remaining_rows = []
    for row in shuffled.to_dict("records"):
        if val_sample_count < target_sample_count and len(val_patient_ids) < n_patients - 1:
            val_patient_ids.append(row["patient_id"])
            val_sample_count += int(row["sample_count"])
        else:
            remaining_rows.append(row)
    remaining = pd.DataFrame(remaining_rows)

    all_classes = set(patient_labels["patient_label"].tolist())
    val_classes = set(patient_labels.loc[patient_labels["patient_id"].isin(val_patient_ids), "patient_label"].tolist())

    if len(all_classes) > 1 and len(val_classes) < 2:
        while len(val_classes) < 2 and len(val_patient_ids) < n_patients - 1:
            missing_classes = all_classes - val_classes
            if remaining.empty:
                break
            candidate_rows = remaining[remaining["patient_label"].isin(missing_classes)].sort_values("sample_count")
            if candidate_rows.empty:
                break

            candidate_idx = None
            for idx, candidate_row in candidate_rows.iterrows():
                next_sample_count = val_sample_count + int(candidate_row["sample_count"])
                if next_sample_count <= max_sample_count:
                    candidate_idx = idx
                    break
            if candidate_idx is None:
                break

            candidate = remaining.loc[candidate_idx]
            val_patient_ids.append(candidate["patient_id"])
            val_classes.add(candidate["patient_label"])
            val_sample_count += int(candidate["sample_count"])
            remaining = remaining.drop(index=candidate_idx).reset_index(drop=True)

    val_mask = train_df["patient_id"].isin(val_patient_ids)
    if val_mask.all():
        raise ValueError("Holdout validation split consumed all training rows; reduce --holdout-val-percent.")

    train_df.loc[val_mask, "split"] = "val"

    val_df = train_df[val_mask]
    remaining_train_df = train_df[~val_mask]
    val_patient_count = val_df["patient_id"].nunique()
    val_sample_count = len(val_df)
    val_sample_percent = val_sample_count / max(len(train_df), 1) * 100.0
    print(
        f"Holdout validation split: {val_patient_count}/{n_patients} patients, "
        f"{val_sample_count}/{len(train_df)} samples ({val_sample_percent:.2f}%)"
    )
    print(f"  Train sample {label_col} distribution:\n{remaining_train_df[label_col].value_counts().sort_index()}")
    print(f"  Val sample {label_col} distribution:\n{val_df[label_col].value_counts().sort_index()}")
    if len(all_classes) > 1 and len(val_classes) < 2:
        print(
            "  Warning: validation split still has one class after reaching "
            f"holdout max percent ({val_max_percent:g}%)."
        )

    return train_df


def assign_train_pool_folds(
    train_df,
    n_folds,
    seed,
    label_col,
    holdout_val_percent=0.0,
    holdout_val_max_percent=0.0,
):
    train_df = train_df.copy().reset_index(drop=True)
    if n_folds == 0:
        return assign_holdout_validation(train_df, holdout_val_percent, holdout_val_max_percent, seed, label_col)

    patient_labels = train_df.groupby("patient_id")[label_col].max().reset_index()
    patient_labels.columns = ["patient_id", "patient_label"]

    if len(patient_labels) < n_folds:
        raise ValueError(
            f"Not enough unique patients ({len(patient_labels)}) to create {n_folds} folds."
        )

    train_df["fold"] = -1
    patient_ids = patient_labels["patient_id"].values
    patient_labels_array = patient_labels["patient_label"].values
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (_, val_idx) in enumerate(splitter.split(patient_ids, patient_labels_array, groups=patient_ids)):
        val_patients = patient_ids[val_idx]
        train_df.loc[train_df["patient_id"].isin(val_patients), "fold"] = fold_idx
        n_pos = int(patient_labels_array[val_idx].sum())
        n_total = int(len(val_idx))
        ratio = (n_pos / n_total * 100.0) if n_total else 0.0
        print(f"  Fold {fold_idx}: {n_total} patients, {n_pos} positive ({ratio:.1f}%)")

    if (train_df["fold"] < 0).any():
        raise AssertionError("Some training samples were not assigned a fold.")
    return train_df


def save_per_fold_csvs(result_df, output_path, n_folds):
    fold_dir = output_path.parent / "fold_split_csvs"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_indices = [0] if n_folds == 0 else list(range(n_folds))
    for fold_idx in fold_indices:
        fold_df = result_df.copy()
        if n_folds == 0:
            fold_df["split"] = fold_df["split"].where(fold_df["split"].isin(["val", "test"]), "train")
        else:
            fold_df["split"] = "train"
            fold_df.loc[fold_df["fold"] == fold_idx, "split"] = "val"
            fold_df.loc[fold_df["fold"] == -1, "split"] = "test"

        fold_path = fold_dir / f"{output_path.stem}_fold{fold_idx}{output_path.suffix}"
        fold_df.to_csv(fold_path, index=False)
        print(f"Saved fold-view CSV: {fold_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare split CSVs for Mammo-CLIP finetuning")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file with fold column")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds. Use 0 to disable CV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="cohort",
        choices=["cohort", "split"],
        help="Use cohort-driven splitting or legacy split-column based splitting.",
    )
    parser.add_argument("--cohort-col", type=str, default="cohort_num", help="Cohort column name")
    parser.add_argument("--label-col", type=str, default="cancer", help="Target label column for stratification")
    parser.add_argument("--train-cohorts", type=str, default="1-8", help="Train cohort spec, e.g. 1-8,12")
    parser.add_argument("--test-cohorts", type=str, default="9-10", help="Test cohort spec, e.g. 9-10")
    parser.add_argument(
        "--holdout-val-percent",
        type=float,
        default=0.0,
        help="Only used when n_folds=0: percent of training samples to reserve for validation by patient.",
    )
    parser.add_argument(
        "--holdout-val-max-percent",
        type=float,
        default=0.0,
        help=(
            "Only used when n_folds=0: maximum validation sample percent if the initial "
            "validation split has only one class. Use 0 to disable expansion."
        ),
    )
    args = parser.parse_args()

    if args.n_folds < 0:
        raise ValueError("--n_folds must be >= 0.")
    _validate_percent("--holdout-val-percent", args.holdout_val_percent)
    _validate_percent("--holdout-val-max-percent", args.holdout_val_max_percent)
    if args.n_folds > 0 and (args.holdout_val_percent > 0 or args.holdout_val_max_percent > 0):
        print("Ignoring holdout validation options because n_folds > 0.")

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    required_columns = {"patient_id", "image_id", args.label_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    backup_col = backup_existing_split(df)
    if backup_col is not None:
        print(f"Backed up input split column to '{backup_col}'.")

    df = build_base_split(df, args)
    print(f"Prepared split distribution:\n{df['split'].value_counts()}")
    print(f"{args.label_col} distribution:\n{df[args.label_col].value_counts()}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\nTraining-pool samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training-pool {args.label_col} rate: {train_df[args.label_col].mean():.4f}")
    print(f"Unique patients in training pool: {train_df['patient_id'].nunique()}")

    train_df = assign_train_pool_folds(
        train_df,
        args.n_folds,
        args.seed,
        args.label_col,
        holdout_val_percent=args.holdout_val_percent,
        holdout_val_max_percent=args.holdout_val_max_percent,
    )
    test_df["fold"] = -1

    result_df = pd.concat([train_df, test_df], ignore_index=True)
    if args.n_folds == 0:
        result_df["split"] = result_df["split"].where(result_df["split"].isin(["val", "test"]), "train")
    else:
        result_df["split"] = result_df["split"].where(result_df["split"] == "test", "train")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved main split CSV: {output_path}")
    print(f"Total output samples: {len(result_df)}")

    if args.n_folds == 0:
        print("Fold distribution: disabled (n_folds=0, all non-test rows use fold=0)")
        print(f"Final split distribution:\n{result_df['split'].value_counts()}")
    else:
        print(f"Fold distribution:\n{train_df['fold'].value_counts().sort_index()}")

    save_per_fold_csvs(result_df, output_path, args.n_folds)


if __name__ == "__main__":
    main()
