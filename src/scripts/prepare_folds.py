"""
Prepare split CSVs for Mammo-CLIP finetuning.

Supports either:
1. Legacy split-column based partitioning
2. Cohort-driven partitioning via a CSV cohort column such as `cohort_num`
"""

import argparse
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


def assign_train_pool_folds(train_df, n_folds, seed):
    train_df = train_df.copy().reset_index(drop=True)
    if n_folds == 0:
        train_df["fold"] = 0
        return train_df

    patient_labels = train_df.groupby("patient_id")["cancer"].max().reset_index()
    patient_labels.columns = ["patient_id", "patient_cancer"]

    if len(patient_labels) < n_folds:
        raise ValueError(
            f"Not enough unique patients ({len(patient_labels)}) to create {n_folds} folds."
        )

    train_df["fold"] = -1
    patient_ids = patient_labels["patient_id"].values
    patient_cancer = patient_labels["patient_cancer"].values
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (_, val_idx) in enumerate(splitter.split(patient_ids, patient_cancer, groups=patient_ids)):
        val_patients = patient_ids[val_idx]
        train_df.loc[train_df["patient_id"].isin(val_patients), "fold"] = fold_idx
        n_pos = int(patient_cancer[val_idx].sum())
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
            fold_df["split"] = fold_df["split"].where(fold_df["split"] == "test", "train")
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
    parser.add_argument("--train-cohorts", type=str, default="1-8", help="Train cohort spec, e.g. 1-8,12")
    parser.add_argument("--test-cohorts", type=str, default="9-10", help="Test cohort spec, e.g. 9-10")
    args = parser.parse_args()

    if args.n_folds < 0:
        raise ValueError("--n_folds must be >= 0.")

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    required_columns = {"patient_id", "image_id", "cancer"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    backup_col = backup_existing_split(df)
    if backup_col is not None:
        print(f"Backed up input split column to '{backup_col}'.")

    df = build_base_split(df, args)
    print(f"Prepared split distribution:\n{df['split'].value_counts()}")
    print(f"Cancer distribution:\n{df['cancer'].value_counts()}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"\nTraining-pool samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training-pool cancer rate: {train_df['cancer'].mean():.4f}")
    print(f"Unique patients in training pool: {train_df['patient_id'].nunique()}")

    train_df = assign_train_pool_folds(train_df, args.n_folds, args.seed)
    test_df["fold"] = -1

    result_df = pd.concat([train_df, test_df], ignore_index=True)
    result_df["split"] = result_df["split"].where(result_df["split"] == "test", "train")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved main split CSV: {output_path}")
    print(f"Total output samples: {len(result_df)}")

    if args.n_folds == 0:
        print("Fold distribution: disabled (n_folds=0, all training-pool rows use fold=0)")
    else:
        print(f"Fold distribution:\n{train_df['fold'].value_counts().sort_index()}")

    save_per_fold_csvs(result_df, output_path, args.n_folds)


if __name__ == "__main__":
    main()
