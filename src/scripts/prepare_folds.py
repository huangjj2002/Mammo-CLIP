"""
预处理脚本：将 train_with_test_data.csv 按 patient 层次进行5折分层划分。

用法:
    python prepare_folds.py --csv_path "G:\data\train_with_test_data.csv" --output_path "G:\data\train_with_test_folds.csv" --n_folds 5 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    parser = argparse.ArgumentParser(description="Prepare 5-fold stratified split for Mammo-CLIP finetuning")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file with fold column")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Split distribution:\n{df['split'].value_counts()}")
    print(f"Cancer distribution:\n{df['cancer'].value_counts()}")

    # Separate training and test data
    train_df = df[df['split'] == 'training'].copy().reset_index(drop=True)
    test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training cancer rate: {train_df['cancer'].mean():.4f}")
    print(f"Unique patients in training: {train_df['patient_id'].nunique()}")

    # Stratified Group K-Fold split by patient_id, stratified by cancer
    # Get patient-level labels (a patient is positive if any of their images is positive)
    patient_labels = train_df.groupby('patient_id')['cancer'].max().reset_index()
    patient_labels.columns = ['patient_id', 'patient_cancer']

    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # Assign fold to each row in train_df
    train_df['fold'] = -1

    patient_ids = patient_labels['patient_id'].values
    patient_cancer = patient_labels['patient_cancer'].values

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(patient_ids, patient_cancer, groups=patient_ids)):
        val_patients = patient_ids[val_idx]
        train_df.loc[train_df['patient_id'].isin(val_patients), 'fold'] = fold_idx
        n_pos = patient_cancer[val_idx].sum()
        n_total = len(val_idx)
        print(f"  Fold {fold_idx}: {n_total} patients, {n_pos} positive ({n_pos/n_total*100:.1f}%)")

    # Verify all training samples have a fold
    assert (train_df['fold'] >= 0).all(), "Some training samples were not assigned a fold!"

    # Mark test data with fold = -1
    test_df['fold'] = -1

    # Combine and save
    result_df = pd.concat([train_df, test_df], ignore_index=True)
    result_df.to_csv(args.output_path, index=False)
    print(f"\nSaved to: {args.output_path}")
    print(f"Total output samples: {len(result_df)}")
    print(f"Fold distribution:\n{result_df[result_df['split'] == 'training']['fold'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()