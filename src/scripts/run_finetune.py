"""
One-click finetuning helper for Mammo-CLIP.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Mammo-CLIP One-Click Finetuning Pipeline")
    parser.add_argument("--csv_path", default=r"/mnt/g/data/train_with_test_data_mini.csv", type=str, help="Path to original CSV file")
    parser.add_argument("--data_dir", default=r"/mnt/g/data", type=str, help="Data directory containing images")
    parser.add_argument("--img_dir", default="images_png", type=str, help="Image directory name relative to data_dir")
    parser.add_argument("--clip_chk_pt_path", default="./model/b5-model-best-epoch-7.tar", type=str, help="Path to Mammo-CLIP checkpoint")
    parser.add_argument("--arch", default="breast_clip_det_b5_period_n_ft", type=str, help="Model architecture")
    parser.add_argument("--label", default="cancer", type=str, help="Label column in CSV")
    parser.add_argument("--n_folds", default=5, type=int, help="Number of folds. Use 0 to disable CV")
    parser.add_argument("--epochs", default=30, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--img_size", nargs="+", default=[912, 1520], type=int, help="Input image size: width height")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--weighted_BCE", default="y", type=str, help="Use weighted BCE loss (y/n)")
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
    parser.add_argument("--skip_prepare", action="store_true", help="Skip split CSV preparation")
    parser.add_argument("--split-mode", choices=["cohort", "split"], default="cohort", help="Split source mode")
    parser.add_argument("--cohort-col", default="cohort_num", help="Cohort column name")
    parser.add_argument("--train-cohorts", default="1-8", help="Train cohort spec, e.g. 1-8,12")
    parser.add_argument("--test-cohorts", default="9-10", help="Test cohort spec, e.g. 9-10")
    parser.add_argument(
        "--holdout-val-percent",
        default=20.0,
        type=float,
        help="Only used when n_folds=0: percent of training samples to reserve for validation",
    )
    parser.add_argument(
        "--holdout-val-max-percent",
        default=20.0,
        type=float,
        help="Only used when n_folds=0: max validation sample percent if the initial val split has one class",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codebase_dir = os.path.join(os.path.dirname(script_dir), "codebase")
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if not os.path.isabs(args.clip_chk_pt_path):
        args.clip_chk_pt_path = os.path.abspath(args.clip_chk_pt_path)

    folds_csv_path = os.path.join(project_root, "folds", "finetune_holdout_seed42.csv")
    csv_filename = folds_csv_path

    if not args.skip_prepare:
        print("\n" + "=" * 60)
        print("Step 1: Preparing cohort-aware split CSVs...")
        print("=" * 60)
        prepare_cmd = [
            sys.executable,
            os.path.join(script_dir, "prepare_folds.py"),
            "--csv_path", args.csv_path,
            "--output_path", folds_csv_path,
            "--n_folds", str(args.n_folds),
            "--seed", str(args.seed),
            "--split-mode", args.split_mode,
            "--cohort-col", args.cohort_col,
            "--label-col", args.label,
            "--train-cohorts", args.train_cohorts,
            "--test-cohorts", args.test_cohorts,
            "--holdout-val-percent", str(args.holdout_val_percent),
            "--holdout-val-max-percent", str(args.holdout_val_max_percent),
        ]
        print(f"Running: {' '.join(prepare_cmd)}")
        result = subprocess.run(prepare_cmd, cwd=script_dir)
        if result.returncode != 0:
            print("Error in split preparation!")
            sys.exit(1)
        print("Split preparation completed!")
    else:
        print(f"Skipping split preparation. Using existing: {folds_csv_path}")

    print("\n" + "=" * 60)
    if args.n_folds == 0:
        print("Step 2: Training classifier with train/test evaluation...")
    else:
        print(f"Step 2: Training classifier with {args.n_folds}-fold CV...")
    print("=" * 60)
    train_cmd = [
        sys.executable,
        os.path.join(codebase_dir, "train_classifier.py"),
        "--data-dir", args.data_dir,
        "--img-dir", args.img_dir,
        "--csv-file", csv_filename,
        "--clip_chk_pt_path", args.clip_chk_pt_path,
        "--dataset", "custom",
        "--arch", args.arch,
        "--label", args.label,
        "--n_folds", str(args.n_folds),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size[0]), str(args.img_size[1]),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--weighted-BCE", args.weighted_BCE,
        "--patience", str(args.patience),
        "--num-workers", "0",
        "--device", "cuda",
        "--apex", "y",
    ]
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd=codebase_dir)
    if result.returncode != 0:
        print("Error in training!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All done! Check the 'outputs' directory for results.")
    print("  - OOF predictions: outputs/custom/zz/classifier/.../*_oof_outputs.csv")
    print("  - Per-fold all-data predictions: outputs/custom/zz/classifier/.../fold*_all_predictions.csv")
    print("  - Ensemble all-data predictions: outputs/custom/zz/classifier/.../ensemble_all_predictions.csv")
    print("  - Per-fold metrics: outputs/custom/zz/classifier/.../fold*_metrics.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
