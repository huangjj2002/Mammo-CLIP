"""
一键微调脚本：自动完成数据预处理 → 5折训练 → 测试集预测 → Ensemble输出

使用方法:
    python run_finetune.py

    # 或自定义参数
    python run_finetune.py \
        --csv_path "G:\data\train_with_test_data.csv" \
        --data_dir "G:\data" \
        --img_dir "images" \
        --epochs 30 \
        --batch_size 8 \
        --lr 5e-5

数据目录结构要求:
    G:\data\
    ├── train_with_test_data.csv   (原始CSV文件)
    ├── images\                    (图片目录)
    │   ├── 10006\                 (patient_id)
    │   │   ├── 462822612.png      (image_id.png)
    │   │   └── ...
    │   ├── 10011\
    │   └── ...
    └── ...
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Mammo-CLIP One-Click Finetuning Pipeline")
    # 数据路径
    parser.add_argument("--csv_path", default=r"/mnt/g/data/train_with_test_data_mini.csv", type=str,
                        help="Path to original CSV file")
    parser.add_argument("--data_dir", default=r"/mnt/g/data", type=str,
                        help="Data directory (containing CSV and images)")
    parser.add_argument("--img_dir", default="images_png", type=str,
                        help="Image directory name (relative to data_dir)")
    parser.add_argument("--clip_chk_pt_path", default="./model/b5-model-best-epoch-7.tar", type=str,
                        help="Path to Mammo-CLIP pretrained checkpoint")

    # 训练参数
    parser.add_argument("--arch", default="breast_clip_det_b5_period_n_ft", type=str,
                        help="Model architecture: breast_clip_det_b5_period_n_ft or breast_clip_det_b5_period_n_lp")
    parser.add_argument("--label", default="cancer", type=str, help="Label column in CSV")
    parser.add_argument("--n_folds", default=5, type=int, help="Number of CV folds")
    parser.add_argument("--epochs", default=30, type=int, help="Training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--img_size", nargs="+", default=[912, 1520], type=int,
                        help="Input image size as: width height")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--weighted_BCE", default="n", type=str,
                        help="Use weighted BCE loss for imbalanced data (y/n)")
    parser.add_argument("--patience", default=10, type=int,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip fold preparation (if already done)")

    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    codebase_dir = os.path.join(os.path.dirname(script_dir), "codebase")
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # 将相对路径转换为绝对路径，避免 subprocess cwd 变化导致路径失效
    if not os.path.isabs(args.clip_chk_pt_path):
        args.clip_chk_pt_path = os.path.abspath(args.clip_chk_pt_path)

    # 生成的 folds CSV 路径
    folds_csv_path = os.path.join(args.data_dir, "train_with_test_folds.csv")
    csv_filename = "train_with_test_folds.csv"

    # ===================== Step 1: 准备折划分 =====================
    if not args.skip_prepare:
        print("\n" + "=" * 60)
        print("Step 1: Preparing 5-fold stratified split...")
        print("=" * 60)
        prepare_cmd = [
            sys.executable,
            os.path.join(script_dir, "prepare_folds.py"),
            "--csv_path", args.csv_path,
            "--output_path", folds_csv_path,
            "--n_folds", str(args.n_folds),
            "--seed", str(args.seed),
        ]
        print(f"Running: {' '.join(prepare_cmd)}")
        result = subprocess.run(prepare_cmd, cwd=script_dir)
        if result.returncode != 0:
            print("Error in fold preparation!")
            sys.exit(1)
        print("Fold preparation completed!")
    else:
        print(f"Skipping fold preparation. Using existing: {folds_csv_path}")

    # ===================== Step 2: 训练分类器 =====================
    print("\n" + "=" * 60)
    print("Step 2: Training classifier with 5-fold CV...")
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
    print("=" * 60)


if __name__ == "__main__":
    main()
