"""
一键微调脚本：自动完成数据预处理 → 5折训练 → 测试集预测 → Ensemble输出

使用方法:
    修改下方配置区域的变量后直接运行即可：
    python run_finetune.py

数据目录结构要求:
    /mnt/g/data/
    ├── train_with_test_data_mini.csv   (原始CSV文件)
    ├── images_png\                     (图片目录)
    │   ├── 10006\                      (patient_id)
    │   │   ├── 462822612.png           (image_id.png)
    │   │   └── ...
    │   ├── 10011\
    │   └── ...
    └── ...
"""

# =============================================================================
# ========================= 配置区域（按需修改） ==============================
# =============================================================================

# ---- 路径 ----
CSV_PATH          = r"/mnt/g/data/train_with_test_data_mini.csv"  # 原始CSV文件路径
DATA_DIR          = r"/mnt/g/data"                                 # 数据根目录
IMG_DIR           = "images_png"                                   # 图片目录（相对于 DATA_DIR，也可用绝对路径）
CLIP_CHK_PT_PATH  = "./model/b5-model-best-epoch-7.tar"          # Mammo-CLIP预训练权重路径

# ---- 数据集 / 任务 ----
LABEL             = "cancer"                          # CSV中的标签列名
ARCH              = "breast_clip_det_b5_period_n_ft"  # 模型架构: "breast_clip_det_b5_period_n_ft"=全量微调, "breast_clip_det_b5_period_n_lp"=线性探针

# ---- 训练 ----
N_FOLDS           = 5         # 交叉验证折数
EPOCHS            = 5         # 每折最大训练轮数
PATIENCE          = 10        # 早停耐心值（0=禁用）
BATCH_SIZE        = 2         # 批大小
LR                = 5e-5      # 学习率
SEED              = 42        # 随机种子
WEIGHTED_BCE      = "y"       # 是否使用加权BCE ("y"/"n")

# ---- 图片 ----
IMG_SIZE          = [912, 1520]  # 图片尺寸 [宽, 高]

# ---- 系统 ----
DEVICE            = "cuda"    # 设备 ("cuda" 或 "cpu")
NUM_WORKERS       = 0         # 数据加载线程数
APEX              = "y"       # 混合精度 ("y"/"n")

# ---- 流程控制 ----
SKIP_PREPARE      = False     # 是否跳过折划分准备（已准备好时设为 True）

# =============================================================================
# =========================== 运行（无需修改） ===============================
# =============================================================================

import os
import subprocess
import sys


def ensure_nltk_punkt():
    """确保 NLTK punkt tokenizer 数据已下载，避免训练时重复下载或网络超时"""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
        print("[NLTK] punkt tokenizer already exists, skip download.")
    except LookupError:
        print("[NLTK] punkt tokenizer not found, attempting to download...")
        try:
            nltk.download("punkt", quiet=True)
            print("[NLTK] punkt tokenizer downloaded successfully.")
        except Exception as e:
            print(f"[NLTK WARNING] Failed to download punkt: {e}")
            print("[NLTK WARNING] If the server cannot connect to the internet, you can manually download punkt:")
            print("  Method 1: Run  python -c \"import nltk; nltk.download('punkt')\"")
            print("  Method 2: Download from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip")
            print("             and extract to ~/nltk_data/tokenizers/punkt/")


def main():
    # ===================== Step 0: 检查 NLTK punkt =====================
    ensure_nltk_punkt()

    # 获取脚本所在目录（项目根目录）
    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "src", "scripts")
    codebase_dir = os.path.join(project_root, "src", "codebase")

    # 将相对路径转换为绝对路径，避免 subprocess cwd 变化导致路径失效
    clip_chk_pt_path = CLIP_CHK_PT_PATH
    if not os.path.isabs(clip_chk_pt_path):
        clip_chk_pt_path = os.path.abspath(clip_chk_pt_path)

    # 生成的 folds CSV 路径
    folds_csv_path = os.path.join(DATA_DIR, "train_with_test_folds.csv")
    csv_filename = "train_with_test_folds.csv"

    # ===================== Step 1: 准备折划分 =====================
    if not SKIP_PREPARE:
        print("\n" + "=" * 60)
        print("Step 1: Preparing 5-fold stratified split...")
        print("=" * 60)
        prepare_cmd = [
            sys.executable,
            os.path.join(scripts_dir, "prepare_folds.py"),
            "--csv_path", CSV_PATH,
            "--output_path", folds_csv_path,
            "--n_folds", str(N_FOLDS),
            "--seed", str(SEED),
        ]
        print(f"Running: {' '.join(prepare_cmd)}")
        result = subprocess.run(prepare_cmd, cwd=scripts_dir)
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
        "--data-dir", DATA_DIR,
        "--img-dir", IMG_DIR,
        "--csv-file", csv_filename,
        "--clip_chk_pt_path", clip_chk_pt_path,
        "--dataset", "custom",
        "--arch", ARCH,
        "--label", LABEL,
        "--n_folds", str(N_FOLDS),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--img-size", str(IMG_SIZE[0]), str(IMG_SIZE[1]),
        "--lr", str(LR),
        "--seed", str(SEED),
        "--weighted-BCE", WEIGHTED_BCE,
        "--patience", str(PATIENCE),
        "--num-workers", str(NUM_WORKERS),
        "--device", DEVICE,
        "--apex", APEX,
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