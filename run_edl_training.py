"""
Mammo-CLIP + Evidential Deep Learning (EDL) 微调脚本

功能:
  1. 加载 Mammo-CLIP 预训练骨干网络（不冻结，全量微调）
  2. 将线性分类头替换为 EDL 分类头，输出 evidence 构造 Dirichlet 分布
  3. 使用类交叉熵损失函数（digamma）+ KL散度正则化进行优化
  4. 五折交叉验证训练
  5. 训练结束后自动调用测试模块
  6. 生成包含 evidence、alpha、probability、uncertainty、fold 列的 CSV 文件
  7. 5折 ensemble 预测结果

使用方法:
    直接修改下方配置区域的参数，然后运行:
    python run_edl_training.py --train-mode full
    python run_edl_training.py --train-mode head_only
    
    或者不做任何修改，使用默认参数运行。

输出文件:
    outputs/edl/.../edl_fold{i}_all_predictions.csv    - 每折全量预测
    outputs/edl/.../edl_ensemble_all_predictions.csv    - Ensemble全量预测
    outputs/edl/.../edl_seed_42_n_folds_5_oof_outputs.csv  - OOF预测

注意:
    - 本脚本不修改原项目任何代码
    - 所有EDL相关代码位于 src/codebase/edl_*.py
"""

# =============================================================================
# ========================= 配置区域 ==============================
# =============================================================================

# ---------- 数据路径 ----------
CSV_PATH           = r"/opt/localdata/Data/dh/dh_preprocessed/hjj_images/embed_data_testcohort_enriched.csv"
DATA_DIR           = r"/opt/localdata/Data/dh/dh_preprocessed/hjj_images"

IMG_DIR            = "images_png"
CLIP_CHK_PT_PATH   = "./model/b5-model-best-epoch-7.tar"
MODEL_SAVE_DIR     = "best_model"
CSV_SAVE_DIR       = "output"

# ---------- 任务配置 ----------
LABEL              = "cancer"
ARCH               = "breast_clip_det_b5_period_n_ft"

# ---------- 交叉验证 ----------
N_FOLDS            = 5
EPOCHS             = 25
PATIENCE           = 3
BATCH_SIZE         = 32
LR                 = 5e-5
SEED               = 42
WEIGHTED_BCE       = "y"       # 对EDL逐样本loss启用类别不平衡加权

# ---------- 图像尺寸 ----------
IMG_SIZE           = [912, 1520]

# ---------- 设备 ----------
DEVICE             = "cuda"
NUM_WORKERS        = 4
APEX               = "y"
GPU_ID             = 0

# ---------- 是否跳过数据准备 ----------
SKIP_PREPARE       = False

# ---------- EDL training mode ----------
# full: train image encoder + EDL head
# head_only: freeze image encoder and train only the EDL head
TRAIN_MODE         = "full"

# ---------- EDL专属参数 ----------
EDL_LOSS_TYPE      = "digamma"     # 损失函数类型: 'digamma'(推荐), 'log', 'mse'
EDL_NUM_CLASSES    = 2             # 类别数（二分类=2，EDL需要为每个类别输出evidence）
EDL_KL_WEIGHT      = 0.1           # KL正则项权重，对齐参考MIL项目
EDL_ANNEALING_START = 0            # KL退火开始epoch
EDL_ANNEALING_EPOCHS = 10          # KL退火到1.0所需epoch数
EDL_DROPOUT        = 0.0           # EDL分类头Dropout
EDL_HIDDEN_DIM     = None          # EDL分类头隐藏层维度（None=直接线性映射）


# =============================================================================
# ========================= 以下为执行逻辑，一般无需修改 ==============================
# =============================================================================

import os
import subprocess
import sys
import argparse


def ensure_nltk_punkt():
    """确保NLTK punkt tokenizer已下载"""
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
    parser = argparse.ArgumentParser(description="Run Mammo-CLIP EDL training.")
    parser.add_argument(
        "--train-mode",
        choices=["full", "head_only"],
        default=TRAIN_MODE,
        help="full trains the image encoder and EDL head; head_only freezes the image encoder.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=GPU_ID,
        help="CUDA device id to use, e.g. 0, 1, 2, or 3.",
    )
    parser.add_argument(
        "--model-save-dir",
        default=MODEL_SAVE_DIR,
        help="Directory under the project root for saving best-model checkpoints.",
    )
    parser.add_argument(
        "--csv-save-dir",
        default=CSV_SAVE_DIR,
        help="Directory under the project root for saving prediction CSVs and loss curves.",
    )
    cli_args = parser.parse_args()

    ensure_nltk_punkt()

    gpu_id = cli_args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU] Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")

    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "src", "scripts")
    codebase_dir = os.path.join(project_root, "src", "codebase")

    clip_chk_pt_path = CLIP_CHK_PT_PATH
    if not os.path.isabs(clip_chk_pt_path):
        clip_chk_pt_path = os.path.abspath(clip_chk_pt_path)

    folds_csv_path = os.path.join(DATA_DIR, "train_with_test_folds.csv")
    csv_filename = "train_with_test_folds.csv"

    # ===================== Step 1: 准备五折划分 =====================
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

    # ===================== Step 2: EDL训练 + 自动测试 =====================
    print("\n" + "=" * 60)
    print("Step 2: EDL Training with 5-fold CV + Auto-testing...")
    print("=" * 60)

    # 直接调用EDL训练模块（而非通过subprocess，以便在同一进程中运行）
    import torch
    sys.path.insert(0, codebase_dir)

    from edl_trainer import do_edl_experiments
    from utils import get_Paths, seed_all
    from pathlib import Path

    # 构建args对象
    args = argparse.Namespace()

    # 路径参数
    args.data_dir = DATA_DIR
    args.img_dir = IMG_DIR
    args.csv_file = csv_filename
    args.clip_chk_pt_path = clip_chk_pt_path
    args.dataset = "custom"
    args.arch = ARCH
    args.label = LABEL
    args.tensorboard_path = "./logs"
    args.checkpoints = os.path.join(project_root, cli_args.model_save_dir)
    args.output_path = os.path.join(project_root, cli_args.csv_save_dir)

    # 模型参数
    args.model_type = "classifier"
    args.VER = "edl"
    args.detector_threshold = 0.1
    args.swin_encoder = "microsoft/swin-tiny-patch4-window7-224"
    args.pretrained_swin_encoder = True
    args.swin_model_type = True

    # 数据增强参数
    args.epochs_warmup = 0
    args.num_cycles = 0.5
    args.alpha = 10
    args.sigma = 15
    args.p = 1.0
    args.mean = 0.3089279
    args.std = 0.25053555408335154
    args.focal_alpha = 0.6
    args.focal_gamma = 2.0

    # 训练参数
    args.n_folds = N_FOLDS
    args.seed = SEED
    args.batch_size = BATCH_SIZE
    args.num_workers = NUM_WORKERS
    args.epochs = EPOCHS
    args.lr = LR
    args.weight_decay = 1e-4
    args.warmup_epochs = 1
    args.img_size = IMG_SIZE
    args.device = DEVICE
    args.apex = True if APEX == "y" else False
    args.gpu_id = gpu_id
    args.print_freq = 5000
    args.log_freq = 1000
    args.running_interactive = False
    args.data_frac = 1.0
    args.weighted_BCE = WEIGHTED_BCE
    args.patience = PATIENCE
    args.balanced_dataloader = "n"
    args.train_mode = cli_args.train_mode
    args.freeze_backbone = "y" if args.train_mode == "head_only" else "n"

    # EDL专属参数
    args.num_classes = EDL_NUM_CLASSES
    args.edl_loss_type = EDL_LOSS_TYPE
    args.edl_kl_weight = EDL_KL_WEIGHT
    args.edl_annealing_start = EDL_ANNEALING_START
    args.edl_annealing_epochs = EDL_ANNEALING_EPOCHS
    args.edl_dropout = EDL_DROPOUT
    args.edl_hidden_dim = EDL_HIDDEN_DIM

    # 设置种子
    seed_all(args.seed)

    # 设置路径（EDL使用独立输出目录，直接构造避免跨平台路径分隔符问题）
    args.root = f"lr_{args.lr}_epochs_{args.epochs}_edl_{args.edl_loss_type}_{args.label}_data_frac_{args.data_frac}_mode_{args.train_mode}"
    chk_pt_path = Path(args.checkpoints) / args.dataset / "edl_classifier" / args.arch / args.root
    output_path = Path(args.output_path) / args.dataset / "zz" / "edl_classifier" / args.arch / args.root
    tb_logs_path = Path(args.tensorboard_path) / args.dataset / "edl_classifier" / args.arch / args.root

    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    print("====================> EDL Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print(f"data_dir: {args.data_dir}")
    print(f"img_dir: {args.img_dir}")
    print(f"csv_file: {args.csv_file}")
    print(f"clip_checkpoint: {args.clip_chk_pt_path}")
    print(f"num_classes: {args.num_classes}")
    print(f"edl_loss_type: {args.edl_loss_type}")
    print(f"edl_kl_weight: {args.edl_kl_weight}")
    print(f"edl_annealing_start: {args.edl_annealing_start}")
    print(f"edl_annealing_epochs: {args.edl_annealing_epochs}")
    print(f"edl_dropout: {args.edl_dropout}")
    print(f"edl_hidden_dim: {args.edl_hidden_dim}")
    print(f"train_mode: {args.train_mode}")
    print(f"freeze_backbone: {args.freeze_backbone}")
    device = DEVICE if DEVICE != "cuda" else ('cuda' if torch.cuda.is_available() else 'cpu')
    args.apex = args.apex and str(device).startswith("cuda")

    print('device:', device)
    print('torch version:', torch.__version__)
    print("====================> EDL Paths <====================")

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 运行EDL实验（包含训练 + 自动测试）
    do_edl_experiments(args, device)

    print("\n" + "=" * 60)
    print("All done! Check the outputs directory for EDL results:")
    print(f"  - Per-fold predictions: {output_path}/edl_fold*_all_predictions.csv")
    print(f"  - Ensemble predictions: {output_path}/edl_ensemble_all_predictions.csv")
    print(f"  - OOF predictions: {output_path}/edl_seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
