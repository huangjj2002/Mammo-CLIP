"""
Mammo-CLIP + Prototype Evidential Deep Learning (Prototype EDL) entrypoint.

Example command:
python run_prototype_edl_training.py ^
  --csv-path /path/to/data.csv ^
  --data-dir /path/to/data ^
  --img-dir images_png ^
  --clip-chk-pt-path ./model/b5-model-best-epoch-7.tar ^
  --label cancer ^
  --n-folds 0 ^
  --holdout-val-percent 20 ^
  --epochs 25 ^
  --edl-proto-k 4 ^
  --edl-proto-topk 3 ^
  --edl-proto-attract-weight 0.1 ^
  --edl-proto-separation-weight 0.1 ^
  --edl-proto-diversity-weight 0.01
"""

SKIP_BAD_BATCHES = True

CSV_PATH = r"/home/dhao4/workspace/hjj_workspace/data/data.csv"
DATA_DIR = r"/home/dhao4/workspace/hjj_workspace/data"

IMG_DIR = "images_png"
CLIP_CHK_PT_PATH = "./model/b5-model-best-epoch-7.tar"
MODEL_SAVE_DIR = "best_model"
CSV_SAVE_DIR = "output"
TENSORBOARD_DIR = "logs"

LABEL = "cancer"
ARCH = "breast_clip_det_b5_period_n_ft"

N_FOLDS = 0
EPOCHS = 25
PATIENCE = 3
BATCH_SIZE = 8
LR = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
SEED = 42
WEIGHTED_BCE = "y"

IMG_SIZE = [912, 1520]

DEVICE = "cuda"
NUM_WORKERS = 4
APEX = "y"
GPU_ID = 3

SKIP_PREPARE = False
FOLDS_CSV_PATH = "folds/prototype_edl_holdout_seed42.csv"
OVERWRITE_FOLDS = False
SPLIT_MODE = "cohort"
COHORT_COL = "cohort_num"
TRAIN_COHORTS = "1-8"
TEST_COHORTS = "9-10"
HOLDOUT_VAL_PERCENT = 20.0
HOLDOUT_VAL_MAX_PERCENT = 20.0

TRAIN_MODE = "full"
RESUME_TRAINING = False

EDL_LOSS_TYPE = "digamma"
EDL_NUM_CLASSES = 2
EDL_KL_WEIGHT = 0.1
EDL_ANNEALING_START = 0
EDL_ANNEALING_EPOCHS = 10

EDL_PROTO_K = 4
EDL_PROTO_TOPK = 3
EDL_PROTO_TEMPERATURE = 1.0
EDL_PROTO_NORMALIZE = "y"
EDL_PROTO_INIT = "kmeans"
EDL_PROTO_ATTRACT_WEIGHT = 0.1
EDL_PROTO_SEPARATION_WEIGHT = 0.1
EDL_PROTO_DIVERSITY_WEIGHT = 0.01
EDL_PROTO_MARGIN = 1.0
EDL_PROTO_BALANCE_CLASSES = "y"

import argparse
import os
import subprocess
import sys


def ensure_nltk_punkt():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
        print("[NLTK] punkt tokenizer already exists, skip download.")
    except LookupError:
        print("[NLTK] punkt tokenizer not found, attempting to download...")
        try:
            nltk.download("punkt", quiet=True)
            print("[NLTK] punkt tokenizer downloaded successfully.")
        except Exception as exc:
            print(f"[NLTK WARNING] Failed to download punkt: {exc}")
            print("[NLTK WARNING] If the server cannot connect to the internet, you can manually download punkt:")
            print("  Method 1: Run  python -c \"import nltk; nltk.download('punkt')\"")
            print("  Method 2: Download from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip")
            print("             and extract to ~/nltk_data/tokenizers/punkt/")


def build_parser():
    parser = argparse.ArgumentParser(description="Run Mammo-CLIP Prototype EDL training.")
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to the input CSV file.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing images.")
    parser.add_argument("--img-dir", default=IMG_DIR, help="Image directory relative to data-dir.")
    parser.add_argument("--clip-chk-pt-path", default=CLIP_CHK_PT_PATH, help="Path to Mammo-CLIP checkpoint.")
    parser.add_argument("--label", default=LABEL, help="Target label column.")
    parser.add_argument("--arch", default=ARCH, help="Classifier architecture.")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS, help="Number of folds. Use 0 to disable CV.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=float, default=WARMUP_EPOCHS, help="Warmup epochs.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--weighted-bce", default=WEIGHTED_BCE, help="Whether to use class weights: y/n.")
    parser.add_argument("--img-size", nargs=2, type=int, default=IMG_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", default=DEVICE, help="Training device, e.g. cuda or cpu.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Dataloader worker count.")
    parser.add_argument("--apex", default=APEX, help="Whether to enable AMP: y/n.")
    parser.add_argument("--gpu-id", type=int, default=GPU_ID, help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--skip-prepare", action="store_true", default=SKIP_PREPARE, help="Skip split CSV preparation.")
    parser.add_argument("--prepare", dest="skip_prepare", action="store_false", help="Run split CSV preparation.")
    parser.add_argument(
        "--folds-csv-path",
        default=FOLDS_CSV_PATH,
        help="Where to write/read the prepared fold CSV. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--overwrite-folds",
        action="store_true",
        default=OVERWRITE_FOLDS,
        help="Allow split preparation to overwrite an existing fold CSV.",
    )
    parser.add_argument("--split-mode", choices=["cohort", "split"], default=SPLIT_MODE, help="Split source mode.")
    parser.add_argument("--cohort-col", default=COHORT_COL, help="Cohort column name for cohort mode.")
    parser.add_argument("--train-cohorts", default=TRAIN_COHORTS, help="Train cohort spec, e.g. 1-8,12.")
    parser.add_argument("--test-cohorts", default=TEST_COHORTS, help="Test cohort spec, e.g. 9-10.")
    parser.add_argument(
        "--holdout-val-percent",
        type=float,
        default=HOLDOUT_VAL_PERCENT,
        help="Only used when n-folds=0: percent of training samples to reserve for validation.",
    )
    parser.add_argument(
        "--holdout-val-max-percent",
        type=float,
        default=HOLDOUT_VAL_MAX_PERCENT,
        help=(
            "Only used when n-folds=0: max validation sample percent if the initial "
            "validation split has only one class."
        ),
    )
    parser.add_argument(
        "--train-mode",
        choices=["full", "head_only"],
        default=TRAIN_MODE,
        help="full trains the image encoder and prototype EDL head; head_only freezes the image encoder.",
    )
    parser.add_argument("--model-save-dir", default=MODEL_SAVE_DIR, help="Project-local checkpoint directory.")
    parser.add_argument("--csv-save-dir", default=CSV_SAVE_DIR, help="Project-local prediction directory.")
    parser.add_argument("--tensorboard-dir", default=TENSORBOARD_DIR, help="Project-local tensorboard directory.")
    parser.add_argument("--resume", action="store_true", default=RESUME_TRAINING, help="Resume from last checkpoint.")
    parser.add_argument("--edl-proto-k", "--edl_proto_k", dest="edl_proto_k", type=int, default=EDL_PROTO_K, help="Prototypes per class.")
    parser.add_argument(
        "--edl-proto-topk",
        "--edl_proto_topk",
        dest="edl_proto_topk",
        type=int,
        default=EDL_PROTO_TOPK,
        help="Top-k prototypes to export per class.",
    )
    parser.add_argument(
        "--edl-proto-temperature",
        "--edl_proto_temperature",
        dest="edl_proto_temperature",
        type=float,
        default=EDL_PROTO_TEMPERATURE,
    )
    parser.add_argument(
        "--edl-proto-normalize",
        "--edl_proto_normalize",
        dest="edl_proto_normalize",
        choices=["y", "n"],
        default=EDL_PROTO_NORMALIZE,
    )
    parser.add_argument(
        "--edl-proto-init",
        "--edl_proto_init",
        dest="edl_proto_init",
        choices=["kmeans", "random"],
        default=EDL_PROTO_INIT,
    )
    parser.add_argument(
        "--edl-proto-attract-weight",
        "--edl_proto_attract_weight",
        dest="edl_proto_attract_weight",
        type=float,
        default=EDL_PROTO_ATTRACT_WEIGHT,
        help="Weight for pulling samples toward same-class prototypes.",
    )
    parser.add_argument(
        "--edl-proto-separation-weight",
        "--edl_proto_separation_weight",
        dest="edl_proto_separation_weight",
        type=float,
        default=EDL_PROTO_SEPARATION_WEIGHT,
        help="Weight for pushing samples away from other-class prototypes.",
    )
    parser.add_argument(
        "--edl-proto-diversity-weight",
        "--edl_proto_diversity_weight",
        dest="edl_proto_diversity_weight",
        type=float,
        default=EDL_PROTO_DIVERSITY_WEIGHT,
        help="Weight for keeping prototypes within each class separated.",
    )
    parser.add_argument(
        "--edl-proto-margin",
        "--edl_proto_margin",
        dest="edl_proto_margin",
        type=float,
        default=EDL_PROTO_MARGIN,
        help="Distance margin used by prototype separation and diversity losses.",
    )
    parser.add_argument(
        "--edl-proto-balance-classes",
        "--edl_proto_balance_classes",
        dest="edl_proto_balance_classes",
        choices=["y", "n"],
        default=EDL_PROTO_BALANCE_CLASSES,
        help="Average attraction/separation by class before averaging the batch.",
    )
    parser.add_argument(
        "--skip-bad-batches",
        dest="skip_bad_batches",
        action="store_true",
        default=SKIP_BAD_BATCHES,
        help="Skip recoverable malformed batches instead of aborting.",
    )
    parser.add_argument(
        "--no-skip-bad-batches",
        dest="skip_bad_batches",
        action="store_false",
        help="Disable bad-batch skipping and fail fast.",
    )
    return parser


def main():
    cli_args = build_parser().parse_args()
    if cli_args.edl_proto_k <= 0:
        raise ValueError("--edl-proto-k must be positive.")
    if cli_args.edl_proto_topk <= 0:
        raise ValueError("--edl-proto-topk must be positive.")
    for loss_name in (
        "edl_proto_attract_weight",
        "edl_proto_separation_weight",
        "edl_proto_diversity_weight",
    ):
        if getattr(cli_args, loss_name) < 0:
            raise ValueError(f"--{loss_name.replace('_', '-')} must be non-negative.")
    if cli_args.edl_proto_margin <= 0:
        raise ValueError("--edl-proto-margin must be positive.")
    ensure_nltk_punkt()

    gpu_id = cli_args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU] Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")

    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "src", "scripts")
    codebase_dir = os.path.join(project_root, "src", "codebase")

    clip_chk_pt_path = cli_args.clip_chk_pt_path
    if not os.path.isabs(clip_chk_pt_path):
        clip_chk_pt_path = os.path.abspath(clip_chk_pt_path)

    folds_csv_path = cli_args.folds_csv_path or os.path.join(project_root, "folds", "prototype_edl_holdout_seed42.csv")
    if not os.path.isabs(folds_csv_path):
        folds_csv_path = os.path.abspath(os.path.join(project_root, folds_csv_path))
    csv_filename = folds_csv_path

    if not cli_args.skip_prepare:
        if os.path.exists(folds_csv_path) and not cli_args.overwrite_folds:
            print(f"Refusing to overwrite existing fold CSV: {folds_csv_path}")
            print("Use --skip-prepare to reuse it, --folds-csv-path for a new file, or --overwrite-folds to replace it.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Step 1: Preparing cohort-aware split CSVs...")
        print("=" * 60)
        prepare_cmd = [
            sys.executable,
            os.path.join(scripts_dir, "prepare_folds.py"),
            "--csv_path", cli_args.csv_path,
            "--output_path", folds_csv_path,
            "--n_folds", str(cli_args.n_folds),
            "--seed", str(cli_args.seed),
            "--split-mode", cli_args.split_mode,
            "--cohort-col", cli_args.cohort_col,
            "--label-col", cli_args.label,
            "--train-cohorts", cli_args.train_cohorts,
            "--test-cohorts", cli_args.test_cohorts,
            "--holdout-val-percent", str(cli_args.holdout_val_percent),
            "--holdout-val-max-percent", str(cli_args.holdout_val_max_percent),
        ]
        print(f"Running: {' '.join(prepare_cmd)}")
        result = subprocess.run(prepare_cmd, cwd=scripts_dir)
        if result.returncode != 0:
            print("Error in split preparation!")
            sys.exit(1)
        print("Split preparation completed!")
    else:
        if not os.path.exists(folds_csv_path):
            print(f"Requested --skip-prepare, but fold CSV does not exist: {folds_csv_path}")
            sys.exit(1)
        print(f"Skipping split preparation. Using existing: {folds_csv_path}")

    print("\n" + "=" * 60)
    print("Model entry: prototype EDL fine-tuning")
    if cli_args.n_folds == 0:
        print("Step 2: Prototype EDL training with holdout validation/test evaluation...")
    else:
        print(f"Step 2: Prototype EDL training with {cli_args.n_folds}-fold CV + auto-testing...")
    print("=" * 60)

    import torch
    from pathlib import Path

    sys.path.insert(0, codebase_dir)
    from edl_proto_trainer import do_prototype_edl_experiments
    from utils import seed_all

    args = argparse.Namespace()
    args.data_dir = cli_args.data_dir
    args.img_dir = cli_args.img_dir
    args.csv_file = csv_filename
    args.clip_chk_pt_path = clip_chk_pt_path
    args.dataset = "custom"
    args.arch = cli_args.arch
    args.label = cli_args.label
    args.tensorboard_path = os.path.join(project_root, cli_args.tensorboard_dir)
    args.checkpoints = os.path.join(project_root, cli_args.model_save_dir)
    args.output_path = os.path.join(project_root, cli_args.csv_save_dir)

    args.model_type = "prototype_edl_classifier"
    args.VER = "prototype_edl"
    args.detector_threshold = 0.1
    args.swin_encoder = "microsoft/swin-tiny-patch4-window7-224"
    args.pretrained_swin_encoder = True
    args.swin_model_type = True

    args.epochs_warmup = 0
    args.num_cycles = 0.5
    args.alpha = 10
    args.sigma = 15
    args.p = 1.0
    args.mean = 0.3089279
    args.std = 0.25053555408335154
    args.focal_alpha = 0.6
    args.focal_gamma = 2.0

    args.n_folds = cli_args.n_folds
    args.seed = cli_args.seed
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.epochs = cli_args.epochs
    args.lr = cli_args.lr
    args.weight_decay = cli_args.weight_decay
    args.warmup_epochs = cli_args.warmup_epochs
    args.img_size = cli_args.img_size
    args.device = cli_args.device
    args.apex = cli_args.apex == "y"
    args.gpu_id = gpu_id
    args.print_freq = 5000
    args.log_freq = 1000
    args.running_interactive = False
    args.data_frac = 1.0
    args.weighted_BCE = cli_args.weighted_bce
    args.patience = cli_args.patience
    args.balanced_dataloader = "n"
    args.train_mode = cli_args.train_mode
    args.freeze_backbone = "y" if args.train_mode == "head_only" else "n"
    args.resume = cli_args.resume
    args.skip_bad_batches = cli_args.skip_bad_batches

    args.num_classes = EDL_NUM_CLASSES
    args.edl_loss_type = EDL_LOSS_TYPE
    args.edl_kl_weight = EDL_KL_WEIGHT
    args.edl_annealing_start = EDL_ANNEALING_START
    args.edl_annealing_epochs = EDL_ANNEALING_EPOCHS
    args.edl_proto_k = cli_args.edl_proto_k
    args.edl_proto_topk = cli_args.edl_proto_topk
    args.edl_proto_temperature = cli_args.edl_proto_temperature
    args.edl_proto_normalize = cli_args.edl_proto_normalize == "y"
    args.edl_proto_init = cli_args.edl_proto_init
    args.edl_proto_attract_weight = cli_args.edl_proto_attract_weight
    args.edl_proto_separation_weight = cli_args.edl_proto_separation_weight
    args.edl_proto_diversity_weight = cli_args.edl_proto_diversity_weight
    args.edl_proto_margin = cli_args.edl_proto_margin
    args.edl_proto_balance_classes = cli_args.edl_proto_balance_classes == "y"

    seed_all(args.seed)

    args.root = (
        f"lr_{args.lr}_epochs_{args.epochs}_prototype_edl_{args.edl_loss_type}_{args.label}_"
        f"k_{args.edl_proto_k}_proto_a{args.edl_proto_attract_weight}_"
        f"s{args.edl_proto_separation_weight}_d{args.edl_proto_diversity_weight}_"
        f"m{args.edl_proto_margin}_bal_{'y' if args.edl_proto_balance_classes else 'n'}_mode_{args.train_mode}"
    )
    chk_pt_path = Path(args.checkpoints) / args.dataset / "prototype_edl_classifier" / args.arch / args.root
    output_path = Path(args.output_path) / args.dataset / "zz" / "prototype_edl_classifier" / args.arch / args.root
    tb_logs_path = Path(args.tensorboard_path) / args.dataset / "prototype_edl_classifier" / args.arch / args.root

    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    print("====================> Prototype EDL Paths <====================")
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
    print(f"edl_proto_k: {args.edl_proto_k}")
    print(f"edl_proto_topk: {args.edl_proto_topk}")
    print(f"edl_proto_temperature: {args.edl_proto_temperature}")
    print(f"edl_proto_normalize: {args.edl_proto_normalize}")
    print(f"edl_proto_init: {args.edl_proto_init}")
    print(f"edl_proto_attract_weight: {args.edl_proto_attract_weight}")
    print(f"edl_proto_separation_weight: {args.edl_proto_separation_weight}")
    print(f"edl_proto_diversity_weight: {args.edl_proto_diversity_weight}")
    print(f"edl_proto_margin: {args.edl_proto_margin}")
    print(f"edl_proto_balance_classes: {args.edl_proto_balance_classes}")
    print(f"train_mode: {args.train_mode}")
    print(f"freeze_backbone: {args.freeze_backbone}")
    print(f"resume: {args.resume}")
    print(f"skip_bad_batches: {args.skip_bad_batches}")
    device = args.device if args.device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
    args.apex = args.apex and str(device).startswith("cuda")

    print("device:", device)
    print("torch version:", torch.__version__)
    print("====================> Prototype EDL Paths <====================")

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    do_prototype_edl_experiments(args, device)

    print("\n" + "=" * 60)
    print("All done! Check the outputs directory for Prototype EDL results:")
    print(f"  - Per-fold predictions: {output_path}/prototype_edl_fold*_all_predictions.csv")
    print(f"  - Ensemble predictions: {output_path}/prototype_edl_ensemble_all_predictions.csv")
    print(f"  - OOF predictions when n_folds>0: {output_path}/prototype_edl_seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv")
    print(f"  - Per-fold metrics: {output_path}/prototype_edl_fold*_metrics.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
