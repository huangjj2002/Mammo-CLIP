import argparse
import os
import subprocess
import sys


CSV_PATH = r"/home/dhao4/workspace/hjj_workspace/data/data.csv"
DATA_DIR = r"/home/dhao4/workspace/hjj_workspace/data"
IMG_DIR = "images_png"
CLIP_CHK_PT_PATH = "./model/b5-model-best-epoch-7.tar"

LABEL = "cancer"
ARCH = "breast_clip_det_b5_period_n_ft"

N_FOLDS = 0
EPOCHS = 25
PATIENCE = 3
BATCH_SIZE = 8
LR = 5e-5
SEED = 42
WEIGHTED_BCE = "y"

IMG_SIZE = [912, 1520]

DEVICE = "cuda"
NUM_WORKERS = 4
APEX = "y"
GPU_ID = 4

SKIP_PREPARE = True
FOLDS_CSV_PATH = None
OVERWRITE_FOLDS = False
SPLIT_MODE = "cohort"
COHORT_COL = "cohort_num"
TRAIN_COHORTS = "1-8"
TEST_COHORTS = "9-10"


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
    parser = argparse.ArgumentParser(description="Run Mammo-CLIP finetuning.")
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
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--weighted-bce", default=WEIGHTED_BCE, help="Whether to use weighted BCE: y/n.")
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
    return parser


def main():
    cli_args = build_parser().parse_args()
    ensure_nltk_punkt()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu_id)
    print(f"[GPU] Using GPU {cli_args.gpu_id} (CUDA_VISIBLE_DEVICES={cli_args.gpu_id})")

    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "src", "scripts")
    codebase_dir = os.path.join(project_root, "src", "codebase")

    clip_chk_pt_path = cli_args.clip_chk_pt_path
    if not os.path.isabs(clip_chk_pt_path):
        clip_chk_pt_path = os.path.abspath(clip_chk_pt_path)

    folds_csv_path = cli_args.folds_csv_path or os.path.join(project_root, "train_with_test_folds.csv")
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
            "--train-cohorts", cli_args.train_cohorts,
            "--test-cohorts", cli_args.test_cohorts,
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
    if cli_args.n_folds == 0:
        print("Step 2: Training classifier with train/test evaluation...")
    else:
        print(f"Step 2: Training classifier with {cli_args.n_folds}-fold CV...")
    print("=" * 60)
    train_cmd = [
        sys.executable,
        os.path.join(codebase_dir, "train_classifier.py"),
        "--data-dir", cli_args.data_dir,
        "--img-dir", cli_args.img_dir,
        "--csv-file", csv_filename,
        "--clip_chk_pt_path", clip_chk_pt_path,
        "--dataset", "custom",
        "--arch", cli_args.arch,
        "--label", cli_args.label,
        "--n_folds", str(cli_args.n_folds),
        "--epochs", str(cli_args.epochs),
        "--batch-size", str(cli_args.batch_size),
        "--img-size", str(cli_args.img_size[0]), str(cli_args.img_size[1]),
        "--lr", str(cli_args.lr),
        "--seed", str(cli_args.seed),
        "--weighted-BCE", cli_args.weighted_bce,
        "--patience", str(cli_args.patience),
        "--num-workers", str(cli_args.num_workers),
        "--device", cli_args.device,
        "--apex", cli_args.apex,
    ]
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd=codebase_dir)
    if result.returncode != 0:
        print("Error in training!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All done! Check the outputs directory for results.")
    print("  - OOF-style predictions: outputs/custom/zz/classifier/.../*_oof_outputs.csv")
    print("  - Holdout test predictions when n_folds=0: outputs/custom/zz/classifier/.../*_test_outputs.csv")
    print("  - Per-fold loss curves: outputs/custom/zz/classifier/.../fold*_loss_curve.png")
    print("  - Combined loss curves: outputs/custom/zz/classifier/.../loss_curves_summary.png")
    print("  - Per-fold all-data predictions: outputs/custom/zz/classifier/.../fold*_all_predictions.csv")
    print("  - Ensemble all-data predictions: outputs/custom/zz/classifier/.../ensemble_all_predictions.csv")
    print("  - Per-fold metrics: outputs/custom/zz/classifier/.../fold*_metrics.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
