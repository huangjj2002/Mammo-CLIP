
# =============================================================================
# ========================= 配置区域 ==============================
# =============================================================================


CSV_PATH          = r"/opt/localdata/Data/dh/dh_preprocessed/hjj_images/embed_data_testcohort_enriched.csv"    # 原始CSV文件路径
DATA_DIR          = r"/opt/localdata/Data/dh/dh_preprocessed/hjj_images"                                 # 数据根目录
IMG_DIR           = "images_png"                                   # 图片目录
CLIP_CHK_PT_PATH  = "./model/b5-model-best-epoch-7.tar"          # Mammo-CLIP预训练权重路径


LABEL             = "cancer"                         
ARCH              = "breast_clip_det_b5_period_n_ft"  


N_FOLDS           = 5         # 交叉验证折数
EPOCHS            = 25         # 每折最大训练轮数
PATIENCE          = 5        # 早停
BATCH_SIZE        = 16         
LR                = 5e-5      
SEED              = 42       
WEIGHTED_BCE      = "y"       


IMG_SIZE          = [912, 1520]  


DEVICE            = "cuda"    
NUM_WORKERS       = 4         
APEX              = "y"       
GPU_ID            = 0         # 使用的GPU编号


SKIP_PREPARE      = False    



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
        except Exception as e:
            print(f"[NLTK WARNING] Failed to download punkt: {e}")
            print("[NLTK WARNING] If the server cannot connect to the internet, you can manually download punkt:")
            print("  Method 1: Run  python -c \"import nltk; nltk.download('punkt')\"")
            print("  Method 2: Download from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.zip")
            print("             and extract to ~/nltk_data/tokenizers/punkt/")


def main():

    ensure_nltk_punkt()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    print(f"[GPU] Using GPU {GPU_ID} (CUDA_VISIBLE_DEVICES={GPU_ID})")

    project_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(project_root, "src", "scripts")
    codebase_dir = os.path.join(project_root, "src", "codebase")


    clip_chk_pt_path = CLIP_CHK_PT_PATH
    if not os.path.isabs(clip_chk_pt_path):
        clip_chk_pt_path = os.path.abspath(clip_chk_pt_path)


    folds_csv_path = os.path.join(DATA_DIR, "train_with_test_folds.csv")
    csv_filename = "train_with_test_folds.csv"


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