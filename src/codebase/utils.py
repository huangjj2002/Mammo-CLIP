import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(args):
    return "cuda" if args.device == "cuda" else 'cpu'


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_Paths(args):
    chk_pt_path = Path(f"{args.checkpoints}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")
    output_path = Path(f"{args.output_path}/{args.dataset}/zz/{args.model_type}/{args.arch}/{args.root}")
    tb_logs_path = Path(f"{args.tensorboard_path}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")

    return chk_pt_path, output_path, tb_logs_path


def resolve_run_id(run_id=None):
    if run_id is None or str(run_id).strip() == "":
        run_id = time.strftime("%Y%m%d_%H%M%S")
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id).strip())
    return run_id.strip("_") or time.strftime("%Y%m%d_%H%M%S")


def append_run_id(root, run_id=None, compact=False):
    resolved_run_id = resolve_run_id(run_id)
    if compact and run_id is not None and str(run_id).strip() != "":
        return f"run_{resolved_run_id}", resolved_run_id
    return f"{root}_run_{resolved_run_id}", resolved_run_id


def _normalize_group_cols(group_col="patient_id", group_cols=None):
    if group_cols is None:
        group_cols = group_col
    if isinstance(group_cols, str):
        group_cols = [col.strip() for col in group_cols.split(",") if col.strip()]
    else:
        group_cols = [str(col).strip() for col in group_cols if str(col).strip()]
    return group_cols or ["patient_id"]


def _normalize_score_agg(score_agg):
    score_agg = str(score_agg or "mean").strip().lower()
    if score_agg not in {"mean", "max"}:
        raise ValueError(f"Unsupported score aggregation: {score_agg}. Use mean or max.")
    return score_agg


def patient_level_aggregate(df, label_col, score_col, group_col="patient_id", group_cols=None, score_agg="mean"):
    group_cols = _normalize_group_cols(group_col=group_col, group_cols=group_cols)
    score_agg = _normalize_score_agg(score_agg)
    missing = [col for col in (*group_cols, label_col, score_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) for patient-level aggregation: {missing}")

    if df.empty:
        return pd.DataFrame(columns=[*group_cols, label_col, score_col])

    return (
        df.groupby(group_cols, dropna=False)
        .agg({label_col: "max", score_col: score_agg})
        .reset_index()
    )


def attach_patient_mean_predictions(
    df,
    image_scores,
    image_score_col="image_prediction_prob",
    patient_score_col="patient_prediction_prob",
    output_score_col="prediction_prob",
    label_col="prediction_label",
    group_col="patient_id",
    group_cols=None,
    score_agg="mean",
    threshold=0.5,
):
    group_cols = _normalize_group_cols(group_col=group_col, group_cols=group_cols)
    score_agg = _normalize_score_agg(score_agg)
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) for prediction aggregation: {missing}")

    result_df = df.copy()
    result_df[image_score_col] = np.asarray(image_scores)
    result_df[patient_score_col] = result_df.groupby(group_cols, dropna=False)[image_score_col].transform(score_agg)
    result_df[output_score_col] = result_df[patient_score_col]
    result_df[label_col] = (result_df[output_score_col] >= threshold).astype(int)
    result_df["prediction_group_cols"] = ",".join(group_cols)
    result_df["prediction_score_agg"] = score_agg
    result_df["prediction_threshold"] = float(threshold)
    result_df["prediction_image_score_col"] = image_score_col
    return result_df


def audit_laterality_label_mixes(df, label_col, context="input", group_cols=("patient_id", "laterality")):
    missing = [col for col in (*group_cols, label_col) if col not in df.columns]
    if missing:
        return

    grouped = df.groupby(list(group_cols))[label_col].agg(["min", "max"])
    mixed_count = int((grouped["min"] != grouped["max"]).sum())
    if mixed_count:
        print(
            f"[AUDIT] {context}: found {mixed_count} {group_cols} groups with mixed {label_col} labels. "
            "Labels are kept row-level to match the original classifier path."
        )


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
