"""
Evidential Deep Learning (EDL) 训练/验证/测试模块

实现完整�?折交叉验证训练流程：
  - 每折训练 �?保存最佳模�?
  - 训练结束后自动调用测试模�?
  - 生成包含evidence、uncertainty、fold列的CSV文件
  - 5折ensemble预测
"""

import gc
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from edl_model import BreastClipEDLClassifier
from edl_loss import EDLLoss
from Datasets.dataset_concepts import MammoDataset, collator_mammo_dataset_w_concepts
from Datasets.dataset_utils import get_eval_transforms, get_transforms
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import auroc
from utils import (
    AverageMeter,
    audit_laterality_label_mixes,
    attach_patient_mean_predictions,
    patient_level_aggregate,
    seed_all,
    timeSince,
)


def _amp_enabled(args, device):
    return bool(args.apex) and str(device).startswith("cuda") and torch.cuda.is_available()


def _cuda_postfix(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return {
            "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
            "CUDA-Util": f"{torch.cuda.utilization(device)}%",
        }
    return {}


def _empty_cuda_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _edl_annealing_gate_epoch(args):
    annealing_start = float(getattr(args, "edl_annealing_start", 0))
    annealing_epochs = max(float(getattr(args, "edl_annealing_epochs", 10)), 1.0)
    return annealing_start + annealing_epochs


def _is_edl_annealing_complete(args, epoch):
    return float(epoch) >= _edl_annealing_gate_epoch(args)


def _edl_annealing_gate_visible_epoch(args):
    return int(math.ceil(_edl_annealing_gate_epoch(args))) + 1


def _assert_finite_tensor(name, tensor, img_paths=None):
    if torch.isfinite(tensor).all():
        return

    message = f"Non-finite values detected in {name}."
    if img_paths:
        preview = ", ".join(str(path) for path in img_paths[:4])
        message += f" Example img_path(s): {preview}"
    raise ValueError(message)


def _fold_best_model_path(args):
    model_name = f'edl_{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc.pth'
    return args.chk_pt_path / model_name


def _fold_last_checkpoint_path(args):
    model_name = f'edl_{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_last.pth'
    return args.chk_pt_path / model_name


def _skip_log_path(args):
    return args.output_path / f'edl_fold{args.cur_fold}_skipped_batches.log'


def _append_skip_log(args, stage, epoch, step, img_paths, reason):
    if not img_paths:
        return

    log_path = _skip_log_path(args)
    with log_path.open("a", encoding="utf-8") as fp:
        for img_path in img_paths:
            fp.write(
                f"stage={stage}\tepoch={epoch + 1}\tstep={step}\timg_path={img_path}\treason={reason}\n"
            )


def _is_recoverable_batch_error(exc):
    recoverable_markers = (
        "Failed to read image",
        "Empty image",
        "Non-finite values detected",
        "Non-finite train loss detected",
        "Non-finite valid loss detected",
        "zero dynamic range image",
    )
    text = str(exc)
    return any(marker in text for marker in recoverable_markers)


def _filter_valid_subbatch(data):
    fallback_flags = data.get("is_fallback", [])
    if not fallback_flags:
        return data, [], []

    valid_indices = [idx for idx, is_fallback in enumerate(fallback_flags) if not is_fallback]
    bad_paths = [data["img_path"][idx] for idx, is_fallback in enumerate(fallback_flags) if is_fallback]
    bad_errors = [data.get("error", [""] * len(fallback_flags))[idx] for idx, is_fallback in enumerate(fallback_flags) if is_fallback]

    if len(valid_indices) == len(fallback_flags):
        return data, [], []

    filtered = {
        "x": data["x"][valid_indices],
        "y": data["y"][valid_indices],
        "img_path": [data["img_path"][idx] for idx in valid_indices],
        "is_fallback": [data["is_fallback"][idx] for idx in valid_indices],
        "error": [data.get("error", [""] * len(fallback_flags))[idx] for idx in valid_indices],
    }
    return filtered, bad_paths, bad_errors


def _valid_indices_from_fallback_flags(fallback_flags, batch_size):
    if fallback_flags:
        return [idx for idx, is_fallback in enumerate(fallback_flags) if not is_fallback]
    return list(range(batch_size))


def _placeholder_prediction_batch(args, batch_size):
    evidence = np.zeros((batch_size, args.num_classes), dtype=np.float32)
    probs = np.full((batch_size, args.num_classes), 1.0 / float(args.num_classes), dtype=np.float32)
    uncertainty = np.ones((batch_size, 1), dtype=np.float32)
    return evidence, probs, uncertainty


def _weighted_bce_enabled(args):
    return str(getattr(args, "weighted_BCE", "n")).strip().lower() == "y"


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return value


def _class_count_summary(df, label_col):
    if df is None:
        return {"total": 0, "positive": 0, "negative": 0, "missing_label": 0}
    if label_col not in df.columns:
        return {"total": int(len(df)), "positive": 0, "negative": 0, "missing_label": int(len(df))}
    labels = pd.to_numeric(df[label_col], errors="coerce")
    return {
        "total": int(len(df)),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "missing_label": int(labels.isna().sum()),
    }


def _edl_run_config_path(args):
    return args.output_path / "edl_run_config.json"


def _read_existing_run_config(args):
    path = _edl_run_config_path(args)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_edl_run_config(args, split_summary=None, fold_diagnostic=None):
    args.output_path.mkdir(parents=True, exist_ok=True)
    config = _read_existing_run_config(args)
    tracked_args = [
        "dataset",
        "model_type",
        "arch",
        "root",
        "run_id",
        "seed",
        "n_folds",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "weighted_BCE",
        "train_mode",
        "freeze_backbone",
        "edl_loss_type",
        "edl_kl_weight",
        "edl_annealing_start",
        "edl_annealing_epochs",
        "edl_dropout",
        "edl_hidden_dim",
        "num_classes",
        "label",
        "patience",
        "skip_bad_batches",
    ]
    config.update(
        {
            "module": "EDL",
            "weighted_BCE_enabled": _weighted_bce_enabled(args),
            "csv_file": str(getattr(args, "_resolved_csv_path", getattr(args, "csv_file", ""))),
            "data_dir": str(getattr(args, "data_dir", "")),
            "img_dir": str(getattr(args, "img_dir", "")),
            "clip_chk_pt_path": str(getattr(args, "clip_chk_pt_path", "")),
            "output_path": str(getattr(args, "output_path", "")),
            "chk_pt_path": str(getattr(args, "chk_pt_path", "")),
            "tb_logs_path": str(getattr(args, "tb_logs_path", "")),
            "args": {name: getattr(args, name, None) for name in tracked_args},
        }
    )
    if split_summary is not None:
        config["split_summary"] = split_summary
    if fold_diagnostic is not None:
        existing = {
            str(item.get("fold")): item
            for item in config.get("fold_diagnostics", [])
            if isinstance(item, dict)
        }
        existing[str(fold_diagnostic.get("fold"))] = fold_diagnostic
        config["fold_diagnostics"] = [
            existing[key] for key in sorted(existing, key=lambda item: int(item) if str(item).lstrip("-").isdigit() else str(item))
        ]

    path = _edl_run_config_path(args)
    path.write_text(json.dumps(_json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _loss_component_meters():
    return {
        "data_loss": AverageMeter(),
        "unweighted_data_loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "total_loss": AverageMeter(),
        "annealing_coef": AverageMeter(),
    }


def _update_loss_component_meters(meters, criterion, batch_size):
    attr_map = {
        "data_loss": "last_data_loss",
        "unweighted_data_loss": "last_unweighted_data_loss",
        "kl_loss": "last_kl_loss",
        "total_loss": "last_total_loss",
        "annealing_coef": "last_annealing_coef",
    }
    for key, attr in attr_map.items():
        value = getattr(criterion, attr, None)
        if value is not None and np.isfinite(float(value)):
            meters[key].update(float(value), batch_size)


def _loss_component_summary(losses, meters, skipped_batches=0):
    stats = {"loss": float(losses.avg), "skipped_batches": int(skipped_batches)}
    for key, meter in meters.items():
        stats[key] = float(meter.avg) if meter.count else float("nan")
    return stats


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def _safe_auroc(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(auroc(labels, scores))
    except Exception:
        return float("nan")


def _score_quantiles(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {
            "score_min": float("nan"),
            "score_q10": float("nan"),
            "score_q25": float("nan"),
            "score_q50": float("nan"),
            "score_q75": float("nan"),
            "score_q90": float("nan"),
            "score_q95": float("nan"),
            "score_max": float("nan"),
            "score_mean": float("nan"),
        }
    return {
        "score_min": float(values.min()),
        "score_q10": float(values.quantile(0.10)),
        "score_q25": float(values.quantile(0.25)),
        "score_q50": float(values.quantile(0.50)),
        "score_q75": float(values.quantile(0.75)),
        "score_q90": float(values.quantile(0.90)),
        "score_q95": float(values.quantile(0.95)),
        "score_max": float(values.max()),
        "score_mean": float(values.mean()),
    }


def _threshold_diagnostics(df, label_col, score_col, prefix, threshold=0.5):
    if df.empty or label_col not in df.columns or score_col not in df.columns:
        return {
            f"{prefix}_sample_n": int(len(df)),
            f"{prefix}_positive_n": 0,
            f"{prefix}_negative_n": 0,
            f"{prefix}_aucroc": float("nan"),
            f"{prefix}_pred_pos_at_0_5": 0,
        }
    labels = pd.to_numeric(df[label_col], errors="coerce")
    scores = pd.to_numeric(df[score_col], errors="coerce")
    valid_mask = labels.notna() & scores.notna() & np.isfinite(scores)
    labels = labels.loc[valid_mask].astype(int).to_numpy()
    scores = scores.loc[valid_mask].astype(float).to_numpy()
    preds = (scores >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    out = {
        f"{prefix}_sample_n": int(len(labels)),
        f"{prefix}_positive_n": int((labels == 1).sum()),
        f"{prefix}_negative_n": int((labels == 0).sum()),
        f"{prefix}_aucroc": _safe_auroc(labels, scores),
        f"{prefix}_bacc_at_0_5": (sensitivity + specificity) / 2.0,
        f"{prefix}_sensitivity_at_0_5": sensitivity,
        f"{prefix}_specificity_at_0_5": specificity,
        f"{prefix}_ppv_at_0_5": _safe_div(tp, tp + fp),
        f"{prefix}_npv_at_0_5": _safe_div(tn, tn + fn),
        f"{prefix}_tp_at_0_5": tp,
        f"{prefix}_tn_at_0_5": tn,
        f"{prefix}_fp_at_0_5": fp,
        f"{prefix}_fn_at_0_5": fn,
        f"{prefix}_pred_pos_at_0_5": int(preds.sum()),
        f"{prefix}_pred_neg_at_0_5": int(len(preds) - preds.sum()),
    }
    for key, value in _score_quantiles(scores).items():
        out[f"{prefix}_{key}"] = value
    for name, mask in [("pos", labels == 1), ("neg", labels == 0)]:
        for key, value in _score_quantiles(scores[mask]).items():
            out[f"{prefix}_{name}_{key}"] = value
    return out


def _append_epoch_history(history, epoch, values):
    old_len = len(history.get("epochs", []))
    history.setdefault("epochs", []).append(epoch)
    for key, existing in list(history.items()):
        if key == "epochs" or not isinstance(existing, list):
            continue
        while len(existing) < old_len:
            existing.append(float("nan"))
        if key not in values and len(existing) == old_len:
            existing.append(float("nan"))
    for key, value in values.items():
        if key not in history:
            history[key] = [float("nan")] * old_len
        history[key].append(value)


def _history_column(history, key, length):
    values = list(history.get(key, []))
    if len(values) < length:
        values.extend([float("nan")] * (length - len(values)))
    return values[:length]


def _load_last_checkpoint_if_available(args, model, optimizer, scheduler, scaler):
    start_epoch = 0
    best_aucroc = 0.0
    epochs_no_improve = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_aucroc": [],
    }

    last_ckpt_path = _fold_last_checkpoint_path(args)
    if not getattr(args, "resume", False) or not last_ckpt_path.exists():
        return start_epoch, best_aucroc, epochs_no_improve, history

    ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler_state = ckpt.get("scaler")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_aucroc = float(ckpt.get("best_aucroc", 0.0))
    epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
    history = ckpt.get("history", history)
    if not _is_edl_annealing_complete(args, int(ckpt.get("epoch", -1))):
        epochs_no_improve = 0

    print(
        f"Resumed fold {args.cur_fold} from {last_ckpt_path} "
        f"(next_epoch={start_epoch + 1}, best_aucroc={best_aucroc:.4f})"
    )
    if not _is_edl_annealing_complete(args, start_epoch - 1):
        print(
            f"Early stopping counter reset on resume; EDL annealing gate opens at visible epoch "
            f"{_edl_annealing_gate_visible_epoch(args)}."
        )
    return start_epoch, best_aucroc, epochs_no_improve, history


def _save_last_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_aucroc, epochs_no_improve, history):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "best_aucroc": best_aucroc,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "train_mode": getattr(args, "train_mode", "full"),
        "freeze_backbone": getattr(args, "freeze_backbone", "n"),
    }
    torch.save(checkpoint, _fold_last_checkpoint_path(args))


def _compute_fold_class_weights(args):
    if str(getattr(args, "weighted_BCE", "n")).lower() != "y":
        return None

    fold_train = args.train_folds
    n_neg = int((fold_train[args.label] == 0).sum())
    n_pos = int((fold_train[args.label] == 1).sum())
    if n_pos <= 0:
        print(f"Fold {args.cur_fold} has no positive samples; using unweighted EDL CE.")
        return None

    w_neg = 1.0
    w_pos = float(n_neg / n_pos)
    print(
        f"Fold {args.cur_fold} class weights -> n_neg={n_neg}, n_pos={n_pos}, "
        f"w_neg={w_neg:.6f}, w_pos={w_pos:.6f}"
    )
    return [w_neg, w_pos]


def _save_fold_loss_curve(args, history):
    if not history["epochs"]:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epochs"], history["train_loss"], label="train_loss", color="tab:blue", linewidth=2)
    ax1.plot(history["epochs"], history["valid_loss"], label="valid_loss", color="tab:orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history["epochs"], history["valid_aucroc"], label="valid_aucroc", color="tab:green", linewidth=2)
    ax2.set_ylabel("AUC-ROC")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    ax1.set_title(f"EDL Fold {args.cur_fold} Training Curve")

    fig.tight_layout()
    curve_path = args.output_path / f"edl_fold{args.cur_fold}_loss_curve.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"Fold {args.cur_fold} loss curve saved to: {curve_path}")


def _fold_indices(args):
    return [0] if args.n_folds == 0 else list(range(args.n_folds))


def _normalized_split(df):
    if "split" not in df.columns:
        return pd.Series("train", index=df.index)
    return df["split"].astype(str).str.strip().str.lower()


def _build_fold_split_view(df, fold, n_folds):
    fold_view = df.copy()
    if "split" not in fold_view.columns:
        fold_view["split"] = "train"
    if n_folds == 0:
        split_values = _normalized_split(fold_view)
        fold_view["split"] = split_values.where(split_values.isin(["val", "test"]), "train")
        return fold_view

    fold_view["split"] = "train"
    fold_view.loc[fold_view["fold"] == fold, "split"] = "val"
    fold_view.loc[fold_view["fold"] == -1, "split"] = "test"
    return fold_view


def _metrics_csv_path(args):
    return args.output_path / f"edl_fold{args.cur_fold}_metrics.csv"


def _save_fold_metrics_csv(args, history, eval_split):
    length = len(history.get("epochs", []))
    metrics_data = {
        "epoch": _history_column(history, "epochs", length),
        "train_loss": _history_column(history, "train_loss", length),
        "eval_loss": _history_column(history, "valid_loss", length),
        "eval_aucroc": _history_column(history, "valid_aucroc", length),
        "eval_split": [eval_split] * length,
    }
    reserved = {"epochs", "train_loss", "valid_loss", "valid_aucroc"}
    for key in sorted(history):
        if key in reserved or not isinstance(history[key], list):
            continue
        metrics_data[key] = _history_column(history, key, length)
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(_metrics_csv_path(args), index=False)
    print(f"Fold {args.cur_fold} metrics saved to: {_metrics_csv_path(args)}")
    return metrics_df


def _record_fold_summary(args, metrics_df, summaries):
    if metrics_df.empty:
        return

    best_idx = metrics_df["eval_aucroc"].idxmax()
    best_row = metrics_df.loc[best_idx]
    summaries.append(
        {
            "fold": args.cur_fold,
            "eval_split": best_row["eval_split"],
            "best_epoch": int(best_row["epoch"]),
            "best_aucroc": float(best_row["eval_aucroc"]),
            "checkpoint_path": str(_fold_best_model_path(args)),
            "metrics_csv": str(_metrics_csv_path(args)),
        }
    )


def do_edl_experiments(args, device):
    """
    EDL experiment entrypoint.
    """
    args.model_base_name = args.arch
    args.data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = args.data_dir / csv_path
    args._resolved_csv_path = csv_path
    args.df = pd.read_csv(csv_path)
    args.df = args.df.fillna(0)
    audit_laterality_label_mixes(args.df, args.label, context="EDL input CSV")
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    split_values = _normalized_split(args.df)
    train_pool_mask = args.df["fold"] >= 0
    holdout_val_mask = train_pool_mask & (split_values == "val")
    holdout_train_mask = train_pool_mask & (split_values != "val") & (split_values != "test")
    test_mask = (args.df["fold"] == -1) | (split_values == "test")

    train_df = args.df[train_pool_mask].reset_index(drop=True)
    holdout_train_df = args.df[holdout_train_mask].reset_index(drop=True)
    holdout_val_df = args.df[holdout_val_mask].reset_index(drop=True)
    test_df = args.df[test_mask].reset_index(drop=True)
    predict_df = args.df.reset_index(drop=True)
    print(f"Training-pool samples: {len(train_df)}")
    if args.n_folds == 0:
        print(f"Holdout train samples: {len(holdout_train_df)}")
        print(f"Holdout val samples: {len(holdout_val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Prediction output samples (all rows): {len(predict_df)}")
    args._edl_split_summary = {
        "train_pool": _class_count_summary(train_df, args.label),
        "holdout_train": _class_count_summary(holdout_train_df, args.label),
        "holdout_val": _class_count_summary(holdout_val_df, args.label),
        "test": _class_count_summary(test_df, args.label),
        "predict_all": _class_count_summary(predict_df, args.label),
    }
    config_path = _write_edl_run_config(args, split_summary=args._edl_split_summary)
    print(f"EDL run config saved to: {config_path}")

    if args.n_folds == 0 and len(holdout_val_df) == 0 and len(test_df) == 0:
        raise ValueError("n_folds=0 requires a non-empty validation split or test split for per-epoch evaluation.")

    oof_df = pd.DataFrame()
    fold_prediction_arrays = []
    fold_summaries = []

    for fold in _fold_indices(args):
        args.cur_fold = fold
        seed_all(args.seed)

        if args.n_folds == 0:
            args.train_folds = holdout_train_df.copy().reset_index(drop=True)
            if len(holdout_val_df) > 0:
                args.valid_folds = holdout_val_df.copy().reset_index(drop=True)
                args.eval_split = "val"
            else:
                args.valid_folds = test_df.copy().reset_index(drop=True)
                args.eval_split = "test"
        else:
            args.train_folds = train_df[train_df["fold"] != fold].reset_index(drop=True)
            args.valid_folds = train_df[train_df["fold"] == fold].reset_index(drop=True)
            args.eval_split = "val"
        print(f"\n=== Fold {fold}: train={len(args.train_folds)}, eval={len(args.valid_folds)} ({args.eval_split}) ===")

        eval_df, metrics_df = edl_train_loop(args, device)
        if args.n_folds > 0:
            oof_df = pd.concat([oof_df, eval_df])
        _record_fold_summary(args, metrics_df, fold_summaries)

        best_model_path = _fold_best_model_path(args)
        if len(predict_df) > 0 and best_model_path.exists():
            fold_results = edl_predict_on_dataset(args, predict_df, best_model_path, device, fold)
            fold_results = _build_fold_split_view(fold_results, fold, args.n_folds)
            fold_prediction_arrays.append(fold_results)
            fold_csv_path = args.output_path / f"edl_fold{fold}_all_predictions.csv"
            fold_results.to_csv(fold_csv_path, index=False)
            print(f"Fold {fold} predictions saved to: {fold_csv_path}")
            print(f"Fold {fold} all-data predictions completed.")

    if args.n_folds > 0 and len(oof_df) > 0:
        oof_df = oof_df.reset_index(drop=True)
        print('\n================ CV (Out-of-Fold) ================')
        oof_agg = patient_level_aggregate(oof_df, args.label, 'prediction_prob')
        aucroc_val = auroc(gt=oof_agg[args.label].values.astype(int), pred=oof_agg['prediction_prob'].values)
        print(f'OOF AUC-ROC: {aucroc_val:.4f}')
        oof_df.to_csv(args.output_path / f'edl_seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv', index=False)

    print("\n" + "=" * 60)
    print("Auto-testing: Generating predictions for all data...")
    print("=" * 60)

    if len(predict_df) > 0 and len(fold_prediction_arrays) > 0:
        ensemble_output = predict_df.copy()
        all_evidence_cols = [f'evidence_{i}' for i in range(args.num_classes)]
        all_alpha_cols = [f'alpha_{i}' for i in range(args.num_classes)]
        all_prob_cols = [f'probability_{i}' for i in range(args.num_classes)]

        image_score_col = 'image_prediction_prob'
        for col in all_evidence_cols + all_alpha_cols + all_prob_cols + ['total_uncertainty', image_score_col]:
            ensemble_output[col] = np.mean([fd[col].values for fd in fold_prediction_arrays], axis=0)

        ensemble_output = attach_patient_mean_predictions(
            ensemble_output,
            ensemble_output[image_score_col].values,
            image_score_col=image_score_col,
        )
        ensemble_output['prediction_label'] = (ensemble_output['prediction_prob'] >= 0.5).astype(int)
        ensemble_csv_path = args.output_path / 'edl_ensemble_all_predictions.csv'
        ensemble_output['model_fold'] = 'ensemble'
        ensemble_output.to_csv(ensemble_csv_path, index=False)
        print(f"\n  Ensemble all-data predictions saved to: {ensemble_csv_path}")

        print(f"\n  Ensemble prediction stats:")
        print(f"    prediction_prob: mean={ensemble_output['prediction_prob'].mean():.4f}, min={ensemble_output['prediction_prob'].min():.4f}, max={ensemble_output['prediction_prob'].max():.4f}")
        print(f"    total_uncertainty: mean={ensemble_output['total_uncertainty'].mean():.4f}, min={ensemble_output['total_uncertainty'].min():.4f}, max={ensemble_output['total_uncertainty'].max():.4f}")
        for i in range(args.num_classes):
            print(f"    evidence_{i}: mean={ensemble_output[f'evidence_{i}'].mean():.4f}")
            print(f"    alpha_{i}: mean={ensemble_output[f'alpha_{i}'].mean():.4f}")
            print(f"    probability_{i}: mean={ensemble_output[f'probability_{i}'].mean():.4f}")
    elif len(predict_df) > 0:
        print("Warning: no fold predictions available; ensemble predictions were not generated.")

    if fold_summaries:
        summary_df = pd.DataFrame(fold_summaries)
        summary_path = args.output_path / 'edl_fold_metrics_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"EDL fold metrics summary saved to: {summary_path}")

    print("\n================ EDL Done! ================")

def edl_train_loop(args, device):
    """
    Single-fold EDL training loop.
    """
    print(f'\n================== EDL fold: {args.cur_fold} training ======================')

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    train_loader, valid_loader = edl_get_dataloader(args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    num_classes = args.num_classes
    print(f"Creating EDL model with {num_classes} classes")
    model = BreastClipEDLClassifier(
        args, ckpt=ckpt, num_classes=num_classes,
        dropout=args.edl_dropout, hidden_dim=args.edl_hidden_dim
    )
    model = model.to(device)
    print(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters: {trainable_param_count:,} / {total_params:,}")
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check args.train_mode/freeze settings.")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs == 1:
        warmup_steps = len(train_loader)
    elif args.warmup_epochs == 0.1:
        warmup_steps = args.epochs
    else:
        warmup_steps = 10
    lr_config = {
        'total_epochs': args.epochs,
        'warmup_steps': warmup_steps,
        'total_steps': len(train_loader) * args.epochs
    }
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(args, device))

    logger = SummaryWriter(args.tb_logs_path / f'edl_fold{args.cur_fold}')
    class_weights = _compute_fold_class_weights(args)
    if _weighted_bce_enabled(args) and class_weights is None:
        raise ValueError(
            f"weighted_BCE=y but class_weights could not be computed for fold {args.cur_fold}. "
            "Check train_folds labels before training."
        )
    _write_edl_run_config(
        args,
        split_summary=getattr(args, "_edl_split_summary", None),
        fold_diagnostic={
            "fold": int(args.cur_fold),
            "eval_split": getattr(args, "eval_split", "val"),
            "train_counts": _class_count_summary(args.train_folds, args.label),
            "eval_counts": _class_count_summary(args.valid_folds, args.label),
            "weighted_BCE_enabled": _weighted_bce_enabled(args),
            "class_weights": class_weights,
            "class_weight_negative": class_weights[0] if class_weights is not None else None,
            "class_weight_positive": class_weights[1] if class_weights is not None and len(class_weights) > 1 else None,
            "best_checkpoint_path": str(_fold_best_model_path(args)),
            "last_checkpoint_path": str(_fold_last_checkpoint_path(args)),
            "metrics_csv": str(_metrics_csv_path(args)),
        },
    )
    criterion = EDLLoss(
        num_classes=num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
    )

    start_epoch, best_aucroc, epochs_no_improve, history = _load_last_checkpoint_if_available(
        args, model, optimizer, scheduler, scaler
    )
    eval_name = getattr(args, 'eval_split', 'val')

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        criterion.current_epoch = epoch
        annealing_complete = _is_edl_annealing_complete(args, epoch)
        if not annealing_complete:
            print(
                f"Epoch {epoch + 1} - early stopping paused until visible epoch "
                f"{_edl_annealing_gate_visible_epoch(args)} while EDL annealing is active."
            )

        train_stats = edl_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, logger, device
        )
        avg_loss = train_stats["loss"]

        valid_stats, predictions = edl_valid_fn(
            valid_loader, model, criterion, args, device, epoch, logger=logger
        )
        avg_val_loss = valid_stats["loss"]
        args.valid_folds = attach_patient_mean_predictions(
            args.valid_folds,
            predictions,
            image_score_col='image_prediction_prob',
        )

        valid_agg = patient_level_aggregate(args.valid_folds, args.label, 'prediction_prob')
        aucroc_val = auroc(valid_agg[args.label].values.astype(int), valid_agg['prediction_prob'].values)
        patient_diag = _threshold_diagnostics(valid_agg, args.label, "prediction_prob", "eval_patient")
        image_diag = _threshold_diagnostics(args.valid_folds, args.label, "image_prediction_prob", "eval_image")
        elapsed = time.time() - start_time

        print(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  '
            f'AUC-ROC: {aucroc_val:.4f}  '
            f"BACC@0.5: {patient_diag['eval_patient_bacc_at_0_5']:.4f}  "
            f"Pred_Pos@0.5: {patient_diag['eval_patient_pred_pos_at_0_5']}  "
            f'time: {elapsed:.0f}s'
        )
        logger.add_scalar(f'{eval_name}/{args.label}/AUC-ROC', aucroc_val, epoch + 1)
        logger.add_scalar(f'{eval_name}/{args.label}/BACC@0.5', patient_diag['eval_patient_bacc_at_0_5'], epoch + 1)
        logger.add_scalar(f'{eval_name}/{args.label}/Pred_Pos@0.5', patient_diag['eval_patient_pred_pos_at_0_5'], epoch + 1)
        logger.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        logger.add_scalar('valid/epoch_loss', avg_val_loss, epoch + 1)
        logger.add_scalar('train/data_loss', train_stats["data_loss"], epoch + 1)
        logger.add_scalar('train/unweighted_data_loss', train_stats["unweighted_data_loss"], epoch + 1)
        logger.add_scalar('train/kl_loss', train_stats["kl_loss"], epoch + 1)
        logger.add_scalar('valid/data_loss', valid_stats["data_loss"], epoch + 1)
        logger.add_scalar('valid/unweighted_data_loss', valid_stats["unweighted_data_loss"], epoch + 1)
        logger.add_scalar('valid/kl_loss', valid_stats["kl_loss"], epoch + 1)

        history_values = {
            "train_loss": avg_loss,
            "valid_loss": avg_val_loss,
            "valid_aucroc": aucroc_val,
            "train_data_loss": train_stats["data_loss"],
            "train_unweighted_data_loss": train_stats["unweighted_data_loss"],
            "train_kl_loss": train_stats["kl_loss"],
            "train_total_loss": train_stats["total_loss"],
            "train_annealing_coef": train_stats["annealing_coef"],
            "train_skipped_batches": train_stats["skipped_batches"],
            "eval_data_loss": valid_stats["data_loss"],
            "eval_unweighted_data_loss": valid_stats["unweighted_data_loss"],
            "eval_kl_loss": valid_stats["kl_loss"],
            "eval_total_loss": valid_stats["total_loss"],
            "eval_annealing_coef": valid_stats["annealing_coef"],
            "eval_skipped_batches": valid_stats["skipped_batches"],
        }
        history_values.update(patient_diag)
        history_values.update(image_diag)
        _append_epoch_history(history, epoch + 1, history_values)

        if epoch == 0 or best_aucroc < aucroc_val:
            best_aucroc = aucroc_val
            epochs_no_improve = 0
            print(f'Epoch {epoch + 1} - Save Best AUC-ROC: {best_aucroc:.4f} Model')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                    'predictions': predictions,
                    'epoch': epoch,
                    'auroc': aucroc_val,
                    'best_aucroc': best_aucroc,
                    'epochs_no_improve': epochs_no_improve,
                    'history': history,
                    'train_mode': getattr(args, "train_mode", "full"),
                    'freeze_backbone': getattr(args, "freeze_backbone", "n"),
                }, _fold_best_model_path(args)
            )
        else:
            epochs_no_improve = epochs_no_improve + 1 if annealing_complete else 0

        _save_last_checkpoint(
            args, model, optimizer, scheduler, scaler, epoch, best_aucroc, epochs_no_improve, history
        )

        if annealing_complete and args.patience > 0 and epochs_no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, '
                  f'best AUC-ROC: {best_aucroc:.4f}')
            break

        print(f'[Fold{args.cur_fold}], Best AUC-ROC: {best_aucroc:.4f}')

    best_model_path = _fold_best_model_path(args)
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location='cpu', weights_only=False)['predictions']
        args.valid_folds = attach_patient_mean_predictions(
            args.valid_folds,
            predictions,
            image_score_col='image_prediction_prob',
        )
    else:
        print(f"Warning: No best model checkpoint found at {best_model_path}")

    metrics_df = _save_fold_metrics_csv(args, history, eval_name)
    _save_fold_loss_curve(args, history)
    logger.close()
    _empty_cuda_cache(device)
    gc.collect()
    return args.valid_folds.copy(), metrics_df

def edl_predict_on_dataset(args, df, model_path, device, fold):
    """
    使用单个EDL模型对全量数据进行预测，返回详细的evidence、alpha、概率、不确定性信�?
    
    Args:
        args: 参数对象
        df: 全量数据DataFrame
        model_path: 模型路径
        device: 计算设备
        fold: 当前fold编号
    
    Returns:
        result_df: 包含所有预测详情的DataFrame
    """
    print(f'\n=== EDL Predicting all data with fold {fold} model ===')

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    model = BreastClipEDLClassifier(
        args, ckpt=ckpt, num_classes=args.num_classes,
        dropout=0.0, hidden_dim=args.edl_hidden_dim
    )
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    predict_dataset = MammoDataset(args=args, df=df, transform=get_eval_transforms(args))
    predict_loader = DataLoader(
        predict_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts
    )

    all_evidence = []
    all_probs = []
    all_uncertainty = []
    amp_enabled = _amp_enabled(args, device)

    with torch.no_grad():
        for step, data in tqdm(enumerate(predict_loader), desc=f"EDL Predicting fold{fold}", total=len(predict_loader)):
            batch_size = len(data.get("img_path", []))
            fallback_flags = list(data.get("is_fallback", []))
            valid_indices = _valid_indices_from_fallback_flags(fallback_flags, batch_size)
            batch_evidence, batch_probs, batch_uncertainty = _placeholder_prediction_batch(args, batch_size)
            try:
                data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
                if fallback_paths:
                    reason = fallback_errors[0] if fallback_errors else "fallback image used during prediction"
                    _append_skip_log(args, "predict_fallback", 0, step, fallback_paths, reason)
                    if data["x"].size(0) == 0:
                        all_evidence.append(batch_evidence)
                        all_probs.append(batch_probs)
                        all_uncertainty.append(batch_uncertainty)
                        print(f"[WARN] Prediction batch contains only fallback images at step {step}: {reason}")
                        continue

                inputs = data['x'].to(device)
                if (
                    args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
                    args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
                    args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
                    args.arch.lower() == "breast_clip_det_b2_period_n_lp"
                ):
                    inputs = inputs.squeeze(1).permute(0, 3, 1, 2)

                _assert_finite_tensor("prediction inputs", inputs, data.get('img_path'))

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    evidence = model(inputs)
                _assert_finite_tensor("prediction evidence", evidence, data.get('img_path'))
                
            # 计算Dirichlet参数
                probs = BreastClipEDLClassifier.compute_probabilities(evidence)
                uncertainty = BreastClipEDLClassifier.compute_uncertainty(evidence)
                _assert_finite_tensor("prediction probabilities", probs, data.get('img_path'))
                _assert_finite_tensor("prediction uncertainty", uncertainty, data.get('img_path'))

                batch_evidence[valid_indices] = evidence.cpu().numpy()
                batch_probs[valid_indices] = probs.cpu().numpy()
                batch_uncertainty[valid_indices] = uncertainty.cpu().numpy()
                all_evidence.append(batch_evidence)
                all_probs.append(batch_probs)
                all_uncertainty.append(batch_uncertainty)
            except Exception as exc:
                if not getattr(args, "skip_bad_batches", False) or not _is_recoverable_batch_error(exc):
                    raise

                _append_skip_log(args, "predict_skip", 0, step, data.get("img_path", []), str(exc))
                all_evidence.append(batch_evidence)
                all_probs.append(batch_probs)
                all_uncertainty.append(batch_uncertainty)
                print(f"[WARN] Prediction batch recovered with placeholders: {exc}")

    evidence_array = np.concatenate(all_evidence)       # [N, K]
    alpha_array = evidence_array + 1                     # [N, K]
    probs_array = np.concatenate(all_probs)              # [N, K]
    uncertainty_array = np.concatenate(all_uncertainty)  # [N, 1]

    # 构建结果DataFrame
    result_df = df.copy()
    result_df['model_fold'] = fold

    num_classes = evidence_array.shape[1]
    for i in range(num_classes):
        result_df[f'evidence_{i}'] = evidence_array[:, i]
        result_df[f'alpha_{i}'] = alpha_array[:, i]
        result_df[f'probability_{i}'] = probs_array[:, i]

    result_df['total_uncertainty'] = uncertainty_array.flatten()
    # 正类概率（类�?），兼容原有格式
    result_df = attach_patient_mean_predictions(
        result_df,
        probs_array[:, -1],
        image_score_col='image_prediction_prob',
    )

    print(f"Fold {fold} prediction stats: "
          f"prob_mean={result_df['prediction_prob'].mean():.4f}, "
          f"uncertainty_mean={result_df['total_uncertainty'].mean():.4f}")

    _empty_cuda_cache(device)
    gc.collect()

    return result_df


def edl_get_dataloader(args):
    """创建训练和验证数据加载器"""
    train_tfm = get_transforms(args)
    val_tfm = get_eval_transforms(args)

    train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
    valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collator_mammo_dataset_w_concepts
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts
    )

    return train_loader, valid_loader


def edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, logger, device):
    """EDL训练一个epoch"""
    model.train()
    amp_enabled = scaler.is_enabled()
    losses = AverageMeter()
    component_meters = _loss_component_meters()
    start = end = time.time()
    skipped_batches = 0

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
        data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
        if fallback_paths:
            reason = fallback_errors[0] if fallback_errors else "fallback image used during training"
            _append_skip_log(args, "train_fallback", epoch, step, fallback_paths, reason)
            if data["x"].size(0) == 0:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"[WARN] Skipping empty train batch at epoch {epoch + 1} step {step}: {reason}")
                continue

        inputs = data['x'].to(device)
        if (
            args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
            args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)

        batch_size = inputs.size(0)
        labels = data['y'].long().to(device)
        if not torch.isfinite(inputs).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "train_skip", epoch, step, data.get('img_path', []), "non-finite train inputs")
            print(f"[WARN] Skipping train batch at epoch {epoch + 1} step {step}: non-finite train inputs")
            continue

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            evidence = model(inputs)
            loss = criterion(evidence, labels)
        if not torch.isfinite(evidence).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "train_skip", epoch, step, data.get('img_path', []), "non-finite train evidence")
            print(f"[WARN] Skipping train batch at epoch {epoch + 1} step {step}: non-finite train evidence")
            continue
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "train_skip", epoch, step, data.get('img_path', []), "non-finite train loss")
            print(f"[WARN] Skipping train batch at epoch {epoch + 1} step {step}: non-finite train loss")
            continue

        losses.update(float(loss.item()), batch_size)
        _update_loss_component_meters(component_meters, criterion, batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()

        postfix = {
            "lr": [optimizer.param_groups[0]['lr']],
            "loss": f"{losses.avg:.4f}",
            "skipped": skipped_batches,
        }
        postfix.update(_cuda_postfix(device))
        progress_iter.set_postfix(postfix)

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)
            logger.add_scalar('train/skipped_batches', skipped_batches, index)

    if skipped_batches > 0:
        print(f"Epoch {epoch + 1}: skipped {skipped_batches} train batch(es).")

    return _loss_component_summary(losses, component_meters, skipped_batches)


def edl_valid_fn(valid_loader, model, criterion, args, device, epoch=1, logger=None):
    """EDL验证一个epoch"""
    losses = AverageMeter()
    component_meters = _loss_component_meters()
    model.eval()
    preds = []
    skipped_batches = 0
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
        raw_batch_size = len(data.get("img_path", []))
        fallback_flags = list(data.get("is_fallback", []))
        valid_indices = _valid_indices_from_fallback_flags(fallback_flags, raw_batch_size)
        batch_pos_probs = np.full(raw_batch_size, 0.5, dtype=np.float32)

        data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
        if fallback_paths:
            reason = fallback_errors[0] if fallback_errors else "fallback image used during validation"
            _append_skip_log(args, "valid_fallback", epoch, step, fallback_paths, reason)
            if data["x"].size(0) == 0:
                skipped_batches += 1
                preds.append(batch_pos_probs)
                print(f"[WARN] Validation batch contains only fallback images at epoch {epoch + 1} step {step}: {reason}")
                continue

        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        if (
            args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
            args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)

        labels = data['y'].long().to(device)

        if not torch.isfinite(inputs).all():
            skipped_batches += 1
            _append_skip_log(args, "valid_skip", epoch, step, data.get('img_path', []), "non-finite valid inputs")
            print(f"[WARN] Recovering valid batch at epoch {epoch + 1} step {step}: non-finite valid inputs")
            loss = torch.tensor(0.0)
            losses.update(float(loss.item()), batch_size)
            preds.append(batch_pos_probs)
            continue
        with torch.no_grad():
            evidence = model(inputs)
            loss = criterion(evidence, labels)

            # 计算正类概率用于AUC-ROC评估
            if not torch.isfinite(evidence).all():
                skipped_batches += 1
                _append_skip_log(args, "valid_skip", epoch, step, data.get('img_path', []), "non-finite valid evidence")
                print(f"[WARN] Recovering valid batch at epoch {epoch + 1} step {step}: non-finite valid evidence")
                loss = torch.tensor(0.0)
                losses.update(float(loss.item()), batch_size)
                preds.append(batch_pos_probs)
                continue
            if not torch.isfinite(loss):
                skipped_batches += 1
                _append_skip_log(args, "valid_skip", epoch, step, data.get('img_path', []), "non-finite valid loss")
                print(f"[WARN] Recovering valid batch at epoch {epoch + 1} step {step}: non-finite valid loss")
                loss = torch.tensor(0.0)
                losses.update(float(loss.item()), batch_size)
                preds.append(batch_pos_probs)
                continue
            probs = BreastClipEDLClassifier.compute_probabilities(evidence)
            if not torch.isfinite(probs).all():
                skipped_batches += 1
                _append_skip_log(args, "valid_skip", epoch, step, data.get('img_path', []), "non-finite valid probabilities")
                print(f"[WARN] Recovering valid batch at epoch {epoch + 1} step {step}: non-finite valid probabilities")
                loss = torch.tensor(0.0)
                losses.update(float(loss.item()), batch_size)
                preds.append(batch_pos_probs)
                continue
            # 取正类（类别1）的概率
            pos_probs = probs[:, -1].cpu().numpy()

        losses.update(loss.item(), batch_size)
        _update_loss_component_meters(component_meters, criterion, batch_size)
        batch_pos_probs[valid_indices] = pos_probs
        preds.append(batch_pos_probs)

        postfix = {
            "loss": f"{losses.avg:.4f}",
        }
        postfix.update(_cuda_postfix(device))
        progress_iter.set_postfix(postfix)

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    predictions = np.concatenate(preds)
    if skipped_batches > 0:
        print(f"Epoch {epoch + 1}: recovered {skipped_batches} valid batch(es).")

    return _loss_component_summary(losses, component_meters, skipped_batches), predictions



