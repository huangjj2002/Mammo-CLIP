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
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    append_run_id,
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


def _training_schedule(args):
    return str(getattr(args, "edl_training_schedule", "joint") or "joint").strip().lower()


def _fold_stage_best_checkpoint_path(args, stage_name):
    safe_stage = str(stage_name).replace(" ", "_")
    model_name = f"edl_{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_{safe_stage}_stage_best.pth"
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


def _class_weight_mode(args):
    mode = getattr(args, "class_weight_mode", None)
    if mode is None or str(mode).strip() == "":
        return "inverse" if _weighted_bce_enabled(args) else "none"
    mode = str(mode).strip().lower()
    if mode not in {"none", "inverse", "effective"}:
        raise ValueError(f"Unsupported class_weight_mode: {mode}")
    return mode


def _balanced_sampler_mode(args):
    mode = getattr(args, "balanced_sampler", None)
    if mode is None:
        mode = getattr(args, "balanced_dataloader", "none")
    mode = str(mode).strip().lower()
    if mode in {"n", "no", "false", "0", "none", ""}:
        return "none"
    if mode in {"y", "yes", "true", "1", "image"}:
        return "image"
    raise ValueError(f"Unsupported balanced_sampler: {mode}")


def _prediction_group_cols(args):
    group_cols = getattr(args, "prediction_group_cols", "patient_id")
    if isinstance(group_cols, str):
        group_cols = [col.strip() for col in group_cols.split(",") if col.strip()]
    else:
        group_cols = [str(col).strip() for col in group_cols if str(col).strip()]
    return group_cols or ["patient_id"]


def _prediction_score_agg(args):
    score_agg = str(getattr(args, "prediction_score_agg", "mean") or "mean").strip().lower()
    if score_agg not in {"mean", "max"}:
        raise ValueError(f"Unsupported prediction_score_agg: {score_agg}. Use mean or max.")
    return score_agg


def _prediction_threshold(args):
    return float(getattr(args, "prediction_threshold", 0.5))


def _prediction_aggregation_config(args):
    return {
        "prediction_group_cols": _prediction_group_cols(args),
        "prediction_score_agg": _prediction_score_agg(args),
        "prediction_threshold": _prediction_threshold(args),
    }


def _attach_prediction_scores(args, df, image_scores, image_score_col="image_prediction_prob"):
    return attach_patient_mean_predictions(
        df,
        image_scores,
        image_score_col=image_score_col,
        group_cols=_prediction_group_cols(args),
        score_agg=_prediction_score_agg(args),
        threshold=_prediction_threshold(args),
    )


def _aggregate_prediction_scores(args, df, label_col, score_col):
    return patient_level_aggregate(
        df,
        label_col,
        score_col,
        group_cols=_prediction_group_cols(args),
        score_agg=_prediction_score_agg(args),
    )


def _best_metric_name(args):
    metric = str(getattr(args, "edl_best_metric", "eval_aucroc") or "eval_aucroc").strip()
    aliases = {"aucroc": "eval_aucroc", "auroc": "eval_aucroc", "valid_aucroc": "eval_aucroc"}
    return aliases.get(metric.lower(), metric)


def _current_best_metric_value(args, history_values, aucroc_value):
    metric_name = _best_metric_name(args)
    if metric_name == "eval_aucroc":
        return float(aucroc_value)
    if metric_name not in history_values:
        available = sorted(key for key, value in history_values.items() if np.isscalar(value))
        raise KeyError(
            f"Requested --edl-best-metric {metric_name!r} is not available. "
            f"Available scalar metrics include: {available}"
        )
    return float(history_values[metric_name])


def _load_best_metric_value_from_checkpoint(args, checkpoint, default=0.0):
    metric_name = _best_metric_name(args)
    checkpoint_metric = checkpoint.get("best_metric_name")
    if checkpoint_metric == metric_name:
        return float(checkpoint.get("best_metric_value", checkpoint.get("best_aucroc", default)))
    if checkpoint_metric is None and metric_name == "eval_aucroc":
        return float(checkpoint.get("best_aucroc", default))
    if checkpoint_metric is not None:
        print(
            f"Checkpoint best metric {checkpoint_metric!r} differs from requested {metric_name!r}; "
            "resetting best metric value for this resume."
        )
    return float(default)


def _best_metric_column(args, metrics_df):
    metric_name = _best_metric_name(args)
    if metric_name == "eval_aucroc":
        return "eval_aucroc"
    if metric_name not in metrics_df.columns:
        raise KeyError(f"Best metric {metric_name!r} is missing from metrics CSV columns.")
    return metric_name


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
        "balanced_sampler",
        "balanced_dataloader",
        "class_weight_mode",
        "effective_beta",
        "edl_focal_gamma",
        "train_mode",
        "freeze_backbone",
        "edl_loss_type",
        "edl_kl_weight",
        "edl_annealing_start",
        "edl_annealing_epochs",
        "edl_dropout",
        "edl_hidden_dim",
        "prediction_group_cols",
        "prediction_score_agg",
        "prediction_threshold",
        "edl_best_metric",
        "edl_training_schedule",
        "bce_warmstart_path",
        "bce_stage_epochs",
        "bce_stage_lr",
        "bce_stage_patience",
        "edl_stage_patience",
        "staged_freeze_encoder",
        "num_classes",
        "label",
        "patience",
        "skip_bad_batches",
    ]
    config.update(
        {
            "module": "EDL",
            "weighted_BCE_enabled": _weighted_bce_enabled(args),
            "class_weight_mode_resolved": _class_weight_mode(args),
            "class_weight_info": getattr(args, "_edl_class_weight_info", None),
            "balanced_sampler_resolved": _balanced_sampler_mode(args),
            "balanced_sampler_stats": getattr(args, "_balanced_sampler_stats", None),
            "prediction_aggregation": _prediction_aggregation_config(args),
            "best_metric_name": _best_metric_name(args),
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
        "class_weighted_data_loss": AverageMeter(),
        "focal_data_loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "total_loss": AverageMeter(),
        "annealing_coef": AverageMeter(),
        "focal_factor_mean": AverageMeter(),
        "sample_weight_mean": AverageMeter(),
        "focal_weighted_denominator": AverageMeter(),
    }


def _update_loss_component_meters(meters, criterion, batch_size):
    attr_map = {
        "data_loss": "last_data_loss",
        "unweighted_data_loss": "last_unweighted_data_loss",
        "class_weighted_data_loss": "last_class_weighted_data_loss",
        "focal_data_loss": "last_focal_data_loss",
        "kl_loss": "last_kl_loss",
        "total_loss": "last_total_loss",
        "annealing_coef": "last_annealing_coef",
        "focal_factor_mean": "last_focal_factor_mean",
        "sample_weight_mean": "last_sample_weight_mean",
        "focal_weighted_denominator": "last_focal_weighted_denominator",
    }
    for key, attr in attr_map.items():
        value = getattr(criterion, attr, None)
        if value is not None and np.isfinite(float(value)):
            meters[key].update(float(value), batch_size)
    class_counts = getattr(criterion, "last_class_counts", None)
    class_attr_map = {
        "data_loss_mean": "last_class_data_loss_means",
        "weighted_loss_mean": "last_class_weighted_loss_means",
        "focal_weighted_loss_mean": "last_class_focal_weighted_loss_means",
        "focal_factor_mean": "last_class_focal_factor_means",
    }
    if class_counts is None:
        return
    for class_idx, class_count in enumerate(class_counts):
        if int(class_count) <= 0:
            continue
        for suffix, attr in class_attr_map.items():
            values = getattr(criterion, attr, None)
            if values is None or class_idx >= len(values):
                continue
            value = values[class_idx]
            if value is not None and np.isfinite(float(value)):
                key = f"label{class_idx}_{suffix}"
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(float(value), int(class_count))


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


def _evidence_store():
    return {"labels": [], "evidence": [], "probability": [], "uncertainty": []}


def _record_evidence_batch(store, labels, evidence):
    with torch.no_grad():
        evidence = evidence.detach().float()
        probs = BreastClipEDLClassifier.compute_probabilities(evidence)
        uncertainty = BreastClipEDLClassifier.compute_uncertainty(evidence)
        store["labels"].append(labels.detach().long().cpu().numpy())
        store["evidence"].append(evidence.cpu().numpy())
        store["probability"].append(probs.cpu().numpy())
        store["uncertainty"].append(uncertainty.cpu().numpy().reshape(-1))


def _summary_stats(prefix, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_q10": float("nan"),
            f"{prefix}_q50": float("nan"),
            f"{prefix}_q90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
    }


def _evidence_summary(store, prefix, num_classes):
    if not store["labels"]:
        return {}

    labels = np.concatenate(store["labels"]).astype(int)
    evidence = np.concatenate(store["evidence"], axis=0).astype(float)
    probs = np.concatenate(store["probability"], axis=0).astype(float)
    uncertainty = np.concatenate(store["uncertainty"]).astype(float)
    alpha_sum = evidence.sum(axis=1) + float(num_classes)

    out = {}
    for label_value in range(int(num_classes)):
        mask = labels == label_value
        label_prefix = f"{prefix}_label{label_value}"
        out[f"{label_prefix}_n"] = int(mask.sum())
        for class_idx in range(int(num_classes)):
            out.update(_summary_stats(f"{label_prefix}_evidence_{class_idx}", evidence[mask, class_idx]))
        out.update(_summary_stats(f"{label_prefix}_alpha_sum", alpha_sum[mask]))
        out.update(_summary_stats(f"{label_prefix}_total_uncertainty", uncertainty[mask]))
        out.update(_summary_stats(f"{label_prefix}_probability_1", probs[mask, -1]))
    return out


def _threshold_diagnostics(df, label_col, score_col, prefix, threshold=0.5):
    if df.empty or label_col not in df.columns or score_col not in df.columns:
        return {
            f"{prefix}_sample_n": int(len(df)),
            f"{prefix}_positive_n": 0,
            f"{prefix}_negative_n": 0,
            f"{prefix}_aucroc": float("nan"),
            f"{prefix}_pred_pos_at_0_5": 0,
            f"{prefix}_threshold": float(threshold),
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
        f"{prefix}_threshold": float(threshold),
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
    best_aucroc = _load_best_metric_value_from_checkpoint(args, ckpt, default=0.0)
    epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
    history = ckpt.get("history", history)
    if not _is_edl_annealing_complete(args, int(ckpt.get("epoch", -1))):
        epochs_no_improve = 0

    print(
        f"Resumed fold {args.cur_fold} from {last_ckpt_path} "
        f"(next_epoch={start_epoch + 1}, best_{_best_metric_name(args)}={best_aucroc:.4f})"
    )
    if not _is_edl_annealing_complete(args, start_epoch - 1):
        print(
            f"Early stopping counter reset on resume; EDL annealing gate opens at visible epoch "
            f"{_edl_annealing_gate_visible_epoch(args)}."
        )
    return start_epoch, best_aucroc, epochs_no_improve, history


def _save_last_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_aucroc,
    epochs_no_improve,
    history,
    stage_name="joint",
    stage_epoch=None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "stage": stage_name,
        "stage_epoch": stage_epoch,
        "best_aucroc": best_aucroc,
        "best_metric_name": _best_metric_name(args),
        "best_metric_value": best_aucroc,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "training_schedule": _training_schedule(args),
        "bce_warmstart_path": getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
        "train_mode": getattr(args, "train_mode", "full"),
        "freeze_backbone": getattr(args, "freeze_backbone", "n"),
    }
    torch.save(checkpoint, _fold_last_checkpoint_path(args))


def _compute_fold_class_weights(args):
    mode = _class_weight_mode(args)
    fold_train = args.train_folds
    n_neg = int((fold_train[args.label] == 0).sum())
    n_pos = int((fold_train[args.label] == 1).sum())
    info = {
        "fold": int(getattr(args, "cur_fold", -1)),
        "mode": mode,
        "n_neg": n_neg,
        "n_pos": n_pos,
        "effective_beta": getattr(args, "effective_beta", None),
        "class_weights": None,
    }

    if mode == "none":
        args._edl_class_weight_info = info
        return None

    if n_pos <= 0:
        print(f"Fold {args.cur_fold} has no positive samples; using unweighted EDL CE.")
        args._edl_class_weight_info = info
        return None
    if n_neg <= 0:
        print(f"Fold {args.cur_fold} has no negative samples; using unweighted EDL CE.")
        args._edl_class_weight_info = info
        return None

    if mode == "inverse":
        w_neg = 1.0
        w_pos = float(n_neg / n_pos)
    else:
        beta = float(getattr(args, "effective_beta", 0.9999))
        if beta < 0.0 or beta >= 1.0:
            raise ValueError("--effective-beta must be in [0, 1).")
        if beta == 0.0:
            weights = np.array([1.0, 1.0], dtype=np.float64)
        else:
            effective_neg = (1.0 - np.power(beta, n_neg)) / (1.0 - beta)
            effective_pos = (1.0 - np.power(beta, n_pos)) / (1.0 - beta)
            weights = np.array([1.0 / effective_neg, 1.0 / effective_pos], dtype=np.float64)
        weights = weights / np.mean(weights)
        w_neg = float(weights[0])
        w_pos = float(weights[1])

    class_weights = [w_neg, w_pos]
    info["class_weights"] = class_weights
    args._edl_class_weight_info = info
    print(
        f"Fold {args.cur_fold} class weights ({mode}) -> n_neg={n_neg}, n_pos={n_pos}, "
        f"w_neg={w_neg:.6f}, w_pos={w_pos:.6f}"
    )
    return class_weights


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


def _stage_patience(args, stage_name, default=None):
    if stage_name == "edl_head":
        return int(getattr(args, "edl_stage_patience", getattr(args, "patience", default if default is not None else 0)))
    if stage_name == "joint":
        return int(getattr(args, "patience", default if default is not None else 0))
    return int(default if default is not None else 0)


def _auto_bce_run_id(args):
    return f"{getattr(args, 'run_id', 'edl')}_bce_warmstart"


def _auto_bce_root(args):
    base_root = (
        f"lr_{args.bce_stage_lr}_epochs_{args.bce_stage_epochs}_weighted_BCE_{args.weighted_BCE}_"
        f"{args.label}_data_frac_{getattr(args, 'data_frac', 1.0)}"
    )
    root, _ = append_run_id(base_root, _auto_bce_run_id(args))
    return root


def _auto_bce_checkpoint_dir(args):
    return Path(args.checkpoints) / args.dataset / "classifier" / args.arch / _auto_bce_root(args)


def _bce_checkpoint_candidates(path, fold):
    if not path.exists():
        return []
    patterns = [f"*fold{fold}_best_aucroc*.pth", f"*fold{fold}_best_acc*.pth"]
    candidates = []
    for pattern in patterns:
        candidates.extend(path.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(path.rglob(pattern))
    unique = {str(candidate.resolve()): candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_bce_warmstart_path(args, fold):
    path_value = getattr(args, "bce_warmstart_path", None)
    if not path_value:
        raise ValueError("staged EDL needs a BCE warm-start path after Stage 1 setup.")

    path_text = str(path_value)
    if "{fold}" in path_text:
        path_text = path_text.format(fold=fold)
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(getattr(args, "project_root", Path.cwd())) / path

    if path.is_file():
        resolved = path
    elif path.is_dir():
        candidates = _bce_checkpoint_candidates(path, fold)
        if not candidates:
            raise FileNotFoundError(f"No BCE checkpoint matching fold{fold} found in {path}.")
        if len(candidates) > 1:
            print(f"[WARN] Multiple BCE checkpoints found for fold{fold}; using newest: {candidates[0]}")
        resolved = candidates[0]
    else:
        raise FileNotFoundError(f"BCE warm-start path does not exist: {path}")

    args._resolved_bce_warmstart_path = str(resolved)
    return resolved


def _run_bce_warmstart_training(args):
    codebase_dir = Path(getattr(args, "codebase_dir", Path(__file__).parent))
    train_script = codebase_dir / "train_classifier.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--tensorboard-path",
        str(args.tensorboard_path),
        "--checkpoints",
        str(args.checkpoints),
        "--output_path",
        str(args.output_path),
        "--data-dir",
        str(args.data_dir),
        "--img-dir",
        str(args.img_dir),
        "--clip_chk_pt_path",
        str(args.clip_chk_pt_path),
        "--csv-file",
        str(args.csv_file),
        "--dataset",
        str(args.dataset),
        "--data_frac",
        str(getattr(args, "data_frac", 1.0)),
        "--arch",
        str(args.arch),
        "--label",
        str(args.label),
        "--VER",
        "bce_warmstart",
        "--n_folds",
        str(args.n_folds),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.bce_stage_epochs),
        "--lr",
        str(args.bce_stage_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--img-size",
        *[str(size) for size in args.img_size],
        "--device",
        str(args.device),
        "--apex",
        "y" if getattr(args, "apex", False) else "n",
        "--print-freq",
        str(args.print_freq),
        "--log-freq",
        str(args.log_freq),
        "--model-type",
        "classifier",
        "--weighted-BCE",
        str(args.weighted_BCE),
        "--patience",
        str(args.bce_stage_patience),
        "--balanced-dataloader",
        str(getattr(args, "balanced_dataloader", "n")),
        "--run-id",
        _auto_bce_run_id(args),
    ]
    print("\n================ Stage 1: BCE warm-start training ================")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(codebase_dir))
    if result.returncode != 0:
        raise RuntimeError(f"BCE warm-start training failed with exit code {result.returncode}.")
    checkpoint_dir = _auto_bce_checkpoint_dir(args)
    print(f"BCE warm-start checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def _ensure_bce_warmstart(args):
    if _training_schedule(args) != "staged":
        return None

    if getattr(args, "bce_warmstart_path", None):
        return Path(str(args.bce_warmstart_path))

    checkpoint_dir = _auto_bce_checkpoint_dir(args)
    missing_folds = []
    for fold in _fold_indices(args):
        if not _bce_checkpoint_candidates(checkpoint_dir, fold):
            missing_folds.append(fold)
    if missing_folds:
        checkpoint_dir = _run_bce_warmstart_training(args)
    else:
        print(f"Reusing existing automatic BCE warm-start directory: {checkpoint_dir}")

    args.bce_warmstart_path = str(checkpoint_dir)
    return checkpoint_dir


def _record_fold_summary(args, metrics_df, summaries):
    if metrics_df.empty:
        return

    metric_col = _best_metric_column(args, metrics_df)
    best_idx = metrics_df[metric_col].idxmax()
    best_row = metrics_df.loc[best_idx]
    summaries.append(
        {
            "fold": args.cur_fold,
            "eval_split": best_row["eval_split"],
            "best_epoch": int(best_row["epoch"]),
            "best_aucroc": float(best_row["eval_aucroc"]),
            "best_metric_name": metric_col,
            "best_metric_value": float(best_row[metric_col]),
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
    _ensure_bce_warmstart(args)

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
        oof_agg = _aggregate_prediction_scores(args, oof_df, args.label, 'prediction_prob')
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

        ensemble_output = _attach_prediction_scores(
            args,
            ensemble_output,
            ensemble_output[image_score_col].values,
            image_score_col=image_score_col,
        )
        ensemble_output['prediction_label'] = (ensemble_output['prediction_prob'] >= _prediction_threshold(args)).astype(int)
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


def _create_edl_model(args, ckpt, device):
    num_classes = args.num_classes
    print(f"Creating EDL model with {num_classes} classes")
    model = BreastClipEDLClassifier(
        args,
        ckpt=ckpt,
        num_classes=num_classes,
        dropout=args.edl_dropout,
        hidden_dim=args.edl_hidden_dim,
    )
    return model.to(device)


def _load_bce_warmstart_for_fold(args, model):
    bce_checkpoint_path = _resolve_bce_warmstart_path(args, args.cur_fold)
    checkpoint = torch.load(bce_checkpoint_path, map_location="cpu", weights_only=False)
    loaded_count = model.load_bce_encoder_state(checkpoint, strict=True)
    print(f"Loaded {loaded_count} image_encoder tensors from BCE warm-start: {bce_checkpoint_path}")
    return bce_checkpoint_path


def _build_stage_optimizer_scheduler(args, model, train_loader, stage_epochs, lr, device):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters: {trainable_param_count:,} / {total_params:,}")
    if not trainable_params:
        raise ValueError("No trainable parameters found for this EDL stage.")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    if args.warmup_epochs == 1:
        warmup_steps = len(train_loader)
    elif args.warmup_epochs == 0.1:
        warmup_steps = stage_epochs
    else:
        warmup_steps = 10
    lr_config = {
        "total_epochs": stage_epochs,
        "warmup_steps": warmup_steps,
        "total_steps": len(train_loader) * stage_epochs,
    }
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(args, device))
    return optimizer, scheduler, scaler


def _build_edl_criterion(args, class_weights):
    return EDLLoss(
        num_classes=args.num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
        focal_gamma=getattr(args, "edl_focal_gamma", 0.0),
    )


def _run_edl_stage(
    args,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    logger,
    device,
    history,
    eval_name,
    stage_name,
    stage_epochs,
    best_aucroc,
    epochs_no_improve,
):
    stage_patience = _stage_patience(args, stage_name)
    stage_best_path = _fold_stage_best_checkpoint_path(args, stage_name)
    stage_best_metric = float("-inf")
    stage_no_improve = 0
    best_metric_name = _best_metric_name(args)
    print(
        f"\n================ Stage: {stage_name} ({stage_epochs} epoch(s), "
        f"patience={stage_patience}) ================"
    )

    for epoch in range(stage_epochs):
        start_time = time.time()
        criterion.current_epoch = epoch
        annealing_complete = _is_edl_annealing_complete(args, epoch)
        if not annealing_complete:
            print(
                f"Epoch {epoch + 1} [{stage_name}] - EDL annealing is active; "
                f"stage early stopping still uses patience={stage_patience}."
            )

        train_stats = edl_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, logger, device
        )
        avg_loss = train_stats["loss"]

        valid_stats, predictions = edl_valid_fn(
            valid_loader, model, criterion, args, device, epoch, logger=logger
        )
        avg_val_loss = valid_stats["loss"]
        args.valid_folds = _attach_prediction_scores(
            args,
            args.valid_folds,
            predictions,
            image_score_col="image_prediction_prob",
        )

        valid_agg = _aggregate_prediction_scores(args, args.valid_folds, args.label, "prediction_prob")
        aucroc_val = auroc(valid_agg[args.label].values.astype(int), valid_agg["prediction_prob"].values)
        threshold = _prediction_threshold(args)
        patient_diag = _threshold_diagnostics(valid_agg, args.label, "prediction_prob", "eval_patient", threshold)
        image_diag = _threshold_diagnostics(args.valid_folds, args.label, "image_prediction_prob", "eval_image", threshold)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1} [{stage_name}] - avg_train_loss: {avg_loss:.4f}  "
            f"avg_{eval_name}_loss: {avg_val_loss:.4f}  AUC-ROC: {aucroc_val:.4f}  "
            f"BACC@0.5: {patient_diag['eval_patient_bacc_at_0_5']:.4f}  "
            f"Pred_Pos@0.5: {patient_diag['eval_patient_pred_pos_at_0_5']}  time: {elapsed:.0f}s"
        )
        logger.add_scalar(f"{eval_name}/{args.label}/AUC-ROC", aucroc_val, epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/BACC@0.5", patient_diag["eval_patient_bacc_at_0_5"], epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/Pred_Pos@0.5", patient_diag["eval_patient_pred_pos_at_0_5"], epoch + 1)
        logger.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
        logger.add_scalar("valid/epoch_loss", avg_val_loss, epoch + 1)
        logger.add_scalar("train/data_loss", train_stats["data_loss"], epoch + 1)
        logger.add_scalar("train/unweighted_data_loss", train_stats["unweighted_data_loss"], epoch + 1)
        logger.add_scalar("train/kl_loss", train_stats["kl_loss"], epoch + 1)
        logger.add_scalar("valid/data_loss", valid_stats["data_loss"], epoch + 1)
        logger.add_scalar("valid/unweighted_data_loss", valid_stats["unweighted_data_loss"], epoch + 1)
        logger.add_scalar("valid/kl_loss", valid_stats["kl_loss"], epoch + 1)

        history_values = {
            "stage": stage_name,
            "stage_epoch": epoch + 1,
            "train_loss": avg_loss,
            "valid_loss": avg_val_loss,
            "valid_aucroc": aucroc_val,
            "train_data_loss": train_stats["data_loss"],
            "train_unweighted_data_loss": train_stats["unweighted_data_loss"],
            "train_class_weighted_data_loss": train_stats["class_weighted_data_loss"],
            "train_focal_data_loss": train_stats["focal_data_loss"],
            "train_kl_loss": train_stats["kl_loss"],
            "train_total_loss": train_stats["total_loss"],
            "train_annealing_coef": train_stats["annealing_coef"],
            "train_focal_factor_mean": train_stats["focal_factor_mean"],
            "train_sample_weight_mean": train_stats["sample_weight_mean"],
            "train_focal_weighted_denominator": train_stats["focal_weighted_denominator"],
            "train_skipped_batches": train_stats["skipped_batches"],
            "eval_data_loss": valid_stats["data_loss"],
            "eval_unweighted_data_loss": valid_stats["unweighted_data_loss"],
            "eval_class_weighted_data_loss": valid_stats["class_weighted_data_loss"],
            "eval_focal_data_loss": valid_stats["focal_data_loss"],
            "eval_kl_loss": valid_stats["kl_loss"],
            "eval_total_loss": valid_stats["total_loss"],
            "eval_annealing_coef": valid_stats["annealing_coef"],
            "eval_focal_factor_mean": valid_stats["focal_factor_mean"],
            "eval_sample_weight_mean": valid_stats["sample_weight_mean"],
            "eval_focal_weighted_denominator": valid_stats["focal_weighted_denominator"],
            "eval_skipped_batches": valid_stats["skipped_batches"],
            "prediction_threshold": threshold,
            "prediction_score_agg": _prediction_score_agg(args),
            "prediction_group_cols": ",".join(_prediction_group_cols(args)),
            "best_metric_name": best_metric_name,
        }
        history_values.update(patient_diag)
        history_values.update(image_diag)
        history_values.update({f"train_{key}": value for key, value in train_stats.items() if key.startswith("label")})
        history_values.update({f"eval_{key}": value for key, value in valid_stats.items() if key.startswith("label")})
        history_values.update({key: value for key, value in train_stats.items() if key.startswith("train_")})
        history_values.update({key: value for key, value in valid_stats.items() if key.startswith("eval_")})
        best_metric_value = _current_best_metric_value(args, history_values, aucroc_val)
        history_values["selected_best_metric_value"] = best_metric_value
        _append_epoch_history(history, epoch + 1, history_values)

        improved = np.isfinite(best_metric_value) and (not np.isfinite(stage_best_metric) or stage_best_metric < best_metric_value)
        if improved:
            stage_best_metric = best_metric_value
            stage_no_improve = 0
            best_aucroc = best_metric_value
            epochs_no_improve = 0
            print(
                f"Epoch {epoch + 1} - Save Best {best_metric_name}: {best_aucroc:.4f} "
                f"(AUC-ROC: {aucroc_val:.4f}) EDL Model"
            )
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "predictions": predictions,
                "epoch": epoch,
                "stage": stage_name,
                "stage_epoch": epoch + 1,
                "auroc": aucroc_val,
                "best_aucroc": aucroc_val,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_aucroc,
                "prediction_aggregation": _prediction_aggregation_config(args),
                "epochs_no_improve": epochs_no_improve,
                "history": history,
                "training_schedule": _training_schedule(args),
                "bce_warmstart_path": getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
                "train_mode": getattr(args, "train_mode", "full"),
                "freeze_backbone": getattr(args, "freeze_backbone", "n"),
            }
            torch.save(checkpoint, _fold_best_model_path(args))
            torch.save(checkpoint, stage_best_path)
        else:
            stage_no_improve += 1
            epochs_no_improve = stage_no_improve

        _save_last_checkpoint(
            args,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_aucroc,
            epochs_no_improve,
            history,
            stage_name=stage_name,
            stage_epoch=epoch + 1,
        )

        if stage_patience > 0 and stage_no_improve >= stage_patience:
            print(
                f"Stage early stopping at epoch {epoch + 1} [{stage_name}]: "
                f"no improvement for {stage_patience} epochs, best {best_metric_name}: {stage_best_metric:.4f}"
            )
            break

        print(f"[Fold{args.cur_fold}], Best {_best_metric_name(args)}: {best_aucroc:.4f}")

    return best_aucroc, epochs_no_improve


def _edl_staged_train_loop(args, device):
    print(f"\n================== EDL fold: {args.cur_fold} staged training ======================")
    if getattr(args, "resume", False):
        print("[WARN] --resume is ignored for staged EDL; staged runs restart from the BCE warm-start checkpoint.")

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    train_loader, valid_loader = edl_get_dataloader(args)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}")

    model = _create_edl_model(args, ckpt, device)
    bce_checkpoint_path = _load_bce_warmstart_for_fold(args, model)
    if getattr(args, "staged_freeze_encoder", True):
        model.set_encoder_trainable(False)
        print("Stage setup: BCE encoder frozen; training EDL head only.")
    else:
        model.set_encoder_trainable(True)
        print("Stage setup: BCE encoder remains trainable.")
    model.set_edl_head_trainable(True)
    print(model)

    logger = SummaryWriter(args.tb_logs_path / f"edl_fold{args.cur_fold}")
    class_weights = _compute_fold_class_weights(args)
    class_weight_mode = _class_weight_mode(args)
    if class_weight_mode != "none" and class_weights is None:
        raise ValueError(
            f"class_weight_mode={class_weight_mode} but class_weights could not be computed for fold {args.cur_fold}. "
            "Check train_folds labels before training."
        )
    _write_edl_run_config(
        args,
        split_summary=getattr(args, "_edl_split_summary", None),
        fold_diagnostic={
            "fold": int(args.cur_fold),
            "eval_split": getattr(args, "eval_split", "val"),
            "training_schedule": "staged",
            "bce_warmstart_path": str(bce_checkpoint_path),
            "bce_stage_patience": getattr(args, "bce_stage_patience", None),
            "edl_stage_patience": getattr(args, "edl_stage_patience", None),
            "staged_freeze_encoder": bool(getattr(args, "staged_freeze_encoder", True)),
            "train_counts": _class_count_summary(args.train_folds, args.label),
            "eval_counts": _class_count_summary(args.valid_folds, args.label),
            "weighted_BCE_enabled": _weighted_bce_enabled(args),
            "class_weight_mode": class_weight_mode,
            "class_weight_info": getattr(args, "_edl_class_weight_info", None),
            "class_weights": class_weights,
            "balanced_sampler": _balanced_sampler_mode(args),
            "balanced_sampler_stats": getattr(args, "_balanced_sampler_stats", None),
            "edl_focal_gamma": getattr(args, "edl_focal_gamma", 0.0),
            "prediction_aggregation": _prediction_aggregation_config(args),
            "best_metric_name": _best_metric_name(args),
            "best_checkpoint_path": str(_fold_best_model_path(args)),
            "last_checkpoint_path": str(_fold_last_checkpoint_path(args)),
            "metrics_csv": str(_metrics_csv_path(args)),
        },
    )

    optimizer, scheduler, scaler = _build_stage_optimizer_scheduler(args, model, train_loader, args.epochs, args.lr, device)
    criterion = _build_edl_criterion(args, class_weights)
    history = {"epochs": [], "train_loss": [], "valid_loss": [], "valid_aucroc": []}
    eval_name = getattr(args, "eval_split", "val")
    best_aucroc, epochs_no_improve = _run_edl_stage(
        args,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        logger,
        device,
        history,
        eval_name,
        stage_name="edl_head",
        stage_epochs=args.epochs,
        best_aucroc=0.0,
        epochs_no_improve=0,
    )

    best_model_path = _fold_best_model_path(args)
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location="cpu", weights_only=False)["predictions"]
        args.valid_folds = _attach_prediction_scores(
            args,
            args.valid_folds,
            predictions,
            image_score_col="image_prediction_prob",
        )
    else:
        print(f"Warning: No best model checkpoint found at {best_model_path}")

    metrics_df = _save_fold_metrics_csv(args, history, eval_name)
    _save_fold_loss_curve(args, history)
    logger.close()
    _empty_cuda_cache(device)
    gc.collect()
    return args.valid_folds.copy(), metrics_df


def edl_train_loop(args, device):
    """
    Single-fold EDL training loop.
    """
    if _training_schedule(args) == "staged":
        return _edl_staged_train_loop(args, device)

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
    class_weight_mode = _class_weight_mode(args)
    if class_weight_mode != "none" and class_weights is None:
        raise ValueError(
            f"class_weight_mode={class_weight_mode} but class_weights could not be computed for fold {args.cur_fold}. "
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
            "class_weight_mode": class_weight_mode,
            "class_weight_info": getattr(args, "_edl_class_weight_info", None),
            "class_weights": class_weights,
            "class_weight_negative": class_weights[0] if class_weights is not None else None,
            "class_weight_positive": class_weights[1] if class_weights is not None and len(class_weights) > 1 else None,
            "balanced_sampler": _balanced_sampler_mode(args),
            "balanced_sampler_stats": getattr(args, "_balanced_sampler_stats", None),
            "edl_focal_gamma": getattr(args, "edl_focal_gamma", 0.0),
            "prediction_aggregation": _prediction_aggregation_config(args),
            "best_metric_name": _best_metric_name(args),
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
        focal_gamma=getattr(args, "edl_focal_gamma", 0.0),
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
        args.valid_folds = _attach_prediction_scores(
            args,
            args.valid_folds,
            predictions,
            image_score_col='image_prediction_prob',
        )

        valid_agg = _aggregate_prediction_scores(args, args.valid_folds, args.label, 'prediction_prob')
        aucroc_val = auroc(valid_agg[args.label].values.astype(int), valid_agg['prediction_prob'].values)
        threshold = _prediction_threshold(args)
        patient_diag = _threshold_diagnostics(valid_agg, args.label, "prediction_prob", "eval_patient", threshold)
        image_diag = _threshold_diagnostics(args.valid_folds, args.label, "image_prediction_prob", "eval_image", threshold)
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
        for loss_key in (
            "class_weighted_data_loss",
            "focal_data_loss",
            "focal_factor_mean",
            "sample_weight_mean",
        ):
            if loss_key in train_stats and np.isfinite(float(train_stats[loss_key])):
                logger.add_scalar(f"train/{loss_key}", train_stats[loss_key], epoch + 1)
            if loss_key in valid_stats and np.isfinite(float(valid_stats[loss_key])):
                logger.add_scalar(f"valid/{loss_key}", valid_stats[loss_key], epoch + 1)

        history_values = {
            "stage": "joint",
            "stage_epoch": epoch + 1,
            "train_loss": avg_loss,
            "valid_loss": avg_val_loss,
            "valid_aucroc": aucroc_val,
            "train_data_loss": train_stats["data_loss"],
            "train_unweighted_data_loss": train_stats["unweighted_data_loss"],
            "train_class_weighted_data_loss": train_stats["class_weighted_data_loss"],
            "train_focal_data_loss": train_stats["focal_data_loss"],
            "train_kl_loss": train_stats["kl_loss"],
            "train_total_loss": train_stats["total_loss"],
            "train_annealing_coef": train_stats["annealing_coef"],
            "train_focal_factor_mean": train_stats["focal_factor_mean"],
            "train_sample_weight_mean": train_stats["sample_weight_mean"],
            "train_focal_weighted_denominator": train_stats["focal_weighted_denominator"],
            "train_skipped_batches": train_stats["skipped_batches"],
            "eval_data_loss": valid_stats["data_loss"],
            "eval_unweighted_data_loss": valid_stats["unweighted_data_loss"],
            "eval_class_weighted_data_loss": valid_stats["class_weighted_data_loss"],
            "eval_focal_data_loss": valid_stats["focal_data_loss"],
            "eval_kl_loss": valid_stats["kl_loss"],
            "eval_total_loss": valid_stats["total_loss"],
            "eval_annealing_coef": valid_stats["annealing_coef"],
            "eval_focal_factor_mean": valid_stats["focal_factor_mean"],
            "eval_sample_weight_mean": valid_stats["sample_weight_mean"],
            "eval_focal_weighted_denominator": valid_stats["focal_weighted_denominator"],
            "eval_skipped_batches": valid_stats["skipped_batches"],
            "prediction_threshold": threshold,
            "prediction_score_agg": _prediction_score_agg(args),
            "prediction_group_cols": ",".join(_prediction_group_cols(args)),
            "best_metric_name": _best_metric_name(args),
        }
        history_values.update(patient_diag)
        history_values.update(image_diag)
        history_values.update({f"train_{key}": value for key, value in train_stats.items() if key.startswith("label")})
        history_values.update({f"eval_{key}": value for key, value in valid_stats.items() if key.startswith("label")})
        history_values.update({key: value for key, value in train_stats.items() if key.startswith("train_")})
        history_values.update({key: value for key, value in valid_stats.items() if key.startswith("eval_")})
        best_metric_name = _best_metric_name(args)
        best_metric_value = _current_best_metric_value(args, history_values, aucroc_val)
        history_values["selected_best_metric_value"] = best_metric_value
        _append_epoch_history(history, epoch + 1, history_values)

        if np.isfinite(best_metric_value) and (epoch == 0 or best_aucroc < best_metric_value):
            best_aucroc = best_metric_value
            epochs_no_improve = 0
            print(
                f'Epoch {epoch + 1} - Save Best {best_metric_name}: {best_aucroc:.4f} '
                f'(AUC-ROC: {aucroc_val:.4f}) Model'
            )
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                    'predictions': predictions,
                    'epoch': epoch,
                    'auroc': aucroc_val,
                    'best_aucroc': aucroc_val,
                    'best_metric_name': best_metric_name,
                    'best_metric_value': best_aucroc,
                    'prediction_aggregation': _prediction_aggregation_config(args),
                    'epochs_no_improve': epochs_no_improve,
                    'history': history,
                    'training_schedule': _training_schedule(args),
                    'bce_warmstart_path': getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
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
                  f'best {_best_metric_name(args)}: {best_aucroc:.4f}')
            break

        print(f'[Fold{args.cur_fold}], Best {_best_metric_name(args)}: {best_aucroc:.4f}')

    best_model_path = _fold_best_model_path(args)
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location='cpu', weights_only=False)['predictions']
        args.valid_folds = _attach_prediction_scores(
            args,
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
    result_df = _attach_prediction_scores(
        args,
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

    sampler = None
    shuffle_train = True
    sampler_mode = _balanced_sampler_mode(args)
    args._balanced_sampler_stats = {
        "mode": sampler_mode,
        "enabled": False,
        "num_samples": int(len(args.train_folds)),
    }
    if sampler_mode == "image":
        labels = pd.to_numeric(args.train_folds[args.label], errors="coerce")
        pos_mask = (labels == 1).to_numpy()
        neg_mask = (labels == 0).to_numpy()
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        weights = np.zeros(len(labels), dtype=np.float64)
        if n_pos <= 0 or n_neg <= 0:
            print(
                f"[WARN] balanced-sampler=image disabled for fold {args.cur_fold}: "
                f"n_pos={n_pos}, n_neg={n_neg}."
            )
        else:
            weights[pos_mask] = 0.5 / float(n_pos)
            weights[neg_mask] = 0.5 / float(n_neg)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle_train = False
            args._balanced_sampler_stats.update(
                {
                    "enabled": True,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "expected_positive_fraction": 0.5,
                    "replacement": True,
                }
            )
            print(
                f"Fold {args.cur_fold} balanced image sampler -> "
                f"n_pos={n_pos}, n_neg={n_neg}, num_samples={len(weights)}, replacement=True"
            )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle_train, sampler=sampler,
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
    evidence_store = _evidence_store()
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
        _record_evidence_batch(evidence_store, labels, evidence)

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

    stats = _loss_component_summary(losses, component_meters, skipped_batches)
    stats.update(_evidence_summary(evidence_store, "train", args.num_classes))
    return stats


def edl_valid_fn(valid_loader, model, criterion, args, device, epoch=1, logger=None):
    """EDL验证一个epoch"""
    losses = AverageMeter()
    component_meters = _loss_component_meters()
    evidence_store = _evidence_store()
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
        _record_evidence_batch(evidence_store, labels, evidence)
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

    stats = _loss_component_summary(losses, component_meters, skipped_batches)
    stats.update(_evidence_summary(evidence_store, "eval", args.num_classes))
    return stats, predictions



