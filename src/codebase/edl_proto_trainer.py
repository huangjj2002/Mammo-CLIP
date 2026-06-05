"""
Prototype + EDL training, validation, and prediction helpers.

This module mirrors the existing EDL data flow and output conventions while
adding per-class prototype explanations.
"""

import gc
import json
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
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasets.dataset_concepts import MammoDataset, collator_mammo_dataset_w_concepts
from Datasets.dataset_utils import get_eval_transforms
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from edl_loss import EDLLoss
from edl_proto_model import BreastClipPrototypeEDLClassifier
from edl_trainer import (
    _aggregate_prediction_scores,
    _amp_enabled,
    _append_epoch_history,
    _append_skip_log,
    _assert_finite_tensor,
    _attach_prediction_scores,
    _best_metric_column,
    _best_metric_name,
    _class_weight_mode,
    _current_best_metric_value,
    _cuda_postfix,
    _edl_annealing_gate_visible_epoch,
    _evidence_store,
    _evidence_summary,
    _is_edl_annealing_complete,
    _json_safe,
    _load_best_metric_value_from_checkpoint,
    _compute_fold_class_weights,
    _empty_cuda_cache,
    _filter_valid_subbatch,
    _is_recoverable_batch_error,
    _loss_component_meters,
    _loss_component_summary,
    _placeholder_prediction_batch,
    _prediction_aggregation_config,
    _prediction_group_cols,
    _prediction_score_agg,
    _prediction_threshold,
    _record_evidence_batch,
    _threshold_diagnostics,
    _update_loss_component_meters,
    _valid_indices_from_fallback_flags,
    _wrong_evidence_penalty_config,
    edl_get_dataloader,
    edl_valid_fn,
)
from metrics import auroc
from utils import (
    AverageMeter,
    append_run_id,
    audit_laterality_label_mixes,
    seed_all,
    timeSince,
)


PROTO_HISTORY_KEYS = (
    "train_edl_loss",
    "train_proto_loss",
    "train_proto_loss_raw",
    "train_proto_attract_loss",
    "train_proto_separation_loss",
    "train_proto_diversity_loss",
)


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


def _fold_best_model_path(args):
    model_name = f"{args.model_base_name}_prototype_edl_seed_{args.seed}_fold{args.cur_fold}_best_aucroc.pth"
    return args.chk_pt_path / model_name


def _fold_last_checkpoint_path(args):
    model_name = f"{args.model_base_name}_prototype_edl_seed_{args.seed}_fold{args.cur_fold}_last_checkpoint.pth"
    return args.chk_pt_path / model_name


def _fold_stage_best_checkpoint_path(args, stage_name):
    safe_stage = str(stage_name).replace(" ", "_")
    model_name = f"{args.model_base_name}_prototype_edl_seed_{args.seed}_fold{args.cur_fold}_{safe_stage}_stage_best.pth"
    return args.chk_pt_path / model_name


def _metrics_csv_path(args):
    return args.output_path / f"prototype_edl_fold{args.cur_fold}_metrics.csv"


def _training_schedule(args):
    return str(getattr(args, "proto_training_schedule", "joint") or "joint").strip().lower()


def _stage_manifest_path(args):
    return args.output_path / "prototype_edl_stage_manifest.json"


def _write_stage_manifest(args, updates=None, fold_updates=None):
    path = _stage_manifest_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    if updates:
        manifest.update(_json_safe(updates))
    if fold_updates is not None:
        fold_key = str(getattr(args, "cur_fold", "unknown"))
        manifest.setdefault("folds", {})
        current = manifest["folds"].get(fold_key, {})
        current.update(_json_safe(fold_updates))
        manifest["folds"][fold_key] = current

    path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return path


def _stage_patience(args, stage_name, default=None):
    if stage_name == "prototype_warmup":
        return int(getattr(args, "proto_warmup_patience", default if default is not None else 0))
    if stage_name == "edl_calibration":
        return int(getattr(args, "edl_stage_patience", getattr(args, "patience", default if default is not None else 0)))
    if stage_name == "joint":
        return int(getattr(args, "patience", default if default is not None else 0))
    return int(default if default is not None else 0)


def _empty_history():
    history = {"epochs": [], "train_loss": [], "valid_loss": [], "valid_aucroc": []}
    for key in PROTO_HISTORY_KEYS:
        history[key] = []
    return history


def _ensure_proto_history_keys(history):
    epoch_count = len(history.get("epochs", []))
    for key in PROTO_HISTORY_KEYS:
        values = list(history.get(key, []))
        if len(values) < epoch_count:
            values = values + [np.nan] * (epoch_count - len(values))
        history[key] = values
    return history


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
        "prototype_initialized_from": getattr(args, "_prototype_initialized_from", None),
        "train_mode": getattr(args, "train_mode", "full"),
        "freeze_backbone": getattr(args, "freeze_backbone", "n"),
        "edl_proto_k": getattr(args, "edl_proto_k", None),
        "edl_proto_temperature": getattr(args, "edl_proto_temperature", None),
        "edl_proto_normalize": getattr(args, "edl_proto_normalize", None),
        "edl_proto_attract_weight": getattr(args, "edl_proto_attract_weight", None),
        "edl_proto_separation_weight": getattr(args, "edl_proto_separation_weight", None),
        "edl_proto_diversity_weight": getattr(args, "edl_proto_diversity_weight", None),
        "edl_proto_loss_weight": getattr(args, "edl_proto_loss_weight", 1.0),
        "edl_proto_margin": getattr(args, "edl_proto_margin", None),
        "edl_proto_balance_classes": getattr(args, "edl_proto_balance_classes", None),
        "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
    }
    torch.save(checkpoint, _fold_last_checkpoint_path(args))


def _load_last_checkpoint_if_available(args, model, optimizer, scheduler, scaler):
    history = _empty_history()
    if not getattr(args, "resume", False):
        return 0, 0.0, 0, history

    checkpoint_path = _fold_last_checkpoint_path(args)
    if not checkpoint_path.exists():
        print(f"No prototype EDL checkpoint to resume for fold {args.cur_fold}: {checkpoint_path}")
        return 0, 0.0, 0, history

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler.is_enabled() and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    history = _ensure_proto_history_keys(checkpoint.get("history", history))
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_aucroc = _load_best_metric_value_from_checkpoint(args, checkpoint, default=0.0)
    epochs_no_improve = int(checkpoint.get("epochs_no_improve", 0))
    if not _is_edl_annealing_complete(args, int(checkpoint.get("epoch", -1))):
        epochs_no_improve = 0
    print(
        f"Resumed prototype EDL fold {args.cur_fold} from {checkpoint_path} "
        f"(next_epoch={start_epoch + 1}, best_{_best_metric_name(args)}={best_aucroc:.4f})"
    )
    if not _is_edl_annealing_complete(args, start_epoch - 1):
        print(
            f"Early stopping counter reset on resume; EDL annealing gate opens at visible epoch "
            f"{_edl_annealing_gate_visible_epoch(args)}."
        )
    return start_epoch, best_aucroc, epochs_no_improve, history


def _save_fold_metrics_csv(args, history, eval_split):
    metric_data = {
        "epoch": history["epochs"],
        "train_loss": history["train_loss"],
        "eval_loss": history["valid_loss"],
        "eval_aucroc": history["valid_aucroc"],
        "eval_split": [eval_split] * len(history["epochs"]),
        "edl_proto_loss_weight": [getattr(args, "edl_proto_loss_weight", 1.0)] * len(history["epochs"]),
    }
    reserved = {"epochs", "train_loss", "valid_loss", "valid_aucroc"}
    for key in sorted(history):
        if key in reserved or not isinstance(history[key], list):
            continue
        values = history.get(key, [])
        if len(values) == len(history["epochs"]):
            metric_data[key] = values

    metrics_df = pd.DataFrame(metric_data)
    metrics_df.to_csv(_metrics_csv_path(args), index=False)
    print(f"Fold {args.cur_fold} metrics saved to: {_metrics_csv_path(args)}")
    return metrics_df


def _auto_bce_run_id(args):
    return f"{getattr(args, 'run_id', 'prototype_edl')}_bce_warmstart"


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
        raise ValueError("staged mode needs a BCE warm-start path after Stage 1 setup.")

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

    manifest_updates = {
        "training_schedule": "staged",
        "checkpoint_selection_metric": _best_metric_name(args),
        "prediction_aggregation": _prediction_aggregation_config(args),
        "bce_stage": {
            "provided_warmstart_path": getattr(args, "bce_warmstart_path", None),
            "epochs": getattr(args, "bce_stage_epochs", None),
            "lr": getattr(args, "bce_stage_lr", None),
            "patience": getattr(args, "bce_stage_patience", None),
            "weighted_BCE": getattr(args, "weighted_BCE", None),
        },
        "prototype_warmup_stage": {
            "epochs": getattr(args, "proto_warmup_epochs", None),
            "patience": getattr(args, "proto_warmup_patience", None),
            "kl_weight": 0.0,
            "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
            "freeze_encoder": bool(getattr(args, "staged_freeze_encoder", True)),
        },
        "edl_calibration_stage": {
            "epochs": getattr(args, "epochs", None),
            "patience": getattr(args, "edl_stage_patience", getattr(args, "patience", None)),
            "kl_weight": getattr(args, "edl_kl_weight", None),
            "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
            "freeze_encoder": bool(getattr(args, "staged_freeze_encoder", True)),
            "freeze_prototypes": bool(getattr(args, "edl_stage_freeze_prototypes", True)),
        },
    }

    if getattr(args, "bce_warmstart_path", None):
        manifest_updates["bce_stage"]["source"] = "provided"
        _write_stage_manifest(args, manifest_updates)
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
    manifest_updates["bce_stage"]["source"] = "auto_trained"
    manifest_updates["bce_stage"]["auto_checkpoint_dir"] = str(checkpoint_dir)
    _write_stage_manifest(args, manifest_updates)
    return checkpoint_dir


def _save_fold_loss_curve(args, history):
    if not history["epochs"]:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epochs"], history["train_loss"], label="train_loss", color="tab:blue", linewidth=2)
    ax.plot(history["epochs"], history["valid_loss"], label="eval_loss", color="tab:orange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Prototype EDL Fold {args.cur_fold} Training Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    curve_path = args.output_path / f"prototype_edl_fold{args.cur_fold}_loss_curve.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"Fold {args.cur_fold} loss curve saved to: {curve_path}")


def _record_fold_summary(args, metrics_df, summaries):
    if metrics_df.empty:
        return

    summary_metrics = metrics_df
    if _training_schedule(args) == "staged" and "stage" in metrics_df.columns:
        non_warmup_metrics = metrics_df[metrics_df["stage"] != "prototype_warmup"]
        if not non_warmup_metrics.empty:
            summary_metrics = non_warmup_metrics

    metric_col = _best_metric_column(args, summary_metrics)
    best_idx = summary_metrics[metric_col].idxmax()
    best_row = summary_metrics.loc[best_idx]
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


def _prepare_inputs_for_arch(args, inputs):
    if (
        args.arch.lower() == "breast_clip_det_b5_period_n_ft"
        or args.arch.lower() == "breast_clip_det_b5_period_n_lp"
        or args.arch.lower() == "breast_clip_det_b2_period_n_ft"
        or args.arch.lower() == "breast_clip_det_b2_period_n_lp"
    ):
        return inputs.squeeze(1).permute(0, 3, 1, 2)
    return inputs


def _class_balanced_mean(values, labels, num_classes, balance_classes):
    if not balance_classes:
        return values.mean()

    class_means = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.any():
            class_means.append(values[class_mask].mean())
    if not class_means:
        return values.mean()
    return torch.stack(class_means).mean()


def _prototype_diversity_loss(model, margin):
    prototypes = model.prototype_head.prototypes
    if model.prototype_head.normalize:
        prototypes = F.normalize(prototypes, p=2, dim=-1)

    per_class_losses = []
    for class_idx in range(prototypes.shape[0]):
        if prototypes.shape[1] < 2:
            continue
        pairwise_distance = torch.pdist(prototypes[class_idx], p=2)
        per_class_losses.append(F.relu(margin - pairwise_distance).pow(2).mean())

    if not per_class_losses:
        return prototypes.new_zeros(())
    return torch.stack(per_class_losses).mean()


def _compute_prototype_regularization(model, details, labels, args):
    distances = details["prototype_distance"].clamp_min(0.0)
    labels = labels.long()
    num_classes = distances.shape[1]
    if torch.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"Prototype EDL labels must be in [0, {num_classes - 1}].")

    batch_indices = torch.arange(labels.shape[0], device=labels.device)
    margin = float(getattr(args, "edl_proto_margin", 1.0))
    balance_classes = bool(getattr(args, "edl_proto_balance_classes", True))

    own_distances = distances[batch_indices, labels]
    nearest_own = own_distances.min(dim=-1).values.sqrt()
    attract_loss = _class_balanced_mean(nearest_own, labels, num_classes, balance_classes)

    other_mask = F.one_hot(labels, num_classes=num_classes).bool().unsqueeze(-1)
    other_distances = distances.masked_fill(other_mask, float("inf")).flatten(start_dim=1)
    nearest_other = other_distances.min(dim=1).values.sqrt()
    separation_loss = _class_balanced_mean(
        F.relu(margin - nearest_other).pow(2),
        labels,
        num_classes,
        balance_classes,
    )

    diversity_loss = _prototype_diversity_loss(model, margin)

    attract_weight = float(getattr(args, "edl_proto_attract_weight", 0.0))
    separation_weight = float(getattr(args, "edl_proto_separation_weight", 0.0))
    diversity_weight = float(getattr(args, "edl_proto_diversity_weight", 0.0))
    proto_loss_weight = float(getattr(args, "edl_proto_loss_weight", 1.0))
    proto_loss_raw = (
        attract_weight * attract_loss
        + separation_weight * separation_loss
        + diversity_weight * diversity_loss
    )
    proto_loss = proto_loss_weight * proto_loss_raw
    stats = {
        "prototype_loss": proto_loss.detach(),
        "prototype_loss_raw": proto_loss_raw.detach(),
        "attract_loss": attract_loss.detach(),
        "separation_loss": separation_loss.detach(),
        "diversity_loss": diversity_loss.detach(),
    }
    return proto_loss, stats


def prototype_edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, logger, device):
    model.train()

    amp_enabled = scaler.is_enabled()
    losses = AverageMeter()
    edl_losses = AverageMeter()
    proto_losses = AverageMeter()
    proto_raw_losses = AverageMeter()
    attract_losses = AverageMeter()
    separation_losses = AverageMeter()
    diversity_losses = AverageMeter()
    component_meters = _loss_component_meters()
    evidence_store = _evidence_store()
    start = time.time()
    skipped_batches = 0

    stage_name = getattr(args, "_current_stage", "joint")
    stage_epoch = int(getattr(args, "_current_stage_epoch", epoch + 1))
    stage_total_epochs = int(getattr(args, "_stage_total_epochs", args.epochs))
    progress_iter = tqdm(
        enumerate(train_loader),
        desc=f"[{stage_name} {stage_epoch:03d}/{stage_total_epochs:03d} prototype train]",
        total=len(train_loader),
    )
    for step, data in progress_iter:
        data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
        if fallback_paths:
            reason = fallback_errors[0] if fallback_errors else "fallback image used during prototype training"
            _append_skip_log(args, "prototype_train_fallback", epoch, step, fallback_paths, reason)
            if data["x"].size(0) == 0:
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"[WARN] Skipping empty prototype train batch at epoch {epoch + 1} step {step}: {reason}")
                continue

        inputs = _prepare_inputs_for_arch(args, data["x"].to(device))
        batch_size = inputs.size(0)
        labels = data["y"].long().to(device)

        if not torch.isfinite(inputs).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "prototype_train_skip", epoch, step, data.get("img_path", []), "non-finite train inputs")
            print(f"[WARN] Skipping prototype train batch at epoch {epoch + 1} step {step}: non-finite train inputs")
            continue

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            evidence, details = model(inputs, return_details=True)
            edl_loss = criterion(evidence, labels)
            proto_loss, proto_stats = _compute_prototype_regularization(model, details, labels, args)
            loss = edl_loss + proto_loss

        if not torch.isfinite(evidence).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "prototype_train_skip", epoch, step, data.get("img_path", []), "non-finite train evidence")
            print(f"[WARN] Skipping prototype train batch at epoch {epoch + 1} step {step}: non-finite train evidence")
            continue
        if not torch.isfinite(details["prototype_distance"]).all():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "prototype_train_skip", epoch, step, data.get("img_path", []), "non-finite prototype distance")
            print(f"[WARN] Skipping prototype train batch at epoch {epoch + 1} step {step}: non-finite prototype distance")
            continue
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            _append_skip_log(args, "prototype_train_skip", epoch, step, data.get("img_path", []), "non-finite train loss")
            print(f"[WARN] Skipping prototype train batch at epoch {epoch + 1} step {step}: non-finite train loss")
            continue

        losses.update(float(loss.item()), batch_size)
        edl_losses.update(float(edl_loss.item()), batch_size)
        proto_losses.update(float(proto_loss.item()), batch_size)
        proto_raw_losses.update(float(proto_stats["prototype_loss_raw"].item()), batch_size)
        attract_losses.update(float(proto_stats["attract_loss"].item()), batch_size)
        separation_losses.update(float(proto_stats["separation_loss"].item()), batch_size)
        diversity_losses.update(float(proto_stats["diversity_loss"].item()), batch_size)
        _update_loss_component_meters(component_meters, criterion, batch_size)
        _record_evidence_batch(evidence_store, labels, evidence)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        postfix = {
            "lr": [optimizer.param_groups[0]["lr"]],
            "loss": f"{losses.avg:.4f}",
            "edl": f"{edl_losses.avg:.4f}",
            "proto": f"{proto_losses.avg:.4f}",
            "proto_raw": f"{proto_raw_losses.avg:.4f}",
            "skipped": skipped_batches,
        }
        postfix.update(_cuda_postfix(device))
        progress_iter.set_postfix(postfix)

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "EDL: {edl.avg:.4f} Proto: {proto.avg:.4f} ProtoRaw: {proto_raw.avg:.4f} "
                "LR: {lr:.8f}".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    edl=edl_losses,
                    proto=proto_losses,
                    proto_raw=proto_raw_losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar("train/epoch", epoch, index)
            logger.add_scalar("train/iter_loss", losses.avg, index)
            logger.add_scalar("train/iter_edl_loss", edl_losses.avg, index)
            logger.add_scalar("train/iter_proto_loss", proto_losses.avg, index)
            logger.add_scalar("train/iter_proto_loss_raw", proto_raw_losses.avg, index)
            logger.add_scalar("train/iter_proto_attract_loss", attract_losses.avg, index)
            logger.add_scalar("train/iter_proto_separation_loss", separation_losses.avg, index)
            logger.add_scalar("train/iter_proto_diversity_loss", diversity_losses.avg, index)
            logger.add_scalar("train/iter_lr", optimizer.param_groups[0]["lr"], index)
            logger.add_scalar("train/skipped_batches", skipped_batches, index)

    if skipped_batches > 0:
        print(f"Epoch {epoch + 1}: skipped {skipped_batches} prototype train batch(es).")

    stats = {
        "edl_loss": edl_losses.avg,
        "prototype_loss": proto_losses.avg,
        "prototype_loss_raw": proto_raw_losses.avg,
        "attract_loss": attract_losses.avg,
        "separation_loss": separation_losses.avg,
        "diversity_loss": diversity_losses.avg,
    }
    component_stats = _loss_component_summary(edl_losses, component_meters, skipped_batches)
    for key, value in component_stats.items():
        stats[f"edl_{key}"] = value
    stats.update(_evidence_summary(evidence_store, "train", args.num_classes))
    return losses.avg, stats


def _collect_train_embeddings(args, model, device):
    init_dataset = MammoDataset(args=args, df=args.train_folds, transform=get_eval_transforms(args))
    init_loader = DataLoader(
        init_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts,
    )

    features = []
    labels = []
    amp_enabled = _amp_enabled(args, device)
    model.eval()
    with torch.no_grad():
        for step, data in tqdm(enumerate(init_loader), desc=f"Prototype init fold{args.cur_fold}", total=len(init_loader)):
            data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
            if fallback_paths:
                reason = fallback_errors[0] if fallback_errors else "fallback image used during prototype init"
                _append_skip_log(args, "prototype_init_fallback", 0, step, fallback_paths, reason)
                if data["x"].size(0) == 0:
                    print(f"[WARN] Prototype init batch contains only fallback images at step {step}: {reason}")
                    continue

            inputs = _prepare_inputs_for_arch(args, data["x"].to(device))
            batch_labels = data["y"].long()
            _assert_finite_tensor("prototype init inputs", inputs, data.get("img_path"))
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                batch_features = model.extract_features(inputs)
            _assert_finite_tensor("prototype init features", batch_features, data.get("img_path"))
            features.append(batch_features.detach().float().cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    if not features:
        raise ValueError(f"Fold {args.cur_fold} has no valid training embeddings for prototype initialization.")

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def _fit_classwise_kmeans(features, labels, prototypes_per_class, seed):
    centers = []
    global_features = features
    for class_idx in range(2):
        class_features = features[labels == class_idx]
        if len(class_features) == 0:
            print(
                f"[WARN] No training samples for class {class_idx}; "
                "initializing prototypes from all training embeddings."
            )
            class_features = global_features

        if len(class_features) >= prototypes_per_class:
            kmeans = KMeans(n_clusters=prototypes_per_class, random_state=seed, n_init=10)
            class_centers = kmeans.fit(class_features).cluster_centers_
        else:
            print(
                f"[WARN] Class {class_idx} has only {len(class_features)} sample(s); "
                f"repeating centers to reach K={prototypes_per_class}."
            )
            repeat_count = int(np.ceil(prototypes_per_class / max(len(class_features), 1)))
            class_centers = np.tile(class_features, (repeat_count, 1))[:prototypes_per_class]
        centers.append(class_centers)

    return torch.tensor(np.stack(centers, axis=0), dtype=torch.float32)


def _initialize_prototypes_from_train(args, model, device):
    if getattr(args, "edl_proto_init", "kmeans").lower() != "kmeans":
        print("Prototype initialization skipped; using random learnable prototypes.")
        args._prototype_initialized_from = "random"
        return

    features, labels = _collect_train_embeddings(args, model, device)
    prototypes = _fit_classwise_kmeans(features, labels, args.edl_proto_k, args.seed)
    model.initialize_prototypes(prototypes.to(device))
    if _training_schedule(args) == "staged" and getattr(args, "_resolved_bce_warmstart_path", None):
        args._prototype_initialized_from = "bce_encoder_kmeans"
    else:
        args._prototype_initialized_from = "current_encoder_kmeans"
    print(f"Initialized prototype tensor with shape: {tuple(prototypes.shape)}")


def _create_model(args, ckpt):
    return BreastClipPrototypeEDLClassifier(
        args,
        ckpt=ckpt,
        num_classes=args.num_classes,
        prototypes_per_class=args.edl_proto_k,
        temperature=args.edl_proto_temperature,
        normalize=args.edl_proto_normalize,
    )


def _load_bce_warmstart_for_fold(args, model):
    bce_checkpoint_path = _resolve_bce_warmstart_path(args, args.cur_fold)
    checkpoint = torch.load(bce_checkpoint_path, map_location="cpu", weights_only=False)
    loaded_count = model.load_bce_encoder_state(checkpoint, strict=True)
    print(f"Loaded {loaded_count} image_encoder tensors from BCE warm-start: {bce_checkpoint_path}")
    _write_stage_manifest(
        args,
        fold_updates={
            "bce_warmstart_path": str(bce_checkpoint_path),
            "bce_encoder_tensors_loaded": loaded_count,
        },
    )
    return bce_checkpoint_path


def _set_stage_trainability(args, model, stage_name):
    if stage_name == "joint":
        return

    freeze_encoder = bool(getattr(args, "staged_freeze_encoder", True))
    model.set_encoder_trainable(not freeze_encoder)
    if stage_name == "prototype_warmup":
        model.set_prototypes_trainable(True)
        model.set_evidence_weights_trainable(True)
    elif stage_name == "edl_calibration":
        model.set_prototypes_trainable(not bool(getattr(args, "edl_stage_freeze_prototypes", True)))
        model.set_evidence_weights_trainable(True)
    else:
        raise ValueError(f"Unknown Prototype EDL training stage: {stage_name}")


def _build_stage_optimizer_scheduler(args, model, train_loader, stage_epochs, lr, device):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters: {trainable_param_count:,} / {total_params:,}")
    if not trainable_params:
        raise ValueError("No trainable parameters found for this Prototype EDL stage.")

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


def _build_edl_criterion(args, class_weights, kl_weight):
    return EDLLoss(
        num_classes=args.num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
        focal_gamma=getattr(args, "edl_focal_gamma", 0.0),
        wrong_evidence_penalty_weight=getattr(args, "edl_wrong_evidence_penalty_weight", 0.0),
        wrong_evidence_margin=getattr(args, "edl_wrong_evidence_margin", 0.05),
        wrong_evidence_class_balanced=getattr(args, "edl_wrong_evidence_class_balanced", True),
    )


def _run_training_stage(
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
    global_epoch_offset,
    best_aucroc,
    epochs_no_improve,
    save_best,
    stage_patience=None,
    restore_stage_best=False,
):
    last_predictions = None
    if stage_epochs <= 0:
        return best_aucroc, epochs_no_improve, last_predictions

    args._current_stage = stage_name
    args._stage_total_epochs = stage_epochs
    best_metric_name = _best_metric_name(args)
    stage_patience = _stage_patience(args, stage_name, default=stage_patience)
    stage_best_metric = float("-inf")
    stage_no_improve = 0
    stage_best_path = _fold_stage_best_checkpoint_path(args, stage_name)
    print(
        f"\n================ Stage: {stage_name} ({stage_epochs} epoch(s), "
        f"patience={stage_patience}) ================"
    )

    for stage_epoch in range(stage_epochs):
        global_epoch = global_epoch_offset + stage_epoch
        args._current_stage_epoch = stage_epoch + 1
        start_time = time.time()
        criterion_epoch = global_epoch if stage_name == "joint" else stage_epoch
        criterion.current_epoch = criterion_epoch
        annealing_complete = True if stage_name == "prototype_warmup" else _is_edl_annealing_complete(args, criterion_epoch)
        if save_best and not annealing_complete:
            print(
                f"Epoch {global_epoch + 1} ({stage_name} {stage_epoch + 1}) - EDL annealing is active; "
                f"stage early stopping still uses patience={stage_patience}."
            )

        avg_loss, train_loss_stats = prototype_edl_train_fn(
            train_loader, model, criterion, optimizer, global_epoch, args, scheduler, scaler, logger, device
        )
        valid_stats, predictions = edl_valid_fn(
            valid_loader, model, criterion, args, device, global_epoch, logger=logger
        )
        last_predictions = predictions
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
            f"Epoch {global_epoch + 1} [{stage_name} {stage_epoch + 1}/{stage_epochs}] - "
            f"avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  "
            f"proto_loss: {train_loss_stats['prototype_loss']:.4f}  "
            f"proto_raw: {train_loss_stats['prototype_loss_raw']:.4f}  AUC-ROC: {aucroc_val:.4f}  "
            f"BACC@0.5: {patient_diag['eval_patient_bacc_at_0_5']:.4f}  "
            f"Pred_Pos@0.5: {patient_diag['eval_patient_pred_pos_at_0_5']}  time: {elapsed:.0f}s"
        )
        logger.add_scalar(f"{eval_name}/{args.label}/AUC-ROC", aucroc_val, global_epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/BACC@0.5", patient_diag["eval_patient_bacc_at_0_5"], global_epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/Pred_Pos@0.5", patient_diag["eval_patient_pred_pos_at_0_5"], global_epoch + 1)
        logger.add_scalar("train/epoch_loss", avg_loss, global_epoch + 1)
        logger.add_scalar("train/epoch_edl_loss", train_loss_stats["edl_loss"], global_epoch + 1)
        logger.add_scalar("train/epoch_proto_loss", train_loss_stats["prototype_loss"], global_epoch + 1)
        logger.add_scalar("train/epoch_proto_loss_raw", train_loss_stats["prototype_loss_raw"], global_epoch + 1)
        logger.add_scalar("train/edl_proto_loss_weight", args.edl_proto_loss_weight, global_epoch + 1)
        logger.add_scalar("train/epoch_proto_attract_loss", train_loss_stats["attract_loss"], global_epoch + 1)
        logger.add_scalar("train/epoch_proto_separation_loss", train_loss_stats["separation_loss"], global_epoch + 1)
        logger.add_scalar("train/epoch_proto_diversity_loss", train_loss_stats["diversity_loss"], global_epoch + 1)
        logger.add_scalar("valid/epoch_loss", avg_val_loss, global_epoch + 1)
        for loss_key in (
            "edl_wrong_evidence_penalty",
            "edl_margin_violation_mean",
            "edl_total_evidence_mean",
        ):
            if loss_key in train_loss_stats and np.isfinite(float(train_loss_stats[loss_key])):
                logger.add_scalar(f"train/{loss_key}", train_loss_stats[loss_key], global_epoch + 1)
        for loss_key in (
            "wrong_evidence_penalty",
            "margin_violation_mean",
            "total_evidence_mean",
        ):
            if loss_key in valid_stats and np.isfinite(float(valid_stats[loss_key])):
                logger.add_scalar(f"valid/{loss_key}", valid_stats[loss_key], global_epoch + 1)

        history_values = {
            "stage": stage_name,
            "stage_epoch": stage_epoch + 1,
            "train_loss": avg_loss,
            "train_edl_loss": train_loss_stats["edl_loss"],
            "train_proto_loss": train_loss_stats["prototype_loss"],
            "train_proto_loss_raw": train_loss_stats["prototype_loss_raw"],
            "train_proto_attract_loss": train_loss_stats["attract_loss"],
            "train_proto_separation_loss": train_loss_stats["separation_loss"],
            "train_proto_diversity_loss": train_loss_stats["diversity_loss"],
            "valid_loss": avg_val_loss,
            "valid_aucroc": aucroc_val,
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
            "eval_wrong_evidence_penalty": valid_stats["wrong_evidence_penalty"],
            "eval_margin_violation_mean": valid_stats["margin_violation_mean"],
            "eval_total_evidence_mean": valid_stats["total_evidence_mean"],
            "eval_skipped_batches": valid_stats["skipped_batches"],
            "prediction_threshold": threshold,
            "prediction_score_agg": _prediction_score_agg(args),
            "prediction_group_cols": ",".join(_prediction_group_cols(args)),
            "best_metric_name": best_metric_name,
        }
        history_values.update(patient_diag)
        history_values.update(image_diag)
        history_values.update({f"eval_{key}": value for key, value in valid_stats.items() if key.startswith("label")})
        for key, value in train_loss_stats.items():
            if key.startswith("edl_"):
                history_values[f"train_{key}"] = value
            elif key.startswith("train_"):
                history_values[key] = value
            elif key.startswith("label"):
                history_values[f"train_{key}"] = value
        history_values.update({key: value for key, value in valid_stats.items() if key.startswith("eval_")})
        best_metric_value = _current_best_metric_value(args, history_values, aucroc_val)
        history_values["selected_best_metric_value"] = best_metric_value
        _append_epoch_history(history, global_epoch + 1, history_values)

        stage_improved = np.isfinite(best_metric_value) and (
            not np.isfinite(stage_best_metric) or stage_best_metric < best_metric_value
        )
        if stage_improved:
            stage_best_metric = best_metric_value
            stage_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "predictions": predictions,
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch + 1,
                    "auroc": aucroc_val,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": stage_best_metric,
                    "prediction_aggregation": _prediction_aggregation_config(args),
                    "training_schedule": _training_schedule(args),
                    "bce_warmstart_path": getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
                    "prototype_initialized_from": getattr(args, "_prototype_initialized_from", None),
                    "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
                    "history": history,
                },
                stage_best_path,
            )
        else:
            stage_no_improve += 1

        has_best_checkpoint = bool(getattr(args, "_best_checkpoint_written", False))
        if save_best and np.isfinite(best_metric_value) and (not has_best_checkpoint or best_aucroc < best_metric_value):
            best_aucroc = best_metric_value
            epochs_no_improve = 0
            args._best_checkpoint_written = True
            print(
                f"Epoch {global_epoch + 1} - Save Best {best_metric_name}: {best_aucroc:.4f} "
                f"(AUC-ROC: {aucroc_val:.4f}) Prototype EDL Model"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "predictions": predictions,
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch + 1,
                    "auroc": aucroc_val,
                    "best_aucroc": aucroc_val,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_aucroc,
                    "prediction_aggregation": _prediction_aggregation_config(args),
                    "epochs_no_improve": epochs_no_improve,
                    "history": history,
                    "training_schedule": _training_schedule(args),
                    "bce_warmstart_path": getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
                    "prototype_initialized_from": getattr(args, "_prototype_initialized_from", None),
                    "train_mode": getattr(args, "train_mode", "full"),
                    "freeze_backbone": getattr(args, "freeze_backbone", "n"),
                    "edl_proto_k": args.edl_proto_k,
                    "edl_proto_temperature": args.edl_proto_temperature,
                    "edl_proto_normalize": args.edl_proto_normalize,
                    "edl_proto_attract_weight": args.edl_proto_attract_weight,
                    "edl_proto_separation_weight": args.edl_proto_separation_weight,
                    "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                    "edl_proto_loss_weight": args.edl_proto_loss_weight,
                    "edl_proto_margin": args.edl_proto_margin,
                    "edl_proto_balance_classes": args.edl_proto_balance_classes,
                    "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
                },
                _fold_best_model_path(args),
            )
        elif save_best:
            epochs_no_improve += 1

        _save_last_checkpoint(
            args,
            model,
            optimizer,
            scheduler,
            scaler,
            global_epoch,
            best_aucroc,
            epochs_no_improve,
            history,
            stage_name=stage_name,
            stage_epoch=stage_epoch + 1,
        )

        if stage_patience > 0 and stage_no_improve >= stage_patience:
            print(
                f"Stage early stopping at epoch {global_epoch + 1} [{stage_name}]: "
                f"no improvement for {stage_patience} epochs, best {best_metric_name}: {stage_best_metric:.4f}"
            )
            break

        if save_best:
            print(f"[Fold{args.cur_fold}], Best {_best_metric_name(args)}: {best_aucroc:.4f}")

    if restore_stage_best and stage_best_path.exists():
        checkpoint = torch.load(stage_best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(
            f"Restored {stage_name} best model from {stage_best_path} "
            f"({best_metric_name}={float(checkpoint.get('best_metric_value', float('nan'))):.4f})"
        )

    _write_stage_manifest(
        args,
        fold_updates={
            f"{stage_name}_best_metric_name": best_metric_name,
            f"{stage_name}_best_metric_value": stage_best_metric if np.isfinite(stage_best_metric) else None,
            f"{stage_name}_stage_best_checkpoint": str(stage_best_path) if stage_best_path.exists() else None,
            f"{stage_name}_patience": stage_patience,
            f"{stage_name}_stopped_after_epoch": global_epoch + 1 if "global_epoch" in locals() else None,
        },
    )
    return best_aucroc, epochs_no_improve, last_predictions


def do_prototype_edl_experiments(args, device):
    args.model_base_name = args.arch
    args.data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = args.data_dir / csv_path
    args.df = pd.read_csv(csv_path)
    args.df = args.df.fillna(0)
    audit_laterality_label_mixes(args.df, args.label, context="Prototype EDL input CSV")
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

    if args.n_folds == 0 and len(holdout_val_df) == 0 and len(test_df) == 0:
        raise ValueError("n_folds=0 requires a non-empty validation split or test split for per-epoch evaluation.")

    oof_df = pd.DataFrame()
    fold_prediction_arrays = []
    fold_summaries = []
    _write_stage_manifest(
        args,
        {
            "training_schedule": _training_schedule(args),
            "checkpoint_selection_metric": _best_metric_name(args),
            "prediction_aggregation": _prediction_aggregation_config(args),
        },
    )
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

        eval_df, metrics_df = prototype_edl_train_loop(args, device)
        if args.n_folds > 0:
            oof_df = pd.concat([oof_df, eval_df])
        _record_fold_summary(args, metrics_df, fold_summaries)

        best_model_path = _fold_best_model_path(args)
        if len(predict_df) > 0 and best_model_path.exists():
            fold_results = prototype_edl_predict_on_dataset(args, predict_df, best_model_path, device, fold)
            fold_results = _build_fold_split_view(fold_results, fold, args.n_folds)
            fold_prediction_arrays.append(fold_results)
            fold_csv_path = args.output_path / f"prototype_edl_fold{fold}_all_predictions.csv"
            fold_results.to_csv(fold_csv_path, index=False)
            print(f"Fold {fold} predictions saved to: {fold_csv_path}")

    if args.n_folds > 0 and len(oof_df) > 0:
        oof_df = oof_df.reset_index(drop=True)
        print("\n================ Prototype EDL CV (Out-of-Fold) ================")
        oof_agg = _aggregate_prediction_scores(args, oof_df, args.label, "prediction_prob")
        aucroc_val = auroc(gt=oof_agg[args.label].values.astype(int), pred=oof_agg["prediction_prob"].values)
        print(f"OOF AUC-ROC: {aucroc_val:.4f}")
        oof_df.to_csv(
            args.output_path / f"prototype_edl_seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv",
            index=False,
        )

    if len(predict_df) > 0 and len(fold_prediction_arrays) > 0:
        ensemble_output = predict_df.copy()
        all_evidence_cols = [f"evidence_{i}" for i in range(args.num_classes)]
        all_alpha_cols = [f"alpha_{i}" for i in range(args.num_classes)]
        all_prob_cols = [f"probability_{i}" for i in range(args.num_classes)]

        image_score_col = "image_prediction_prob"
        for col in all_evidence_cols + all_alpha_cols + all_prob_cols + ["total_uncertainty", image_score_col]:
            ensemble_output[col] = np.mean([fd[col].values for fd in fold_prediction_arrays], axis=0)

        ensemble_output = _attach_prediction_scores(
            args,
            ensemble_output,
            ensemble_output[image_score_col].values,
            image_score_col=image_score_col,
        )
        ensemble_output["prediction_label"] = (ensemble_output["prediction_prob"] >= _prediction_threshold(args)).astype(int)
        ensemble_output["prediction_score"] = ensemble_output["prediction_prob"]
        ensemble_output["predicted_class"] = ensemble_output["prediction_label"]
        ensemble_output["uncertainty"] = ensemble_output["total_uncertainty"]
        ensemble_output["model_fold"] = "ensemble"
        ensemble_csv_path = args.output_path / "prototype_edl_ensemble_all_predictions.csv"
        ensemble_output.to_csv(ensemble_csv_path, index=False)
        print(f"\nPrototype EDL ensemble all-data predictions saved to: {ensemble_csv_path}")
    elif len(predict_df) > 0:
        print("Warning: no fold predictions available; ensemble predictions were not generated.")

    if fold_summaries:
        summary_df = pd.DataFrame(fold_summaries)
        summary_path = args.output_path / "prototype_edl_fold_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Prototype EDL fold metrics summary saved to: {summary_path}")

    print("\n================ Prototype EDL Done! ================")


def _prototype_edl_staged_train_loop(args, device):
    print(f"\n================== Prototype EDL fold: {args.cur_fold} staged training ======================")
    if getattr(args, "resume", False):
        print("[WARN] --resume is ignored for staged Prototype EDL; staged runs restart from the BCE warm-start checkpoint.")
    if not hasattr(args, "edl_proto_loss_weight"):
        args.edl_proto_loss_weight = 1.0

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    train_loader, valid_loader = edl_get_dataloader(args)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}")

    model = _create_model(args, ckpt).to(device)
    bce_checkpoint_path = _load_bce_warmstart_for_fold(args, model)
    if getattr(args, "staged_freeze_encoder", True):
        model.set_encoder_trainable(False)
        print("Stage setup: BCE encoder frozen.")
    else:
        model.set_encoder_trainable(True)
        print("Stage setup: BCE encoder trainable.")

    _initialize_prototypes_from_train(args, model, device)
    print(model)
    _write_stage_manifest(
        args,
        fold_updates={
            "prototype_initialized_from": getattr(args, "_prototype_initialized_from", None),
            "bce_warmstart_path": str(bce_checkpoint_path),
            "staged_freeze_encoder": bool(getattr(args, "staged_freeze_encoder", True)),
            "edl_stage_freeze_prototypes": bool(getattr(args, "edl_stage_freeze_prototypes", True)),
            "bce_stage_patience": getattr(args, "bce_stage_patience", None),
            "proto_warmup_patience": getattr(args, "proto_warmup_patience", None),
            "edl_stage_patience": getattr(args, "edl_stage_patience", getattr(args, "patience", None)),
            "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
        },
    )

    logger = SummaryWriter(args.tb_logs_path / f"prototype_edl_fold{args.cur_fold}")
    class_weights = _compute_fold_class_weights(args)
    class_weight_mode = _class_weight_mode(args)
    if class_weight_mode != "none" and class_weights is None:
        raise ValueError(
            f"class_weight_mode={class_weight_mode} but class_weights could not be computed for fold {args.cur_fold}. "
            "Check train_folds labels before training."
        )

    eval_name = getattr(args, "eval_split", "val")
    history = _empty_history()
    best_aucroc = 0.0
    epochs_no_improve = 0
    args._best_checkpoint_written = False
    global_epoch_offset = 0

    warmup_epochs = int(getattr(args, "proto_warmup_epochs", 0))
    if warmup_epochs > 0:
        _set_stage_trainability(args, model, "prototype_warmup")
        warmup_optimizer, warmup_scheduler, warmup_scaler = _build_stage_optimizer_scheduler(
            args, model, train_loader, warmup_epochs, args.lr, device
        )
        warmup_criterion = _build_edl_criterion(args, class_weights, kl_weight=0.0)
        best_aucroc, epochs_no_improve, _ = _run_training_stage(
            args,
            model,
            train_loader,
            valid_loader,
            warmup_criterion,
            warmup_optimizer,
            warmup_scheduler,
            warmup_scaler,
            logger,
            device,
            history,
            eval_name,
            stage_name="prototype_warmup",
            stage_epochs=warmup_epochs,
            global_epoch_offset=global_epoch_offset,
            best_aucroc=best_aucroc,
            epochs_no_improve=epochs_no_improve,
            save_best=False,
            stage_patience=args.proto_warmup_patience,
            restore_stage_best=True,
        )
        global_epoch_offset += warmup_epochs

    _set_stage_trainability(args, model, "edl_calibration")
    edl_optimizer, edl_scheduler, edl_scaler = _build_stage_optimizer_scheduler(
        args, model, train_loader, args.epochs, args.lr, device
    )
    edl_criterion = _build_edl_criterion(args, class_weights, kl_weight=args.edl_kl_weight)
    best_aucroc, epochs_no_improve, _ = _run_training_stage(
        args,
        model,
        train_loader,
        valid_loader,
        edl_criterion,
        edl_optimizer,
        edl_scheduler,
        edl_scaler,
        logger,
        device,
        history,
        eval_name,
        stage_name="edl_calibration",
        stage_epochs=args.epochs,
        global_epoch_offset=global_epoch_offset,
        best_aucroc=best_aucroc,
        epochs_no_improve=epochs_no_improve,
        save_best=True,
        stage_patience=args.edl_stage_patience,
        restore_stage_best=False,
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
    _write_stage_manifest(
        args,
        fold_updates={
            "best_metric_name": _best_metric_name(args),
            "best_metric_value": best_aucroc,
            "metrics_csv": str(_metrics_csv_path(args)),
            "best_checkpoint_path": str(best_model_path),
        },
    )
    logger.close()
    _empty_cuda_cache(device)
    gc.collect()
    return args.valid_folds.copy(), metrics_df


def prototype_edl_train_loop(args, device):
    if _training_schedule(args) == "staged":
        return _prototype_edl_staged_train_loop(args, device)

    print(f"\n================== Prototype EDL fold: {args.cur_fold} training ======================")
    if not hasattr(args, "edl_proto_loss_weight"):
        args.edl_proto_loss_weight = 1.0

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    train_loader, valid_loader = edl_get_dataloader(args)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}")

    model = _create_model(args, ckpt).to(device)
    _initialize_prototypes_from_train(args, model, device)
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
        "total_epochs": args.epochs,
        "warmup_steps": warmup_steps,
        "total_steps": len(train_loader) * args.epochs,
    }
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(args, device))

    logger = SummaryWriter(args.tb_logs_path / f"prototype_edl_fold{args.cur_fold}")
    class_weights = _compute_fold_class_weights(args)
    class_weight_mode = _class_weight_mode(args)
    if class_weight_mode != "none" and class_weights is None:
        raise ValueError(
            f"class_weight_mode={class_weight_mode} but class_weights could not be computed for fold {args.cur_fold}. "
            "Check train_folds labels before training."
        )
    criterion = EDLLoss(
        num_classes=args.num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
        focal_gamma=getattr(args, "edl_focal_gamma", 0.0),
        wrong_evidence_penalty_weight=getattr(args, "edl_wrong_evidence_penalty_weight", 0.0),
        wrong_evidence_margin=getattr(args, "edl_wrong_evidence_margin", 0.05),
        wrong_evidence_class_balanced=getattr(args, "edl_wrong_evidence_class_balanced", True),
    )

    start_epoch, best_aucroc, epochs_no_improve, history = _load_last_checkpoint_if_available(
        args, model, optimizer, scheduler, scaler
    )
    eval_name = getattr(args, "eval_split", "val")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        criterion.current_epoch = epoch
        annealing_complete = _is_edl_annealing_complete(args, epoch)
        if not annealing_complete:
            print(
                f"Epoch {epoch + 1} - early stopping paused until visible epoch "
                f"{_edl_annealing_gate_visible_epoch(args)} while EDL annealing is active."
            )

        avg_loss, train_loss_stats = prototype_edl_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, logger, device
        )
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
            f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  "
            f"proto_loss: {train_loss_stats['prototype_loss']:.4f}  "
            f"proto_raw: {train_loss_stats['prototype_loss_raw']:.4f}  AUC-ROC: {aucroc_val:.4f}  "
            f"BACC@0.5: {patient_diag['eval_patient_bacc_at_0_5']:.4f}  "
            f"Pred_Pos@0.5: {patient_diag['eval_patient_pred_pos_at_0_5']}  time: {elapsed:.0f}s"
        )
        logger.add_scalar(f"{eval_name}/{args.label}/AUC-ROC", aucroc_val, epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/BACC@0.5", patient_diag["eval_patient_bacc_at_0_5"], epoch + 1)
        logger.add_scalar(f"{eval_name}/{args.label}/Pred_Pos@0.5", patient_diag["eval_patient_pred_pos_at_0_5"], epoch + 1)
        logger.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
        logger.add_scalar("train/epoch_edl_loss", train_loss_stats["edl_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_loss", train_loss_stats["prototype_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_loss_raw", train_loss_stats["prototype_loss_raw"], epoch + 1)
        logger.add_scalar("train/edl_proto_loss_weight", args.edl_proto_loss_weight, epoch + 1)
        logger.add_scalar("train/epoch_proto_attract_loss", train_loss_stats["attract_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_separation_loss", train_loss_stats["separation_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_diversity_loss", train_loss_stats["diversity_loss"], epoch + 1)
        logger.add_scalar("valid/epoch_loss", avg_val_loss, epoch + 1)
        for loss_key in (
            "edl_wrong_evidence_penalty",
            "edl_margin_violation_mean",
            "edl_total_evidence_mean",
        ):
            if loss_key in train_loss_stats and np.isfinite(float(train_loss_stats[loss_key])):
                logger.add_scalar(f"train/{loss_key}", train_loss_stats[loss_key], epoch + 1)
        for loss_key in (
            "wrong_evidence_penalty",
            "margin_violation_mean",
            "total_evidence_mean",
        ):
            if loss_key in valid_stats and np.isfinite(float(valid_stats[loss_key])):
                logger.add_scalar(f"valid/{loss_key}", valid_stats[loss_key], epoch + 1)

        history_values = {
            "stage": "joint",
            "stage_epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_edl_loss": train_loss_stats["edl_loss"],
            "train_proto_loss": train_loss_stats["prototype_loss"],
            "train_proto_loss_raw": train_loss_stats["prototype_loss_raw"],
            "train_proto_attract_loss": train_loss_stats["attract_loss"],
            "train_proto_separation_loss": train_loss_stats["separation_loss"],
            "train_proto_diversity_loss": train_loss_stats["diversity_loss"],
            "valid_loss": avg_val_loss,
            "valid_aucroc": aucroc_val,
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
            "eval_wrong_evidence_penalty": valid_stats["wrong_evidence_penalty"],
            "eval_margin_violation_mean": valid_stats["margin_violation_mean"],
            "eval_total_evidence_mean": valid_stats["total_evidence_mean"],
            "eval_skipped_batches": valid_stats["skipped_batches"],
            "prediction_threshold": threshold,
            "prediction_score_agg": _prediction_score_agg(args),
            "prediction_group_cols": ",".join(_prediction_group_cols(args)),
            "best_metric_name": _best_metric_name(args),
        }
        history_values.update(patient_diag)
        history_values.update(image_diag)
        history_values.update({f"eval_{key}": value for key, value in valid_stats.items() if key.startswith("label")})
        for key, value in train_loss_stats.items():
            if key.startswith("edl_"):
                history_values[f"train_{key}"] = value
            elif key.startswith("train_"):
                history_values[key] = value
            elif key.startswith("label"):
                history_values[f"train_{key}"] = value
        history_values.update({key: value for key, value in valid_stats.items() if key.startswith("eval_")})
        best_metric_name = _best_metric_name(args)
        best_metric_value = _current_best_metric_value(args, history_values, aucroc_val)
        history_values["selected_best_metric_value"] = best_metric_value
        _append_epoch_history(history, epoch + 1, history_values)

        if np.isfinite(best_metric_value) and (epoch == 0 or best_aucroc < best_metric_value):
            best_aucroc = best_metric_value
            epochs_no_improve = 0
            print(
                f"Epoch {epoch + 1} - Save Best {best_metric_name}: {best_aucroc:.4f} "
                f"(AUC-ROC: {aucroc_val:.4f}) Prototype EDL Model"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "predictions": predictions,
                    "epoch": epoch,
                    "auroc": aucroc_val,
                    "best_aucroc": aucroc_val,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_aucroc,
                    "prediction_aggregation": _prediction_aggregation_config(args),
                    "epochs_no_improve": epochs_no_improve,
                    "history": history,
                    "training_schedule": _training_schedule(args),
                    "bce_warmstart_path": getattr(args, "_resolved_bce_warmstart_path", getattr(args, "bce_warmstart_path", None)),
                    "prototype_initialized_from": getattr(args, "_prototype_initialized_from", None),
                    "train_mode": getattr(args, "train_mode", "full"),
                    "freeze_backbone": getattr(args, "freeze_backbone", "n"),
                    "edl_proto_k": args.edl_proto_k,
                    "edl_proto_temperature": args.edl_proto_temperature,
                    "edl_proto_normalize": args.edl_proto_normalize,
                    "edl_proto_attract_weight": args.edl_proto_attract_weight,
                    "edl_proto_separation_weight": args.edl_proto_separation_weight,
                    "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                    "edl_proto_loss_weight": args.edl_proto_loss_weight,
                    "edl_proto_margin": args.edl_proto_margin,
                    "edl_proto_balance_classes": args.edl_proto_balance_classes,
                    "wrong_evidence_penalty": _wrong_evidence_penalty_config(args),
                },
                _fold_best_model_path(args),
            )
        else:
            epochs_no_improve = epochs_no_improve + 1 if annealing_complete else 0

        _save_last_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_aucroc, epochs_no_improve, history)

        if annealing_complete and args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, "
                f"best {_best_metric_name(args)}: {best_aucroc:.4f}"
            )
            break

        print(f"[Fold{args.cur_fold}], Best {_best_metric_name(args)}: {best_aucroc:.4f}")

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


def _prototype_topk_batch(details, topk, batch_size, valid_indices):
    proto_evidence = details["prototype_evidence"]
    proto_similarity = details["prototype_similarity"]
    effective_topk = min(int(topk), proto_evidence.shape[-1])
    top_evidence, top_indices = torch.topk(proto_evidence, k=effective_topk, dim=-1)
    top_similarity = torch.gather(proto_similarity, dim=-1, index=top_indices)

    idx_batch = np.full((batch_size, proto_evidence.shape[1], effective_topk), -1, dtype=np.int64)
    evidence_batch = np.zeros((batch_size, proto_evidence.shape[1], effective_topk), dtype=np.float32)
    similarity_batch = np.zeros((batch_size, proto_evidence.shape[1], effective_topk), dtype=np.float32)

    idx_batch[valid_indices] = top_indices.cpu().numpy()
    evidence_batch[valid_indices] = top_evidence.cpu().numpy()
    similarity_batch[valid_indices] = top_similarity.cpu().numpy()
    return idx_batch, evidence_batch, similarity_batch


def prototype_edl_predict_on_dataset(args, df, model_path, device, fold):
    print(f"\n=== Prototype EDL Predicting all data with fold {fold} model ===")

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    model = _create_model(args, ckpt)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    predict_dataset = MammoDataset(args=args, df=df, transform=get_eval_transforms(args))
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts,
    )

    all_evidence = []
    all_probs = []
    all_uncertainty = []
    all_top_idx = []
    all_top_evidence = []
    all_top_similarity = []
    amp_enabled = _amp_enabled(args, device)
    effective_topk = min(int(args.edl_proto_topk), int(args.edl_proto_k))

    with torch.no_grad():
        for step, data in tqdm(
            enumerate(predict_loader), desc=f"Prototype EDL Predicting fold{fold}", total=len(predict_loader)
        ):
            batch_size = len(data.get("img_path", []))
            fallback_flags = list(data.get("is_fallback", []))
            valid_indices = _valid_indices_from_fallback_flags(fallback_flags, batch_size)
            batch_evidence, batch_probs, batch_uncertainty = _placeholder_prediction_batch(args, batch_size)
            top_idx = np.full((batch_size, args.num_classes, effective_topk), -1, dtype=np.int64)
            top_evidence = np.zeros((batch_size, args.num_classes, effective_topk), dtype=np.float32)
            top_similarity = np.zeros((batch_size, args.num_classes, effective_topk), dtype=np.float32)

            try:
                data, fallback_paths, fallback_errors = _filter_valid_subbatch(data)
                if fallback_paths:
                    reason = fallback_errors[0] if fallback_errors else "fallback image used during prediction"
                    _append_skip_log(args, "prototype_predict_fallback", 0, step, fallback_paths, reason)
                    if data["x"].size(0) == 0:
                        all_evidence.append(batch_evidence)
                        all_probs.append(batch_probs)
                        all_uncertainty.append(batch_uncertainty)
                        all_top_idx.append(top_idx)
                        all_top_evidence.append(top_evidence)
                        all_top_similarity.append(top_similarity)
                        print(f"[WARN] Prediction batch contains only fallback images at step {step}: {reason}")
                        continue

                inputs = _prepare_inputs_for_arch(args, data["x"].to(device))
                _assert_finite_tensor("prototype prediction inputs", inputs, data.get("img_path"))
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    evidence, details = model(inputs, return_details=True)
                _assert_finite_tensor("prototype prediction evidence", evidence, data.get("img_path"))

                probs = BreastClipPrototypeEDLClassifier.compute_probabilities(evidence)
                uncertainty = BreastClipPrototypeEDLClassifier.compute_uncertainty(evidence)
                _assert_finite_tensor("prototype prediction probabilities", probs, data.get("img_path"))
                _assert_finite_tensor("prototype prediction uncertainty", uncertainty, data.get("img_path"))

                batch_evidence[valid_indices] = evidence.cpu().numpy()
                batch_probs[valid_indices] = probs.cpu().numpy()
                batch_uncertainty[valid_indices] = uncertainty.cpu().numpy()
                top_idx, top_evidence, top_similarity = _prototype_topk_batch(
                    details, effective_topk, batch_size, valid_indices
                )

                all_evidence.append(batch_evidence)
                all_probs.append(batch_probs)
                all_uncertainty.append(batch_uncertainty)
                all_top_idx.append(top_idx)
                all_top_evidence.append(top_evidence)
                all_top_similarity.append(top_similarity)
            except Exception as exc:
                if not getattr(args, "skip_bad_batches", False) or not _is_recoverable_batch_error(exc):
                    raise

                _append_skip_log(args, "prototype_predict_skip", 0, step, data.get("img_path", []), str(exc))
                all_evidence.append(batch_evidence)
                all_probs.append(batch_probs)
                all_uncertainty.append(batch_uncertainty)
                all_top_idx.append(top_idx)
                all_top_evidence.append(top_evidence)
                all_top_similarity.append(top_similarity)
                print(f"[WARN] Prediction batch recovered with placeholders: {exc}")

    evidence_array = np.concatenate(all_evidence)
    alpha_array = evidence_array + 1
    probs_array = np.concatenate(all_probs)
    uncertainty_array = np.concatenate(all_uncertainty)
    top_idx_array = np.concatenate(all_top_idx)
    top_evidence_array = np.concatenate(all_top_evidence)
    top_similarity_array = np.concatenate(all_top_similarity)

    result_df = df.copy()
    result_df["model_fold"] = fold
    num_classes = evidence_array.shape[1]
    for i in range(num_classes):
        result_df[f"evidence_{i}"] = evidence_array[:, i]
        result_df[f"alpha_{i}"] = alpha_array[:, i]
        result_df[f"probability_{i}"] = probs_array[:, i]

    result_df["total_uncertainty"] = uncertainty_array.flatten()
    result_df = _attach_prediction_scores(
        args,
        result_df,
        probs_array[:, -1],
        image_score_col="image_prediction_prob",
    )
    result_df["prediction_score"] = result_df["prediction_prob"]
    result_df["predicted_class"] = result_df["prediction_label"]
    result_df["uncertainty"] = result_df["total_uncertainty"]

    for class_idx in range(num_classes):
        for rank_idx in range(top_idx_array.shape[-1]):
            rank = rank_idx + 1
            result_df[f"proto_c{class_idx}_top{rank}_idx"] = top_idx_array[:, class_idx, rank_idx]
            result_df[f"proto_c{class_idx}_top{rank}_evidence"] = top_evidence_array[:, class_idx, rank_idx]
            result_df[f"proto_c{class_idx}_top{rank}_similarity"] = top_similarity_array[:, class_idx, rank_idx]

    print(
        f"Fold {fold} prototype prediction stats: "
        f"prob_mean={result_df['prediction_prob'].mean():.4f}, "
        f"uncertainty_mean={result_df['total_uncertainty'].mean():.4f}"
    )

    _empty_cuda_cache(device)
    gc.collect()
    return result_df
