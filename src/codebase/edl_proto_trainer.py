"""
Prototype + EDL training, validation, and prediction helpers.

This module mirrors the existing EDL data flow and output conventions while
adding per-class prototype explanations.
"""

import gc
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
    _amp_enabled,
    _append_skip_log,
    _assert_finite_tensor,
    _cuda_postfix,
    _edl_annealing_gate_visible_epoch,
    _is_edl_annealing_complete,
    _compute_fold_class_weights,
    _empty_cuda_cache,
    _filter_valid_subbatch,
    _is_recoverable_batch_error,
    _placeholder_prediction_batch,
    _valid_indices_from_fallback_flags,
    edl_get_dataloader,
    edl_valid_fn,
)
from metrics import auroc
from utils import AverageMeter, seed_all, timeSince


PROTO_HISTORY_KEYS = (
    "train_edl_loss",
    "train_proto_loss",
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


def _metrics_csv_path(args):
    return args.output_path / f"prototype_edl_fold{args.cur_fold}_metrics.csv"


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
        "edl_proto_k": getattr(args, "edl_proto_k", None),
        "edl_proto_temperature": getattr(args, "edl_proto_temperature", None),
        "edl_proto_normalize": getattr(args, "edl_proto_normalize", None),
        "edl_proto_attract_weight": getattr(args, "edl_proto_attract_weight", None),
        "edl_proto_separation_weight": getattr(args, "edl_proto_separation_weight", None),
        "edl_proto_diversity_weight": getattr(args, "edl_proto_diversity_weight", None),
        "edl_proto_margin": getattr(args, "edl_proto_margin", None),
        "edl_proto_balance_classes": getattr(args, "edl_proto_balance_classes", None),
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
    best_aucroc = float(checkpoint.get("best_aucroc", 0.0))
    epochs_no_improve = int(checkpoint.get("epochs_no_improve", 0))
    if not _is_edl_annealing_complete(args, int(checkpoint.get("epoch", -1))):
        epochs_no_improve = 0
    print(
        f"Resumed prototype EDL fold {args.cur_fold} from {checkpoint_path} "
        f"(next_epoch={start_epoch + 1}, best_aucroc={best_aucroc:.4f})"
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
    }
    for key in PROTO_HISTORY_KEYS:
        values = history.get(key, [])
        if len(values) == len(history["epochs"]):
            metric_data[key] = values

    metrics_df = pd.DataFrame(metric_data)
    metrics_df.to_csv(_metrics_csv_path(args), index=False)
    print(f"Fold {args.cur_fold} metrics saved to: {_metrics_csv_path(args)}")
    return metrics_df


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
    proto_loss = (
        attract_weight * attract_loss
        + separation_weight * separation_loss
        + diversity_weight * diversity_loss
    )
    stats = {
        "prototype_loss": proto_loss.detach(),
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
    attract_losses = AverageMeter()
    separation_losses = AverageMeter()
    diversity_losses = AverageMeter()
    start = time.time()
    skipped_batches = 0

    progress_iter = tqdm(
        enumerate(train_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch prototype train]",
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
        attract_losses.update(float(proto_stats["attract_loss"].item()), batch_size)
        separation_losses.update(float(proto_stats["separation_loss"].item()), batch_size)
        diversity_losses.update(float(proto_stats["diversity_loss"].item()), batch_size)

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
            "skipped": skipped_batches,
        }
        postfix.update(_cuda_postfix(device))
        progress_iter.set_postfix(postfix)

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "EDL: {edl.avg:.4f} Proto: {proto.avg:.4f} "
                "LR: {lr:.8f}".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    edl=edl_losses,
                    proto=proto_losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar("train/epoch", epoch, index)
            logger.add_scalar("train/iter_loss", losses.avg, index)
            logger.add_scalar("train/iter_edl_loss", edl_losses.avg, index)
            logger.add_scalar("train/iter_proto_loss", proto_losses.avg, index)
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
        "attract_loss": attract_losses.avg,
        "separation_loss": separation_losses.avg,
        "diversity_loss": diversity_losses.avg,
    }
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
        return

    features, labels = _collect_train_embeddings(args, model, device)
    prototypes = _fit_classwise_kmeans(features, labels, args.edl_proto_k, args.seed)
    model.initialize_prototypes(prototypes.to(device))
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


def do_prototype_edl_experiments(args, device):
    args.model_base_name = args.arch
    args.data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = args.data_dir / csv_path
    args.df = pd.read_csv(csv_path)
    args.df = args.df.fillna(0)
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
        oof_agg = oof_df.groupby("patient_id").agg({args.label: "max", "prediction_prob": "mean"}).reset_index()
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

        for col in all_evidence_cols + all_alpha_cols + all_prob_cols + ["total_uncertainty", "prediction_prob"]:
            ensemble_output[col] = np.mean([fd[col].values for fd in fold_prediction_arrays], axis=0)

        ensemble_output["prediction_label"] = (ensemble_output["prediction_prob"] >= 0.5).astype(int)
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


def prototype_edl_train_loop(args, device):
    print(f"\n================== Prototype EDL fold: {args.cur_fold} training ======================")

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
    criterion = EDLLoss(
        num_classes=args.num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
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
        avg_val_loss, predictions = edl_valid_fn(
            valid_loader, model, criterion, args, device, epoch, logger=logger
        )
        args.valid_folds["prediction_prob"] = predictions

        valid_agg = args.valid_folds[["patient_id", args.label, "prediction_prob", "fold"]].groupby(
            ["patient_id"]
        ).mean()
        aucroc_val = auroc(valid_agg[args.label].values.astype(int), valid_agg["prediction_prob"].values)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  "
            f"proto_loss: {train_loss_stats['prototype_loss']:.4f}  AUC-ROC: {aucroc_val:.4f}  time: {elapsed:.0f}s"
        )
        logger.add_scalar(f"{eval_name}/{args.label}/AUC-ROC", aucroc_val, epoch + 1)
        logger.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
        logger.add_scalar("train/epoch_edl_loss", train_loss_stats["edl_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_loss", train_loss_stats["prototype_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_attract_loss", train_loss_stats["attract_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_separation_loss", train_loss_stats["separation_loss"], epoch + 1)
        logger.add_scalar("train/epoch_proto_diversity_loss", train_loss_stats["diversity_loss"], epoch + 1)
        logger.add_scalar("valid/epoch_loss", avg_val_loss, epoch + 1)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_edl_loss"].append(train_loss_stats["edl_loss"])
        history["train_proto_loss"].append(train_loss_stats["prototype_loss"])
        history["train_proto_attract_loss"].append(train_loss_stats["attract_loss"])
        history["train_proto_separation_loss"].append(train_loss_stats["separation_loss"])
        history["train_proto_diversity_loss"].append(train_loss_stats["diversity_loss"])
        history["valid_loss"].append(avg_val_loss)
        history["valid_aucroc"].append(aucroc_val)

        if epoch == 0 or best_aucroc < aucroc_val:
            best_aucroc = aucroc_val
            epochs_no_improve = 0
            print(f"Epoch {epoch + 1} - Save Best AUC-ROC: {best_aucroc:.4f} Prototype EDL Model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "predictions": predictions,
                    "epoch": epoch,
                    "auroc": aucroc_val,
                    "best_aucroc": best_aucroc,
                    "epochs_no_improve": epochs_no_improve,
                    "history": history,
                    "train_mode": getattr(args, "train_mode", "full"),
                    "freeze_backbone": getattr(args, "freeze_backbone", "n"),
                    "edl_proto_k": args.edl_proto_k,
                    "edl_proto_temperature": args.edl_proto_temperature,
                    "edl_proto_normalize": args.edl_proto_normalize,
                    "edl_proto_attract_weight": args.edl_proto_attract_weight,
                    "edl_proto_separation_weight": args.edl_proto_separation_weight,
                    "edl_proto_diversity_weight": args.edl_proto_diversity_weight,
                    "edl_proto_margin": args.edl_proto_margin,
                    "edl_proto_balance_classes": args.edl_proto_balance_classes,
                },
                _fold_best_model_path(args),
            )
        else:
            epochs_no_improve = epochs_no_improve + 1 if annealing_complete else 0

        _save_last_checkpoint(args, model, optimizer, scheduler, scaler, epoch, best_aucroc, epochs_no_improve, history)

        if annealing_complete and args.patience > 0 and epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, "
                f"best AUC-ROC: {best_aucroc:.4f}"
            )
            break

        print(f"[Fold{args.cur_fold}], Best AUC-ROC: {best_aucroc:.4f}")

    best_model_path = _fold_best_model_path(args)
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location="cpu", weights_only=False)["predictions"]
        args.valid_folds["prediction_prob"] = predictions
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
    result_df["prediction_prob"] = probs_array[:, -1]
    result_df["prediction_label"] = (result_df["prediction_prob"] >= 0.5).astype(int)
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
