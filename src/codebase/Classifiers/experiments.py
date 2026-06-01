import gc
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_concepts import MammoDataset, collator_mammo_dataset_w_concepts
from Datasets.dataset_utils import get_eval_transforms, get_transforms
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import auroc
from utils import (
    AverageMeter,
    attach_patient_mean_predictions,
    audit_laterality_label_mixes,
    patient_level_aggregate,
    seed_all,
    timeSince,
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


def _best_model_path(args):
    if args.label.lower() in {"density", "birads"}:
        model_name = f"{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth"
    else:
        model_name = f"{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth"
    return args.chk_pt_path / model_name


def _metrics_csv_path(args):
    return args.output_path / f"fold{args.cur_fold}_metrics.csv"


def _loss_curve_path(args):
    return args.output_path / f"fold{args.cur_fold}_loss_curve.png"


def _save_fold_metrics(args, history_rows):
    metrics_df = pd.DataFrame(history_rows)
    metrics_df.to_csv(_metrics_csv_path(args), index=False)
    print(f"Fold {args.cur_fold} metrics saved to: {_metrics_csv_path(args)}")
    return metrics_df


def _save_loss_curve(args, metrics_df):
    required_cols = {"epoch", "train_loss", "eval_loss"}
    if metrics_df.empty or not required_cols.issubset(metrics_df.columns):
        print(f"Fold {args.cur_fold} loss curve skipped: no epoch loss history.")
        return

    train_rows = metrics_df[metrics_df["train_loss"].notna()]
    eval_rows = metrics_df[metrics_df["eval_loss"].notna()]
    if train_rows.empty and eval_rows.empty:
        print(f"Fold {args.cur_fold} loss curve skipped: all loss values are empty.")
        return

    eval_name = getattr(args, "eval_split", "eval")
    if "eval_split" in metrics_df.columns and metrics_df["eval_split"].notna().any():
        eval_name = str(metrics_df["eval_split"].dropna().iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    if not train_rows.empty:
        ax.plot(train_rows["epoch"], train_rows["train_loss"], marker="o", linewidth=2, label="train loss")
    if not eval_rows.empty:
        ax.plot(eval_rows["epoch"], eval_rows["eval_loss"], marker="o", linewidth=2, label=f"{eval_name} loss")
    ax.set_title(f"Fold {args.cur_fold} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    curve_path = _loss_curve_path(args)
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Fold {args.cur_fold} loss curve saved to: {curve_path}")


def _save_combined_loss_curves(args, metrics_history):
    valid_metrics = []
    for metrics_df in metrics_history:
        if metrics_df.empty or not {"fold", "epoch", "train_loss", "eval_loss"}.issubset(metrics_df.columns):
            continue
        if metrics_df[["train_loss", "eval_loss"]].notna().any().any():
            valid_metrics.append(metrics_df)

    if not valid_metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metrics_df in valid_metrics:
        fold = int(metrics_df["fold"].iloc[0])
        train_rows = metrics_df[metrics_df["train_loss"].notna()]
        eval_rows = metrics_df[metrics_df["eval_loss"].notna()]
        if not train_rows.empty:
            axes[0].plot(train_rows["epoch"], train_rows["train_loss"], marker="o", linewidth=1.5, label=f"fold{fold}")
        if not eval_rows.empty:
            axes[1].plot(eval_rows["epoch"], eval_rows["eval_loss"], marker="o", linewidth=1.5, label=f"fold{fold}")

    axes[0].set_title("Train Loss")
    axes[1].set_title("Eval Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()

    curve_path = args.output_path / "loss_curves_summary.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Combined loss curves saved to: {curve_path}")


def _save_eval_predictions(args, oof_df):
    if len(oof_df) == 0:
        return

    oof_df = oof_df.reset_index(drop=True)
    agg = oof_df.groupby("patient_id").agg({
        args.label: "max",
        "prediction": "mean",
    }).reset_index()

    eval_name = getattr(args, "eval_split", "eval")
    if args.n_folds > 0:
        print("\n================ CV (Out-of-Fold) ================")
        aucroc_value = auroc(gt=agg[args.label].values.astype(int), pred=agg["prediction"].values)
        print(f"OOF AUC-ROC: {aucroc_value:.4f}")
    else:
        print(f"\n================ Holdout {eval_name.upper()} Predictions ================")
        aucroc_value = auroc(gt=agg[args.label].values.astype(int), pred=agg["prediction"].values)
        print(f"{eval_name.upper()} AUC-ROC from best checkpoint: {aucroc_value:.4f}")

    oof_path = args.output_path / f"seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF-style predictions saved to: {oof_path}")

    if args.n_folds == 0:
        eval_path = args.output_path / f"seed_{args.seed}_{eval_name}_outputs.csv"
        oof_df.to_csv(eval_path, index=False)
        print(f"Holdout {eval_name} predictions also saved to: {eval_path}")


def _best_metric_column(args):
    return "eval_accuracy" if args.label.lower() in {"density", "birads"} else "eval_aucroc"


def _record_fold_summary(args, metrics_df, summaries):
    if metrics_df.empty:
        return

    metric_col = _best_metric_column(args)
    if metric_col not in metrics_df.columns:
        return

    valid_rows = metrics_df[metric_col].notna()
    if not valid_rows.any():
        return

    best_idx = metrics_df.loc[valid_rows, metric_col].idxmax()
    best_row = metrics_df.loc[best_idx]
    summaries.append(
        {
            "fold": args.cur_fold,
            "eval_split": best_row["eval_split"],
            "best_epoch": int(best_row["epoch"]),
            "metric_name": metric_col,
            "best_metric": float(best_row[metric_col]),
            "checkpoint_path": str(_best_model_path(args)),
            "metrics_csv": str(_metrics_csv_path(args)),
        }
    )


def do_experiments(args, device):
    if "efficientnetv2" in args.arch:
        args.model_base_name = "efficientv2_s"
    elif "efficientnet_b5_ns" in args.arch:
        args.model_base_name = "efficientnetb5"
    else:
        args.model_base_name = args.arch

    args.data_dir = Path(args.data_dir)
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = args.data_dir / csv_path

    args.df = pd.read_csv(csv_path)
    args.df = args.df.fillna(0)
    audit_laterality_label_mixes(args.df, args.label, context="classifier input CSV")
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
    metrics_history = []

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

        if args.inference_mode == "y":
            eval_df, metrics_df = inference_loop(args)
        else:
            eval_df, metrics_df = train_loop(args, device)

        eval_df = eval_df.copy()
        eval_df["eval_split"] = args.eval_split
        oof_df = pd.concat([oof_df, eval_df])

        _record_fold_summary(args, metrics_df, fold_summaries)
        if not metrics_df.empty:
            metrics_with_fold = metrics_df.copy()
            metrics_with_fold["fold"] = fold
            metrics_history.append(metrics_with_fold)

        best_model_path = _best_model_path(args)
        if len(predict_df) > 0 and best_model_path.exists():
            fold_output = predict_on_dataset(args, predict_df, best_model_path, device, fold)
            fold_output = _build_fold_split_view(fold_output, fold, args.n_folds)
            fold_csv_path = args.output_path / f"fold{fold}_all_predictions.csv"
            fold_output.to_csv(fold_csv_path, index=False)
            fold_prediction_arrays.append(fold_output["prediction_prob"].values)
            print(f"Fold {fold} all-data predictions saved to: {fold_csv_path}")

    _save_eval_predictions(args, oof_df)
    _save_combined_loss_curves(args, metrics_history)

    if len(predict_df) > 0 and len(fold_prediction_arrays) > 0:
        ensemble_predictions = np.mean(fold_prediction_arrays, axis=0)
        ensemble_output = predict_df.copy()
        ensemble_output["patient_prediction_prob"] = ensemble_predictions
        ensemble_output["prediction_prob"] = ensemble_predictions
        ensemble_output["prediction_label"] = (ensemble_predictions >= 0.5).astype(int)
        ensemble_csv_path = args.output_path / "ensemble_all_predictions.csv"
        ensemble_output.to_csv(ensemble_csv_path, index=False)
        print(f"\nEnsemble all-data predictions saved to: {ensemble_csv_path}")
    elif len(predict_df) > 0:
        print("Warning: no fold predictions available; ensemble_all_predictions.csv was not written.")

    if fold_summaries:
        summary_df = pd.DataFrame(fold_summaries)
        summary_path = args.output_path / "fold_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Fold metrics summary saved to: {summary_path}")

    print("\n================ Done! ================")


def train_loop(args, device):
    print(f"\n================== fold: {args.cur_fold} training ======================")
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    if args.clip_chk_pt_path is not None:
        ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
        if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = None
        ckpt = None

    if args.running_interactive:
        args.train_folds = args.train_folds.sample(min(1000, len(args.train_folds)))
        args.valid_folds = args.valid_folds.sample(n=min(1000, len(args.valid_folds)))

    train_loader, valid_loader = get_dataloader(args)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}")

    n_class = 1
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads":
        n_class = 3

    optimizer = None
    scheduler = None
    mapper = None
    attr_embs = None
    if "breast_clip" in args.arch:
        print(f"Architecture: {args.arch}")
        print(args.image_encoder_type)
        model = BreastClipClassifier(args, ckpt=ckpt, n_class=n_class)
        print("Model is loaded")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup_epochs == 0.1:
            warmup_steps = args.epochs
        elif args.warmup_epochs == 1:
            warmup_steps = len(train_loader)
        else:
            warmup_steps = 10
        lr_config = {
            "total_epochs": args.epochs,
            "warmup_steps": warmup_steps,
            "total_steps": len(train_loader) * args.epochs,
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)

    model = model.to(device)
    print(model)

    logger = SummaryWriter(args.tb_logs_path / f"fold{args.cur_fold}")

    if args.label.lower() in {"density", "birads"}:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to(device)
        print(f"pos_wt: {pos_wt}")
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    best_aucroc = 0.0
    best_acc = 0.0
    epochs_no_improve = 0
    history_rows = []
    eval_name = getattr(args, "eval_split", "val")

    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device
        )

        if (
            "efficientnetv2" in args.arch or "efficientnet_b5_ns" in args.arch
            or "efficientnet_b5_ns-detect" in args.arch or "efficientnetv2-detect" in args.arch
        ):
            scheduler.step()

        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, mapper=mapper, attr_embs=attr_embs, logger=logger
        )
        args.valid_folds["prediction"] = predictions

        if args.label.lower() in {"density", "birads"}:
            valid_agg = args.valid_folds[["patient_id", args.label, "prediction", "fold"]].groupby(["patient_id"]).mean()
            correct_predictions = (valid_agg[args.label] == valid_agg["prediction"]).sum()
            total_predictions = len(valid_agg)
            accuracy = correct_predictions / total_predictions
            valid_agg[args.label] = valid_agg[args.label].astype(int)
            valid_agg["prediction"] = valid_agg["prediction"].astype(int)
            f1 = f1_score(valid_agg[args.label], valid_agg["prediction"], average="macro")

            print(
                f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  "
                f"accuracy: {accuracy * 100:.4f}   f1: {f1 * 100:.4f}"
            )
            logger.add_scalar(f"{eval_name}/{args.label}/accuracy", accuracy, epoch + 1)
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_loss": avg_val_loss,
                    "eval_split": eval_name,
                    "eval_aucroc": np.nan,
                    "eval_accuracy": accuracy,
                    "eval_f1": f1,
                }
            )

            if epoch == 0 or best_acc < accuracy:
                best_acc = accuracy
                epochs_no_improve = 0
                print(f"Epoch {epoch + 1} - Save Best acc: {best_acc * 100:.4f} Model")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "predictions": predictions,
                        "epoch": epoch,
                        "accuracy": accuracy,
                        "f1": f1,
                    },
                    _best_model_path(args),
                )
            else:
                epochs_no_improve += 1
        else:
            valid_agg = patient_level_aggregate(args.valid_folds, args.label, "prediction")
            aucroc_value = auroc(valid_agg[args.label].values.astype(int), valid_agg["prediction"].values)
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_{eval_name}_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )
            print(f"Epoch {epoch + 1} - {eval_name.upper()} AUC-ROC Score: {aucroc_value:.4f}")
            logger.add_scalar(f"{eval_name}/{args.label}/AUC-ROC", aucroc_value, epoch + 1)
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_loss": avg_val_loss,
                    "eval_split": eval_name,
                    "eval_aucroc": aucroc_value,
                    "eval_accuracy": np.nan,
                    "eval_f1": np.nan,
                }
            )

            if epoch == 0 or best_aucroc < aucroc_value:
                best_aucroc = aucroc_value
                epochs_no_improve = 0
                print(f"Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "predictions": predictions,
                        "epoch": epoch,
                        "auroc": aucroc_value,
                    },
                    _best_model_path(args),
                )
            else:
                epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            if args.label.lower() in {"density", "birads"}:
                print(
                    f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, "
                    f"best Accuracy: {best_acc * 100:.4f}"
                )
            else:
                print(
                    f"Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, "
                    f"best AUC-ROC: {best_aucroc:.4f}"
                )
            break

        if args.label.lower() in {"density", "birads"}:
            print(f"[Fold{args.cur_fold}], Best Accuracy: {best_acc * 100:.4f}")
        else:
            print(f"[Fold{args.cur_fold}], Best AUC-ROC: {best_aucroc:.4f}")

    best_model_path = _best_model_path(args)
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location="cpu", weights_only=False)["predictions"]
        args.valid_folds["prediction"] = predictions
    else:
        print(f"Warning: No best model checkpoint found at {best_model_path}")

    metrics_df = _save_fold_metrics(args, history_rows)
    _save_loss_curve(args, metrics_df)
    logger.close()
    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds.copy(), metrics_df


def inference_loop(args):
    print(f"================== fold: {args.cur_fold} validating ======================")
    print(args.valid_folds.shape)
    best_model_path = _best_model_path(args)
    predictions = torch.load(best_model_path, map_location="cpu", weights_only=False)["predictions"]
    print(f"predictions: {predictions.shape} {type(predictions)}")
    args.valid_folds["prediction"] = predictions

    valid_agg = patient_level_aggregate(args.valid_folds, args.label, "prediction")
    aucroc_value = auroc(valid_agg[args.label].values.astype(int), valid_agg["prediction"].values)
    print(f"Fold {args.cur_fold} AUC-ROC: {aucroc_value:.4f}")
    metrics_df = _save_fold_metrics(
        args,
        [
            {
                "epoch": 0,
                "train_loss": np.nan,
                "eval_loss": np.nan,
                "eval_split": getattr(args, "eval_split", "val"),
                "eval_aucroc": aucroc_value,
                "eval_accuracy": np.nan,
                "eval_f1": np.nan,
            }
        ],
    )
    return args.valid_folds.copy(), metrics_df


def predict_on_dataset(args, df, model_path, device, fold):
    print(f"\n=== Predicting all data with fold {fold} model ===")

    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    model = BreastClipClassifier(args, ckpt=ckpt, n_class=1)
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

    preds = []
    with torch.no_grad():
        for data in tqdm(predict_loader, desc=f"Predicting fold{fold}"):
            inputs = data["x"].to(device)
            if (
                args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "breast_clip_det_b2_period_n_lp"
            ):
                inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
            elif args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
                inputs = inputs.squeeze(1)

            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)

            preds.append(y_preds.squeeze(1).sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)

    predict_output = attach_patient_mean_predictions(df, predictions)
    predictions_agg = predict_output["prediction_prob"].values

    print(
        f"Fold {fold} all-data prediction stats: mean={predictions_agg.mean():.4f}, "
        f"min={predictions_agg.min():.4f}, max={predictions_agg.max():.4f}"
    )

    torch.cuda.empty_cache()
    gc.collect()
    return predict_output


def ensemble_predictions_for_dataset(args, df, best_model_paths, device):
    print("\n================ Ensemble All-Data Predictions ================")

    all_predictions = []
    for fold, model_path in enumerate(best_model_paths):
        if model_path.exists():
            fold_output = predict_on_dataset(args, df, model_path, device, fold)
            all_predictions.append(fold_output["prediction_prob"].values)
        else:
            print(f"Warning: Model not found: {model_path}")

    if len(all_predictions) == 0:
        print("Error: No model predictions available for ensemble!")
        return np.zeros(len(df))

    ensemble_pred = np.mean(all_predictions, axis=0)
    print(
        f"Ensemble prediction stats: mean={ensemble_pred.mean():.4f}, "
        f"min={ensemble_pred.min():.4f}, max={ensemble_pred.max():.4f}"
    )
    return ensemble_pred


def get_dataloader(args):
    train_tfm = None
    val_tfm = None

    if args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
        import torchvision

        color_jitter_transform = torchvision.transforms.ColorJitter(
            brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1
        )
        normalize_transform = torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        train_tfm = torchvision.transforms.Compose([
            color_jitter_transform,
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ])
        val_tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ])
    else:
        train_tfm = get_transforms(args)
        val_tfm = get_eval_transforms(args)

    train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
    valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator_mammo_dataset_w_concepts,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts,
    )

    return train_loader, valid_loader


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]", total=len(train_loader))
    for step, data in progress_iter:
        inputs = data["x"].to(device)
        if (
            args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
            args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
            inputs = inputs.squeeze(1)

        batch_size = inputs.size(0)
        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({"img": inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                scores = torch.nn.functional.normalize(scores, p=2, dim=1)
                inputs_dict = {"img": inputs, "scores": scores}
                with torch.cuda.amp.autocast(enabled=args.apex):
                    y_preds = model(inputs_dict)
        else:
            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)
        if args.label == "density" or args.label.lower() == "birads":
            labels = data["y"].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data["y"].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if "breast_clip" in args.arch:
            scheduler.step()
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]["lr"]],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] Elapsed {remain:s} Loss: {loss.val:.4f}({loss.avg:.4f}) LR: {lr:.8f}".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar("train/epoch", epoch, index)
            logger.add_scalar("train/iter_loss", losses.avg, index)
            logger.add_scalar("train/iter_lr", optimizer.param_groups[0]["lr"], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, mapper=None, attr_embs=None, logger=None):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]", total=len(valid_loader))
    for step, data in progress_iter:
        inputs = data["x"].to(device)
        batch_size = inputs.size(0)
        if (
            args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
            args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
            args.arch.lower() == "breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == "swin_tiny_custom_norm" or args.arch.lower() == "swin_base_custom_norm":
            inputs = inputs.squeeze(1)

        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({"img": inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                inputs_dict = {"img": inputs, "scores": scores}
                with torch.no_grad():
                    y_preds = model(inputs_dict)
        else:
            with torch.no_grad():
                y_preds = model(inputs)

        if args.label == "density" or args.label.lower() == "birads":
            labels = data["y"].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data["y"].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        if args.label == "density" or args.label.lower() == "birads":
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().to("cpu").numpy())

        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] Elapsed {remain:s} Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar("valid/iter_loss", losses.avg, index)

    if args.label == "density" or args.label.lower() == "birads":
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)
    return losses.avg, predictions
