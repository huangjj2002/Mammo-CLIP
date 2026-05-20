"""
Evidential Deep Learning (EDL) 训练/验证/测试模块

实现完整的5折交叉验证训练流程：
  - 每折训练 → 保存最佳模型
  - 训练结束后自动调用测试模块
  - 生成包含evidence、uncertainty、fold列的CSV文件
  - 5折ensemble预测
"""

import gc
import time
from pathlib import Path

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
from utils import seed_all, AverageMeter, timeSince


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


def do_edl_experiments(args, device):
    """
    EDL实验主入口：5折交叉验证训练 + 测试
    
    Args:
        args: 参数对象（需包含 data_dir, csv_file, n_folds, label 等）
        device: 计算设备
    """
    args.model_base_name = args.arch
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    train_df = args.df[args.df['fold'] >= 0].reset_index(drop=True)
    test_df = args.df[args.df['fold'] == -1].reset_index(drop=True)
    predict_df = args.df.reset_index(drop=True)
    print(f"Training samples (5 folds): {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Prediction output samples (all rows): {len(predict_df)}")

    oof_df = pd.DataFrame()
    fold_prediction_arrays = []

    # ===================== 5折训练 =====================
    for fold in range(args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)

        args.train_folds = train_df[train_df['fold'] != fold].reset_index(drop=True)
        args.valid_folds = train_df[train_df['fold'] == fold].reset_index(drop=True)
        print(f"\n=== Fold {fold}: train={len(args.train_folds)}, valid={len(args.valid_folds)} ===")

        _oof_df = edl_train_loop(args, device)
        oof_df = pd.concat([oof_df, _oof_df])

        # 保存最佳模型路径
        model_name = f'edl_{args.model_base_name}_seed_{args.seed}_fold{fold}_best_aucroc.pth'
        best_model_path = args.chk_pt_path / model_name

        # 每折训练完成后，用该模型对全量数据进行预测
        if len(predict_df) > 0 and best_model_path.exists():
            fold_results = edl_predict_on_dataset(args, predict_df, best_model_path, device, fold)
            fold_prediction_arrays.append(fold_results)
            fold_csv_path = args.output_path / f'edl_fold{fold}_all_predictions.csv'
            fold_results.to_csv(fold_csv_path, index=False)
            print(f"Fold {fold} predictions saved to: {fold_csv_path}")
            print(f"Fold {fold} all-data predictions completed.")

    # ===================== OOF结果汇总 =====================
    if len(oof_df) > 0:
        oof_df = oof_df.reset_index(drop=True)
        print('\n================ CV (Out-of-Fold) ================')
        oof_agg = oof_df.groupby('patient_id').agg({
            args.label: 'max',
            'prediction_prob': 'mean'
        }).reset_index()
        aucroc_val = auroc(gt=oof_agg[args.label].values.astype(int), pred=oof_agg['prediction_prob'].values)
        print(f'OOF AUC-ROC: {aucroc_val:.4f}')
        oof_df.to_csv(args.output_path / f'edl_seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv', index=False)

    # ===================== 自动测试：全量数据预测 =====================
    print("\n" + "=" * 60)
    print("Auto-testing: Generating predictions for all data...")
    print("=" * 60)

    if len(predict_df) > 0 and len(fold_prediction_arrays) > 0:
        # Ensemble：5折模型的预测取平均
        ensemble_output = predict_df.copy()

        # 收集所有fold的详细预测结果
        all_evidence_cols = [f'evidence_{i}' for i in range(args.num_classes)]
        all_alpha_cols = [f'alpha_{i}' for i in range(args.num_classes)]
        all_prob_cols = [f'probability_{i}' for i in range(args.num_classes)]

        # 逐列平均
        for col in all_evidence_cols + all_alpha_cols + all_prob_cols + ['total_uncertainty', 'prediction_prob']:
            ensemble_output[col] = np.mean([fd[col].values for fd in fold_prediction_arrays], axis=0)

        ensemble_output['prediction_label'] = (ensemble_output['prediction_prob'] >= 0.5).astype(int)

        # 保存ensemble结果
        ensemble_csv_path = args.output_path / 'edl_ensemble_all_predictions.csv'
        ensemble_output['model_fold'] = 'ensemble'
        ensemble_output.to_csv(ensemble_csv_path, index=False)
        print(f"\n  Ensemble all-data predictions saved to: {ensemble_csv_path}")

        # 打印ensemble统计
        print(f"\n  Ensemble prediction stats:")
        print(f"    prediction_prob: mean={ensemble_output['prediction_prob'].mean():.4f}, "
              f"min={ensemble_output['prediction_prob'].min():.4f}, "
              f"max={ensemble_output['prediction_prob'].max():.4f}")
        print(f"    total_uncertainty: mean={ensemble_output['total_uncertainty'].mean():.4f}, "
              f"min={ensemble_output['total_uncertainty'].min():.4f}, "
              f"max={ensemble_output['total_uncertainty'].max():.4f}")
        for i in range(args.num_classes):
            print(f"    evidence_{i}: mean={ensemble_output[f'evidence_{i}'].mean():.4f}")
            print(f"    alpha_{i}: mean={ensemble_output[f'alpha_{i}'].mean():.4f}")
            print(f"    probability_{i}: mean={ensemble_output[f'probability_{i}'].mean():.4f}")

    elif len(predict_df) > 0:
        print("Warning: no fold predictions available; ensemble predictions were not generated.")

    print("\n================ EDL Done! ================")


def edl_train_loop(args, device):
    """
    EDL单折训练循环
    
    Args:
        args: 参数对象
        device: 计算设备
    Returns:
        valid_folds: 包含OOF预测结果的DataFrame
    """
    print(f'\n================== EDL fold: {args.cur_fold} training ======================')

    # 加载预训练权重获取编码器配置
    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    # 创建数据加载器
    train_loader, valid_loader = edl_get_dataloader(args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    # 创建EDL模型
    num_classes = args.num_classes
    print(f"Creating EDL model with {num_classes} classes")
    model = BreastClipEDLClassifier(
        args, ckpt=ckpt, num_classes=num_classes,
        dropout=args.edl_dropout, hidden_dim=args.edl_hidden_dim
    )
    model = model.to(device)
    print(model)

    # 优化器
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters: {trainable_param_count:,} / {total_params:,}")
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check args.train_mode/freeze settings.")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
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

    # TensorBoard
    logger = SummaryWriter(args.tb_logs_path / f'edl_fold{args.cur_fold}')

    class_weights = _compute_fold_class_weights(args)

    # EDL损失函数（类交叉熵损失 + KL正则化）
    criterion = EDLLoss(
        num_classes=num_classes,
        loss_type=args.edl_loss_type,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
        class_weights=class_weights,
    )

    best_aucroc = 0.
    epochs_no_improve = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_aucroc": [],
    }

    for epoch in range(args.epochs):
        start_time = time.time()

        # 更新损失函数中的epoch（用于KL退火）
        criterion.current_epoch = epoch

        # 训练
        avg_loss = edl_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device
        )

        # 验证
        avg_val_loss, predictions = edl_valid_fn(
            valid_loader, model, criterion, args, device, epoch, logger=logger
        )
        args.valid_folds['prediction_prob'] = predictions

        # 计算AUC-ROC
        valid_agg = args.valid_folds[['patient_id', args.label, 'prediction_prob', 'fold']].groupby(
            ['patient_id']).mean()
        aucroc_val = auroc(valid_agg[args.label].values.astype(int), valid_agg['prediction_prob'].values)
        elapsed = time.time() - start_time

        print(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  '
            f'AUC-ROC: {aucroc_val:.4f}  time: {elapsed:.0f}s'
        )
        logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc_val, epoch + 1)
        logger.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
        logger.add_scalar('valid/epoch_loss', avg_val_loss, epoch + 1)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["valid_loss"].append(avg_val_loss)
        history["valid_aucroc"].append(aucroc_val)

        # 保存最佳模型
        if epoch == 0 or best_aucroc < aucroc_val:
            best_aucroc = aucroc_val
            epochs_no_improve = 0
            model_name = f'edl_{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc.pth'
            print(f'Epoch {epoch + 1} - Save Best AUC-ROC: {best_aucroc:.4f} Model')
            torch.save(
                {
                    'model': model.state_dict(),
                    'predictions': predictions,
                    'epoch': epoch,
                    'auroc': aucroc_val,
                    'train_mode': getattr(args, "train_mode", "full"),
                    'freeze_backbone': getattr(args, "freeze_backbone", "n"),
                }, args.chk_pt_path / model_name
            )
        else:
            epochs_no_improve += 1

        # 早停
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch + 1}: no improvement for {args.patience} epochs, '
                  f'best AUC-ROC: {best_aucroc:.4f}')
            break

        print(f'[Fold{args.cur_fold}], Best AUC-ROC: {best_aucroc:.4f}')

    # 加载最佳模型的预测结果
    model_name = f'edl_{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc.pth'
    best_model_path = args.chk_pt_path / model_name
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location='cpu', weights_only=False)['predictions']
        args.valid_folds['prediction_prob'] = predictions
    else:
        print(f"Warning: No best model checkpoint found at {best_model_path}")

    _save_fold_loss_curve(args, history)
    logger.close()
    _empty_cuda_cache(device)
    gc.collect()
    return args.valid_folds


def edl_predict_on_dataset(args, df, model_path, device, fold):
    """
    使用单个EDL模型对全量数据进行预测，返回详细的evidence、alpha、概率、不确定性信息
    
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
        for data in tqdm(predict_loader, desc=f"EDL Predicting fold{fold}"):
            inputs = data['x'].to(device)
            if (
                args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "breast_clip_det_b2_period_n_lp"
            ):
                inputs = inputs.squeeze(1).permute(0, 3, 1, 2)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                evidence = model(inputs)

            # 计算Dirichlet参数
            alpha = BreastClipEDLClassifier.compute_dirichlet_params(evidence)
            probs = BreastClipEDLClassifier.compute_probabilities(evidence)
            uncertainty = BreastClipEDLClassifier.compute_uncertainty(evidence)

            all_evidence.append(evidence.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_uncertainty.append(uncertainty.cpu().numpy())

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
    # 正类概率（类别1），兼容原有格式
    result_df['prediction_prob'] = probs_array[:, -1]
    result_df['prediction_label'] = (result_df['prediction_prob'] >= 0.5).astype(int)

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


def edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device):
    """EDL训练一个epoch"""
    model.train()
    amp_enabled = _amp_enabled(args, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    losses = AverageMeter()
    start = end = time.time()

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
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

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            evidence = model(inputs)
            loss = criterion(evidence, labels)

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()

        postfix = {
            "lr": [optimizer.param_groups[0]['lr']],
            "loss": f"{losses.avg:.4f}",
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

    return losses.avg


def edl_valid_fn(valid_loader, model, criterion, args, device, epoch=1, logger=None):
    """EDL验证一个epoch"""
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
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

        with torch.no_grad():
            evidence = model(inputs)
            loss = criterion(evidence, labels)

            # 计算正类概率用于AUC-ROC评估
            probs = BreastClipEDLClassifier.compute_probabilities(evidence)
            # 取正类（类别1）的概率
            pos_probs = probs[:, -1].cpu().numpy()

        losses.update(loss.item(), batch_size)
        preds.append(pos_probs)

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
    return losses.avg, predictions
