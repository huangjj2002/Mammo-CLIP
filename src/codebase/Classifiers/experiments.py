

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_concepts import MammoDataset, collator_mammo_dataset_w_concepts
from Datasets.dataset_utils import get_eval_transforms, get_transforms
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import pfbeta_binarized, pr_auc, compute_auprc, auroc, compute_accuracy_np_array
from utils import seed_all, AverageMeter, timeSince


def do_experiments(args, device):

    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
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

    for fold in range(args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)

        args.train_folds = train_df[train_df['fold'] != fold].reset_index(drop=True)
        args.valid_folds = train_df[train_df['fold'] == fold].reset_index(drop=True)
        print(f"\n=== Fold {fold}: train={len(args.train_folds)}, valid={len(args.valid_folds)} ===")

        if args.inference_mode == 'y':
            _oof_df = inference_loop(args)
        else:
            _oof_df = train_loop(args, device)

        oof_df = pd.concat([oof_df, _oof_df])

   
        model_name = f'{args.model_base_name}_seed_{args.seed}_fold{fold}_best_aucroc_ver{args.VER}.pth'
        best_model_path = args.chk_pt_path / model_name


        if len(predict_df) > 0 and best_model_path.exists():
            fold_predictions = predict_on_dataset(args, predict_df, best_model_path, device, fold)
            fold_output = predict_df.copy()
            fold_output['prediction_prob'] = fold_predictions
            fold_output['prediction_label'] = (fold_predictions >= 0.5).astype(int)
            fold_csv_path = args.output_path / f'fold{fold}_all_predictions.csv'
            fold_output.to_csv(fold_csv_path, index=False)
            fold_prediction_arrays.append(fold_predictions)
            print(f"Fold {fold} all-data predictions saved to: {fold_csv_path}")


    if len(oof_df) > 0:
        oof_df = oof_df.reset_index(drop=True)
        print('\n================ CV (Out-of-Fold) ================')
  
        oof_agg = oof_df.groupby('patient_id').agg({
            args.label: 'max',
            'prediction': 'mean'
        }).reset_index()
        aucroc = auroc(gt=oof_agg[args.label].values.astype(int), pred=oof_agg['prediction'].values)
        print(f'OOF AUC-ROC: {aucroc:.4f}')
        oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_oof_outputs.csv', index=False)


    if len(predict_df) > 0 and len(fold_prediction_arrays) > 0:
        ensemble_predictions = np.mean(fold_prediction_arrays, axis=0)
        ensemble_output = predict_df.copy()
        ensemble_output['prediction_prob'] = ensemble_predictions
        ensemble_output['prediction_label'] = (ensemble_predictions >= 0.5).astype(int)
        ensemble_csv_path = args.output_path / 'ensemble_all_predictions.csv'
        ensemble_output.to_csv(ensemble_csv_path, index=False)
        print(f"\nEnsemble all-data predictions saved to: {ensemble_csv_path}")
    elif len(predict_df) > 0:
        print("Warning: no fold predictions available; ensemble_all_predictions.csv was not written.")

    print("\n================ Done! ================")


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
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
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    n_class = 1
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads":
        n_class = 3

    optimizer = None
    scheduler = None
    mapper = None
    attr_embs = None
    if 'breast_clip' in args.arch:
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
            'total_epochs': args.epochs,
            'warmup_steps': warmup_steps,
            'total_steps': len(train_loader) * args.epochs
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler()

    model = model.to(device)
    print(model)

    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

    if args.label.lower() == "density" or args.label.lower() == "birads":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to(device)
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    best_aucroc = 0.
    best_acc = 0
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device
        )

        if (
                'efficientnetv2' in args.arch or 'efficientnet_b5_ns' in args.arch
                or 'efficientnet_b5_ns-detect' in args.arch or 'efficientnetv2-detect' in args.arch
        ):
            scheduler.step()

        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, mapper=mapper, attr_embs=attr_embs, logger=logger
        )
        args.valid_folds['prediction'] = predictions


        valid_agg = args.valid_folds[['patient_id', args.label, 'prediction', 'fold']].groupby(
            ['patient_id']).mean()

        if args.label.lower() == "density" or args.label.lower() == "birads":
            correct_predictions = (valid_agg[args.label] == valid_agg['prediction']).sum()
            total_predictions = len(valid_agg)
            accuracy = correct_predictions / total_predictions
            valid_agg[args.label] = valid_agg[args.label].astype(int)
            valid_agg['prediction'] = valid_agg['prediction'].astype(int)
            f1 = f1_score(valid_agg[args.label], valid_agg['prediction'], average='macro')

            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  '
                f'accuracy: {accuracy * 100:.4f}   f1: {f1 * 100:.4f}'
            )
            logger.add_scalar(f'valid/{args.label}/accuracy', accuracy, epoch + 1)

            if epoch == 0 or best_acc < accuracy:
                best_acc = accuracy
                epochs_no_improve = 0
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save Best acc: {best_acc * 100:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'accuracy': accuracy,
                        'f1': f1,
                    }, args.chk_pt_path / model_name
                )
            else:
                epochs_no_improve += 1
        else:
            aucroc = auroc(valid_agg[args.label].values.astype(int), valid_agg['prediction'].values)
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
            )
            print(f'Epoch {epoch + 1} - AUC-ROC Score: {aucroc:.4f}')
            logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc, epoch + 1)

            if epoch == 0 or best_aucroc < aucroc:
                best_aucroc = aucroc
                epochs_no_improve = 0
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'auroc': aucroc,
                    }, args.chk_pt_path / model_name
                )
            else:
                epochs_no_improve += 1


        if args.patience > 0 and epochs_no_improve >= args.patience:
            if args.label.lower() == "density" or args.label.lower() == "birads":
                print(f'Early stopping at epoch {epoch+1}: no improvement for {args.patience} epochs, '
                      f'best Accuracy: {best_acc * 100:.4f}')
            else:
                print(f'Early stopping at epoch {epoch+1}: no improvement for {args.patience} epochs, '
                      f'best AUC-ROC: {best_aucroc:.4f}')
            break


        if args.label.lower() == "density" or args.label.lower() == "birads":
            print(f'[Fold{args.cur_fold}], Best Accuracy: {best_acc * 100:.4f}')
        else:
            print(f'[Fold{args.cur_fold}], AUC-ROC Score: {best_aucroc:.4f}')


    if args.label.lower() == "density" or args.label.lower() == "birads":
        model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
    else:
        model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'

    best_model_path = args.chk_pt_path / model_name
    if best_model_path.exists():
        predictions = torch.load(best_model_path, map_location='cpu', weights_only=False)['predictions']
        args.valid_folds['prediction'] = predictions
    else:
        print(f"Warning: No best model checkpoint found at {best_model_path}")

    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds


def inference_loop(args):
    print(f'================== fold: {args.cur_fold} validating ======================')
    print(args.valid_folds.shape)
    model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
    predictions = torch.load(
        args.chk_pt_path / model_name,
        map_location='cpu', weights_only=False)['predictions']
    print(f'predictions: {predictions.shape}', type(predictions))
    args.valid_folds['prediction'] = predictions

    valid_agg = args.valid_folds[['patient_id', args.label, 'prediction', 'fold']].groupby(
        ['patient_id']).mean()
    aucroc = auroc(valid_agg[args.label].values.astype(int), valid_agg['prediction'].values)
    print(f'Fold {args.cur_fold} AUC-ROC: {aucroc:.4f}')
    return args.valid_folds.copy()


def predict_on_dataset(args, df, model_path, device, fold):
    """使用单个模型对test集进行预测"""
    print(f'\n=== Predicting all data with fold {fold} model ===')


    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu", weights_only=False)
    if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
    elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
        args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    model = BreastClipClassifier(args, ckpt=ckpt, n_class=1)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

  
    predict_dataset = MammoDataset(args=args, df=df, transform=get_eval_transforms(args))
    predict_loader = DataLoader(
        predict_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=collator_mammo_dataset_w_concepts
    )


    preds = []
    with torch.no_grad():
        for data in tqdm(predict_loader, desc=f"Predicting fold{fold}"):
            inputs = data['x'].to(device)
            if (
                    args.arch.lower() == "breast_clip_det_b5_period_n_ft" or
                    args.arch.lower() == "breast_clip_det_b5_period_n_lp" or
                    args.arch.lower() == "breast_clip_det_b2_period_n_ft" or
                    args.arch.lower() == "breast_clip_det_b2_period_n_lp"
            ):
                inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
            elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
                inputs = inputs.squeeze(1)

            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)

            preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

    predictions = np.concatenate(preds)


    predict_temp = df.copy()
    predict_temp['prediction_prob'] = predictions
    patient_pred = predict_temp.groupby('patient_id')['prediction_prob'].mean()

    predictions_agg = df['patient_id'].map(patient_pred).values

    print(f"Fold {fold} all-data prediction stats: mean={predictions_agg.mean():.4f}, "
          f"min={predictions_agg.min():.4f}, max={predictions_agg.max():.4f}")

    torch.cuda.empty_cache()
    gc.collect()

    return predictions_agg


def ensemble_predictions_for_dataset(args, df, best_model_paths, device):
    """使用多个模型的预测取平均（ensemble）"""
    print(f'\n================ Ensemble All-Data Predictions ================')

    all_predictions = []
    for fold, model_path in enumerate(best_model_paths):
        if model_path.exists():
            fold_preds = predict_on_dataset(args, df, model_path, device, fold)
            all_predictions.append(fold_preds)
        else:
            print(f"Warning: Model not found: {model_path}")

    if len(all_predictions) == 0:
        print("Error: No model predictions available for ensemble!")
        return np.zeros(len(df))


    ensemble_pred = np.mean(all_predictions, axis=0)
    print(f"Ensemble prediction stats: mean={ensemble_pred.mean():.4f}, "
          f"min={ensemble_pred.min():.4f}, max={ensemble_pred.max():.4f}")

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
            normalize_transform
        ])
        val_tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
    else:
        train_tfm = get_transforms(args)
        val_tfm = get_eval_transforms(args)

    train_dataset = MammoDataset(args=args, df=args.train_folds, transform=train_tfm)
    valid_dataset = MammoDataset(args=args, df=args.valid_folds, transform=val_tfm)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collator_mammo_dataset_w_concepts
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=collator_mammo_dataset_w_concepts
    )

    return train_loader, valid_loader


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
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
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        batch_size = inputs.size(0)
        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
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
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.cuda.amp.autocast(enabled=args.apex):
                    y_preds = model(inputs_dict)
        else:
            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)
        if args.label == "density" or args.label.lower() == "birads":
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if 'breast_clip' in args.arch:
            scheduler.step()
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

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


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, mapper=None, attr_embs=None, logger=None):
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
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.no_grad():
                    y_preds = model(inputs_dict)
        else:
            with torch.no_grad():
                y_preds = model(inputs)

        if args.label == "density" or args.label.lower() == "birads":
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        if args.label == "density" or args.label.lower() == "birads":
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())

        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

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

    if args.label == "density" or args.label.lower() == "birads":
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)
    return losses.avg, predictions
