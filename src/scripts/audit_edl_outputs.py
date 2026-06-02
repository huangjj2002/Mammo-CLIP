from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


THRESHOLD = 0.5


def rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def split_values(df: pd.DataFrame) -> pd.Series:
    if "split" not in df.columns:
        return pd.Series("unknown", index=df.index)
    return df["split"].astype(str).str.strip().str.lower()


def choose_image_score_col(df: pd.DataFrame) -> str:
    for col in ("image_prediction_prob", "probability_1", "prediction_score", "prediction_prob"):
        if col in df.columns:
            return col
    raise ValueError("No image-level score column found")


def choose_patient_score_col(df: pd.DataFrame) -> str:
    for col in ("patient_prediction_prob", "prediction_prob"):
        if col in df.columns:
            return col
    raise ValueError("No patient-level score column found")


def class_counts(df: pd.DataFrame, label_col: str = "cancer") -> dict[str, int]:
    labels = pd.to_numeric(df[label_col], errors="coerce")
    return {
        "sample_n": int(len(labels)),
        "positive_n": int((labels == 1).sum()),
        "negative_n": int((labels == 0).sum()),
    }


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= THRESHOLD).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "AUC": rank_auc(y_true, y_score),
        "BACC@0.5": (sensitivity + specificity) / 2.0,
        "Sensitivity@0.5": sensitivity,
        "Specificity@0.5": specificity,
        "PPV@0.5": safe_div(tp, tp + fp),
        "NPV@0.5": safe_div(tn, tn + fn),
        "TP@0.5": tp,
        "TN@0.5": tn,
        "FP@0.5": fp,
        "FN@0.5": fn,
        "Pred_Pos@0.5": int(y_pred.sum()),
        "Pred_Neg@0.5": int(len(y_pred) - y_pred.sum()),
    }


def quantile_stats(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {
            "Score_Min": float("nan"),
            "Score_Q10": float("nan"),
            "Score_Q50": float("nan"),
            "Score_Q90": float("nan"),
            "Score_Max": float("nan"),
            "Score_Mean": float("nan"),
        }
    return {
        "Score_Min": float(values.min()),
        "Score_Q10": float(values.quantile(0.10)),
        "Score_Q50": float(values.quantile(0.50)),
        "Score_Q90": float(values.quantile(0.90)),
        "Score_Max": float(values.max()),
        "Score_Mean": float(values.mean()),
    }


def patient_frame(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        df.groupby("patient_id", as_index=False)
        .agg(cancer=("cancer", "max"), prediction_prob=(score_col, "mean"))
    )


def add_threshold_rows(rows: list[dict[str, object]], model: str, source_name: str, df: pd.DataFrame) -> None:
    splits = split_values(df)
    for split in ("train", "val", "test"):
        part = df[splits == split].copy()
        if part.empty:
            continue
        image_score_col = choose_image_score_col(part)
        image_scores = pd.to_numeric(part[image_score_col], errors="coerce")
        image_labels = pd.to_numeric(part["cancer"], errors="coerce")
        valid = image_scores.notna() & image_labels.notna() & np.isfinite(image_scores)
        row = {
            "Model": model,
            "Source": source_name,
            "Split": split,
            "Grain": "image",
            "Score_Column": image_score_col,
            "Sample_N": int(valid.sum()),
            "Patient_N": int(part.loc[valid, "patient_id"].nunique()) if "patient_id" in part else int(valid.sum()),
            "Positive_N": int(image_labels.loc[valid].sum()),
            "Negative_N": int(valid.sum() - image_labels.loc[valid].sum()),
        }
        row.update(threshold_metrics(image_labels.loc[valid].to_numpy(), image_scores.loc[valid].to_numpy()))
        row.update(quantile_stats(image_scores.loc[valid]))
        rows.append(row)

        patient_score_col = choose_patient_score_col(part)
        patients = patient_frame(part, patient_score_col)
        patient_scores = pd.to_numeric(patients["prediction_prob"], errors="coerce")
        patient_labels = pd.to_numeric(patients["cancer"], errors="coerce")
        valid = patient_scores.notna() & patient_labels.notna() & np.isfinite(patient_scores)
        row = {
            "Model": model,
            "Source": source_name,
            "Split": split,
            "Grain": "patient",
            "Score_Column": patient_score_col,
            "Sample_N": int(valid.sum()),
            "Patient_N": int(valid.sum()),
            "Positive_N": int(patient_labels.loc[valid].sum()),
            "Negative_N": int(valid.sum() - patient_labels.loc[valid].sum()),
        }
        row.update(threshold_metrics(patient_labels.loc[valid].to_numpy(), patient_scores.loc[valid].to_numpy()))
        row.update(quantile_stats(patient_scores.loc[valid]))
        rows.append(row)


def edl_loss_components(
    evidence: np.ndarray,
    labels: np.ndarray,
    loss_type: str,
    class_weights: list[float] | None,
    kl_weight: float,
    annealing_coef: float,
) -> dict[str, float]:
    output = torch.as_tensor(evidence, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
    alpha = torch.relu(output) + 1.0
    s = torch.sum(alpha, dim=1, keepdim=True)
    if loss_type == "digamma":
        per_sample_data = torch.sum(target_onehot * (torch.digamma(s) - torch.digamma(alpha)), dim=1)
    elif loss_type == "log":
        per_sample_data = torch.sum(target_onehot * (torch.log(s + 1e-10) - torch.log(alpha + 1e-10)), dim=1)
    elif loss_type == "mse":
        err = torch.sum((target_onehot - alpha / s) ** 2, dim=1)
        var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1)
        per_sample_data = err + var
    else:
        raise ValueError(f"Unsupported EDL loss type: {loss_type}")

    beta = torch.ones_like(alpha)
    kl_alpha = target_onehot + (1 - target_onehot) * alpha
    s_alpha = torch.sum(kl_alpha, dim=1, keepdim=True)
    s_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(kl_alpha), dim=1, keepdim=True)
    ln_b_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
    dg0 = torch.digamma(s_alpha)
    dg1 = torch.digamma(kl_alpha)
    per_sample_kl = torch.sum((kl_alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + ln_b + ln_b_uni

    unweighted_data = per_sample_data.mean()
    if class_weights is None:
        weighted_data = unweighted_data
    else:
        weights = torch.as_tensor(class_weights, dtype=torch.float32)[target]
        weighted_data = (per_sample_data * weights).sum() / weights.sum().clamp_min(1e-8)
    kl = per_sample_kl.mean()
    weighted_total = weighted_data + kl_weight * annealing_coef * kl
    unweighted_total = unweighted_data + kl_weight * annealing_coef * kl
    return {
        "Unweighted_Data_Loss": float(unweighted_data),
        "Weighted_Data_Loss": float(weighted_data),
        "KL_Loss": float(kl),
        "Unweighted_Total_Loss": float(unweighted_total),
        "Weighted_Total_Loss": float(weighted_total),
    }


def evidence_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("evidence_")]
    return sorted(cols, key=lambda item: int(item.rsplit("_", 1)[-1]))


def metric_reference(edl_dir: Path) -> dict[str, object]:
    metric_files = sorted(edl_dir.rglob("edl_fold0_metrics.csv"))
    if not metric_files:
        return {}
    metrics = pd.read_csv(metric_files[0])
    if metrics.empty or "eval_aucroc" not in metrics:
        return {"Metrics_File": str(metric_files[0])}
    best = metrics.loc[metrics["eval_aucroc"].idxmax()]
    return {
        "Metrics_File": str(metric_files[0]),
        "Best_Epoch": int(best.get("epoch", -1)),
        "Best_Eval_AUC": float(best.get("eval_aucroc", float("nan"))),
        "Best_Eval_Loss": float(best.get("eval_loss", float("nan"))),
    }


def add_loss_rows(
    rows: list[dict[str, object]],
    pred_path: Path,
    ref: dict[str, object],
    args: argparse.Namespace,
) -> None:
    df = pd.read_csv(pred_path)
    cols = evidence_columns(df)
    if not cols:
        return
    splits = split_values(df)
    train_counts = class_counts(df[splits == "train"])
    class_weights = None
    if train_counts["positive_n"] > 0:
        class_weights = [1.0, float(train_counts["negative_n"] / train_counts["positive_n"])]
    for split in ("train", "val", "test"):
        part = df[splits == split].copy()
        if part.empty:
            continue
        evidence = part[cols].to_numpy(dtype=np.float32)
        labels = pd.to_numeric(part["cancer"], errors="coerce").astype(int).to_numpy()
        components = edl_loss_components(
            evidence,
            labels,
            args.loss_type,
            class_weights,
            args.kl_weight,
            args.annealing_coef,
        )
        row = {
            "Prediction_File": str(pred_path),
            "Prediction_Source": pred_path.name,
            "Split": split,
            "Sample_N": int(len(part)),
            "Positive_N": int((labels == 1).sum()),
            "Negative_N": int((labels == 0).sum()),
            "Train_Negative_N": train_counts["negative_n"],
            "Train_Positive_N": train_counts["positive_n"],
            "Class_Weight_Negative": class_weights[0] if class_weights else None,
            "Class_Weight_Positive": class_weights[1] if class_weights else None,
            "Loss_Type": args.loss_type,
            "KL_Weight": args.kl_weight,
            "Annealing_Coef": args.annealing_coef,
        }
        row.update(ref)
        row.update(components)
        if split == "val" and "Best_Eval_Loss" in ref:
            row["Abs_Diff_BestEval_vs_UnweightedTotal"] = abs(ref["Best_Eval_Loss"] - row["Unweighted_Total_Loss"])
            row["Abs_Diff_BestEval_vs_WeightedTotal"] = abs(ref["Best_Eval_Loss"] - row["Weighted_Total_Loss"])
            row["Saved_Loss_Closer_To"] = (
                "weighted"
                if row["Abs_Diff_BestEval_vs_WeightedTotal"] < row["Abs_Diff_BestEval_vs_UnweightedTotal"]
                else "unweighted"
            )
        rows.append(row)


def add_direction_rows(rows: list[dict[str, object]], pred_path: Path) -> None:
    df = pd.read_csv(pred_path)
    if "probability_1" not in df.columns:
        return
    splits = split_values(df)
    for split in ("train", "val", "test"):
        part = df[splits == split].copy()
        if part.empty:
            continue
        labels = pd.to_numeric(part["cancer"], errors="coerce")
        scores = pd.to_numeric(part["probability_1"], errors="coerce")
        valid = labels.notna() & scores.notna() & np.isfinite(scores)
        labels = labels.loc[valid].astype(int).to_numpy()
        scores = scores.loc[valid].astype(float).to_numpy()
        rows.append(
            {
                "Prediction_File": str(pred_path),
                "Split": split,
                "Sample_N": int(len(labels)),
                "AUC_probability_1": rank_auc(labels, scores),
                "AUC_1_minus_probability_1": rank_auc(labels, 1.0 - scores),
                "Direction_Check": (
                    "ok_probability_1_higher_is_positive"
                    if rank_auc(labels, scores) >= rank_auc(labels, 1.0 - scores)
                    else "warning_reverse_is_higher"
                ),
            }
        )


def find_prediction_files(edl_dir: Path) -> list[Path]:
    files = []
    files.extend(sorted(edl_dir.rglob("edl_fold*_all_predictions.csv")))
    files.extend(sorted(edl_dir.rglob("edl_ensemble_all_predictions.csv")))
    seen = set()
    out = []
    for path in files:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit existing MammoCLIP EDL prediction outputs without retraining.")
    parser.add_argument("--edl-dir", type=Path, required=True, help="EDL output directory or its parent.")
    parser.add_argument("--origin-dir", type=Path, default=None, help="Optional Origin output directory for threshold comparison.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for audit CSV files.")
    parser.add_argument("--loss-type", default="digamma", choices=["digamma", "log", "mse"])
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--annealing-coef", type=float, default=1.0)
    args = parser.parse_args()

    edl_dir = args.edl_dir
    output_dir = args.output_dir or edl_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    edl_prediction_files = find_prediction_files(edl_dir)
    if not edl_prediction_files:
        raise FileNotFoundError(f"No EDL all-prediction CSV files found under {edl_dir}")

    ref = metric_reference(edl_dir)
    loss_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for pred_path in edl_prediction_files:
        add_loss_rows(loss_rows, pred_path, ref, args)
        add_direction_rows(direction_rows, pred_path)
        add_threshold_rows(threshold_rows, "EDL", pred_path.name, pd.read_csv(pred_path))

    if args.origin_dir is not None:
        for origin_path in sorted(args.origin_dir.rglob("*ensemble_all_predictions.csv")):
            add_threshold_rows(threshold_rows, "Origin", origin_path.name, pd.read_csv(origin_path))

    loss_df = pd.DataFrame(loss_rows)
    direction_df = pd.DataFrame(direction_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    loss_path = output_dir / "edl_loss_reconstruction.csv"
    direction_path = output_dir / "edl_probability_direction_audit.csv"
    threshold_path = output_dir / "edl_origin_threshold_diagnostics.csv"
    loss_df.to_csv(loss_path, index=False, encoding="utf-8-sig")
    direction_df.to_csv(direction_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {loss_path}")
    print(f"Wrote {direction_path}")
    print(f"Wrote {threshold_path}")
    if not loss_df.empty and "Saved_Loss_Closer_To" in loss_df.columns:
        print(loss_df[["Prediction_Source", "Split", "Best_Eval_Loss", "Unweighted_Total_Loss", "Weighted_Total_Loss", "Saved_Loss_Closer_To"]].to_string(index=False))
    if not threshold_df.empty:
        cols = ["Model", "Source", "Split", "Grain", "Score_Column", "Sample_N", "Positive_N", "AUC", "BACC@0.5", "Sensitivity@0.5", "Specificity@0.5", "Pred_Pos@0.5"]
        print(threshold_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
