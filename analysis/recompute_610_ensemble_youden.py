from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


RESULT_ROOT = Path(r"G:\610\Mammo-CLIP")
OUTPUT_DIR = RESULT_ROOT / "analysis" / "youden_ensemble_20260610"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 610


GRAINS = [
    {
        "name": "image",
        "group_cols": [],
        "score_agg": "none",
        "label_agg": "row",
    },
    {
        "name": "patient_id_mean",
        "group_cols": ["patient_id"],
        "score_agg": "mean",
        "label_agg": "max",
    },
    {
        "name": "patient_id_laterality_max",
        "group_cols": ["patient_id", "laterality"],
        "score_agg": "max",
        "label_agg": "max",
    },
]


def discover_runs() -> list[dict]:
    runs: list[dict] = []
    for base, kind, pred_name, manifest_name, split_col in [
        (
            RESULT_ROOT / "dst_results",
            "DST",
            "dst_all_predictions.csv",
            "dst_manifest.json",
            "dst_split",
        ),
        (
            RESULT_ROOT / "embedding_edl_results",
            "Embedding Proto-EDL",
            "proto_edl_all_predictions.csv",
            "proto_edl_manifest.json",
            "proto_edl_split",
        ),
    ]:
        for run_dir in sorted(base.glob("*5fold_loss_run*")):
            k_match = re.search(r"reg_k(\d+)", run_dir.name)
            k_value = int(k_match.group(1)) if k_match else math.nan
            runs.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": run_dir,
                    "model": kind,
                    "k": k_value,
                    "prediction_file": pred_name,
                    "manifest_file": manifest_name,
                    "split_col": split_col,
                }
            )
    return runs


def t_crit_975(df: int) -> float:
    lookup = {
        1: 12.706204736,
        2: 4.302652730,
        3: 3.182446305,
        4: 2.776445105,
        5: 2.570581836,
        6: 2.446911851,
        7: 2.364624252,
        8: 2.306004135,
        9: 2.262157163,
        10: 2.228138852,
    }
    return lookup.get(df, 1.959963984)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_stat = tpr - fpr
    finite = np.isfinite(thresholds)
    if finite.any():
        finite_idx = np.where(finite)[0]
        best = finite_idx[np.argmax(j_stat[finite_idx])]
    else:
        best = int(np.argmax(j_stat))
    return float(thresholds[best])


def threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if threshold is None:
        threshold = safe_youden_threshold(y_true, y_score)

    auc = safe_auc(y_true, y_score)
    if not np.isfinite(threshold):
        return {
            "n": int(y_true.size),
            "positives": int(y_true.sum()),
            "auc": auc,
            "youden_threshold": float("nan"),
            "bacc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "pred_pos": float("nan"),
        }

    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    bacc = (sensitivity + specificity) / 2
    return {
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "auc": auc,
        "youden_threshold": float(threshold),
        "bacc": float(bacc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "pred_pos": int(pred.sum()),
    }


def prepare_grain(df: pd.DataFrame, score_col: str, grain: dict) -> pd.DataFrame:
    cols = ["cancer", score_col] + grain["group_cols"]
    tmp = df[cols].copy()
    tmp = tmp.dropna(subset=["cancer", score_col])
    tmp["cancer"] = tmp["cancer"].astype(int)
    tmp[score_col] = tmp[score_col].astype(float)
    if not grain["group_cols"]:
        return tmp.rename(columns={score_col: "score"})[["cancer", "score"]]
    grouped = (
        tmp.groupby(grain["group_cols"], dropna=False)
        .agg(cancer=("cancer", "max"), score=(score_col, grain["score_agg"]))
        .reset_index(drop=True)
    )
    return grouped[["cancer", "score"]]


def read_prediction(path: Path, split_col: str) -> pd.DataFrame:
    needed = [
        "embedding_row",
        "source_row",
        "patient_id",
        "image_id",
        "laterality",
        "split",
        "cancer",
        split_col,
        "image_prediction_prob",
    ]
    return pd.read_csv(path, usecols=needed)


def fold_metrics_for_run(run: dict) -> list[dict]:
    rows: list[dict] = []
    for fold_dir in sorted(run["run_dir"].glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        df = read_prediction(fold_dir / run["prediction_file"], run["split_col"])
        for eval_split in ["train", "val", "test", "all"]:
            split_df = df if eval_split == "all" else df[df[run["split_col"]] == eval_split]
            for grain in GRAINS:
                prepared = prepare_grain(split_df, "image_prediction_prob", grain)
                metrics = threshold_metrics(
                    prepared["cancer"].to_numpy(), prepared["score"].to_numpy()
                )
                rows.append(
                    {
                        "run_name": run["run_name"],
                        "model": run["model"],
                        "k": run["k"],
                        "fold": fold,
                        "eval_split": eval_split,
                        "split_col": run["split_col"],
                        "grain": grain["name"],
                        "score_agg": grain["score_agg"],
                        "threshold_rule": "youden_on_eval_split",
                        **metrics,
                    }
                )
    return rows


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "n",
        "positives",
        "auc",
        "youden_threshold",
        "bacc",
        "sensitivity",
        "specificity",
        "pred_pos",
    ]
    records = []
    group_cols = ["run_name", "model", "k", "eval_split", "grain", "score_agg"]
    for keys, grp in fold_df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["fold_count"] = int(grp["fold"].nunique())
        for metric in metric_cols:
            vals = grp[metric].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            half = (
                t_crit_975(vals.size - 1) * std / math.sqrt(vals.size)
                if vals.size > 1 and np.isfinite(std)
                else float("nan")
            )
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
            rec[f"{metric}_ci95_low"] = mean - half if np.isfinite(half) else float("nan")
            rec[f"{metric}_ci95_high"] = mean + half if np.isfinite(half) else float("nan")
        records.append(rec)
    return pd.DataFrame(records)


def ensemble_predictions_for_run(run: dict) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    fold_score_cols = []
    for fold_dir in sorted(run["run_dir"].glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        df = read_prediction(fold_dir / run["prediction_file"], run["split_col"])
        df = df[df[run["split_col"]] == "test"].copy()
        score_col = f"score_fold_{fold}"
        fold_score_cols.append(score_col)
        cols = [
            "embedding_row",
            "source_row",
            "patient_id",
            "image_id",
            "laterality",
            "split",
            "cancer",
            run["split_col"],
            "image_prediction_prob",
        ]
        df = df[cols].rename(columns={"image_prediction_prob": score_col})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[
                    [
                        "source_row",
                        "patient_id",
                        "image_id",
                        "laterality",
                        "cancer",
                        score_col,
                    ]
                ],
                on=["source_row", "patient_id", "image_id", "laterality", "cancer"],
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise RuntimeError(f"No fold predictions found for {run['run_name']}")
    merged["ensemble_score_mean"] = merged[fold_score_cols].mean(axis=1)
    merged["ensemble_score_std"] = merged[fold_score_cols].std(axis=1, ddof=1)
    return merged


def bootstrap_ci(prepared: pd.DataFrame, rng: np.random.Generator) -> dict:
    y = prepared["cancer"].to_numpy().astype(int)
    s = prepared["score"].to_numpy().astype(float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    boot_rows = []
    for _ in range(BOOTSTRAP_N):
        sample_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        metrics = threshold_metrics(y[sample_idx], s[sample_idx])
        boot_rows.append(metrics)
    boot = pd.DataFrame(boot_rows)
    result = {}
    for metric in ["auc", "youden_threshold", "bacc", "sensitivity", "specificity", "pred_pos"]:
        vals = boot[metric].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        result[f"{metric}_bootstrap_mean"] = float(np.mean(vals))
        result[f"{metric}_bootstrap_std"] = float(np.std(vals, ddof=1))
        result[f"{metric}_ci95_low"] = float(np.percentile(vals, 2.5))
        result[f"{metric}_ci95_high"] = float(np.percentile(vals, 97.5))
    return result


def ensemble_metrics_for_run(run: dict, rng: np.random.Generator) -> tuple[list[dict], pd.DataFrame]:
    ensemble = ensemble_predictions_for_run(run)
    records = []
    for grain in GRAINS:
        prepared = prepare_grain(ensemble, "ensemble_score_mean", grain)
        point = threshold_metrics(prepared["cancer"].to_numpy(), prepared["score"].to_numpy())
        ci = bootstrap_ci(prepared, rng)
        records.append(
            {
                "run_name": run["run_name"],
                "model": run["model"],
                "k": run["k"],
                "eval_split": "test",
                "split_col": run["split_col"],
                "grain": grain["name"],
                "score_agg": grain["score_agg"],
                "threshold_rule": "youden_on_ensemble_test",
                "bootstrap_n": BOOTSTRAP_N,
                **point,
                **ci,
            }
        )
    return records, ensemble


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def extract_loss_history(run: dict) -> list[dict]:
    rows: list[dict] = []
    for fold_dir in sorted(run["run_dir"].glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        manifest_path = fold_dir / run["manifest_file"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_epoch = manifest.get("best_epoch")
        for item in manifest.get("history", []):
            rows.append(
                {
                    "run_name": run["run_name"],
                    "model": run["model"],
                    "k": run["k"],
                    "fold": fold,
                    "best_epoch": best_epoch,
                    **item,
                }
            )
    return rows


def plot_loss_curves(loss_df: pd.DataFrame) -> None:
    if loss_df.empty:
        return

    for run_name, grp in loss_df.groupby("run_name", sort=True):
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
        for fold, fold_grp in grp.groupby("fold"):
            fold_grp = fold_grp.sort_values("epoch")
            ax.plot(
                fold_grp["epoch"],
                fold_grp["train_loss"],
                color="#1f77b4",
                alpha=0.18,
                linewidth=1.0,
            )
            ax.plot(
                fold_grp["epoch"],
                fold_grp["eval_loss"],
                color="#d62728",
                alpha=0.18,
                linewidth=1.0,
            )
        summary = (
            grp.groupby("epoch")
            .agg(
                train_loss_mean=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                eval_loss_mean=("eval_loss", "mean"),
                eval_loss_std=("eval_loss", "std"),
            )
            .reset_index()
        )
        x = summary["epoch"].to_numpy()
        train_mean = summary["train_loss_mean"].to_numpy()
        train_std = summary["train_loss_std"].fillna(0).to_numpy()
        eval_mean = summary["eval_loss_mean"].to_numpy()
        eval_std = summary["eval_loss_std"].fillna(0).to_numpy()
        ax.plot(x, train_mean, color="#1f77b4", linewidth=2.2, label="train loss mean")
        ax.fill_between(x, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.10)
        ax.plot(x, eval_mean, color="#d62728", linewidth=2.2, label="val loss mean")
        ax.fill_between(x, eval_mean - eval_std, eval_mean + eval_std, color="#d62728", alpha=0.10)
        ax.set_title(run_name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"loss_curve_{safe_slug(run_name)}.png")
        plt.close(fig)

    run_names = list(loss_df.groupby("run_name").groups.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160, sharex=False, sharey=False)
    for ax, run_name in zip(axes.ravel(), run_names):
        grp = loss_df[loss_df["run_name"] == run_name]
        summary = (
            grp.groupby("epoch")
            .agg(train_loss=("train_loss", "mean"), eval_loss=("eval_loss", "mean"))
            .reset_index()
        )
        ax.plot(summary["epoch"], summary["train_loss"], color="#1f77b4", label="train")
        ax.plot(summary["epoch"], summary["eval_loss"], color="#d62728", label="val")
        ax.set_title(run_name, fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(run_names) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "loss_curves_all_runs.png")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    if not runs:
        raise RuntimeError(f"No runs found under {RESULT_ROOT}")

    fold_rows: list[dict] = []
    ensemble_rows: list[dict] = []
    loss_rows: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for run in runs:
        fold_rows.extend(fold_metrics_for_run(run))
        run_ensemble_rows, ensemble = ensemble_metrics_for_run(run, rng)
        ensemble_rows.extend(run_ensemble_rows)
        ensemble.to_csv(
            OUTPUT_DIR / f"ensemble_test_predictions_{safe_slug(run['run_name'])}.csv",
            index=False,
        )
        loss_rows.extend(extract_loss_history(run))

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_folds(fold_df)
    ensemble_df = pd.DataFrame(ensemble_rows)
    loss_df = pd.DataFrame(loss_rows)

    fold_df.to_csv(OUTPUT_DIR / "fold_youden_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "fold_youden_metrics_summary_mean_std_ci.csv", index=False)
    ensemble_df.to_csv(OUTPUT_DIR / "ensemble_youden_metrics_bootstrap_ci.csv", index=False)
    loss_df.to_csv(OUTPUT_DIR / "loss_history_long.csv", index=False)
    plot_loss_curves(loss_df)

    print(f"Wrote outputs to {OUTPUT_DIR}")
    print("Runs:")
    for run in runs:
        print(f"- {run['model']} k={run['k']}: {run['run_name']}")


if __name__ == "__main__":
    main()
