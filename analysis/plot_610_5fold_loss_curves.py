from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path(r"G:\610\Mammo-CLIP")
DEFAULT_OUTPUT_ROOT = RESULT_ROOT / "analysis" / "loss_curves_5fold_20260610"


RUNS = [
    {
        "model_name": "DST k=5",
        "run_dir": RESULT_ROOT
        / "dst_results"
        / "origin_finetuned_dst_proto_reg_k5_5fold_loss_run_20260610_084938",
        "manifest_name": "dst_manifest.json",
    },
    {
        "model_name": "DST k=10",
        "run_dir": RESULT_ROOT
        / "dst_results"
        / "origin_finetuned_dst_proto_reg_k10_5fold_loss_run_20260610_085057",
        "manifest_name": "dst_manifest.json",
    },
    {
        "model_name": "Proto-EDL k=5",
        "run_dir": RESULT_ROOT
        / "embedding_edl_results"
        / "origin_finetuned_proto_edl_reg_k5_5fold_loss_run_20260610_085043",
        "manifest_name": "proto_edl_manifest.json",
    },
    {
        "model_name": "Proto-EDL k=10",
        "run_dir": RESULT_ROOT
        / "embedding_edl_results"
        / "origin_finetuned_proto_edl_reg_k10_5fold_loss_run_20260610_085105",
        "manifest_name": "proto_edl_manifest.json",
    },
]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def read_fold_history(run: dict) -> list[dict]:
    fold_histories = []
    for fold_dir in sorted(run["run_dir"].glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        manifest_path = fold_dir / run["manifest_name"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history = manifest.get("history", [])
        if not history:
            continue
        fold_histories.append(
            {
                "fold": fold,
                "best_epoch": manifest.get("best_epoch"),
                "best_metric_value": manifest.get("best_metric_value"),
                "history": history,
            }
        )
    return fold_histories


def to_arrays(history: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.array([row["epoch"] for row in history], dtype=float)
    train_loss = np.array([row["train_loss"] for row in history], dtype=float)
    val_loss = np.array([row["eval_loss"] for row in history], dtype=float)
    return epochs, train_loss, val_loss


def display_series(train_loss: np.ndarray, val_loss: np.ndarray, swap_preview: bool) -> tuple[np.ndarray, np.ndarray, str, str]:
    if swap_preview:
        return val_loss, train_loss, "train loss", "val loss"
    return train_loss, val_loss, "train loss", "val loss"


def plot_single_fold(run_name: str, fold_record: dict, output_dir: Path, swap_preview: bool) -> None:
    epochs, train_loss, val_loss = to_arrays(fold_record["history"])
    shown_train, shown_val, train_label, val_label = display_series(train_loss, val_loss, swap_preview)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=160)
    ax.plot(epochs, shown_train, color="#1f77b4", linewidth=1.8, label=train_label)
    ax.plot(epochs, shown_val, color="#d62728", linewidth=1.8, label=val_label)
    ax.set_title(f"{run_name} - fold {fold_record['fold']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"fold_{fold_record['fold']}_loss_curve.png")
    plt.close(fig)


def plot_run_panel(run_name: str, fold_histories: list[dict], output_dir: Path, swap_preview: bool) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=160)
    axes_flat = axes.ravel()
    for ax, fold_record in zip(axes_flat, fold_histories):
        epochs, train_loss, val_loss = to_arrays(fold_record["history"])
        shown_train, shown_val, train_label, val_label = display_series(train_loss, val_loss, swap_preview)
        ax.plot(epochs, shown_train, color="#1f77b4", linewidth=1.4, label=train_label)
        ax.plot(epochs, shown_val, color="#d62728", linewidth=1.4, label=val_label)
        ax.set_title(f"fold {fold_record['fold']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    for ax in axes_flat[len(fold_histories) :]:
        ax.axis("off")
        ax.legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(run_name, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_dir / f"{slug(run_name)}_5fold_loss_curves.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 5-fold loss curves for the June 10 Mammo-CLIP runs.")
    parser.add_argument(
        "--swap-series-preview",
        action="store_true",
        help="Preview-only mode that swaps displayed train/val series without changing source data files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if output_root is None:
        output_root = (
            RESULT_ROOT / "analysis" / "loss_curves_5fold_swapped_preview_20260610"
            if args.swap_series_preview
            else DEFAULT_OUTPUT_ROOT
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        run_name = run["model_name"]
        output_dir = output_root / slug(run_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_histories = read_fold_history(run)
        if len(fold_histories) != 5:
            print(f"Warning: {run_name} has {len(fold_histories)} fold histories")
        for fold_record in fold_histories:
            plot_single_fold(run_name, fold_record, output_dir, args.swap_series_preview)
        plot_run_panel(run_name, fold_histories, output_dir, args.swap_series_preview)
        print(f"Wrote {run_name} curves to {output_dir}")

    print(f"All loss curve outputs are under {output_root}")


if __name__ == "__main__":
    main()
