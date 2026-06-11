from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


RESULT_DIR = Path(r"G:\610\Mammo-CLIP\analysis\youden_ensemble_20260610")
OUTPUT_DIR = Path(r"G:\610\Mammo-CLIP\analysis\ppt_metric_tables_20260611")

ENSEMBLE_CSV = RESULT_DIR / "ensemble_youden_metrics_bootstrap_ci.csv"
FOLD_SUMMARY_CSV = RESULT_DIR / "fold_youden_metrics_summary_mean_std_ci.csv"

MODEL_ORDER = [
    ("DST", 5, "DST k=5"),
    ("DST", 10, "DST k=10"),
    ("Embedding Proto-EDL", 5, "Proto-EDL k=5"),
    ("Embedding Proto-EDL", 10, "Proto-EDL k=10"),
]

GRAINS = [
    ("image", "Image-Level Test Metrics"),
    ("patient_id_mean", "Patient-Level Test Metrics (patient_id mean)"),
    ("patient_id_laterality_max", "Patient-Level Test Metrics (patient_id + laterality max)"),
]

METRICS = [
    ("auc", "AUC"),
    ("bacc", "bACC"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
]


def fmt(x: float) -> str:
    return f"{x:.3f}"


def build_table_dataframe() -> pd.DataFrame:
    ensemble = pd.read_csv(ENSEMBLE_CSV)
    fold = pd.read_csv(FOLD_SUMMARY_CSV)
    fold = fold[fold["eval_split"] == "test"].copy()

    merged = ensemble.merge(
        fold,
        on=["model", "k", "grain"],
        how="inner",
        suffixes=("_ens", "_fold"),
        validate="one_to_one",
    )
    label_map = {(model, k): label for model, k, label in MODEL_ORDER}
    merged["model_label"] = merged.apply(lambda r: label_map[(r["model"], r["k"])], axis=1)
    return merged


def cell_text(row: pd.Series, metric: str) -> str:
    return (
        f"Ens {fmt(row[metric])}\n"
        f"Mean+/-SD {fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_std'])}\n"
        f"95% CI [{fmt(row[f'{metric}_ci95_low_fold'])}, {fmt(row[f'{metric}_ci95_high_fold'])}]"
    )


def export_flat_csv(grain_df: pd.DataFrame, grain: str) -> Path:
    rows = []
    for model, k, label in MODEL_ORDER:
        row = grain_df[(grain_df["model"] == model) & (grain_df["k"] == k)].iloc[0]
        record = {"Model": label}
        for metric, header in METRICS:
            record[f"{header}_ensemble"] = row[metric]
            record[f"{header}_mean"] = row[f"{metric}_mean"]
            record[f"{header}_std"] = row[f"{metric}_std"]
            record[f"{header}_ci95_low"] = row[f"{metric}_ci95_low_fold"]
            record[f"{header}_ci95_high"] = row[f"{metric}_ci95_high_fold"]
        rows.append(record)
    out_path = OUTPUT_DIR / f"ppt_table_{grain}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def draw_table_image(grain_df: pd.DataFrame, grain: str, title: str) -> Path:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_color = "#10233f"
    note_color = "#5b6472"
    border = "#d6dbe4"
    header_fill = "#163a63"
    header_text = "white"
    row_alt = "#f6f8fb"
    row_base = "white"

    ax.text(0.05, 0.945, title, fontsize=24, fontweight="bold", color=title_color, va="top")
    ax.text(
        0.05,
        0.905,
        "Each cell shows: Ensemble point estimate | 5-fold mean+/-std | 5-fold 95% CI",
        fontsize=11.5,
        color=note_color,
        va="top",
    )
    ax.text(
        0.05,
        0.88,
        "bACC, Sensitivity, and Specificity are computed at the Youden-optimal threshold; AUC is threshold-free.",
        fontsize=11.5,
        color=note_color,
        va="top",
    )

    left = 0.05
    top = 0.83
    table_width = 0.90
    header_h = 0.095
    row_h = 0.155
    model_w = 0.17
    metric_w = (table_width - model_w) / 4
    col_widths = [model_w] + [metric_w] * 4
    headers = ["Model"] + [name for _, name in METRICS]

    x = left
    for width, header in zip(col_widths, headers):
        ax.add_patch(
            Rectangle((x, top - header_h), width, header_h, facecolor=header_fill, edgecolor=border, linewidth=1.2)
        )
        ax.text(
            x + width / 2,
            top - header_h / 2,
            header,
            ha="center",
            va="center",
            color=header_text,
            fontsize=13.5,
            fontweight="bold",
        )
        x += width

    for row_idx, (_, _, label) in enumerate(MODEL_ORDER):
        y_top = top - header_h - row_idx * row_h
        fill = row_alt if row_idx % 2 else row_base
        x = left
        row = grain_df[grain_df["model_label"] == label].iloc[0]
        values = [label] + [cell_text(row, metric) for metric, _ in METRICS]
        for col_idx, (width, value) in enumerate(zip(col_widths, values)):
            ax.add_patch(Rectangle((x, y_top - row_h), width, row_h, facecolor=fill, edgecolor=border, linewidth=1.0))
            if col_idx == 0:
                ax.text(
                    x + 0.015,
                    y_top - row_h / 2,
                    value,
                    ha="left",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="#142033",
                )
            else:
                ax.text(
                    x + width / 2,
                    y_top - row_h / 2,
                    value,
                    ha="center",
                    va="center",
                    fontsize=11.2,
                    color="#142033",
                    linespacing=1.45,
                )
            x += width

    ax.text(
        0.05,
        0.055,
        "Source: G:/610/Mammo-CLIP/analysis/youden_ensemble_20260610",
        fontsize=10.5,
        color=note_color,
        va="center",
    )

    out_path = OUTPUT_DIR / f"ppt_table_{grain}.png"
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_table_dataframe()
    for grain, title in GRAINS:
        grain_df = merged[merged["grain"] == grain].copy()
        export_flat_csv(grain_df, grain)
        draw_table_image(grain_df, grain, title)
    print(f"Wrote PPT table assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
