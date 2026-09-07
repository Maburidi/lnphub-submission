#!/usr/bin/env python3
"""Plot grouped-bar comparisons for multimodal random 5-fold CV metrics."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lnphub_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lnphub_oracles.cv_comparison import (
    EXPERIMENT_LABELS,
    EXPERIMENT_ORDER,
    build_summary_table,
    collect_cv_results,
    model_order,
)


EXPERIMENT_COLORS = {
    "all_layers": "#2f5f8f",
    "molecular_formulations": "#d18f2f",
    "molecular_only": "#4f8a5b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one grouped-bar figure per metric. The x-axis is model, "
            "and each model has bars for the three experiment types. LiON is "
            "shown only for the all-layers experiment by default."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("."),
        help="Repository/results root. Default: current working directory.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional summary CSV from collect_multimodal_cv_results.py. "
            "If omitted, metrics are collected directly from results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparisons/multimodal_random_5fold_cv/figures"),
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        help="Figure formats to write. Default: png svg.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metrics to plot. Default: every numeric metric except n.",
    )
    parser.add_argument(
        "--include-lion-non-all-layers",
        action="store_true",
        help="Plot LiON bars for molecular/formulation and molecular-only experiments too.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = results_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv:
        summary_path = args.summary_csv
        if not summary_path.is_absolute():
            summary_path = results_root / summary_path
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = build_summary_table(collect_cv_results(results_root))

    if not args.include_lion_non_all_layers:
        is_lion = summary_df["model"].str.startswith("LiON", na=False)
        is_not_all_layers = summary_df["experiment"].ne("all_layers")
        summary_df = summary_df.loc[~(is_lion & is_not_all_layers)].copy()

    available_metrics = [
        column[: -len("_mean")]
        for column in summary_df.columns
        if column.endswith("_mean") and column[: -len("_mean")] != "n"
    ]
    metrics = args.metrics or available_metrics
    missing_metrics = sorted(set(metrics).difference(available_metrics))
    if missing_metrics:
        raise ValueError(
            "Requested metrics are not available in the summary table: "
            + ", ".join(missing_metrics)
        )

    for metric in metrics:
        figure_paths = plot_metric(summary_df, metric, output_dir, args.formats)
        print(f"Wrote {metric}: " + ", ".join(str(path) for path in figure_paths))


def plot_metric(
    summary_df: pd.DataFrame, metric: str, output_dir: Path, formats: list[str]
) -> list[Path]:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    plot_df = summary_df[
        ["experiment", "experiment_label", "category", "model", mean_column, std_column]
    ].copy()
    plot_df = plot_df.dropna(subset=[mean_column])

    models = model_order(plot_df["model"].unique())
    experiments = [experiment for experiment in EXPERIMENT_ORDER if experiment in set(plot_df["experiment"])]

    x = np.arange(len(models))
    bar_width = min(0.23, 0.75 / max(len(experiments), 1))
    offsets = (np.arange(len(experiments)) - (len(experiments) - 1) / 2) * bar_width

    width = max(12, min(28, 0.52 * len(models) + 5))
    fig, ax = plt.subplots(figsize=(width, 6.5))

    for offset, experiment in zip(offsets, experiments):
        values = []
        errors = []
        for model in models:
            match = plot_df[
                plot_df["experiment"].eq(experiment) & plot_df["model"].eq(model)
            ]
            if match.empty:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(float(match[mean_column].iloc[0]))
                errors.append(float(match[std_column].fillna(0.0).iloc[0]))

        positions = x + offset
        valid = ~np.isnan(values)
        values_array = np.asarray(values, dtype=float)
        errors_array = np.asarray(errors, dtype=float)
        ax.bar(
            positions[valid],
            values_array[valid],
            width=bar_width * 0.92,
            yerr=errors_array[valid],
            capsize=3,
            label=EXPERIMENT_LABELS[experiment],
            color=EXPERIMENT_COLORS.get(experiment),
            edgecolor="#202020",
            linewidth=0.45,
        )

    ax.set_title(_metric_title(metric), fontsize=14, pad=14)
    ax.set_ylabel(_metric_title(metric))
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Experiment", frameon=False, ncol=min(3, len(experiments)))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    _set_metric_axis_limits(ax, metric, plot_df[mean_column], plot_df[std_column])
    fig.tight_layout()

    figure_paths = []
    for extension in formats:
        figure_path = output_dir / f"{_slugify(metric)}.{extension.lstrip('.')}"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        figure_paths.append(figure_path)
    plt.close(fig)
    return figure_paths


def _metric_title(metric: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "mcc": "MCC",
        "roc_auc_ovr_macro": "ROC AUC OVR Macro",
        "average_precision_macro": "Average Precision Macro",
        "log_loss": "Log Loss",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def _set_metric_axis_limits(
    ax: plt.Axes, metric: str, values: pd.Series, errors: pd.Series
) -> None:
    finite_values = pd.to_numeric(values, errors="coerce").dropna()
    finite_errors = pd.to_numeric(errors, errors="coerce").fillna(0.0)
    if finite_values.empty:
        return
    lower = float((finite_values - finite_errors.reindex(finite_values.index, fill_value=0.0)).min())
    upper = float((finite_values + finite_errors.reindex(finite_values.index, fill_value=0.0)).max())

    if metric != "mcc" and lower >= 0 and upper <= 1:
        ax.set_ylim(0, min(1.0, max(upper * 1.12, 0.05)))
    elif metric == "mcc":
        pad = max(0.05, (upper - lower) * 0.12)
        ax.set_ylim(max(-1.0, lower - pad), min(1.0, upper + pad))
    else:
        pad = max(0.05, (upper - lower) * 0.12)
        ax.set_ylim(max(0.0, lower - pad), upper + pad)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


if __name__ == "__main__":
    main()
