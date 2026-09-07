"""Utilities for comparing multimodal 5-fold CV result folders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENT_ROOTS = {
    "all_layers": Path("results/multimodal_all/random_5fold_CV"),
    "molecular_formulations": Path(
        "results/multimodal_molecular_formulations/random_5fold_CV"
    ),
    "molecular_only": Path("results/multimodal_molecular_only/random_5fold_CV"),
}

EXPERIMENT_LABELS = {
    "all_layers": "All layers",
    "molecular_formulations": "Molecular + formulations",
    "molecular_only": "Molecular only",
}

EXPERIMENT_ORDER = list(EXPERIMENT_ROOTS)
TEST_SPLITS = {"test", "untouched_test"}

ID_COLUMNS = {
    "experiment",
    "experiment_label",
    "category",
    "model",
    "method",
    "source_path",
    "split",
    "fold",
    "encoder",
    "validation_study",
}


def discover_metrics_files(results_root: Path) -> list[Path]:
    """Find per-model metrics.csv files in the three target 5-fold CV folders."""
    metrics_files: list[Path] = []
    for experiment, relative_root in EXPERIMENT_ROOTS.items():
        root = results_root / relative_root
        if not root.exists():
            raise FileNotFoundError(f"Missing expected experiment root: {root}")
        for metrics_path in root.rglob("metrics.csv"):
            if metrics_path.is_file():
                metrics_files.append(metrics_path)
    return sorted(metrics_files)


def collect_cv_results(results_root: Path) -> pd.DataFrame:
    """Read all fold, mean, and std rows from target multimodal 5-fold CV results."""
    rows = []
    for metrics_path in discover_metrics_files(results_root):
        experiment = _experiment_from_path(results_root, metrics_path)
        df = pd.read_csv(metrics_path)

        if "split" not in df.columns:
            raise ValueError(f"{metrics_path} is missing a split column")

        sub_experiment = _sub_experiment_from_path(results_root, metrics_path, experiment)
        model_dir = metrics_path.parent.name
        fallback_method = _model_label_from_dir(model_dir)

        df = df.copy()
        df.insert(0, "source_path", str(metrics_path.relative_to(results_root)))
        df.insert(0, "model", _canonical_model_name(fallback_method))
        df.insert(0, "category", _category_from_sub_experiment(sub_experiment))
        df.insert(0, "experiment_label", EXPERIMENT_LABELS[experiment])
        df.insert(0, "experiment", experiment)

        if "method" not in df.columns:
            df["method"] = fallback_method
        else:
            df["method"] = df["method"].fillna(fallback_method)
            df["model"] = df["method"].map(_canonical_model_name)

        rows.append(df)

    if not rows:
        raise FileNotFoundError("No per-model metrics.csv files were found")

    combined = pd.concat(rows, ignore_index=True, sort=False)
    return _sort_results(combined)


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric metric columns, excluding identifier columns."""
    metrics = []
    for column in df.columns:
        if column in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            metrics.append(column)
    return metrics


def build_summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Create one mean/std row per experiment, category, and model."""
    metrics = metric_columns(long_df)
    mean_rows = long_df[long_df["split"].eq("cv_mean")]
    std_rows = long_df[long_df["split"].eq("cv_std")]

    keys = ["experiment", "experiment_label", "category", "model", "method", "source_path"]
    summary = mean_rows[keys].copy()
    for metric in metrics:
        summary[f"{metric}_mean"] = mean_rows[metric].to_numpy()

    std_lookup = std_rows.set_index(keys)
    for metric in metrics:
        values = []
        for key in summary[keys].itertuples(index=False, name=None):
            if key in std_lookup.index:
                values.append(std_lookup.loc[key, metric])
            else:
                values.append(np.nan)
        summary[f"{metric}_std"] = values

    return _sort_results(summary)


def build_test_results_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Return one table containing all held-out test metric rows."""
    test_rows = long_df[long_df["split"].isin(TEST_SPLITS)].copy()
    test_rows.insert(test_rows.columns.get_loc("split"), "original_split", test_rows["split"])
    test_rows["split"] = "test"
    return _sort_results(test_rows)


def build_formatted_summary_table(summary_df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """Create a compact table with each metric formatted as mean +/- std."""
    keys = ["experiment_label", "category", "model"]
    formatted = summary_df[keys].copy()

    mean_columns = [column for column in summary_df.columns if column.endswith("_mean")]
    for mean_column in mean_columns:
        metric = mean_column[: -len("_mean")]
        std_column = f"{metric}_std"
        if std_column not in summary_df.columns:
            continue
        formatted[metric] = [
            _format_mean_std(mean, std, precision)
            for mean, std in zip(summary_df[mean_column], summary_df[std_column])
        ]

    return formatted


def model_order(models: Iterable[str]) -> list[str]:
    preferred = [
        "RDKit Descriptors + Random Forest",
        "RDKit Descriptors + XGBoost",
        "RDKit Descriptors + MLP",
        "ECFP4 + Random Forest",
        "ECFP4 + XGBoost",
        "ECFP4 + MLP",
        "AGILE-Frozen",
        "ChemBERTa-MTR-Frozen",
        "LiON-Frozen",
        "MoLFormer-Frozen",
        "MolCLR-GIN-Frozen",
        "Uni-Mol-Frozen",
        "AGILE-Finetuned",
        "ChemBERTa-MTR-Finetuned",
        "LiON-Finetuned",
        "MoLFormer-Finetuned",
        "MolCLR-GIN-Finetuned",
        "Uni-Mol-Finetuned",
        "D-MPNN",
        "GCN",
        "GIN",
    ]
    model_set = set(models)
    ordered = [model for model in preferred if model in model_set]
    ordered.extend(sorted(model_set.difference(ordered)))
    return ordered


def _experiment_from_path(results_root: Path, metrics_path: Path) -> str:
    relative_path = metrics_path.relative_to(results_root)
    for experiment, root in EXPERIMENT_ROOTS.items():
        if relative_path.is_relative_to(root):
            return experiment
    raise ValueError(f"Could not map metrics file to a target experiment: {metrics_path}")


def _sub_experiment_from_path(
    results_root: Path, metrics_path: Path, experiment: str
) -> str:
    experiment_root = results_root / EXPERIMENT_ROOTS[experiment]
    relative_path = metrics_path.relative_to(experiment_root)
    if len(relative_path.parts) < 2:
        raise ValueError(f"Unexpected metrics file location: {metrics_path}")
    return relative_path.parts[0]


def _category_from_sub_experiment(sub_experiment: str) -> str:
    lower = sub_experiment.lower()
    if "finetuned" in lower:
        return "Fine-tuned encoder"
    if "frozen" in lower:
        return "Frozen embedding"
    if "gnn" in lower:
        return "GNN"
    if "fingerprint" in lower or "descriptor" in lower:
        return "Descriptor/fingerprint"
    return sub_experiment.replace("_", " ")


def _canonical_model_name(method: str) -> str:
    model = str(method)
    replacements = {
        "Formulation/PhysChem": "",
        "Formulation + PhysChem": "",
        "Formulation_PhysChem": "",
        "Formulation": "",
        "RDKit_Descriptors": "RDKit Descriptors",
        "Random_Forest": "Random Forest",
        "ChemBERTa_MTR": "ChemBERTa-MTR",
        "MolCLR_GIN": "MolCLR-GIN",
        "Uni_Mol": "Uni-Mol",
        "D_MPNN": "D-MPNN",
        "_Frozen": "-Frozen",
        "_Finetuned": "-Finetuned",
    }
    for old, new in replacements.items():
        model = model.replace(old, new)
    model = model.replace("_plus_", " + ")
    model = model.replace("_", " ")
    model = re.sub(r"\s*\+\s*\+\s*", " + ", model)
    model = re.sub(r"\s+\+\s+(Random Forest|XGBoost|MLP)$", r" + \1", model)
    model = re.sub(r"\s+", " ", model).strip(" +")
    return model


def _model_label_from_dir(model_dir: str) -> str:
    return _canonical_model_name(model_dir)


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _sort_results(df: pd.DataFrame) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_experiment_order"] = sortable["experiment"].map(
        {experiment: index for index, experiment in enumerate(EXPERIMENT_ORDER)}
    )
    sortable["_model_order"] = sortable["model"].map(
        {model: index for index, model in enumerate(model_order(sortable["model"].dropna()))}
    )
    sort_columns = [
        column
        for column in ["_experiment_order", "category", "_model_order", "model", "split"]
        if column in sortable.columns
    ]
    sortable = sortable.sort_values(sort_columns).drop(
        columns=["_experiment_order", "_model_order"]
    )
    return sortable.reset_index(drop=True)
