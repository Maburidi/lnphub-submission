#!/usr/bin/env python3
"""Generate LNPHub manuscript and supplementary figures from the final CSV.

Usage:
    python generate_lnphub_figures.py path/to/final_lnphub.csv

The script treats the input CSV as read-only, derives all numbers from the file,
and writes figures, source data, logs, and summary tables under figures_lnphub/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import platform
import re
import sys
import textwrap
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    RDLogger = None
    rdFingerprintGenerator = None

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None


RANDOM_SEED = 42
DPI = 700

BASE = Path(__file__).resolve().parents[1]
DIRS = {
    "main": BASE / "main",
    "supplementary": BASE / "supplementary",
    "panels": BASE / "panels",
    "tables": BASE / "tables",
    "source": BASE / "source_data",
    "scripts": BASE / "scripts",
}

COLORS = {
    "ink": "#1f2933",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "blue": "#2474A6",
    "teal": "#1B9E77",
    "orange": "#D95F02",
    "purple": "#7570B3",
    "rose": "#C44E52",
    "gold": "#E6AB02",
    "green": "#2A9D55",
    "gray": "#8a8f98",
}

FAMILY_COLORS = OrderedDict(
    [
        ("Physicochemical", "#2474A6"),
        ("Delivery", "#1B9E77"),
        ("Viability", "#E6AB02"),
        ("Toxicity", "#D95F02"),
        ("Editing", "#7570B3"),
        ("Uptake", "#66A61E"),
        ("Immune", "#C44E52"),
        ("Other", "#8A8F98"),
    ]
)

DIMENSION_BLOCKS = OrderedDict(
    [
        (
            "Ionizable lipid\nstructure",
            [
                "il_id",
                "il_smiles",
                "il_class",
                "reaction_type",
                "il_architecture_type",
                "il_head_smiles",
                "il_head_class",
                "il_head_architecture",
                "il_linker_smiles",
                "il_linker_class",
                "il_tail_count",
                "il_tail1_smiles",
                "il_tail1_class",
                "il_tail2_smiles",
                "il_tail2_class",
                "il_tail3_smiles",
                "il_tail3_class",
                "heavy_atoms",
                "rings",
                "aromatic_rings",
                "rotatable_bonds",
                "van_der_waals_molecular_volume",
                "topological_polar_surface_area",
                "hydrogen_bond_donors",
                "hydrogen_bond_acceptors",
                "logp",
                "molar_refractivity",
                "fraction_sp3_carbons",
                "molecular_weight",
                "has_ester",
                "has_carbonate",
                "has_disulfide",
            ],
        ),
        (
            "Formulation",
            [
                "helper_lipid_name",
                "helper_lipid_smiles",
                "helper_lipid_mol_ratio",
                "cholesterol_name",
                "cholesterol_smiles",
                "cholesterol_mol_ratio",
                "peg_lipid_name",
                "peg_lipid_smiles",
                "peg_lipid_mol_ratio",
                "il_mol_ratio",
                "il_to_nucleic_acid_mass_ratio",
                "il_to_nucleic_acid_charge_ratio",
                "total_lipid_to_cargo_ratio",
                "total_flow_rate_ml_min",
            ],
        ),
        (
            "Physicochemical",
            [
                "particle_size_nm",
                "pdi",
                "zeta_potential_mv",
                "zeta_potential_metric",
                "encapsulation_efficiency_percent",
                "measured_pka",
            ],
        ),
        ("Delivery", ["delivery", "delivery_metric"]),
        (
            "Viability",
            [
                "viability_igrov1_percent",
                "viability_mtt_0_5_ug_per_ml",
                "viability_mtt_5_ug_per_ml",
            ],
        ),
        (
            "Toxicity",
            [
                "toxicity_cera_umol_per_l",
                "toxicity_hemolysis_percent",
                "toxicity_urea_umol_per_l",
                "toxicity_alt_u_per_l",
                "toxicity_ast_u_per_l",
                "toxicity_bun_mg_per_dl",
            ],
        ),
        (
            "Editing",
            [
                "editing_gfp_negative_cells_percent",
                "editing_relative_fvii_ec50_mg_per_kg",
                "base_editing_percent",
            ],
        ),
        ("Uptake", ["cellular_metabolic_uptake_percent", "uptake_cy5_mluc_lnps_percent"]),
        (
            "Immune",
            [
                "immune_tnf_alpha_pg_ml",
                "immune_il1a_pg_ml",
                "immune_il6_pg_ml",
                "immune_ifn_gamma_pg_ml",
                "immune_gm_csf_pg_ml",
            ],
        ),
        (
            "Protein / other\noutcomes",
            ["protein_abundance_value", "mrna_binding_efficiency_percent", "gfp_mfi_transfection"],
        ),
    ]
)

ENDPOINT_FAMILIES = OrderedDict(
    [
        (
            "Physicochemical",
            [
                "particle_size_nm",
                "pdi",
                "zeta_potential_mv",
                "encapsulation_efficiency_percent",
                "measured_pka",
            ],
        ),
        ("Delivery", ["delivery"]),
        (
            "Viability",
            [
                "viability_igrov1_percent",
                "viability_mtt_0_5_ug_per_ml",
                "viability_mtt_5_ug_per_ml",
            ],
        ),
        (
            "Toxicity",
            [
                "toxicity_cera_umol_per_l",
                "toxicity_hemolysis_percent",
                "toxicity_urea_umol_per_l",
                "toxicity_alt_u_per_l",
                "toxicity_ast_u_per_l",
                "toxicity_bun_mg_per_dl",
            ],
        ),
        (
            "Editing",
            [
                "editing_gfp_negative_cells_percent",
                "editing_relative_fvii_ec50_mg_per_kg",
                "base_editing_percent",
            ],
        ),
        ("Uptake", ["cellular_metabolic_uptake_percent", "uptake_cy5_mluc_lnps_percent"]),
        (
            "Immune",
            [
                "immune_tnf_alpha_pg_ml",
                "immune_il1a_pg_ml",
                "immune_il6_pg_ml",
                "immune_ifn_gamma_pg_ml",
                "immune_gm_csf_pg_ml",
            ],
        ),
        ("Other", ["protein_abundance_value", "mrna_binding_efficiency_percent", "gfp_mfi_transfection"]),
    ]
)

FORMULATION_RATIO_COLS = [
    "il_mol_ratio",
    "helper_lipid_mol_ratio",
    "cholesterol_mol_ratio",
    "peg_lipid_mol_ratio",
    "il_to_nucleic_acid_mass_ratio",
    "il_to_nucleic_acid_charge_ratio",
    "total_lipid_to_cargo_ratio",
]

TAXONOMY_COLS = [
    "il_class",
    "reaction_type",
    "il_architecture_type",
    "il_head_class",
    "il_linker_class",
    "il_tail_count",
]


def mkdirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def fmt_int(x: float | int) -> str:
    if pd.isna(x):
        return "NA"
    return f"{int(round(float(x))):,}"


def clean_label(x) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return "Not reported"
    return str(x).replace("_", " ")


def save_source(df: pd.DataFrame, name: str) -> Path:
    path = DIRS["source"] / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_fig(fig: plt.Figure, base_name: str, folder: Path) -> list[Path]:
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = folder / f"{base_name}.{ext}"
        fig.savefig(path, dpi=DPI if ext == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")


def present_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def non_missing_mask(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.notna() & s.astype(str).str.strip().ne("")
    return s.notna()


def row_has_any(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = present_cols(df, cols)
    if not cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | non_missing_mask(df[c])
    return mask


def first_existing(df: pd.DataFrame, options: Iterable[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def extract_year(study_id, publication_link=None) -> float:
    text = f"{study_id or ''} {publication_link or ''}"
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", text)]
    years = [y for y in years if 1990 <= y <= dt.date.today().year]
    return float(years[-1]) if years else np.nan


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def endpoint_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for family_cols in ENDPOINT_FAMILIES.values():
        cols.extend(present_cols(df, family_cols))
    return cols


def audit_dataset(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    endpoint_cols = endpoint_columns(df)
    audit = {
        "total_records": len(df),
        "unique_study_id": df["study_id"].nunique(dropna=True),
        "unique_publication_link": df["publication_link"].nunique(dropna=True),
        "unique_lnp_id": df["lnp_id"].nunique(dropna=True),
        "unique_il_id": df["il_id"].nunique(dropna=True),
        "unique_il_smiles": df["il_smiles"].nunique(dropna=True),
        "number_of_columns": len(df.columns),
    }
    rows = [{"metric": k, "value": v} for k, v in audit.items()]
    for col in ["experimental_setting", "cargo_type", first_existing(df, ["delivery_metric", "delivery_value_method"])]:
        if col and col in df.columns:
            for category, count in df[col].fillna("Not reported").value_counts(dropna=False).items():
                rows.append({"metric": f"{col}:{category}", "value": int(count)})
    for col in endpoint_cols:
        rows.append({"metric": f"non_null:{col}", "value": int(non_missing_mask(df[col]).sum())})
    audit_long = pd.DataFrame(rows)

    completeness = []
    for col in df.columns:
        nn = int(non_missing_mask(df[col]).sum())
        completeness.append(
            {
                "column": col,
                "non_null_count": nn,
                "missing_count": int(len(df) - nn),
                "percent_complete": nn / len(df) * 100 if len(df) else 0,
                "studies_with_value": int(df.loc[non_missing_mask(df[col]), "study_id"].nunique()) if "study_id" in df else np.nan,
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )
    completeness_df = pd.DataFrame(completeness)
    save_source(audit_long, "dataset_audit")
    completeness_df.to_csv(DIRS["tables"] / "field_completeness_table.csv", index=False)
    return audit, audit_long, completeness_df


def write_audit_and_mapping(df: pd.DataFrame, audit: dict) -> None:
    mapping_rows = []
    panel_map = OrderedDict(
        [
            ("Fig2a_dataset_scale", ["study_id", "publication_link", "record_id", "lnp_id", "il_id", "il_smiles", "biological_model", "biological_target", "cargo_type", "experimental_setting"]),
            ("Fig2b_study_dimension_heatmap", sum([v for v in DIMENSION_BLOCKS.values()], [])),
            ("Fig2c_experimental_landscape", ["experimental_setting", "cargo_type", "study_id", "lnp_id"]),
            ("Fig2d_biological_landscape", ["biological_model", "biological_target", "route_of_administration", "experimental_setting", "study_id", "lnp_id"]),
            ("Fig2e_chemical_space_il_class", ["il_smiles", "il_class", "study_id", "reaction_type", "il_architecture_type", "il_tail_count"]),
            ("Fig2f_structural_diversity_alluvial", ["il_smiles", "reaction_type", "il_architecture_type", "il_tail_count"]),
            ("Fig2g_outcome_landscape", endpoint_columns(df) + ["study_id", "lnp_id"]),
        ]
    )
    for panel, cols in panel_map.items():
        for col in cols:
            mapping_rows.append({"figure_or_panel": panel, "column": col, "available": col in df.columns})
    mapping = pd.DataFrame(mapping_rows).drop_duplicates()
    mapping.to_csv(DIRS["tables"] / "proposed_figure_data_mappings.csv", index=False)

    lines = [
        "LNPHub dataset audit and proposed figure/data mappings",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Dataset audit",
    ]
    lines.extend([f"- {k}: {fmt_int(v)}" for k, v in audit.items()])
    lines.append("")
    lines.append("Detected value columns")
    lines.append(f"- delivery value: {first_existing(df, ['delivery_value', 'delivery'])}")
    lines.append(f"- delivery metric/method: {first_existing(df, ['delivery_value_method', 'delivery_metric'])}")
    lines.append(f"- zeta metadata: {first_existing(df, ['zeta_potential_method', 'zeta_potential_metric'])}")
    lines.append("")
    lines.append("Dimension blocks")
    for block, cols in DIMENSION_BLOCKS.items():
        available = present_cols(df, cols)
        lines.append(f"- {block.replace(chr(10), ' ')}: {len(available)} available columns")
    (DIRS["tables"] / "dataset_audit_and_mappings.txt").write_text("\n".join(lines) + "\n")


def study_dimension_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study, sdf in df.groupby("study_id", dropna=False):
        row = {"study_id": study, "records": len(sdf)}
        for block, cols in DIMENSION_BLOCKS.items():
            cols = present_cols(df, cols)
            row[block] = float(row_has_any(sdf, cols).mean() * 100) if cols else np.nan
        rows.append(row)
    matrix = pd.DataFrame(rows).sort_values("records", ascending=False)
    save_source(matrix, "Fig2b_study_dimension_heatmap_matrix")
    return matrix


def field_coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = endpoint_columns(df)
    rows = []
    for study, sdf in df.groupby("study_id", dropna=False):
        row = {"study_id": study, "records": len(sdf)}
        for col in cols:
            row[col] = float(non_missing_mask(sdf[col]).mean() * 100)
        rows.append(row)
    mat = pd.DataFrame(rows).sort_values("records", ascending=False)
    save_source(mat, "SuppFig2_field_level_coverage_matrix")
    return mat


def categorical_summary(df: pd.DataFrame, col: str, source_name: str, include_missing: bool = True) -> pd.DataFrame:
    vals = df[col].fillna("Not reported") if include_missing else df[col].dropna()
    work = df.copy()
    work[col] = vals
    if not include_missing:
        work = work.loc[work[col].notna()]
    out = (
        work.groupby(col, dropna=False)
        .agg(records=("record_id", "count"), studies=("study_id", pd.Series.nunique), unique_lnps=("lnp_id", pd.Series.nunique), unique_il_smiles=("il_smiles", pd.Series.nunique))
        .reset_index()
        .rename(columns={col: "category"})
        .sort_values("records", ascending=False)
    )
    out["category"] = out["category"].map(clean_label)
    save_source(out, source_name)
    return out


def top_n_with_other(summary: pd.DataFrame, n: int, label_col: str = "category") -> pd.DataFrame:
    if len(summary) <= n:
        return summary.copy()
    top = summary.head(n).copy()
    rest = summary.iloc[n:]
    numeric = rest.select_dtypes(include=[np.number]).sum().to_dict()
    other = {label_col: "Other"}
    other.update(numeric)
    return pd.concat([top, pd.DataFrame([other])], ignore_index=True)


def draw_stat_infographic(ax: plt.Axes, stats: OrderedDict[str, int]) -> None:
    ax.axis("off")
    positions = [
        (0.05, 0.63, 0.42, 0.30, 24),
        (0.53, 0.63, 0.42, 0.30, 24),
        (0.05, 0.29, 0.42, 0.26, 20),
        (0.53, 0.29, 0.42, 0.26, 20),
    ]
    dominant = list(stats.items())[:4]
    for (label, value), (x, y, w, h, fs) in zip(dominant, positions):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", linewidth=0.8, edgecolor="#d8dee9", facecolor="#f8fafc")
        ax.add_patch(box)
        ax.text(x + 0.03, y + h * 0.58, fmt_int(value), fontsize=fs, fontweight="bold", color=COLORS["ink"], ha="left", va="center")
        ax.text(x + 0.03, y + h * 0.24, label, fontsize=8, color=COLORS["muted"], ha="left", va="center")
    small = list(stats.items())[4:]
    xs = np.linspace(0.08, 0.92, 4)
    ys = [0.12, -0.02]
    for idx, (label, value) in enumerate(small):
        x = xs[idx % 4]
        y = ys[idx // 4]
        ax.text(x, y + 0.08, fmt_int(value), fontsize=13, fontweight="bold", color=COLORS["blue"], ha="center", va="center")
        ax.text(x, y - 0.01, label, fontsize=6.8, color=COLORS["muted"], ha="center", va="center", wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1)


def figure_2a(df: pd.DataFrame) -> list[Path]:
    outcome_categories = sum(1 for cols in ENDPOINT_FAMILIES.values() if present_cols(df, cols) and row_has_any(df, cols).any())
    stats = OrderedDict(
        [
            ("studies", df["study_id"].nunique(dropna=True)),
            ("experimental records", len(df)),
            ("unique LNPs", df["lnp_id"].nunique(dropna=True)),
            ("ionizable lipids", df["il_smiles"].nunique(dropna=True)),
            ("publications", df["publication_link"].nunique(dropna=True)),
            ("biological models", df["biological_model"].nunique(dropna=True)),
            ("biological targets", df["biological_target"].nunique(dropna=True)),
            ("cargo types", df["cargo_type"].nunique(dropna=True)),
            ("experimental settings", df["experimental_setting"].nunique(dropna=True)),
            ("outcome families", outcome_categories),
            ("retained fields", len(df.columns)),
        ]
    )
    save_source(pd.DataFrame({"statistic": list(stats.keys()), "value": list(stats.values())}), "Fig2a_dataset_scale")
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    add_panel_label(ax, "a", -0.02, 1.0)
    draw_stat_infographic(ax, stats)
    return save_fig(fig, "Fig2a_dataset_scale", DIRS["panels"])


def draw_heatmap(ax: plt.Axes, data: pd.DataFrame, cmap="mako", cbar: bool = True) -> None:
    arr = data.to_numpy(dtype=float)
    if sns:
        sns.heatmap(
            data,
            ax=ax,
            vmin=0,
            vmax=100,
            cmap=cmap,
            cbar=cbar,
            cbar_kws={"label": "% records with value", "shrink": 0.75},
            linewidths=0.25,
            linecolor="white",
        )
    else:
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100, cmap=plt.get_cmap("viridis"))
        if cbar:
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% records with value")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def figure_2b(df: pd.DataFrame) -> list[Path]:
    matrix = study_dimension_matrix(df)
    values = matrix.set_index("study_id").drop(columns=["records"])
    fig = plt.figure(figsize=(8.2, 8.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "b", -0.12, 1.02)
    draw_heatmap(ax, values, cmap="mako", cbar=True)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5.5)
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_bar.barh(np.arange(len(matrix)) + 0.5, matrix["records"], color=COLORS["gray"], height=0.8)
    ax_bar.set_xlabel("Records", fontsize=7)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.invert_yaxis()
    return save_fig(fig, "Fig2b_study_dimension_heatmap", DIRS["panels"])


def draw_ranked_bar(ax: plt.Axes, data: pd.DataFrame, label_col: str, value_col: str, color: str, max_items: int = 12, other: bool = True) -> None:
    d = top_n_with_other(data, max_items, label_col) if other else data.head(max_items).copy()
    d = d.sort_values(value_col, ascending=True)
    y = np.arange(len(d))
    ax.barh(y, d[value_col], color=color, height=0.65)
    ax.set_yticks(y, d[label_col])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_int(x)))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, val in zip(y, d[value_col]):
        ax.text(val, yi, f" {fmt_int(val)}", va="center", fontsize=6.5, color=COLORS["muted"])


def figure_2c(df: pd.DataFrame) -> list[Path]:
    setting = categorical_summary(df, "experimental_setting", "Fig2c_experimental_setting")
    cargo = categorical_summary(df, "cargo_type", "Fig2c_cargo_type")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.55})
    add_panel_label(axes[0], "c", -0.18, 1.04)
    draw_ranked_bar(axes[0], setting, "category", "records", COLORS["blue"], max_items=8, other=False)
    axes[0].set_xlabel("Experimental records")
    axes[0].set_title("Experimental setting", loc="left", fontweight="bold", pad=3)
    draw_ranked_bar(axes[1], cargo, "category", "records", COLORS["teal"], max_items=8, other=False)
    axes[1].set_xlabel("Experimental records")
    axes[1].set_title("Cargo type", loc="left", fontweight="bold", pad=3)
    return save_fig(fig, "Fig2c_experimental_landscape", DIRS["panels"])


def figure_2d(df: pd.DataFrame) -> list[Path]:
    models = categorical_summary(df, "biological_model", "Fig2d_biological_model")
    targets = categorical_summary(df, "biological_target", "Fig2d_biological_target")
    routes_df = df[df["route_of_administration"].notna()].copy()
    routes = categorical_summary(routes_df, "route_of_administration", "Fig2d_route_of_administration", include_missing=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9], "wspace": 0.7})
    add_panel_label(axes[0], "d", -0.22, 1.03)
    draw_ranked_bar(axes[0], models[models["category"] != "Not reported"], "category", "records", COLORS["purple"], max_items=12, other=True)
    axes[0].set_xlabel("Records")
    axes[0].set_title("Biological model", loc="left", fontweight="bold", pad=3)
    draw_ranked_bar(axes[1], targets[targets["category"] != "Not reported"], "category", "records", COLORS["orange"], max_items=12, other=True)
    axes[1].set_xlabel("Records")
    axes[1].set_title("Biological target", loc="left", fontweight="bold", pad=3)
    draw_ranked_bar(axes[2], routes, "category", "records", COLORS["green"], max_items=8, other=False)
    axes[2].set_xlabel("In vivo records")
    axes[2].set_title("Administration route", loc="left", fontweight="bold", pad=3)
    return save_fig(fig, "Fig2d_biological_landscape", DIRS["panels"])


def unique_il_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = present_cols(
        df,
        [
            "study_id",
            "publication_link",
            "il_id",
            "il_smiles",
            "il_class",
            "reaction_type",
            "il_architecture_type",
            "il_tail_count",
            "il_head_class",
            "il_linker_class",
        ],
    )
    u = df.loc[non_missing_mask(df["il_smiles"]), cols].drop_duplicates("il_smiles").copy()
    u["publication_year"] = [extract_year(r.get("study_id"), r.get("publication_link")) for _, r in u.iterrows()]
    return u


def chemical_embedding(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out_path = DIRS["source"] / "Fig2e_chemical_space_embedding.csv"
    if out_path.exists():
        emb = pd.read_csv(out_path)
        info = {"valid_molecules": int(emb["rdkit_valid"].sum()), "invalid_molecules": int((~emb["rdkit_valid"]).sum()), "embedding_reused": True}
        return emb, info
    u = unique_il_dataframe(df)
    if Chem is None or rdFingerprintGenerator is None:
        warnings.warn("RDKit unavailable; using molecular descriptor fallback for chemical embedding.")
        desc_cols = present_cols(df, ["heavy_atoms", "rings", "aromatic_rings", "rotatable_bonds", "molecular_weight", "logp", "topological_polar_surface_area"])
        tmp = df.drop_duplicates("il_smiles")[["il_smiles"] + desc_cols].dropna(subset=["il_smiles"]).copy()
        u = u.merge(tmp, on="il_smiles", how="left")
        x = u[desc_cols].fillna(u[desc_cols].median(numeric_only=True)).to_numpy()
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x) if PCA else np.column_stack([x[:, 0], x[:, 1]])
        u["UMAP1"], u["UMAP2"] = coords[:, 0], coords[:, 1]
        u["rdkit_valid"] = False
        save_source(u, "Fig2e_chemical_space_embedding")
        return u, {"valid_molecules": 0, "invalid_molecules": len(u), "embedding_reused": False}
    RDLogger.DisableLog("rdApp.*")
    mols = []
    valid = []
    for smi in u["il_smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        mols.append(mol)
        valid.append(mol is not None)
    u["rdkit_valid"] = valid
    valid_df = u[u["rdkit_valid"]].copy().reset_index(drop=True)
    invalid_df = u[~u["rdkit_valid"]].copy()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for mol in [m for m in mols if m is not None]:
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        fps.append(arr)
    fp = np.vstack(fps).astype(bool)
    if umap is not None and len(valid_df) > 2:
        reducer = umap.UMAP(n_components=2, metric="jaccard", n_neighbors=30, min_dist=0.08, random_state=RANDOM_SEED, low_memory=True)
        coords = reducer.fit_transform(fp)
    elif PCA is not None:
        coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(fp.astype(float))
    else:
        coords = np.column_stack([np.arange(len(valid_df)), np.zeros(len(valid_df))])
    valid_df["UMAP1"], valid_df["UMAP2"] = coords[:, 0], coords[:, 1]
    if len(invalid_df):
        invalid_df["UMAP1"], invalid_df["UMAP2"] = np.nan, np.nan
    emb = pd.concat([valid_df, invalid_df], ignore_index=True)
    save_source(emb, "Fig2e_chemical_space_embedding")
    return emb, {"valid_molecules": len(valid_df), "invalid_molecules": len(invalid_df), "embedding_reused": False}


def scatter_by_category(ax: plt.Axes, emb: pd.DataFrame, color_col: str, max_legend: int = 12) -> None:
    d = emb[emb["rdkit_valid"] & emb["UMAP1"].notna()].copy()
    if color_col not in d:
        ax.scatter(d["UMAP1"], d["UMAP2"], s=5, color=COLORS["blue"], alpha=0.6, linewidth=0)
        return
    d[color_col] = d[color_col].fillna("Not reported").astype(str)
    counts = d[color_col].value_counts()
    keep = counts.head(max_legend).index
    d["_cat"] = np.where(d[color_col].isin(keep), d[color_col], "Other")
    cats = list(pd.Series(d["_cat"]).drop_duplicates())
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set2").colors)
    for i, cat in enumerate(cats):
        sub = d[d["_cat"] == cat]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=5, alpha=0.65, linewidth=0, color=palette[i % len(palette)], label=clean_label(cat))
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, markerscale=2, handletextpad=0.2)


def figure_2e(df: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, dict]:
    emb, info = chemical_embedding(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    add_panel_label(ax, "e", -0.08, 1.03)
    scatter_by_category(ax, emb, "il_class", max_legend=10)
    paths = save_fig(fig, "Fig2e_chemical_space_il_class", DIRS["panels"])
    for col in ["study_id", "reaction_type", "il_architecture_type", "il_tail_count", "publication_year"]:
        if col in emb:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            scatter_by_category(ax, emb, col, max_legend=12)
            save_fig(fig, f"SuppFig10_chemical_space_{col}", DIRS["supplementary"])
    return paths, emb, info


def alluvial_positions(categories: list[str], counts: dict[str, int]) -> dict[str, tuple[float, float]]:
    total = sum(counts.values()) or 1
    gap = 0.018
    scale = (1 - gap * (len(categories) - 1)) / total
    y = 0.0
    pos = {}
    for cat in categories:
        h = counts[cat] * scale
        pos[cat] = (y, y + h)
        y += h + gap
    return pos


def draw_alluvial(ax: plt.Axes, data: pd.DataFrame, stages: list[str], color_stage: str) -> None:
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(stages))
    stage_cats = {}
    stage_pos = {}
    for st in stages:
        counts = data.groupby(st)["count"].sum().sort_values(ascending=False).to_dict()
        cats = list(counts.keys())
        stage_cats[st] = cats
        stage_pos[st] = alluvial_positions(cats, counts)
    color_counts = data.groupby(color_stage)["count"].sum().sort_values(ascending=False)
    palette = dict(zip(color_counts.index, plt.get_cmap("Set2").colors * 4))
    for i in range(len(stages) - 1):
        left, right = stages[i], stages[i + 1]
        group_cols = list(dict.fromkeys([left, right, color_stage]))
        flow = data.groupby(group_cols, dropna=False)["count"].sum().reset_index()
        left_offsets = {k: stage_pos[left][k][0] for k in stage_cats[left]}
        right_offsets = {k: stage_pos[right][k][0] for k in stage_cats[right]}
        for _, r in flow.sort_values("count", ascending=False).iterrows():
            lcat, rcat = r[left], r[right]
            cnt = r["count"]
            l0 = left_offsets[lcat]
            r0 = right_offsets[rcat]
            l_total = data.groupby(left)["count"].sum()[lcat]
            r_total = data.groupby(right)["count"].sum()[rcat]
            l_height = (stage_pos[left][lcat][1] - stage_pos[left][lcat][0]) * cnt / l_total
            r_height = (stage_pos[right][rcat][1] - stage_pos[right][rcat][0]) * cnt / r_total
            left_offsets[lcat] += l_height
            right_offsets[rcat] += r_height
            verts = [
                (xs[i], l0),
                ((xs[i] + xs[i + 1]) / 2, l0),
                ((xs[i] + xs[i + 1]) / 2, r0),
                (xs[i + 1], r0),
                (xs[i + 1], r0 + r_height),
                ((xs[i] + xs[i + 1]) / 2, r0 + r_height),
                ((xs[i] + xs[i + 1]) / 2, l0 + l_height),
                (xs[i], l0 + l_height),
                (xs[i], l0),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            path = MplPath(verts, codes)
            ax.add_patch(patches.PathPatch(path, facecolor=palette.get(r[color_stage], COLORS["gray"]), alpha=0.45, edgecolor="none"))
    for x, st in zip(xs, stages):
        ax.text(x, 1.04, st.replace("_", " "), ha="center", va="bottom", fontsize=8, fontweight="bold")
        for cat in stage_cats[st]:
            y0, y1 = stage_pos[st][cat]
            ax.add_patch(patches.Rectangle((x - 0.018, y0), 0.036, y1 - y0, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.5))
            if y1 - y0 > 0.035:
                ax.text(x + 0.025, (y0 + y1) / 2, clean_label(cat), va="center", ha="left", fontsize=6.3, color=COLORS["ink"])
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.1)


def structural_flow_data(df: pd.DataFrame) -> pd.DataFrame:
    u = unique_il_dataframe(df)
    stages = ["reaction_type", "il_architecture_type", "il_tail_count"]
    for col in stages:
        u[col] = u[col].map(clean_label)
    flow_raw = u.groupby(stages, dropna=False).size().reset_index(name="count")
    save_source(flow_raw, "Fig2f_structural_diversity_alluvial_raw")
    # Group extremely sparse first-stage categories for legibility; raw source remains preserved above.
    reaction_counts = u["reaction_type"].value_counts()
    rare = reaction_counts[reaction_counts < max(10, len(u) * 0.005)].index
    u["reaction_type_plot"] = np.where(u["reaction_type"].isin(rare), "Other sparse reactions", u["reaction_type"])
    plot = (
        u.groupby(["reaction_type_plot", "il_architecture_type", "il_tail_count"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"reaction_type_plot": "reaction_type"})
    )
    save_source(plot, "Fig2f_structural_diversity_alluvial_plot")
    return plot


def figure_2f(df: pd.DataFrame) -> list[Path]:
    flow = structural_flow_data(df)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    add_panel_label(ax, "f", -0.04, 1.0)
    draw_alluvial(ax, flow, ["reaction_type", "il_architecture_type", "il_tail_count"], "reaction_type")
    return save_fig(fig, "Fig2f_structural_diversity_alluvial", DIRS["panels"])


def endpoint_summary(df: pd.DataFrame, source_name: str = "Fig2g_outcome_landscape") -> pd.DataFrame:
    rows = []
    for family, cols in ENDPOINT_FAMILIES.items():
        for col in present_cols(df, cols):
            mask = non_missing_mask(df[col])
            rows.append(
                {
                    "family": family,
                    "endpoint": col,
                    "measurements": int(mask.sum()),
                    "studies": int(df.loc[mask, "study_id"].nunique()),
                    "unique_lnps": int(df.loc[mask, "lnp_id"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    save_source(out, source_name)
    return out


def figure_2g(df: pd.DataFrame) -> list[Path]:
    out = endpoint_summary(df)
    d = out.sort_values(["family", "measurements"], ascending=[True, False]).copy()
    d["endpoint_label"] = d["endpoint"].str.replace("_", " ", regex=False)
    d["x"] = np.log10(d["measurements"].clip(lower=1))
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    add_panel_label(ax, "g", -0.08, 1.02)
    sizes = 18 + np.sqrt(d["studies"].clip(lower=1)) * 18
    colors = [FAMILY_COLORS.get(f, COLORS["gray"]) for f in d["family"]]
    ax.scatter(d["measurements"], y, s=sizes, c=colors, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yticks(y, d["endpoint_label"])
    ax.set_xlabel("Non-null measurements (log scale)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [patches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items() if f in set(d["family"])]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right")
    return save_fig(fig, "Fig2g_outcome_landscape", DIRS["panels"])


def figure2_combined(df: pd.DataFrame, emb: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(12.0, 18.0))
    gs = gridspec.GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[0.85, 0.82, 0.95, 1.0, 1.05],
        width_ratios=[1, 1.25],
        hspace=0.58,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a", -0.03, 1.03)
    outcome_categories = sum(1 for cols in ENDPOINT_FAMILIES.values() if present_cols(df, cols) and row_has_any(df, cols).any())
    stats = OrderedDict(
        [
            ("studies", df["study_id"].nunique(dropna=True)),
            ("experimental records", len(df)),
            ("unique LNPs", df["lnp_id"].nunique(dropna=True)),
            ("ionizable lipids", df["il_smiles"].nunique(dropna=True)),
            ("publications", df["publication_link"].nunique(dropna=True)),
            ("biological models", df["biological_model"].nunique(dropna=True)),
            ("biological targets", df["biological_target"].nunique(dropna=True)),
            ("cargo types", df["cargo_type"].nunique(dropna=True)),
            ("experimental settings", df["experimental_setting"].nunique(dropna=True)),
            ("outcome families", outcome_categories),
            ("retained fields", len(df.columns)),
        ]
    )
    draw_stat_infographic(ax_a, stats)
    ax_b = fig.add_subplot(gs[0:2, 1])
    add_panel_label(ax_b, "b", -0.09, 1.02)
    m = study_dimension_matrix(df).set_index("study_id").drop(columns=["records"])
    draw_heatmap(ax_b, m, cmap="mako", cbar=True)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), fontsize=4.8)

    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.55)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    add_panel_label(ax_c1, "c", -0.28, 1.06)
    draw_ranked_bar(ax_c1, categorical_summary(df, "experimental_setting", "Fig2c_experimental_setting_combined"), "category", "records", COLORS["blue"], max_items=5, other=False)
    ax_c1.set_title("Experimental setting", loc="left", fontweight="bold", pad=2)
    ax_c1.set_xlabel("Records")
    draw_ranked_bar(ax_c2, categorical_summary(df, "cargo_type", "Fig2c_cargo_type_combined"), "category", "records", COLORS["teal"], max_items=6, other=False)
    ax_c2.set_title("Cargo type", loc="left", fontweight="bold", pad=2)
    ax_c2.set_xlabel("Records")

    gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], wspace=0.62, width_ratios=[1.1, 1.1, 0.85])
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])
    ax_d3 = fig.add_subplot(gs_d[0, 2])
    add_panel_label(ax_d1, "d", -0.14, 1.05)
    draw_ranked_bar(ax_d1, categorical_summary(df, "biological_model", "Fig2d_biological_model_combined").query("category != 'Not reported'"), "category", "records", COLORS["purple"], max_items=10, other=True)
    ax_d1.set_title("Biological model", loc="left", fontweight="bold", pad=2)
    ax_d1.set_xlabel("Records")
    draw_ranked_bar(ax_d2, categorical_summary(df, "biological_target", "Fig2d_biological_target_combined").query("category != 'Not reported'"), "category", "records", COLORS["orange"], max_items=10, other=True)
    ax_d2.set_title("Biological target", loc="left", fontweight="bold", pad=2)
    ax_d2.set_xlabel("Records")
    routes_df = df[df["route_of_administration"].notna()].copy()
    draw_ranked_bar(ax_d3, categorical_summary(routes_df, "route_of_administration", "Fig2d_route_of_administration_combined", include_missing=False), "category", "records", COLORS["green"], max_items=8, other=False)
    ax_d3.set_title("Administration route", loc="left", fontweight="bold", pad=2)
    ax_d3.set_xlabel("In vivo records")

    ax_e = fig.add_subplot(gs[3, 0])
    add_panel_label(ax_e, "e", -0.09, 1.04)
    scatter_by_category(ax_e, emb, "il_class", max_legend=8)
    ax_f = fig.add_subplot(gs[3, 1])
    add_panel_label(ax_f, "f", -0.08, 1.0)
    draw_alluvial(ax_f, structural_flow_data(df), ["reaction_type", "il_architecture_type", "il_tail_count"], "reaction_type")

    ax_g = fig.add_subplot(gs[4, :])
    add_panel_label(ax_g, "g", -0.04, 1.04)
    out = endpoint_summary(df, "Fig2g_outcome_landscape_combined")
    d = out.sort_values(["family", "measurements"], ascending=[True, False]).copy()
    y = np.arange(len(d))[::-1]
    ax_g.scatter(d["measurements"], y, s=18 + np.sqrt(d["studies"].clip(lower=1)) * 16, c=[FAMILY_COLORS.get(f, COLORS["gray"]) for f in d["family"]], alpha=0.88, edgecolor="white", linewidth=0.3)
    ax_g.set_xscale("log")
    ax_g.set_yticks(y, d["endpoint"].str.replace("_", " ", regex=False), fontsize=5.5)
    ax_g.set_xlabel("Non-null measurements")
    ax_g.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    return save_fig(fig, "Figure2_LNPHub_multidimensional_landscape", DIRS["main"])


def supplementary_1(df: pd.DataFrame) -> list[Path]:
    out = (
        df.groupby("study_id")
        .agg(records=("record_id", "count"), unique_lnps=("lnp_id", pd.Series.nunique), unique_ionizable_lipids=("il_smiles", pd.Series.nunique))
        .reset_index()
        .sort_values("records", ascending=True)
    )
    save_source(out, "SuppFig1_dataset_contribution_by_study")
    fig, ax = plt.subplots(figsize=(7.5, 11.0))
    y = np.arange(len(out))
    for col, color, offset in [("records", COLORS["blue"], -0.18), ("unique_lnps", COLORS["teal"], 0), ("unique_ionizable_lipids", COLORS["orange"], 0.18)]:
        ax.scatter(out[col], y + offset, s=14, label=col.replace("_", " "), color=color, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yticks(y, out["study_id"], fontsize=5.2)
    ax.set_xlabel("Count (log scale)")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    return save_fig(fig, "Supplementary_Fig1_dataset_contribution_by_study", DIRS["supplementary"])


def supplementary_2(df: pd.DataFrame) -> list[Path]:
    mat = field_coverage_matrix(df)
    values = mat.set_index("study_id").drop(columns=["records"])
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    draw_heatmap(ax, values, cmap="mako", cbar=True)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    ax.set_xticklabels([t.get_text().replace("_", " ") for t in ax.get_xticklabels()], fontsize=5.3)
    return save_fig(fig, "Supplementary_Fig2_field_level_coverage_heatmap", DIRS["supplementary"])


def supplementary_3_4(df: pd.DataFrame, col: str, fig_no: int) -> list[Path]:
    out = categorical_summary(df, col, f"SuppFig{fig_no}_{col}_distribution")
    d = out[out["category"] != "Not reported"].sort_values("records", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(4, len(d) * 0.16)))
    y = np.arange(len(d))
    ax.barh(y, d["records"], color=COLORS["blue"] if col == "biological_model" else COLORS["orange"], alpha=0.82, height=0.65, label="records")
    ax.scatter(d["unique_lnps"], y, s=16, color=COLORS["teal"], label="unique LNPs", zorder=3)
    ax.scatter(d["studies"], y, s=16, color=COLORS["rose"], label="studies", zorder=3)
    ax.set_yticks(y, d["category"], fontsize=6)
    ax.set_xlabel("Count")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_int(x)))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    return save_fig(fig, f"Supplementary_Fig{fig_no}_{col}_distribution", DIRS["supplementary"])


def supplementary_5(df: pd.DataFrame) -> list[Path]:
    matrix = pd.crosstab(df["cargo_type"].fillna("Not reported"), df["experimental_setting"].fillna("Not reported"))
    norm = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100
    save_source(matrix.reset_index(), "SuppFig5_cargo_by_experimental_setting_records")
    save_source(norm.reset_index(), "SuppFig5_cargo_by_experimental_setting_percent_within_cargo")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), gridspec_kw={"wspace": 0.45})
    sns.heatmap(matrix, ax=axes[0], cmap="Blues", annot=True, fmt=",.0f", cbar_kws={"label": "Records"}) if sns else axes[0].imshow(matrix)
    axes[0].set_xlabel("Experimental setting")
    axes[0].set_ylabel("Cargo type")
    sns.heatmap(norm, ax=axes[1], cmap="Greens", annot=True, fmt=".0f", cbar_kws={"label": "% within cargo"}) if sns else axes[1].imshow(norm)
    axes[1].set_xlabel("Experimental setting")
    axes[1].set_ylabel("")
    return save_fig(fig, "Supplementary_Fig5_cargo_experimental_setting_matrix", DIRS["supplementary"])


def supplementary_6(df: pd.DataFrame) -> list[Path]:
    work = df[(df["experimental_setting"] == "in_vivo") & df["route_of_administration"].notna() & df["biological_target"].notna()].copy()
    matrix = pd.crosstab(work["biological_target"], work["route_of_administration"])
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).head(25).index]
    save_source(matrix.reset_index(), "SuppFig6_biological_target_by_route_in_vivo")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sns.heatmap(matrix, ax=ax, cmap="YlGnBu", annot=True, fmt=",.0f", cbar_kws={"label": "In vivo records"}) if sns else ax.imshow(matrix)
    ax.set_xlabel("Administration route")
    ax.set_ylabel("Biological target")
    return save_fig(fig, "Supplementary_Fig6_biological_target_administration_route", DIRS["supplementary"])


def numeric_summary(df: pd.DataFrame, cols: list[str], source_name: str) -> pd.DataFrame:
    rows = []
    for col in present_cols(df, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            rows.append({"field": col, "n": len(s), "median": s.median(), "q1": s.quantile(0.25), "q3": s.quantile(0.75), "iqr": s.quantile(0.75) - s.quantile(0.25), "min": s.min(), "max": s.max()})
    out = pd.DataFrame(rows)
    save_source(out, source_name)
    return out


def supplementary_distribution(df: pd.DataFrame, cols: list[str], fig_no: int, name: str, log_if_skewed: bool = False) -> list[Path]:
    cols = [c for c in present_cols(df, cols) if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5]
    stats = numeric_summary(df, cols, f"SuppFig{fig_no}_{name}_summary_statistics")
    long = []
    log_notes = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        plot_s = s.copy()
        log_scale = False
        if log_if_skewed and (s > 0).all() and s.quantile(0.95) / max(s.quantile(0.50), 1e-12) > 8:
            plot_s = np.log10(s)
            log_scale = True
            log_notes.append(col)
        long.extend([{"field": col, "value": v, "plot_value": pv, "log10_transformed": log_scale} for v, pv in zip(s, plot_s)])
    long_df = pd.DataFrame(long)
    save_source(long_df, f"SuppFig{fig_no}_{name}_plot_values")
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, len(cols) * 0.55)))
    if sns and not long_df.empty:
        sns.violinplot(data=long_df, y="field", x="plot_value", ax=ax, color="#d8ecf3", inner=None, linewidth=0.6)
        sns.boxplot(data=long_df, y="field", x="plot_value", ax=ax, width=0.16, color="white", fliersize=1, linewidth=0.6)
        sns.stripplot(data=long_df.sample(min(len(long_df), 4000), random_state=RANDOM_SEED), y="field", x="plot_value", ax=ax, color=COLORS["blue"], size=1.4, alpha=0.22)
    ax.set_ylabel("")
    ax.set_xlabel("Value" + (f" (log10 for: {', '.join(log_notes)})" if log_notes else ""))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    return save_fig(fig, f"Supplementary_Fig{fig_no}_{name}", DIRS["supplementary"])


def supplementary_9(df: pd.DataFrame) -> list[Path]:
    u = unique_il_dataframe(df)
    frames = []
    for col in present_cols(u, TAXONOMY_COLS):
        tmp = u.copy()
        tmp[col] = tmp[col].map(clean_label)
        summ = tmp.groupby(col).size().reset_index(name="unique_ionizable_lipids").rename(columns={col: "category"}).sort_values("unique_ionizable_lipids", ascending=False)
        summ.insert(0, "variable", col)
        frames.append(summ)
    out = pd.concat(frames, ignore_index=True)
    save_source(out, "SuppFig9_ionizable_lipid_taxonomy")
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.0), gridspec_kw={"hspace": 0.55, "wspace": 0.55})
    for ax, col in zip(axes.flat, present_cols(u, TAXONOMY_COLS)):
        d = out[out["variable"] == col].sort_values("unique_ionizable_lipids", ascending=False)
        draw_ranked_bar(ax, d, "category", "unique_ionizable_lipids", COLORS["blue"], max_items=10, other=True)
        ax.set_title(col.replace("_", " "), loc="left", fontweight="bold", pad=2)
        ax.set_xlabel("Unique ionizable lipids")
    return save_fig(fig, "Supplementary_Fig9_ionizable_lipid_taxonomy", DIRS["supplementary"])


def supplementary_10_grid(emb: pd.DataFrame) -> list[Path]:
    cols = [c for c in ["study_id", "reaction_type", "il_architecture_type", "il_tail_count", "il_class", "publication_year"] if c in emb]
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), gridspec_kw={"hspace": 0.45, "wspace": 0.75})
    for ax, col in zip(axes.flat, cols):
        scatter_by_category(ax, emb, col, max_legend=8)
        ax.set_title(col.replace("_", " "), loc="left", fontweight="bold")
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    return save_fig(fig, "Supplementary_Fig10_chemical_space_annotations", DIRS["supplementary"])


def supplementary_11(df: pd.DataFrame) -> list[Path]:
    metric_col = first_existing(df, ["delivery_value_method", "delivery_metric"])
    value_col = first_existing(df, ["delivery_value", "delivery"])
    work = df[[c for c in ["study_id", metric_col, value_col] if c]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    cont = work[(work[metric_col] == "expression") & work[value_col].notna()].copy()
    disc = work[(work[metric_col] == "expression_discretized") & work[value_col].notna()].copy()
    save_source(cont, "SuppFig11_delivery_expression_continuous")
    save_source(disc, "SuppFig11_delivery_expression_discretized")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"wspace": 0.55})
    top = cont["study_id"].value_counts().head(25).index
    if sns and len(cont):
        sns.boxplot(data=cont[cont["study_id"].isin(top)], y="study_id", x=value_col, ax=axes[0], color="#d8ecf3", fliersize=1, linewidth=0.6)
    axes[0].set_title("expression", loc="left", fontweight="bold")
    axes[0].set_xlabel("Delivery")
    axes[0].set_ylabel("Study")
    if len(disc):
        mat = pd.crosstab(disc["study_id"], disc[value_col]).loc[disc["study_id"].value_counts().head(25).index]
        sns.heatmap(mat, ax=axes[1], cmap="YlGnBu", cbar_kws={"label": "Records"}) if sns else axes[1].imshow(mat)
    axes[1].set_title("expression discretized", loc="left", fontweight="bold")
    axes[1].set_xlabel("Discretized value")
    axes[1].set_ylabel("")
    return save_fig(fig, "Supplementary_Fig11_delivery_distributions_by_study", DIRS["supplementary"])


def supplementary_12(df: pd.DataFrame) -> list[Path]:
    blocks = OrderedDict(
        [
            ("delivery", ["delivery"]),
            ("viability", ENDPOINT_FAMILIES["Viability"]),
            ("toxicity", ENDPOINT_FAMILIES["Toxicity"]),
            ("editing", ENDPOINT_FAMILIES["Editing"]),
            ("uptake", ENDPOINT_FAMILIES["Uptake"]),
            ("immune", ENDPOINT_FAMILIES["Immune"]),
            ("protein abundance", ["protein_abundance_value"]),
            ("mRNA binding", ["mrna_binding_efficiency_percent"]),
            ("GFP transfection", ["gfp_mfi_transfection"]),
        ]
    )
    rows = []
    for study, sdf in df.groupby("study_id"):
        row = {"study_id": study, "records": len(sdf)}
        for block, cols in blocks.items():
            row[block] = int(row_has_any(sdf, cols).sum())
        rows.append(row)
    mat = pd.DataFrame(rows).sort_values("records", ascending=False)
    save_source(mat, "SuppFig12_biological_outcome_coverage_by_study")
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    values = mat.set_index("study_id").drop(columns=["records"])
    sns.heatmap(values, ax=ax, cmap="mako", cbar_kws={"label": "Records with measurement"}) if sns else ax.imshow(values)
    ax.set_xlabel("Endpoint family")
    ax.set_ylabel("Study")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5.2)
    return save_fig(fig, "Supplementary_Fig12_biological_outcome_coverage_by_study", DIRS["supplementary"])


def supplementary_13(df: pd.DataFrame, completeness: pd.DataFrame) -> list[Path]:
    block_rows = []
    for block, cols in DIMENSION_BLOCKS.items():
        cols = present_cols(df, cols)
        if cols:
            block_rows.append({"block": block.replace("\n", " "), "percent_complete": float(np.mean([non_missing_mask(df[c]).mean() * 100 for c in cols])), "columns": len(cols)})
    blocks = pd.DataFrame(block_rows).sort_values("percent_complete")
    important = completeness[completeness["column"].isin(endpoint_columns(df) + ["study_id", "lnp_id", "il_smiles", "cargo_type", "experimental_setting", "biological_model", "biological_target"])].sort_values("percent_complete")
    save_source(blocks, "SuppFig13_completeness_by_major_block")
    save_source(important, "SuppFig13_completeness_important_fields")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"wspace": 0.58})
    draw_ranked_bar(axes[0], blocks.rename(columns={"block": "category"}), "category", "percent_complete", COLORS["blue"], max_items=20, other=False)
    axes[0].set_xlabel("% complete")
    axes[0].set_title("Major data blocks", loc="left", fontweight="bold")
    draw_ranked_bar(axes[1], important.rename(columns={"column": "category"}), "category", "percent_complete", COLORS["teal"], max_items=24, other=False)
    axes[1].set_xlabel("% complete")
    axes[1].set_title("Scientifically important fields", loc="left", fontweight="bold")
    return save_fig(fig, "Supplementary_Fig13_missingness_completeness_landscape", DIRS["supplementary"])


def supplementary_14(df: pd.DataFrame) -> list[Path]:
    work = df.copy()
    work["publication_year"] = [extract_year(r.get("study_id"), r.get("publication_link")) for _, r in work.iterrows()]
    work = work.dropna(subset=["publication_year"]).copy()
    work["publication_year"] = work["publication_year"].astype(int)
    rows = []
    for year, ydf in work.groupby("publication_year"):
        up_to = work[work["publication_year"] <= year]
        rows.append({"publication_year": year, "studies": ydf["study_id"].nunique(), "cumulative_records": len(up_to), "cumulative_unique_ionizable_lipids": up_to["il_smiles"].nunique(), "cumulative_unique_lnps": up_to["lnp_id"].nunique()})
    out = pd.DataFrame(rows).sort_values("publication_year")
    save_source(out, "SuppFig14_temporal_growth")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.bar(out["publication_year"], out["studies"], color=COLORS["gray"], alpha=0.55, label="studies")
    ax1.set_ylabel("Studies")
    ax2 = ax1.twinx()
    for col, color in [("cumulative_records", COLORS["blue"]), ("cumulative_unique_ionizable_lipids", COLORS["orange"]), ("cumulative_unique_lnps", COLORS["teal"])]:
        ax2.plot(out["publication_year"], out[col], marker="o", ms=3, color=color, label=col.replace("_", " "))
    ax2.set_ylabel("Cumulative count")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_int(x)))
    ax1.set_xlabel("Publication year derived from study identifier/publication text")
    lines, labels = ax2.get_legend_handles_labels()
    b_lines, b_labels = ax1.get_legend_handles_labels()
    ax2.legend(b_lines + lines, b_labels + labels, frameon=False, loc="upper left")
    return save_fig(fig, "Supplementary_Fig14_temporal_growth_lnp_corpus", DIRS["supplementary"])


def supplementary_tables(df: pd.DataFrame, completeness: pd.DataFrame) -> None:
    phys_mask = row_has_any(df, ENDPOINT_FAMILIES["Physicochemical"])
    rows = []
    for study, sdf in df.groupby("study_id"):
        pub_links = "; ".join(sorted(sdf["publication_link"].dropna().astype(str).unique()))
        rows.append(
            {
                "study_id": study,
                "publication_link": pub_links,
                "publication_year": extract_year(study, pub_links),
                "experimental_records": len(sdf),
                "unique_lnp_id": sdf["lnp_id"].nunique(),
                "unique_il_id": sdf["il_id"].nunique(),
                "unique_il_smiles": sdf["il_smiles"].nunique(),
                "experimental_settings_represented": "; ".join(sorted(sdf["experimental_setting"].dropna().astype(str).unique())),
                "biological_models_count": sdf["biological_model"].nunique(),
                "biological_targets_count": sdf["biological_target"].nunique(),
                "cargo_types": "; ".join(sorted(sdf["cargo_type"].dropna().astype(str).unique())),
                "delivery_records": int(row_has_any(sdf, ["delivery"]).sum()),
                "physicochemical_records": int(phys_mask.loc[sdf.index].sum()),
                "editing_records": int(row_has_any(sdf, ENDPOINT_FAMILIES["Editing"]).sum()),
                "uptake_records": int(row_has_any(sdf, ENDPOINT_FAMILIES["Uptake"]).sum()),
                "immune_records": int(row_has_any(sdf, ENDPOINT_FAMILIES["Immune"]).sum()),
                "toxicity_records": int(row_has_any(sdf, ENDPOINT_FAMILIES["Toxicity"]).sum()),
            }
        )
    table1 = pd.DataFrame(rows).sort_values("experimental_records", ascending=False)
    table1.to_csv(DIRS["tables"] / "Supplementary_Table1_study_summary.csv", index=False)

    vocab_cols = [
        "experimental_setting",
        "cargo_type",
        "biological_model",
        "biological_target",
        "route_of_administration",
        "il_class",
        "reaction_type",
        "il_architecture_type",
        "il_head_class",
        "il_linker_class",
        first_existing(df, ["delivery_value_method", "delivery_metric"]),
    ]
    vocab_rows = []
    for col in [c for c in vocab_cols if c and c in df.columns]:
        tmp = categorical_summary(df, col, f"Table2_controlled_vocabulary_{col}")
        for _, r in tmp.iterrows():
            vocab_rows.append({"variable": col, "category": r["category"], "number_of_records": r["records"], "number_of_studies": r["studies"], "number_of_unique_LNPs": r["unique_lnps"]})
    table2 = pd.DataFrame(vocab_rows)
    table2.to_csv(DIRS["tables"] / "Supplementary_Table2_controlled_vocabulary_summary.csv", index=False)
    completeness.to_csv(DIRS["tables"] / "Supplementary_Table3_field_completeness.csv", index=False)
    with pd.ExcelWriter(DIRS["tables"] / "Supplementary_Table1_study_summary.xlsx", engine="xlsxwriter") as writer:
        table1.to_excel(writer, index=False, sheet_name="Study summary")
    with pd.ExcelWriter(DIRS["tables"] / "Supplementary_Table2_controlled_vocabulary_summary.xlsx", engine="xlsxwriter") as writer:
        table2.to_excel(writer, index=False, sheet_name="Controlled vocabulary")
    with pd.ExcelWriter(DIRS["tables"] / "Supplementary_Table3_field_completeness.xlsx", engine="xlsxwriter") as writer:
        completeness.to_excel(writer, index=False, sheet_name="Field completeness")


def results_statistics(df: pd.DataFrame) -> None:
    rows = []
    def add(section, metric, value, extra=""):
        rows.append({"section": section, "metric": metric, "value": value, "extra": extra})
    add("DATASET SCALE", "total records", len(df))
    add("DATASET SCALE", "studies", df["study_id"].nunique(dropna=True))
    add("DATASET SCALE", "publications", df["publication_link"].nunique(dropna=True))
    add("DATASET SCALE", "unique LNPs", df["lnp_id"].nunique(dropna=True))
    add("DATASET SCALE", "unique ionizable lipids", df["il_smiles"].nunique(dropna=True))
    for col, section in [("il_class", "CHEMICAL DIVERSITY"), ("reaction_type", "CHEMICAL DIVERSITY"), ("il_architecture_type", "CHEMICAL DIVERSITY"), ("il_tail_count", "CHEMICAL DIVERSITY"), ("experimental_setting", "EXPERIMENTAL DIVERSITY"), ("cargo_type", "EXPERIMENTAL DIVERSITY"), ("route_of_administration", "EXPERIMENTAL DIVERSITY")]:
        if col in df:
            if col == "route_of_administration":
                route_den = int(non_missing_mask(df[col]).sum())
                for cat, count in df.loc[non_missing_mask(df[col]), col].value_counts().items():
                    pct = count / max(route_den, 1) * 100
                    add(section, f"{col}:{clean_label(cat)}", int(count), f"{pct:.1f}% of route-bearing records")
                add(section, "route_of_administration:missing/not applicable", int((~non_missing_mask(df[col])).sum()), f"{(~non_missing_mask(df[col])).mean() * 100:.1f}% of all records")
            else:
                for cat, count in df[col].fillna("Not reported").value_counts().items():
                    pct = count / len(df) * 100
                    add(section, f"{col}:{clean_label(cat)}", int(count), f"{pct:.1f}% of all records")
    for col, title in [("biological_model", "models"), ("biological_target", "targets")]:
        add("BIOLOGICAL DIVERSITY", f"unique biological {title}", df[col].nunique(dropna=True))
        for cat, count in df.loc[non_missing_mask(df[col]), col].value_counts().head(10).items():
            add("BIOLOGICAL DIVERSITY", f"top {col}:{clean_label(cat)}", int(count))
    for col in present_cols(df, ENDPOINT_FAMILIES["Physicochemical"]):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            add("PHYSICOCHEMICAL COVERAGE", f"{col}:observations", len(s), f"studies={df.loc[s.index, 'study_id'].nunique()}; median={s.median():.4g}; IQR={s.quantile(.25):.4g}-{s.quantile(.75):.4g}")
    for _, r in endpoint_summary(df, "results_statistics_endpoint_summary").iterrows():
        add("BIOLOGICAL OUTCOMES", r["endpoint"], int(r["measurements"]), f"family={r['family']}; studies={int(r['studies'])}; unique_LNPs={int(r['unique_lnps'])}")
    dim = study_dimension_matrix(df)
    dim_cols = [c for c in dim.columns if c not in ("study_id", "records")]
    populated = (dim[dim_cols] > 0).sum(axis=1)
    add("DATA COVERAGE", "average populated outcome dimensions per study", f"{populated.mean():.2f}")
    add("DATA COVERAGE", "median populated outcome dimensions per study", f"{populated.median():.0f}")
    add("DATA COVERAGE", "range populated outcome dimensions per study", f"{populated.min()}-{populated.max()}")
    top_studies = "; ".join(dim.loc[populated.sort_values(ascending=False).head(10).index, "study_id"].astype(str))
    add("DATA COVERAGE", "studies covering greatest number of dimensions", top_studies)
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["tables"] / "results_statistics.csv", index=False)
    lines = []
    for section, sdf in out.groupby("section", sort=False):
        lines.append(section)
        lines.append("-" * len(section))
        for _, r in sdf.iterrows():
            extra = f" ({r['extra']})" if str(r["extra"]) else ""
            lines.append(f"{r['metric']}: {r['value']}{extra}")
        lines.append("")
    (DIRS["tables"] / "results_statistics.txt").write_text("\n".join(lines))


def suspicious_data_report(df: pd.DataFrame, chem_info: dict) -> None:
    rows = []
    for col in ["experimental_setting", "cargo_type", "biological_model", "biological_target", "route_of_administration", "il_smiles", "delivery"]:
        if col in df:
            rows.append({"issue": f"missing {col}", "count": int((~non_missing_mask(df[col])).sum()), "note": "Not modified; reported as missing/not reported in plots where relevant."})
    rows.append({"issue": "invalid RDKit SMILES", "count": chem_info.get("invalid_molecules", np.nan), "note": "Excluded only from chemical-space fingerprint embedding; retained in source metadata."})
    if "route_of_administration" in df and "experimental_setting" in df:
        not_invivo_routes = df[df["route_of_administration"].notna() & (df["experimental_setting"] != "in_vivo")]
        rows.append({"issue": "route present outside in_vivo", "count": len(not_invivo_routes), "note": "Route matrix restricted to in_vivo route-bearing records."})
    pd.DataFrame(rows).to_csv(DIRS["tables"] / "suspicious_data_issues.csv", index=False)


def package_versions() -> str:
    modules = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    for name, obj in [("seaborn", sns), ("rdkit", Chem), ("umap", umap)]:
        try:
            modules[name] = getattr(obj, "__version__", "available") if obj is not None else "unavailable"
        except Exception:
            modules[name] = "available"
    return "\n".join(f"{k}: {v}" for k, v in modules.items())


def generate(csv_path: Path) -> None:
    mkdirs()
    set_style()
    log_lines = [f"LNPHub figure generation started {dt.datetime.now().isoformat(timespec='seconds')}", f"Input CSV: {csv_path}", "", "Package versions", package_versions(), ""]
    df = load_data(csv_path)
    audit, audit_long, completeness = audit_dataset(df)
    write_audit_and_mapping(df, audit)
    print("DATASET AUDIT")
    for k, v in audit.items():
        print(f"{k}: {fmt_int(v)}")
    print("\nExperimental setting counts")
    print(df["experimental_setting"].fillna("Not reported").value_counts().to_string())
    print("\nCargo type counts")
    print(df["cargo_type"].fillna("Not reported").value_counts().to_string())
    metric_col = first_existing(df, ["delivery_value_method", "delivery_metric"])
    if metric_col:
        print("\nDelivery metric counts")
        print(df[metric_col].fillna("Not reported").value_counts().to_string())
    print("\nEndpoint non-null counts")
    for c in endpoint_columns(df):
        print(f"{c}: {int(non_missing_mask(df[c]).sum())}")
    generated = []
    for func in [figure_2a, figure_2b, figure_2c, figure_2d]:
        paths = func(df)
        generated.extend(paths)
        log_lines.append(f"Generated {paths[0].stem}")
    paths, emb, chem_info = figure_2e(df)
    generated.extend(paths)
    print(f"\nRDKit valid molecules: {chem_info['valid_molecules']}")
    print(f"RDKit invalid molecules: {chem_info['invalid_molecules']}")
    for func in [figure_2f, figure_2g]:
        paths = func(df)
        generated.extend(paths)
        log_lines.append(f"Generated {paths[0].stem}")
    generated.extend(figure2_combined(df, emb))
    for func in [
        supplementary_1,
        supplementary_2,
        lambda d: supplementary_3_4(d, "biological_model", 3),
        lambda d: supplementary_3_4(d, "biological_target", 4),
        supplementary_5,
        supplementary_6,
        lambda d: supplementary_distribution(d, ENDPOINT_FAMILIES["Physicochemical"], 7, "physicochemical_distributions"),
        lambda d: supplementary_distribution(d, FORMULATION_RATIO_COLS, 8, "formulation_space_diversity", log_if_skewed=True),
        supplementary_9,
        supplementary_11,
        supplementary_12,
        lambda d: supplementary_13(d, completeness),
        supplementary_14,
    ]:
        try:
            generated.extend(func(df))
        except Exception as exc:
            warnings.warn(f"Supplementary figure skipped after error: {exc}")
            log_lines.append(f"WARNING supplementary figure error: {exc}")
    generated.extend(supplementary_10_grid(emb))
    supplementary_tables(df, completeness)
    results_statistics(df)
    suspicious_data_report(df, chem_info)
    log_lines.append("")
    log_lines.append("Generated figure files")
    log_lines.extend(str(p) for p in generated)
    log_lines.append("")
    log_lines.append(f"LNPHub figure generation finished {dt.datetime.now().isoformat(timespec='seconds')}")
    (DIRS["tables"] / "generation_log.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\nGenerated {len(generated)} figure files.")
    print(f"Log: {DIRS['tables'] / 'generation_log.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(args.csv_path)
    with warnings.catch_warnings(record=False):
        warnings.simplefilter("default")
        generate(args.csv_path)


if __name__ == "__main__":
    main()
