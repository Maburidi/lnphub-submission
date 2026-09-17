from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lnphub_matplotlib_cache")

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


# ============================================================
# Nature-style Oracle benchmark bar plot
# Descriptors/FP -> GNNs -> Frozen -> Fine-tuned
# 5-fold CV summary: bars = mean, error bars = std
# ============================================================


# ============================================================
# 1. Load data
# ============================================================

# Prefer the full merged 5-fold CV summary if it exists.
# Fall back to the recovered Colab-display summary.
CSV_PATH = Path(
    "results/multimodal_lasttrial_5fold_cv/merged_results/"
    "lasttrial_5fold_cv_summary_mean_std.csv"
)

if not CSV_PATH.exists():
    CSV_PATH = Path(
        "results/multimodal_lasttrial_5fold_cv/merged_results/"
        "lasttrial_5fold_cv_summary_mean_std_recovered_from_colab_display.csv"
    )

df = pd.read_csv(CSV_PATH)


# ============================================================
# 2. Metric to plot
# ============================================================

# Other good options:
# "balanced_accuracy"
# "mcc"
# "roc_auc_ovr_macro"
# "average_precision_macro"
# "accuracy"

METRIC = "roc_auc_ovr_macro"
VALUE_COL = f"{METRIC}_mean" if f"{METRIC}_mean" in df.columns else METRIC
ERROR_COL = f"{METRIC}_std" if f"{METRIC}_std" in df.columns else None

if VALUE_COL not in df.columns:
    raise ValueError(f"Could not find metric column for {METRIC}. Available columns: {list(df.columns)}")

df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
if ERROR_COL is not None:
    df[ERROR_COL] = pd.to_numeric(df[ERROR_COL], errors="coerce").fillna(0.0)
else:
    df["_metric_std"] = 0.0
    ERROR_COL = "_metric_std"

df = df.dropna(subset=[VALUE_COL]).copy()


# ============================================================
# 3. Desired category order
# ============================================================

category_order = [
    "Descriptor/fingerprint",
    "GNN",
    "Frozen embedding",
    "Fine-tuned encoder",
]

category_display = {
    "Descriptor/fingerprint": "Descriptors / fingerprints",
    "GNN": "GNNs",
    "Frozen embedding": "Frozen molecular models",
    "Fine-tuned encoder": "Fine-tuned molecular models",
}


# ============================================================
# 4. Input-setting order
# ============================================================

context_order = [
    "SMILES only",
    "SMILES + formulation",
    "SMILES + formulation + experimental context",
    "SMILES + formulation + experimental context + physicochemical",
]

context_display = {
    "SMILES only": "SMILES only",
    "SMILES + formulation": "+ formulation",
    "SMILES + formulation + experimental context":
        "+ formulation + experimental context",
    "SMILES + formulation + experimental context + physicochemical":
        "+ formulation + experimental context + physchem",
}


# ============================================================
# 5. Nature-style category colors
# ============================================================

# Restrained, publication-friendly palette
base_colors = {
    "Descriptor/fingerprint": "#3C5488",  # deep blue
    "GNN": "#00A087",                    # teal
    "Frozen embedding": "#E6A23C",       # amber
    "Fine-tuned encoder": "#B23A48",     # wine/red
}


def lighten(color, amount=0.5):
    """
    Mix a color with white.
    amount=0 -> original color
    amount=1 -> white
    """
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * amount)


# Four shades represent increasing amount of information
shade_amounts = {
    "SMILES only": 0.68,
    "SMILES + formulation": 0.45,
    "SMILES + formulation + experimental context": 0.22,
    "SMILES + formulation + experimental context + physicochemical": 0.00,
}


# ============================================================
# 6. Simplify method names for publication
# ============================================================

def clean_method(method):
    s = str(method)

    # Remove split descriptions
    s = re.sub(r"\s+Random Split$", "", s, flags=re.I)
    s = re.sub(r"\s+random 80/10/10$", "", s, flags=re.I)

    # Remove input-setting information because this is already
    # represented by the four bars
    removable_feature_phrases = [
        " + Formulation + Context + PhysChem",
        " + Formulation + Experimental Context + PhysChem",
        " + Formulation + Context",
        " + Formulation + Experimental Context",
        " + Formulation/PhysChem",
        " + Formulation + PhysChem",
        " + Selected Covariates",
        " + Tabular Features",
        " + Formulation",
    ]
    for phrase in removable_feature_phrases:
        s = s.replace(phrase, "")

    # Remove training-mode suffixes because category headers
    # already indicate Frozen vs Fine-tuned
    s = s.replace("-Frozen", "")
    s = s.replace("-Finetuned", "")

    # Compact publication labels
    s = s.replace("RDKit Descriptors", "RDKit Desc.")
    s = s.replace("Random Forest", "RF")
    s = s.replace("XGBoost", "XGB")
    s = s.replace("ChemBERTa-MTR", "ChemBERTa")
    s = s.replace("MolCLR-GIN", "MolCLR")

    return s


df["model_short"] = df["method"].map(clean_method)


# ============================================================
# 7. Model order within each family
# ============================================================

# Frozen and fine-tuned use exactly the same encoder order,
# which makes direct visual comparison easier.

model_order = {

    "Descriptor/fingerprint": [
        "ECFP4 + MLP",
        "ECFP4 + RF",
        "ECFP4 + XGB",
        "RDKit Desc. + MLP",
        "RDKit Desc. + RF",
        "RDKit Desc. + XGB",
    ],

    "GNN": [
        "GCN",
        "GIN",
        "D-MPNN",
    ],

    "Frozen embedding": [
        "AGILE",
        "ChemBERTa",
        "LiON",
        "MolCLR",
        "MoLFormer",
        "Uni-Mol",
    ],

    "Fine-tuned encoder": [
        "AGILE",
        "ChemBERTa",
        "LiON",
        "MolCLR",
        "MoLFormer",
        "Uni-Mol",
    ],
}


# ============================================================
# 8. Publication typography
# ============================================================

mpl.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 9,

    "axes.labelsize": 10,
    "axes.titlesize": 11,

    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8.5,

    "axes.linewidth": 0.8,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    # Keep text editable in Illustrator
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 9. Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(15.2, 5.4),
    dpi=180
)


# Individual bar width
bar_width = 0.18

# Gap between model groups inside the same family
within_model_gap = 0.28

# Large gap between model families
category_gap = 1.35


x = 0.0

xticks = []
xticklabels = []

category_spans = {}

# Save bar information so we can identify the best result
bar_records = []


# ============================================================
# 10. Draw bars
# ============================================================

for category in category_order:

    cat_start = None

    for model in model_order[category]:

        subset = df[
            (df["category_label"] == category) &
            (df["model_short"] == model)
        ]

        if subset.empty:
            continue

        # Center x-position for model label
        center = x + 1.5 * bar_width

        xticks.append(center)
        xticklabels.append(model)

        # ----------------------------------------------------
        # Four bars = four input settings
        # ----------------------------------------------------

        for j, context in enumerate(context_order):

            row = subset[
                subset["experiment_label"] == context
            ]

            if row.empty:
                continue

            xpos = x + j * bar_width

            value = float(row.iloc[0][VALUE_COL])
            error = float(row.iloc[0][ERROR_COL]) if ERROR_COL in row.columns else 0.0

            color = lighten(
                base_colors[category],
                shade_amounts[context]
            )

            ax.bar(
                xpos,
                value,

                width=bar_width * 0.88,

                color=color,
                edgecolor=base_colors[category],

                linewidth=0.55,

                zorder=3,
            )

            ax.errorbar(
                xpos,
                value,
                yerr=error,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.7,
                capsize=2.2,
                capthick=0.7,
                zorder=4,
            )

            bar_records.append(
                (
                    category,
                    model,
                    context,
                    xpos,
                    value,
                    error,
                )
            )

            if cat_start is None:
                cat_start = xpos - bar_width * 0.55

        # Move to next model
        x += (4 * bar_width) + within_model_gap

    if cat_start is None:
        continue

    # Save category boundaries
    cat_end = x - within_model_gap + bar_width * 0.45

    category_spans[category] = (
        cat_start,
        cat_end
    )

    # Large visual separation between families
    x += category_gap


# ============================================================
# 11. Y-axis
# ============================================================

metric_labels = {

    "macro_f1": "Macro-F1 score",

    "balanced_accuracy":
        "Balanced Accuracy",

    "accuracy":
        "Accuracy",

    "mcc":
        "Matthews correlation coefficient",

    "roc_auc_ovr_macro":
        "ROC-AUC",

    "average_precision_macro":
        "Average Precision",
}


ax.set_ylabel(
    metric_labels.get(
        METRIC,
        METRIC.replace("_", " ").title()
    )
)


# Bar plots should normally retain a zero baseline.
upper_limit = max(
    0.70,
    (df[VALUE_COL] + df[ERROR_COL]).max() + 0.055
)

ax.set_ylim(
    0,
    upper_limit
)


# ============================================================
# 12. X labels
# ============================================================

ax.set_xticks(xticks)

ax.set_xticklabels(
    xticklabels,
    rotation=55,
    ha="right",
    rotation_mode="anchor"
)


# ============================================================
# 13. Grid
# ============================================================

ax.yaxis.set_major_locator(
    mpl.ticker.MultipleLocator(0.10)
)

ax.yaxis.set_minor_locator(
    mpl.ticker.MultipleLocator(0.05)
)


ax.grid(
    axis="y",
    which="major",
    color="#D9D9D9",
    linewidth=0.65,
    alpha=0.75,
    zorder=0
)

ax.grid(
    axis="y",
    which="minor",
    color="#ECECEC",
    linewidth=0.45,
    alpha=0.65,
    zorder=0
)


# ============================================================
# 14. Clean Nature-style axes
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#333333")
ax.spines["bottom"].set_color("#333333")

ax.tick_params(
    axis="x",
    length=0,
    pad=4
)

ax.tick_params(
    axis="y",
    length=3,
    color="#333333"
)

ax.set_axisbelow(True)


# ============================================================
# 15. Category headers
# ============================================================

y_top = ax.get_ylim()[1]


visible_categories = [
    category for category in category_order
    if category in category_spans
]

for i, category in enumerate(visible_categories):

    start, end = category_spans[category]

    mid = (start + end) / 2

    # Category title
    ax.text(
        mid,
        y_top * 1.012,

        category_display[category],

        ha="center",
        va="bottom",

        fontsize=10,
        fontweight="bold",

        color=base_colors[category],

        clip_on=False,
    )

    # Colored line underneath category title
    ax.plot(
        [start, end],
        [y_top * 1.002, y_top * 1.002],

        color=base_colors[category],

        linewidth=2.3,

        solid_capstyle="round",

        clip_on=False,
    )

    # Light separator between categories
    if i < len(visible_categories) - 1:

        next_start, _ = category_spans[
            visible_categories[i + 1]
        ]

        sep_x = (end + next_start) / 2

        ax.axvline(
            sep_x,

            color="#D0D0D0",

            linewidth=0.7,

            linestyle=(0, (2, 3)),

            zorder=1
        )


# ============================================================
# 16. Annotate best result from each family
# ============================================================

for category in visible_categories:

    records = [
        r for r in bar_records
        if r[0] == category
    ]

    if not records:
        continue

    best = max(
        records,
        key=lambda r: r[4]
    )

    _, _, _, xpos, value, error = best

    ax.text(
        xpos,
        value + error + 0.010,

        f"{value:.3f}",

        ha="center",
        va="bottom",

        fontsize=7.3,
        fontweight="bold",

        color="#222222"
    )


# ============================================================
# 17. Legend
# ============================================================

# Gray proxies emphasize that lightness means input setting,
# while the actual bar hue means model family.

legend_handles = [

    Patch(
        facecolor="#E1E1E1",
        edgecolor="#777777",
        label="SMILES only"
    ),

    Patch(
        facecolor="#C8C8C8",
        edgecolor="#777777",
        label="SMILES + formulation"
    ),

    Patch(
        facecolor="#969696",
        edgecolor="#666666",
        label="SMILES + formulation + experimental context"
    ),

    Patch(
        facecolor="#6F6F6F",
        edgecolor="#555555",
        label="SMILES + formulation + experimental context + physchem"
    ),
]


ax.legend(
    handles=legend_handles,

    title="Input setting",

    ncol=2,

    frameon=False,

    loc="upper center",
    bbox_to_anchor=(0.5, -0.31),

    handlelength=1.2,
    columnspacing=1.8,

    fontsize=11,
    title_fontsize=13.5
)


# ============================================================
# 18. Layout
# ============================================================

plt.subplots_adjust(
    left=0.07,
    right=0.995,
    top=0.87,
    bottom=0.36
)


# ============================================================
# 19. Save publication-quality outputs
# ============================================================

out_dir = Path(
    "results/multimodal_lasttrial_5fold_cv/merged_results/"
    "publication_barplots"
)
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / f"oracle_{METRIC}_5fold_cv_nature_barplot.pdf"
png_path = out_dir / f"oracle_{METRIC}_5fold_cv_nature_barplot.png"

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight"
)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.close(fig)
