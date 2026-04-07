#!/usr/bin/env python3
"""
Supplementary Figure: AlphaFold3 structural validation across all case-study proteins.

Reads af3_structural_validation.csv and produces:
  Panel A: RMSD to native structure (grouped bar chart, all proteins x 4 conditions)
  Panel B: pTM confidence scores (same layout)
  Panel C: RMSD vs pTM scatter (coloured by protein)
  Panel D: Heatmap summary (protein x condition, RMSD values)
"""

import numpy as np
import pandas as pd
import matplotlib
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent
PIPELINE_ROOT = BASE.parent
DATA_DIR = PIPELINE_ROOT / "data"
DATA_CSV = DATA_DIR / "af3_results" / "analysis" / "af3_validation_combined.csv"
OUT_DIR = DATA_DIR / "af3_results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
TEXT_SCALE = 1.35
BASE_FONT = int(round(8 * TEXT_SCALE))
TITLE_FONT = int(round(9 * TEXT_SCALE))
TICK_FONT = int(round(7 * TEXT_SCALE))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": BASE_FONT,
    "axes.titlesize": TITLE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Condition display
CONDITION_ORDER = [
    "unconstrained",
    "unconstrained_denovo_glycans",
    "full_sequon_fixed",
    "full_sequon_fixed_full_glycans",
    "full_sequon_fixed_with_glycans",
]
CONDITION_LABELS = {
    "unconstrained": "Unconstrained",
    "unconstrained_denovo_glycans": "Uncons. +\nde novo gly",
    "full_sequon_fixed": "Fixed",
    "full_sequon_fixed_full_glycans": "Fixed +\nfull gly",
    "full_sequon_fixed_with_glycans": "Fixed +\ngly",
}
CONDITION_COLORS = {
    "unconstrained": "#e74c3c",
    "unconstrained_denovo_glycans": "#e67e22",
    "full_sequon_fixed": "#2980b9",
    "full_sequon_fixed_full_glycans": "#16a085",
    "full_sequon_fixed_with_glycans": "#27ae60",
}
PROTEIN_ORDER = ["1GQV", "1ATJ", "1RUZ", "5EQG", "1C1Z"]
PROTEIN_COLORS = {
    "1GQV": "#9b59b6",
    "1ATJ": "#e74c3c",
    "1RUZ": "#2980b9",
    "5EQG": "#27ae60",
    "1C1Z": "#e67e22",
}
PROTEIN_DESCRIPTIONS = {
    "1GQV": "Neuraminidase\n(single chain)",
    "1ATJ": "HRP C1A\n(6 chains)",
    "1RUZ": "1918 HA\n(6 chains)",
    "5EQG": "GLUT1\n(single chain)",
    "1C1Z": "β2-GPI\n(single chain)",
}


def make_panel_a(ax, df):
    """Grouped bar chart: RMSD across proteins and conditions."""
    n_proteins = len(PROTEIN_ORDER)
    n_conditions = len(CONDITION_ORDER)
    bar_width = 0.18
    x = np.arange(n_proteins)

    for i, cond in enumerate(CONDITION_ORDER):
        rmsds = []
        for prot in PROTEIN_ORDER:
            row = df[(df["pdb_id"] == prot) & (df["condition"] == cond)]
            rmsds.append(row["mean_rmsd"].values[0] if len(row) else 0)

        offset = (i - n_conditions / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, rmsds,
            bar_width, color=CONDITION_COLORS[cond],
            edgecolor="white", linewidth=0.4,
            label=CONDITION_LABELS[cond].replace("\n", " "),
        )

        for bar, val in zip(bars, rmsds):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=4.5 * TEXT_SCALE, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([PROTEIN_DESCRIPTIONS[p] for p in PROTEIN_ORDER],
                       fontsize=5.5 * TEXT_SCALE)
    ax.set_ylabel("RMSD to native structure (Å)")
    ax.set_title("AF3-predicted structure accuracy")
    ax.legend(fontsize=5.5 * TEXT_SCALE, loc="upper right", ncol=2, framealpha=0.9)

    # Threshold lines
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.text(n_proteins - 0.5, 1.05, "1 Å", fontsize=5 * TEXT_SCALE, color="gray", ha="right")
    ax.text(n_proteins - 0.5, 2.05, "2 Å", fontsize=5 * TEXT_SCALE, color="gray", ha="right")


def make_panel_b(ax, df):
    """Grouped bar chart: pTM across proteins and conditions."""
    n_proteins = len(PROTEIN_ORDER)
    n_conditions = len(CONDITION_ORDER)
    bar_width = 0.18
    x = np.arange(n_proteins)

    for i, cond in enumerate(CONDITION_ORDER):
        ptms = []
        for prot in PROTEIN_ORDER:
            row = df[(df["pdb_id"] == prot) & (df["condition"] == cond)]
            ptms.append(row["ptm"].values[0] if len(row) else 0)

        offset = (i - n_conditions / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, ptms,
            bar_width, color=CONDITION_COLORS[cond],
            edgecolor="white", linewidth=0.4,
            label=CONDITION_LABELS[cond].replace("\n", " "),
        )

        for bar, val in zip(bars, ptms):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=4.5 * TEXT_SCALE, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([PROTEIN_DESCRIPTIONS[p] for p in PROTEIN_ORDER],
                       fontsize=5.5 * TEXT_SCALE)
    ax.set_ylabel("pTM score")
    ax.set_title("AF3 prediction confidence")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(n_proteins - 0.5, 0.51, "0.5 (low confidence)", fontsize=5 * TEXT_SCALE,
            color="gray", ha="right")


def make_panel_c(ax, df):
    """Scatter: RMSD vs pTM, coloured by protein, shaped by condition."""
    markers = {
        "unconstrained": "o",
        "unconstrained_denovo_glycans": "D",
        "full_sequon_fixed": "s",
        "full_sequon_fixed_full_glycans": "P",
        "full_sequon_fixed_with_glycans": "^",
    }

    for _, row in df.iterrows():
        prot = row["pdb_id"]
        cond = row["condition"]
        ax.scatter(
            row["ptm"], row["mean_rmsd"],
            c=PROTEIN_COLORS[prot],
            marker=markers.get(cond, "o"),
            s=80, alpha=0.85,
            edgecolors="white", linewidths=0.5,
            zorder=3,
        )

    # Legends
    from matplotlib.lines import Line2D
    protein_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PROTEIN_COLORS[p],
               markersize=8, label=p)
        for p in PROTEIN_ORDER
    ]
    condition_handles = [
        Line2D([0], [0], marker=markers[c], color="w", markerfacecolor="gray",
               markersize=8, label=CONDITION_LABELS[c].replace("\n", " "))
        for c in CONDITION_ORDER
    ]

    leg1 = ax.legend(handles=protein_handles, loc="upper left",
                     fontsize=5.5 * TEXT_SCALE, title="Protein", title_fontsize=6 * TEXT_SCALE,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=condition_handles, loc="upper right",
              fontsize=5.5 * TEXT_SCALE, title="Condition", title_fontsize=6 * TEXT_SCALE,
              framealpha=0.9)

    ax.set_xlabel("pTM score")
    ax.set_ylabel("RMSD to native (Å)")
    ax.set_title("Confidence vs structural accuracy")

    # Correlation
    valid = df.dropna(subset=["mean_rmsd", "ptm"])
    if len(valid) > 3:
        r, p = stats.spearmanr(valid["ptm"], valid["mean_rmsd"])
        ax.text(
            0.05, 0.05,
            f"Spearman ρ = {r:.2f}\np = {p:.3f}\nn = {len(valid)}",
            transform=ax.transAxes, fontsize=5.5 * TEXT_SCALE,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )


def make_panel_d(ax, df):
    """Heatmap: RMSD by protein x condition."""
    pivot = df.pivot(index="pdb_id", columns="condition", values="mean_rmsd")
    pivot = pivot.reindex(index=PROTEIN_ORDER, columns=CONDITION_ORDER)

    # Use log scale for colour to handle 1C1Z outlier
    import matplotlib.colors as mcolors
    vmin, vmax = 0.3, 6.0
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(PROTEIN_ORDER)):
        for j in range(len(CONDITION_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 2.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5 * TEXT_SCALE, fontweight="bold", color=color)

    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        fontsize=5.5 * TEXT_SCALE,
    )
    ax.set_yticks(range(len(PROTEIN_ORDER)))
    ax.set_yticklabels(PROTEIN_ORDER, fontsize=6.5 * TEXT_SCALE)
    ax.set_title("RMSD summary (Å)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("RMSD (Å)", fontsize=6 * TEXT_SCALE)


def make_supplementary_figure():
    """Assemble the 4-panel supplementary figure."""
    df = pd.read_csv(DATA_CSV)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.95, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                fontsize=14 * TEXT_SCALE, fontweight="bold", va="top")

    print("=" * 60)
    print("GENERATING SUPPLEMENTARY FIGURE: AF3 VALIDATION")
    print("=" * 60)

    print("\n--- Panel A: RMSD ---")
    make_panel_a(ax_a, df)

    print("\n--- Panel B: pTM ---")
    make_panel_b(ax_b, df)

    print("\n--- Panel C: RMSD vs pTM ---")
    make_panel_c(ax_c, df)

    print("\n--- Panel D: Heatmap ---")
    make_panel_d(ax_d, df)

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    for prot in PROTEIN_ORDER:
        prot_df = df[df["pdb_id"] == prot]
        print(f"  {prot}: RMSD range [{prot_df['mean_rmsd'].min():.2f}, {prot_df['mean_rmsd'].max():.2f}], "
              f"pTM range [{prot_df['ptm'].min():.2f}, {prot_df['ptm'].max():.2f}]")

    # Statistical tests
    print("\n--- Statistical Tests ---")
    # Paired comparison: does adding glycans to fixed designs change RMSD?
    fixed_no_gly = df[df["condition"] == "full_sequon_fixed"]["mean_rmsd"].values
    fixed_gly = df[df["condition"] == "full_sequon_fixed_with_glycans"]["mean_rmsd"].values
    if len(fixed_no_gly) == len(fixed_gly):
        stat, p = stats.wilcoxon(fixed_no_gly, fixed_gly)
        direction = "lower" if np.mean(fixed_gly) < np.mean(fixed_no_gly) else "higher"
        print(f"  Fixed+gly vs Fixed RMSD: Wilcoxon p={p:.4f} "
              f"(glycans {direction}: {np.mean(fixed_gly):.3f} vs {np.mean(fixed_no_gly):.3f})")

    unc = df[df["condition"] == "unconstrained"]["mean_rmsd"].values
    unc_gly = df[df["condition"] == "unconstrained_denovo_glycans"]["rmsd"].values
    if len(unc) == len(unc_gly):
        stat, p = stats.wilcoxon(unc, unc_gly)
        print(f"  Uncons+gly vs Uncons RMSD: Wilcoxon p={p:.4f} "
              f"({np.mean(unc_gly):.3f} vs {np.mean(unc):.3f})")

    # Overall: does condition affect RMSD? Kruskal-Wallis
    groups = [df[df["condition"] == c]["mean_rmsd"].dropna().values for c in CONDITION_ORDER]
    stat, p = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis across conditions: H={stat:.3f}, p={p:.4f}")

    # Spearman correlation pTM vs RMSD
    valid = df.dropna(subset=["rmsd", "ptm"])
    r, p = stats.spearmanr(valid["ptm"], valid["mean_rmsd"])
    print(f"  Spearman pTM vs RMSD: ρ={r:.3f}, p={p:.4f}")

    # Save
    out_png = OUT_DIR / "Supplementary_AF3_validation.png"
    out_pdf = OUT_DIR / "Supplementary_AF3_validation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure saved to: {out_png}")
    print(f"PDF saved to: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate supplementary AF3 validation figure."
    )
    parser.add_argument(
        "--csv",
        default=str(DATA_CSV),
        help="Input CSV with AF3 metrics and RMSD values.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for figure files.",
    )
    args = parser.parse_args()

    DATA_CSV = Path(args.csv).resolve()
    OUT_DIR = Path(args.out_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    make_supplementary_figure()
