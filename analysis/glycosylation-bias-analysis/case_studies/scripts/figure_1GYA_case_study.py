#!/usr/bin/env python3
"""
1GYA (CD2) Case Study Figure: A glycan-dependent folder where the computational
pipeline fails to flag the problem.

Layout (2 rows x 3 columns):
A) Protein schematic: sequence with charge cluster + glycosite highlighted
B) MPNN design outcomes at Asn65: near-universal N->D replacement
C) AF3 RMSD comparison across conditions (bar chart)
D) AF3 confidence (pTM / ranking score) across conditions
E) Local glycosite distortion (local RMSD + SASA change)
F) De novo sequon creation map
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

warnings.filterwarnings("ignore")

# Paths
BASE = Path(
    "/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias"
)
PIPELINE_DATA = BASE / "experiments" / "MPNN_to_AF3_analysis" / "case_study_pipeline" / "data"
CROSS = PIPELINE_DATA / "cross_protein_comparison"
AF3_DIR = PIPELINE_DATA / "af3_results" / "1GYA"
OUT_DIR = PIPELINE_DATA / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
TEXT_SCALE = 1.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": int(round(8 * TEXT_SCALE)),
        "axes.titlesize": int(round(9.5 * TEXT_SCALE)),
        "axes.labelsize": int(round(8 * TEXT_SCALE)),
        "xtick.labelsize": int(round(7 * TEXT_SCALE)),
        "ytick.labelsize": int(round(7 * TEXT_SCALE)),
        "legend.fontsize": int(round(6.5 * TEXT_SCALE)),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Colors
COL_POS = "#e74c3c"       # positive charge (K, R, H)
COL_NEG = "#3498db"       # negative charge (D, E)
COL_GLYCO = "#27ae60"     # glycosite
COL_NEUTRAL = "#ecf0f1"   # neutral residues
COL_FIXED = "#2980b9"
COL_UNCONS = "#e74c3c"
COL_WT = "#7f8c8d"
COL_GLYCAN = "#27ae60"
COL_DENOVO = "#e67e22"

SEQ = "KEITNALETWGALGQDINLDIPSFQMSDDIDDIKWEKTSDKKKIAQFRKEKETFKEKDTYKLFKNGTLKIKHLKTDDQDIYKVSIYDTKGKNVLEKIFDLKIQER"


def panel_a_sequence_map(ax):
    """Show the charge environment around Asn65 with two-row sequence display."""
    # Show residues 55-80 — the critical region around the glycosite
    start, end = 54, 80
    region = SEQ[start:end]
    n = len(region)

    box_w = 1.0
    box_h = 1.2
    fs_aa = 9 * TEXT_SCALE

    for i, aa in enumerate(region):
        pos_1idx = start + i + 1
        if pos_1idx == 65:
            color = COL_GLYCO
            fontweight = "bold"
        elif aa in ("K", "R", "H"):
            color = COL_POS
            fontweight = "bold"
        elif aa in ("D", "E"):
            color = COL_NEG
            fontweight = "normal"
        else:
            color = "#d5dbdb"
            fontweight = "normal"

        rect = plt.Rectangle((i * box_w, 0), box_w * 0.95, box_h,
                              facecolor=color, edgecolor="white",
                              linewidth=0.8, alpha=0.9, zorder=2)
        ax.add_patch(rect)

        text_color = "white" if color in (COL_POS, COL_NEG, COL_GLYCO) else "#2c3e50"
        ax.text(i * box_w + box_w / 2, box_h / 2, aa,
                ha="center", va="center", fontsize=fs_aa,
                fontweight=fontweight, color=text_color, family="monospace", zorder=3)

        # Position labels
        if pos_1idx % 5 == 0 or pos_1idx == 65:
            ax.text(i * box_w + box_w / 2, -0.25, str(pos_1idx),
                    ha="center", va="top", fontsize=5.5 * TEXT_SCALE, color="#7f8c8d")

    # Bracket for sequon NGT (positions 65-67)
    gs = (64 - start) * box_w
    bracket_y = box_h + 0.15
    ax.plot([gs, gs, gs + 3 * box_w, gs + 3 * box_w],
            [bracket_y, bracket_y + 0.15, bracket_y + 0.15, bracket_y],
            color=COL_GLYCO, linewidth=2.0, zorder=4)
    ax.text(gs + 1.5 * box_w, bracket_y + 0.25, "N-G-T sequon",
            ha="center", va="bottom", fontsize=7 * TEXT_SCALE,
            color=COL_GLYCO, fontweight="bold")

    # Charge annotation
    ax.text(n * box_w / 2, -0.85,
            "6 positive charges in 14-residue span (K61, K64, K69, K71, H72, K74)\n"
            "Glycan stabilises fold by counterbalancing this charge cluster",
            ha="center", va="top", fontsize=5.8 * TEXT_SCALE,
            color=COL_POS, fontstyle="italic")

    # Legend
    legend_items = [
        mpatches.Patch(color=COL_POS, label="Positive (K/R/H)"),
        mpatches.Patch(color=COL_NEG, label="Negative (D/E)"),
        mpatches.Patch(color=COL_GLYCO, label="Glycosite (N65)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=5.5 * TEXT_SCALE,
              framealpha=0.9, handlelength=1.0, handletextpad=0.4)

    ax.set_xlim(-0.5, n * box_w + 0.5)
    ax.set_ylim(-1.5, box_h + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("CD2 (1GYA): charge cluster around glycosite Asn65",
                 fontsize=9 * TEXT_SCALE)


def panel_b_mpnn_replacement(ax):
    """Show that MPNN almost universally replaces N65 with D."""
    # Data: 63/64 -> DGT, 1/64 -> NGT
    labels = ["N\u2192D\n(DGT)", "N retained\n(NGT)"]
    counts = [63, 1]
    colors = [COL_UNCONS, COL_GLYCO]

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.0,
                  width=0.55, alpha=0.9)

    for bar, count in zip(bars, counts):
        pct = count / 64 * 100
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
                f"{count}/64\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=7 * TEXT_SCALE, fontweight="bold")

    ax.set_ylabel("Number of designs")
    ax.set_ylim(0, 75)
    ax.set_title("MPNN designs at Asn65\n(unconstrained)")

    # Add note about conservative substitution
    ax.text(0.5, 0.65, "N\u2192D: same geometry,\nabolishes glycosylation",
            transform=ax.transAxes, ha="center", fontsize=5.8 * TEXT_SCALE,
            fontstyle="italic", color="#7f8c8d",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                      edgecolor="#dee2e6", alpha=0.9))


def panel_c_af3_rmsd(ax):
    """AF3 RMSD across all conditions."""
    rmsd_df = pd.read_csv(AF3_DIR / "rmsd_result.csv")
    # Filter to per-chain (not mean aggregates)
    rmsd_df = rmsd_df[rmsd_df["ref_chain"] == "A"]

    conditions = {
        "1GYA_wt": ("WT\nsequence", COL_WT),
        "1GYA_unconstrained": ("Unconstrained\n(N\u2192D)", COL_UNCONS),
        "1GYA_full_sequon_fixed": ("Sequon\nfixed", COL_FIXED),
        "1GYA_full_sequon_fixed_with_glycans": ("Fixed +\nglycans", COL_GLYCAN),
        "1GYA_unconstrained_denovo_glycans": ("Uncons. +\nde novo gly", COL_DENOVO),
    }

    x_pos = []
    heights = []
    colors = []
    labels = []

    for i, (struct_name, (label, color)) in enumerate(conditions.items()):
        row = rmsd_df[rmsd_df["structure"] == struct_name]
        if len(row) > 0:
            x_pos.append(i)
            heights.append(row.iloc[0]["rmsd"])
            colors.append(color)
            labels.append(label)

    bars = ax.bar(x_pos, heights, color=colors, edgecolor="white", linewidth=0.8,
                  width=0.65, alpha=0.9)

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=6.5 * TEXT_SCALE)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=5.8 * TEXT_SCALE)
    ax.set_ylabel("RMSD to crystal (\u00c5)")
    ax.set_title("AF3 global RMSD")
    ax.set_ylim(0, 2.0)

    # Reference line at 2.0
    ax.axhline(y=1.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(len(x_pos) - 0.5, 1.52, "all < 1.5\u00c5", fontsize=5.5 * TEXT_SCALE,
            color="gray", ha="right")


def panel_d_af3_confidence(ax):
    """AF3 pTM and ranking score across conditions."""
    conditions_order = [
        ("wt", "WT", COL_WT),
        ("unconstrained", "Uncons.\n(N\u2192D)", COL_UNCONS),
        ("full_sequon_fixed", "Sequon\nfixed", COL_FIXED),
        ("full_sequon_fixed_with_glycans", "Fixed +\ngly", COL_GLYCAN),
        ("unconstrained_denovo_glycans", "Uncons. +\nde novo", COL_DENOVO),
    ]

    ptms = []
    rankings = []
    labels = []
    colors = []

    for cond, label, color in conditions_order:
        conf_path = AF3_DIR / "confidences" / f"1GYA_{cond}.json"
        with open(conf_path) as f:
            data = json.load(f)
        ptms.append(data.get("ptm", 0))
        rankings.append(data.get("ranking_score", 0))
        labels.append(label)
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ptms, width, color=colors, alpha=0.7,
                   edgecolor="white", linewidth=0.6, label="pTM")
    bars2 = ax.bar(x + width / 2, rankings, width, color=colors, alpha=1.0,
                   edgecolor="white", linewidth=0.6, label="Ranking score")

    # Add hatching to ranking score bars to distinguish
    for bar in bars2:
        bar.set_hatch("//")

    for bar, val in zip(bars1, ptms):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5 * TEXT_SCALE)
    for bar, val in zip(bars2, rankings):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5 * TEXT_SCALE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5 * TEXT_SCALE)
    ax.set_ylabel("Score")
    ax.set_title("AF3 confidence metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6 * TEXT_SCALE, loc="lower right", framealpha=0.9)

    # Highlight: all designs pass quality threshold
    ax.axhline(y=0.7, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.02, 0.71, "all pass\nquality threshold",
            transform=ax.get_yaxis_transform(), fontsize=5 * TEXT_SCALE,
            color="gray", va="bottom")


def panel_e_local_distortion(ax):
    """Local glycosite metrics: local RMSD and SASA."""
    local = pd.read_csv(PIPELINE_DATA / "af3_results" / "analysis" / "local_glycosite_metrics.csv")
    gya = local[local["pdb_id"] == "1GYA"].copy()

    conditions_order = [
        ("full_sequon_fixed", "Sequon fixed", COL_FIXED),
        ("full_sequon_fixed_with_glycans", "Fixed + gly", COL_GLYCAN),
        ("unconstrained", "Uncons. (N\u2192D)", COL_UNCONS),
        ("unconstrained_denovo_glycans", "Uncons. + de novo", COL_DENOVO),
    ]

    labels = []
    local_rmsds = []
    sasas_af3 = []
    colors = []
    crystal_sasa = gya.iloc[0]["sasa_crystal"]

    for cond, label, color in conditions_order:
        row = gya[gya["condition"] == cond]
        if len(row) > 0:
            labels.append(label)
            local_rmsds.append(row.iloc[0]["local_rmsd_8A"])
            sasas_af3.append(row.iloc[0]["sasa_af3"])
            colors.append(color)

    x = np.arange(len(labels))

    # Local RMSD bars
    bars = ax.bar(x, local_rmsds, color=colors, edgecolor="white", linewidth=0.8,
                  width=0.6, alpha=0.9)

    for bar, val in zip(bars, local_rmsds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.05,
                f"{val:.1f}\u00c5", ha="center", va="bottom", fontsize=6 * TEXT_SCALE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5 * TEXT_SCALE, rotation=15, ha="right")
    ax.set_ylabel("Local RMSD 8\u00c5 sphere (\u00c5)")
    ax.set_title("Local distortion at\nglycosite (Asn65)")
    ax.set_ylim(0, 4.0)

    # Add SASA info as secondary annotation
    for i, (sasa, color) in enumerate(zip(sasas_af3, colors)):
        delta_sasa = sasa - crystal_sasa
        sign = "+" if delta_sasa > 0 else ""
        ax.text(i, -0.5, f"SASA: {sasa:.0f}\n({sign}{delta_sasa:.0f})",
                ha="center", fontsize=4.8 * TEXT_SCALE, color="#7f8c8d")

    ax.text(0.5, -0.22, f"Crystal SASA: {crystal_sasa:.0f} \u00c5\u00b2",
            transform=ax.transAxes, ha="center", fontsize=5 * TEXT_SCALE,
            color="#7f8c8d", fontstyle="italic")


def panel_f_denovo_sequons(ax):
    """De novo sequon creation map along the sequence."""
    denovo = pd.read_csv(PIPELINE_DATA / "outputs" / "output_1GYA" / "denovo_sequons.csv")
    unc = denovo[denovo["condition"] == "unconstrained"]

    # Count by position
    pos_counts = unc.groupby("position_0idx").size().reset_index(name="n_designs")
    pos_counts["position_1idx"] = pos_counts["position_0idx"] + 1

    # Total designs = 64
    n_designs = 64
    pos_counts["fraction"] = pos_counts["n_designs"] / n_designs

    # Bar chart along sequence
    ax.bar(pos_counts["position_1idx"], pos_counts["fraction"] * 100,
           color=COL_DENOVO, edgecolor="white", linewidth=0.5, width=1.5, alpha=0.85)

    # Mark the native sequon position
    ax.axvline(x=65, color=COL_GLYCO, linestyle="-", linewidth=2.0, alpha=0.7)
    ax.text(65, max(pos_counts["fraction"] * 100) * 0.95, "native\nAsn65",
            ha="center", fontsize=5.5 * TEXT_SCALE, color=COL_GLYCO, fontweight="bold")

    # Label top de novo positions
    top = pos_counts.nlargest(3, "n_designs")
    for _, row in top.iterrows():
        ax.text(row["position_1idx"], row["fraction"] * 100 + 2,
                f"pos {int(row['position_1idx'])}\n({int(row['n_designs'])}/64)",
                ha="center", fontsize=5 * TEXT_SCALE, color="#2c3e50")

    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Designs with de novo\nsequon (%)")
    ax.set_title("De novo sequon creation\n(unconstrained designs)")
    ax.set_xlim(0, 106)
    ax.set_ylim(0, max(pos_counts["fraction"] * 100) + 15)

    # Note
    n_positions = len(pos_counts)
    ax.text(0.98, 0.95,
            f"{n_positions} positions gain\nde novo sequons\n"
            f"(native N65 lost in\n63/64 designs)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5 * TEXT_SCALE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                      edgecolor="#f0e68c", alpha=0.9))


def make_figure():
    fig = plt.figure(figsize=(16, 10))

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                           left=0.06, right=0.97, top=0.93, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # Panel labels
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                          ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.12, 1.10, label, transform=ax.transAxes,
                fontsize=14 * TEXT_SCALE, fontweight="bold", va="top")

    fig.suptitle("1GYA (CD2): Glycan-dependent folding — a computational blind spot",
                 fontsize=11 * TEXT_SCALE, fontweight="bold", y=0.98)

    print("Generating 1GYA case study figure...")

    panel_a_sequence_map(ax_a)
    print("  Panel A: sequence charge map")

    panel_b_mpnn_replacement(ax_b)
    print("  Panel B: MPNN replacement pattern")

    panel_c_af3_rmsd(ax_c)
    print("  Panel C: AF3 RMSD")

    panel_d_af3_confidence(ax_d)
    print("  Panel D: AF3 confidence")

    panel_e_local_distortion(ax_e)
    print("  Panel E: local glycosite distortion")

    panel_f_denovo_sequons(ax_f)
    print("  Panel F: de novo sequons")

    # Save
    for fmt in ["png", "pdf"]:
        out_path = OUT_DIR / f"1GYA_case_study.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out_path}")

    # Also save to experiments/figures
    exp_dir = BASE / "experiments" / "figures"
    for fmt in ["png", "pdf"]:
        out_path = exp_dir / f"1GYA_case_study.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out_path}")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    make_figure()
