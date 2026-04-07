#!/usr/bin/env python3
"""
IgG1 Fc Case Study: Glycosylation at Asn297 controls Fc domain conformation.

Compares three IgG1 Fc crystal structures:
  3AVE — glycosylated, "open" Cg2 conformation (reference)
  1L6X — glycosylated, well-cited open-conformation structure
  3S7G — aglycosylated (Borrok et al.), "closed" conformation

Key biology:
  - Asn297 glycan (NST sequon) holds the two Cg2 domains apart
  - Without glycan, Cg2 domains collapse inward ("closed" conformation)
  - This conformation change abolishes FcgR binding and effector function
  - ProteinMPNN sees only backbone geometry -> glycan presence/absence
    changes what MPNN "learns" about this position

Layout (2 rows x 3 columns):
A) Schematic: glycosylated vs aglycosylated Fc conformations
B) MPNN sequon retention: 3AVE vs 3S7G vs 1L6X comparison
C) Per-chain retention detail (3AVE has asymmetric chains)
D) AF3 global RMSD across conditions
E) AF3 confidence metrics (pTM, ranking score)
F) Local glycosite distortion at Asn297
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
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(
    "/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias"
)
PIPELINE_DATA = BASE / "experiments" / "MPNN_to_AF3_analysis" / "case_study_pipeline" / "data"
OUT_DIR = PIPELINE_DATA / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

COL_GLYCO = "#27ae60"
COL_AGLYCO = "#e74c3c"
COL_1L6X = "#3498db"
COL_FIXED = "#2980b9"
COL_NONLY = "#9b59b6"
COL_DENOVO = "#e67e22"

STRUCTURES = {
    "3AVE": {"label": "3AVE\n(glycosylated)", "color": COL_GLYCO, "short": "3AVE"},
    "3S7G": {"label": "3S7G\n(aglycosylated)", "color": COL_AGLYCO, "short": "3S7G"},
    "1L6X": {"label": "1L6X\n(glycosylated)", "color": COL_1L6X, "short": "1L6X"},
}


def load_retention_data():
    """Load retention data for all three structures."""
    data = {}
    for pdb in STRUCTURES:
        ret_path = PIPELINE_DATA / "outputs" / f"output_{pdb}" / "all_conditions_retention.csv"
        data[pdb] = pd.read_csv(ret_path)
    return data


def panel_a_conformation_schematic(ax):
    """Schematic showing glycosylated (open) vs aglycosylated (closed) Fc."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # --- Glycosylated (open) on left ---
    # Two Cg2 domains spread apart
    left_cg2_x, right_cg2_x = 1.5, 3.5
    cg2_y = 2.5
    cg2_w, cg2_h = 1.0, 2.5

    # Hinge at top
    ax.plot([2.0, 2.5, 3.0], [5.5, 5.8, 5.5], "k-", linewidth=2.5)

    # Cg2 domains
    rect_l = plt.Rectangle((left_cg2_x, cg2_y), cg2_w, cg2_h,
                            facecolor="#aed6f1", edgecolor="#2c3e50", linewidth=1.5,
                            zorder=2)
    rect_r = plt.Rectangle((right_cg2_x, cg2_y), cg2_w, cg2_h,
                            facecolor="#aed6f1", edgecolor="#2c3e50", linewidth=1.5,
                            zorder=2)
    ax.add_patch(rect_l)
    ax.add_patch(rect_r)

    # Cg3 domains
    rect_l3 = plt.Rectangle((left_cg2_x, 0.5), cg2_w, 1.8,
                             facecolor="#d5dbdb", edgecolor="#2c3e50", linewidth=1.5)
    rect_r3 = plt.Rectangle((right_cg2_x, 0.5), cg2_w, 1.8,
                             facecolor="#d5dbdb", edgecolor="#2c3e50", linewidth=1.5)
    ax.add_patch(rect_l3)
    ax.add_patch(rect_r3)

    # Linkers
    ax.plot([2.0, 2.0], [5.5, cg2_y + cg2_h], color="#2c3e50", linewidth=2)
    ax.plot([3.0, 3.0], [5.5, cg2_y + cg2_h], color="#2c3e50", linewidth=2)

    # Glycans (tree shapes between domains)
    for gx in [2.3, 2.7]:
        ax.plot([gx, gx], [3.2, 4.2], color=COL_GLYCO, linewidth=3, alpha=0.8, zorder=3)
        ax.plot(gx, 4.3, "^", color=COL_GLYCO, markersize=10, zorder=3)

    ax.text(2.5, 3.0, "N297", ha="center", fontsize=5.5 * TEXT_SCALE,
            color=COL_GLYCO, fontweight="bold", zorder=4)

    ax.text(2.0, cg2_y + cg2_h / 2, "C\u03b32", ha="center", va="center",
            fontsize=6 * TEXT_SCALE, color="#2c3e50")
    ax.text(3.5, cg2_y + cg2_h / 2 + 0.5, "C\u03b32", ha="center", va="center",
            fontsize=6 * TEXT_SCALE, color="#2c3e50")
    ax.text(2.5, 6.2, "Glycosylated\n(open)", ha="center",
            fontsize=7 * TEXT_SCALE, fontweight="bold", color=COL_GLYCO)

    # --- Aglycosylated (closed) on right ---
    mid_x = 7.5
    # Two Cg2 domains collapsed together
    ax.plot([mid_x - 0.5, mid_x, mid_x + 0.5], [5.5, 5.8, 5.5], "k-", linewidth=2.5)

    rect_l2 = plt.Rectangle((mid_x - 0.8, cg2_y), cg2_w, cg2_h,
                             facecolor="#f5b7b1", edgecolor="#2c3e50", linewidth=1.5,
                             zorder=2)
    rect_r2 = plt.Rectangle((mid_x - 0.2, cg2_y), cg2_w, cg2_h,
                             facecolor="#f5b7b1", edgecolor="#2c3e50", linewidth=1.5,
                             zorder=2)
    ax.add_patch(rect_l2)
    ax.add_patch(rect_r2)

    rect_l32 = plt.Rectangle((mid_x - 1.0, 0.5), cg2_w, 1.8,
                              facecolor="#d5dbdb", edgecolor="#2c3e50", linewidth=1.5)
    rect_r32 = plt.Rectangle((mid_x + 0.0, 0.5), cg2_w, 1.8,
                              facecolor="#d5dbdb", edgecolor="#2c3e50", linewidth=1.5)
    ax.add_patch(rect_l32)
    ax.add_patch(rect_r32)

    ax.plot([mid_x - 0.5, mid_x - 0.3], [5.5, cg2_y + cg2_h], color="#2c3e50", linewidth=2)
    ax.plot([mid_x + 0.5, mid_x + 0.3], [5.5, cg2_y + cg2_h], color="#2c3e50", linewidth=2)

    # X marks where glycan was
    ax.plot(mid_x, 3.7, "x", color=COL_AGLYCO, markersize=12, markeredgewidth=3, zorder=3)
    ax.text(mid_x, 3.0, "no glycan", ha="center", fontsize=5 * TEXT_SCALE,
            color=COL_AGLYCO, fontstyle="italic")

    ax.text(mid_x, 6.2, "Aglycosylated\n(closed)", ha="center",
            fontsize=7 * TEXT_SCALE, fontweight="bold", color=COL_AGLYCO)

    # Arrow between
    ax.annotate("", xy=(5.5, 4.0), xytext=(4.5, 4.0),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))
    ax.text(5.0, 4.5, "remove\nglycan", ha="center", fontsize=5.5 * TEXT_SCALE,
            color="#7f8c8d")

    ax.text(5.0, 0.1, "Loss of Fc\u03b3R binding\n& effector function",
            ha="center", fontsize=6 * TEXT_SCALE, fontstyle="italic",
            color="#7f8c8d")

    ax.set_title("IgG1 Fc: glycan controls\nC\u03b32 domain conformation",
                 fontsize=9 * TEXT_SCALE)


def panel_b_retention_comparison(ax):
    """Compare Asn297 (NST) retention across the three structures."""
    ret_data = load_retention_data()

    # Per-structure mean retention (average across chains)
    results = []
    for pdb, info in STRUCTURES.items():
        df = ret_data[pdb]
        for cond in ["unconstrained", "n_only_fixed", "full_sequon_fixed"]:
            c = df[df["condition"] == cond]
            if len(c) == 0:
                continue
            n_ret = c["n_retained"].mean() * 100
            func_ret = c["functional"].mean() * 100
            results.append({
                "pdb": pdb, "condition": cond,
                "n_retention": n_ret, "functional": func_ret,
                "color": info["color"],
            })

    results_df = pd.DataFrame(results)

    # Bar chart: unconstrained N retention by structure
    x = np.arange(3)
    width = 0.25
    cond_labels = ["Unconstrained", "N-only fixed", "Full sequon fixed"]
    cond_keys = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]
    cond_colors = ["#e74c3c", "#9b59b6", "#2980b9"]

    for i, (cond, clabel, ccol) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
        vals = []
        for pdb in STRUCTURES:
            row = results_df[(results_df["pdb"] == pdb) & (results_df["condition"] == cond)]
            vals.append(row["functional"].values[0] if len(row) > 0 else 0)

        bars = ax.bar(x + i * width, vals, width, color=ccol, alpha=0.85,
                      edgecolor="white", linewidth=0.6, label=clabel)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", fontsize=5.5 * TEXT_SCALE, rotation=0)

    ax.set_xticks(x + width)
    ax.set_xticklabels([STRUCTURES[p]["short"] for p in STRUCTURES],
                       fontsize=7 * TEXT_SCALE)
    ax.set_ylabel("Functional sequon\nretention (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Sequon retention at Asn297\nacross IgG1 Fc structures")
    ax.legend(fontsize=5.5 * TEXT_SCALE, loc="upper left", framealpha=0.9)


def load_af3_data():
    """Load AF3 validation data for IgG structures."""
    analysis_dir = PIPELINE_DATA / "af3_results" / "analysis"

    # Confidence + RMSD from combined CSV
    combined = pd.read_csv(analysis_dir / "af3_validation_combined.csv")
    igg = combined[combined["pdb_id"].isin(STRUCTURES.keys())].copy()

    # Local glycosite metrics
    local = pd.read_csv(analysis_dir / "local_glycosite_metrics.csv")
    local_igg = local[local["pdb_id"].isin(STRUCTURES.keys())].copy()

    return igg, local_igg


def panel_c_per_chain_detail(ax):
    """Per-chain retention showing asymmetry in 3AVE."""
    ret_data = load_retention_data()

    entries = []
    for pdb in ["3AVE", "3S7G", "1L6X"]:
        df = ret_data[pdb]
        unc = df[df["condition"] == "unconstrained"]
        for (ch, pos), grp in unc.groupby(["chain", "position_0idx"]):
            n_ret_pct = grp["n_retained"].mean() * 100
            func_pct = grp["functional"].mean() * 100
            entries.append({
                "label": f"{pdb}:{ch}",
                "pdb": pdb,
                "n_retention": n_ret_pct,
                "functional": func_pct,
                "color": STRUCTURES[pdb]["color"],
            })

    entries_df = pd.DataFrame(entries)
    y = np.arange(len(entries_df))

    ax.barh(y, entries_df["n_retention"], color=entries_df["color"],
            edgecolor="white", linewidth=0.8, height=0.6, alpha=0.85,
            label="N retained")

    for i, row in entries_df.iterrows():
        ax.text(row["n_retention"] + 1.5, i, f"{row['n_retention']:.1f}%",
                va="center", fontsize=6 * TEXT_SCALE)

    ax.set_yticks(y)
    ax.set_yticklabels(entries_df["label"], fontsize=6.5 * TEXT_SCALE)
    ax.set_xlabel("N retention (%)")
    ax.set_title("Per-chain Asn297 retention\n(unconstrained)")
    ax.set_xlim(0, 80)

    ax.text(0.98, 0.95,
            "3S7G: 0% across all\n4 chains (closed\nconformation)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5 * TEXT_SCALE, color=COL_AGLYCO,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec",
                      edgecolor=COL_AGLYCO, alpha=0.9))

    ax.text(0.98, 0.55,
            "3AVE: 55% chain A\nvs 14% chain B\n(asymmetric crystal\ncontacts?)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5 * TEXT_SCALE, color=COL_GLYCO,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor=COL_GLYCO, alpha=0.9))


def panel_d_af3_rmsd(ax):
    """AF3 global RMSD across conditions — designs fold well despite sequon loss."""
    igg, local_igg = load_af3_data()

    # Use median global_rmsd per structure/condition from local glycosite data
    # (since af3_validation_combined has NaN RMSD for IgG)
    rmsd_data = local_igg.groupby(["pdb_id", "condition"])["global_rmsd"].median().reset_index()

    cond_keys = ["unconstrained", "full_sequon_fixed", "full_sequon_fixed_with_glycans"]
    cond_labels = ["Unconstrained", "Full sequon fixed", "Fixed + glycans"]
    cond_colors = ["#e74c3c", "#2980b9", "#27ae60"]

    x = np.arange(len(STRUCTURES))
    width = 0.25

    for i, (cond, clabel, ccol) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
        vals = []
        for pdb in STRUCTURES:
            row = rmsd_data[(rmsd_data["pdb_id"] == pdb) & (rmsd_data["condition"] == cond)]
            vals.append(row["global_rmsd"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, color=ccol, alpha=0.85,
                      edgecolor="white", linewidth=0.6, label=clabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{v:.2f}", ha="center", fontsize=5 * TEXT_SCALE, rotation=0)

    ax.set_xticks(x + width)
    ax.set_xticklabels([STRUCTURES[p]["short"] for p in STRUCTURES],
                       fontsize=7 * TEXT_SCALE)
    ax.set_ylabel("Median global RMSD (\u00c5)")
    ax.set_title("AF3 structural validation\n(RMSD vs crystal)")
    ax.legend(fontsize=5.5 * TEXT_SCALE, loc="upper right", framealpha=0.9)

    # Reference line at 2A
    ax.axhline(y=2.0, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.02, 0.92, "2\u00c5 threshold",
            transform=ax.transAxes, ha="left", fontsize=5.5 * TEXT_SCALE,
            color="#95a5a6", fontstyle="italic")
    ax.set_ylim(0, 2.5)

    ax.text(0.5, 0.05,
            "All conditions < 2\u00c5 \u2192 pipeline doesn't flag glycan loss",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5.5 * TEXT_SCALE, fontstyle="italic", color="#c0392b",
            fontweight="bold")


def panel_e_af3_confidence(ax):
    """AF3 confidence metrics — pTM and ranking score side by side."""
    igg, _ = load_af3_data()

    cond_keys = ["unconstrained", "full_sequon_fixed", "full_sequon_fixed_with_glycans"]
    cond_labels = ["Uncons.", "Fixed", "Fixed+gly"]
    cond_colors = ["#e74c3c", "#2980b9", "#27ae60"]

    x = np.arange(len(STRUCTURES))
    width = 0.25

    for i, (cond, clabel, ccol) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
        ptm_vals = []
        rank_vals = []
        for pdb in STRUCTURES:
            row = igg[(igg["pdb_id"] == pdb) & (igg["condition"] == cond)]
            ptm_vals.append(row["ptm"].values[0] if len(row) > 0 else 0)
            rank_vals.append(row["ranking_score"].values[0] if len(row) > 0 else 0)

        pos = x + i * width
        ax.bar(pos, ptm_vals, width, color=ccol, alpha=0.75,
               edgecolor="white", linewidth=0.6, label=clabel)
        ax.scatter(pos, rank_vals, marker="D", color="black",
                   s=25, zorder=3, edgecolors="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([STRUCTURES[p]["short"] for p in STRUCTURES],
                       fontsize=7 * TEXT_SCALE)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("AF3 confidence\n(bars=pTM, \u25c6=ranking score)")
    ax.legend(fontsize=5.5 * TEXT_SCALE, loc="upper right", framealpha=0.9)

    # Highlight 3S7G lower confidence
    ax.text(0.02, 0.05,
            "3S7G: lower confidence\n(4-chain tetramer)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5 * TEXT_SCALE, fontstyle="italic", color=COL_AGLYCO)


def panel_f_local_glycosite(ax):
    """Local RMSD around Asn297 across structures and conditions."""
    _, local_igg = load_af3_data()

    # Median local RMSD per structure/condition (reduces outlier effect)
    local_agg = local_igg.groupby(["pdb_id", "condition"]).agg(
        local_rmsd=("local_rmsd_8A", "median"),
        sasa_af3=("sasa_af3", "mean"),
        sasa_crystal=("sasa_crystal", "mean"),
    ).reset_index()

    cond_keys = ["unconstrained", "full_sequon_fixed", "full_sequon_fixed_with_glycans"]
    cond_labels = ["Unconstrained", "Fixed", "Fixed + glycans"]
    cond_colors = ["#e74c3c", "#2980b9", "#27ae60"]

    x = np.arange(len(STRUCTURES))
    width = 0.25

    for i, (cond, clabel, ccol) in enumerate(zip(cond_keys, cond_labels, cond_colors)):
        vals = []
        for pdb in STRUCTURES:
            row = local_agg[(local_agg["pdb_id"] == pdb) & (local_agg["condition"] == cond)]
            vals.append(row["local_rmsd"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, color=ccol, alpha=0.85,
                      edgecolor="white", linewidth=0.6, label=clabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.1f}", ha="center", fontsize=5 * TEXT_SCALE, rotation=0)

    ax.set_xticks(x + width)
    ax.set_xticklabels([STRUCTURES[p]["short"] for p in STRUCTURES],
                       fontsize=7 * TEXT_SCALE)
    ax.set_ylabel("Local RMSD at N297\n(8\u00c5 sphere, \u00c5)")
    ax.set_title("Glycosite distortion\naround Asn297")
    ax.legend(fontsize=5.5 * TEXT_SCALE, loc="upper left", framealpha=0.9)

    ax.text(0.98, 0.95,
            "Fixed+glycans: lower\nlocal RMSD (glycan\nstabilises site)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5 * TEXT_SCALE, fontstyle="italic", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8",
                      edgecolor="#27ae60", alpha=0.9))


def make_figure():
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45,
                           left=0.06, right=0.97, top=0.91, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                          ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.12, 1.10, label, transform=ax.transAxes,
                fontsize=14 * TEXT_SCALE, fontweight="bold", va="top")

    fig.suptitle(
        "IgG1 Fc: Glycan at Asn297 controls domain conformation "
        "- structure-dependent MPNN bias",
        fontsize=10.5 * TEXT_SCALE, fontweight="bold", y=0.97)

    print("Generating IgG Fc case study figure...")

    panel_a_conformation_schematic(ax_a)
    print("  Panel A: conformation schematic")

    panel_b_retention_comparison(ax_b)
    print("  Panel B: retention comparison")

    panel_c_per_chain_detail(ax_c)
    print("  Panel C: per-chain detail")

    panel_d_af3_rmsd(ax_d)
    print("  Panel D: AF3 global RMSD")

    panel_e_af3_confidence(ax_e)
    print("  Panel E: AF3 confidence")

    panel_f_local_glycosite(ax_f)
    print("  Panel F: local glycosite distortion")

    for fmt in ["png", "pdf"]:
        for out_dir in [OUT_DIR, BASE / "experiments" / "figures"]:
            path = out_dir / f"IgG_Fc_case_study.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {path}")

    plt.close()
    print("Done!")


if __name__ == "__main__":
    make_figure()
