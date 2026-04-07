#!/usr/bin/env python3
"""
Poster panel figures — harmonised with the SVG layout palette.

Generates:
  1. poster_panel2_conditions.pdf/png  — 3-condition functional retention
  2. poster_panel2_tiers.pdf/png       — Validated vs motif-only vs non-sequon
  3. poster_panel3_cd2.pdf/png         — CD2 schematic (folded vs degraded)
  4. poster_panel1_sameness.pdf/png    — Sameness diagram (conditions x metrics)
  5. poster_panel2_geometry.pdf/png    — Geometry breakdown (secondary structure)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "cross_protein_comparison"
FIG_DIR = BASE / "data" / "figures" / "poster"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# POSTER PALETTE (from SVG)
# ============================================================
C_RED = "#E24B4A"         # danger / broken / unconstrained
C_RED_LIGHT = "#FDF0F0"
C_AMBER = "#BA7517"       # warning / n-only fixed
C_AMBER_LIGHT = "#FDF5E6"
C_GREEN = "#1D9E75"       # good / functional / fixed
C_GREEN_LIGHT = "#E1F5EE"
C_PURPLE = "#5349B7"      # SugarFix / CD2 / validated
C_PURPLE_LIGHT = "#EEEDFE"
C_GRAY = "#5F5E5A"        # neutral / motif-only
C_GRAY_LIGHT = "#F1EFE8"
C_DARK = "#1F1E1D"        # text
C_MID = "#3D3D3A"         # secondary text
C_BORDER = "#5F5E5A"

# Condition colors
C_UNC = C_RED
C_NONLY = C_AMBER
C_FIXED = C_GREEN

# Tier colors
C_VALIDATED = C_PURPLE
C_MOTIF = "#95A5A6"       # warm gray
C_NONSEQ = C_DARK

def poster_style():
    """Apply poster-wide matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_BORDER,
        "axes.labelcolor": C_DARK,
        "xtick.color": C_MID,
        "ytick.color": C_MID,
        "text.color": C_DARK,
    })

def save(fig, name):
    for fmt in ["png", "pdf"]:
        p = FIG_DIR / f"{name}.{fmt}"
        fig.savefig(p)
        print(f"  Saved: {p}")


# ============================================================
# PANEL 2A: 3-Condition Functional Retention
# ============================================================
def panel2_conditions():
    """Bar chart: per-protein functional retention across 3 conditions."""
    pooled = pd.read_csv(DATA / "pooled_retention.csv")

    cond_stats = {}
    for cond, label, color in [
        ("unconstrained", "Unconstrained", C_UNC),
        ("n_only_fixed", "N-only fixed", C_NONLY),
        ("full_sequon_fixed", "Full sequon\nfixed", C_FIXED),
    ]:
        d = pooled[pooled["condition"] == cond]
        if d.empty:
            continue
        by_prot = d.groupby("pdb_id")["functional"].mean() * 100
        cond_stats[label] = {
            "median": by_prot.median(),
            "mean": by_prot.mean(),
            "sem": by_prot.sem(),
            "values": by_prot.values,
            "color": color,
            "n": len(by_prot),
        }

    fig, ax = plt.subplots(figsize=(4.5, 4))

    labels = list(cond_stats.keys())
    x = np.arange(len(labels))

    for i, (label, s) in enumerate(cond_stats.items()):
        # Strip plot (jittered dots)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(s["values"]))
        ax.scatter(i + jitter, s["values"], s=12, color=s["color"],
                   alpha=0.35, edgecolors="none", zorder=2)
        # Box
        bp = ax.boxplot([s["values"]], positions=[i], widths=0.4,
                        patch_artist=True, showfliers=False, zorder=3,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color=s["color"], linewidth=1.2),
                        capprops=dict(color=s["color"], linewidth=1.2))
        bp["boxes"][0].set(facecolor=s["color"], alpha=0.7, edgecolor=s["color"])

        # Median annotation
        ax.text(i, s["median"] + 4, f'{s["median"]:.1f}%',
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=s["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Functional sequon retention (%)")
    ax.set_ylim(-5, 115)
    ax.set_title("Sequon retention across design conditions\n(84 glycoproteins)")

    # Wilcoxon between unc and n-only
    if "Unconstrained" in cond_stats and "N-only fixed" in cond_stats:
        u_vals = cond_stats["Unconstrained"]["values"]
        n_vals = cond_stats["N-only fixed"]["values"]
        min_len = min(len(u_vals), len(n_vals))
        if min_len >= 5:
            stat, pval = stats.wilcoxon(u_vals[:min_len], n_vals[:min_len])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            y_br = 95
            ax.plot([0, 0, 1, 1], [y_br-2, y_br, y_br, y_br-2],
                    color=C_MID, linewidth=1)
            ax.text(0.5, y_br + 1, sig, ha="center", fontsize=11,
                    color=C_MID, fontweight="bold")

    fig.tight_layout()
    save(fig, "poster_panel2_conditions")
    plt.close(fig)


# ============================================================
# PANEL 2B: Validated vs Motif-only vs Non-sequon
# ============================================================
def panel2_tiers():
    """Bar chart: N retention by site class (per-protein means)."""
    pooled = pd.read_csv(DATA / "pooled_retention.csv")
    positions = pd.read_csv(DATA / "all_n_positions_with_rsa.csv")

    unc_pooled = pooled[pooled["condition"] == "unconstrained"]
    tier_map = (
        unc_pooled.groupby(["pdb_id", "position_0idx"])["evidence_tier"]
        .first().reset_index()
        .rename(columns={"position_0idx": "position"})
    )

    pos = positions[positions["condition"] == "unconstrained"].copy()
    pos = pos.merge(tier_map, on=["pdb_id", "position"], how="left")

    is_seq = pos["is_sequon"] == True
    pos["site_class"] = "non-sequon"
    pos.loc[is_seq & pos["evidence_tier"].isin(
        ["experimental", "pdb_evidence", "curator_inferred"]), "site_class"] = "validated"
    pos.loc[is_seq & (pos["evidence_tier"] == "motif_only"), "site_class"] = "motif_only"
    pos.loc[is_seq & (pos["site_class"] == "non-sequon"), "site_class"] = "validated"

    groups = {
        "Validated\nsequon N": (pos[pos["site_class"] == "validated"], C_VALIDATED),
        "Motif-only\nsequon N": (pos[pos["site_class"] == "motif_only"], C_MOTIF),
        "Non-sequon\nN": (pos[pos["site_class"] == "non-sequon"], C_NONSEQ),
    }

    fig, ax = plt.subplots(figsize=(4, 4))

    labels = list(groups.keys())
    means = []
    sems = []
    colors = []
    ns = []

    for label, (data, color) in groups.items():
        by_prot = data.groupby("pdb_id")["n_retention_pct"].mean()
        means.append(by_prot.mean())
        sems.append(by_prot.sem())
        colors.append(color)
        ns.append(len(by_prot))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.6, zorder=3)

    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + sems[i] + 2.5, f"{m:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color=colors[i])
        ax.text(i, -4, f"n={n}", ha="center", fontsize=8, color=C_MID)

    # Significance bracket: validated vs non-sequon
    val_data = pos[pos["site_class"] == "validated"].groupby("pdb_id")["n_retention_pct"].mean()
    nonseq_data = pos[pos["site_class"] == "non-sequon"].groupby("pdb_id")["n_retention_pct"].mean()
    stat, pval = stats.mannwhitneyu(val_data, nonseq_data)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    y_br = max(means) + max(sems) + 8
    ax.plot([0, 0, 2, 2], [y_br-1, y_br, y_br, y_br-1], color=C_MID, linewidth=1)
    ax.text(1, y_br + 1, sig, ha="center", fontsize=11, color=C_MID, fontweight="bold")

    # ns bracket: validated vs motif-only
    ax.text(0.5, means[0] + sems[0] + 9, "ns (p=0.52)",
            ha="center", fontsize=8, color=C_MID, style="italic")
    ax.plot([0, 0, 1, 1],
            [means[0] + sems[0] + 5, means[0] + sems[0] + 7,
             means[0] + sems[0] + 7, means[0] + sems[0] + 5],
            color=C_MID, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean N retention (%)\n(per-protein means)")
    ax.set_ylim(-8, y_br + 12)
    ax.set_title("N retention by site class")

    fig.tight_layout()
    save(fig, "poster_panel2_tiers")
    plt.close(fig)


# ============================================================
# PANEL 2C: Geometry breakdown (secondary structure)
# ============================================================
def panel2_geometry():
    """Grouped bar: N retention by secondary structure, split by tier."""
    pooled = pd.read_csv(DATA / "pooled_retention.csv")
    positions = pd.read_csv(DATA / "all_n_positions_with_rsa.csv")

    unc_pooled = pooled[pooled["condition"] == "unconstrained"]
    tier_map = (
        unc_pooled.groupby(["pdb_id", "position_0idx"])["evidence_tier"]
        .first().reset_index()
        .rename(columns={"position_0idx": "position"})
    )

    pos = positions[positions["condition"] == "unconstrained"].copy()
    pos = pos.merge(tier_map, on=["pdb_id", "position"], how="left")

    is_seq = pos["is_sequon"] == True
    pos["site_class"] = "non-sequon"
    pos.loc[is_seq & pos["evidence_tier"].isin(
        ["experimental", "pdb_evidence", "curator_inferred"]), "site_class"] = "validated"
    pos.loc[is_seq & (pos["evidence_tier"] == "motif_only"), "site_class"] = "motif_only"
    pos.loc[is_seq & (pos["site_class"] == "non-sequon"), "site_class"] = "validated"

    if "ss" not in pos.columns:
        print("  Skipping geometry panel — no 'ss' column in data")
        return

    ss_categories = ["Helix", "Sheet", "Coil"]
    site_classes = [
        ("Validated", "validated", C_VALIDATED),
        ("Motif-only", "motif_only", C_MOTIF),
        ("Non-sequon", "non-sequon", C_NONSEQ),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    n_ss = len(ss_categories)
    n_cls = len(site_classes)
    width = 0.22
    x = np.arange(n_ss)

    for j, (cls_label, cls_key, color) in enumerate(site_classes):
        means = []
        sems = []
        for ss in ss_categories:
            subset = pos[(pos["site_class"] == cls_key) & (pos["ss"] == ss)]
            by_prot = subset.groupby("pdb_id")["n_retention_pct"].mean()
            means.append(by_prot.mean() if len(by_prot) > 0 else 0)
            sems.append(by_prot.sem() if len(by_prot) > 1 else 0)

        offset = (j - 1) * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=sems, capsize=3,
                      color=color, edgecolor="white", linewidth=0.8,
                      label=cls_label, alpha=0.85, zorder=3)

        for i, m in enumerate(means):
            if m > 0:
                ax.text(x[i] + offset, m + sems[i] + 1.5, f"{m:.0f}%",
                        ha="center", fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ss_categories)
    ax.set_ylabel("Mean N retention (%)")
    ax.set_title("N retention by secondary structure")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_ylim(0, 65)

    fig.tight_layout()
    save(fig, "poster_panel2_geometry")
    plt.close(fig)


# ============================================================
# PANEL 3: CD2 Schematic
# ============================================================
def panel3_cd2():
    """Conceptual schematic: glycan-dependent folding of CD2."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # --- LEFT: Wild-type with glycan (folds) ---
    # Protein body
    body_l = FancyBboxPatch((0.5, 1.8), 3.5, 3.2, boxstyle="round,pad=0.15",
                             facecolor=C_GREEN_LIGHT, edgecolor=C_GREEN,
                             linewidth=1.5, zorder=2)
    ax.add_patch(body_l)

    # Glycan tree (branching Y shape)
    # Stem
    ax.plot([2.25, 2.25], [4.0, 5.0], color=C_PURPLE, linewidth=3, zorder=3)
    # Branches
    ax.plot([2.25, 1.5], [5.0, 5.8], color=C_PURPLE, linewidth=2.5, zorder=3)
    ax.plot([2.25, 3.0], [5.0, 5.8], color=C_PURPLE, linewidth=2.5, zorder=3)
    # Sugar circles
    for pos in [(2.25, 4.2), (2.25, 4.6), (2.25, 5.0),
                (1.85, 5.4), (2.65, 5.4),
                (1.5, 5.8), (3.0, 5.8)]:
        ax.plot(*pos, 'o', color=C_PURPLE, markersize=7, zorder=4)

    ax.text(2.25, 6.3, "N-glycan", ha="center", fontsize=10,
            color=C_PURPLE, fontweight="bold")

    # N65 label
    ax.text(2.25, 3.8, "N65", ha="center", fontsize=10,
            color=C_GREEN, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=C_GREEN, linewidth=0.8))

    # "Folded" label
    ax.text(2.25, 1.2, "Folded", ha="center", fontsize=12,
            color=C_GREEN, fontweight="bold")
    ax.text(2.25, 0.7, "Wild-type CD2", ha="center", fontsize=9,
            color=C_MID)

    # --- RIGHT: MPNN design without glycan (cannot fold) ---
    # Protein body (partially unfolded — wavy outline)
    body_r = FancyBboxPatch((6.0, 1.8), 3.5, 3.2, boxstyle="round,pad=0.15",
                             facecolor=C_RED_LIGHT, edgecolor=C_RED,
                             linewidth=1.5, linestyle="--", zorder=2)
    ax.add_patch(body_r)

    # Unfolded squiggle lines inside
    for y_off in [2.3, 2.9, 3.5, 4.1]:
        xs = np.linspace(6.3, 9.2, 40)
        ys = y_off + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 40))
        ax.plot(xs, ys, color=C_RED, alpha=0.25, linewidth=1.5, zorder=3)

    # D65 label (mutated)
    ax.text(7.75, 3.8, "D65", ha="center", fontsize=10,
            color=C_RED, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=C_RED, linewidth=0.8))

    # X over glycan attachment
    ax.plot([7.4, 8.1], [4.8, 5.5], color=C_RED, linewidth=2.5, zorder=4)
    ax.plot([7.4, 8.1], [5.5, 4.8], color=C_RED, linewidth=2.5, zorder=4)
    ax.text(7.75, 6.0, "No glycan", ha="center", fontsize=10,
            color=C_RED, fontweight="bold")

    # "Cannot fold" label
    ax.text(7.75, 1.2, "ER degradation", ha="center", fontsize=12,
            color=C_RED, fontweight="bold")
    ax.text(7.75, 0.7, "MPNN design (98% N65\u2192D)", ha="center", fontsize=9,
            color=C_MID)

    # --- CENTER: Arrow ---
    ax.annotate("", xy=(5.7, 3.4), xytext=(4.3, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=C_MID,
                                lw=2, mutation_scale=15))
    ax.text(5.0, 3.9, "MPNN\nredesign", ha="center", fontsize=8,
            color=C_MID, style="italic")

    # --- BOTTOM: Pipeline verdict boxes ---
    box_g = FancyBboxPatch((0.5, -0.3), 3.5, 0.8, boxstyle="round,pad=0.1",
                            facecolor=C_GREEN_LIGHT, edgecolor=C_GREEN,
                            linewidth=1.2, zorder=2)
    ax.add_patch(box_g)
    ax.text(2.25, 0.1, "Pipeline: RMSD 1.4 A \u2192 \"pass\"",
            ha="center", fontsize=8.5, color=C_GREEN, fontweight="bold", zorder=5)

    box_r = FancyBboxPatch((6.0, -0.3), 3.5, 0.8, boxstyle="round,pad=0.1",
                            facecolor=C_RED_LIGHT, edgecolor=C_RED,
                            linewidth=1.2, zorder=2)
    ax.add_patch(box_r)
    ax.text(7.75, 0.1, "Reality: cannot fold without glycan",
            ha="center", fontsize=8.5, color=C_RED, fontweight="bold", zorder=5)

    ax.set_ylim(-0.6, 6.8)

    fig.tight_layout()
    save(fig, "poster_panel3_cd2")
    plt.close(fig)


# ============================================================
# PANEL 1: Sameness Diagram (conditions x metrics grid)
# ============================================================
def panel1_sameness():
    """The centrepiece: conditions-vs-metrics grid showing the blind spot."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Column positions and labels
    cols = {
        "Unconstrained": (2.0, C_RED),
        "N-only\nfixed": (4.2, C_AMBER),
        "Full sequon\nfixed": (6.4, C_GREEN),
        "Fixed +\nglycans": (8.6, C_GREEN),
    }

    # Row positions and labels
    rows = {
        "Sequon": 4.0,
        "RMSD": 3.0,
        "Confidence": 2.0,
        "Biology": 1.0,
    }

    # Column headers
    for label, (cx, color) in cols.items():
        ax.text(cx, 5.1, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)

    # Row labels
    for label, ry in rows.items():
        ax.text(0.5, ry, label, ha="right", va="center", fontsize=10,
                fontweight="bold", color=C_MID)

    def cell(cx, ry, text, bg_color, text_color, alpha=0.25):
        w, h = 1.6, 0.65
        rect = FancyBboxPatch((cx - w/2, ry - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=bg_color, alpha=alpha,
                               edgecolor="none", zorder=2)
        ax.add_patch(rect)
        ax.text(cx, ry, text, ha="center", va="center", fontsize=9,
                color=text_color, zorder=3)

    def span_cell(cx_start, cx_end, ry, text, bg_color, text_color, alpha=0.15):
        w = cx_end - cx_start + 1.6
        h = 0.65
        rect = FancyBboxPatch((cx_start - 0.8, ry - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=bg_color, alpha=alpha,
                               edgecolor="none", zorder=2)
        ax.add_patch(rect)
        cx_mid = (cx_start + cx_end) / 2
        ax.text(cx_mid, ry, text, ha="center", va="center", fontsize=9,
                color=text_color, zorder=3)

    # Row 1: Sequon (diverges)
    cell(2.0, 4.0, "4% kept", C_RED, C_RED, 0.2)
    cell(4.2, 4.0, "54% kept", C_AMBER, C_AMBER, 0.2)
    cell(6.4, 4.0, "100%", C_GREEN, C_GREEN, 0.2)
    cell(8.6, 4.0, "100%", C_GREEN, C_GREEN, 0.2)

    # Row 2: RMSD (all same)
    span_cell(2.0, 8.6, 3.0,
              "All < 2 A  \u2014  no significant difference (p = 0.42)",
              C_GREEN, C_GREEN, 0.1)

    # Row 3: Confidence (all same)
    span_cell(2.0, 8.6, 2.0,
              "All high confidence  \u2014  no significant difference",
              C_GREEN, C_GREEN, 0.1)

    # Row 4: Biology (diverges)
    cell(2.0, 1.0, "Broken", C_RED, C_RED, 0.2)
    cell(4.2, 1.0, "At risk", C_AMBER, C_AMBER, 0.2)
    cell(6.4, 1.0, "Functional", C_GREEN, C_GREEN, 0.2)
    cell(8.6, 1.0, "Functional", C_GREEN, C_GREEN, 0.2)

    # Annotation
    ax.text(5.3, 0.2,
            "Rows 2\u20133 are identical across conditions \u2014 that is the blind spot",
            ha="center", fontsize=10, fontweight="bold", color=C_RED,
            style="italic")

    # Curly brace or bracket highlighting rows 2-3
    ax.annotate("", xy=(0.65, 2.55), xytext=(0.65, 3.45),
                arrowprops=dict(arrowstyle="-", color=C_RED,
                                lw=1.5, connectionstyle="arc3,rad=0.3"))
    ax.text(0.45, 2.5, "}", ha="center", va="center", fontsize=28,
            color=C_RED, fontweight="bold", rotation=0,
            fontfamily="serif")

    fig.tight_layout()
    save(fig, "poster_panel1_sameness")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    poster_style()

    print("Panel 1: Sameness diagram...")
    panel1_sameness()

    print("Panel 2a: Condition strip plot...")
    panel2_conditions()

    print("Panel 2b: Evidence tier bars...")
    panel2_tiers()

    print("Panel 2c: Geometry breakdown...")
    panel2_geometry()

    print("Panel 3: CD2 schematic...")
    panel3_cd2()

    print(f"\nAll panels saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
