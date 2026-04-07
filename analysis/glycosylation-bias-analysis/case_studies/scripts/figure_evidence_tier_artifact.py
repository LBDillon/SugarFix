#!/usr/bin/env python3
"""
Supplementary Figure: Evidence tier enrichment artifact explanation.

Shows how homo-oligomeric row inflation created a spurious ~3x difference
between validated and motif-only functional retention, and how per-site /
per-protein aggregation removes it.

Panels:
  A) Rows per site by evidence tier, colored by chain count — shows that
     multi-chain proteins dominate differently in each tier
  B) Top proteins by row contribution, split validated vs motif-only,
     annotated with chain count and functional retention
  C) Pooled vs per-site vs per-protein functional retention side by side
  D) Per-protein paired scatter: validated vs motif-only N retention
     (35 proteins with both), reproducing Panel A of Figure 1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "cross_protein_comparison"
FIG_DIR = BASE / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALSO_SAVE = Path(__file__).resolve().parent.parent.parent.parent / "figures"
ALSO_SAVE.mkdir(parents=True, exist_ok=True)

# ============================================================
# STYLE
# ============================================================
TEXT_SCALE = 1.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": int(round(8 * TEXT_SCALE)),
    "axes.titlesize": int(round(9 * TEXT_SCALE)),
    "axes.labelsize": int(round(8 * TEXT_SCALE)),
    "xtick.labelsize": int(round(7 * TEXT_SCALE)),
    "ytick.labelsize": int(round(7 * TEXT_SCALE)),
    "legend.fontsize": int(round(7 * TEXT_SCALE)),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COL_VALIDATED = "#8e44ad"
COL_MOTIF = "#95a5a6"
COL_NONSEQ = "#2c3e50"
COL_POOLED = "#e74c3c"
COL_PERSITE = "#f39c12"
COL_PERPROT = "#27ae60"

# ============================================================
# LOAD DATA
# ============================================================
pooled = pd.read_csv(DATA / "pooled_retention.csv")
positions = pd.read_csv(DATA / "all_n_positions_with_rsa.csv")

unc = pooled[pooled["condition"] == "unconstrained"].copy()

# Classify tiers
unc["tier_group"] = "other"
unc.loc[unc["evidence_tier"].isin(
    ["experimental", "pdb_evidence", "curator_inferred"]), "tier_group"] = "validated"
unc.loc[unc["evidence_tier"] == "motif_only", "tier_group"] = "motif_only"

val = unc[unc["tier_group"] == "validated"]
mot = unc[unc["tier_group"] == "motif_only"]

# ============================================================
# PANEL A: Row count distribution by tier, showing chain inflation
# ============================================================
def make_panel_a(ax):
    """Rows per site, split by tier group."""
    for tier, data, color, label in [
        ("validated", val, COL_VALIDATED, "Validated"),
        ("motif_only", mot, COL_MOTIF, "Motif-only"),
    ]:
        site_rows = data.groupby(["pdb_id", "chain", "position_0idx"]).size()
        bins = [0, 8, 16, 32, 64, 128, 260]
        labels_b = ["1-8", "9-16", "17-32", "33-64", "65-128", "129-256"]
        binned = pd.cut(site_rows, bins=bins, labels=labels_b)
        counts = binned.value_counts().reindex(labels_b, fill_value=0)
        pcts = counts / counts.sum() * 100
        x = np.arange(len(labels_b))
        width = 0.35
        offset = -0.18 if tier == "validated" else 0.18
        ax.bar(x + offset, pcts.values, width, color=color, label=label,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(np.arange(len(labels_b)))
    ax.set_xticklabels(labels_b, rotation=30, ha="right")
    ax.set_xlabel("Rows per site in pooled CSV")
    ax.set_ylabel("% of sites")
    ax.set_title("(A) Design-row inflation\nby evidence tier")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# PANEL B: Top proteins driving each tier's pooled rate
# ============================================================
def make_panel_b(ax):
    """Horizontal bar: top 8 proteins by row count for each tier, annotated."""
    tiers = [("Validated", val, COL_VALIDATED), ("Motif-only", mot, COL_MOTIF)]
    all_bars = []
    for tier_label, data, color in tiers:
        by_prot = data.groupby("pdb_id").agg(
            n_rows=("functional", "count"),
            n_chains=("chain", "nunique"),
            func_rate=("functional", "mean"),
        ).sort_values("n_rows", ascending=False).head(6)
        for pdb_id, row in by_prot.iterrows():
            all_bars.append({
                "label": f"{pdb_id} ({int(row['n_chains'])}ch)",
                "rows": row["n_rows"],
                "func": row["func_rate"] * 100,
                "tier": tier_label,
                "color": color,
            })

    df = pd.DataFrame(all_bars)
    # Sort: validated first, then motif-only, each descending by rows
    val_df = df[df["tier"] == "Validated"].sort_values("rows", ascending=True)
    mot_df = df[df["tier"] == "Motif-only"].sort_values("rows", ascending=True)
    df_sorted = pd.concat([mot_df, val_df])

    y = np.arange(len(df_sorted))
    ax.barh(y, df_sorted["rows"].values, color=df_sorted["color"].values,
            edgecolor="white", linewidth=0.5, height=0.7)

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["rows"] + 10, i, f"{row['func']:.0f}%",
                va="center", fontsize=int(round(6 * TEXT_SCALE)),
                color=row["color"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["label"].values, fontsize=int(round(6 * TEXT_SCALE)))
    ax.set_xlabel("Rows in pooled CSV")
    ax.set_title("(B) Top proteins by row count\n(annotation = func. retention)")

    # Add tier separator
    n_mot = len(mot_df)
    ax.axhline(n_mot - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.5, n_mot - 0.7, "Motif-only",
            ha="center", fontsize=int(round(6 * TEXT_SCALE)), color=COL_MOTIF)
    ax.text(ax.get_xlim()[1] * 0.5, n_mot + 0.2, "Validated",
            ha="center", fontsize=int(round(6 * TEXT_SCALE)), color=COL_VALIDATED)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# PANEL C: Pooled vs per-site vs per-protein rates
# ============================================================
def make_panel_c(ax):
    """Grouped bar chart comparing aggregation methods."""
    # Pooled
    val_pooled = val["functional"].mean() * 100
    mot_pooled = mot["functional"].mean() * 100

    # Per-site means
    val_site = val.groupby(["pdb_id", "chain", "position_0idx"])["functional"].mean()
    mot_site = mot.groupby(["pdb_id", "chain", "position_0idx"])["functional"].mean()
    val_persite = val_site.mean() * 100
    mot_persite = mot_site.mean() * 100

    # Per-protein means
    val_prot = val.groupby("pdb_id")["functional"].mean()
    mot_prot = mot.groupby("pdb_id")["functional"].mean()
    val_perprot = val_prot.mean() * 100
    mot_perprot = mot_prot.mean() * 100

    # Per-protein SEM
    val_perprot_sem = val_prot.sem() * 100
    mot_perprot_sem = mot_prot.sem() * 100

    methods = ["Pooled\n(per-design)", "Per-site\nmeans", "Per-protein\nmeans"]
    val_vals = [val_pooled, val_persite, val_perprot]
    mot_vals = [mot_pooled, mot_persite, mot_perprot]
    val_errs = [0, 0, val_perprot_sem]
    mot_errs = [0, 0, mot_perprot_sem]

    x = np.arange(3)
    w = 0.32
    bars_v = ax.bar(x - w/2, val_vals, w, yerr=val_errs, capsize=4,
                    color=COL_VALIDATED, label="Validated", edgecolor="white")
    bars_m = ax.bar(x + w/2, mot_vals, w, yerr=mot_errs, capsize=4,
                    color=COL_MOTIF, label="Motif-only", edgecolor="white")

    # Annotate values
    for bars in [bars_v, bars_m]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=int(round(6 * TEXT_SCALE)),
                    fontweight="bold")

    # Ratio annotations
    for i, (v, m) in enumerate(zip(val_vals, mot_vals)):
        if m > 0:
            ratio = v / m
            ax.text(i, max(v, m) + 6, f"{ratio:.1f}x",
                    ha="center", fontsize=int(round(6.5 * TEXT_SCALE)),
                    color=COL_POOLED if ratio > 2 else "gray", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Functional retention (%)")
    ax.set_title("(C) Aggregation method changes\nthe apparent tier difference")
    ax.set_ylim(0, 35)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw attention to pooled
    ax.annotate("Artifact", xy=(0, max(val_pooled, mot_pooled) + 8),
                fontsize=int(round(7 * TEXT_SCALE)), color=COL_POOLED,
                ha="center", fontweight="bold")


# ============================================================
# PANEL D: Per-protein paired scatter (N retention)
# ============================================================
def make_panel_d(ax):
    """Per-protein validated vs motif-only N retention scatter."""
    pos = positions[positions["condition"] == "unconstrained"].copy()

    # Merge evidence tier from pooled
    tier_map = (
        unc.groupby(["pdb_id", "position_0idx"])["tier_group"]
        .first().reset_index()
        .rename(columns={"position_0idx": "position"})
    )
    pos = pos.merge(tier_map, on=["pdb_id", "position"], how="left")

    # Classify
    is_seq = pos["is_sequon"] == True
    val_pos = pos[is_seq & (pos["tier_group"] == "validated")]
    mot_pos = pos[is_seq & (pos["tier_group"] == "motif_only")]

    val_by_prot = val_pos.groupby("pdb_id")["n_retention_pct"].mean()
    mot_by_prot = mot_pos.groupby("pdb_id")["n_retention_pct"].mean()

    common = sorted(set(val_by_prot.index) & set(mot_by_prot.index))

    if len(common) >= 3:
        x = [mot_by_prot[p] for p in common]
        y = [val_by_prot[p] for p in common]

        ax.scatter(x, y, s=40, color=COL_VALIDATED, alpha=0.7, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)

        stat, pval = stats.wilcoxon([val_by_prot[p] for p in common],
                                     [mot_by_prot[p] for p in common])

        ax.text(0.05, 0.95,
                f"n={len(common)} proteins\nWilcoxon p={pval:.2f}",
                transform=ax.transAxes, va="top",
                fontsize=int(round(7 * TEXT_SCALE)),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Motif-only N retention (%)")
    ax.set_ylabel("Validated N retention (%)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")
    ax.set_title("(D) Per-protein N retention:\nno tier difference")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# COMPOSE FIGURE
# ============================================================
def main():
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    make_panel_a(ax_a)
    make_panel_b(ax_b)
    make_panel_c(ax_c)
    make_panel_d(ax_d)

    fig.suptitle(
        "Supplementary: Evidence tier enrichment is an aggregation artifact",
        fontsize=int(round(11 * TEXT_SCALE)), fontweight="bold", y=0.98
    )

    for fmt in ["png", "pdf"]:
        out = FIG_DIR / f"evidence_tier_artifact.{fmt}"
        fig.savefig(out)
        print(f"Saved: {out}")
        out2 = ALSO_SAVE / f"evidence_tier_artifact.{fmt}"
        fig.savefig(out2)
        print(f"Saved: {out2}")

    plt.close(fig)


if __name__ == "__main__":
    main()
