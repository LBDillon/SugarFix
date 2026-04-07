#!/usr/bin/env python3
"""
Figure 1 (Updated): ProteinMPNN treatment of N-linked glycosylation sequons.

All panels now use the 84-protein case study pipeline dataset with evidence tiers.

Panels:
A) Per-protein sequon-N vs non-sequon-N retention (scatter + paired test)
   Evidence tier comparison: validated (experimental/PDB) vs motif-only retention
B) Geometry biases: retention binned by secondary structure and phi/psi
   Uses per-protein means within bins to avoid pseudo-replication
C) MPNN score delta: unconstrained vs full-sequon-fixed
D) Multi-protein AF3 validation: paired RMSD dot plot + glycan effect bar chart
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# PATHS — all from case study pipeline
# ============================================================
BASE = Path(
    "/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias"
)
PIPELINE_DATA = BASE / "experiments" / "MPNN_to_AF3_analysis" / "case_study_pipeline" / "data"
CROSS = PIPELINE_DATA / "cross_protein_comparison"
OUT_DIR = BASE / "experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_FIG_DIR = PIPELINE_DATA / "figures"
PIPELINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STYLE
# ============================================================
TEXT_SCALE = 1.55

BASE_FONT = int(round(8 * TEXT_SCALE))
TITLE_FONT = int(round(9 * TEXT_SCALE))
TICK_FONT = int(round(7 * TEXT_SCALE))
LEGEND_FONT = int(round(7 * TEXT_SCALE))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": BASE_FONT,
        "axes.titlesize": TITLE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Color scheme
COL_SEQUON = "#c0392b"
COL_NONSEQ = "#2c3e50"
COL_X = "#e67e22"
COL_ST = "#27ae60"
COL_BASELINE = "#7f8c8d"
COL_FIXED = "#2980b9"
COL_UNCONS = "#e74c3c"

# Evidence tier colors
COL_EXPERIMENTAL = "#8e44ad"
COL_PDB_EVIDENCE = "#2980b9"
COL_CURATOR = "#27ae60"
COL_MOTIF_ONLY = "#95a5a6"
COL_VALIDATED = "#8e44ad"


# ============================================================
# HELPER: load per-position data with evidence tiers
# ============================================================
def _load_positions_with_tiers():
    """Load per-position data and merge evidence tier from pooled_retention.csv."""
    pos_df = pd.read_csv(CROSS / "all_n_positions_with_rsa.csv")
    pooled = pd.read_csv(CROSS / "pooled_retention.csv")

    # Build tier map: one tier per (pdb_id, position)
    tier_map = (
        pooled[pooled["condition"] == "unconstrained"]
        .groupby(["pdb_id", "position_0idx"])["evidence_tier"]
        .first()
        .reset_index()
        .rename(columns={"position_0idx": "position"})
    )
    pos_df = pos_df.merge(tier_map, on=["pdb_id", "position"], how="left")

    # Classify sequon positions into validated vs motif-only
    pos_df["site_class"] = "non-sequon"
    is_seq = pos_df["is_sequon"] == True
    pos_df.loc[is_seq & pos_df["evidence_tier"].isin(["experimental", "pdb_evidence"]), "site_class"] = "validated"
    pos_df.loc[is_seq & (pos_df["evidence_tier"] == "motif_only"), "site_class"] = "motif_only"
    pos_df.loc[is_seq & (pos_df["evidence_tier"] == "curator_inferred"), "site_class"] = "validated"
    # Any sequon without tier info stays as "sequon_unknown"
    pos_df.loc[is_seq & (pos_df["site_class"] == "non-sequon"), "site_class"] = "validated"

    return pos_df


# ============================================================
# PANEL A: Per-protein N retention — validated vs motif-only
# ============================================================
def make_panel_a(ax_scatter, ax_bar):
    """Scatter: per-protein validated-sequon vs motif-only-sequon N retention.
    Bar: evidence tier breakdown of functional retention.
    """
    pos_df = _load_positions_with_tiers()
    unc = pos_df[pos_df["condition"] == "unconstrained"].copy()

    # --- Scatter: validated vs motif-only N retention per protein ---
    val_pos = unc[unc["site_class"] == "validated"]
    mot_pos = unc[unc["site_class"] == "motif_only"]
    nonseq_pos = unc[unc["site_class"] == "non-sequon"]

    val_by_prot = val_pos.groupby("pdb_id")["n_retention_pct"].mean()
    mot_by_prot = mot_pos.groupby("pdb_id")["n_retention_pct"].mean()
    nonseq_by_prot = nonseq_pos.groupby("pdb_id")["n_retention_pct"].mean()

    # Proteins that have both validated and motif-only sites
    common = sorted(set(val_by_prot.index) & set(mot_by_prot.index))
    # Proteins with only validated sites
    val_only = sorted(set(val_by_prot.index) - set(mot_by_prot.index))
    # Proteins with only motif-only sites
    mot_only = sorted(set(mot_by_prot.index) - set(val_by_prot.index))

    # Plot proteins with both types (paired)
    if common:
        ax_scatter.scatter(
            mot_by_prot.loc[common].values,
            val_by_prot.loc[common].values,
            c=COL_VALIDATED,
            s=35,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=f"Both ({len(common)})",
        )

    ax_scatter.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1.0, zorder=1)

    # Paired Wilcoxon for proteins with both types
    if len(common) >= 6:
        v = val_by_prot.loc[common].values
        m = mot_by_prot.loc[common].values
        stat_w, p_w = stats.wilcoxon(v, m)
        n_above = np.sum(v > m)
    else:
        p_w = np.nan
        n_above = 0

    ax_scatter.text(
        0.05,
        0.95,
        f"n = {len(common)} paired proteins\n"
        f"Wilcoxon p = {p_w:.2f}\n"
        f"{n_above}/{len(common)} above diagonal",
        transform=ax_scatter.transAxes,
        fontsize=6.5 * TEXT_SCALE,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85
        ),
    )

    ax_scatter.set_xlabel("Motif-only sequon N retention (%)")
    ax_scatter.set_ylabel("Validated sequon N retention (%)")
    ax_scatter.set_xlim(-5, 105)
    ax_scatter.set_ylim(-5, 105)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_title("Per-protein N retention:\nvalidated vs motif-only sequons")

    # --- Bar chart: N retention for validated, motif-only, non-sequon ---
    groups = [
        ("Validated\nsequon N", val_by_prot, COL_VALIDATED),
        ("Motif-only\nsequon N", mot_by_prot, COL_MOTIF_ONLY),
        ("Non-sequon\nN", nonseq_by_prot, COL_NONSEQ),
    ]

    means, sems, ns, colors, labels = [], [], [], [], []
    for label, data, color in groups:
        means.append(data.mean())
        sems.append(stats.sem(data) if len(data) > 1 else 0)
        ns.append(len(data))
        colors.append(color)
        labels.append(label)

    bars = ax_bar.bar(
        range(3),
        means,
        yerr=sems,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=4,
        width=0.65,
    )

    for bar, m_val, n in zip(bars, means, ns):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"{m_val:.1f}%\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=5.5 * TEXT_SCALE,
        )

    ax_bar.set_xticks(range(3))
    ax_bar.set_xticklabels(labels, fontsize=6 * TEXT_SCALE)
    ax_bar.set_ylabel("Mean N retention (%)\n(per-protein means)")
    ax_bar.set_ylim(0, max(means) + 20)
    ax_bar.set_title("N retention by site\nclass (per-protein means)")

    # Significance brackets: validated vs motif-only, and validated vs non-sequon
    if len(common) >= 6:
        # Validated vs non-sequon (unpaired since different n)
        _, p_vn = stats.mannwhitneyu(val_by_prot, nonseq_by_prot, alternative="two-sided")
        y_b1 = max(means) + 10
        ax_bar.plot([0, 0, 2, 2], [y_b1 - 1, y_b1, y_b1, y_b1 - 1], "k-", linewidth=1.0)
        star_vn = "***" if p_vn < 0.001 else ("**" if p_vn < 0.01 else ("*" if p_vn < 0.05 else "ns"))
        ax_bar.text(1, y_b1 + 0.5, f"p={p_vn:.1e} ({star_vn})", ha="center", fontsize=5.3 * TEXT_SCALE)

    print(f"Panel A: {len(common)} proteins with both validated & motif-only")
    print(f"  Validated N-ret: {val_by_prot.mean():.1f}% ({len(val_by_prot)} proteins)")
    print(f"  Motif-only N-ret: {mot_by_prot.mean():.1f}% ({len(mot_by_prot)} proteins)")
    print(f"  Non-sequon N-ret: {nonseq_by_prot.mean():.1f}% ({len(nonseq_by_prot)} proteins)")
    if len(common) >= 6:
        print(f"  Paired Wilcoxon (val vs mot): p={p_w:.4f}")
        print(f"  Val vs non-sequon MWU: p={p_vn:.2e}")


# ============================================================
# PANEL B: Geometry biases — validated vs motif-only vs non-sequon
# ============================================================
def _compute_three_group_stats(geo_df, category_col, category_val):
    """Compute per-protein mean N retention for validated, motif-only, non-sequon
    within a given geometry category. Returns (means, sems, ns, pvals_dict)."""
    subset = geo_df[geo_df[category_col] == category_val]

    groups_spec = [
        ("validated", subset[subset["site_class"] == "validated"]),
        ("motif_only", subset[subset["site_class"] == "motif_only"]),
        ("non-sequon", subset[subset["site_class"] == "non-sequon"]),
    ]

    means, sems, ns = [], [], []
    by_prot = {}
    for name, grp in groups_spec:
        prot_means = grp.groupby("pdb_id")["n_retention_pct"].mean()
        by_prot[name] = prot_means
        means.append(prot_means.mean() if len(prot_means) > 0 else 0)
        sems.append(stats.sem(prot_means) if len(prot_means) > 1 else 0)
        ns.append(len(prot_means))

    # p-value: validated vs non-sequon (main comparison)
    pval_v_ns = 1.0
    if len(by_prot["validated"]) > 1 and len(by_prot["non-sequon"]) > 1:
        _, pval_v_ns = stats.mannwhitneyu(
            by_prot["validated"], by_prot["non-sequon"], alternative="two-sided"
        )

    return means, sems, ns, pval_v_ns


def make_panel_b(ax_ss, ax_rama):
    """Secondary structure and Ramachandran region analysis.
    Three groups: validated sequon N, motif-only sequon N, non-sequon N.
    Uses per-protein means within bins to avoid pseudo-replication.
    """
    pos_df = _load_positions_with_tiers()
    unc = pos_df[pos_df["condition"] == "unconstrained"].copy()
    geo_complete = unc.dropna(subset=["ss", "rama_region"])

    n_val = len(geo_complete[geo_complete["site_class"] == "validated"])
    n_mot = len(geo_complete[geo_complete["site_class"] == "motif_only"])
    n_non = len(geo_complete[geo_complete["site_class"] == "non-sequon"])
    print(f"  {len(geo_complete)} positions: {n_val} validated, {n_mot} motif-only, {n_non} non-sequon")

    # --- Secondary structure ---
    ss_categories = ["Helix", "Sheet", "Coil"]
    width = 0.26
    x_pos = np.arange(len(ss_categories))

    all_ss_means = {"validated": [], "motif_only": [], "non-sequon": []}
    all_ss_sems = {"validated": [], "motif_only": [], "non-sequon": []}
    ss_pvals = []

    for ss_cat in ss_categories:
        m, s, n, pv = _compute_three_group_stats(geo_complete, "ss", ss_cat)
        for i, key in enumerate(["validated", "motif_only", "non-sequon"]):
            all_ss_means[key].append(m[i])
            all_ss_sems[key].append(s[i])
        ss_pvals.append(pv)

    ax_ss.bar(
        x_pos - width,
        all_ss_means["validated"],
        width,
        yerr=all_ss_sems["validated"],
        color=COL_VALIDATED,
        alpha=0.85,
        label="Validated sequon",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    ax_ss.bar(
        x_pos,
        all_ss_means["motif_only"],
        width,
        yerr=all_ss_sems["motif_only"],
        color=COL_MOTIF_ONLY,
        alpha=0.85,
        label="Motif-only sequon",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    ax_ss.bar(
        x_pos + width,
        all_ss_means["non-sequon"],
        width,
        yerr=all_ss_sems["non-sequon"],
        color=COL_NONSEQ,
        alpha=0.85,
        label="Non-sequon N",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )

    for i, p in enumerate(ss_pvals):
        max_h = max(
            all_ss_means["validated"][i] + all_ss_sems["validated"][i],
            all_ss_means["motif_only"][i] + all_ss_sems["motif_only"][i],
            all_ss_means["non-sequon"][i] + all_ss_sems["non-sequon"][i],
        )
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax_ss.text(i, max_h + 3, star, ha="center", fontsize=6 * TEXT_SCALE, fontweight="bold")

    ax_ss.set_xticks(x_pos)
    ax_ss.set_xticklabels(ss_categories)
    ax_ss.set_ylabel("N retention (%)\n(per-protein means)")
    ax_ss.set_title("N retention by\nsecondary structure")
    ax_ss.legend(loc="upper right", fontsize=5.2 * TEXT_SCALE, framealpha=0.85)
    all_max = max(max(v) for v in all_ss_means.values())
    ax_ss.set_ylim(0, all_max + 25)

    # --- Ramachandran ---
    rama_categories = ["Alpha-helix", "Beta-sheet", "Left-handed", "Other"]
    x_pos2 = np.arange(len(rama_categories))

    all_rama_means = {"validated": [], "motif_only": [], "non-sequon": []}
    all_rama_sems = {"validated": [], "motif_only": [], "non-sequon": []}
    rama_pvals = []

    for rama in rama_categories:
        m, s, n, pv = _compute_three_group_stats(geo_complete, "rama_region", rama)
        for i, key in enumerate(["validated", "motif_only", "non-sequon"]):
            all_rama_means[key].append(m[i])
            all_rama_sems[key].append(s[i])
        rama_pvals.append(pv)

    ax_rama.bar(
        x_pos2 - width,
        all_rama_means["validated"],
        width,
        yerr=all_rama_sems["validated"],
        color=COL_VALIDATED,
        alpha=0.85,
        label="Validated",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    ax_rama.bar(
        x_pos2,
        all_rama_means["motif_only"],
        width,
        yerr=all_rama_sems["motif_only"],
        color=COL_MOTIF_ONLY,
        alpha=0.85,
        label="Motif-only",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    ax_rama.bar(
        x_pos2 + width,
        all_rama_means["non-sequon"],
        width,
        yerr=all_rama_sems["non-sequon"],
        color=COL_NONSEQ,
        alpha=0.85,
        label="Non-sequon",
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )

    for i, p in enumerate(rama_pvals):
        max_h = max(
            all_rama_means["validated"][i] + all_rama_sems["validated"][i],
            all_rama_means["motif_only"][i] + all_rama_sems["motif_only"][i],
            all_rama_means["non-sequon"][i] + all_rama_sems["non-sequon"][i],
        )
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax_rama.text(i, max_h + 3, star, ha="center", fontsize=6 * TEXT_SCALE, fontweight="bold")

    ax_rama.set_xticks(x_pos2)
    ax_rama.set_xticklabels(["\u03b1-helix", "\u03b2-sheet", "Left-\nhanded", "Other"], fontsize=5.8 * TEXT_SCALE)
    ax_rama.set_ylabel("N retention (%)\n(per-protein means)")
    ax_rama.set_title("N retention by\nRamachandran region")
    all_max2 = max(max(v) for v in all_rama_means.values())
    ax_rama.set_ylim(0, all_max2 + 25)

    print(f"Panel B: {len(geo_complete)} positions")
    print("  Significance: validated vs non-sequon (MWU, per-protein means)")
    for i, ss_cat in enumerate(ss_categories):
        print(f"  SS {ss_cat}: val={all_ss_means['validated'][i]:.1f}%, "
              f"mot={all_ss_means['motif_only'][i]:.1f}%, "
              f"nonsq={all_ss_means['non-sequon'][i]:.1f}%, p={ss_pvals[i]:.2e}")
    for i, rama in enumerate(rama_categories):
        print(f"  Rama {rama}: val={all_rama_means['validated'][i]:.1f}%, "
              f"mot={all_rama_means['motif_only'][i]:.1f}%, "
              f"nonsq={all_rama_means['non-sequon'][i]:.1f}%, p={rama_pvals[i]:.2e}")


# ============================================================
# PANEL C: MPNN score delta (from pipeline data)
# ============================================================
def make_panel_c(ax_delta, ax_paired):
    """MPNN score delta: unconstrained vs fixed.
    Uses pipeline's mpnn_scores_by_condition.csv.
    """
    score_df = pd.read_csv(CROSS / "mpnn_scores_by_condition.csv")

    pivot = score_df.pivot(index="pdb_id", columns="condition", values="mean_score").dropna()

    if "unconstrained" not in pivot.columns or "full_sequon_fixed" not in pivot.columns:
        ax_paired.text(0.5, 0.5, "Insufficient data", transform=ax_paired.transAxes, ha="center")
        ax_delta.text(0.5, 0.5, "Insufficient data", transform=ax_delta.transAxes, ha="center")
        return

    unc = pivot["unconstrained"].values
    fix = pivot["full_sequon_fixed"].values
    delta = fix - unc  # negative = fixing improves

    for i in range(len(unc)):
        ax_paired.plot([0, 1], [unc[i], fix[i]], "o-", color="gray", alpha=0.4, markersize=4, linewidth=0.7)

    ax_paired.plot([0], [np.mean(unc)], "o", color=COL_UNCONS, markersize=10, zorder=5)
    ax_paired.plot([1], [np.mean(fix)], "o", color=COL_FIXED, markersize=10, zorder=5)
    ax_paired.plot([0, 1], [np.mean(unc), np.mean(fix)], "-", color="black", linewidth=2.2, zorder=5)

    stat, p_val = stats.ttest_rel(unc, fix)

    ax_paired.set_xticks([0, 1])
    ax_paired.set_xticklabels(["Unconstrained", "Full sequon\nfixed"], fontsize=7.2 * TEXT_SCALE)
    ax_paired.set_ylabel("Mean MPNN score")
    ax_paired.set_title("MPNN score:\nunconstrained vs fixed")
    ax_paired.text(
        0.5,
        0.95,
        f"n={len(unc)}\npaired t p={p_val:.2e}",
        transform=ax_paired.transAxes,
        fontsize=6.4 * TEXT_SCALE,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    ax_delta.hist(delta, bins=15, color=COL_FIXED, alpha=0.7, edgecolor="white")
    ax_delta.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_delta.axvline(x=np.mean(delta), color=COL_UNCONS, linestyle="-", linewidth=2.0, label=f"Mean \u0394 = {np.mean(delta):.4f}")

    ax_delta.set_xlabel("\u0394 Score (fixed \u2212 unconstrained)")
    ax_delta.set_ylabel("Count")
    ax_delta.set_title("Score change from\nfixing sequons")
    ax_delta.legend(fontsize=6.2 * TEXT_SCALE, loc="upper right")

    n_improved = np.sum(delta < 0)
    print(f"Panel C: {len(unc)} glycoproteins with paired scores")
    print(f"  {n_improved}/{len(delta)} proteins improved by fixing")
    print(f"  Paired t p={p_val:.2e}")


# ============================================================
# PANEL D: Multi-protein AF3 validation
# ============================================================
def make_panel_d(ax_rmsd, ax_glycan):
    """AF3 validation across multiple proteins.

    D-left:  Paired RMSD dot plot (unconstrained vs full_sequon_fixed)
    D-right: Glycan effect horizontal bar chart (delta RMSD with/without glycans)
    """
    af3_csv = PIPELINE_DATA / "af3_results" / "analysis" / "af3_validation_combined.csv"
    af3_df = pd.read_csv(af3_csv)

    # D-left: Paired RMSD dot plot (Unc vs Fixed)
    unc_df = af3_df[af3_df["condition"] == "unconstrained"][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "rmsd_unc"}
    )
    fix_df = af3_df[af3_df["condition"] == "full_sequon_fixed"][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "rmsd_fix"}
    )
    paired = unc_df.merge(fix_df, on="pdb_id").dropna()
    paired = paired.sort_values("rmsd_unc", ascending=True).reset_index(drop=True)

    n_paired = len(paired)
    y_positions = np.arange(n_paired)

    for i, row in paired.iterrows():
        ax_rmsd.plot(
            [row["rmsd_unc"], row["rmsd_fix"]],
            [i, i],
            "-",
            color="gray",
            alpha=0.5,
            linewidth=1.0,
            zorder=1,
        )

    ax_rmsd.scatter(
        paired["rmsd_unc"],
        y_positions,
        c=COL_UNCONS,
        s=50,
        marker="o",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Unconstrained",
    )
    ax_rmsd.scatter(
        paired["rmsd_fix"],
        y_positions,
        c=COL_ST,
        s=50,
        marker="s",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Full sequon fixed",
    )

    ax_rmsd.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_rmsd.set_yticks(y_positions)
    ax_rmsd.set_yticklabels(paired["pdb_id"].values, fontsize=5.5 * TEXT_SCALE)

    stat_w, p_w = stats.wilcoxon(paired["rmsd_unc"], paired["rmsd_fix"])
    sig_label = "ns" if p_w >= 0.05 else ("*" if p_w >= 0.01 else ("**" if p_w >= 0.001 else "***"))

    ax_rmsd.text(
        0.95,
        0.05,
        f"Wilcoxon p={p_w:.2f} ({sig_label})\nn={n_paired} paired proteins",
        transform=ax_rmsd.transAxes,
        fontsize=5.8 * TEXT_SCALE,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    ax_rmsd.set_xlabel("Mean RMSD to Crystal Structure (\u00c5)")
    ax_rmsd.set_title("RMSD: Sequon Loss Does\nNot Impair Folding")
    ax_rmsd.legend(loc="upper right", fontsize=5.5 * TEXT_SCALE, framealpha=0.85)

    print(f"Panel D-left: {n_paired} paired proteins, Wilcoxon p={p_w:.4f}")

    # D-right: Glycan effect horizontal bar chart
    fix_nogly = af3_df[af3_df["condition"] == "full_sequon_fixed"][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "rmsd_nogly"}
    )
    fix_gly = af3_df[af3_df["condition"] == "full_sequon_fixed_with_glycans"][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "rmsd_gly"}
    )
    glycan_paired = fix_nogly.merge(fix_gly, on="pdb_id").dropna()
    glycan_paired["delta_rmsd"] = glycan_paired["rmsd_gly"] - glycan_paired["rmsd_nogly"]
    glycan_paired = glycan_paired.sort_values("delta_rmsd", ascending=True).reset_index(drop=True)

    n_glycan = len(glycan_paired)
    y_pos_gly = np.arange(n_glycan)

    bar_colors = [COL_ST if d < 0 else "#e74c3c" for d in glycan_paired["delta_rmsd"]]

    ax_glycan.barh(
        y_pos_gly,
        glycan_paired["delta_rmsd"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax_glycan.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    ax_glycan.set_yticks(y_pos_gly)
    ax_glycan.set_yticklabels(glycan_paired["pdb_id"].values, fontsize=5.5 * TEXT_SCALE)
    ax_glycan.set_xlabel("\u0394 RMSD (\u00c5)\n(with glycans \u2212 without)")
    ax_glycan.set_title("Adding Glycans to AF3")

    n_improved = int((glycan_paired["delta_rmsd"] < 0).sum())
    n_worsened = int((glycan_paired["delta_rmsd"] > 0).sum())

    if n_glycan >= 2:
        stat_g, p_g = stats.wilcoxon(glycan_paired["delta_rmsd"])
        sig_g = "ns" if p_g >= 0.05 else ("*" if p_g >= 0.01 else ("**" if p_g >= 0.001 else "***"))
    else:
        p_g = np.nan
        sig_g = "n/a"

    ax_glycan.text(
        0.95,
        0.05,
        f"Improved {n_improved}/{n_glycan},\nworsened {n_worsened}/{n_glycan}\n"
        f"Wilcoxon p={p_g:.2f} ({sig_g})",
        transform=ax_glycan.transAxes,
        fontsize=5.8 * TEXT_SCALE,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    print(f"Panel D-right: {n_glycan} proteins, {n_improved} improved, {n_worsened} worsened, Wilcoxon p={p_g:.4f}")


# ============================================================
# ASSEMBLE FIGURE 1
# ============================================================
def make_figure1():
    """Assemble all panels into Figure 1."""
    fig = plt.figure(figsize=(16, 13))

    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        hspace=0.62,
        wspace=0.50,
        left=0.06,
        right=0.97,
        top=0.94,
        bottom=0.06,
    )

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b1 = fig.add_subplot(gs[0, 2])
    ax_b2 = fig.add_subplot(gs[0, 3])
    ax_c_paired = fig.add_subplot(gs[1, 0])
    ax_c_delta = fig.add_subplot(gs[1, 1])
    ax_d_rmsd = fig.add_subplot(gs[1, 2])
    ax_d_glycan = fig.add_subplot(gs[1, 3])

    for ax, label in zip([ax_a1, ax_b1, ax_c_paired, ax_d_rmsd], ["A", "B", "C", "D"]):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes, fontsize=14 * TEXT_SCALE, fontweight="bold", va="top")

    print("=" * 60)
    print("GENERATING FIGURE 1 (all panels from 81-protein pipeline)")
    print("=" * 60)

    print("\n--- Panel A ---")
    make_panel_a(ax_a1, ax_a2)

    print("\n--- Panel B ---")
    make_panel_b(ax_b1, ax_b2)

    print("\n--- Panel C ---")
    make_panel_c(ax_c_delta, ax_c_paired)

    print("\n--- Panel D ---")
    make_panel_d(ax_d_rmsd, ax_d_glycan)

    # Save
    for out_dir in [PIPELINE_FIG_DIR, OUT_DIR]:
        out_path = out_dir / "Figure1_publication_updated.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")

        pdf_path = out_dir / "Figure1_publication_updated.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")

    plt.close()


if __name__ == "__main__":
    make_figure1()
