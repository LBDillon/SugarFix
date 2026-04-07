#!/usr/bin/env python3
"""
Cross-protein comparison of ProteinMPNN sequon treatment.

Auto-discovers all output_* directories and aggregates results to generate:
- Summary table and statistics
- Multi-panel comparison figures
- RSA-retention analysis pooled across all proteins
- De novo sequon comparison (normalized by sequence length)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"

CONDITIONS = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]
COND_LABELS = {"unconstrained": "Unconstrained", "n_only_fixed": "N-only Fixed",
               "full_sequon_fixed": "Full Sequon Fixed"}
COND_COLORS = {"unconstrained": "#d95f02", "n_only_fixed": "#1b9e77",
               "full_sequon_fixed": "#7570b3"}


def _sig_label(pval):
    """Return significance stars for a p-value."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return "ns"


def _annotate_sig(ax, x1, x2, y, pval, fontsize=10):
    """Draw a significance bracket between two x positions."""
    label = _sig_label(pval)
    ax.plot([x1, x1, x2, x2], [y, y + 1.5, y + 1.5, y], color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y + 1.5, f"{label}\np={pval:.1e}",
            ha="center", va="bottom", fontsize=fontsize)


def discover_proteins(base_dir: Path):
    """Auto-discover proteins from output_* directories that have retention data."""
    outputs_dir = base_dir / "outputs"
    proteins = {}
    for output_dir in sorted(outputs_dir.glob("output_*")):
        ret_path = output_dir / "all_conditions_retention.csv"
        if not ret_path.exists():
            continue
        pdb_id = output_dir.name.replace("output_", "")

        info = {"name": pdb_id, "type": "Unknown", "total_length": 0}
        struct_info_path = base_dir / "prep" / pdb_id / "structure" / "structure_info.json"
        if struct_info_path.exists():
            try:
                with open(struct_info_path) as f:
                    sinfo = json.load(f)
                chains = sinfo.get("asymmetric_unit", {}).get("chains", [])
                n_chains = len(chains)
                total_length = sum(c.get("length", 0) for c in chains)
                info["type"] = f"{n_chains}-chain" if n_chains > 1 else "single-chain"
                info["total_length"] = total_length
            except (json.JSONDecodeError, KeyError):
                pass

        proteins[pdb_id] = info
    return proteins


def load_all_data(base_dir: Path, protein_ids: list):
    """Load retention, structural, and de novo data for all proteins."""
    retention_dfs = []
    structural_dfs = []
    denovo_dfs = []

    for pdb_id in protein_ids:
        output_dir = base_dir / "outputs" / f"output_{pdb_id}"

        ret_path = output_dir / "all_conditions_retention.csv"
        if ret_path.exists():
            df = pd.read_csv(ret_path)
            df["pdb_id"] = pdb_id
            retention_dfs.append(df)

        struct_path = output_dir / "structural_context.csv"
        if struct_path.exists():
            df = pd.read_csv(struct_path)
            df["pdb_id"] = pdb_id
            structural_dfs.append(df)

        denovo_path = output_dir / "denovo_sequons.csv"
        if denovo_path.exists():
            df = pd.read_csv(denovo_path)
            df["pdb_id"] = pdb_id
            denovo_dfs.append(df)

    retention = pd.concat(retention_dfs, ignore_index=True) if retention_dfs else pd.DataFrame()
    structural = pd.concat(structural_dfs, ignore_index=True) if structural_dfs else pd.DataFrame()
    denovo = pd.concat(denovo_dfs, ignore_index=True) if denovo_dfs else pd.DataFrame()

    return retention, structural, denovo


def compute_summary(retention, structural, denovo, protein_info):
    """Compute per-protein summary statistics."""
    rows = []
    for pdb_id, info in protein_info.items():
        prot_ret = retention[retention["pdb_id"] == pdb_id]
        prot_struct = structural[structural["pdb_id"] == pdb_id]
        prot_denovo = denovo[denovo["pdb_id"] == pdb_id] if not denovo.empty else pd.DataFrame()

        if prot_ret.empty:
            continue

        n_chains = prot_ret["chain"].nunique()
        n_sequons = prot_struct.shape[0] if not prot_struct.empty else 0
        total_length = info.get("total_length", 0)

        row = {
            "PDB": pdb_id,
            "Protein": info["name"],
            "Type": info["type"],
            "Chains": n_chains,
            "Sequons": n_sequons,
            "TotalLength": total_length,
            "SequonDensity": (n_sequons / total_length * 100) if total_length > 0 else 0,
        }

        for cond in CONDITIONS:
            cond_data = prot_ret[prot_ret["condition"] == cond]
            if not cond_data.empty:
                row[f"{cond}_n_ret"] = cond_data["n_retained"].mean() * 100
                row[f"{cond}_exact"] = cond_data["exact_match"].mean() * 100
                row[f"{cond}_func"] = cond_data["functional"].mean() * 100
            else:
                row[f"{cond}_n_ret"] = np.nan
                row[f"{cond}_exact"] = np.nan
                row[f"{cond}_func"] = np.nan

        # Evidence tier counts
        if not prot_struct.empty and "evidence_tier" in prot_struct.columns:
            n_validated = len(prot_struct[prot_struct["evidence_tier"].isin(
                ["experimental", "pdb_evidence"])])
            row["ValidatedSites"] = n_validated
            row["MotifOnlySites"] = n_sequons - n_validated
        else:
            row["ValidatedSites"] = np.nan
            row["MotifOnlySites"] = np.nan

        # RSA correlation (unconstrained)
        if not prot_struct.empty and prot_struct["rsa"].notna().sum() >= 3:
            valid = prot_struct.dropna(subset=["rsa", "n_retention"])
            if len(valid) >= 3 and valid["n_retention"].std() > 0:
                rho, pval = stats.spearmanr(valid["rsa"], valid["n_retention"])
                row["rsa_rho"] = rho
                row["rsa_pval"] = pval
            else:
                row["rsa_rho"] = np.nan
                row["rsa_pval"] = np.nan
        else:
            row["rsa_rho"] = np.nan
            row["rsa_pval"] = np.nan

        # De novo stats (unconstrained)
        dn_unconstrained = prot_denovo[prot_denovo["condition"] == "unconstrained"] if not prot_denovo.empty else pd.DataFrame()
        if not dn_unconstrained.empty:
            n_designs = dn_unconstrained["design"].nunique()
            row["denovo_avg"] = len(dn_unconstrained) / n_designs if n_designs > 0 else 0
            row["denovo_positions"] = dn_unconstrained.groupby(["chain", "position_0idx"]).ngroups
        else:
            row["denovo_avg"] = 0.0
            row["denovo_positions"] = 0

        # Normalized de novo: per 100 residues per design
        if total_length > 0:
            row["denovo_per_100res"] = (row["denovo_avg"] / total_length) * 100
        else:
            row["denovo_per_100res"] = 0.0

        # MPNN scores
        unconstrained = prot_ret[prot_ret["condition"] == "unconstrained"]
        if not unconstrained.empty and "score" in unconstrained.columns:
            row["mpnn_score_mean"] = unconstrained["score"].dropna().mean()

        rows.append(row)

    return pd.DataFrame(rows)


def create_comparison_figures(retention, structural, denovo, summary, output_dir):
    """Create comprehensive cross-protein comparison figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_proteins = len(summary)

    # Sort by unconstrained functional retention
    summary_sorted = summary.sort_values("unconstrained_func", ascending=False).reset_index(drop=True)
    pdb_order_sorted = list(summary_sorted["PDB"])

    valid_struct = structural.dropna(subset=["rsa", "n_retention"])

    # =========================================================================
    # FIGURE 1: Main results (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # --- Panel A: Functional retention across conditions (paired box + strip) ---
    ax = axes[0, 0]
    plot_data = []
    for cond in CONDITIONS:
        col = f"{cond}_func"
        for val in summary[col].dropna():
            plot_data.append({"Condition": COND_LABELS[cond], "Functional Retention (%)": val,
                              "cond_key": cond})
    plot_df = pd.DataFrame(plot_data)
    bp = sns.boxplot(data=plot_df, x="Condition", y="Functional Retention (%)",
                     order=[COND_LABELS[c] for c in CONDITIONS],
                     palette=[COND_COLORS[c] for c in CONDITIONS],
                     width=0.5, ax=ax, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                     boxprops=dict(alpha=0.7), flierprops=dict(markersize=3))
    sns.stripplot(data=plot_df, x="Condition", y="Functional Retention (%)",
                  order=[COND_LABELS[c] for c in CONDITIONS],
                  color="black", alpha=0.25, size=4, jitter=0.2, ax=ax)
    # Medians
    for i, cond in enumerate(CONDITIONS):
        vals = summary[f"{cond}_func"].dropna()
        ax.text(i, vals.median() + 2, f"{vals.median():.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    # Significance: unconstrained vs n_only
    unc_vals = summary["unconstrained_func"].dropna().values
    nonly_vals = summary["n_only_fixed_func"].dropna().values
    stat, p_unc_nonly = stats.wilcoxon(unc_vals, nonly_vals)
    _annotate_sig(ax, 0, 1, 105, p_unc_nonly, fontsize=8)
    ax.set_ylim(-5, 125)
    ax.set_xlabel("")
    ax.set_title("A. Functional Retention by Condition", fontsize=13, fontweight="bold")

    # --- Panel B: Unconstrained retention breakdown (N retained vs functional vs exact) ---
    ax = axes[0, 1]
    metrics = ["unconstrained_n_ret", "unconstrained_func", "unconstrained_exact"]
    metric_labels = ["N Retained", "Functional\n(valid N-X-S/T)", "Exact Match\n(identical triplet)"]
    metric_colors = ["#3498db", "#2ecc71", "#9b59b6"]
    bp_data = [summary[m].dropna().values for m in metrics]
    bp = ax.boxplot(bp_data, tick_labels=metric_labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                    flierprops=dict(markersize=3))
    for patch, color in zip(bp["boxes"], metric_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(bp_data):
        jitter = np.random.normal(0, 0.04, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color="black", alpha=0.25, s=12, zorder=3)
        med = np.median(vals)
        ax.text(i + 1, med + 2, f"{med:.1f}%", ha="center", fontsize=9, fontweight="bold")
    # Significance: N retained vs functional (paired)
    n_ret = summary["unconstrained_n_ret"].dropna().values
    func = summary["unconstrained_func"].dropna().values
    stat, p_nret_func = stats.wilcoxon(n_ret, func)
    _annotate_sig(ax, 1, 2, max(n_ret.max(), func.max()) + 5, p_nret_func, fontsize=8)
    ax.set_ylabel("Retention (%)", fontsize=12)
    ax.set_ylim(-5, max(n_ret.max(), 100) + 25)
    ax.set_title(f"B. Unconstrained: What Is Retained? (n={n_proteins})",
                 fontsize=13, fontweight="bold")

    # --- Panel C: RSA vs functional retention (pooled scatter) ---
    ax = axes[1, 0]
    if not valid_struct.empty:
        ax.scatter(valid_struct["rsa"], valid_struct["n_retention"],
                   alpha=0.35, s=25, c="#4c78a8", edgecolors="none")
        if len(valid_struct) >= 5:
            rho, pval = stats.spearmanr(valid_struct["rsa"], valid_struct["n_retention"])
            z = np.polyfit(valid_struct["rsa"], valid_struct["n_retention"], 1)
            x_line = np.linspace(0, valid_struct["rsa"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="red", alpha=0.7, linewidth=2)
            sig = _sig_label(pval)
            ax.text(0.02, 0.98,
                    f"n={len(valid_struct)} sequon sites\nSpearman rho={rho:.3f}\np={pval:.2e} ({sig})",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
        ax.axvspan(0, 20, alpha=0.06, color="blue")
        ax.axvspan(20, 50, alpha=0.06, color="yellow")
        ax.axvspan(50, 100, alpha=0.06, color="red")
        ax.text(10, -8, "Buried", ha="center", fontsize=9, color="gray")
        ax.text(35, -8, "Intermediate", ha="center", fontsize=9, color="gray")
        ax.text(75, -8, "Exposed", ha="center", fontsize=9, color="gray")
    ax.set_xlabel("RSA (%)", fontsize=12)
    ax.set_ylabel("N Retention (%)", fontsize=12)
    ax.set_title("C. RSA vs N Retention (Pooled)", fontsize=13, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-12, 105)

    # --- Panel D: Retention by RSA category with significance ---
    ax = axes[1, 1]
    if not valid_struct.empty:
        cat_order = ["Buried", "Intermediate", "Exposed"]
        cat_colors = {"Buried": "#3498db", "Intermediate": "#f1c40f", "Exposed": "#e74c3c"}

        # Use violin + strip for richer display
        cat_data = valid_struct[valid_struct["rsa_category"].isin(cat_order)].copy()
        cat_data["rsa_category"] = pd.Categorical(cat_data["rsa_category"], categories=cat_order, ordered=True)
        sns.violinplot(data=cat_data, x="rsa_category", y="n_retention",
                       order=cat_order, palette=cat_colors, alpha=0.6, ax=ax,
                       inner=None, cut=0)
        sns.stripplot(data=cat_data, x="rsa_category", y="n_retention",
                      order=cat_order, color="black", alpha=0.2, size=3, jitter=0.2, ax=ax)

        # Means with SEM error bars
        for i, cat in enumerate(cat_order):
            vals = cat_data[cat_data["rsa_category"] == cat]["n_retention"]
            mean = vals.mean()
            ax.plot(i, mean, "D", color="red", markersize=7, zorder=5)
            ax.text(i, mean + 3, f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold")

        # Pairwise significance
        groups = {c: valid_struct[valid_struct["rsa_category"] == c]["n_retention"].values
                  for c in cat_order}
        # Kruskal-Wallis overall
        kw_groups = [groups[c] for c in cat_order if len(groups[c]) > 0]
        if len(kw_groups) >= 2:
            stat, kw_pval = stats.kruskal(*kw_groups)
            ax.text(0.98, 0.98,
                    f"Kruskal-Wallis p={kw_pval:.2e} ({_sig_label(kw_pval)})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

        # Pairwise: Buried vs Exposed
        if len(groups.get("Buried", [])) > 0 and len(groups.get("Exposed", [])) > 0:
            stat, p_be = stats.mannwhitneyu(groups["Buried"], groups["Exposed"])
            _annotate_sig(ax, 0, 2, 92, p_be, fontsize=8)

        ax.set_xticklabels([f"{c}\n(n={len(groups.get(c, []))})" for c in cat_order], fontsize=10)

    ax.set_ylabel("N Retention (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("D. Retention by Burial Category", fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 115)

    plt.suptitle(f"ProteinMPNN Sequon Retention Analysis (n={n_proteins} proteins, "
                 f"{len(valid_struct)} sequon sites)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig1_path = fig_dir / "cross_protein_main.png"
    plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig1_path}")

    # =========================================================================
    # FIGURE 2: De novo and condition effects (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # --- Panel A: N-only fixed — what fraction of fixed-N sites remain functional? ---
    ax = axes[0, 0]
    n_only_n_ret = summary["n_only_fixed_n_ret"].dropna().values
    n_only_func = summary["n_only_fixed_func"].dropna().values
    # N is always fixed so n_ret should be ~100%. Show functional vs 100 as the key message.
    ax.hist(n_only_func, bins=20, color=COND_COLORS["n_only_fixed"],
            alpha=0.7, edgecolor="white")
    ax.axvline(x=np.median(n_only_func), color="red", linestyle="--", linewidth=2)
    ax.text(np.median(n_only_func) + 1, ax.get_ylim()[1] * 0.9,
            f"Median: {np.median(n_only_func):.1f}%",
            fontsize=11, fontweight="bold", color="red")
    ax.set_xlabel("Functional Retention (%)", fontsize=12)
    ax.set_ylabel("Number of Proteins", fontsize=12)
    ax.set_title("A. N-only Fixed: Flanking Mutations Still Destroy Sequons",
                 fontsize=12, fontweight="bold")
    ax.text(0.98, 0.75,
            f"N is always retained (fixed),\nbut flanking S/T positions\nare mutated in "
            f"{100 - np.median(n_only_func):.0f}% of cases\n(median across {len(n_only_func)} proteins)",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    # --- Panel B: De novo sequon density (normalized per 100 residues) ---
    ax = axes[0, 1]
    denovo_norm = summary["denovo_per_100res"].values
    prob_sequon = (1 / 20) * (18 / 20) * (2 / 20) * 100  # Expected per 100 residues
    ax.hist(denovo_norm, bins=20, color="#d95f02", alpha=0.7, edgecolor="white")
    ax.axvline(x=np.median(denovo_norm), color="red", linestyle="--", linewidth=2,
               label=f"Median: {np.median(denovo_norm):.2f}")
    ax.axvline(x=prob_sequon, color="blue", linestyle=":", linewidth=2,
               label=f"Random expectation: {prob_sequon:.2f}")
    ax.set_xlabel("De Novo Sequons per 100 Residues per Design", fontsize=12)
    ax.set_ylabel("Number of Proteins", fontsize=12)
    ax.set_title("B. De Novo Sequon Creation Rate (Length-Normalized)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # One-sample Wilcoxon: is de novo rate significantly above random?
    stat, p_denovo = stats.wilcoxon(denovo_norm - prob_sequon)
    sig = _sig_label(p_denovo)
    ax.text(0.98, 0.7,
            f"vs random: p={p_denovo:.2e} ({sig})\n"
            f"Median rate is {np.median(denovo_norm)/prob_sequon:.1f}x\nabove random expectation",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    # --- Panel C: Sequon density vs functional retention ---
    ax = axes[1, 0]
    valid_density = summary[summary["TotalLength"] > 0].copy()
    ax.scatter(valid_density["SequonDensity"], valid_density["unconstrained_func"],
               s=50, c="#4c78a8", alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Sequon Density (sequons per 100 residues)", fontsize=12)
    ax.set_ylabel("Unconstrained Functional Retention (%)", fontsize=12)
    ax.set_title("C. Does Sequon Density Predict Retention?",
                 fontsize=12, fontweight="bold")
    if len(valid_density) >= 5:
        rho, pval = stats.spearmanr(valid_density["SequonDensity"],
                                    valid_density["unconstrained_func"])
        sig = _sig_label(pval)
        z = np.polyfit(valid_density["SequonDensity"], valid_density["unconstrained_func"], 1)
        x_line = np.linspace(valid_density["SequonDensity"].min(),
                             valid_density["SequonDensity"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
        ax.text(0.02, 0.98, f"Spearman rho={rho:.2f}\np={pval:.2e} ({sig})",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # --- Panel D: MPNN score vs functional retention ---
    ax = axes[1, 1]
    if "mpnn_score_mean" in summary.columns:
        valid_scores = summary.dropna(subset=["mpnn_score_mean", "unconstrained_func"])
        if not valid_scores.empty:
            ax.scatter(valid_scores["mpnn_score_mean"], valid_scores["unconstrained_func"],
                       s=50, c="#4c78a8", alpha=0.6, edgecolors="black", linewidth=0.5)
            if len(valid_scores) >= 5:
                rho, pval = stats.spearmanr(valid_scores["mpnn_score_mean"],
                                            valid_scores["unconstrained_func"])
                sig = _sig_label(pval)
                z = np.polyfit(valid_scores["mpnn_score_mean"],
                               valid_scores["unconstrained_func"], 1)
                x_line = np.linspace(valid_scores["mpnn_score_mean"].min(),
                                     valid_scores["mpnn_score_mean"].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
                ax.text(0.02, 0.98, f"Spearman rho={rho:.2f}\np={pval:.2e} ({sig})",
                        transform=ax.transAxes, va="top", fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("Mean MPNN Score (Unconstrained)", fontsize=12)
    ax.set_ylabel("Functional Retention (%)", fontsize=12)
    ax.set_title("D. Design Confidence vs Sequon Retention",
                 fontsize=12, fontweight="bold")

    plt.suptitle(f"Condition Effects and De Novo Analysis (n={n_proteins})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig2_path = fig_dir / "cross_protein_effects.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig2_path}")

    # =========================================================================
    # FIGURE 3: Per-protein ranked bar chart (sorted by retention)
    # =========================================================================
    fig_height = max(6, n_proteins * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = np.arange(n_proteins)
    func_vals = summary_sorted["unconstrained_func"].values
    n_ret_vals = summary_sorted["unconstrained_n_ret"].values

    ax.barh(y_pos, n_ret_vals, height=0.7, color="#3498db", alpha=0.4, label="N Retained")
    ax.barh(y_pos, func_vals, height=0.7, color="#2ecc71", alpha=0.8, label="Functional")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{p} ({int(summary_sorted.loc[summary_sorted['PDB']==p, 'Sequons'].values[0])} seq)"
         for p in pdb_order_sorted], fontsize=8)
    ax.set_xlabel("Retention (%)", fontsize=12)
    ax.set_title(f"Unconstrained Sequon Retention by Protein (n={n_proteins})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()

    # Add median line
    med_func = np.median(func_vals)
    ax.axvline(x=med_func, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(med_func + 1, n_proteins - 1, f"Median: {med_func:.1f}%",
            fontsize=9, color="red", fontweight="bold")

    plt.tight_layout()
    fig3_path = fig_dir / "cross_protein_ranked.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig3_path}")

    return fig1_path, fig2_path, fig3_path


def print_statistical_tests(retention, structural, denovo, summary):
    """Print key statistical tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    n_proteins = len(summary)

    # 1. Pooled RSA-retention correlation
    valid = structural.dropna(subset=["rsa", "n_retention"])
    if len(valid) >= 5:
        rho, pval = stats.spearmanr(valid["rsa"], valid["n_retention"])
        print(f"\n1. Pooled RSA vs N Retention (n={len(valid)} sequon sites from {n_proteins} proteins):")
        print(f"   Spearman rho = {rho:.3f}, p = {pval:.2e} ({_sig_label(pval)})")

    # 2. Retention by RSA category
    print(f"\n2. Retention by RSA Category (pooled across {n_proteins} proteins):")
    for cat in ["Buried", "Intermediate", "Exposed"]:
        subset = valid[valid["rsa_category"] == cat]
        if not subset.empty:
            mean = subset["n_retention"].mean()
            sem = subset["n_retention"].sem()
            print(f"   {cat:15s}: {mean:.1f}% +/- {sem:.1f}% (n={len(subset)} sites)")

    groups = [valid[valid["rsa_category"] == c]["n_retention"].values
              for c in ["Buried", "Intermediate", "Exposed"]
              if len(valid[valid["rsa_category"] == c]) > 0]
    if len(groups) >= 2:
        stat, pval = stats.kruskal(*groups)
        print(f"   Kruskal-Wallis: H = {stat:.2f}, p = {pval:.2e} ({_sig_label(pval)})")

    # Pairwise Mann-Whitney
    cat_names = ["Buried", "Intermediate", "Exposed"]
    cat_groups = {c: valid[valid["rsa_category"] == c]["n_retention"].values for c in cat_names}
    for i, c1 in enumerate(cat_names):
        for c2 in cat_names[i+1:]:
            if len(cat_groups[c1]) > 0 and len(cat_groups[c2]) > 0:
                stat, pval = stats.mannwhitneyu(cat_groups[c1], cat_groups[c2])
                print(f"   {c1} vs {c2}: Mann-Whitney p = {pval:.2e} ({_sig_label(pval)})")

    # 3. Condition comparisons (paired Wilcoxon)
    print(f"\n3. Condition Effects (paired across {n_proteins} proteins):")
    unc_func = summary["unconstrained_func"].dropna().values
    nonly_func = summary["n_only_fixed_func"].dropna().values
    if len(unc_func) > 0 and len(nonly_func) > 0 and len(unc_func) == len(nonly_func):
        stat, pval = stats.wilcoxon(unc_func, nonly_func)
        print(f"   Unconstrained vs N-only (functional): Wilcoxon p = {pval:.2e} ({_sig_label(pval)})")
        print(f"     Unconstrained median: {np.median(unc_func):.1f}%")
        print(f"     N-only median:        {np.median(nonly_func):.1f}%")

    # 4. N-only functional gap
    print(f"\n4. N-only Fixed: Functional Gap (N retained but flanking mutated):")
    gaps = summary["n_only_fixed_n_ret"] - summary["n_only_fixed_func"]
    gaps = gaps.dropna()
    print(f"   Mean gap: {gaps.mean():.1f}% (median: {gaps.median():.1f}%)")
    print(f"   Proteins where >50% of fixed-N sites lose function: {(gaps > 50).sum()}/{len(gaps)}")

    # 5. De novo sequon statistics (normalized)
    print(f"\n5. De Novo Sequon Generation (Unconstrained, normalized):")
    denovo_norm = summary["denovo_per_100res"]
    prob_sequon = (1 / 20) * (18 / 20) * (2 / 20) * 100
    print(f"   Rate per 100 residues: {denovo_norm.median():.2f} (median), {denovo_norm.mean():.2f} (mean)")
    print(f"   Random expectation:    {prob_sequon:.2f} per 100 residues")
    print(f"   Fold enrichment:       {denovo_norm.median()/prob_sequon:.1f}x (median)")
    if (denovo_norm > 0).sum() >= 5:
        stat, pval = stats.wilcoxon(denovo_norm - prob_sequon)
        print(f"   Wilcoxon (vs random):  p = {pval:.2e} ({_sig_label(pval)})")

    # 6. Evidence tier analysis
    if "evidence_tier" in retention.columns and retention["evidence_tier"].notna().any():
        print(f"\n6. Retention by Evidence Tier (Unconstrained, pooled):")
        unc_ret = retention[retention["condition"] == "unconstrained"]
        for tier_name in ["experimental", "pdb_evidence", "curator_inferred", "motif_only"]:
            tier_data = unc_ret[unc_ret["evidence_tier"] == tier_name]
            if not tier_data.empty:
                n_ret = tier_data["n_retained"].mean() * 100
                func = tier_data["functional"].mean() * 100
                n_sites = tier_data.groupby(["pdb_id", "chain", "position_0idx"]).ngroups
                print(f"   {tier_name:20s}: N Ret={n_ret:.1f}%, Func={func:.1f}% (n={n_sites} sites)")

        # Compare validated vs motif_only
        validated = unc_ret[unc_ret["evidence_tier"].isin(["experimental", "pdb_evidence"])]
        motif_only = unc_ret[unc_ret["evidence_tier"] == "motif_only"]
        if not validated.empty and not motif_only.empty:
            val_func = validated["functional"].mean() * 100
            mo_func = motif_only["functional"].mean() * 100
            print(f"\n   Validated vs motif_only functional retention:")
            print(f"     Validated:  {val_func:.1f}%")
            print(f"     Motif-only: {mo_func:.1f}%")
            # Mann-Whitney test
            val_grouped = validated.groupby(["pdb_id", "chain", "position_0idx"])["functional"].mean()
            mo_grouped = motif_only.groupby(["pdb_id", "chain", "position_0idx"])["functional"].mean()
            if len(val_grouped) >= 3 and len(mo_grouped) >= 3:
                stat, pval = stats.mannwhitneyu(val_grouped.values, mo_grouped.values)
                print(f"     Mann-Whitney: p = {pval:.2e} ({_sig_label(pval)})")

    # 7. Overall summary
    print(f"\n7. Overall Summary:")
    unc_func_s = summary["unconstrained_func"].dropna()
    print(f"   Median unconstrained functional retention: {unc_func_s.median():.1f}%")
    print(f"   Mean unconstrained functional retention:   {unc_func_s.mean():.1f}%")
    print(f"   Proteins with <25% functional retention:   {(unc_func_s < 25).sum()}/{len(unc_func_s)} "
          f"({(unc_func_s < 25).sum()/len(unc_func_s)*100:.0f}%)")
    print(f"   Proteins with >75% functional retention:   {(unc_func_s > 75).sum()}/{len(unc_func_s)} "
          f"({(unc_func_s > 75).sum()/len(unc_func_s)*100:.0f}%)")

    # Evidence tier overall counts
    if "ValidatedSites" in summary.columns and summary["ValidatedSites"].notna().any():
        total_validated = int(summary["ValidatedSites"].sum())
        total_motif = int(summary["MotifOnlySites"].sum())
        total_all = total_validated + total_motif
        print(f"\n   Evidence tier breakdown ({total_all} total sequon sites):")
        print(f"     Validated (experimental + pdb_evidence): {total_validated} ({total_validated/total_all*100:.0f}%)")
        print(f"     Motif-only (no external validation):     {total_motif} ({total_motif/total_all*100:.0f}%)")


def print_figure_explainers(summary, structural, denovo):
    """Print a plain-language explanation of every figure panel."""
    n_proteins = len(summary)
    n_sites = len(structural.dropna(subset=["rsa", "n_retention"])) if not structural.empty else 0
    unc_func = summary["unconstrained_func"].dropna()
    nonly_func = summary["n_only_fixed_func"].dropna()
    denovo_norm = summary["denovo_per_100res"]
    prob_sequon = (1 / 20) * (18 / 20) * (2 / 20) * 100

    print("\n" + "=" * 70)
    print("FIGURE EXPLAINERS")
    print("=" * 70)

    # --- Figure 1 ---
    print("\n--- Figure 1: cross_protein_main.png ---")
    print("  Overall title: ProteinMPNN Sequon Retention Analysis")
    print(f"  Data: {n_proteins} glycoproteins, {n_sites} total sequon sites\n")

    print("  Panel A – Functional Retention by Condition")
    print("    What it shows: Box-and-strip plots comparing per-protein functional")
    print("    sequon retention (%) across three ProteinMPNN design conditions:")
    print("    Unconstrained (no residues fixed), N-only Fixed (asparagine fixed),")
    print("    and Full Sequon Fixed (N, X, S/T all fixed). Each dot is one protein.")
    print("    Red diamond = mean. Wilcoxon signed-rank test shown between conditions.")
    print(f"    Key finding: Unconstrained designs retain a median of {unc_func.median():.1f}%")
    print(f"    of functional sequons; fixing just the N raises this to {nonly_func.median():.1f}%,")
    print("    but flanking mutations still destroy roughly half of the sequon motifs.\n")

    print("  Panel B – Unconstrained: What Is Retained?")
    print("    What it shows: Three metrics for the unconstrained condition only,")
    print("    arranged left-to-right by strictness:")
    print("      'N Retained' – the asparagine at position 1 of the sequon is kept")
    print("      'Functional' – the full N-X-S/T motif is preserved (X != P)")
    print("      'Exact Match' – the designed triplet is identical to wild-type")
    print("    Paired Wilcoxon test between N Retained and Functional quantifies how")
    print("    often keeping N is not enough because flanking residues are mutated.\n")

    print("  Panel C – RSA vs N Retention (Pooled)")
    print("    What it shows: Scatter of relative solvent accessibility (RSA, %) vs")
    print("    asparagine retention (%) for every sequon site across all proteins.")
    print("    Background shading indicates burial categories (Buried <20%, Intermediate")
    print("    20-50%, Exposed >50%). Dashed red line = linear trend. Spearman rho and")
    print("    p-value quantify monotonic association.")
    print("    Key question: Are buried sequons harder for MPNN to redesign away?\n")

    print("  Panel D – Retention by Burial Category")
    print("    What it shows: Violin + strip plots grouping sequon sites into Buried,")
    print("    Intermediate, and Exposed bins by RSA. Red diamonds = means.")
    print("    Kruskal-Wallis tests overall group difference; Mann-Whitney bracket")
    print("    compares Buried vs Exposed. This is the categorical complement to Panel C.")

    # --- Figure 2 ---
    print("\n--- Figure 2: cross_protein_effects.png ---")
    print("  Overall title: Condition Effects and De Novo Analysis\n")

    print("  Panel A – N-only Fixed: Flanking Mutations Still Destroy Sequons")
    print("    What it shows: Histogram of per-protein functional retention under the")
    print("    N-only Fixed condition. Because the asparagine is fixed, N retention is")
    print("    ~100% by construction – but the S/T at position +2 is free to mutate.")
    print("    The histogram therefore reveals how often flanking mutations break the")
    print("    sequon even when the asparagine is preserved.")
    med_nonly = nonly_func.median()
    print(f"    Key finding: Median functional retention is {med_nonly:.1f}%, meaning")
    print(f"    MPNN mutates the flanking S/T in ~{100 - med_nonly:.0f}% of sequon sites.\n")

    print("  Panel B – De Novo Sequon Creation Rate (Length-Normalized)")
    print("    What it shows: Histogram of de novo N-X-S/T sequons generated per 100")
    print("    residues per design under the unconstrained condition. The blue dotted")
    print(f"    line marks the random expectation ({prob_sequon:.2f} per 100 residues,")
    print("    calculated as P(N) x P(not P) x P(S or T) = 1/20 x 18/20 x 2/20).")
    print("    One-sample Wilcoxon tests whether the observed rate significantly")
    print("    exceeds random.")
    fold = np.median(denovo_norm) / prob_sequon if prob_sequon > 0 else 0
    print(f"    Key finding: Median rate is {np.median(denovo_norm):.2f} per 100 residues")
    print(f"    ({fold:.1f}x above random expectation).\n")

    print("  Panel C – Does Sequon Density Predict Retention?")
    print("    What it shows: Scatter of wild-type sequon density (sequons per 100")
    print("    residues) vs unconstrained functional retention (%). Tests whether")
    print("    proteins that are more densely glycosylated have higher or lower")
    print("    retention. Spearman correlation and p-value annotated.\n")

    print("  Panel D – Design Confidence vs Sequon Retention")
    print("    What it shows: Scatter of mean ProteinMPNN score (negative log-likelihood;")
    print("    lower = more confident) vs unconstrained functional retention (%).")
    print("    Tests whether MPNN's own confidence in its designs correlates with its")
    print("    tendency to preserve or remove sequons. Spearman test annotated.\n")

    # --- Figure 3 ---
    print("--- Figure 3: cross_protein_ranked.png ---")
    print("  Title: Unconstrained Sequon Retention by Protein\n")

    print("  What it shows: Horizontal bar chart with one bar per protein, sorted")
    print("  from highest to lowest unconstrained functional retention. Light blue")
    print("  bars = N Retained (asparagine kept); green overlay = Functional")
    print("  (full N-X-S/T motif intact). The gap between the two bars shows sites")
    print("  where N was retained but flanking residues were mutated. Red dashed")
    print("  line = median across all proteins. Protein labels include sequon count.")
    n_low = (unc_func < 25).sum()
    n_high = (unc_func > 75).sum()
    print(f"  Key finding: {n_low}/{len(unc_func)} proteins ({n_low/len(unc_func)*100:.0f}%)")
    print(f"  have <25% functional retention; {n_high}/{len(unc_func)} ({n_high/len(unc_func)*100:.0f}%)")
    print("  have >75%. This shows the breadth of the bias across diverse glycoproteins.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", default=str(PIPELINE_ROOT))
    args = parser.parse_args()
    base_dir = Path(args.pipeline_root) / "data"
    output_dir = base_dir / "cross_protein_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-PROTEIN COMPARISON ANALYSIS")
    print("=" * 70)

    print("\nDiscovering proteins...")
    protein_info = discover_proteins(base_dir)
    protein_ids = list(protein_info.keys())
    print(f"  Found {len(protein_ids)} proteins with output data")

    print("\nLoading data...")
    retention, structural, denovo = load_all_data(base_dir, protein_ids)

    proteins_found = retention["pdb_id"].unique() if not retention.empty else []
    print(f"  Loaded retention data for {len(proteins_found)} proteins")
    print(f"  Total retention rows: {len(retention)}")
    print(f"  Total structural context rows: {len(structural)}")
    print(f"  Total de novo rows: {len(denovo)}")

    print("\nComputing summary...")
    summary = compute_summary(retention, structural, denovo, protein_info)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE ({len(summary)} proteins)")
    print("=" * 70)
    has_tiers = "ValidatedSites" in summary.columns and summary["ValidatedSites"].notna().any()
    tier_header = " {'Valid':>5} {'MoOnly':>6}" if has_tiers else ""
    print(f"\n{'PDB':<6} {'Ch':>3} {'Seq':>4} {'Len':>5} "
          f"{'N Ret':>7} {'Func':>7} {'Exact':>7} {'DN/100':>7} {'RSA rho':>8}"
          + (" {'Valid':>5} {'MoOnly':>6}" if has_tiers else ""))
    print("-" * (65 + (13 if has_tiers else 0)))
    for _, row in summary.sort_values("unconstrained_func", ascending=False).iterrows():
        rsa_str = f"{row['rsa_rho']:.2f}" if pd.notna(row.get("rsa_rho")) else "N/A"
        tier_str = ""
        if has_tiers and pd.notna(row.get("ValidatedSites")):
            tier_str = f" {int(row['ValidatedSites']):>5} {int(row['MotifOnlySites']):>6}"
        print(f"{row['PDB']:<6} "
              f"{row['Chains']:>3} {row['Sequons']:>4} {row['TotalLength']:>5} "
              f"{row['unconstrained_n_ret']:>6.1f}% "
              f"{row['unconstrained_func']:>6.1f}% "
              f"{row['unconstrained_exact']:>6.1f}% "
              f"{row['denovo_per_100res']:>6.2f} "
              f"{rsa_str:>8}{tier_str}")

    # Pooled totals
    print("-" * 65)
    all_unc = retention[retention["condition"] == "unconstrained"]
    print(f"{'POOL':<6} "
          f"{summary['Chains'].sum():>3} {summary['Sequons'].sum():>4} "
          f"{summary['TotalLength'].sum():>5} "
          f"{all_unc['n_retained'].mean()*100:>6.1f}% "
          f"{all_unc['functional'].mean()*100:>6.1f}% "
          f"{all_unc['exact_match'].mean()*100:>6.1f}% "
          f"{summary['denovo_per_100res'].mean():>6.2f} "
          f"{'':>8}")

    # Statistical tests
    print_statistical_tests(retention, structural, denovo, summary)

    # Create figures
    print("\n" + "=" * 70)
    print("CREATING FIGURES")
    print("=" * 70)
    figs = create_comparison_figures(retention, structural, denovo, summary, output_dir)

    # Save CSVs
    summary_path = output_dir / "protein_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    structural_path = output_dir / "pooled_structural_context.csv"
    structural.to_csv(structural_path, index=False)
    print(f"  Saved: {structural_path}")

    retention_path = output_dir / "pooled_retention.csv"
    retention.to_csv(retention_path, index=False)
    print(f"  Saved: {retention_path}")

    if not denovo.empty:
        denovo_path = output_dir / "pooled_denovo.csv"
        denovo.to_csv(denovo_path, index=False)
        print(f"  Saved: {denovo_path}")

    # Print figure explainers
    print_figure_explainers(summary, structural, denovo)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {output_dir}/")
    print(f"  - protein_comparison_summary.csv")
    print(f"  - pooled_structural_context.csv")
    print(f"  - pooled_retention.csv")
    print(f"  - figures/cross_protein_main.png")
    print(f"  - figures/cross_protein_effects.png")
    print(f"  - figures/cross_protein_ranked.png")


if __name__ == "__main__":
    main()
