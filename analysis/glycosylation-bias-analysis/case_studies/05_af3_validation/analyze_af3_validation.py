#!/usr/bin/env python3
"""Analyze AF3 structural validation results: RMSD + confidence metrics.

Combines per-chain RMSD data from PyMOL alignments with AF3 confidence
metrics (pTM, ipTM, ranking_score) to assess whether ProteinMPNN designs
that lost glycosylation sequons still fold correctly.

Reads from:
  af3_results/*/rmsd_result.csv     (PyMOL chain-by-chain RMSD)
  af3_results/aggregate_summary.csv (AF3 confidence, best seed per condition)

Outputs:
  af3_results/analysis/
    af3_validation_combined.csv     (merged RMSD + confidence per protein/condition)
    af3_validation_main.png         (Figure: RMSD + pTM comparison)
    af3_validation_glycan_effect.png (Figure: effect of adding glycans)
"""

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
AF3_DIR = DATA_DIR / "af3_results"

# Proteins to exclude from analysis (with reasons)
# 1C1Z: Beta-2-glycoprotein-I — monomeric protein that appears as 4 identical
# chains in the crystal structure due to crystallographic symmetry. AF3 struggles
# with this ambiguous oligomeric state, producing RMSD 2-5A vs <1A for other proteins.
EXCLUDE_PDBS = {"1C1Z"}

# Conditions to analyze (exclude unconstrained_custom)
STANDARD_CONDITIONS = [
    "unconstrained",
    "full_sequon_fixed",
    "full_sequon_fixed_full_glycans",
    "full_sequon_fixed_with_glycans",
    "unconstrained_denovo_glycans",
]

# Labels and colors
COND_LABELS = {
    "unconstrained": "Unconstrained",
    "full_sequon_fixed": "Full Sequon\nFixed",
    "full_sequon_fixed_full_glycans": "Full Sequon Fixed\n+ Full Glycans",
    "full_sequon_fixed_with_glycans": "Full Sequon Fixed\n+ Glycans",
    "unconstrained_denovo_glycans": "Unconstrained\n+ De Novo Glycans",
}
COND_SHORT = {
    "unconstrained": "Unc",
    "full_sequon_fixed": "Fixed",
    "full_sequon_fixed_full_glycans": "Fixed+FullGlyc",
    "full_sequon_fixed_with_glycans": "Fixed+Glyc",
    "unconstrained_denovo_glycans": "Unc+DeNovo",
}
COND_COLORS = {
    "unconstrained": "#d95f02",
    "full_sequon_fixed": "#1b9e77",
    "full_sequon_fixed_full_glycans": "#17a589",
    "full_sequon_fixed_with_glycans": "#66a61e",
    "unconstrained_denovo_glycans": "#e7298a",
}

# Protein-only conditions (no glycan ligands in AF3 input)
PROTEIN_ONLY = {"unconstrained", "full_sequon_fixed"}
# Conditions with glycan ligands
WITH_GLYCANS = {
    "full_sequon_fixed_full_glycans",
    "full_sequon_fixed_with_glycans",
    "unconstrained_denovo_glycans",
}
FIXED_GLYCAN_CANDIDATES = [
    "full_sequon_fixed_full_glycans",
    "full_sequon_fixed_with_glycans",
]


def _preferred_fixed_glycan_condition(merged):
    available = set(merged["condition"].dropna())
    for cond in FIXED_GLYCAN_CANDIDATES:
        if cond in available:
            return cond
    return None


def _sig_label(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return "ns"


def load_rmsd_data():
    """Load all per-protein RMSD CSVs and return combined DataFrame."""
    rows = []
    for pdb_dir in sorted(AF3_DIR.iterdir()):
        if not pdb_dir.is_dir():
            continue
        # Try both naming conventions
        for fname in ["rmsd_result.csv", "rmsd_results.csv"]:
            csv_path = pdb_dir / fname
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["pdb_id"] = pdb_dir.name
                rows.append(df)
                break

    if not rows:
        print("ERROR: No RMSD CSVs found in af3_results/*/")
        sys.exit(1)

    combined = pd.concat(rows, ignore_index=True)

    # Parse condition from structure name (e.g., "4ARN_unconstrained" -> "unconstrained")
    def parse_condition(structure_name):
        pdb = structure_name.split("_")[0]
        cond = structure_name[len(pdb) + 1:]
        return cond

    combined["condition"] = combined["structure"].apply(parse_condition)

    # Filter out unconstrained_custom and excluded proteins
    combined = combined[combined["condition"] != "unconstrained_custom"].copy()
    if EXCLUDE_PDBS:
        n_before = len(combined)
        combined = combined[~combined["pdb_id"].isin(EXCLUDE_PDBS)].copy()
        n_excluded = n_before - len(combined)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} rows from {EXCLUDE_PDBS}")

    # Separate mean rows from per-chain rows
    mean_rmsd = combined[combined["ref_chain"] == "mean"].copy()
    chain_rmsd = combined[combined["ref_chain"] != "mean"].copy()

    return mean_rmsd, chain_rmsd


def load_confidence_data():
    """Load aggregate confidence summary."""
    path = AF3_DIR / "aggregate_summary.csv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    df = pd.read_csv(path)
    df = df[df["condition"] != "unconstrained_custom"].copy()
    if EXCLUDE_PDBS:
        df = df[~df["pdb_id"].isin(EXCLUDE_PDBS)].copy()

    # Extract protein-chain pTMs (exclude glycan chains which are all ~0.04)
    # For protein-only conditions, the first N chain_X_ptm columns are real chains
    # For glycan conditions, real protein chains come first, then 0.04s for glycans
    protein_chain_ptms = []
    for _, row in df.iterrows():
        chain_cols = [c for c in df.columns if c.startswith("chain_") and c.endswith("_ptm")]
        ptms = []
        for col in chain_cols:
            val = row.get(col)
            if pd.notna(val) and val > 0.1:  # Real protein chains have pTM >> 0.04
                ptms.append(val)
        protein_chain_ptms.append(np.mean(ptms) if ptms else np.nan)

    df["protein_chain_ptm"] = protein_chain_ptms

    return df


def merge_data(mean_rmsd, confidence):
    """Merge RMSD and confidence data."""
    # Mean RMSD per protein/condition
    rmsd_agg = mean_rmsd[["pdb_id", "condition", "rmsd"]].rename(columns={"rmsd": "mean_rmsd"})

    merged = pd.merge(rmsd_agg, confidence,
                      on=["pdb_id", "condition"], how="outer")
    return merged


def create_main_figure(merged, chain_rmsd, output_dir):
    """Figure: Clean AF3 structural validation — 3 focused panels.

    A. Per-protein RMSD: unconstrained vs fixed (paired dot plot)
       → Main message: both conditions fold equally well
    B. Protein-chain pTM distribution across protein-only conditions
       → Main message: AF3 is confident in both designs
    C. Effect of adding glycans on RMSD (delta bar plot)
       → Main message: glycans don't significantly help or hurt
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    n_proteins = merged["pdb_id"].nunique()

    # ---- Prepare paired data: unconstrained vs full_sequon_fixed ----
    unc = merged[merged["condition"] == "unconstrained"][["pdb_id", "mean_rmsd", "protein_chain_ptm"]]
    fsf = merged[merged["condition"] == "full_sequon_fixed"][["pdb_id", "mean_rmsd", "protein_chain_ptm"]]
    paired = pd.merge(unc, fsf, on="pdb_id", suffixes=("_unc", "_fsf")).dropna(subset=["mean_rmsd_unc", "mean_rmsd_fsf"])
    paired = paired.sort_values("mean_rmsd_unc")

    # --- Panel A: Paired dot plot — RMSD unconstrained vs fixed ---
    ax = axes[0]
    y = np.arange(len(paired))
    ax.scatter(paired["mean_rmsd_unc"], y, color=COND_COLORS["unconstrained"],
               s=70, zorder=3, label="Unconstrained", edgecolors="black", linewidth=0.5)
    ax.scatter(paired["mean_rmsd_fsf"], y, color=COND_COLORS["full_sequon_fixed"],
               s=70, zorder=3, label="Full Sequon Fixed", marker="s",
               edgecolors="black", linewidth=0.5)
    # Connect pairs
    for i, (_, row) in enumerate(paired.iterrows()):
        ax.plot([row["mean_rmsd_unc"], row["mean_rmsd_fsf"]], [i, i],
                color="gray", linewidth=1, alpha=0.5, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(paired["pdb_id"], fontsize=10)
    ax.set_xlabel("Mean RMSD to Crystal Structure (A)", fontsize=12)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(fontsize=10, loc="lower right")

    # Wilcoxon test
    stat, pval = stats.wilcoxon(paired["mean_rmsd_unc"], paired["mean_rmsd_fsf"])
    sig = _sig_label(pval)
    ax.set_title(f"A. RMSD: Sequon Loss Does Not Impair Folding\n"
                 f"Wilcoxon p={pval:.2f} ({sig}), n={len(paired)} proteins",
                 fontsize=11, fontweight="bold")

    # --- Panel B: pTM strip plot — protein-only conditions ---
    ax = axes[1]
    protein_only_conds = ["unconstrained", "full_sequon_fixed"]
    ptm_rows = []
    for _, row in merged.iterrows():
        if row["condition"] in protein_only_conds and pd.notna(row.get("protein_chain_ptm")):
            ptm_rows.append({
                "Condition": "Unconstrained" if row["condition"] == "unconstrained" else "Full Sequon Fixed",
                "pTM": row["protein_chain_ptm"],
                "PDB": row["pdb_id"],
            })
    ptm_df = pd.DataFrame(ptm_rows)
    order = ["Unconstrained", "Full Sequon Fixed"]
    palette = {"Unconstrained": COND_COLORS["unconstrained"],
               "Full Sequon Fixed": COND_COLORS["full_sequon_fixed"]}

    sns.stripplot(data=ptm_df, x="Condition", y="pTM", order=order,
                  palette=palette, size=9, alpha=0.7, jitter=0.15, ax=ax,
                  edgecolor="black", linewidth=0.5)
    # Add median lines
    for i, cond in enumerate(order):
        vals = ptm_df[ptm_df["Condition"] == cond]["pTM"]
        med = vals.median()
        ax.plot([i - 0.25, i + 0.25], [med, med], color="black", linewidth=2.5, zorder=5)
        ax.text(i + 0.28, med, f"{med:.2f}", fontsize=10, va="center", fontweight="bold")

    # Wilcoxon on paired pTM
    paired_ptm = pd.merge(
        merged[merged["condition"] == "unconstrained"][["pdb_id", "protein_chain_ptm"]],
        merged[merged["condition"] == "full_sequon_fixed"][["pdb_id", "protein_chain_ptm"]],
        on="pdb_id", suffixes=("_unc", "_fsf")
    ).dropna()
    if len(paired_ptm) >= 5:
        stat, pval_ptm = stats.wilcoxon(paired_ptm["protein_chain_ptm_unc"],
                                        paired_ptm["protein_chain_ptm_fsf"])
        sig_ptm = _sig_label(pval_ptm)
    else:
        pval_ptm, sig_ptm = np.nan, "n/a"

    ax.set_xlabel("")
    ax.set_ylabel("Protein Chain pTM", fontsize=12)
    ax.set_title(f"B. AF3 Confidence Is High for Both Conditions\n"
                 f"Wilcoxon p={pval_ptm:.2f} ({sig_ptm})",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.4)
    ax.text(1.45, 0.805, "pTM = 0.8", fontsize=8, color="gray")

    # --- Panel C: Delta RMSD when adding glycans (horizontal bar) ---
    ax = axes[2]
    # Full sequon fixed: with vs without glycans
    glycan_condition = _preferred_fixed_glycan_condition(merged)
    fsf_plain = merged[merged["condition"] == "full_sequon_fixed"][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "without"})
    fsf_glyc = merged[merged["condition"] == glycan_condition][["pdb_id", "mean_rmsd"]].rename(
        columns={"mean_rmsd": "with_glyc"})
    glyc_paired = pd.merge(fsf_plain, fsf_glyc, on="pdb_id").dropna()
    glyc_paired["delta"] = glyc_paired["with_glyc"] - glyc_paired["without"]
    glyc_paired = glyc_paired.sort_values("delta")

    if glycan_condition and not glyc_paired.empty:
        bar_colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in glyc_paired["delta"]]
        y_pos = np.arange(len(glyc_paired))
        ax.barh(y_pos, glyc_paired["delta"], color=bar_colors,
                alpha=0.8, edgecolor="black", linewidth=0.5, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(glyc_paired["pdb_id"], fontsize=10)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Delta RMSD (A): with glycans - without", fontsize=11)

        n_better = (glyc_paired["delta"] < 0).sum()
        n_worse = (glyc_paired["delta"] > 0).sum()
        if len(glyc_paired) >= 5:
            stat, pval_g = stats.wilcoxon(glyc_paired["delta"])
            sig_g = _sig_label(pval_g)
        else:
            pval_g, sig_g = np.nan, "n/a"

        ax.set_title(f"C. {COND_LABELS[glycan_condition].replace(chr(10), ' ')}\n"
                     f"Improved {n_better}/{len(glyc_paired)}, "
                     f"worsened {n_worse}/{len(glyc_paired)}, "
                     f"Wilcoxon p={pval_g:.2f} ({sig_g})",
                     fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No paired glycan condition available",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("C. Glycan Condition Comparison",
                     fontsize=11, fontweight="bold")

        
    plt.suptitle(f"AF3 Structural Validation of ProteinMPNN Designs (n={n_proteins} proteins)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "af3_validation_main.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _annotate_sig(ax, x1, x2, y, pval, fontsize=9, direction="up"):
    """Draw a significance bracket."""
    label = _sig_label(pval)
    offset = 0.02 if direction == "up" else -0.02
    ax.plot([x1, x1, x2, x2],
            [y, y + offset, y + offset, y], color="black", linewidth=1)
    va = "bottom" if direction == "up" else "top"
    ax.text((x1 + x2) / 2, y + offset, f"{label}\np={pval:.2e}",
            ha="center", va=va, fontsize=fontsize)


def print_statistical_summary(merged):
    """Print all key statistical tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    n_proteins = merged["pdb_id"].nunique()

    # 1. Overall RMSD by condition
    print(f"\n1. Mean RMSD by Condition (n={n_proteins} proteins):")
    for cond in STANDARD_CONDITIONS:
        vals = merged[merged["condition"] == cond]["mean_rmsd"].dropna()
        if not vals.empty:
            print(f"   {COND_SHORT[cond]:>12s}: {vals.median():.3f} A (median), "
                  f"{vals.mean():.3f} A (mean), n={len(vals)}")

    # 2. Paired test: unconstrained vs full_sequon_fixed
    print(f"\n2. Unconstrained vs Full Sequon Fixed (paired Wilcoxon):")
    unc = merged[merged["condition"] == "unconstrained"].set_index("pdb_id")["mean_rmsd"]
    fsf = merged[merged["condition"] == "full_sequon_fixed"].set_index("pdb_id")["mean_rmsd"]
    common = unc.index.intersection(fsf.index)
    if len(common) >= 5:
        stat, pval = stats.wilcoxon(unc[common], fsf[common])
        print(f"   Wilcoxon p = {pval:.4e} ({_sig_label(pval)})")
        diff = unc[common] - fsf[common]
        print(f"   Mean delta RMSD (Unc - Fixed): {diff.mean():+.3f} A")
        print(f"   Unconstrained higher in {(diff > 0).sum()}/{len(diff)} proteins")
    else:
        print(f"   Insufficient paired data (n={len(common)})")

    # 3. Protein-chain pTM by condition
    print(f"\n3. Protein-Chain pTM by Condition:")
    for cond in STANDARD_CONDITIONS:
        vals = merged[merged["condition"] == cond]["protein_chain_ptm"].dropna()
        if not vals.empty:
            print(f"   {COND_SHORT[cond]:>12s}: {vals.median():.3f} (median), "
                  f"{vals.mean():.3f} (mean), n={len(vals)}")

    # 4. Effect of adding glycans to full_sequon_fixed
    glycan_condition = _preferred_fixed_glycan_condition(merged)
    glycan_label = COND_SHORT.get(glycan_condition, "glycan condition")
    print(f"\n4. Effect of Adding Glycans (Full Sequon Fixed):")
    fsf_rmsd = merged[merged["condition"] == "full_sequon_fixed"].set_index("pdb_id")["mean_rmsd"]
    fsf_g_rmsd = merged[merged["condition"] == glycan_condition].set_index("pdb_id")["mean_rmsd"]
    common_g = fsf_rmsd.index.intersection(fsf_g_rmsd.index)
    if len(common_g) >= 5:
        delta = fsf_g_rmsd[common_g] - fsf_rmsd[common_g]
        stat, pval = stats.wilcoxon(delta)
        print(f"   Using comparison condition: {glycan_label}")
        print(f"   RMSD delta (with_glycans - without): {delta.median():+.3f} A (median)")
        print(f"   Wilcoxon p = {pval:.4e} ({_sig_label(pval)})")
        print(f"   Glycans improved RMSD in {(delta < 0).sum()}/{len(delta)} proteins")

    fsf_ptm = merged[merged["condition"] == "full_sequon_fixed"].set_index("pdb_id")["protein_chain_ptm"]
    fsf_g_ptm = merged[merged["condition"] == glycan_condition].set_index("pdb_id")["protein_chain_ptm"]
    common_ptm = fsf_ptm.dropna().index.intersection(fsf_g_ptm.dropna().index)
    if len(common_ptm) >= 5:
        delta_ptm = fsf_g_ptm[common_ptm] - fsf_ptm[common_ptm]
        stat, pval = stats.wilcoxon(delta_ptm)
        print(f"   pTM delta (with_glycans - without): {delta_ptm.median():+.3f} (median)")
        print(f"   Wilcoxon p = {pval:.4e} ({_sig_label(pval)})")

    # 5. Effect of adding de novo glycans to unconstrained
    print(f"\n5. Effect of Adding De Novo Glycans (Unconstrained):")
    unc_rmsd = merged[merged["condition"] == "unconstrained"].set_index("pdb_id")["mean_rmsd"]
    dn_rmsd = merged[merged["condition"] == "unconstrained_denovo_glycans"].set_index("pdb_id")["mean_rmsd"]
    common_dn = unc_rmsd.index.intersection(dn_rmsd.index)
    if len(common_dn) >= 5:
        delta = dn_rmsd[common_dn] - unc_rmsd[common_dn]
        stat, pval = stats.wilcoxon(delta)
        print(f"   RMSD delta (denovo - without): {delta.median():+.3f} A (median)")
        print(f"   Wilcoxon p = {pval:.4e} ({_sig_label(pval)})")
        print(f"   De novo glycans improved RMSD in {(delta < 0).sum()}/{len(delta)} proteins")

    # 6. RMSD-pTM correlation
    print(f"\n6. RMSD vs pTM Correlation (pooled across conditions):")
    valid = merged.dropna(subset=["mean_rmsd", "protein_chain_ptm"])
    valid = valid[valid["condition"].isin(STANDARD_CONDITIONS)]
    if len(valid) >= 5:
        rho, pval = stats.spearmanr(valid["mean_rmsd"], valid["protein_chain_ptm"])
        print(f"   Spearman rho = {rho:.3f}, p = {pval:.2e} ({_sig_label(pval)})")

    # 7. Key conclusion
    print(f"\n7. Key Conclusion:")
    all_rmsd = merged[merged["condition"].isin(STANDARD_CONDITIONS)]["mean_rmsd"].dropna()
    n_under_1 = (all_rmsd < 1.0).sum()
    n_under_2 = (all_rmsd < 2.0).sum()
    print(f"   {n_under_1}/{len(all_rmsd)} predictions ({n_under_1/len(all_rmsd)*100:.0f}%) "
          f"have RMSD < 1.0 A")
    print(f"   {n_under_2}/{len(all_rmsd)} predictions ({n_under_2/len(all_rmsd)*100:.0f}%) "
          f"have RMSD < 2.0 A")
    print(f"   Designs that lost glycosylation sequons fold to near-native structures")


def main():
    print("=" * 70)
    print("AF3 STRUCTURAL VALIDATION ANALYSIS")
    print("=" * 70)

    output_dir = AF3_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading RMSD data...")
    mean_rmsd, chain_rmsd = load_rmsd_data()
    print(f"  {len(mean_rmsd)} mean RMSD entries across {mean_rmsd['pdb_id'].nunique()} proteins")
    print(f"  {len(chain_rmsd)} per-chain RMSD entries")

    print("\nLoading confidence data...")
    confidence = load_confidence_data()
    print(f"  {len(confidence)} confidence entries")

    print("\nMerging data...")
    merged = merge_data(mean_rmsd, confidence)
    merged = merged[merged["condition"].isin(STANDARD_CONDITIONS)].copy()
    print(f"  {len(merged)} merged entries across {merged['pdb_id'].nunique()} proteins")

    # Save combined CSV
    csv_path = output_dir / "af3_validation_combined.csv"
    cols_to_save = ["pdb_id", "condition", "mean_rmsd", "ptm", "iptm",
                    "ranking_score", "protein_chain_ptm", "fraction_disordered", "has_clash"]
    merged[cols_to_save].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("PER-PROTEIN SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'PDB':<6} {'Condition':<28} {'RMSD':>6} {'pTM':>5} {'ChainpTM':>9} {'Rank':>5}")
    print(f"  {'-' * 62}")
    current = None
    for _, row in merged.sort_values(["pdb_id", "condition"]).iterrows():
        if row["pdb_id"] != current:
            if current is not None:
                print()
            current = row["pdb_id"]
        rmsd = f"{row['mean_rmsd']:.3f}" if pd.notna(row.get("mean_rmsd")) else "N/A"
        ptm = f"{row['ptm']:.2f}" if pd.notna(row.get("ptm")) else "N/A"
        cptm = f"{row['protein_chain_ptm']:.2f}" if pd.notna(row.get("protein_chain_ptm")) else "N/A"
        rank = f"{row['ranking_score']:.2f}" if pd.notna(row.get("ranking_score")) else "N/A"
        cond_short = COND_SHORT.get(row["condition"], row["condition"][:12])
        print(f"  {row['pdb_id']:<6} {cond_short:<28} {rmsd:>6} {ptm:>5} {cptm:>9} {rank:>5}")

    # Statistical tests
    print_statistical_summary(merged)

    # Create figures
    print(f"\n{'=' * 70}")
    print("CREATING FIGURES")
    print(f"{'=' * 70}")
    create_main_figure(merged, chain_rmsd, output_dir)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutputs in: {output_dir}/")
    print(f"  - af3_validation_combined.csv")
    print(f"  - af3_validation_main.png")


if __name__ == "__main__":
    main()
