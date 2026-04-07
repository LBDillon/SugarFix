#!/usr/bin/env python3
"""Select the best candidate proteins for full AF3 structural validation.

Reads the cross-protein comparison summary and per-protein structural context
to pick a representative, scientifically informative subset for the expensive
MPNN → AF3 → RMSD/pTM pipeline.

Selection criteria
------------------
Hard filters (must pass all):
  1. >= 3 sequon sites          — need enough sites per protein for meaningful comparison
  2. Total length <= 2000 res   — AF3 server practical limit
  3. MPNN score < 1.0           — designs above this are low-confidence and may not fold
  4. <= 4 chains                — simpler to interpret RMSD; avoids large oligomers

Soft scoring (ranked by composite score):
  - Sequon count (more = more data points per protein)
  - Contrast between conditions (large gap unconstrained→n_only = clear signal)
  - Spread of RSA categories across sequon sites (buried + exposed = richer story)
  - Moderate functional retention in unconstrained (not 0%, not 100% — most informative)

Stratified sampling:
  After scoring, picks proteins across retention bins to ensure the final set
  covers the full retention spectrum, not just one region.

Output
------
  - af3_candidates.csv          — ranked list with scores and selection rationale
  - Console summary with the final pick list
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"
TARGET_N = 15  # how many candidates to select


def load_data():
    """Load summary and structural context from the cross-protein comparison."""
    comp_dir = DATA_DIR / "cross_protein_comparison"
    summary_path = comp_dir / "protein_comparison_summary.csv"
    struct_path = comp_dir / "pooled_structural_context.csv"

    if not summary_path.exists():
        print(f"ERROR: Run compare_all_proteins.py first.\n  Missing: {summary_path}")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    structural = pd.read_csv(struct_path) if struct_path.exists() else pd.DataFrame()
    return summary, structural


def compute_rsa_spread(structural, pdb_id):
    """Count how many RSA categories (Buried, Intermediate, Exposed) a protein's sequon sites span."""
    prot = structural[structural["pdb_id"] == pdb_id]
    if prot.empty or "rsa_category" not in prot.columns:
        return 0
    return prot["rsa_category"].nunique()


def apply_hard_filters(summary):
    """Apply must-pass filters, return filtered DataFrame and report exclusions."""
    n_before = len(summary)
    reasons = []

    mask_sequons = summary["Sequons"] >= 3
    mask_length = summary["TotalLength"] <= 2000
    mask_score = summary["mpnn_score_mean"] < 1.0
    mask_chains = summary["Chains"] <= 4

    for _, row in summary.iterrows():
        fails = []
        if not mask_sequons.loc[row.name]:
            fails.append(f"sequons={int(row['Sequons'])}<3")
        if not mask_length.loc[row.name]:
            fails.append(f"length={int(row['TotalLength'])}>2000")
        if not mask_score.loc[row.name]:
            fails.append(f"mpnn_score={row['mpnn_score_mean']:.2f}>=1.0")
        if not mask_chains.loc[row.name]:
            fails.append(f"chains={int(row['Chains'])}>4")
        if fails:
            reasons.append((row["PDB"], "; ".join(fails)))

    mask = mask_sequons & mask_length & mask_score & mask_chains
    filtered = summary[mask].copy()

    print(f"\n  Hard filters: {n_before} → {len(filtered)} proteins")
    if reasons:
        print(f"  Excluded {len(reasons)} proteins:")
        for pdb, reason in reasons:
            print(f"    {pdb}: {reason}")

    return filtered


def score_candidates(filtered, structural):
    """Score each protein on informativeness for AF3 validation."""
    scores = []
    for _, row in filtered.iterrows():
        pdb_id = row["PDB"]

        # 1. Sequon count score (log-scaled, more is better, diminishing returns)
        seq_score = np.log2(row["Sequons"]) / np.log2(42)  # normalize to max in dataset
        seq_score = min(seq_score, 1.0)

        # 2. Condition contrast: gap between unconstrained and n_only_fixed functional
        #    Bigger gap = clearer demonstration that fixing N helps but isn't enough
        gap = row["n_only_fixed_func"] - row["unconstrained_func"]
        gap_score = gap / 100.0  # normalize to 0-1

        # 3. RSA spread: proteins with sequons in multiple burial categories
        rsa_spread = compute_rsa_spread(structural, pdb_id)
        rsa_score = rsa_spread / 3.0  # max 3 categories

        # 4. "Interesting" retention: not 0% and not 100%. Peak informativeness ~10-50%
        #    Use a bell-curve-like score centered on 25%
        ret = row["unconstrained_func"]
        if ret == 0:
            ret_score = 0.3  # still useful (shows complete loss) but less informative
        elif ret >= 90:
            ret_score = 0.1  # not much to validate — sequons are kept anyway
        else:
            # Peak at ~25%, drops off above 60%
            ret_score = 1.0 - abs(ret - 25) / 75
            ret_score = max(ret_score, 0.2)

        # 5. Length preference: shorter is cheaper/faster for AF3
        length_score = 1.0 - (row["TotalLength"] / 2000.0)
        length_score = max(length_score, 0.0)

        # Composite score (weighted)
        composite = (
            0.25 * seq_score
            + 0.25 * gap_score
            + 0.15 * rsa_score
            + 0.20 * ret_score
            + 0.15 * length_score
        )

        scores.append({
            "PDB": pdb_id,
            "Sequons": int(row["Sequons"]),
            "TotalLength": int(row["TotalLength"]),
            "Chains": int(row["Chains"]),
            "unconstrained_func": round(row["unconstrained_func"], 1),
            "n_only_fixed_func": round(row["n_only_fixed_func"], 1),
            "condition_gap": round(gap, 1),
            "mpnn_score": round(row["mpnn_score_mean"], 3),
            "rsa_spread": rsa_spread,
            "seq_score": round(seq_score, 3),
            "gap_score": round(gap_score, 3),
            "rsa_score": round(rsa_score, 3),
            "ret_score": round(ret_score, 3),
            "length_score": round(length_score, 3),
            "composite_score": round(composite, 3),
        })

    scored = pd.DataFrame(scores).sort_values("composite_score", ascending=False)
    return scored


def stratified_select(scored, target_n=TARGET_N):
    """Select proteins across retention bins for balanced representation."""
    # Define retention bins
    bins = [
        ("very_low", 0, 5),       # ~0% retention (strong bias)
        ("low", 5, 15),           # low retention
        ("moderate", 15, 40),     # moderate retention
        ("high", 40, 101),        # high retention (rare, but important controls)
    ]

    # Count available per bin
    bin_counts = {}
    for label, lo, hi in bins:
        mask = (scored["unconstrained_func"] >= lo) & (scored["unconstrained_func"] < hi)
        bin_counts[label] = mask.sum()

    total_available = sum(bin_counts.values())
    print(f"\n  Retention bin distribution ({total_available} candidates):")
    for label, lo, hi in bins:
        print(f"    {label:12s} [{lo:3d}-{hi:3d}%): {bin_counts[label]} proteins")

    # Allocate slots proportional to bin size, with minimum 1 per non-empty bin
    selected = []
    remaining_slots = target_n

    # First pass: ensure at least 2 from each non-empty bin (or all if <2 available)
    for label, lo, hi in bins:
        mask = (scored["unconstrained_func"] >= lo) & (scored["unconstrained_func"] < hi)
        bin_df = scored[mask].head(min(2, bin_counts[label]))
        selected.extend(bin_df["PDB"].tolist())
        remaining_slots -= len(bin_df)

    # Second pass: fill remaining slots by composite score, avoiding duplicates
    already = set(selected)
    for _, row in scored.iterrows():
        if remaining_slots <= 0:
            break
        if row["PDB"] not in already:
            selected.append(row["PDB"])
            already.add(row["PDB"])
            remaining_slots -= 1

    # Mark selections in scored dataframe
    scored["selected"] = scored["PDB"].isin(selected)
    scored["selection_rank"] = 0
    for i, pdb in enumerate(selected, 1):
        scored.loc[scored["PDB"] == pdb, "selection_rank"] = i

    return scored, selected


def main():
    print("=" * 70)
    print("AF3 CANDIDATE SELECTION")
    print("=" * 70)

    summary, structural = load_data()
    print(f"\n  Loaded {len(summary)} proteins from cross-protein comparison")

    # Step 1: Hard filters
    filtered = apply_hard_filters(summary)

    if filtered.empty:
        print("\n  No proteins pass hard filters!")
        sys.exit(1)

    # Step 2: Score candidates
    print(f"\n  Scoring {len(filtered)} candidates...")
    scored = score_candidates(filtered, structural)

    # Step 3: Stratified selection
    scored, selected = stratified_select(scored, target_n=TARGET_N)

    # Save full scored list
    output_dir = DATA_DIR / "cross_protein_comparison"
    output_path = output_dir / "af3_candidates.csv"
    scored.to_csv(output_path, index=False)
    print(f"\n  Saved full ranked list: {output_path}")

    # Print selection
    print(f"\n{'=' * 70}")
    print(f"SELECTED {len(selected)} CANDIDATES FOR AF3 VALIDATION")
    print(f"{'=' * 70}")
    print(f"\n  {'#':>2} {'PDB':<6} {'Seq':>4} {'Len':>5} {'Ch':>3} "
          f"{'Unc Func':>9} {'N-only':>7} {'Gap':>5} {'MPNN':>6} {'RSA':>4} {'Score':>6}")
    print(f"  {'-' * 68}")

    selected_df = scored[scored["selected"]].sort_values("selection_rank")
    for _, row in selected_df.iterrows():
        print(f"  {int(row['selection_rank']):>2} {row['PDB']:<6} "
              f"{row['Sequons']:>4} {row['TotalLength']:>5} {row['Chains']:>3} "
              f"{row['unconstrained_func']:>8.1f}% {row['n_only_fixed_func']:>6.1f}% "
              f"{row['condition_gap']:>5.1f} {row['mpnn_score']:>6.3f} "
              f"{row['rsa_spread']:>4} {row['composite_score']:>6.3f}")

    # Summary statistics of selected set
    sel_summary = selected_df
    print(f"\n  Selection coverage:")
    print(f"    Retention range:  {sel_summary['unconstrained_func'].min():.1f}% – "
          f"{sel_summary['unconstrained_func'].max():.1f}%")
    print(f"    Sequon range:     {sel_summary['Sequons'].min()} – {sel_summary['Sequons'].max()}")
    print(f"    Length range:     {sel_summary['TotalLength'].min()} – {sel_summary['TotalLength'].max()} residues")
    print(f"    Chain range:      {sel_summary['Chains'].min()} – {sel_summary['Chains'].max()}")
    print(f"    Single-chain:     {(sel_summary['Chains'] == 1).sum()}")
    print(f"    Multi-chain:      {(sel_summary['Chains'] > 1).sum()}")
    total_af3_jobs = len(selected) * 5  # 2 conditions x (plain + glycan) + 1 denovo
    print(f"\n  AF3 jobs needed:    ~{total_af3_jobs} predictions "
          f"(5 per protein: 2 conditions x 2 glycan states + 1 de novo)")

    # Print the PDB list for easy copy-paste
    print(f"\n  PDB list for pipeline:")
    print(f"    {' '.join(selected)}")

    print(f"\n  Criteria applied:")
    print(f"    Hard: >= 3 sequons, <= 2000 residues, MPNN score < 1.0, <= 4 chains")
    print(f"    Soft: sequon count, condition contrast, RSA spread,")
    print(f"          informative retention range, protein length")
    print(f"    Stratified: covers very low (<5%), low (5-15%), moderate (15-40%),")
    print(f"                and high (>40%) unconstrained functional retention")


if __name__ == "__main__":
    main()
