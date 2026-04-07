#!/usr/bin/env python3
"""
Calculate baseline per-residue recovery rates in unconstrained ProteinMPNN designs.

This script answers the critical question: "Is 8-9% sequon retention evidence of
active bias against sequons, or just general low sequence recovery?"

Author: Laura Dillon
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict, Counter
import json

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MANIFEST_FILE = BASE_DIR / "data/glyco_benchmark/manifests/glycoprotein_details_simple.csv"
DESIGNS_DIR = BASE_DIR / "data/glyco_benchmark/designs"
OUTPUT_DIR = BASE_DIR / "experiments/07_baseline_recovery_analysis/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Amino acids
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def is_sequon(triplet):
    """Check if a triplet is an N-X-S/T sequon (X != P)."""
    if len(triplet) != 3:
        return False
    return (triplet[0] == 'N' and
            triplet[1] != 'P' and
            triplet[2] in ['S', 'T'])

def load_manifest():
    """Load protein manifest with native sequences."""
    df = pd.read_csv(MANIFEST_FILE)
    return df

def load_designs(pdb_id, chain_id):
    """Load designed sequences from unconstrained FASTA file."""
    fasta_file = DESIGNS_DIR / f"{pdb_id}_{chain_id}_unconstrained.fasta"

    if not fasta_file.exists():
        print(f"Warning: No unconstrained designs for {pdb_id}_{chain_id}")
        return []

    designs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        designs.append(str(record.seq))

    return designs

def calculate_per_residue_recovery(native_seq, designed_seqs):
    """Calculate per-residue recovery rate."""
    if not designed_seqs:
        return None

    # Filter designs to match native sequence length
    valid_designs = [d for d in designed_seqs if len(d) == len(native_seq)]

    if not valid_designs:
        print(f"  Warning: No designs match native length ({len(native_seq)})")
        return None

    if len(valid_designs) < len(designed_seqs):
        print(f"  Warning: {len(designed_seqs) - len(valid_designs)} designs filtered due to length mismatch")

    results = []

    for pos in range(len(native_seq)):
        native_aa = native_seq[pos]

        # Count matches across all designs
        matches = sum(1 for design in valid_designs if design[pos] == native_aa)
        recovery_rate = 100 * matches / len(valid_designs)

        results.append({
            'position': pos,
            'native_aa': native_aa,
            'recovery_rate': recovery_rate,
            'n_matches': matches,
            'n_designs': len(valid_designs)
        })

    return pd.DataFrame(results)

def calculate_triplet_recovery(native_seq, designed_seqs):
    """Calculate triplet recovery rates."""
    if not designed_seqs:
        return None

    # Filter designs to match native sequence length
    valid_designs = [d for d in designed_seqs if len(d) == len(native_seq)]

    if not valid_designs:
        return None

    results = []

    for pos in range(len(native_seq) - 2):
        native_triplet = native_seq[pos:pos+3]
        is_seq = is_sequon(native_triplet)

        # Count exact triplet matches
        matches = sum(1 for design in valid_designs
                     if design[pos:pos+3] == native_triplet)
        recovery_rate = 100 * matches / len(valid_designs)

        results.append({
            'position': pos,
            'native_triplet': native_triplet,
            'is_sequon': is_seq,
            'recovery_rate': recovery_rate,
            'n_matches': matches,
            'n_designs': len(valid_designs)
        })

    return pd.DataFrame(results)

def main():
    """Main analysis pipeline."""

    print("="*70)
    print("BASELINE PER-RESIDUE RECOVERY ANALYSIS")
    print("="*70)
    print()

    # Load manifest
    print("Loading protein manifest...")
    manifest = load_manifest()
    print(f"Found {len(manifest)} proteins\n")

    # Storage for results
    all_per_residue_results = []
    all_triplet_results = []
    protein_summaries = []

    # Process each protein
    for idx, row in manifest.iterrows():
        pdb_id = row['pdb_id']
        chain_id = row['chain_id']
        native_seq = row['sequence']

        if pd.isna(native_seq) or len(native_seq) == 0:
            print(f"Skipping {pdb_id}_{chain_id} - no sequence in manifest")
            continue

        print(f"Processing {pdb_id}_{chain_id} ({len(native_seq)} residues)...")

        # Load designs
        designs = load_designs(pdb_id, chain_id)
        if not designs:
            continue

        print(f"  Loaded {len(designs)} designs")

        # Calculate per-residue recovery
        per_res_df = calculate_per_residue_recovery(native_seq, designs)
        if per_res_df is not None:
            per_res_df['protein'] = f"{pdb_id}_{chain_id}"
            all_per_residue_results.append(per_res_df)

            overall_recovery = per_res_df['recovery_rate'].mean()
            print(f"  Overall per-residue recovery: {overall_recovery:.1f}%")

        # Calculate triplet recovery
        triplet_df = calculate_triplet_recovery(native_seq, designs)
        if triplet_df is not None:
            triplet_df['protein'] = f"{pdb_id}_{chain_id}"
            all_triplet_results.append(triplet_df)

            overall_triplet = triplet_df['recovery_rate'].mean()
            sequon_triplet = triplet_df[triplet_df['is_sequon']]['recovery_rate'].mean()
            non_sequon_triplet = triplet_df[~triplet_df['is_sequon']]['recovery_rate'].mean()

            print(f"  Overall triplet recovery: {overall_triplet:.1f}%")
            print(f"  Sequon triplet recovery: {sequon_triplet:.1f}%")
            print(f"  Non-sequon triplet recovery: {non_sequon_triplet:.1f}%")

        # Summary for this protein
        protein_summaries.append({
            'protein': f"{pdb_id}_{chain_id}",
            'length': len(native_seq),
            'n_designs': len(designs),
            'overall_per_residue_recovery': per_res_df['recovery_rate'].mean() if per_res_df is not None else np.nan,
            'overall_triplet_recovery': triplet_df['recovery_rate'].mean() if triplet_df is not None else np.nan,
            'sequon_triplet_recovery': triplet_df[triplet_df['is_sequon']]['recovery_rate'].mean() if triplet_df is not None else np.nan,
            'non_sequon_triplet_recovery': triplet_df[~triplet_df['is_sequon']]['recovery_rate'].mean() if triplet_df is not None else np.nan,
        })

        print()

    # Combine all results
    print("Combining results across all proteins...")
    per_residue_all = pd.concat(all_per_residue_results, ignore_index=True)
    triplet_all = pd.concat(all_triplet_results, ignore_index=True)
    summary_df = pd.DataFrame(protein_summaries)

    # Save detailed results
    print("Saving detailed results...")
    per_residue_all.to_csv(OUTPUT_DIR / "per_residue_recovery_detailed.csv", index=False)
    triplet_all.to_csv(OUTPUT_DIR / "triplet_recovery_detailed.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "protein_recovery_summary.csv", index=False)

    # Calculate overall statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)

    overall_per_res = per_residue_all['recovery_rate'].mean()
    overall_per_res_std = per_residue_all['recovery_rate'].std()

    overall_triplet = triplet_all['recovery_rate'].mean()
    overall_triplet_std = triplet_all['recovery_rate'].std()

    sequon_triplets = triplet_all[triplet_all['is_sequon']]
    sequon_recovery = sequon_triplets['recovery_rate'].mean()
    sequon_recovery_std = sequon_triplets['recovery_rate'].std()

    non_sequon_triplets = triplet_all[~triplet_all['is_sequon']]
    non_sequon_recovery = non_sequon_triplets['recovery_rate'].mean()
    non_sequon_recovery_std = non_sequon_triplets['recovery_rate'].std()

    print(f"\nPer-Residue Recovery:")
    print(f"  Mean: {overall_per_res:.1f}% ± {overall_per_res_std:.1f}%")
    print(f"  Median: {per_residue_all['recovery_rate'].median():.1f}%")
    print(f"  Range: {per_residue_all['recovery_rate'].min():.1f}% - {per_residue_all['recovery_rate'].max():.1f}%")

    print(f"\nTriplet Recovery:")
    print(f"  All triplets: {overall_triplet:.1f}% ± {overall_triplet_std:.1f}%")
    print(f"  Sequon triplets (N-X-S/T): {sequon_recovery:.1f}% ± {sequon_recovery_std:.1f}% (n={len(sequon_triplets)})")
    print(f"  Non-sequon triplets: {non_sequon_recovery:.1f}% ± {non_sequon_recovery_std:.1f}% (n={len(non_sequon_triplets)})")

    # Per-amino-acid recovery
    print(f"\nPer-Amino-Acid Recovery:")
    aa_recovery = per_residue_all.groupby('native_aa')['recovery_rate'].agg(['mean', 'std', 'count'])
    aa_recovery = aa_recovery.sort_values('mean', ascending=False)

    for aa in AA_LIST:
        if aa in aa_recovery.index:
            mean_rec = aa_recovery.loc[aa, 'mean']
            std_rec = aa_recovery.loc[aa, 'std']
            count = aa_recovery.loc[aa, 'count']
            print(f"  {aa}: {mean_rec:5.1f}% ± {std_rec:4.1f}% (n={int(count):4d})")

    # Highlight sequon-relevant amino acids
    print(f"\nSequon-Relevant Amino Acids:")
    for aa in ['N', 'S', 'T']:
        if aa in aa_recovery.index:
            mean_rec = aa_recovery.loc[aa, 'mean']
            std_rec = aa_recovery.loc[aa, 'std']
            print(f"  {aa}: {mean_rec:.1f}% ± {std_rec:.1f}%")

    # Statistical comparison
    from scipy import stats

    print(f"\nStatistical Comparison:")
    print(f"  Sequon vs Non-Sequon Triplets:")
    t_stat, p_value = stats.mannwhitneyu(
        sequon_triplets['recovery_rate'],
        non_sequon_triplets['recovery_rate'],
        alternative='two-sided'
    )
    print(f"    Mann-Whitney U test: p = {p_value:.4f}")

    if p_value < 0.05:
        if sequon_recovery < non_sequon_recovery:
            print(f"    ✓ Sequons are recovered LESS than other triplets (p < 0.05)")
        else:
            print(f"    ✓ Sequons are recovered MORE than other triplets (p < 0.05)")
    else:
        print(f"    No significant difference (p >= 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((sequon_recovery_std**2 + non_sequon_recovery_std**2) / 2)
    cohens_d = (sequon_recovery - non_sequon_recovery) / pooled_std
    print(f"    Cohen's d: {cohens_d:.3f}")

    # Interpretation
    print(f"\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if overall_per_res > 30:
        print(f"\nBaseline per-residue recovery is HIGH ({overall_per_res:.1f}%).")
        print(f"Sequon triplet recovery ({sequon_recovery:.1f}%) is MUCH LOWER than baseline.")
        print(f"→ This suggests ProteinMPNN ACTIVELY DISFAVORS N-X-S/T sequons.")
    elif overall_per_res < 15:
        print(f"\nBaseline per-residue recovery is LOW ({overall_per_res:.1f}%).")
        print(f"Sequon triplet recovery ({sequon_recovery:.1f}%) is similar to overall triplet recovery.")
        print(f"→ This suggests ProteinMPNN has GENERAL LOW FIDELITY, not specific anti-sequon bias.")
    else:
        print(f"\nBaseline per-residue recovery is MODERATE ({overall_per_res:.1f}%).")
        print(f"Sequon triplet recovery ({sequon_recovery:.1f}%) is lower than baseline.")
        print(f"→ Further analysis needed to determine if this is statistically significant.")

    # Save summary statistics
    summary_stats = {
        'overall_per_residue_recovery_mean': float(overall_per_res),
        'overall_per_residue_recovery_std': float(overall_per_res_std),
        'overall_triplet_recovery_mean': float(overall_triplet),
        'overall_triplet_recovery_std': float(overall_triplet_std),
        'sequon_triplet_recovery_mean': float(sequon_recovery),
        'sequon_triplet_recovery_std': float(sequon_recovery_std),
        'non_sequon_triplet_recovery_mean': float(non_sequon_recovery),
        'non_sequon_triplet_recovery_std': float(non_sequon_recovery_std),
        'mannwhitney_p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'n_sequon_triplets': int(len(sequon_triplets)),
        'n_non_sequon_triplets': int(len(non_sequon_triplets)),
        'n_proteins': len(summary_df),
        'total_residues': len(per_residue_all)
    }

    with open(OUTPUT_DIR / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - per_residue_recovery_detailed.csv")
    print(f"  - triplet_recovery_detailed.csv")
    print(f"  - protein_recovery_summary.csv")
    print(f"  - summary_statistics.json")
    print("\nDone!")

if __name__ == "__main__":
    main()
