#!/usr/bin/env python3
"""
Corrected Sequon Analysis

This analysis addresses the fundamental problem with comparing "sequon retention"
between glycosylated and non-glycosylated proteins. The N-X-S/T motifs in
non-glycosylated proteins are NOT sequons - they're coincidental patterns.

Better comparisons:
1. Sequon-N vs non-sequon-N retention within glycosylated proteins
2. Observed vs expected (null model) sequon retention in glycosylated proteins
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict


def parse_fasta_sequences(fasta_path):
    """Parse FASTA file and return sequences dict.

    Returns dict with full header (without '>') as key and sequence as value.
    """
    sequences = {}
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_seq)
                # Keep the full header (without '>') to check for 'sample='
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            sequences[current_header] = ''.join(current_seq)

    return sequences


def find_sequon_positions(sequence):
    """Find all N-X-S/T positions where X != P (0-indexed)."""
    sequons = []
    for i in range(len(sequence) - 2):
        if sequence[i] == 'N':
            x = sequence[i+1]
            st = sequence[i+2]
            if x != 'P' and st in ['S', 'T']:
                sequons.append({
                    'position': i,
                    'sequon': sequence[i:i+3],
                    'n_pos': i,
                    'x_pos': i+1,
                    'st_pos': i+2
                })
    return sequons


def find_all_n_positions(sequence):
    """Find all N positions in a sequence (0-indexed)."""
    return [i for i, aa in enumerate(sequence) if aa == 'N']


def analyze_glycosylated_proteins(results_dir, pdb_dir):
    """
    Perform the corrected analysis on glycosylated proteins only.

    Key comparisons:
    1. N retention at sequon positions vs non-sequon positions
    2. Observed vs expected sequon retention (null model)
    """

    results_dir = Path(results_dir)
    pdb_dir = Path(pdb_dir)

    # Collect all data
    all_data = []

    for pdb_subdir in results_dir.iterdir():
        if not pdb_subdir.is_dir():
            continue

        pdb_id = pdb_subdir.name

        # The structure is: results_full_gly/{pdb_id}/designs/unconstrained/seqs/
        seqs_dir = pdb_subdir / 'designs' / 'unconstrained' / 'seqs'

        if not seqs_dir.exists():
            continue

        # Find the design FASTA file
        fasta_files = list(seqs_dir.glob('*.fa'))
        if not fasta_files:
            continue

        fasta_file = fasta_files[0]
        sequences = parse_fasta_sequences(fasta_file)

        if not sequences:
            continue

        # Separate WT and designed sequences
        # First entry (without "sample=") is WT, rest are designs
        wt_seq = None
        designs = []

        for header, seq in sequences.items():
            if 'sample=' not in header:
                # This is the WT
                wt_seq = seq
            else:
                # This is a design
                designs.append(seq)

        if wt_seq is None or not designs:
            print(f"  Skipping {pdb_id}: no WT or designs found")
            continue

        # Treat as single chain (chain A)
        chain = 'A'
        n_designs = len(designs)

        # Find sequon positions and all N positions
        sequons = find_sequon_positions(wt_seq)
        all_n_positions = find_all_n_positions(wt_seq)

        # Identify sequon-N vs non-sequon-N positions
        sequon_n_positions = set(s['n_pos'] for s in sequons)
        non_sequon_n_positions = set(all_n_positions) - sequon_n_positions

        # Calculate retentions
        sequon_n_retained = 0
        sequon_n_total = 0
        non_sequon_n_retained = 0
        non_sequon_n_total = 0

        # Also track X and S/T retention at sequon positions
        x_retained = 0
        x_total = 0
        st_retained = 0
        st_total = 0

        # Full sequon retention
        full_sequon_retained = 0
        full_sequon_total = 0

        for design in designs:
            # Sequon-N retention
            for pos in sequon_n_positions:
                if pos < len(design) and pos < len(wt_seq):
                    sequon_n_total += 1
                    if design[pos] == wt_seq[pos]:
                        sequon_n_retained += 1

            # Non-sequon-N retention
            for pos in non_sequon_n_positions:
                if pos < len(design) and pos < len(wt_seq):
                    non_sequon_n_total += 1
                    if design[pos] == wt_seq[pos]:
                        non_sequon_n_retained += 1

            # X and S/T retention at sequon positions
            for sequon in sequons:
                x_pos = sequon['x_pos']
                st_pos = sequon['st_pos']

                if x_pos < len(design) and x_pos < len(wt_seq):
                    x_total += 1
                    if design[x_pos] == wt_seq[x_pos]:
                        x_retained += 1

                if st_pos < len(design) and st_pos < len(wt_seq):
                    st_total += 1
                    if design[st_pos] == wt_seq[st_pos]:
                        st_retained += 1

                # Full sequon
                full_sequon_total += 1
                n_pos = sequon['n_pos']
                if (n_pos < len(design) and x_pos < len(design) and st_pos < len(design) and
                    n_pos < len(wt_seq) and x_pos < len(wt_seq) and st_pos < len(wt_seq)):
                    if (design[n_pos] == wt_seq[n_pos] and
                        design[x_pos] == wt_seq[x_pos] and
                        design[st_pos] == wt_seq[st_pos]):
                        full_sequon_retained += 1

        all_data.append({
            'pdb_id': pdb_id,
            'chain': chain,
            'n_sequons': len(sequons),
            'n_non_sequon_N': len(non_sequon_n_positions),
            'n_designs': n_designs,
            # Sequon-N
            'sequon_n_retained': sequon_n_retained,
            'sequon_n_total': sequon_n_total,
            'sequon_n_pct': 100 * sequon_n_retained / sequon_n_total if sequon_n_total > 0 else np.nan,
            # Non-sequon-N
            'non_sequon_n_retained': non_sequon_n_retained,
            'non_sequon_n_total': non_sequon_n_total,
            'non_sequon_n_pct': 100 * non_sequon_n_retained / non_sequon_n_total if non_sequon_n_total > 0 else np.nan,
            # X position
            'x_retained': x_retained,
            'x_total': x_total,
            'x_pct': 100 * x_retained / x_total if x_total > 0 else np.nan,
            # S/T position
            'st_retained': st_retained,
            'st_total': st_total,
            'st_pct': 100 * st_retained / st_total if st_total > 0 else np.nan,
            # Full sequon
            'sequon_retained': full_sequon_retained,
            'sequon_total': full_sequon_total,
            'sequon_pct': 100 * full_sequon_retained / full_sequon_total if full_sequon_total > 0 else np.nan,
        })

    return pd.DataFrame(all_data)


def calculate_null_expectation(df):
    """
    Calculate the null expectation for sequon retention.

    If N, X, and S/T were retained independently at their observed rates,
    what would full-sequon retention be?

    Expected = P(N) × P(X) × P(S/T)
    """

    # Aggregate across all proteins
    total_n_retained = df['sequon_n_retained'].sum()
    total_n_observations = df['sequon_n_total'].sum()

    total_x_retained = df['x_retained'].sum()
    total_x_observations = df['x_total'].sum()

    total_st_retained = df['st_retained'].sum()
    total_st_observations = df['st_total'].sum()

    total_sequon_retained = df['sequon_retained'].sum()
    total_sequon_observations = df['sequon_total'].sum()

    # Calculate probabilities
    p_n = total_n_retained / total_n_observations if total_n_observations > 0 else 0
    p_x = total_x_retained / total_x_observations if total_x_observations > 0 else 0
    p_st = total_st_retained / total_st_observations if total_st_observations > 0 else 0

    # Null expectation (independent retention)
    p_expected = p_n * p_x * p_st

    # Observed
    p_observed = total_sequon_retained / total_sequon_observations if total_sequon_observations > 0 else 0

    return {
        'p_n': p_n,
        'p_x': p_x,
        'p_st': p_st,
        'p_expected': p_expected,
        'p_observed': p_observed,
        'ratio': p_observed / p_expected if p_expected > 0 else np.nan,
        'total_n_observations': total_n_observations,
        'total_sequon_observations': total_sequon_observations
    }


def main():
    base_dir = Path(__file__).parent.parent

    # Paths for glycosylated proteins only
    results_gly = base_dir / 'results_full_gly'
    pdb_gly = base_dir / 'PDBs_gly'

    print("="*70)
    print("CORRECTED SEQUON ANALYSIS - GLYCOSYLATED PROTEINS ONLY")
    print("="*70)

    # Run analysis
    print("\nAnalyzing glycosylated proteins...")
    df = analyze_glycosylated_proteins(results_gly, pdb_gly)

    if len(df) == 0:
        print("No data found!")
        return

    print(f"Analyzed {len(df)} protein chains")

    # Save detailed results
    output_file = base_dir / 'corrected_sequon_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved detailed results to: {output_file}")

    # =================================================================
    # ANALYSIS 1: Sequon-N vs Non-sequon-N retention
    # =================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Does MPNN treat sequon-N differently from non-sequon-N?")
    print("="*70)

    # Filter to chains with both sequon and non-sequon N
    df_both = df[(df['sequon_n_total'] > 0) & (df['non_sequon_n_total'] > 0)]

    # Aggregate statistics
    total_sequon_n = df['sequon_n_total'].sum()
    total_sequon_n_retained = df['sequon_n_retained'].sum()
    sequon_n_rate = 100 * total_sequon_n_retained / total_sequon_n if total_sequon_n > 0 else 0

    total_non_sequon_n = df['non_sequon_n_total'].sum()
    total_non_sequon_n_retained = df['non_sequon_n_retained'].sum()
    non_sequon_n_rate = 100 * total_non_sequon_n_retained / total_non_sequon_n if total_non_sequon_n > 0 else 0

    print(f"\n{'Position Type':<25} {'Retained':<12} {'Total':<12} {'Rate':<10}")
    print("-"*60)
    print(f"{'N at sequon sites':<25} {total_sequon_n_retained:<12} {total_sequon_n:<12} {sequon_n_rate:.2f}%")
    print(f"{'N at non-sequon sites':<25} {total_non_sequon_n_retained:<12} {total_non_sequon_n:<12} {non_sequon_n_rate:.2f}%")
    print(f"\n{'Difference:':<25} {sequon_n_rate - non_sequon_n_rate:+.2f}%")

    if sequon_n_rate > non_sequon_n_rate:
        print("\n→ INTERPRETATION: N at sequon sites is BETTER retained than N elsewhere.")
        print("   This suggests MPNN may have learned something about glycosylation contexts.")
    elif sequon_n_rate < non_sequon_n_rate:
        print("\n→ INTERPRETATION: N at sequon sites is WORSE retained than N elsewhere.")
        print("   MPNN may actually destabilize sequon contexts.")
    else:
        print("\n→ INTERPRETATION: N retention is similar regardless of sequon context.")
        print("   MPNN shows no awareness of glycosylation site importance.")

    # =================================================================
    # ANALYSIS 2: Observed vs Expected (Null Model)
    # =================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: Observed vs Expected Sequon Retention (Null Model)")
    print("="*70)

    null = calculate_null_expectation(df)

    print(f"\nIndividual retention rates at sequon positions:")
    print(f"  P(N retained)   = {100*null['p_n']:.2f}%")
    print(f"  P(X retained)   = {100*null['p_x']:.2f}%")
    print(f"  P(S/T retained) = {100*null['p_st']:.2f}%")

    print(f"\nNull expectation (if independent):")
    print(f"  Expected = P(N) × P(X) × P(S/T)")
    print(f"           = {100*null['p_n']:.2f}% × {100*null['p_x']:.2f}% × {100*null['p_st']:.2f}%")
    print(f"           = {100*null['p_expected']:.3f}%")

    print(f"\nObserved full sequon retention: {100*null['p_observed']:.3f}%")
    print(f"\nRatio (Observed / Expected): {null['ratio']:.2f}")

    if null['ratio'] > 1.1:
        print("\n→ INTERPRETATION: Observed retention is HIGHER than null expectation.")
        print("   The three positions are positively correlated - MPNN tends to")
        print("   preserve or disrupt them together, suggesting some structural coupling.")
    elif null['ratio'] < 0.9:
        print("\n→ INTERPRETATION: Observed retention is LOWER than null expectation.")
        print("   MPNN specifically disrupts sequons more than random expectation.")
    else:
        print("\n→ INTERPRETATION: Observed ≈ Expected (within 10%).")
        print("   Sequon positions are retained independently - no special treatment.")

    # =================================================================
    # Per-protein breakdown
    # =================================================================
    print("\n" + "="*70)
    print("PER-PROTEIN BREAKDOWN (Glycosylated only)")
    print("="*70)

    print(f"\n{'PDB':<8} {'Sequons':<8} {'Seq-N%':<10} {'Non-Seq-N%':<12} {'Diff':<10} {'Full Seq%':<10}")
    print("-"*70)

    for _, row in df.iterrows():
        seq_n = row['sequon_n_pct'] if not pd.isna(row['sequon_n_pct']) else 0
        non_seq_n = row['non_sequon_n_pct'] if not pd.isna(row['non_sequon_n_pct']) else 0
        diff = seq_n - non_seq_n if not pd.isna(row['non_sequon_n_pct']) else np.nan
        full_seq = row['sequon_pct'] if not pd.isna(row['sequon_pct']) else 0

        diff_str = f"{diff:+.1f}%" if not pd.isna(diff) else "N/A"
        non_seq_str = f"{non_seq_n:.1f}%" if not pd.isna(row['non_sequon_n_pct']) else "N/A"

        print(f"{row['pdb_id']:<8} {row['n_sequons']:<8} {seq_n:.1f}%{'':<5} {non_seq_str:<12} {diff_str:<10} {full_seq:.1f}%")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n1. KEY QUESTION: Does MPNN recognize glycosylation sites?")
    print(f"   - N retention at sequon sites: {sequon_n_rate:.1f}%")
    print(f"   - N retention at non-sequon sites: {non_sequon_n_rate:.1f}%")
    print(f"   - Difference: {sequon_n_rate - non_sequon_n_rate:+.1f}%")

    print("\n2. NULL MODEL TEST:")
    print(f"   - Expected sequon retention (if independent): {100*null['p_expected']:.2f}%")
    print(f"   - Observed sequon retention: {100*null['p_observed']:.2f}%")
    print(f"   - Ratio: {null['ratio']:.2f}x")

    return df, null


if __name__ == '__main__':
    df, null = main()
