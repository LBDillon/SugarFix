#!/usr/bin/env python3
"""
STEP 4: Sequon Retention Analysis

Analyzes how well original sequons are retained across designs and conditions.
Includes comparative analysis of sequon-N vs non-sequon-N retention.

Usage:
    python 04_analyze_retention.py --pdb_dir ./results/1EO8

Outputs:
    - analysis/retention/retention_by_position.csv
    - analysis/retention/retention_summary.csv
    - analysis/retention/n_position_analysis.csv  (NEW: all N positions)
    - analysis/retention/figures/retention_heatmap.png
    - analysis/retention/figures/retention_barplot.png
    - analysis/retention/figures/retention_comparison.png
    - analysis/retention/figures/sequon_n_vs_nonsequon_n.png  (NEW)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import shared utilities
from mpnn_utils import (
    read_fasta_sequences as read_fasta,
    split_mpnn_concat_seq as split_chains,
    is_functional_sequon as is_valid_sequon,
    get_mpnn_chain_seqs_and_order,
    SEQUON_REGEX
)


def analyze_retention(fa_path, sequons_by_chain, chain_order):
    """Analyze sequon retention for a condition."""
    if not fa_path.exists():
        return None

    sequences = read_fasta(fa_path)
    wt_header, wt_seq = sequences[0]
    wt_chains = split_chains(wt_seq, chain_order)
    designs = sequences[1:]

    results = {
        'n_designs': len(designs),
        'retention_by_position': {},  # (chain, pos) -> count
        'n_fixed_by_position': {},    # (chain, pos) -> count (N still present)
    }

    for chain_id, sequon_list in sequons_by_chain.items():
        for sequon in sequon_list:
            # Support both 'position_0idx' (new) and 'position' (legacy)
            pos = sequon.get('position_0idx', sequon.get('position'))
            key = (chain_id, pos)
            results['retention_by_position'][key] = 0
            results['n_fixed_by_position'][key] = 0

    for header, seq in designs:
        des_chains = split_chains(seq, chain_order)

        for chain_id, sequon_list in sequons_by_chain.items():
            if chain_id not in des_chains:
                continue
            des_seq = des_chains[chain_id]

            for sequon in sequon_list:
                pos = sequon.get('position_0idx', sequon.get('position'))
                key = (chain_id, pos)

                if pos + 3 <= len(des_seq):
                    triplet = des_seq[pos:pos+3]
                    if is_valid_sequon(triplet):
                        results['retention_by_position'][key] += 1
                    if triplet[0] == 'N':
                        results['n_fixed_by_position'][key] += 1

    return results


def analyze_all_n_positions(fa_path, sequons_by_chain, chain_order):
    """
    Analyze retention at EVERY N position, comparing sequon-N vs non-sequon-N.

    This is a key analysis for understanding whether ProteinMPNN treats
    asparagine residues differently based on their sequon context.
    """
    if not fa_path.exists():
        return None

    sequences = read_fasta(fa_path)
    wt_header, wt_seq = sequences[0]
    wt_chains = split_chains(wt_seq, chain_order)
    designs = sequences[1:]

    # Identify all N positions and which are in sequons
    sequon_n_positions = set()
    for chain_id, sequon_list in sequons_by_chain.items():
        for sequon in sequon_list:
            pos = sequon.get('position_0idx', sequon.get('position'))
            sequon_n_positions.add((chain_id, pos))

    all_n_positions = set()
    for chain_id, seq in wt_chains.items():
        for i, aa in enumerate(seq):
            if aa == 'N':
                all_n_positions.add((chain_id, i))

    non_sequon_n_positions = all_n_positions - sequon_n_positions

    # Count retention for each category
    results = {
        'n_designs': len(designs),
        'sequon_n_total': 0,
        'sequon_n_retained': 0,
        'non_sequon_n_total': 0,
        'non_sequon_n_retained': 0,
        'per_position': []  # List of dicts for each N position
    }

    # Initialize per-position tracking
    position_counts = {}
    for chain_id, pos in all_n_positions:
        is_sequon = (chain_id, pos) in sequon_n_positions
        wt_triplet = wt_chains[chain_id][pos:pos+3] if pos + 3 <= len(wt_chains[chain_id]) else wt_chains[chain_id][pos:]
        position_counts[(chain_id, pos)] = {
            'is_sequon': is_sequon,
            'wt_triplet': wt_triplet,
            'n_retained': 0,
            'n_total': 0,
            'exact_sequon_retained': 0,
            'functional_sequon_retained': 0
        }

    # Analyze each design
    for header, seq in designs:
        des_chains = split_chains(seq, chain_order)

        for (chain_id, pos), data in position_counts.items():
            if chain_id not in des_chains:
                continue

            des_seq = des_chains[chain_id]
            wt_seq = wt_chains[chain_id]

            if pos >= len(des_seq) or pos >= len(wt_seq):
                continue

            data['n_total'] += 1

            # Check if N is retained
            if des_seq[pos] == 'N':
                data['n_retained'] += 1

                if data['is_sequon']:
                    results['sequon_n_retained'] += 1
                else:
                    results['non_sequon_n_retained'] += 1

            if data['is_sequon']:
                results['sequon_n_total'] += 1

                # Check exact and functional retention for sequon positions
                if pos + 3 <= len(des_seq) and pos + 3 <= len(wt_seq):
                    des_triplet = des_seq[pos:pos+3]
                    wt_triplet = wt_seq[pos:pos+3]

                    if des_triplet == wt_triplet:
                        data['exact_sequon_retained'] += 1

                    if is_valid_sequon(des_triplet):
                        data['functional_sequon_retained'] += 1
            else:
                results['non_sequon_n_total'] += 1

    # Convert to per-position list
    for (chain_id, pos), data in position_counts.items():
        n_total = data['n_total'] if data['n_total'] > 0 else 1
        results['per_position'].append({
            'chain_id': chain_id,
            'position': pos,
            'is_sequon': data['is_sequon'],
            'wt_triplet': data['wt_triplet'],
            'n_retained': data['n_retained'],
            'n_total': data['n_total'],
            'n_retention_pct': 100.0 * data['n_retained'] / n_total,
            'exact_sequon_retained': data['exact_sequon_retained'] if data['is_sequon'] else None,
            'exact_sequon_pct': 100.0 * data['exact_sequon_retained'] / n_total if data['is_sequon'] else None,
            'functional_retained': data['functional_sequon_retained'] if data['is_sequon'] else None,
            'functional_pct': 100.0 * data['functional_sequon_retained'] / n_total if data['is_sequon'] else None
        })

    return results


def create_n_position_visualization(n_analysis_df, output_dir, pdb_id):
    """Create visualization comparing sequon-N vs non-sequon-N retention."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    if len(n_analysis_df) == 0:
        return

    # Figure: Sequon-N vs Non-Sequon-N retention
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plot comparison
    ax = axes[0]
    plot_data = n_analysis_df[['is_sequon', 'n_retention_pct']].copy()
    plot_data['Category'] = plot_data['is_sequon'].map({True: 'Sequon-N', False: 'Non-Sequon-N'})

    sns.boxplot(data=plot_data, x='Category', y='n_retention_pct', ax=ax,
                palette=['#E74C3C', '#3498DB'])
    ax.set_xlabel('N Position Type', fontsize=12)
    ax.set_ylabel('N Retention (%)', fontsize=12)
    ax.set_title('Asparagine Retention by Context', fontsize=12)
    ax.set_ylim(-5, 105)

    # Add mean values as text
    for i, category in enumerate(['Sequon-N', 'Non-Sequon-N']):
        subset = plot_data[plot_data['Category'] == category]['n_retention_pct']
        if len(subset) > 0:
            mean_val = subset.mean()
            ax.text(i, mean_val + 5, f'Mean: {mean_val:.1f}%', ha='center', fontsize=10)

    # Right: Scatter of all N positions
    ax = axes[1]
    sequon_data = n_analysis_df[n_analysis_df['is_sequon'] == True]
    non_sequon_data = n_analysis_df[n_analysis_df['is_sequon'] == False]

    ax.scatter(range(len(non_sequon_data)), non_sequon_data['n_retention_pct'].values,
               alpha=0.6, label=f'Non-Sequon-N (n={len(non_sequon_data)})', color='#3498DB')
    ax.scatter(range(len(non_sequon_data), len(non_sequon_data) + len(sequon_data)),
               sequon_data['n_retention_pct'].values,
               alpha=0.6, label=f'Sequon-N (n={len(sequon_data)})', color='#E74C3C')

    ax.axhline(non_sequon_data['n_retention_pct'].mean(), color='#3498DB', linestyle='--', alpha=0.7)
    ax.axhline(sequon_data['n_retention_pct'].mean(), color='#E74C3C', linestyle='--', alpha=0.7)

    ax.set_xlabel('N Position (sorted by category)', fontsize=12)
    ax.set_ylabel('N Retention (%)', fontsize=12)
    ax.set_title('Individual N Position Retention', fontsize=12)
    ax.legend()
    ax.set_ylim(-5, 105)

    plt.suptitle(f'{pdb_id}: Sequon-N vs Non-Sequon-N Retention', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / "sequon_n_vs_nonsequon_n.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {fig_dir}/sequon_n_vs_nonsequon_n.png")


def create_visualizations(retention_df, summary_df, output_dir, pdb_id):
    """Create retention analysis visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Retention heatmap
    if len(retention_df) > 0:
        # Pivot for heatmap
        heatmap_data = retention_df.pivot_table(
            index='position_label',
            columns='condition',
            values='retention_pct',
            aggfunc='first'
        )

        # Reorder columns
        col_order = ['Unconstrained', 'N Only Fixed', 'Full Sequon Fixed']
        heatmap_data = heatmap_data[[c for c in col_order if c in heatmap_data.columns]]

        fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.5)))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.0f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={'label': 'Retention %'}
        )

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Sequon Position', fontsize=12)
        ax.set_title(f'{pdb_id}: Sequon Retention by Position', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "retention_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 2: Overall retention bar plot
    if len(summary_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#E74C3C', '#F39C12', '#27AE60']
        bars = ax.bar(summary_df['condition'], summary_df['retention_pct'], color=colors)

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Overall Retention (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: Overall Sequon Retention', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)

        # Add value labels
        for bar, pct in zip(bars, summary_df['retention_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "retention_barplot.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: N-fixed vs full retention comparison
    if len(retention_df) > 0 and 'n_fixed_pct' in retention_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(retention_df['position_label'].unique()))
        width = 0.25

        positions = retention_df['position_label'].unique()

        for i, condition in enumerate(['Unconstrained', 'N Only Fixed', 'Full Sequon Fixed']):
            subset = retention_df[retention_df['condition'] == condition]
            if len(subset) > 0:
                values = [subset[subset['position_label'] == p]['retention_pct'].values[0]
                         if len(subset[subset['position_label'] == p]) > 0 else 0
                         for p in positions]
                ax.bar(x + i * width, values, width, label=condition)

        ax.set_xlabel('Sequon Position', fontsize=12)
        ax.set_ylabel('Retention (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: Retention by Position and Condition', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(positions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(fig_dir / "retention_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 4: Analyze sequon retention')
    parser.add_argument('--pdb_dir', required=True, help='Directory from previous steps')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    designs_dir = pdb_dir / "designs"
    analysis_dir = pdb_dir / "analysis" / "retention"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load info
    with open(structure_dir / "structure_info.json") as f:
        structure_info = json.load(f)

    with open(sequons_dir / "sequons_by_chain.json") as f:
        sequons_by_chain = json.load(f)

    pdb_id = structure_info['pdb_id']

    # Load MPNN chain order if available (from step 2)
    chain_order_path = sequons_dir / "mpnn_chain_order.json"
    if chain_order_path.exists():
        with open(chain_order_path) as f:
            chain_order = json.load(f)["chain_order"]
    else:
        chain_order = [c['chain_id'] for c in structure_info['asymmetric_unit']['chains']
                       if c['model'] == 1]

    print("=" * 70)
    print(f"STEP 4: RETENTION ANALYSIS - {pdb_id}")
    print("=" * 70)

    # Count total sequons
    total_sequons = sum(len(s) for s in sequons_by_chain.values())
    print(f"\nTotal original sequons: {total_sequons}")

    conditions = [
        ('unconstrained', 'Unconstrained'),
        ('n_only_fixed', 'N Only Fixed'),
        ('full_sequon_fixed', 'Full Sequon Fixed')
    ]

    all_results = []
    summary_data = []

    for condition_dir, condition_name in conditions:
        # Look for any .fa file in the seqs directory
        seqs_dir = designs_dir / condition_dir / "seqs"
        fa_path = None
        if seqs_dir.exists():
            fa_files = list(seqs_dir.glob("*.fa"))
            if fa_files:
                fa_path = fa_files[0]

        print(f"\n{condition_name}:")

        results = analyze_retention(fa_path, sequons_by_chain, chain_order) if fa_path else None

        if results is None:
            print(f"  No results found at {fa_path}")
            continue

        n_designs = results['n_designs']
        print(f"  Designs analyzed: {n_designs}")

        # Calculate retention for each position
        total_retained = 0
        total_possible = 0

        for (chain_id, pos), count in results['retention_by_position'].items():
            retention_pct = count / n_designs * 100
            n_fixed_count = results['n_fixed_by_position'][(chain_id, pos)]
            n_fixed_pct = n_fixed_count / n_designs * 100

            # Get sequon string (support both position_0idx and position keys)
            sequon_str = next(
                (s['sequon'] for s in sequons_by_chain[chain_id]
                 if s.get('position_0idx', s.get('position')) == pos),
                'N?S/T'
            )

            all_results.append({
                'condition': condition_name,
                'chain_id': chain_id,
                'position': pos,
                'position_label': f"{chain_id}:{pos} ({sequon_str})",
                'sequon': sequon_str,
                'retained_count': count,
                'n_designs': n_designs,
                'retention_pct': retention_pct,
                'n_fixed_count': n_fixed_count,
                'n_fixed_pct': n_fixed_pct
            })

            total_retained += count
            total_possible += n_designs

            print(f"    {chain_id}:{pos} ({sequon_str}): {retention_pct:.1f}% retained, N fixed: {n_fixed_pct:.1f}%")

        overall_retention = total_retained / total_possible * 100 if total_possible > 0 else 0
        print(f"  Overall retention: {overall_retention:.1f}%")

        summary_data.append({
            'condition': condition_name,
            'total_retained': total_retained,
            'total_possible': total_possible,
            'retention_pct': overall_retention
        })

    # Create DataFrames
    retention_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_data)

    # Save CSVs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    retention_df.to_csv(analysis_dir / "retention_by_position.csv", index=False)
    summary_df.to_csv(analysis_dir / "retention_summary.csv", index=False)
    print(f"  Saved: {analysis_dir}/retention_by_position.csv")
    print(f"  Saved: {analysis_dir}/retention_summary.csv")

    # NEW: Analyze all N positions (sequon-N vs non-sequon-N)
    print(f"\n{'=' * 70}")
    print("SEQUON-N vs NON-SEQUON-N ANALYSIS")
    print("=" * 70)

    # Use unconstrained designs for this analysis
    unconstrained_dir = designs_dir / "unconstrained" / "seqs"
    if unconstrained_dir.exists():
        fa_files = list(unconstrained_dir.glob("*.fa"))
        if fa_files:
            n_results = analyze_all_n_positions(fa_files[0], sequons_by_chain, chain_order)

            if n_results:
                n_analysis_df = pd.DataFrame(n_results['per_position'])

                # Calculate summary statistics
                sequon_n_pct = (100.0 * n_results['sequon_n_retained'] / n_results['sequon_n_total']
                               if n_results['sequon_n_total'] > 0 else 0)
                non_sequon_n_pct = (100.0 * n_results['non_sequon_n_retained'] / n_results['non_sequon_n_total']
                                   if n_results['non_sequon_n_total'] > 0 else 0)

                print(f"\n  Sequon-N positions: {n_results['sequon_n_total'] // n_results['n_designs']} total")
                print(f"    Retention: {sequon_n_pct:.1f}%")
                print(f"\n  Non-Sequon-N positions: {n_results['non_sequon_n_total'] // n_results['n_designs']} total")
                print(f"    Retention: {non_sequon_n_pct:.1f}%")
                print(f"\n  Difference: {sequon_n_pct - non_sequon_n_pct:+.1f}% (sequon - non-sequon)")

                # Save N-position analysis
                n_analysis_df.to_csv(analysis_dir / "n_position_analysis.csv", index=False)
                print(f"\n  Saved: {analysis_dir}/n_position_analysis.csv")

                # Create N-position visualization
                create_n_position_visualization(n_analysis_df, analysis_dir, pdb_id)

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(retention_df, summary_df, analysis_dir, pdb_id)

    # Summary
    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)

    print("\nRetention Summary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['condition']}: {row['retention_pct']:.1f}%")

    print(f"\nOutputs in {analysis_dir}/:")
    print(f"  - retention_by_position.csv")
    print(f"  - retention_summary.csv")
    print(f"  - figures/retention_heatmap.png")
    print(f"  - figures/retention_barplot.png")
    print(f"  - figures/retention_comparison.png")

    print(f"\n→ Next: python 05_analyze_denovo.py --pdb_dir {pdb_dir}")


if __name__ == "__main__":
    main()
