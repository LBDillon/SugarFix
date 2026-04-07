#!/usr/bin/env python3
"""
STEP 5: De Novo Sequon Analysis

Identifies newly created sequons (not in wild-type) and finds hotspot positions.

Usage:
    python 05_analyze_denovo.py --pdb_dir ./results/1EO8

Outputs:
    - analysis/denovo/denovo_positions.csv
    - analysis/denovo/denovo_summary.csv
    - analysis/denovo/hotspots.csv
    - analysis/denovo/figures/denovo_heatmap.png
    - analysis/denovo/figures/hotspot_barplot.png
    - analysis/denovo/figures/denovo_distribution.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SEQUON_PATTERN = r'N[^P][ST]'


def read_fasta(path):
    """Read FASTA file."""
    sequences = []
    current_header = None
    current_seq = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            sequences.append((current_header, ''.join(current_seq)))
    return sequences


def split_chains(concat_seq, chain_order):
    """Split concatenated sequence by /."""
    parts = concat_seq.split('/')
    chains = {}
    for i, chain_id in enumerate(chain_order):
        if i < len(parts):
            chains[chain_id] = parts[i]
    return chains


def analyze_denovo(fa_path, sequons_by_chain, chain_order):
    """Analyze de novo sequon creation for a condition."""
    if not fa_path.exists():
        return None

    sequences = read_fasta(fa_path)
    designs = sequences[1:]  # Skip WT

    # Get original positions
    original_positions = {}
    for chain_id, sequon_list in sequons_by_chain.items():
        original_positions[chain_id] = [s['position'] for s in sequon_list]

    results = {
        'n_designs': len(designs),
        'denovo_counts': {},  # (chain, pos) -> count
        'denovo_per_design': [],  # count per design
        'denovo_details': []  # list of all de novo sequons found
    }

    for design_idx, (header, seq) in enumerate(designs):
        des_chains = split_chains(seq, chain_order)
        design_denovo = 0

        for chain_id in chain_order:
            if chain_id not in des_chains:
                continue
            des_seq = des_chains[chain_id]
            orig_pos = original_positions.get(chain_id, [])

            for match in re.finditer(SEQUON_PATTERN, des_seq):
                pos = match.start()
                if pos not in orig_pos:
                    key = (chain_id, pos)
                    results['denovo_counts'][key] = results['denovo_counts'].get(key, 0) + 1
                    design_denovo += 1

                    results['denovo_details'].append({
                        'design': design_idx + 1,
                        'chain_id': chain_id,
                        'position': pos,
                        'sequon': match.group()
                    })

        results['denovo_per_design'].append(design_denovo)

    return results


def create_visualizations(denovo_df, hotspots_df, summary_df, output_dir, pdb_id):
    """Create de novo analysis visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: De novo distribution per design
    if len(summary_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        conditions = summary_df['condition'].unique()
        x = np.arange(len(conditions))

        means = summary_df['avg_denovo']
        mins = summary_df['min_denovo']
        maxs = summary_df['max_denovo']

        colors = sns.color_palette("Set2", len(conditions))
        bars = ax.bar(x, means, color=colors, yerr=[means - mins, maxs - means], capsize=5)

        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('De Novo Sequons per Design', fontsize=12)
        ax.set_title(f'{pdb_id}: De Novo Sequon Generation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "denovo_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 2: Hotspot bar plot (top positions)
    if len(hotspots_df) > 0:
        top_hotspots = hotspots_df.head(15)  # Top 15

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#E74C3C' if p >= 75 else '#F39C12' if p >= 50 else '#3498DB'
                 for p in top_hotspots['max_pct']]

        bars = ax.barh(range(len(top_hotspots)), top_hotspots['max_pct'], color=colors)

        ax.set_yticks(range(len(top_hotspots)))
        ax.set_yticklabels(top_hotspots['position_label'])
        ax.set_xlabel('Max Occurrence Across Conditions (%)', fontsize=12)
        ax.set_ylabel('Position', fontsize=12)
        ax.set_title(f'{pdb_id}: Top De Novo Sequon Hotspots', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)

        # Add percentage labels
        for bar, pct in zip(bars, top_hotspots['max_pct']):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.0f}%', ha='left', va='center', fontsize=9)

        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "hotspot_barplot.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: Heatmap of de novo positions by condition
    if len(denovo_df) > 0:
        # Get top 20 positions by max occurrence
        position_max = denovo_df.groupby('position_label')['occurrence_pct'].max()
        top_positions = position_max.nlargest(20).index.tolist()

        subset = denovo_df[denovo_df['position_label'].isin(top_positions)]

        if len(subset) > 0:
            heatmap_data = subset.pivot_table(
                index='position_label',
                columns='condition',
                values='occurrence_pct',
                aggfunc='first'
            ).fillna(0)

            # Sort by max value
            heatmap_data['max'] = heatmap_data.max(axis=1)
            heatmap_data = heatmap_data.sort_values('max', ascending=False).drop('max', axis=1)

            fig, ax = plt.subplots(figsize=(10, max(5, len(heatmap_data) * 0.4)))

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                vmin=0,
                vmax=100,
                ax=ax,
                cbar_kws={'label': 'Occurrence %'}
            )

            ax.set_xlabel('Condition', fontsize=12)
            ax.set_ylabel('Position', fontsize=12)
            ax.set_title(f'{pdb_id}: De Novo Sequon Hotspots by Condition', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig(fig_dir / "denovo_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 5: Analyze de novo sequons')
    parser.add_argument('--pdb_dir', required=True, help='Directory from previous steps')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    designs_dir = pdb_dir / "designs"
    analysis_dir = pdb_dir / "analysis" / "denovo"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load info
    with open(structure_dir / "structure_info.json") as f:
        structure_info = json.load(f)

    with open(sequons_dir / "sequons_by_chain.json") as f:
        sequons_by_chain = json.load(f)

    pdb_id = structure_info['pdb_id']
    chain_order = [c['chain_id'] for c in structure_info['asymmetric_unit']['chains']
                   if c['model'] == 1]

    print("=" * 70)
    print(f"STEP 5: DE NOVO SEQUON ANALYSIS - {pdb_id}")
    print("=" * 70)

    conditions = [
        ('unconstrained', 'Unconstrained'),
        ('n_only_fixed', 'N Only Fixed'),
        ('full_sequon_fixed', 'Full Sequon Fixed')
    ]

    all_denovo = []
    summary_data = []
    all_positions = {}  # Track all positions across conditions

    for condition_dir, condition_name in conditions:
        # Look for any .fa file in the seqs directory
        seqs_dir = designs_dir / condition_dir / "seqs"
        fa_path = None
        if seqs_dir.exists():
            fa_files = list(seqs_dir.glob("*.fa"))
            if fa_files:
                fa_path = fa_files[0]

        print(f"\n{condition_name}:")

        results = analyze_denovo(fa_path, sequons_by_chain, chain_order) if fa_path else None

        if results is None:
            print(f"  No results found")
            continue

        n_designs = results['n_designs']
        denovo_counts = results['denovo_counts']
        denovo_per_design = results['denovo_per_design']

        avg_denovo = np.mean(denovo_per_design)
        min_denovo = min(denovo_per_design)
        max_denovo = max(denovo_per_design)

        print(f"  Designs analyzed: {n_designs}")
        print(f"  De novo per design: {avg_denovo:.1f} avg ({min_denovo}-{max_denovo})")
        print(f"  Unique positions: {len(denovo_counts)}")

        summary_data.append({
            'condition': condition_name,
            'n_designs': n_designs,
            'total_denovo': sum(denovo_per_design),
            'avg_denovo': avg_denovo,
            'min_denovo': min_denovo,
            'max_denovo': max_denovo,
            'unique_positions': len(denovo_counts)
        })

        # Record position-level data
        for (chain_id, pos), count in denovo_counts.items():
            occurrence_pct = count / n_designs * 100
            position_label = f"{chain_id}:{pos}"

            all_denovo.append({
                'condition': condition_name,
                'chain_id': chain_id,
                'position': pos,
                'position_label': position_label,
                'count': count,
                'n_designs': n_designs,
                'occurrence_pct': occurrence_pct
            })

            if position_label not in all_positions:
                all_positions[position_label] = {'chain': chain_id, 'pos': pos, 'pcts': []}
            all_positions[position_label]['pcts'].append(occurrence_pct)

        # Top 5 hotspots for this condition
        print("  Top 5 hotspots:")
        sorted_counts = sorted(denovo_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for (chain_id, pos), count in sorted_counts:
            pct = count / n_designs * 100
            print(f"    {chain_id}:{pos}: {pct:.1f}%")

    # Create DataFrames
    denovo_df = pd.DataFrame(all_denovo)
    summary_df = pd.DataFrame(summary_data)

    # Create hotspots DataFrame (max across conditions)
    hotspots_data = []
    for position_label, data in all_positions.items():
        hotspots_data.append({
            'position_label': position_label,
            'chain_id': data['chain'],
            'position': data['pos'],
            'max_pct': max(data['pcts']),
            'avg_pct': np.mean(data['pcts']),
            'n_conditions': len(data['pcts'])
        })

    hotspots_df = pd.DataFrame(hotspots_data)
    hotspots_df = hotspots_df.sort_values('max_pct', ascending=False)

    # Save CSVs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    denovo_df.to_csv(analysis_dir / "denovo_positions.csv", index=False)
    summary_df.to_csv(analysis_dir / "denovo_summary.csv", index=False)
    hotspots_df.to_csv(analysis_dir / "hotspots.csv", index=False)
    print(f"  Saved: {analysis_dir}/denovo_positions.csv")
    print(f"  Saved: {analysis_dir}/denovo_summary.csv")
    print(f"  Saved: {analysis_dir}/hotspots.csv")

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(denovo_df, hotspots_df, summary_df, analysis_dir, pdb_id)

    # Summary
    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    print("\nDe Novo Summary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['condition']}: {row['avg_denovo']:.1f} avg per design")

    print(f"\nTop 5 Hotspots (across all conditions):")
    for _, row in hotspots_df.head(5).iterrows():
        print(f"  {row['position_label']}: {row['max_pct']:.0f}%")

    print(f"\nOutputs in {analysis_dir}/:")
    print(f"  - denovo_positions.csv")
    print(f"  - denovo_summary.csv")
    print(f"  - hotspots.csv")
    print(f"  - figures/denovo_distribution.png")
    print(f"  - figures/hotspot_barplot.png")
    print(f"  - figures/denovo_heatmap.png")

    print(f"\n→ Next: python 06_structural_context.py --pdb_dir {pdb_dir}")


if __name__ == "__main__":
    main()
