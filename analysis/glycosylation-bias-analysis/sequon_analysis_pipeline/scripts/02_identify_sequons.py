#!/usr/bin/env python3
"""
STEP 2: Sequon Identification

Identifies all N-X-S/T sequons in the wild-type protein sequence using
ProteinMPNN's own PDB parsing to ensure correct indexing.

Usage:
    python 02_identify_sequons.py --pdb_dir ./results/1EO8

Outputs:
    - sequons/sequons.csv
    - sequons/sequons_by_chain.json
    - sequons/figures/sequon_map.png
    - sequons/figures/sequon_positions.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

# Import shared utilities for MPNN-consistent parsing
from mpnn_utils import (
    get_mpnn_chain_seqs_and_order,
    find_sequons,
    verify_sequon_positions,
    SEQUON_REGEX
)


def create_sequon_map(chains_data, sequons_df, output_dir, pdb_id):
    """Create visual map of sequons on protein sequence."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Get unique chains
    chains_with_sequons = sequons_df['chain_id'].unique()
    all_chains = list(chains_data.keys())

    n_chains = len(all_chains)
    fig_height = max(4, n_chains * 1.2)

    fig, ax = plt.subplots(figsize=(14, fig_height))

    colors = sns.color_palette("Set2", n_chains)
    sequon_color = '#E74C3C'  # Red for sequons

    y_positions = {}
    for i, chain_id in enumerate(sorted(all_chains)):
        y_positions[chain_id] = n_chains - i - 1

    # Draw chains as horizontal bars
    for chain_id, chain_info in chains_data.items():
        y = y_positions[chain_id]
        length = chain_info['length']
        color = colors[list(chains_data.keys()).index(chain_id) % len(colors)]

        # Draw chain backbone
        ax.barh(y, length, height=0.6, color=color, alpha=0.3, edgecolor=color)

        # Add chain label
        ax.text(-10, y, f"Chain {chain_id}", ha='right', va='center', fontsize=11, fontweight='bold')
        ax.text(length + 5, y, f"({length} aa)", ha='left', va='center', fontsize=9, color='gray')

    # Mark sequon positions
    for _, row in sequons_df.iterrows():
        chain_id = row['chain_id']
        pos = row['position_0idx']
        y = y_positions[chain_id]

        # Draw sequon marker
        ax.barh(y, 3, left=pos, height=0.6, color=sequon_color, alpha=0.8)

        # Add position label (only if not too crowded)
        if len(sequons_df[sequons_df['chain_id'] == chain_id]) < 10:
            ax.text(pos + 1.5, y + 0.4, f"{row['sequon']}\n({pos})",
                   ha='center', va='bottom', fontsize=7, color=sequon_color)

    # Customize plot
    ax.set_xlim(-50, max(c['length'] for c in chains_data.values()) + 50)
    ax.set_ylim(-0.5, n_chains - 0.5)
    ax.set_xlabel('Residue Position (0-indexed)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'{pdb_id}: N-X-S/T Sequon Map', fontsize=14, fontweight='bold')

    # Add legend
    chain_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='Protein chain')
    sequon_patch = mpatches.Patch(color=sequon_color, alpha=0.8, label='N-X-S/T sequon')
    ax.legend(handles=[chain_patch, sequon_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(fig_dir / "sequon_map.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Sequon positions bar plot
    if len(sequons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        sequon_counts = sequons_df.groupby('chain_id').size().reindex(all_chains, fill_value=0)

        colors = ['#E74C3C' if c > 0 else '#BDC3C7' for c in sequon_counts]
        bars = ax.bar(sequon_counts.index, sequon_counts.values, color=colors)

        ax.set_xlabel('Chain ID', fontsize=12)
        ax.set_ylabel('Number of Sequons', fontsize=12)
        ax.set_title(f'{pdb_id}: Sequons per Chain', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, count in zip(bars, sequon_counts.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "sequon_counts.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: Sequon details table
    if len(sequons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(3, len(sequons_df) * 0.4 + 1)))
        ax.axis('off')

        table_data = sequons_df[['chain_id', 'position_0idx', 'sequon']].copy()
        table_data.columns = ['Chain', 'Position (0-idx)', 'Sequon']

        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colColours=['#E74C3C'] * len(table_data.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(table_data.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'{pdb_id}: Sequon Details', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(fig_dir / "sequon_table.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 2: Identify sequons')
    parser.add_argument('--pdb_dir', required=True, help='Directory from Step 1')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    sequons_dir.mkdir(exist_ok=True)

    # Load structure info
    info_path = structure_dir / "structure_info.json"
    if not info_path.exists():
        print(f"ERROR: Run Step 1 first. Missing: {info_path}")
        return

    with open(info_path) as f:
        structure_info = json.load(f)

    pdb_id = structure_info['pdb_id']

    # Get protein-only PDB path
    asymmetric_unit = structure_info['asymmetric_unit']
    pdb_path = Path(asymmetric_unit.get('protein_only_path', asymmetric_unit.get('path')))

    print("=" * 70)
    print(f"STEP 2: SEQUON IDENTIFICATION - {pdb_id}")
    print("=" * 70)

    # CRITICAL: Use ProteinMPNN's parsing for correct indexing
    print(f"\n1. Parsing structure with ProteinMPNN's parser...")
    print(f"   (This ensures sequon positions match ProteinMPNN's internal indexing)")

    chain_seqs, chain_order = get_mpnn_chain_seqs_and_order(pdb_path)

    if not chain_seqs:
        print(f"ERROR: No chains found in {pdb_path}")
        return

    # Build chains_data for visualization compatibility
    chains_data = {}
    for chain_id in chain_order:
        chains_data[chain_id] = {
            'sequence': chain_seqs[chain_id],
            'length': len(chain_seqs[chain_id])
        }

    print(f"   Found {len(chains_data)} chain(s): {', '.join(chain_order)}")

    print(f"\n2. Scanning for N-X-S/T sequons...")

    # Find sequons in each chain
    all_sequons = []
    sequons_by_chain = {}

    for chain_id in chain_order:
        sequence = chain_seqs[chain_id]
        sequons = find_sequons(sequence)

        sequons_by_chain[chain_id] = [
            {'position_0idx': s['position_0idx'], 'sequon': s['sequon']}
            for s in sequons
        ]

        for s in sequons:
            s['chain_id'] = chain_id
            s['position_1idx'] = s['position_0idx'] + 1
            s['n_residue'] = s['sequon'][0]
            s['x_residue'] = s['sequon'][1]
            s['st_residue'] = s['sequon'][2]
            all_sequons.append(s)

        print(f"  Chain {chain_id}: {len(sequons)} sequon(s)")
        for s in sequons:
            print(f"    Position {s['position_0idx']} (1-idx: {s['position_0idx']+1}): {s['sequon']}")

    # CRITICAL: Verify positions are correct
    print(f"\n3. Verifying sequon positions...")
    try:
        verify_sequon_positions(chain_seqs, sequons_by_chain, pdb_id)
        print("   All sequon positions verified successfully")
    except AssertionError as e:
        print(f"   ERROR: Position verification failed: {e}")
        return

    # Create DataFrame
    sequons_df = pd.DataFrame(all_sequons)

    # Save outputs
    print(f"\n4. Saving outputs...")

    if len(sequons_df) > 0:
        csv_path = sequons_dir / "sequons.csv"
        sequons_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    json_path = sequons_dir / "sequons_by_chain.json"
    with open(json_path, 'w') as f:
        json.dump(sequons_by_chain, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save chain order for downstream scripts
    chain_order_path = sequons_dir / "mpnn_chain_order.json"
    with open(chain_order_path, 'w') as f:
        json.dump({"chain_order": chain_order}, f, indent=2)
    print(f"  Saved: {chain_order_path}")

    # Create visualizations
    print(f"\n5. Creating visualizations...")
    if len(sequons_df) > 0:
        create_sequon_map(chains_data, sequons_df, sequons_dir, pdb_id)
    else:
        print("  No sequons found - skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"\nTotal sequons found: {len(sequons_df)}")

    chains_with_sequons = sequons_df['chain_id'].nunique() if len(sequons_df) > 0 else 0
    print(f"Chains with sequons: {chains_with_sequons}/{len(chains_data)}")

    print(f"\nOutputs in {sequons_dir}/:")
    print(f"  - sequons.csv")
    print(f"  - sequons_by_chain.json")
    print(f"  - figures/sequon_map.png")
    print(f"  - figures/sequon_counts.png")
    print(f"  - figures/sequon_table.png")

    print(f"\n→ Next: python 03_run_proteinmpnn.py --pdb_dir {pdb_dir}")


if __name__ == "__main__":
    main()
