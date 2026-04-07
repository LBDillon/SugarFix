#!/usr/bin/env python3
"""
STEP 1: Structure Preparation

Downloads PDB structure and biological assembly, extracts chain information,
and creates summary visualizations.

Usage:
    python 01_prepare_structure.py --pdb_id 1EO8 --output_dir ./results/1EO8

Outputs:
    - structure/{pdb_id}.pdb
    - structure/{pdb_id}_bioassembly.pdb (if different)
    - structure/structure_info.json
    - structure/chain_summary.csv
    - structure/figures/chain_lengths.png
    - structure/figures/structure_overview.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
except ImportError:
    print("Installing BioPython...")
    os.system("pip install biopython")
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1


def download_pdb(pdb_id, output_path, biological_assembly=False):
    """Download PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    if biological_assembly:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb1"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def parse_structure(pdb_path):
    """Parse PDB and extract chain information."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    chains_info = []
    model_count = 0

    for model in structure:
        model_count += 1
        for chain in model:
            chain_id = chain.id
            residues = [r for r in chain.get_residues() if is_aa(r)]

            if len(residues) > 0:
                sequence = ''.join([seq1(r.get_resname()) for r in residues])
                first_res = residues[0].id[1]
                last_res = residues[-1].id[1]

                # Calculate average B-factor
                b_factors = []
                for res in residues:
                    for atom in res:
                        b_factors.append(atom.get_bfactor())
                avg_bfactor = sum(b_factors) / len(b_factors) if b_factors else 0

                chains_info.append({
                    'model': model.id + 1,
                    'chain_id': chain_id,
                    'length': len(residues),
                    'first_residue': first_res,
                    'last_residue': last_res,
                    'avg_bfactor': avg_bfactor,
                    'sequence': sequence
                })

    return chains_info, model_count

def create_protein_only_pdb(input_pdb_path, output_pdb_path, protein_chains):
    """Create a PDB file containing only protein chains."""
    protein_chain_ids = {chain['chain_id'] for chain in protein_chains}
    
    with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21:22]
                if chain_id in protein_chain_ids:
                    outfile.write(line)
            else:
                # Keep header and other non-atom lines
                outfile.write(line)


def create_visualizations(chains_df, output_dir, pdb_id):
    """Create summary visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Chain lengths bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique chains (from model 1 only for clarity)
    model1_chains = chains_df[chains_df['model'] == 1].copy()

    colors = sns.color_palette("husl", len(model1_chains))
    bars = ax.bar(model1_chains['chain_id'], model1_chains['length'], color=colors)

    ax.set_xlabel('Chain ID', fontsize=12)
    ax.set_ylabel('Number of Residues', fontsize=12)
    ax.set_title(f'{pdb_id}: Chain Lengths', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, length in zip(bars, model1_chains['length']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(length), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_dir / "chain_lengths.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Structure overview (if multiple models/chains)
    if len(chains_df) > 4:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Chain counts per model
        model_counts = chains_df.groupby('model').size()
        axes[0].bar(model_counts.index, model_counts.values, color='steelblue')
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('Number of Chains', fontsize=12)
        axes[0].set_title('Chains per Model', fontsize=12)

        # Right: B-factor distribution
        sns.boxplot(data=chains_df, x='chain_id', y='avg_bfactor', ax=axes[1])
        axes[1].set_xlabel('Chain ID', fontsize=12)
        axes[1].set_ylabel('Average B-factor', fontsize=12)
        axes[1].set_title('B-factor by Chain', fontsize=12)

        plt.suptitle(f'{pdb_id}: Structure Overview', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / "structure_overview.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: Summary table as image
    fig, ax = plt.subplots(figsize=(12, max(3, len(model1_chains) * 0.5 + 1)))
    ax.axis('off')

    table_data = model1_chains[['chain_id', 'length', 'first_residue', 'last_residue', 'avg_bfactor']].copy()
    table_data['avg_bfactor'] = table_data['avg_bfactor'].round(1)
    table_data.columns = ['Chain', 'Length', 'First Res', 'Last Res', 'Avg B-factor']

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(table_data.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'{pdb_id}: Chain Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(fig_dir / "chain_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 1: Prepare protein structure')
    parser.add_argument('--pdb_id', required=True, help='PDB ID (e.g., 1EO8)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    output_dir = Path(args.output_dir)
    structure_dir = output_dir / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STEP 1: STRUCTURE PREPARATION - {pdb_id}")
    print("=" * 70)

    # Download asymmetric unit
    print(f"\n1. Downloading asymmetric unit...")
    pdb_path = structure_dir / f"{pdb_id}.pdb"
    if pdb_path.exists():
        print(f"  Already exists: {pdb_path}")
    else:
        if download_pdb(pdb_id, pdb_path, biological_assembly=False):
            print(f"  Downloaded: {pdb_path}")
        else:
            print("  FAILED to download asymmetric unit")
            sys.exit(1)

    # Download biological assembly
    print(f"\n2. Downloading biological assembly...")
    bioassembly_path = structure_dir / f"{pdb_id}_bioassembly.pdb"
    if download_pdb(pdb_id, bioassembly_path, biological_assembly=True):
        print(f"  Downloaded: {bioassembly_path}")
    else:
        print("  No biological assembly available (using asymmetric unit)")
        bioassembly_path = None

    # Parse asymmetric unit
    print(f"\n3. Parsing asymmetric unit...")
    asu_chains, asu_models = parse_structure(pdb_path)
    print(f"  Found {len(asu_chains)} chain(s) in {asu_models} model(s)")

    # Create protein-only PDB for ProteinMPNN
    print(f"\n4. Creating protein-only PDB...")
    protein_only_path = structure_dir / f"{pdb_id}_protein.pdb"
    create_protein_only_pdb(pdb_path, protein_only_path, asu_chains)
    print(f"  Created: {protein_only_path}")

    # Parse biological assembly if different
    bioassembly_chains = None
    if bioassembly_path and bioassembly_path.exists():
        print(f"\n5. Parsing biological assembly...")
        bioassembly_chains, bio_models = parse_structure(bioassembly_path)
        print(f"  Found {len(bioassembly_chains)} chain(s) in {bio_models} model(s)")

        if bio_models > 1:
            print(f"  ⚠️  Multiple models detected - biological assembly is an oligomer")
            print(f"     Consider using 07_oligomer_comparison.py for analysis")

    # Create DataFrames
    asu_df = pd.DataFrame(asu_chains)

    # Save CSV
    csv_path = structure_dir / "chain_summary.csv"
    asu_df.to_csv(csv_path, index=False)
    print(f"\n5. Saved chain summary: {csv_path}")

    # Save JSON with full info
    info = {
        'pdb_id': pdb_id,
        'asymmetric_unit': {
            'path': str(pdb_path),
            'protein_only_path': str(protein_only_path),
            'n_models': asu_models,
            'n_chains': len(asu_chains),
            'chains': asu_chains
        }
    }
    if bioassembly_chains:
        info['biological_assembly'] = {
            'path': str(bioassembly_path),
            'n_models': bio_models,
            'n_chains': len(bioassembly_chains),
            'is_oligomer': bio_models > 1
        }

    json_path = structure_dir / "structure_info.json"
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved structure info: {json_path}")

    # Create visualizations
    print(f"\n6. Creating visualizations...")
    create_visualizations(asu_df, structure_dir, pdb_id)

    # Summary
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in {structure_dir}/:")
    print(f"  - {pdb_id}.pdb (asymmetric unit)")
    if bioassembly_path:
        print(f"  - {pdb_id}_bioassembly.pdb (biological assembly)")
    print(f"  - chain_summary.csv")
    print(f"  - structure_info.json")
    print(f"  - figures/chain_lengths.png")
    print(f"  - figures/chain_summary_table.png")

    print(f"\n→ Next: python 02_identify_sequons.py --pdb_dir {output_dir}")


if __name__ == "__main__":
    main()
