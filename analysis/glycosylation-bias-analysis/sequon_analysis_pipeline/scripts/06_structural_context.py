#!/usr/bin/env python3
"""
STEP 6: Structural Context Analysis

Analyzes structural features (RSA, B-factors, secondary structure) and
correlates them with sequon retention and de novo creation rates.

Usage:
    python 06_structural_context.py --pdb_dir ./results/1EO8

Outputs:
    - analysis/structural/structural_features.csv
    - analysis/structural/figures/rsa_distribution.png
    - analysis/structural/figures/retention_vs_rsa.png
    - analysis/structural/figures/denovo_vs_rsa.png
    - analysis/structural/figures/structural_summary.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from Bio.PDB import PDBParser, SASA, is_aa
except ImportError:
    print("Installing BioPython...")
    import os
    os.system("pip install biopython")
    from Bio.PDB import PDBParser, SASA, is_aa


# Standard amino acid maximum SASA values (Å²) for RSA calculation
MAX_SASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174
}


def calculate_structural_features(pdb_path):
    """Calculate RSA and B-factors for all residues."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # Calculate SASA
    sr = SASA.ShrakeRupley()
    sr.compute(structure, level="R")

    features = []

    for model in structure:
        if model.id > 0:  # Only first model
            break
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if not is_aa(residue):
                    continue

                res_name = residue.get_resname()
                res_id = residue.id[1]

                # Get SASA and calculate RSA
                sasa = residue.sasa if hasattr(residue, 'sasa') else 0

                # Get one-letter code
                from Bio.SeqUtils import seq1
                aa = seq1(res_name)

                max_sasa = MAX_SASA.get(aa, 200)
                rsa = (sasa / max_sasa * 100) if max_sasa > 0 else 0

                # Get B-factor (average of all atoms)
                b_factors = [atom.get_bfactor() for atom in residue]
                avg_bfactor = np.mean(b_factors) if b_factors else 0

                features.append({
                    'chain_id': chain_id,
                    'residue_id': res_id,
                    'residue_name': aa,
                    'sasa': sasa,
                    'rsa': rsa,
                    'bfactor': avg_bfactor
                })

    return pd.DataFrame(features)


def create_visualizations(features_df, sequon_features_df, denovo_features_df,
                         output_dir, pdb_id):
    """Create structural analysis visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: RSA distribution for all residues vs sequon positions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overall RSA distribution
    axes[0].hist(features_df['rsa'], bins=30, color='steelblue', alpha=0.7,
                edgecolor='black', label='All residues')
    if len(sequon_features_df) > 0:
        axes[0].axvline(sequon_features_df['rsa'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Sequon avg: {sequon_features_df['rsa'].mean():.1f}%")
    axes[0].set_xlabel('Relative Solvent Accessibility (%)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('RSA Distribution', fontsize=12)
    axes[0].legend()

    # Right: B-factor distribution
    axes[1].hist(features_df['bfactor'], bins=30, color='steelblue', alpha=0.7,
                edgecolor='black', label='All residues')
    if len(sequon_features_df) > 0:
        axes[1].axvline(sequon_features_df['bfactor'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Sequon avg: {sequon_features_df['bfactor'].mean():.1f}")
    axes[1].set_xlabel('B-factor', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('B-factor Distribution', fontsize=12)
    axes[1].legend()

    plt.suptitle(f'{pdb_id}: Structural Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / "structural_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Retention vs RSA (if retention data available)
    if len(sequon_features_df) > 0 and 'retention_pct' in sequon_features_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            sequon_features_df['rsa'],
            sequon_features_df['retention_pct'],
            c=sequon_features_df['bfactor'],
            cmap='coolwarm',
            s=100,
            alpha=0.7,
            edgecolor='black'
        )

        # Add labels
        for _, row in sequon_features_df.iterrows():
            ax.annotate(
                row['position_label'],
                (row['rsa'], row['retention_pct']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        plt.colorbar(scatter, label='B-factor')
        ax.set_xlabel('Relative Solvent Accessibility (%)', fontsize=12)
        ax.set_ylabel('Retention Rate (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: Sequon Retention vs Structural Features', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "retention_vs_rsa.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: De novo rate vs RSA (if denovo data available)
    if len(denovo_features_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter to top hotspots
        top_denovo = denovo_features_df.nlargest(20, 'occurrence_pct')

        scatter = ax.scatter(
            top_denovo['rsa'],
            top_denovo['occurrence_pct'],
            c=top_denovo['bfactor'],
            cmap='coolwarm',
            s=100,
            alpha=0.7,
            edgecolor='black'
        )

        plt.colorbar(scatter, label='B-factor')
        ax.set_xlabel('Relative Solvent Accessibility (%)', fontsize=12)
        ax.set_ylabel('De Novo Occurrence Rate (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: De Novo Sequons vs Structural Features', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "denovo_vs_rsa.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 4: Summary plot - original sequons with structural context
    if len(sequon_features_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(sequon_features_df))
        width = 0.35

        ax.bar([i - width/2 for i in x], sequon_features_df['rsa'], width,
              label='RSA (%)', color='steelblue', alpha=0.7)
        ax.bar([i + width/2 for i in x], sequon_features_df['bfactor'], width,
              label='B-factor', color='coral', alpha=0.7)

        ax.set_xlabel('Sequon Position', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{pdb_id}: Structural Context of Original Sequons', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequon_features_df['position_label'], rotation=45, ha='right')
        ax.legend()

        # Add retention rate as text if available
        if 'retention_pct' in sequon_features_df.columns:
            for i, (_, row) in enumerate(sequon_features_df.iterrows()):
                ax.annotate(
                    f"{row['retention_pct']:.0f}%",
                    (i, max(row['rsa'], row['bfactor']) + 2),
                    ha='center',
                    fontsize=8,
                    color='green' if row['retention_pct'] > 50 else 'red'
                )

        plt.tight_layout()
        plt.savefig(fig_dir / "sequon_structural_context.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 6: Structural context analysis')
    parser.add_argument('--pdb_dir', required=True, help='Directory from previous steps')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    retention_dir = pdb_dir / "analysis" / "retention"
    denovo_dir = pdb_dir / "analysis" / "denovo"
    analysis_dir = pdb_dir / "analysis" / "structural"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load info
    with open(structure_dir / "structure_info.json") as f:
        structure_info = json.load(f)

    pdb_id = structure_info['pdb_id']
    pdb_path = structure_dir / f"{pdb_id}.pdb"

    print("=" * 70)
    print(f"STEP 6: STRUCTURAL CONTEXT ANALYSIS - {pdb_id}")
    print("=" * 70)

    # Calculate structural features
    print(f"\n1. Calculating structural features...")
    features_df = calculate_structural_features(pdb_path)
    print(f"  Calculated features for {len(features_df)} residues")

    # Load sequon positions
    with open(sequons_dir / "sequons_by_chain.json") as f:
        sequons_by_chain = json.load(f)

    # Create sequon features DataFrame
    print(f"\n2. Extracting sequon structural context...")
    sequon_features = []

    for chain_id, sequon_list in sequons_by_chain.items():
        for sequon in sequon_list:
            pos = sequon['position']

            # Get features for N position (index in features_df)
            chain_features = features_df[features_df['chain_id'] == chain_id]

            # Find the residue - need to match by sequence position
            # Assuming 0-indexed position maps to residue order
            if len(chain_features) > pos:
                row = chain_features.iloc[pos]
                sequon_features.append({
                    'chain_id': chain_id,
                    'position': pos,
                    'position_label': f"{chain_id}:{pos} ({sequon['sequon']})",
                    'sequon': sequon['sequon'],
                    'rsa': row['rsa'],
                    'bfactor': row['bfactor']
                })

    sequon_features_df = pd.DataFrame(sequon_features)

    # Try to load retention data
    retention_path = retention_dir / "retention_by_position.csv"
    if retention_path.exists():
        retention_df = pd.read_csv(retention_path)
        # Get unconstrained retention for comparison
        unconstrained = retention_df[retention_df['condition'] == 'Unconstrained']
        for _, row in unconstrained.iterrows():
            mask = ((sequon_features_df['chain_id'] == row['chain_id']) &
                   (sequon_features_df['position'] == row['position']))
            sequon_features_df.loc[mask, 'retention_pct'] = row['retention_pct']

    # Try to load de novo data
    denovo_features_df = pd.DataFrame()
    denovo_path = denovo_dir / "denovo_positions.csv"
    if denovo_path.exists():
        denovo_df = pd.read_csv(denovo_path)
        # Get unconstrained de novo rates
        unconstrained = denovo_df[denovo_df['condition'] == 'Unconstrained']

        # Map structural features to de novo positions
        denovo_features = []
        for _, row in unconstrained.iterrows():
            chain_features = features_df[features_df['chain_id'] == row['chain_id']]
            if len(chain_features) > row['position']:
                feat_row = chain_features.iloc[row['position']]
                denovo_features.append({
                    'chain_id': row['chain_id'],
                    'position': row['position'],
                    'position_label': row['position_label'],
                    'occurrence_pct': row['occurrence_pct'],
                    'rsa': feat_row['rsa'],
                    'bfactor': feat_row['bfactor']
                })

        denovo_features_df = pd.DataFrame(denovo_features)

    # Print summary
    print(f"\n3. Structural context of original sequons:")
    if len(sequon_features_df) > 0:
        for _, row in sequon_features_df.iterrows():
            ret_str = f", Retention: {row.get('retention_pct', 'N/A'):.1f}%" if 'retention_pct' in row else ""
            print(f"  {row['position_label']}: RSA={row['rsa']:.1f}%, B-factor={row['bfactor']:.1f}{ret_str}")

    # Save outputs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    features_df.to_csv(analysis_dir / "all_residue_features.csv", index=False)
    sequon_features_df.to_csv(analysis_dir / "sequon_features.csv", index=False)
    if len(denovo_features_df) > 0:
        denovo_features_df.to_csv(analysis_dir / "denovo_structural_features.csv", index=False)

    print(f"  Saved: {analysis_dir}/all_residue_features.csv")
    print(f"  Saved: {analysis_dir}/sequon_features.csv")

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations(features_df, sequon_features_df, denovo_features_df,
                         analysis_dir, pdb_id)

    # Summary
    print("\n" + "=" * 70)
    print("STEP 6 COMPLETE")
    print("=" * 70)

    if len(sequon_features_df) > 0:
        print(f"\nSequon Structural Summary:")
        print(f"  Average RSA: {sequon_features_df['rsa'].mean():.1f}%")
        print(f"  Average B-factor: {sequon_features_df['bfactor'].mean():.1f}")

        if 'retention_pct' in sequon_features_df.columns:
            # Correlation
            corr_rsa = sequon_features_df['rsa'].corr(sequon_features_df['retention_pct'])
            corr_bf = sequon_features_df['bfactor'].corr(sequon_features_df['retention_pct'])
            print(f"  Correlation (RSA vs Retention): {corr_rsa:.3f}")
            print(f"  Correlation (B-factor vs Retention): {corr_bf:.3f}")

    print(f"\nOutputs in {analysis_dir}/:")
    print(f"  - all_residue_features.csv")
    print(f"  - sequon_features.csv")
    print(f"  - figures/structural_distributions.png")
    print(f"  - figures/retention_vs_rsa.png")
    print(f"  - figures/sequon_structural_context.png")

    print(f"\n→ Optional: python 07_oligomer_comparison.py --pdb_dir {pdb_dir}")
    print(f"→ Final: python 08_generate_report.py --pdb_dir {pdb_dir}")


if __name__ == "__main__":
    main()
