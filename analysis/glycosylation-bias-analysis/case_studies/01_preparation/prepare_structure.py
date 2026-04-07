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
import ssl
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Work around macOS SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, is_aa
    from Bio.SeqUtils import seq1
except ImportError:
    print("Installing BioPython...")
    os.system("pip install biopython")
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, is_aa
    from Bio.SeqUtils import seq1


def convert_mmcif_to_pdb(mmcif_path, pdb_path):
    """Convert an mmCIF structure file into legacy PDB format."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(mmcif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    return pdb_path.exists()


def download_pdb(pdb_id, output_path, biological_assembly=False):
    """Download a structure from RCSB, falling back from PDB to mmCIF when needed.

    The pipeline mostly expects a `.pdb` file on disk. Some newer or larger
    RCSB entries only provide mmCIF downloads, so this helper will:
    1. try the legacy PDB download,
    2. if that fails, download the mmCIF file alongside it, and
    3. convert the mmCIF into a `.pdb` for downstream tools.
    """
    pdb_id = pdb_id.upper()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if biological_assembly:
        primary_url = f"https://files.rcsb.org/download/{pdb_id}.pdb1"
        fallback_url = f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif"
    else:
        primary_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        fallback_url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    primary_error = None
    try:
        urlretrieve(primary_url, output_path)
        return True
    except Exception as e:
        primary_error = e

    mmcif_path = output_path.with_suffix(".cif")
    try:
        urlretrieve(fallback_url, mmcif_path)
    except Exception as fallback_error:
        print(f"  Error downloading PDB: {primary_error}")
        print(f"  Error downloading mmCIF fallback: {fallback_error}")
        return False

    try:
        if convert_mmcif_to_pdb(mmcif_path, output_path):
            print(
                f"  Downloaded mmCIF fallback for {pdb_id} and converted it to "
                f"{output_path.name}"
            )
            return True
    except Exception as conversion_error:
        print(f"  Error converting mmCIF to PDB: {conversion_error}")

    return False


def parse_structure(pdb_path):
    """Parse PDB and extract chain information."""
    pdb_path = Path(pdb_path)

    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
    else:
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('protein', str(pdb_path))
        except Exception:
            mmcif_path = pdb_path.with_suffix(".cif")
            if not mmcif_path.exists():
                raise
            print(f"  Falling back to mmCIF parser using {mmcif_path.name}")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('protein', str(mmcif_path))

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

def parse_missing_residues(pdb_path):
    """Parse REMARK 465 to find missing residues in the structure.

    Returns dict: chain_id -> list of (resname, resnum) tuples.
    """
    missing = {}
    in_remark465 = False
    header_seen = False

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("REMARK 465"):
                if in_remark465:
                    break
                continue
            in_remark465 = True
            content = line[10:].strip()
            if not content or content.startswith("THE FOLLOWING") or content.startswith("EXPERIMENT"):
                continue
            if content.startswith("M RES"):
                header_seen = True
                continue
            if not header_seen:
                continue
            parts = content.split()
            if len(parts) >= 3:
                # Format: [M] RES C SSSEQ [I]
                # M is optional model number
                try:
                    if parts[0].isdigit():
                        resname, chain, resnum = parts[1], parts[2], int(parts[3])
                    else:
                        resname, chain, resnum = parts[0], parts[1], int(parts[2])
                    missing.setdefault(chain, []).append((resname, resnum))
                except (ValueError, IndexError):
                    continue

    return missing


def create_protein_only_pdb(input_pdb_path, output_pdb_path, protein_chains):
    """Create a PDB file containing only protein chains."""
    protein_chain_ids = {chain['chain_id'] for chain in protein_chains}
    
    with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
        for line in infile:
            if line.startswith('ATOM'):
                chain_id = line[21:22]
                if chain_id in protein_chain_ids:
                    outfile.write(line)
            elif line.startswith('TER'):
                chain_id = line[21:22]
                if chain_id in protein_chain_ids:
                    outfile.write(line)
            elif line.startswith('HETATM'):
                continue
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

    # Check for missing residues
    print(f"\n   Checking for missing residues (REMARK 465)...")
    missing_residues = parse_missing_residues(pdb_path)
    if missing_residues:
        total_missing = sum(len(v) for v in missing_residues.values())
        print(f"  WARNING: {total_missing} missing residues detected:")
        for chain_id, residues in sorted(missing_residues.items()):
            nums = [r[1] for r in residues]
            print(f"    Chain {chain_id}: {len(residues)} missing (resnums: {nums[:10]}{'...' if len(nums) > 10 else ''})")
    else:
        print(f"  No missing residues detected.")

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
    if missing_residues:
        info['missing_residues'] = {
            chain: [{"resname": rn, "resnum": rnum} for rn, rnum in residues]
            for chain, residues in missing_residues.items()
        }
        info['total_missing_residues'] = sum(len(v) for v in missing_residues.values())
    else:
        info['missing_residues'] = {}
        info['total_missing_residues'] = 0

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

    # Extract glycan trees
    print(f"\n7. Extracting glycan trees from PDB...")
    try:
        from extract_pdb_glycans import extract_glycan_trees
        glycan_trees = extract_glycan_trees(pdb_path)
        if glycan_trees:
            glycan_path = structure_dir / "glycan_trees.json"
            with open(glycan_path, "w") as f:
                json.dump(glycan_trees, f, indent=2)
            print(f"  Found {len(glycan_trees)} glycosylation site(s):")
            for site_key, ginfo in sorted(glycan_trees.items()):
                print(f"    {site_key}: {ginfo['residues_string']} ({ginfo['n_sugars']} sugars)")
            print(f"  Saved: {glycan_path}")
        else:
            print(f"  No N-linked glycan trees found in PDB.")
    except ImportError:
        print(f"  Skipped (extract_pdb_glycans not available)")

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

    print(f"\n→ Next: python identify_sequons.py --pdb_dir {output_dir}")


if __name__ == "__main__":
    main()
