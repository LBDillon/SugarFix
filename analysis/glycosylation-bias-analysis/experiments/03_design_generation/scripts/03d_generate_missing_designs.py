#!/usr/bin/env python3
"""
Generate Missing Unconstrained and Single-Fix Designs

This script generates unconstrained and single-fix designs for proteins that were
added during dataset expansion (1CF3, 1EW3, 1EZJ, 1GPE, 1LEG) but didn't have
designs from the original experiment.

Author: Claude Code
Date: 2026-01-20
"""

import os
import csv
from pathlib import Path

# Install ColabDesign if needed
try:
    from colabdesign.mpnn import mk_mpnn_model
except ImportError:
    print("Installing ColabDesign...")
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0")
    from colabdesign.mpnn import mk_mpnn_model

try:
    from Bio.PDB import PDBParser, PPBuilder, is_aa
    from Bio.SeqUtils import seq1
except ImportError:
    print("ERROR: BioPython not installed")
    print("Install with: pip install biopython")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
MANIFEST_PATH = DATA_DIR / "manifests" / "expanded_manifest_validated.csv"
PDB_ROOT = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "designs"

# Only process these 5 proteins
PROTEINS_TO_PROCESS = ['1CF3', '1EW3', '1EZJ', '1GPE', '1LEG']

NUM_SEQS = 32
SAMPLING_TEMP = 0.1
MODEL_NAME = "v_48_030"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_manifest():
    """Load manifest and filter for target proteins."""
    proteins = []
    with open(MANIFEST_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['pdb_id'] in PROTEINS_TO_PROCESS:
                proteins.append(row)
    return proteins

def pdb_to_mpnn_position(pdb_position, pdb_path, chain_id):
    """Convert PDB position to MPNN 0-indexed position."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [r for r in chain.get_residues() if is_aa(r)]
                for mpnn_idx, residue in enumerate(residues):
                    if residue.id[1] == pdb_position:
                        return mpnn_idx
    return None

def run_mpnn_design(mpnn_model, pdb_path, chain_id, fixed_positions=None):
    """Run ProteinMPNN design with optional fixed positions."""

    # Convert fixed positions list to string format if provided
    fix_pos_str = None
    if fixed_positions:
        fix_pos_str = ",".join(map(str, fixed_positions))

    # Prepare structure with optional fixed positions
    mpnn_model.prep_inputs(
        pdb_filename=str(pdb_path),
        chain=chain_id,
        fix_pos=fix_pos_str,
        verbose=False
    )

    # Run design
    out = mpnn_model.sample(
        num=max(1, NUM_SEQS // 32),
        batch=min(32, NUM_SEQS),
        temperature=SAMPLING_TEMP
    )

    return out['seq'], out['score']

def save_fasta(sequences, scores, output_path, condition):
    """Save sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for i, (seq, score) in enumerate(zip(sequences, scores)):
            f.write(f">design_{i}_{condition}_score_{score:.3f}\n")
            f.write(f"{seq}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*80)
    print("GENERATING MISSING DESIGNS FOR EXPANDED DATASET PROTEINS")
    print("="*80)
    print()
    print("Target proteins:", ", ".join(PROTEINS_TO_PROCESS))
    print()

    # Load manifest
    proteins = load_manifest()
    print(f"Found {len(proteins)} proteins to process")
    print()

    # Load ProteinMPNN model once
    print("Loading ProteinMPNN model...")
    mpnn_model = mk_mpnn_model(
        model_name=MODEL_NAME,
        backbone_noise=0.0
    )
    print(f"✓ Loaded model: {MODEL_NAME}")
    print()

    # Process each protein
    for i, protein in enumerate(proteins, 1):
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']

        print(f"[{i}/{len(proteins)}] {pdb_id} chain {chain_id}")

        # Find PDB file
        pdb_path = None
        for subdir in ['glycoproteins', 'controls']:
            for pattern in [f"{pdb_id}_{chain_id}.pdb", f"{pdb_id}.pdb"]:
                candidate = PDB_ROOT / subdir / pattern
                if candidate.exists():
                    pdb_path = candidate
                    break
            if pdb_path:
                break

        if not pdb_path:
            print(f"  ✗ PDB file not found")
            continue

        # Parse glycosite positions
        try:
            glycosite_positions = [int(x.strip()) for x in protein['glycosite_positions'].split(',')]
        except:
            print(f"  ✗ Could not parse glycosite positions")
            continue

        print(f"  Glycosites (PDB): {glycosite_positions}")

        # Convert to MPNN positions
        mpnn_glycosites = []
        for pdb_pos in glycosite_positions:
            mpnn_pos = pdb_to_mpnn_position(pdb_pos, pdb_path, chain_id)
            if mpnn_pos is not None:
                mpnn_glycosites.append(mpnn_pos)

        if len(mpnn_glycosites) == 0:
            print(f"  ✗ No valid glycosite positions found")
            continue

        print(f"  MPNN positions: {mpnn_glycosites}")

        # Generate UNCONSTRAINED designs
        output_file = OUTPUT_DIR / f"{pdb_id}_{chain_id}_unconstrained.fasta"
        if output_file.exists():
            print(f"  Unconstrained: Already exists, skipping")
        else:
            print(f"  Generating unconstrained designs...")
            try:
                sequences, scores = run_mpnn_design(mpnn_model, pdb_path, chain_id,
                                                    fixed_positions=None)
                save_fasta(sequences, scores, output_file, "unconstrained")
                print(f"    ✓ Saved: {output_file.name}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

        # Generate SINGLE-FIX designs (only N positions)
        output_file = OUTPUT_DIR / f"{pdb_id}_{chain_id}_fixed.fasta"
        if output_file.exists():
            print(f"  Single-fix: Already exists, skipping")
        else:
            print(f"  Generating single-fix designs...")
            try:
                sequences, scores = run_mpnn_design(mpnn_model, pdb_path, chain_id,
                                                    fixed_positions=mpnn_glycosites)
                save_fasta(sequences, scores, output_file, "fixed")
                print(f"    ✓ Saved: {output_file.name}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

        print()

    print("="*80)
    print("COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Rerun analysis: python3 scripts/09_analyze_multifix_designs.py")
    print("  2. Regenerate figures: python3 scripts/10_visualize_threeway_comparison.py")
    print()

if __name__ == '__main__':
    main()
