#!/usr/bin/env python3
"""
ProteinMPNN Design Experiment: Expanded Validated Dataset

Runs ProteinMPNN on all proteins from the expanded validated manifest.
Skips proteins that already have design files.

Two conditions per glycoprotein:
1. Unconstrained: MPNN freely redesigns the entire sequence
2. Glycosite-fixed: The asparagine (N) at glycosylation sites is held fixed

Author: Claude Code
Date: 2026-01-20
"""

import os
import csv
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Install ColabDesign if needed
try:
    from colabdesign.mpnn import mk_mpnn_model
except ImportError:
    print("Installing ColabDesign...")
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0")
    from colabdesign.mpnn import mk_mpnn_model

try:
    from Bio.PDB import PDBParser, is_aa
except ImportError:
    print("Installing BioPython...")
    os.system("pip -q install biopython")
    from Bio.PDB import PDBParser, is_aa

# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_PATH = "data/glyco_benchmark/manifests/expanded_manifest_validated.csv"
PDB_ROOT = "data/glyco_benchmark/raw"
OUTPUT_DIR = "data/glyco_benchmark/designs"
NUM_SEQS = 32
SAMPLING_TEMP = 0.1
MODEL_NAME = "v_48_030"

# =============================================================================
# HELPER FUNCTIONS (copied from original script)
# =============================================================================

def find_pdb_file(pdb_id, chain_id, pdb_root):
    """Find PDB file in the directory structure."""
    patterns = [
        f"{pdb_id}_{chain_id}.pdb",
        f"{pdb_id}.pdb",
    ]

    for subdir in ['glycoproteins', 'controls']:
        base_path = Path(pdb_root) / subdir
        for pattern in patterns:
            path = base_path / pattern
            if path.exists():
                return str(path)

    return None

def parse_glycosites(glycosite_string):
    """Parse glycosite positions from manifest string (e.g., '64,172,320')"""
    if not glycosite_string or glycosite_string == '':
        return []
    try:
        return [int(x.strip()) for x in glycosite_string.split(',') if x.strip()]
    except ValueError:
        return []

def pdb_to_mpnn_position(pdb_residue_num, pdb_path, chain_id):
    """
    Convert PDB residue number to MPNN internal position.
    Returns the MPNN position (1-indexed) or None if not found.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    residues = [r for r in chain.get_residues() if is_aa(r)]

                    # Find exact match
                    for mpnn_pos, residue in enumerate(residues, start=1):
                        if residue.id[1] == pdb_residue_num:
                            return mpnn_pos

                    return None

        return None
    except Exception as e:
        print(f"    Error mapping position {pdb_residue_num}: {e}")
        return None

def get_first_nonzero_chain(pdb_path):
    """Get the first chain in the PDB file that contains amino acids."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('temp', pdb_path)

        for model in structure:
            for chain in model:
                residues = [r for r in chain.get_residues() if is_aa(r)]
                if len(residues) > 0:
                    return chain.id

        return None
    except Exception as e:
        print(f"    Error detecting chain: {e}")
        return None

def run_mpnn_design(mpnn_model, pdb_path, chain, fix_pos=None,
                    num_seqs=32, temperature=0.1):
    """
    Run ProteinMPNN design and return results.

    Args:
        fix_pos: String of positions to fix (e.g., "1,5,10") or None

    Returns:
        dict with 'sequences', 'scores', 'seqid' lists
    """
    try:
        mpnn_model.prep_inputs(
            pdb_filename=pdb_path,
            chain=chain,
            fix_pos=fix_pos,
            verbose=False
        )

        out = mpnn_model.sample(
            num=max(1, num_seqs // 32),
            batch=min(32, num_seqs),
            temperature=temperature
        )

        return {
            'sequences': out['seq'],
            'scores': out['score'],
            'seqid': out['seqid']
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None

def save_designs(results, output_path, protein_info, condition):
    """Save designed sequences to FASTA file"""
    if results is None:
        return

    with open(output_path, 'w') as f:
        for i, (seq, score, seqid) in enumerate(zip(
            results['sequences'], results['scores'], results['seqid']
        )):
            pdb_id = protein_info.get('pdb_id', 'unknown')
            header = f">{pdb_id}_{condition}_design{i:02d}|score={score:.3f}|seqid={seqid:.3f}"
            f.write(f"{header}\n{seq}\n")

def check_existing_designs(pdb_id, chain_id):
    """Check which design files already exist for a protein."""
    unconstrained_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_unconstrained.fasta"
    fixed_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_fixed.fasta"

    return {
        'unconstrained': unconstrained_path.exists(),
        'fixed': fixed_path.exists()
    }

def load_manifest(path):
    """Load the expanded manifest."""
    proteins = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            proteins.append(row)
    return proteins

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the design experiment on proteins missing design files."""

    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    proteins = load_manifest(MANIFEST_PATH)
    print(f"="*80)
    print(f"PROTEINMPNN DESIGN EXPERIMENT: EXPANDED VALIDATED DATASET")
    print(f"="*80)
    print(f"\nLoaded {len(proteins)} proteins from expanded manifest")

    # Check which need designs
    proteins_to_run = []
    for protein in proteins:
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']
        existing = check_existing_designs(pdb_id, chain_id)

        if not existing['unconstrained'] or not existing['fixed']:
            proteins_to_run.append({
                'protein': protein,
                'needs_unconstrained': not existing['unconstrained'],
                'needs_fixed': not existing['fixed']
            })

    if len(proteins_to_run) == 0:
        print("\n✓ All proteins already have design files!")
        print("\nTo re-run all designs, delete files in:")
        print(f"  {OUTPUT_DIR}")
        return

    print(f"\n{'='*80}")
    print(f"PROTEINS NEEDING DESIGNS: {len(proteins_to_run)}")
    print(f"{'='*80}")
    for item in proteins_to_run:
        p = item['protein']
        needs = []
        if item['needs_unconstrained']:
            needs.append('unconstrained')
        if item['needs_fixed']:
            needs.append('fixed')
        print(f"  {p['pdb_id']:6s} chain {p['chain_id']:1s}: {', '.join(needs):20s} ({p['n_glycosites']} sequons)")

    # Initialize MPNN model
    print(f"\n{'='*80}")
    print("Loading ProteinMPNN model...")
    try:
        mpnn_model = mk_mpnn_model(MODEL_NAME)
        print(f"✓ Loaded model: {MODEL_NAME}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure ColabDesign is installed")
        return

    # Run designs
    proteins_processed = 0
    proteins_failed = 0

    print(f"\n{'='*80}")
    print("RUNNING DESIGNS")
    print(f"{'='*80}")

    for idx, item in enumerate(proteins_to_run):
        protein = item['protein']
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']
        glycosites_pdb = parse_glycosites(protein.get('glycosite_positions', ''))

        print(f"\n[{idx+1}/{len(proteins_to_run)}] {pdb_id} chain {chain_id}")
        print(f"  Glycosites: {glycosites_pdb}")

        # Find PDB file
        pdb_path = find_pdb_file(pdb_id, chain_id, PDB_ROOT)
        if not pdb_path:
            print(f"  ✗ PDB file not found")
            proteins_failed += 1
            continue

        # Auto-detect chain
        actual_chain = get_first_nonzero_chain(pdb_path)
        if not actual_chain:
            print(f"  ✗ No amino acids found")
            proteins_failed += 1
            continue

        if actual_chain != chain_id:
            chain_id = actual_chain

        # Convert positions
        glycosites_mpnn = []
        for pdb_pos in glycosites_pdb:
            mpnn_pos = pdb_to_mpnn_position(pdb_pos, pdb_path, chain_id)
            if mpnn_pos:
                glycosites_mpnn.append(mpnn_pos)

        # UNCONSTRAINED
        if item['needs_unconstrained']:
            print(f"  Running unconstrained...", end=' ', flush=True)
            results = run_mpnn_design(
                mpnn_model, pdb_path, chain_id,
                fix_pos=None,
                num_seqs=NUM_SEQS,
                temperature=SAMPLING_TEMP
            )

            if results:
                output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_unconstrained.fasta"
                save_designs(results, output_path, protein, "unconstrained")
                print(f"✓ ({len(results['sequences'])} designs)")
            else:
                print(f"✗ FAILED")

        # FIXED
        if item['needs_fixed'] and glycosites_mpnn:
            print(f"  Running fixed...", end=' ', flush=True)
            fix_pos_string = ",".join(str(p) for p in glycosites_mpnn)

            results = run_mpnn_design(
                mpnn_model, pdb_path, chain_id,
                fix_pos=fix_pos_string,
                num_seqs=NUM_SEQS,
                temperature=SAMPLING_TEMP
            )

            if results:
                output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_fixed.fasta"
                save_designs(results, output_path, protein, "fixed")
                print(f"✓ ({len(results['sequences'])} designs)")
            else:
                print(f"✗ FAILED")

        proteins_processed += 1

    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Proteins processed: {proteins_processed}/{len(proteins_to_run)}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_experiment()
