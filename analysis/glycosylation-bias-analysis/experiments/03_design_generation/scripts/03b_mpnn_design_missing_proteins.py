#!/usr/bin/env python3
"""
ProteinMPNN Design Experiment: Generate Missing Designs for Expanded Dataset

This script runs ProteinMPNN on proteins from the expanded manifest that don't
have design files yet. It uses the same two-condition setup:
1. Unconstrained: MPNN freely redesigns the entire sequence
2. Glycosite-fixed: The asparagine (N) at known glycosylation sites is held fixed

Author: Claude Code
Date: 2026-01-20
"""

import os
import csv
from pathlib import Path
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Install ColabDesign if needed
try:
    from colabdesign.mpnn import mk_mpnn_model
except ImportError:
    print("Installing ColabDesign...")
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0")
    from colabdesign.mpnn import mk_mpnn_model

# Import helper functions from original script
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import the helper functions from the original design script
from importlib import import_module
import importlib.util

spec = importlib.util.spec_from_file_location("design_helpers",
    "scripts/03_mpnn_design_experiment.py")
if spec and spec.loader:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    # Import the functions we need
    find_pdb_file = helpers.find_pdb_file
    parse_glycosites = helpers.parse_glycosites
    pdb_to_mpnn_position = helpers.pdb_to_mpnn_position
    get_first_nonzero_chain = helpers.get_first_nonzero_chain
    run_mpnn_design = helpers.run_mpnn_design
    save_designs = helpers.save_designs
    check_sequon = helpers.check_sequon

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
# MAIN EXPERIMENT
# =============================================================================

def check_existing_designs(pdb_id, chain_id):
    """Check which design files already exist for a protein."""
    patterns = [
        f"{pdb_id}_{chain_id}_unconstrained.fasta",
        f"{pdb_id}_{chain_id}_fixed.fasta"
    ]

    existing = {}
    for pattern in patterns:
        path = Path(OUTPUT_DIR) / pattern
        condition = 'unconstrained' if 'unconstrained' in pattern else 'fixed'
        existing[condition] = path.exists()

    return existing

def load_manifest(path):
    """Load the expanded manifest."""
    proteins = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            proteins.append(row)
    return proteins

def run_missing_designs():
    """
    Run ProteinMPNN designs for proteins that don't have design files yet.
    """

    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    proteins = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(proteins)} proteins from expanded manifest")

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
        return

    print(f"\nNeed to generate designs for {len(proteins_to_run)} proteins:")
    for item in proteins_to_run:
        p = item['protein']
        needs = []
        if item['needs_unconstrained']:
            needs.append('unconstrained')
        if item['needs_fixed']:
            needs.append('fixed')
        print(f"  - {p['pdb_id']} chain {p['chain_id']}: {', '.join(needs)}")

    # Initialize MPNN model once
    print("\nLoading ProteinMPNN model...")
    try:
        mpnn_model = mk_mpnn_model(MODEL_NAME)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure ColabDesign is installed and CUDA is available")
        return

    # Run designs
    all_results = []
    proteins_processed = 0
    proteins_failed = 0

    for idx, item in enumerate(proteins_to_run):
        protein = item['protein']
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']
        glycosites_pdb = parse_glycosites(protein.get('glycosite_positions', ''))

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(proteins_to_run)}] Processing: {pdb_id} chain {chain_id}")
        print(f"Glycosylation sites (PDB numbering): {glycosites_pdb}")
        print(f"Conditions needed: {', '.join(['unconstrained' if item['needs_unconstrained'] else '', 'fixed' if item['needs_fixed'] else '']).strip(', ')}")

        # Find PDB file
        pdb_path = find_pdb_file(pdb_id, chain_id, PDB_ROOT)
        if not pdb_path:
            print(f"  ✗ PDB file not found, skipping")
            proteins_failed += 1
            continue

        print(f"  PDB path: {pdb_path}")

        # Auto-detect chain
        actual_chain = get_first_nonzero_chain(pdb_path)
        if not actual_chain:
            print(f"  ✗ No chain with amino acids found in PDB, skipping")
            proteins_failed += 1
            continue

        if actual_chain != chain_id:
            print(f"  ⚠️  Chain mismatch: manifest says '{chain_id}' but file has '{actual_chain}'")
            chain_id = actual_chain

        print(f"  Using chain: {chain_id}")

        # Convert PDB positions to MPNN positions
        glycosites_mpnn = []
        for pdb_pos in glycosites_pdb:
            mpnn_pos = pdb_to_mpnn_position(pdb_pos, pdb_path, chain_id)
            if mpnn_pos:
                glycosites_mpnn.append(mpnn_pos)
                print(f"    Glycosite {pdb_pos} (PDB) → {mpnn_pos} (MPNN)")
            else:
                print(f"    ✗ Could not map glycosite {pdb_pos}")

        # ----- CONDITION 1: UNCONSTRAINED -----
        if item['needs_unconstrained']:
            print(f"\n  CONDITION 1: UNCONSTRAINED design")
            results_unconstrained = run_mpnn_design(
                mpnn_model, pdb_path, chain_id,
                fix_pos=None,
                num_seqs=NUM_SEQS,
                temperature=SAMPLING_TEMP
            )

            if results_unconstrained:
                output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_unconstrained.fasta"
                save_designs(results_unconstrained, output_path, protein, "unconstrained")
                print(f"    ✓ Saved {len(results_unconstrained['sequences'])} designs")
            else:
                print(f"    ✗ Failed to generate unconstrained designs")

        # ----- CONDITION 2: GLYCOSITE-FIXED -----
        if item['needs_fixed'] and glycosites_mpnn:
            print(f"\n  CONDITION 2: GLYCOSITE-FIXED design")
            fix_pos_string = ",".join(str(p) for p in glycosites_mpnn)
            print(f"    Fixing positions: {fix_pos_string}")

            results_fixed = run_mpnn_design(
                mpnn_model, pdb_path, chain_id,
                fix_pos=fix_pos_string,
                num_seqs=NUM_SEQS,
                temperature=SAMPLING_TEMP
            )

            if results_fixed:
                output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_fixed.fasta"
                save_designs(results_fixed, output_path, protein, "fixed")
                print(f"    ✓ Saved {len(results_fixed['sequences'])} designs")
            else:
                print(f"    ✗ Failed to generate fixed designs")
        elif item['needs_fixed'] and not glycosites_mpnn:
            print(f"\n  CONDITION 2: SKIPPED (no valid glycosite positions)")

        proteins_processed += 1

    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"  Proteins processed: {proteins_processed}")
    print(f"  Proteins failed: {proteins_failed}")
    print(f"  Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_missing_designs()
