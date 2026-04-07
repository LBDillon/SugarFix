#!/usr/bin/env python3
"""
ProteinMPNN Design Experiment: Glycoprotein Sequon Retention
Runs two conditions per glycoprotein: unconstrained vs glycosite-fixed

Tests whether ProteinMPNN can design sequences while preserving
N-X-S/T glycosylation sequons.
"""

import os
import re
import csv
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

# Install ColabDesign if needed
try:
    from colabdesign.mpnn import mk_mpnn_model
except ImportError:
    print("Installing ColabDesign...")
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0")
    from colabdesign.mpnn import mk_mpnn_model

# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_PATH = "data/glyco_benchmark/manifests/benchmark_manifest_simple.csv"
PDB_ROOT = "data/glyco_benchmark/raw"
OUTPUT_DIR = "data/glyco_benchmark/designs"
NUM_SEQS = 32
SAMPLING_TEMP = 0.1
MODEL_NAME = "v_48_030"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_manifest(path, protein_class='glycoprotein'):
    """Load benchmark manifest and filter by protein class"""
    proteins = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['protein_class'] == protein_class:
                proteins.append(row)
    return proteins

def find_pdb_file(pdb_id, chain_id, pdb_root):
    """
    Find PDB file in the directory structure.
    Looks in both glycoproteins/ and controls/ directories.
    """
    # Try different naming conventions and locations
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
    
    # Fallback: try without chain specified
    for subdir in ['glycoproteins', 'controls']:
        for pattern in patterns:
            path = Path(pdb_root) / subdir / pattern
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
    MPNN uses 1-indexed positions based on residues in the structure.
    
    Returns the MPNN position (1-indexed) or None if not found.
    """
    try:
        from Bio.PDB import PDBParser, is_aa
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        # Get residues in the specified chain
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    # Get only amino acid residues (exclude heteroatoms)
                    residues = [r for r in chain.get_residues() 
                               if is_aa(r)]
                    
                    # Find the MPNN position (1-indexed)
                    for mpnn_pos, residue in enumerate(residues, start=1):
                        # residue.id[1] is the residue number (PDB numbering)
                        # residue.id[0] is the insertion code (usually ' ')
                        if residue.id[1] == pdb_residue_num:
                            return mpnn_pos
                    
                    # If exact match not found, try fuzzy matching (close residue numbers)
                    # This handles cases where the PDB might have renumbered residues
                    closest = None
                    min_diff = float('inf')
                    for mpnn_pos, residue in enumerate(residues, start=1):
                        diff = abs(residue.id[1] - pdb_residue_num)
                        if diff < min_diff:
                            min_diff = diff
                            closest = mpnn_pos
                    
                    # Only use fuzzy match if very close (within 3 residues)
                    if min_diff <= 3:
                        print(f"      (fuzzy match: PDB {pdb_residue_num} → {pdb_residue_num - min_diff} found → MPNN {closest})")
                        return closest
                    
                    return None
        
        return None
    except Exception as e:
        print(f"    Error mapping position {pdb_residue_num}: {e}")
        return None

def check_sequon(sequence, position):
    """
    Check if N-X-S/T sequon exists at position (0-indexed in sequence).
    Returns: 'NXS', 'NXT', 'partial', or 'none'
    """
    if position < 0 or position + 2 >= len(sequence):
        return 'none'
    
    aa_0 = sequence[position]      # Should be N
    aa_1 = sequence[position + 1]  # Should be anything except P
    aa_2 = sequence[position + 2]  # Should be S or T
    
    if aa_0 != 'N':
        return 'none'
    if aa_1 == 'P':
        return 'partial'  # N-P-S/T is not glycosylated
    if aa_2 == 'S':
        return 'NXS'
    if aa_2 == 'T':
        return 'NXT'
    return 'partial'

def get_first_nonzero_chain(pdb_path):
    """
    Get the first chain in the PDB file that contains amino acids.
    Handles cases where the manifest specifies a chain that's empty or doesn't exist.
    """
    from Bio.PDB import PDBParser, is_aa
    
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
        # Try to load and prepare inputs
        mpnn_model.prep_inputs(
            pdb_filename=pdb_path,
            chain=chain,
            fix_pos=fix_pos,
            verbose=False
        )
        
        # Sample sequences
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
        error_str = str(e).lower()
        
        # Check for specific common errors
        if "0 models" in error_str or "found 0" in error_str:
            print(f"    Error: PDB file formatting issue (check file integrity)")
        elif "not found" in error_str or "chain" in error_str:
            print(f"    Error: Chain or position not recognized by ColabDesign")
            print(f"           (this can happen with fixed positions on multi-chain files)")
        else:
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
            header = f">{protein_info['pdb_id']}_{condition}_design{i:02d}|score={score:.3f}|seqid={seqid:.3f}"
            f.write(f"{header}\n{seq}\n")

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the full design experiment on all glycoproteins"""
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    proteins = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(proteins)} glycoproteins from manifest")
    
    # Initialize MPNN model once
    print("Loading ProteinMPNN model...")
    try:
        mpnn_model = mk_mpnn_model(MODEL_NAME)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure ColabDesign is installed and CUDA is available")
        return
    
    # Results storage
    all_results = []
    proteins_processed = 0
    proteins_failed = 0
    
    for protein_idx, protein in enumerate(proteins):
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']
        glycosites_pdb = parse_glycosites(protein.get('glycosite_positions', ''))
        
        print(f"\n{'='*70}")
        print(f"[{protein_idx+1}/{len(proteins)}] Processing: {pdb_id} chain {chain_id}")
        print(f"Glycosylation sites (PDB numbering): {glycosites_pdb}")
        
        # Find PDB file
        pdb_path = find_pdb_file(pdb_id, chain_id, PDB_ROOT)
        if not pdb_path:
            print(f"  ✗ PDB file not found, skipping")
            proteins_failed += 1
            continue
        
        print(f"  PDB path: {pdb_path}")
        
        # Auto-detect the actual chain with amino acids (handles multi-chain complexes)
        actual_chain = get_first_nonzero_chain(pdb_path)
        if not actual_chain:
            print(f"  ✗ No chain with amino acids found in PDB, skipping")
            proteins_failed += 1
            continue
        
        if actual_chain != chain_id:
            print(f"  ⚠️  Chain mismatch: manifest says '{chain_id}' but file has '{actual_chain}' with residues")
            chain_id = actual_chain  # Use the actual chain
        
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
        print(f"\n  CONDITION 1: UNCONSTRAINED design")
        results_unconstrained = run_mpnn_design(
            mpnn_model, pdb_path, chain_id,
            fix_pos=None,
            num_seqs=NUM_SEQS,
            temperature=SAMPLING_TEMP
        )
        
        if results_unconstrained:
            # Save designs
            output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_unconstrained.fasta"
            save_designs(results_unconstrained, output_path, protein, "unconstrained")
            print(f"    ✓ Saved {len(results_unconstrained['sequences'])} designs")
            
            # Analyze sequon retention
            for seq_idx, seq in enumerate(results_unconstrained['sequences']):
                for site_idx, mpnn_pos in enumerate(glycosites_mpnn):
                    seq_pos = mpnn_pos - 1
                    sequon_status = check_sequon(seq, seq_pos)
                    all_results.append({
                        'pdb_id': pdb_id,
                        'chain_id': chain_id,
                        'condition': 'unconstrained',
                        'design_idx': seq_idx,
                        'glycosite_pdb': glycosites_pdb[site_idx] if site_idx < len(glycosites_pdb) else None,
                        'glycosite_mpnn': mpnn_pos,
                        'sequon_status': sequon_status,
                        'design_score': results_unconstrained['scores'][seq_idx],
                        'seqid': results_unconstrained['seqid'][seq_idx]
                    })
        else:
            print(f"    ✗ Failed to generate unconstrained designs")
        
        # ----- CONDITION 2: GLYCOSITE-FIXED -----
        if glycosites_mpnn:
            print(f"\n  CONDITION 2: GLYCOSITE-FIXED design")
            fix_pos_string = ",".join(str(p) for p in glycosites_mpnn)
            print(f"    Fixing positions: {fix_pos_string}")
            
            # Only attempt fixed design if unconstrained worked
            if results_unconstrained is None:
                print(f"    ✗ Skipped (unconstrained design failed - PDB issue)")
            else:
                results_fixed = run_mpnn_design(
                    mpnn_model, pdb_path, chain_id,
                    fix_pos=fix_pos_string,
                    num_seqs=NUM_SEQS,
                    temperature=SAMPLING_TEMP
                )
                
                if results_fixed:
                    # Save designs
                    output_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}_fixed.fasta"
                    save_designs(results_fixed, output_path, protein, "fixed")
                    print(f"    ✓ Saved {len(results_fixed['sequences'])} designs")
                    
                    # Analyze sequon retention (should be 100%)
                    for seq_idx, seq in enumerate(results_fixed['sequences']):
                        for site_idx, mpnn_pos in enumerate(glycosites_mpnn):
                            seq_pos = mpnn_pos - 1
                            sequon_status = check_sequon(seq, seq_pos)
                            all_results.append({
                                'pdb_id': pdb_id,
                                'chain_id': chain_id,
                                'condition': 'fixed',
                                'design_idx': seq_idx,
                                'glycosite_pdb': glycosites_pdb[site_idx] if site_idx < len(glycosites_pdb) else None,
                                'glycosite_mpnn': mpnn_pos,
                                'sequon_status': sequon_status,
                                'design_score': results_fixed['scores'][seq_idx],
                                'seqid': results_fixed['seqid'][seq_idx]
                            })
                else:
                    print(f"    ✗ Failed to generate fixed designs")
        else:
            print(f"\n  CONDITION 2: SKIPPED (no valid glycosite positions)")
        
        proteins_processed += 1
    
    # Save all results
    if all_results:
        results_path = Path(OUTPUT_DIR) / "sequon_retention_analysis.csv"
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*70}")
        print(f"✓ Saved detailed results to {results_path}")
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY: Sequon Retention Analysis")
        print("="*70)
        
        for condition in ['unconstrained', 'fixed']:
            cond_results = results_df[results_df['condition'] == condition]
            if len(cond_results) > 0:
                total = len(cond_results)
                preserved = len(cond_results[cond_results['sequon_status'].isin(['NXS', 'NXT'])])
                partially = len(cond_results[cond_results['sequon_status'] == 'partial'])
                none = len(cond_results[cond_results['sequon_status'] == 'none'])
                
                print(f"\n{condition.upper()}:")
                print(f"  Total sequons designed: {total}")
                print(f"  Preserved (NXS/NXT):   {preserved} ({100*preserved/total:.1f}%)")
                print(f"  Partial retained:      {partially} ({100*partially/total:.1f}%)")
                print(f"  Lost:                  {none} ({100*none/total:.1f}%)")
                print(f"  Mean score:            {cond_results['design_score'].mean():.3f}")
                print(f"  Mean seqid:            {cond_results['seqid'].mean():.3f}")
        
        # Comparison
        print(f"\n" + "="*70)
        print("COMPARISON: Unconstrained vs Fixed")
        print("="*70)
        
        unconstrained = results_df[results_df['condition'] == 'unconstrained']
        fixed = results_df[results_df['condition'] == 'fixed']
        
        if len(unconstrained) > 0 and len(fixed) > 0:
            print(f"\nScore penalty (fixed - unconstrained):")
            print(f"  {fixed['design_score'].mean() - unconstrained['design_score'].mean():.4f}")
            print(f"  (negative = fixed designs score lower)")
            
            print(f"\nSequence identity penalty (fixed - unconstrained):")
            print(f"  {fixed['seqid'].mean() - unconstrained['seqid'].mean():.4f}")
            print(f"  (negative = fixed designs more different from original)")
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"  Proteins processed: {proteins_processed}")
    print(f"  Proteins failed: {proteins_failed}")
    print(f"  Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_experiment()
