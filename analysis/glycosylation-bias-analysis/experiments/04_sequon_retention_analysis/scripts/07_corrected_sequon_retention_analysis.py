#!/usr/bin/env python3
"""
CORRECTED Sequon Retention Analysis

This script implements the CORRECT analysis methodology:

For UNCONSTRAINED designs:
  - Check if the sequon is preserved AT THE ORIGINAL POSITION
  - NOT: count how many sequons appear anywhere in the sequence

For FIXED designs:
  - Verify that N was actually fixed at the glycosite
  - Check if the full N-X-S/T motif is preserved

This replaces the faulty analysis in scripts/05_analyze_sequon_retention.py

Author: Claude Code
Date: 2026-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import csv

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
DESIGNS_DIR = DATA_DIR / "designs"
MANIFEST_PATH = DATA_DIR / "manifests" / "expanded_manifest_validated.csv"
PDB_ROOT = DATA_DIR / "raw"

OUTPUT_DIR = DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SEQUON CHECKING
# =============================================================================

def check_sequon(sequence: str, position: int) -> Optional[str]:
    """
    Check if N-X-S/T sequon exists at position (0-indexed).

    Returns:
        'NXS': Full sequon with Serine
        'NXT': Full sequon with Threonine
        'N_only': Has N but sequon is broken (N-P-X or N-X-other)
        'no_N': No asparagine at position
        None: Position out of bounds
    """
    if position < 0 or position + 2 >= len(sequence):
        return None

    aa_0 = sequence[position]
    aa_1 = sequence[position + 1]
    aa_2 = sequence[position + 2]

    if aa_0 != 'N':
        return 'no_N'

    if aa_1 == 'P':
        return 'N_only'  # N-P-S/T is not glycosylated

    if aa_2 == 'S':
        return 'NXS'
    elif aa_2 == 'T':
        return 'NXT'
    else:
        return 'N_only'

# =============================================================================
# SEQUENCE LOADING
# =============================================================================

def load_wild_type_sequence(pdb_id: str, chain_id: str) -> Optional[str]:
    """Load wild-type sequence from PDB file."""
    # Try glycoproteins first, then controls
    for subdir in ['glycoproteins', 'controls']:
        for pattern in [f"{pdb_id}_{chain_id}.pdb", f"{pdb_id}.pdb"]:
            pdb_path = PDB_ROOT / subdir / pattern
            if pdb_path.exists():
                try:
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('protein', str(pdb_path))

                    for model in structure:
                        for chain in model:
                            if chain.id == chain_id:
                                residues = [r for r in chain.get_residues() if is_aa(r)]
                                if len(residues) > 0:
                                    sequence = ''.join([seq1(r.get_resname()) for r in residues])
                                    return sequence
                except Exception:
                    continue

    return None

def read_design_sequences(fasta_file: Path) -> List[str]:
    """Read all design sequences from a FASTA file."""
    sequences = []

    if not fasta_file.exists():
        return sequences

    with open(fasta_file, 'r') as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))

    return sequences

# =============================================================================
# POSITION MAPPING
# =============================================================================

def pdb_to_sequence_index(pdb_position: int, pdb_id: str, chain_id: str) -> Optional[int]:
    """
    Convert PDB residue number to 0-indexed sequence position.

    Returns:
        0-indexed position in the sequence, or None if not found
    """
    for subdir in ['glycoproteins', 'controls']:
        for pattern in [f"{pdb_id}_{chain_id}.pdb", f"{pdb_id}.pdb"]:
            pdb_path = PDB_ROOT / subdir / pattern
            if pdb_path.exists():
                try:
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('protein', str(pdb_path))

                    for model in structure:
                        for chain in model:
                            if chain.id == chain_id:
                                residues = [r for r in chain.get_residues() if is_aa(r)]

                                # Find position
                                for seq_idx, residue in enumerate(residues):
                                    if residue.id[1] == pdb_position:
                                        return seq_idx

                                return None
                except Exception:
                    continue

    return None

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_protein(protein_info: Dict) -> List[Dict]:
    """
    Analyze sequon retention for a single protein.

    Returns:
        List of dicts with per-design, per-sequon results
    """
    pdb_id = protein_info['pdb_id']
    chain_id = protein_info['chain_id']
    protein_name = protein_info['protein_name']
    glycosite_positions_str = protein_info['glycosite_positions']
    sequence_length = int(protein_info['sequence_length'])

    # Parse glycosite positions
    try:
        pdb_glycosites = [int(x.strip()) for x in glycosite_positions_str.split(',') if x.strip()]
    except:
        return []

    # Load wild-type sequence
    wt_sequence = load_wild_type_sequence(pdb_id, chain_id)
    if wt_sequence is None:
        print(f"  ⚠️  Could not load WT sequence for {pdb_id} chain {chain_id}")
        return []

    # Map PDB positions to sequence indices
    sequon_sites = []
    for pdb_pos in pdb_glycosites:
        seq_idx = pdb_to_sequence_index(pdb_pos, pdb_id, chain_id)
        if seq_idx is not None:
            # Verify this is actually a sequon in WT
            sequon_status = check_sequon(wt_sequence, seq_idx)
            if sequon_status in ['NXS', 'NXT']:
                sequon_sites.append({
                    'pdb_position': pdb_pos,
                    'seq_index': seq_idx,
                    'wt_sequon_type': sequon_status
                })

    if len(sequon_sites) == 0:
        print(f"  ⚠️  No valid sequons found for {pdb_id} chain {chain_id}")
        return []

    results = []

    # Analyze UNCONSTRAINED designs
    unconstrained_fasta = DESIGNS_DIR / f"{pdb_id}_{chain_id}_unconstrained.fasta"
    if unconstrained_fasta.exists():
        design_sequences = read_design_sequences(unconstrained_fasta)

        for design_idx, design_seq in enumerate(design_sequences):
            for site in sequon_sites:
                seq_idx = site['seq_index']
                pdb_pos = site['pdb_position']

                # Check if sequon is preserved AT THE ORIGINAL POSITION
                sequon_status = check_sequon(design_seq, seq_idx)

                preserved = sequon_status in ['NXS', 'NXT']

                results.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'protein_name': protein_name,
                    'condition': 'unconstrained',
                    'design_idx': design_idx,
                    'glycosite_pdb': pdb_pos,
                    'glycosite_seq_idx': seq_idx,
                    'wt_sequon_type': site['wt_sequon_type'],
                    'design_sequon_status': sequon_status,
                    'sequon_preserved': preserved,
                    'n_preserved': 1 if design_seq[seq_idx] == 'N' else 0,
                    'full_motif_preserved': 1 if preserved else 0
                })

    # Analyze FIXED designs
    fixed_fasta = DESIGNS_DIR / f"{pdb_id}_{chain_id}_fixed.fasta"
    if fixed_fasta.exists():
        design_sequences = read_design_sequences(fixed_fasta)

        for design_idx, design_seq in enumerate(design_sequences):
            for site in sequon_sites:
                seq_idx = site['seq_index']
                pdb_pos = site['pdb_position']

                # Verify N was actually fixed
                n_is_fixed = (design_seq[seq_idx] == 'N')

                # Check if full sequon is preserved
                sequon_status = check_sequon(design_seq, seq_idx)
                preserved = sequon_status in ['NXS', 'NXT']

                results.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'protein_name': protein_name,
                    'condition': 'fixed',
                    'design_idx': design_idx,
                    'glycosite_pdb': pdb_pos,
                    'glycosite_seq_idx': seq_idx,
                    'wt_sequon_type': site['wt_sequon_type'],
                    'design_sequon_status': sequon_status,
                    'sequon_preserved': preserved,
                    'n_preserved': 1 if n_is_fixed else 0,
                    'full_motif_preserved': 1 if preserved else 0
                })

    return results

def main():
    """Run the corrected sequon retention analysis."""

    print("="*80)
    print("CORRECTED SEQUON RETENTION ANALYSIS")
    print("="*80)
    print("\nThis analysis checks:")
    print("  - Unconstrained: Is the sequon preserved AT THE ORIGINAL POSITION?")
    print("  - Fixed: Is the full N-X-S/T motif retained after fixing N?")
    print("\nThis corrects the flawed analysis that counted sequons anywhere in the sequence.")

    # Load manifest
    print(f"\nLoading manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, 'r') as f:
        reader = csv.DictReader(f)
        proteins = list(reader)

    print(f"Found {len(proteins)} proteins")

    # Analyze each protein
    print(f"\n{'='*80}")
    print("ANALYZING PROTEINS")
    print(f"{'='*80}")

    all_results = []
    proteins_analyzed = 0
    proteins_skipped = 0

    for i, protein in enumerate(proteins, 1):
        pdb_id = protein['pdb_id']
        chain_id = protein['chain_id']
        print(f"\n[{i}/{len(proteins)}] {pdb_id} chain {chain_id}...")

        results = analyze_protein(protein)

        if len(results) > 0:
            all_results.extend(results)
            proteins_analyzed += 1

            # Count results by condition
            unconstrained = [r for r in results if r['condition'] == 'unconstrained']
            fixed = [r for r in results if r['condition'] == 'fixed']

            if unconstrained:
                preserved = sum(r['sequon_preserved'] for r in unconstrained)
                total = len(unconstrained)
                print(f"  Unconstrained: {preserved}/{total} sequons preserved ({100*preserved/total:.1f}%)")

            if fixed:
                preserved = sum(r['sequon_preserved'] for r in fixed)
                total = len(fixed)
                print(f"  Fixed: {preserved}/{total} sequons preserved ({100*preserved/total:.1f}%)")
        else:
            proteins_skipped += 1
            print(f"  Skipped (no design files or no valid sequons)")

    # Save detailed results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        # Save detailed results
        detailed_output = OUTPUT_DIR / "sequon_retention_corrected_detailed.csv"
        results_df.to_csv(detailed_output, index=False)
        print(f"✓ Detailed results: {detailed_output}")
        print(f"  ({len(results_df)} design×sequon combinations)")

        # Generate summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")

        for condition in ['unconstrained', 'fixed']:
            cond_data = results_df[results_df['condition'] == condition]

            if len(cond_data) > 0:
                total = len(cond_data)
                n_preserved = cond_data['n_preserved'].sum()
                full_preserved = cond_data['full_motif_preserved'].sum()

                print(f"\n{condition.upper()}:")
                print(f"  Total sequon sites tested: {total}")
                print(f"  N preserved: {n_preserved} ({100*n_preserved/total:.1f}%)")
                print(f"  Full sequon preserved: {full_preserved} ({100*full_preserved/total:.1f}%)")

                # Per-protein breakdown
                protein_summary = cond_data.groupby('pdb_id').agg({
                    'full_motif_preserved': ['sum', 'count']
                }).reset_index()
                protein_summary.columns = ['pdb_id', 'preserved', 'total']
                protein_summary['retention_rate'] = protein_summary['preserved'] / protein_summary['total']

                print(f"\n  Per-protein retention rates:")
                for _, row in protein_summary.iterrows():
                    print(f"    {row['pdb_id']:6s}: {row['preserved']:3.0f}/{row['total']:3.0f} ({100*row['retention_rate']:5.1f}%)")

        # Save summary
        summary_output = OUTPUT_DIR / "sequon_retention_corrected_summary.csv"
        summary_data = []

        for condition in ['unconstrained', 'fixed']:
            cond_data = results_df[results_df['condition'] == condition]

            if len(cond_data) > 0:
                protein_summary = cond_data.groupby(['pdb_id', 'chain_id', 'protein_name']).agg({
                    'full_motif_preserved': ['sum', 'count'],
                    'n_preserved': 'sum'
                }).reset_index()

                protein_summary.columns = ['pdb_id', 'chain_id', 'protein_name',
                                          'sequons_preserved', 'total_sequons',
                                          'n_preserved']

                protein_summary['condition'] = condition
                protein_summary['retention_rate'] = protein_summary['sequons_preserved'] / protein_summary['total_sequons']

                summary_data.append(protein_summary)

        if summary_data:
            summary_df = pd.concat(summary_data, ignore_index=True)
            summary_df.to_csv(summary_output, index=False)
            print(f"\n✓ Summary: {summary_output}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Proteins analyzed: {proteins_analyzed}")
    print(f"Proteins skipped: {proteins_skipped}")

    if len(all_results) > 0:
        print(f"\nNext step:")
        print(f"  Create visualizations with the corrected data")
        print(f"  Run: python3 scripts/08_create_visualizations.py")

if __name__ == '__main__':
    main()
