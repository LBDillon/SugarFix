#!/usr/bin/env python3
"""
Scan All Available PDB Files for N-X-S/T Sequons

This script:
1. Scans all PDB files in the glycoproteins directory
2. Identifies ALL N-X-S/T sequons in each structure
3. Filters for monomeric proteins (single chain, reasonable size)
4. Generates a new manifest with validated sequons

This will help us expand the dataset beyond the original manifest.

Author: Claude Code
Date: 2026-01-20
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

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

PDB_DIR = Path("data/glyco_benchmark/raw/glycoproteins")
OUTPUT_DIR = Path("data/glyco_benchmark/manifests")

# Filtering criteria for good candidates
MIN_LENGTH = 50   # Minimum protein length
MAX_LENGTH = 600  # Maximum protein length (ProteinMPNN works best < 500)
MIN_SEQUONS = 1   # Minimum number of sequons required

# =============================================================================
# SEQUON DETECTION
# =============================================================================

def find_all_sequons(sequence: str) -> List[Dict]:
    """
    Find all N-X-S/T sequons in a sequence.

    Returns:
        List of dicts with sequon information
    """
    sequons = []

    for i in range(len(sequence) - 2):
        aa_0 = sequence[i]
        aa_1 = sequence[i + 1]
        aa_2 = sequence[i + 2]

        # Check for N-X-S/T pattern
        if aa_0 == 'N' and aa_1 != 'P' and aa_2 in ['S', 'T']:
            sequon_type = 'NXS' if aa_2 == 'S' else 'NXT'
            context = get_context(sequence, i)

            sequons.append({
                'position_0indexed': i,
                'position_pdb': i + 1,  # Will be updated with actual PDB numbering
                'sequon_type': sequon_type,
                'triplet': f"{aa_0}{aa_1}{aa_2}",
                'context': context
            })

    return sequons

def get_context(sequence: str, position: int, window: int = 5) -> str:
    """Get sequence context around a position."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 3)  # +3 to include the sequon
    context = sequence[start:end]

    # Mark the sequon position
    relative_pos = position - start
    if 0 <= relative_pos < len(context):
        context = context[:relative_pos] + '[' + context[relative_pos] + ']' + context[relative_pos+1:]

    return context

# =============================================================================
# PDB PARSING
# =============================================================================

def parse_pdb_file(pdb_path: Path) -> Optional[Dict]:
    """
    Parse PDB file and extract sequence information.

    Returns:
        Dict with protein info or None on error
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))

        # Get all chains
        chains_data = []

        for model in structure:
            for chain in model:
                # Get amino acid residues only
                residues = [r for r in chain.get_residues() if is_aa(r)]

                if len(residues) == 0:
                    continue

                # Extract sequence
                try:
                    sequence = ''.join([seq1(r.get_resname()) for r in residues])
                except Exception as e:
                    # Handle non-standard amino acids
                    continue

                # Get PDB residue numbers for mapping
                residue_numbers = [r.id[1] for r in residues]

                chains_data.append({
                    'chain_id': chain.id,
                    'sequence': sequence,
                    'length': len(sequence),
                    'residue_numbers': residue_numbers
                })

        return {
            'pdb_path': str(pdb_path),
            'pdb_id': pdb_path.stem.split('_')[0],
            'chains': chains_data
        }

    except Exception as e:
        print(f"  ERROR parsing {pdb_path.name}: {e}")
        return None

# =============================================================================
# MAIN SCANNING LOGIC
# =============================================================================

def scan_all_pdbs():
    """
    Scan all PDB files in the glycoproteins directory.
    """
    print("="*80)
    print("SCANNING ALL PDB FILES FOR N-X-S/T SEQUONS")
    print("="*80)
    print(f"\nPDB directory: {PDB_DIR}")
    print(f"Filtering criteria:")
    print(f"  - Length: {MIN_LENGTH}-{MAX_LENGTH} residues")
    print(f"  - Minimum sequons: {MIN_SEQUONS}")
    print(f"  - Monomeric chains only (ignoring multi-chain complexes)")

    # Get all PDB files
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    print(f"\nFound {len(pdb_files)} PDB files")

    results = []
    monomeric_candidates = []

    print("\n" + "="*80)
    print("SCANNING RESULTS")
    print("="*80)

    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_data = parse_pdb_file(pdb_file)

        if pdb_data is None:
            continue

        pdb_id = pdb_data['pdb_id']

        # Process each chain
        for chain_data in pdb_data['chains']:
            chain_id = chain_data['chain_id']
            sequence = chain_data['sequence']
            length = chain_data['length']
            residue_numbers = chain_data['residue_numbers']

            # Apply filters
            if length < MIN_LENGTH or length > MAX_LENGTH:
                continue

            # Find sequons
            sequons = find_all_sequons(sequence)

            if len(sequons) < MIN_SEQUONS:
                continue

            # Update sequon positions with actual PDB numbering
            for sequon in sequons:
                seq_idx = sequon['position_0indexed']
                if seq_idx < len(residue_numbers):
                    sequon['position_pdb'] = residue_numbers[seq_idx]

            # This is a good candidate
            candidate = {
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'length': length,
                'n_sequons': len(sequons),
                'sequons': sequons,
                'sequence': sequence
            }

            monomeric_candidates.append(candidate)

            # Print summary
            sequon_positions = [s['position_pdb'] for s in sequons]
            print(f"\n✓ {pdb_id} chain {chain_id} ({length} aa)")
            print(f"  Sequons: {len(sequons)} sites at PDB positions {sequon_positions}")
            for j, s in enumerate(sequons, 1):
                print(f"    {j}. Position {s['position_pdb']:4d} ({s['sequon_type']}): {s['context']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal PDB files scanned: {len(pdb_files)}")
    print(f"Monomeric proteins with sequons: {len(monomeric_candidates)}")

    if len(monomeric_candidates) > 0:
        total_sequons = sum(c['n_sequons'] for c in monomeric_candidates)
        print(f"Total sequon sites: {total_sequons}")

        # Breakdown by number of sequons
        sequon_counts = {}
        for c in monomeric_candidates:
            n = c['n_sequons']
            sequon_counts[n] = sequon_counts.get(n, 0) + 1

        print(f"\nBreakdown by sequon count:")
        for n in sorted(sequon_counts.keys()):
            print(f"  {n} sequon(s): {sequon_counts[n]} protein(s)")

    # Save results
    save_expanded_manifest(monomeric_candidates)

    return monomeric_candidates

def save_expanded_manifest(candidates: List[Dict]):
    """
    Save expanded manifest with all validated monomeric glycoproteins.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create CSV manifest
    manifest_rows = []
    detailed_rows = []

    for candidate in candidates:
        # One row per protein in the manifest
        sequon_positions = [str(s['position_pdb']) for s in candidate['sequons']]

        manifest_rows.append({
            'pdb_id': candidate['pdb_id'],
            'chain_id': candidate['chain_id'],
            'protein_name': f"{candidate['pdb_id']} (auto-detected)",
            'glycosite_positions': ','.join(sequon_positions),
            'n_glycosites': candidate['n_sequons'],
            'sequence_length': candidate['length'],
            'source': 'automated_scan'
        })

        # Detailed rows (one per sequon)
        for sequon in candidate['sequons']:
            detailed_rows.append({
                'pdb_id': candidate['pdb_id'],
                'chain_id': candidate['chain_id'],
                'pdb_position': sequon['position_pdb'],
                'sequence_index': sequon['position_0indexed'],
                'sequon_type': sequon['sequon_type'],
                'triplet': sequon['triplet'],
                'context': sequon['context']
            })

    # Save manifest
    manifest_file = OUTPUT_DIR / "expanded_manifest_validated.csv"
    with open(manifest_file, 'w', newline='') as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\n{'='*80}")
    print("SAVED OUTPUT FILES")
    print(f"{'='*80}")
    print(f"✓ Manifest: {manifest_file}")
    print(f"  ({len(manifest_rows)} proteins)")

    # Save detailed sequon info
    detailed_file = OUTPUT_DIR / "expanded_manifest_sequons_detailed.csv"
    with open(detailed_file, 'w', newline='') as f:
        if detailed_rows:
            writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
            writer.writeheader()
            writer.writerows(detailed_rows)

    print(f"✓ Detailed sequons: {detailed_file}")
    print(f"  ({len(detailed_rows)} sequon sites)")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("1. Review the expanded manifest")
    print("2. Run ProteinMPNN designs using scripts/03_mpnn_design_experiment.py")
    print("3. Analyze sequon retention with the corrected analysis script")

if __name__ == '__main__':
    scan_all_pdbs()
