#!/usr/bin/env python3
"""
Manifest Validation Script: Verify Glycosylation Sites Have N-X-S/T Sequons

This script validates the glycoprotein manifest by:
1. Loading each PDB structure
2. Extracting the amino acid sequence
3. Checking if listed glycosite positions actually contain N-X-S/T motifs
4. Generating a corrected manifest with only valid sequons
5. Producing a detailed diagnostic report

Author: Claude Code
Date: 2026-01-20
"""

import os
import csv
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser, PPBuilder, is_aa
    from Bio.Seq import Seq
except ImportError:
    print("ERROR: BioPython not installed")
    print("Install with: pip install biopython")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_PATH = "data/glyco_benchmark/manifests/benchmark_manifest_simple.csv"
PDB_ROOT = "data/glyco_benchmark/raw"
OUTPUT_DIR = "data/glyco_benchmark/manifests"

# =============================================================================
# SEQUON VALIDATION FUNCTIONS
# =============================================================================

def check_sequon(sequence: str, position: int) -> Optional[str]:
    """
    Check if N-X-S/T sequon exists at position (0-indexed in sequence).

    N-linked glycosylation requires:
    - Position i: Asparagine (N)
    - Position i+1: Any amino acid EXCEPT Proline (X ≠ P)
    - Position i+2: Serine (S) or Threonine (T)

    Args:
        sequence: Protein sequence string
        position: 0-indexed position to check

    Returns:
        'NXS': Full sequon with Serine
        'NXT': Full sequon with Threonine
        'N_only': Has N but sequon is broken
        'no_N': No asparagine at position
        None: Position out of bounds
    """
    # Check bounds
    if position < 0 or position + 2 >= len(sequence):
        return None

    aa_0 = sequence[position]      # Should be N
    aa_1 = sequence[position + 1]  # Should be X (anything except P)
    aa_2 = sequence[position + 2]  # Should be S or T

    # Check asparagine
    if aa_0 != 'N':
        return 'no_N'

    # Check middle position (not Proline)
    if aa_1 == 'P':
        return 'N_only'  # N-P-S/T is not glycosylated

    # Check final position
    if aa_2 == 'S':
        return 'NXS'
    elif aa_2 == 'T':
        return 'NXT'
    else:
        return 'N_only'  # N-X-[not S/T]

def get_sequence_context(sequence: str, position: int, window: int = 5) -> str:
    """
    Get sequence context around a position.

    Args:
        sequence: Protein sequence
        position: 0-indexed position
        window: Number of residues on each side

    Returns:
        String like "...RKQNLSI..." with the position in the middle
    """
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)

    context = sequence[start:end]

    # Mark the position of interest
    relative_pos = position - start
    if 0 <= relative_pos < len(context):
        context = context[:relative_pos] + '[' + context[relative_pos] + ']' + context[relative_pos+1:]

    return context

# =============================================================================
# PDB PARSING FUNCTIONS
# =============================================================================

def extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> Optional[Tuple[str, List[int]]]:
    """
    Extract amino acid sequence from PDB file.

    Returns:
        (sequence_string, residue_numbers_list) or None on error

    Note: Returns actual PDB residue numbers for mapping
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        # Find the specified chain
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    # Get all amino acid residues
                    residues = [r for r in chain.get_residues() if is_aa(r)]

                    if len(residues) == 0:
                        return None

                    # Extract sequence and residue numbers
                    sequence = ''.join([r.get_resname() for r in residues])
                    # Convert 3-letter to 1-letter codes
                    from Bio.SeqUtils import seq1
                    sequence = seq1(sequence)

                    # Get PDB residue numbers
                    residue_numbers = [r.id[1] for r in residues]

                    return sequence, residue_numbers

        return None
    except Exception as e:
        print(f"      ERROR parsing PDB: {e}")
        return None

def find_pdb_file(pdb_id: str, chain_id: str, pdb_root: str) -> Optional[str]:
    """
    Find PDB file in the directory structure.
    """
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

# =============================================================================
# MANIFEST PARSING
# =============================================================================

def parse_glycosites(glycosite_string: str) -> List[int]:
    """Parse glycosite positions from manifest string (e.g., '64,172,320')"""
    if not glycosite_string or glycosite_string == '':
        return []
    try:
        # Remove quotes if present
        glycosite_string = glycosite_string.strip('"\'')
        return [int(x.strip()) for x in glycosite_string.split(',') if x.strip()]
    except ValueError as e:
        print(f"      ERROR parsing glycosites '{glycosite_string}': {e}")
        return []

def load_manifest(manifest_path: str, protein_class: str = 'glycoprotein') -> List[Dict]:
    """Load manifest and filter by protein class"""
    proteins = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['protein_class'] == protein_class:
                proteins.append(row)
    return proteins

# =============================================================================
# VALIDATION LOGIC
# =============================================================================

def validate_protein(protein: Dict, pdb_root: str) -> Dict:
    """
    Validate a single protein's glycosites.

    Returns:
        Dictionary with validation results
    """
    pdb_id = protein['pdb_id']
    chain_id = protein['chain_id']
    protein_name = protein['protein_name']
    glycosite_str = protein.get('glycosite_positions', '')

    result = {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'protein_name': protein_name,
        'listed_glycosites': glycosite_str,
        'n_listed': 0,
        'pdb_file_found': False,
        'sequence_extracted': False,
        'sequence_length': 0,
        'valid_sequons': [],
        'invalid_sites': [],
        'status': 'unknown',
        'notes': []
    }

    # Parse glycosites
    glycosites_pdb = parse_glycosites(glycosite_str)
    result['n_listed'] = len(glycosites_pdb)

    if len(glycosites_pdb) == 0:
        result['status'] = 'no_glycosites_listed'
        return result

    # Find PDB file
    pdb_path = find_pdb_file(pdb_id, chain_id, pdb_root)
    if not pdb_path:
        result['status'] = 'pdb_not_found'
        result['notes'].append(f"PDB file not found for {pdb_id} chain {chain_id}")
        return result

    result['pdb_file_found'] = True

    # Extract sequence
    seq_data = extract_sequence_from_pdb(pdb_path, chain_id)
    if seq_data is None:
        result['status'] = 'sequence_extraction_failed'
        result['notes'].append(f"Could not extract sequence from {pdb_path}")
        return result

    sequence, residue_numbers = seq_data
    result['sequence_extracted'] = True
    result['sequence_length'] = len(sequence)

    # Validate each glycosite
    for pdb_pos in glycosites_pdb:
        # Find this PDB residue number in the sequence
        try:
            seq_idx = residue_numbers.index(pdb_pos)
        except ValueError:
            result['invalid_sites'].append({
                'pdb_position': pdb_pos,
                'reason': 'position_not_in_structure',
                'sequon_status': None,
                'context': None
            })
            continue

        # Check sequon at this position
        sequon_status = check_sequon(sequence, seq_idx)
        context = get_sequence_context(sequence, seq_idx, window=5)

        site_info = {
            'pdb_position': pdb_pos,
            'sequence_index': seq_idx,
            'sequon_status': sequon_status,
            'context': context
        }

        if sequon_status in ['NXS', 'NXT']:
            result['valid_sequons'].append(site_info)
        else:
            site_info['reason'] = sequon_status
            result['invalid_sites'].append(site_info)

    # Determine overall status
    n_valid = len(result['valid_sequons'])
    n_invalid = len(result['invalid_sites'])

    if n_valid == len(glycosites_pdb):
        result['status'] = 'all_valid'
    elif n_valid > 0:
        result['status'] = 'partial_valid'
    else:
        result['status'] = 'no_valid_sequons'

    return result

# =============================================================================
# REPORTING
# =============================================================================

def print_validation_report(results: List[Dict]):
    """Print detailed validation report to console"""

    print("\n" + "="*80)
    print("GLYCOPROTEIN MANIFEST VALIDATION REPORT")
    print("="*80)

    # Summary statistics
    total = len(results)
    all_valid = sum(1 for r in results if r['status'] == 'all_valid')
    partial_valid = sum(1 for r in results if r['status'] == 'partial_valid')
    no_valid = sum(1 for r in results if r['status'] == 'no_valid_sequons')
    errors = sum(1 for r in results if r['status'] in ['pdb_not_found', 'sequence_extraction_failed'])

    print(f"\nSUMMARY:")
    print(f"  Total proteins: {total}")
    print(f"  All glycosites valid: {all_valid}")
    print(f"  Partial glycosites valid: {partial_valid}")
    print(f"  No valid glycosites: {no_valid}")
    print(f"  Errors (PDB not found / extraction failed): {errors}")

    # Detailed results
    print("\n" + "="*80)
    print("DETAILED VALIDATION RESULTS")
    print("="*80)

    # Group by status
    status_groups = defaultdict(list)
    for r in results:
        status_groups[r['status']].append(r)

    # Print all_valid proteins
    if 'all_valid' in status_groups:
        print(f"\n{'─'*80}")
        print(f"✓ PROTEINS WITH ALL VALID SEQUONS ({len(status_groups['all_valid'])})")
        print(f"{'─'*80}")

        for r in status_groups['all_valid']:
            print(f"\n{r['pdb_id']} (chain {r['chain_id']}) - {r['protein_name']}")
            print(f"  Sequence length: {r['sequence_length']} aa")
            print(f"  Valid sequons: {len(r['valid_sequons'])}/{r['n_listed']}")

            for site in r['valid_sequons']:
                print(f"    • PDB position {site['pdb_position']:4d} (seq idx {site['sequence_index']:3d}): "
                      f"{site['sequon_status']:3s} - {site['context']}")

    # Print partial_valid proteins
    if 'partial_valid' in status_groups:
        print(f"\n{'─'*80}")
        print(f"⚠ PROTEINS WITH PARTIAL VALID SEQUONS ({len(status_groups['partial_valid'])})")
        print(f"{'─'*80}")

        for r in status_groups['partial_valid']:
            print(f"\n{r['pdb_id']} (chain {r['chain_id']}) - {r['protein_name']}")
            print(f"  Sequence length: {r['sequence_length']} aa")
            print(f"  Valid sequons: {len(r['valid_sequons'])}/{r['n_listed']}")

            if r['valid_sequons']:
                print(f"  ✓ Valid sites:")
                for site in r['valid_sequons']:
                    print(f"    • PDB position {site['pdb_position']:4d} (seq idx {site['sequence_index']:3d}): "
                          f"{site['sequon_status']:3s} - {site['context']}")

            if r['invalid_sites']:
                print(f"  ✗ Invalid sites:")
                for site in r['invalid_sites']:
                    reason = site.get('reason', 'unknown')
                    pos = site['pdb_position']
                    context = site.get('context', 'N/A')
                    print(f"    • PDB position {pos:4d}: {reason} - {context}")

    # Print no_valid proteins
    if 'no_valid_sequons' in status_groups:
        print(f"\n{'─'*80}")
        print(f"✗ PROTEINS WITH NO VALID SEQUONS ({len(status_groups['no_valid_sequons'])})")
        print(f"{'─'*80}")

        for r in status_groups['no_valid_sequons']:
            print(f"\n{r['pdb_id']} (chain {r['chain_id']}) - {r['protein_name']}")
            print(f"  Sequence length: {r['sequence_length']} aa")
            print(f"  Listed glycosites: {r['n_listed']}")
            print(f"  Valid sequons: 0")

            if r['invalid_sites']:
                print(f"  Reasons:")
                for site in r['invalid_sites']:
                    reason = site.get('reason', 'unknown')
                    pos = site['pdb_position']
                    context = site.get('context', 'N/A')
                    print(f"    • PDB position {pos:4d}: {reason} - {context}")

    # Print errors
    for status in ['pdb_not_found', 'sequence_extraction_failed']:
        if status in status_groups:
            print(f"\n{'─'*80}")
            print(f"⚠ {status.upper().replace('_', ' ')} ({len(status_groups[status])})")
            print(f"{'─'*80}")

            for r in status_groups[status]:
                print(f"\n{r['pdb_id']} (chain {r['chain_id']}) - {r['protein_name']}")
                if r['notes']:
                    for note in r['notes']:
                        print(f"  • {note}")

def save_corrected_manifest(results: List[Dict], output_path: str):
    """
    Save a corrected manifest with only valid glycosites.
    """
    corrected = []

    for r in results:
        if len(r['valid_sequons']) > 0:
            # Create corrected entry
            valid_positions = [str(s['pdb_position']) for s in r['valid_sequons']]

            corrected.append({
                'pdb_id': r['pdb_id'],
                'chain_id': r['chain_id'],
                'protein_name': r['protein_name'],
                'glycosite_positions': ','.join(valid_positions),
                'n_glycosites': len(valid_positions),
                'sequence_length': r['sequence_length'],
                'validation_status': r['status']
            })

    # Write to CSV
    if corrected:
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['pdb_id', 'chain_id', 'protein_name', 'glycosite_positions',
                         'n_glycosites', 'sequence_length', 'validation_status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corrected)

        print(f"\n✓ Saved corrected manifest to: {output_path}")
        print(f"  Contains {len(corrected)} proteins with valid sequons")
    else:
        print(f"\n⚠ No proteins with valid sequons found - no corrected manifest saved")

def save_detailed_csv(results: List[Dict], output_path: str):
    """
    Save detailed validation results to CSV for further analysis.
    """
    rows = []

    for r in results:
        # One row per glycosite
        for site in r['valid_sequons']:
            rows.append({
                'pdb_id': r['pdb_id'],
                'chain_id': r['chain_id'],
                'protein_name': r['protein_name'],
                'pdb_position': site['pdb_position'],
                'sequence_index': site['sequence_index'],
                'sequon_status': site['sequon_status'],
                'context': site['context'],
                'valid': True
            })

        for site in r['invalid_sites']:
            rows.append({
                'pdb_id': r['pdb_id'],
                'chain_id': r['chain_id'],
                'protein_name': r['protein_name'],
                'pdb_position': site['pdb_position'],
                'sequence_index': site.get('sequence_index', None),
                'sequon_status': site.get('sequon_status', site.get('reason', 'unknown')),
                'context': site.get('context', ''),
                'valid': False
            })

    if rows:
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['pdb_id', 'chain_id', 'protein_name', 'pdb_position',
                         'sequence_index', 'sequon_status', 'context', 'valid']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Saved detailed validation results to: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("GLYCOPROTEIN MANIFEST VALIDATOR")
    print("="*80)
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"PDB root: {PDB_ROOT}")

    # Load manifest
    print(f"\nLoading glycoproteins from manifest...")
    proteins = load_manifest(MANIFEST_PATH, protein_class='glycoprotein')
    print(f"Found {len(proteins)} glycoproteins")

    # Validate each protein
    print(f"\nValidating glycosylation sites...")
    results = []

    for i, protein in enumerate(proteins, 1):
        pdb_id = protein['pdb_id']
        print(f"  [{i}/{len(proteins)}] {pdb_id}...", end=' ', flush=True)

        result = validate_protein(protein, PDB_ROOT)
        results.append(result)

        # Print quick status
        if result['status'] == 'all_valid':
            print(f"✓ {len(result['valid_sequons'])} valid sequons")
        elif result['status'] == 'partial_valid':
            print(f"⚠ {len(result['valid_sequons'])}/{result['n_listed']} valid")
        elif result['status'] == 'no_valid_sequons':
            print(f"✗ No valid sequons")
        else:
            print(f"⚠ {result['status']}")

    # Print detailed report
    print_validation_report(results)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    corrected_manifest_path = Path(OUTPUT_DIR) / "benchmark_manifest_validated.csv"
    detailed_results_path = Path(OUTPUT_DIR) / "glycosite_validation_detailed.csv"

    print(f"\n{'='*80}")
    print("SAVING OUTPUTS")
    print(f"{'='*80}")

    save_corrected_manifest(results, corrected_manifest_path)
    save_detailed_csv(results, detailed_results_path)

    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")

    # Final recommendations
    print("\nRECOMMENDATIONS:")

    valid_proteins = [r for r in results if len(r['valid_sequons']) > 0]
    if len(valid_proteins) == 0:
        print("  ⚠ NO PROTEINS WITH VALID SEQUONS FOUND")
        print("  → Check if glycosite positions are in the correct numbering scheme")
        print("  → Verify PDB files contain the expected sequences")
    elif len(valid_proteins) < len(proteins) / 2:
        print(f"  ⚠ Only {len(valid_proteins)}/{len(proteins)} proteins have valid sequons")
        print("  → Use the corrected manifest for experiments")
        print("  → Investigate why many glycosites don't have sequons")
    else:
        print(f"  ✓ {len(valid_proteins)}/{len(proteins)} proteins have valid sequons")
        print("  → Use the corrected manifest for experiments")

    print(f"\nNext steps:")
    print(f"  1. Review the validation report above")
    print(f"  2. Check {detailed_results_path} for full details")
    print(f"  3. Use {corrected_manifest_path} for ProteinMPNN experiments")

if __name__ == '__main__':
    main()
