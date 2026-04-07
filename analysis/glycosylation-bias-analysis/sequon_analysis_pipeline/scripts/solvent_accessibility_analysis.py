#!/usr/bin/env python3
"""
Solvent Accessibility Analysis for Sequon Retention Study

This analysis addresses the main limitation: whether lower sequon-N retention
is due to surface exposure (structural confound) or sequon-specific effects.

Analysis:
1. Calculate RSA (relative solvent accessibility) for all N positions
2. Bin into categories: buried (<20%), intermediate (20-50%), exposed (>50%)
3. Compare sequon-N vs non-sequon-N retention within each bin
4. Analyze functional retention and de novo sequons by RSA bin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from Bio.PDB import PDBParser, SASA
import ast
import warnings
warnings.filterwarnings('ignore')


# Maximum ASA values for amino acids (Tien et al., 2013 - empirical)
MAX_ASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174
}

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def calculate_rsa(pdb_path):
    """Calculate RSA for all residues using BioPython's SASA."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))

    # Calculate SASA
    sr = SASA.ShrakeRupley()
    sr.compute(structure, level="R")

    results = []
    for model in structure:
        for chain in model:
            chain_seq = []
            for residue in chain:
                if residue.id[0] == ' ':  # Skip heteroatoms
                    res_name = residue.resname
                    res_num = residue.id[1]
                    one_letter = AA_MAP.get(res_name, 'X')
                    chain_seq.append((res_num, one_letter))

                    try:
                        asa = residue.sasa
                    except AttributeError:
                        asa = 0

                    if one_letter in MAX_ASA:
                        max_asa = MAX_ASA[one_letter]
                        rsa = min(100, 100 * asa / max_asa) if max_asa > 0 else 0
                    else:
                        rsa = None

                    results.append({
                        'chain': chain.id,
                        'res_num': res_num,
                        'residue': one_letter,
                        'asa': asa,
                        'rsa': rsa
                    })

    return pd.DataFrame(results)


def get_sequence_from_pdb(pdb_path):
    """Extract sequence with position mapping from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))

    sequences = {}
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.id[0] == ' ':
                    res_num = residue.id[1]
                    one_letter = AA_MAP.get(residue.resname, 'X')
                    seq.append((res_num, one_letter))
            sequences[chain.id] = seq

    return sequences


def bin_rsa(rsa):
    """Bin RSA into categories."""
    if rsa is None or np.isnan(rsa):
        return 'unknown'
    elif rsa < 20:
        return 'buried'
    elif rsa < 50:
        return 'intermediate'
    else:
        return 'exposed'


def is_valid_sequon(triplet):
    """Check if a triplet is a valid N-X-S/T sequon (X ≠ P)."""
    if len(triplet) != 3:
        return False
    return (triplet[0] == 'N' and
            triplet[1] != 'P' and
            triplet[2] in ['S', 'T'])


def find_sequons_in_sequence(seq_list):
    """Find sequon positions in a sequence (list of (res_num, aa) tuples)."""
    sequons = []
    for i in range(len(seq_list) - 2):
        triplet = seq_list[i][1] + seq_list[i+1][1] + seq_list[i+2][1]
        if is_valid_sequon(triplet):
            sequons.append({
                'position': seq_list[i][0],
                'sequon': triplet,
                'seq_idx': i
            })
    return sequons


def analyze_protein(pdb_path, pdb_id, design_fasta=None):
    """Analyze a single protein for RSA and sequon retention."""

    # Calculate RSA
    df_rsa = calculate_rsa(pdb_path)

    # Get sequences and find sequons
    sequences = get_sequence_from_pdb(pdb_path)

    all_sequons = []
    for chain_id, seq in sequences.items():
        sequons = find_sequons_in_sequence(seq)
        for s in sequons:
            s['chain'] = chain_id
            all_sequons.append(s)

    # Mark which N residues are sequon-N
    sequon_positions = set((s['chain'], s['position']) for s in all_sequons)

    df_rsa['is_sequon_n'] = df_rsa.apply(
        lambda row: (row['chain'], row['res_num']) in sequon_positions and row['residue'] == 'N',
        axis=1
    )

    # Filter to N residues only
    df_n = df_rsa[df_rsa['residue'] == 'N'].copy()
    df_n['rsa_bin'] = df_n['rsa'].apply(bin_rsa)
    df_n['pdb_id'] = pdb_id

    # If we have designs, analyze retention
    retention_data = None
    if design_fasta and Path(design_fasta).exists():
        retention_data = analyze_designs(design_fasta, sequences, df_n, all_sequons)

    return df_n, all_sequons, retention_data


def analyze_designs(fasta_path, wt_sequences, df_n, sequons):
    """Analyze design retention by position, including RSA data."""

    with open(fasta_path) as f:
        content = f.read()

    entries = content.strip().split('>')[1:]
    if len(entries) < 2:
        return None

    # Parse wild-type (first entry)
    wt_entry = entries[0]
    wt_lines = wt_entry.split('\n')
    wt_seq_combined = ''.join(wt_lines[1:])

    # Handle multi-chain (separated by /)
    wt_chains = wt_seq_combined.split('/')

    # Parse designs
    design_seqs = []
    for entry in entries[1:]:
        lines = entry.strip().split('\n')
        seq = ''.join(lines[1:])
        design_seqs.append(seq.split('/'))

    # Build chain mapping from RSA data
    # df_n has (chain, res_num) - we need to map chain letters to indices
    chain_ids = sorted(df_n['chain'].unique())
    chain_to_idx = {ch: i for i, ch in enumerate(chain_ids)}

    # Build lookup for RSA by (chain_idx, sequential N position within chain)
    rsa_lookup = {}
    for chain_id in chain_ids:
        chain_data = df_n[df_n['chain'] == chain_id].sort_values('res_num')
        chain_idx = chain_to_idx[chain_id]
        # Create mapping from res_num to RSA data
        for _, row in chain_data.iterrows():
            rsa_lookup[(chain_id, row['res_num'])] = {
                'rsa': row['rsa'],
                'rsa_bin': row['rsa_bin']
            }

    # Analyze N retention by position
    results = []

    for chain_idx, wt_chain in enumerate(wt_chains):
        # Get corresponding chain_id
        chain_id = chain_ids[chain_idx] if chain_idx < len(chain_ids) else None

        # Track which N position we're at (to match with PDB numbering)
        # We need to find which res_nums correspond to N positions
        if chain_id:
            chain_n_data = df_n[df_n['chain'] == chain_id].sort_values('res_num')
            n_res_nums = list(chain_n_data['res_num'].values)
        else:
            n_res_nums = []

        n_counter = 0  # Counter for N positions in this chain

        for pos_in_chain, aa in enumerate(wt_chain):
            if aa == 'N':
                retained_count = 0
                total_count = 0

                for design in design_seqs:
                    if chain_idx < len(design) and pos_in_chain < len(design[chain_idx]):
                        total_count += 1
                        if design[chain_idx][pos_in_chain] == 'N':
                            retained_count += 1

                # Check if this is a sequon position
                is_sequon = False
                if pos_in_chain + 2 < len(wt_chain):
                    triplet = wt_chain[pos_in_chain:pos_in_chain+3]
                    is_sequon = is_valid_sequon(triplet)

                # Exact sequon retention (full triplet matches WT exactly)
                # Functional retention (any valid sequon at this position)
                exact_sequon_count = 0
                functional_count = 0
                wt_triplet = wt_chain[pos_in_chain:pos_in_chain+3] if pos_in_chain + 2 < len(wt_chain) else None

                if is_sequon and wt_triplet:
                    for design in design_seqs:
                        if chain_idx < len(design) and pos_in_chain + 2 < len(design[chain_idx]):
                            design_triplet = design[chain_idx][pos_in_chain:pos_in_chain+3]
                            # Exact match - full triplet identical to WT
                            if design_triplet == wt_triplet:
                                exact_sequon_count += 1
                            # Functional - any valid sequon (includes exact + other valid)
                            if is_valid_sequon(design_triplet):
                                functional_count += 1

                # Get RSA for this N position
                rsa_val = None
                rsa_bin_val = 'unknown'
                if n_counter < len(n_res_nums) and chain_id:
                    res_num = n_res_nums[n_counter]
                    rsa_info = rsa_lookup.get((chain_id, res_num), {})
                    rsa_val = rsa_info.get('rsa')
                    rsa_bin_val = rsa_info.get('rsa_bin', 'unknown')

                results.append({
                    'chain_idx': chain_idx,
                    'chain_id': chain_id,
                    'pos_in_chain': pos_in_chain,
                    'is_sequon': is_sequon,
                    'wt_triplet': wt_triplet,
                    'n_retained': retained_count,
                    'n_total': total_count,
                    'n_retention_pct': 100 * retained_count / total_count if total_count > 0 else 0,
                    'exact_sequon_retained': exact_sequon_count,
                    'exact_sequon_pct': 100 * exact_sequon_count / total_count if total_count > 0 and is_sequon else None,
                    'functional_retained': functional_count,
                    'functional_pct': 100 * functional_count / total_count if total_count > 0 and is_sequon else None,
                    'rsa': rsa_val,
                    'rsa_bin': rsa_bin_val
                })

                n_counter += 1

    return pd.DataFrame(results)


def analyze_main_dataset(base_dir):
    """Analyze all glycosylated proteins."""

    print("="*70)
    print("MAIN DATASET ANALYSIS")
    print("="*70)

    # Get list of successful proteins
    protein_file = base_dir / 'corrected_sequon_analysis.csv'
    df_protein = pd.read_csv(protein_file)
    successful_proteins = list(df_protein['pdb_id'].values)

    print(f"Analyzing {len(successful_proteins)} proteins")

    pdb_dir = base_dir / 'PDBs_gly'
    results_dir = base_dir / 'results_full_gly'

    all_n_data = []
    all_retention_data = []

    for pdb_id in successful_proteins:
        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            print(f"  WARNING: PDB not found for {pdb_id}")
            continue

        # Find design file
        design_dir = results_dir / pdb_id / 'designs' / 'unconstrained' / 'seqs'
        fasta_files = list(design_dir.glob('*.fa')) if design_dir.exists() else []
        design_fasta = fasta_files[0] if fasta_files else None

        try:
            df_n, sequons, retention = analyze_protein(pdb_file, pdb_id, design_fasta)
            all_n_data.append(df_n)

            if retention is not None:
                retention['pdb_id'] = pdb_id
                all_retention_data.append(retention)

            n_sequon = df_n['is_sequon_n'].sum()
            n_total = len(df_n)
            print(f"  {pdb_id}: {n_total} N residues, {n_sequon} sequon-N, {len(sequons)} sequons")

        except Exception as e:
            print(f"  ERROR with {pdb_id}: {e}")

    # Combine data
    df_all_n = pd.concat(all_n_data, ignore_index=True)
    df_all_retention = pd.concat(all_retention_data, ignore_index=True) if all_retention_data else None

    return df_all_n, df_all_retention


def analyze_ha(base_dir):
    """Analyze HA case study."""

    print("\n" + "="*70)
    print("HA CASE STUDY ANALYSIS")
    print("="*70)

    pdb_file = base_dir / 'results_HA_case_study' / '1RUZ' / 'structure' / '1RUZ_protein.pdb'
    design_fasta = base_dir / 'results_HA_case_study' / '1RUZ' / 'designs' / 'unconstrained' / 'seqs' / '1RUZ_protein.fa'

    if not pdb_file.exists():
        print(f"ERROR: PDB not found")
        return None, None

    df_n, sequons, retention = analyze_protein(pdb_file, '1RUZ', design_fasta)

    print(f"Total N residues: {len(df_n)}")
    print(f"Sequon-N: {df_n['is_sequon_n'].sum()}")
    print(f"Sequons found: {len(sequons)}")

    for s in sequons[:10]:  # Show first 10
        print(f"  Chain {s['chain']} pos {s['position']}: {s['sequon']}")

    return df_n, retention


def merge_rsa_with_retention(df_n, df_retention, pdb_id):
    """Merge RSA data with retention data using sequential ordering."""

    if df_retention is None or len(df_retention) == 0:
        return None

    # Filter to just this protein
    df_n_prot = df_n[df_n['pdb_id'] == pdb_id].copy()
    df_ret_prot = df_retention[df_retention['pdb_id'] == pdb_id].copy()

    # Get unique chains in order
    chains = df_n_prot['chain'].unique()

    results = []
    for chain_idx, chain_id in enumerate(chains):
        # Get N residues in this chain, sorted by position
        chain_n = df_n_prot[df_n_prot['chain'] == chain_id].sort_values('res_num')

        # Get retention data for this chain
        chain_ret = df_ret_prot[df_ret_prot['chain_idx'] == chain_idx]

        # Create position mapping
        n_positions = chain_n.reset_index()

        for idx, n_row in n_positions.iterrows():
            # Find corresponding retention entry
            pos_in_chain = n_row.name  # Use the sequential index

            # Look for retention data at this N's position
            ret_matches = chain_ret[chain_ret['pos_in_chain'] == n_row['res_num']]

            if len(ret_matches) == 0:
                # Try matching by counting N positions
                continue

            ret_row = ret_matches.iloc[0]

            results.append({
                'pdb_id': pdb_id,
                'chain': chain_id,
                'res_num': n_row['res_num'],
                'rsa': n_row['rsa'],
                'rsa_bin': n_row['rsa_bin'],
                'is_sequon': n_row['is_sequon_n'],
                'n_retention_pct': ret_row['n_retention_pct'],
                'functional_pct': ret_row.get('functional_pct', None)
            })

    return pd.DataFrame(results) if results else None


def print_rsa_analysis(df_n, df_retention, label=""):
    """Print RSA-stratified analysis."""

    print(f"\n{'='*70}")
    print(f"RSA-STRATIFIED ANALYSIS {label}")
    print("="*70)

    # RSA distribution
    print("\n1. RSA DISTRIBUTION OF N RESIDUES")
    print("-"*50)

    for is_sequon in [True, False]:
        subset = df_n[df_n['is_sequon_n'] == is_sequon]
        label_str = "Sequon-N" if is_sequon else "Non-sequon-N"
        print(f"\n{label_str} (n={len(subset)}):")

        for bin_name in ['buried', 'intermediate', 'exposed']:
            count = len(subset[subset['rsa_bin'] == bin_name])
            pct = 100 * count / len(subset) if len(subset) > 0 else 0
            print(f"  {bin_name}: {count} ({pct:.1f}%)")

    # Retention by RSA bin (if we have retention data)
    if df_retention is not None and len(df_retention) > 0:
        print("\n2. N RETENTION BY RSA BIN AND SEQUON STATUS")
        print("-"*50)

        # Group retention by sequon status and summarize
        for is_sequon in [True, False]:
            subset = df_retention[df_retention['is_sequon'] == is_sequon]
            label_str = "Sequon-N" if is_sequon else "Non-sequon-N"

            if len(subset) > 0:
                mean_ret = subset['n_retention_pct'].mean()
                print(f"\n{label_str}: mean N retention = {mean_ret:.1f}% (n={len(subset)} positions)")

        # Compare sequon vs non-sequon
        sequon_ret = df_retention[df_retention['is_sequon']]['n_retention_pct'].mean()
        non_sequon_ret = df_retention[~df_retention['is_sequon']]['n_retention_pct'].mean()

        print(f"\nDifference (Sequon - Non-sequon): {sequon_ret - non_sequon_ret:+.1f}%")

        # Functional retention for sequons
        sequon_data = df_retention[df_retention['is_sequon']]
        if 'functional_pct' in sequon_data.columns:
            func_data = sequon_data[sequon_data['functional_pct'].notna()]
            if len(func_data) > 0:
                mean_exact = func_data['n_retention_pct'].mean()
                mean_func = func_data['functional_pct'].mean()
                print(f"\nSequon positions (n={len(func_data)}):")
                print(f"  Mean N retention: {mean_exact:.1f}%")
                print(f"  Mean functional retention: {mean_func:.1f}%")
                print(f"  Ratio (functional/exact): {mean_func/mean_exact:.2f}x" if mean_exact > 0 else "")

    # KEY ANALYSIS: N retention by RSA bin AND sequon status
    print("\n3. N RETENTION WITHIN EACH RSA BIN (Controlling for Solvent Accessibility)")
    print("-"*70)

    if df_retention is not None and len(df_retention) > 0 and 'rsa_bin' in df_retention.columns:
        print("\n{:15} {:^30} {:^30} {:^10}".format("", "Sequon-N", "Non-Sequon-N", ""))
        print("{:15} {:^15} {:^10} {:^15} {:^10} {:^10}".format(
            "RSA Bin", "Mean ± SD", "n", "Mean ± SD", "n", "Diff"))
        print("-" * 85)

        summary_data = []
        for rsa_bin in ['buried', 'intermediate', 'exposed']:
            # Filter retention data by RSA bin
            bin_data = df_retention[df_retention['rsa_bin'] == rsa_bin]

            sequon_in_bin = bin_data[bin_data['is_sequon'] == True]
            non_sequon_in_bin = bin_data[bin_data['is_sequon'] == False]

            seq_mean = sequon_in_bin['n_retention_pct'].mean() if len(sequon_in_bin) > 0 else np.nan
            seq_std = sequon_in_bin['n_retention_pct'].std() if len(sequon_in_bin) > 0 else np.nan
            seq_n = len(sequon_in_bin)

            non_seq_mean = non_sequon_in_bin['n_retention_pct'].mean() if len(non_sequon_in_bin) > 0 else np.nan
            non_seq_std = non_sequon_in_bin['n_retention_pct'].std() if len(non_sequon_in_bin) > 0 else np.nan
            non_seq_n = len(non_sequon_in_bin)

            diff = seq_mean - non_seq_mean if not (np.isnan(seq_mean) or np.isnan(non_seq_mean)) else np.nan

            summary_data.append({
                'rsa_bin': rsa_bin,
                'sequon_mean': seq_mean,
                'sequon_std': seq_std,
                'sequon_n': seq_n,
                'non_sequon_mean': non_seq_mean,
                'non_sequon_std': non_seq_std,
                'non_sequon_n': non_seq_n,
                'diff': diff
            })

            seq_str = f"{seq_mean:.1f} ± {seq_std:.1f}" if not np.isnan(seq_mean) else "N/A"
            non_seq_str = f"{non_seq_mean:.1f} ± {non_seq_std:.1f}" if not np.isnan(non_seq_mean) else "N/A"
            diff_str = f"{diff:+.1f}%" if not np.isnan(diff) else "N/A"

            print("{:15} {:^15} {:^10} {:^15} {:^10} {:^10}".format(
                rsa_bin, seq_str, f"(n={seq_n})",
                non_seq_str, f"(n={non_seq_n})", diff_str))

        # Print interpretation
        print("\n  Interpretation:")
        print("  - Negative difference means sequon-N retained LESS than non-sequon-N")
        print("  - If differences persist across bins, effect is NOT due to solvent accessibility alone")

        return summary_data
    else:
        print("  (RSA data not available in retention dataframe)")
        return None


def create_n_retention_figure(df_retention, output_path):
    """Create figure showing N retention by RSA and sequon status."""

    if df_retention is None or len(df_retention) == 0:
        print("No retention data for N retention figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sequon_data = df_retention[df_retention['is_sequon']]
    non_sequon_data = df_retention[~df_retention['is_sequon']]

    # Panel 1: Overall N retention - sequon vs non-sequon
    ax1 = axes[0]

    categories = ['Sequon-N', 'Non-sequon-N']
    means = [
        sequon_data['n_retention_pct'].mean() if len(sequon_data) > 0 else 0,
        non_sequon_data['n_retention_pct'].mean() if len(non_sequon_data) > 0 else 0
    ]
    stds = [
        sequon_data['n_retention_pct'].std() if len(sequon_data) > 0 else 0,
        non_sequon_data['n_retention_pct'].std() if len(non_sequon_data) > 0 else 0
    ]
    counts = [len(sequon_data), len(non_sequon_data)]

    x = np.arange(len(categories))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'gray'], edgecolor='black')

    ax1.set_ylabel('N Retention (%)', fontsize=12)
    ax1.set_title('A. Overall N Retention', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{cat}\n(n={c})' for cat, c in zip(categories, counts)])
    ax1.set_ylim(0, 100)

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', fontsize=11)

    # Panel 2: RSA-stratified N retention comparison
    ax2 = axes[1]

    if 'rsa_bin' in df_retention.columns:
        rsa_bins = ['buried', 'intermediate', 'exposed']
        x = np.arange(len(rsa_bins))
        width = 0.35

        sequon_means = []
        non_sequon_means = []
        sequon_ns = []
        non_sequon_ns = []

        for rsa_bin in rsa_bins:
            bin_data = df_retention[df_retention['rsa_bin'] == rsa_bin]
            seq_data = bin_data[bin_data['is_sequon'] == True]
            non_seq_data = bin_data[bin_data['is_sequon'] == False]

            sequon_means.append(seq_data['n_retention_pct'].mean() if len(seq_data) > 0 else 0)
            non_sequon_means.append(non_seq_data['n_retention_pct'].mean() if len(non_seq_data) > 0 else 0)
            sequon_ns.append(len(seq_data))
            non_sequon_ns.append(len(non_seq_data))

        bars1 = ax2.bar(x - width/2, sequon_means, width, label='Sequon-N', color='steelblue', edgecolor='black')
        bars2 = ax2.bar(x + width/2, non_sequon_means, width, label='Non-sequon-N', color='gray', edgecolor='black')

        ax2.set_ylabel('N Retention (%)', fontsize=12)
        ax2.set_title('B. N Retention by RSA Bin\n(Controlling for Solvent Accessibility)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{b}\n(n={s},{ns})' for b, s, ns in zip(rsa_bins, sequon_ns, non_sequon_ns)])
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')

        # Add value labels
        for bar, mean in zip(bars1, sequon_means):
            if mean > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{mean:.0f}', ha='center', fontsize=9)
        for bar, mean in zip(bars2, non_sequon_means):
            if mean > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{mean:.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved N retention figure: {output_path}")


def create_sequon_retention_figure(df_retention, output_path):
    """Create figure showing exact vs functional SEQUON retention by RSA bin.

    Exact sequon retention = full N-X-S/T triplet matches WT exactly
    Functional sequon retention = any valid N-X-S/T (includes exact + other valid)

    Note: Functional should always be >= Exact since functional includes exact.
    """

    if df_retention is None or len(df_retention) == 0:
        print("No retention data for sequon retention figure")
        return

    # Filter to only sequon positions with functional data
    sequon_data = df_retention[df_retention['is_sequon'] & df_retention['functional_pct'].notna()]

    if len(sequon_data) == 0:
        print("No sequon data available for sequon retention figure")
        return

    # Check if we have exact_sequon_pct (new metric)
    has_exact_sequon = 'exact_sequon_pct' in sequon_data.columns

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Overall exact vs functional sequon retention
    ax1 = axes[0]

    if has_exact_sequon:
        exact_mean = sequon_data['exact_sequon_pct'].mean()
    else:
        # Fallback to N retention if exact sequon not available (shouldn't happen after re-run)
        exact_mean = sequon_data['n_retention_pct'].mean()
    func_mean = sequon_data['functional_pct'].mean()

    categories = ['Exact Sequon\nRetention', 'Functional\nSequon Retention']
    means = [exact_mean, func_mean]
    colors = ['steelblue', 'orange']

    bars = ax1.bar(range(2), means, color=colors, edgecolor='black')
    ax1.set_ylabel('Retention (%)', fontsize=12)
    ax1.set_title(f'A. Overall Sequon Retention\n(n={len(sequon_data)} sequon sites)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, max(100, max(means) * 1.2 if max(means) > 0 else 10))

    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{mean:.1f}%', ha='center', fontsize=11)

    if exact_mean > 0:
        ratio = func_mean / exact_mean
        ax1.text(0.5, 0.92, f'Func/Exact Ratio: {ratio:.2f}x',
                transform=ax1.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Panel 2: RSA-stratified exact vs functional sequon retention
    ax2 = axes[1]

    if 'rsa_bin' in sequon_data.columns:
        rsa_bins = ['buried', 'intermediate', 'exposed']
        x = np.arange(len(rsa_bins))
        width = 0.35

        exact_means = []
        func_means = []
        ns = []

        for rsa_bin in rsa_bins:
            bin_data = sequon_data[sequon_data['rsa_bin'] == rsa_bin]

            if has_exact_sequon:
                exact_means.append(bin_data['exact_sequon_pct'].mean() if len(bin_data) > 0 else 0)
            else:
                exact_means.append(bin_data['n_retention_pct'].mean() if len(bin_data) > 0 else 0)
            func_means.append(bin_data['functional_pct'].mean() if len(bin_data) > 0 else 0)
            ns.append(len(bin_data))

        bars1 = ax2.bar(x - width/2, exact_means, width, label='Exact Sequon', color='steelblue', edgecolor='black')
        bars2 = ax2.bar(x + width/2, func_means, width, label='Functional Sequon', color='orange', edgecolor='black')

        ax2.set_ylabel('Retention (%)', fontsize=12)
        ax2.set_title('B. Sequon Retention by RSA Bin\n(Exact vs Functional)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{b}\n(n={n})' for b, n in zip(rsa_bins, ns)])
        ax2.set_ylim(0, max(100, max(max(exact_means), max(func_means)) * 1.2 if max(exact_means) > 0 or max(func_means) > 0 else 10))
        ax2.legend(loc='upper right')

        # Add value labels
        for bar, mean in zip(bars1, exact_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{mean:.1f}', ha='center', fontsize=9)
        for bar, mean in zip(bars2, func_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{mean:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved sequon retention figure: {output_path}")


def main():
    base_dir = Path(__file__).parent.parent

    print("="*70)
    print("SOLVENT ACCESSIBILITY ANALYSIS")
    print("="*70)

    # Create output directory
    output_dir = base_dir / 'solvent_accessibility_analysis'
    output_dir.mkdir(exist_ok=True)

    # Analyze main dataset
    df_all_n, df_all_retention = analyze_main_dataset(base_dir)

    if df_all_n is not None:
        # Save data
        df_all_n.to_csv(output_dir / 'main_dataset_n_rsa.csv', index=False)
        if df_all_retention is not None:
            df_all_retention.to_csv(output_dir / 'main_dataset_retention.csv', index=False)

        # Print analysis
        print_rsa_analysis(df_all_n, df_all_retention, "(Main Dataset)")

        # Create figures (two separate figures)
        create_n_retention_figure(df_all_retention, output_dir / 'fig_main_n_retention.png')
        create_sequon_retention_figure(df_all_retention, output_dir / 'fig_main_sequon_retention.png')

    # Analyze HA
    df_ha_n, df_ha_retention = analyze_ha(base_dir)

    if df_ha_n is not None:
        df_ha_n.to_csv(output_dir / 'ha_n_rsa.csv', index=False)
        if df_ha_retention is not None:
            df_ha_retention.to_csv(output_dir / 'ha_retention.csv', index=False)

        print_rsa_analysis(df_ha_n, df_ha_retention, "(HA Case Study)")
        create_n_retention_figure(df_ha_retention, output_dir / 'fig_ha_n_retention.png')
        create_sequon_retention_figure(df_ha_retention, output_dir / 'fig_ha_sequon_retention.png')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {output_dir}")

    return df_all_n, df_all_retention, df_ha_n, df_ha_retention


if __name__ == '__main__':
    main()
