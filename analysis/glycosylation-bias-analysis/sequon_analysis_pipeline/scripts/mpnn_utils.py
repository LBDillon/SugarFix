#!/usr/bin/env python3
"""
Shared utilities for ProteinMPNN-consistent sequence handling.

This module ensures that all sequon identification and position fixing
uses ProteinMPNN's internal sequence parsing, avoiding indexing mismatches
between BioPython and ProteinMPNN.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

# Add ProteinMPNN to path for importing its utilities
SCRIPT_DIR = Path(__file__).parent.parent.parent
PMPNN_DIR = SCRIPT_DIR / "ProteinMPNN"
if str(PMPNN_DIR) not in sys.path:
    sys.path.insert(0, str(PMPNN_DIR))

# Sequon pattern: N-X-S/T where X != P
SEQUON_REGEX = re.compile(r"N[^P][ST]")


def get_mpnn_chain_seqs_and_order(pdb_path):
    """
    Extract chain sequences using ProteinMPNN's own parsing.

    This ensures that sequon positions match ProteinMPNN's internal indexing,
    which is critical for fixed_positions_jsonl to work correctly.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        tuple: (chain_seqs dict, chain_order list)
            chain_seqs: {chain_id: sequence_string}
            chain_order: list of chain IDs in ProteinMPNN order
    """
    try:
        from protein_mpnn_utils import parse_PDB
    except ImportError:
        raise ImportError(
            f"Could not import protein_mpnn_utils. "
            f"Ensure ProteinMPNN is at: {PMPNN_DIR}"
        )

    pdb_dict_list = parse_PDB(str(pdb_path))
    if not pdb_dict_list:
        return {}, []

    d = pdb_dict_list[0]

    # Extract chain IDs in order
    chain_ids = sorted([k.split("_")[-1] for k in d.keys() if k.startswith("seq_chain_")])

    # Build sequence dict
    chain_seqs = {}
    for ch in chain_ids:
        seq = d.get(f"seq_chain_{ch}", "")
        # ProteinMPNN uses '-' for gaps; convert to 'X' for regex safety
        seq = ''.join([a if a != '-' else 'X' for a in seq])
        chain_seqs[ch] = seq

    return chain_seqs, chain_ids


def find_sequons(sequence):
    """
    Find all N-X-S/T sequons in a sequence.

    Args:
        sequence: Amino acid sequence string

    Returns:
        list of dicts with 'position_0idx' and 'sequon' keys
    """
    return [
        {"position_0idx": m.start(), "sequon": m.group()}
        for m in SEQUON_REGEX.finditer(sequence)
    ]


def is_functional_sequon(triplet):
    """Check if a triplet is a valid N-X-S/T sequon (X != P)."""
    return bool(SEQUON_REGEX.fullmatch(triplet))


def read_fasta_sequences(fa_path):
    """
    Read sequences from a FASTA file.

    Args:
        fa_path: Path to FASTA file

    Returns:
        list of tuples: [(header, sequence), ...]
    """
    records = []
    header, buf = None, []

    for line in Path(fa_path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header = line[1:]
            buf = []
        else:
            buf.append(line)

    if header is not None:
        records.append((header, "".join(buf)))

    return records


def split_mpnn_concat_seq(concat_seq, chain_order):
    """
    Split ProteinMPNN's concatenated sequence by '/' separator.

    ProteinMPNN outputs sequences like "SEQUENCE_A/SEQUENCE_B/SEQUENCE_C"

    Args:
        concat_seq: Concatenated sequence string
        chain_order: List of chain IDs in order

    Returns:
        dict: {chain_id: sequence}
    """
    parts = concat_seq.split("/")
    return {chain_order[i]: parts[i] for i in range(min(len(parts), len(chain_order)))}


def write_fixed_positions_jsonl(target_name, chain_to_pos0, chain_order, output_path):
    """
    Write fixed_positions.jsonl for ProteinMPNN.

    Args:
        target_name: Name of the target (typically PDB stem)
        chain_to_pos0: Dict of {chain_id: [0-indexed positions to fix]}
        chain_order: List of all chain IDs
        output_path: Path to write the JSONL file

    Returns:
        The payload dict that was written
    """
    payload = {target_name: {}}

    for ch in chain_order:
        # Convert 0-indexed to 1-indexed for ProteinMPNN
        positions_1idx = sorted([p + 1 for p in chain_to_pos0.get(ch, [])])
        payload[target_name][ch] = positions_1idx

    Path(output_path).write_text(json.dumps(payload))
    return payload


def verify_sequon_positions(chain_seqs, sequons_by_chain, pdb_id=""):
    """
    Verify that identified sequon positions actually contain valid sequons.

    This is a critical sanity check to ensure indexing is correct.

    Args:
        chain_seqs: Dict of {chain_id: sequence}
        sequons_by_chain: Dict of {chain_id: [{'position_0idx': int, 'sequon': str}, ...]}
        pdb_id: Optional PDB ID for error messages

    Raises:
        AssertionError: If any position doesn't contain the expected sequon
    """
    for ch, seqlist in sequons_by_chain.items():
        seq = chain_seqs.get(ch, "")
        for s in seqlist:
            p0 = s["position_0idx"]
            expected = s["sequon"]

            if p0 + 3 > len(seq):
                raise AssertionError(
                    f"[{pdb_id}] Chain {ch} position {p0}: "
                    f"sequence too short (len={len(seq)})"
                )

            actual = seq[p0:p0+3]
            if actual != expected:
                raise AssertionError(
                    f"[{pdb_id}] Chain {ch} position {p0}: "
                    f"expected '{expected}' but found '{actual}'"
                )

            if not is_functional_sequon(actual):
                raise AssertionError(
                    f"[{pdb_id}] Chain {ch} position {p0}: "
                    f"'{actual}' is not a valid sequon"
                )


def compute_sequence_identity(seq1, seq2):
    """
    Compute sequence identity between two sequences.

    Args:
        seq1, seq2: Sequence strings (must be same length)

    Returns:
        float: Percentage identity (0-100)
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Length mismatch: {len(seq1)} vs {len(seq2)}")

    if not seq1:
        return np.nan

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return 100.0 * matches / len(seq1)


def sanity_check_full_fixed(fa_path, chain_order, sequons_by_chain):
    """
    Verify that FULL sequon fixing preserved all WT triplets exactly.

    Args:
        fa_path: Path to FASTA file with designs
        chain_order: List of chain IDs
        sequons_by_chain: Dict of sequon positions

    Returns:
        bool: True if all designs preserve WT triplets at sequon sites
    """
    records = read_fasta_sequences(fa_path)
    if len(records) < 2:
        return False

    wt_concat = records[0][1]
    designs = [seq for _, seq in records[1:]]
    wt_chains = split_mpnn_concat_seq(wt_concat, chain_order)

    for ch, seqlist in sequons_by_chain.items():
        for s in seqlist:
            p0 = s["position_0idx"]
            wt_triplet = wt_chains[ch][p0:p0+3]

            for design in designs:
                d_chains = split_mpnn_concat_seq(design, chain_order)
                if d_chains[ch][p0:p0+3] != wt_triplet:
                    return False

    return True
