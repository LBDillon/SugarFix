from __future__ import annotations

from pathlib import Path
from typing import Optional


def check_sequon(sequence: str, position: int) -> str:
    """Classify the N-X-S/T sequon at 0-indexed `position`.

    Returns 'NXS', 'NXT', 'partial' (e.g. N-P-S/T, biologically non-glycosylated),
    or 'none'.
    """
    if position < 0 or position + 2 >= len(sequence):
        return "none"
    aa0, aa1, aa2 = sequence[position], sequence[position + 1], sequence[position + 2]
    if aa0 != "N":
        return "none"
    if aa1 == "P":
        return "partial"
    if aa2 == "S":
        return "NXS"
    if aa2 == "T":
        return "NXT"
    return "partial"


def pdb_to_model_position(
    pdb_residue_num: int, pdb_path: str, chain_id: str, fuzzy_window: int = 3
) -> Optional[int]:
    """Map a PDB residue number to 1-indexed position in the chain's AA sequence.

    Models like ProteinMPNN and ESM-IF number residues sequentially along the
    chain's amino-acid residues, not by PDB residue number. Returns None if no
    residue within `fuzzy_window` of the requested number is found.
    """
    from Bio.PDB import PDBParser, is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            residues = [r for r in chain.get_residues() if is_aa(r)]
            for idx, residue in enumerate(residues, start=1):
                if residue.id[1] == pdb_residue_num:
                    return idx
            # Fuzzy fallback for renumbered PDBs.
            closest = None
            min_diff = float("inf")
            for idx, residue in enumerate(residues, start=1):
                diff = abs(residue.id[1] - pdb_residue_num)
                if diff < min_diff:
                    min_diff = diff
                    closest = idx
            if min_diff <= fuzzy_window:
                return closest
            return None
    return None


def first_chain_with_residues(pdb_path: str) -> Optional[str]:
    """First chain in the file that has at least one amino-acid residue."""
    from Bio.PDB import PDBParser, is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    for model in structure:
        for chain in model:
            if any(is_aa(r) for r in chain.get_residues()):
                return chain.id
    return None
