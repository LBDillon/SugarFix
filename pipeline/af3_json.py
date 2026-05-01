"""AlphaFold 3 JSON builders used by SugarFix tutorials and exports.

The standalone AlphaFold 3 codebase and AlphaFold Server use different JSON
dialects.  This module keeps those details out of notebooks and pipeline
scripts, and makes the glycan modelling choice explicit at export time.
"""

from __future__ import annotations

import json
import string
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AF3_SERVER_GLYCAN_CCD = "NAG"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_protein_sequence(sequence: str, unknown_residue: str = "G") -> str:
    """Replace non-standard protein letters with a valid amino acid code."""
    return "".join(aa if aa in STANDARD_AA else unknown_residue for aa in sequence.upper())


def _entity_id_candidates() -> Iterable[str]:
    alphabet = string.ascii_uppercase
    for width in range(1, 4):
        for letters in product(alphabet, repeat=width):
            yield "".join(letters)


def _next_entity_id(used: set[str]) -> str:
    for candidate in _entity_id_candidates():
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("Could not allocate a unique AF3 entity id")


def _normalise_chain_ids(
    n_chains: int,
    chain_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return valid, unique AF3 entity IDs for protein chains."""
    used: set[str] = set()
    out: List[str] = []
    supplied = list(chain_ids or [])

    for idx in range(n_chains):
        candidate = supplied[idx].upper() if idx < len(supplied) else ""
        if candidate.isalpha() and candidate == candidate.upper() and candidate not in used:
            used.add(candidate)
            out.append(candidate)
        else:
            out.append(_next_entity_id(used))
    return out


def _normalise_glycan_record(record: dict) -> Tuple[List[str], List[dict]]:
    """Return CCD codes and within-glycan bond records for one glycan."""
    residues = record.get("ccdCodes", record.get("residues", AF3_SERVER_GLYCAN_CCD))
    if isinstance(residues, str):
        ccd_codes = [part for part in residues.replace(",", "-").split("-") if part]
    else:
        ccd_codes = [str(part) for part in residues]
    if not ccd_codes:
        ccd_codes = [AF3_SERVER_GLYCAN_CCD]
    return ccd_codes, list(record.get("bonds", []))


def make_af3_server_json(
    name: str,
    chain_sequences: Sequence[str],
    glycan_positions: Optional[Dict[int, List[dict]]] = None,
    unknown_residue: str = "G",
) -> List[dict]:
    """Create an AlphaFold Server JSON payload.

    Glycans in this dialect are intentionally stub-like: the server accepts a
    single CCD code at a protein residue position, but not a bonded multi-CCD
    glycan tree.
    """
    sequences = []
    for chain_idx, sequence in enumerate(chain_sequences):
        chain_payload = {
            "sequence": sanitize_protein_sequence(sequence, unknown_residue),
            "count": 1,
        }
        if glycan_positions and glycan_positions.get(chain_idx):
            server_glycans = []
            for record in glycan_positions[chain_idx]:
                residues, _bonds = _normalise_glycan_record(record)
                server_glycans.append(
                    {
                        "position": int(record["position"]),
                        "residues": residues[0],
                    }
                )
            chain_payload["glycans"] = sorted(
                server_glycans,
                key=lambda record: record["position"],
            )
        sequences.append({"proteinChain": chain_payload})

    return [
        {
            "name": name,
            "modelSeeds": [],
            "sequences": sequences,
            "dialect": "alphafoldserver",
            "version": 1,
        }
    ]


def make_af3_full_json(
    name: str,
    chain_sequences: Sequence[str],
    glycan_positions: Optional[Dict[int, List[dict]]] = None,
    chain_ids: Optional[Sequence[str]] = None,
    model_seeds: Optional[Sequence[int]] = None,
    version: int = 4,
    unknown_residue: str = "G",
) -> dict:
    """Create a standalone AlphaFold 3 JSON payload.

    Glycans are represented as ligand entities with CCD codes and
    ``bondedAtomPairs``.  If a glycan record includes a full tree from
    ``extract_pdb_glycans.extract_glycan_trees`` its internal sugar-sugar bonds
    are preserved; otherwise a single NAG residue is emitted as a covalent stub.
    """
    protein_ids = _normalise_chain_ids(len(chain_sequences), chain_ids)
    used_ids = set(protein_ids)
    sequences = []
    bonded_atom_pairs = []

    for chain_id, sequence in zip(protein_ids, chain_sequences):
        sequences.append(
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sanitize_protein_sequence(sequence, unknown_residue),
                }
            }
        )

    for chain_idx, records in sorted((glycan_positions or {}).items()):
        if chain_idx >= len(protein_ids):
            raise IndexError(f"Glycan chain index {chain_idx} has no matching protein chain")
        protein_id = protein_ids[chain_idx]
        for record in sorted(records, key=lambda item: item["position"]):
            residue_position = int(record["position"])
            ccd_codes, glycan_bonds = _normalise_glycan_record(record)
            glycan_id = _next_entity_id(used_ids)
            description = record.get(
                "description",
                f"N-linked glycan attached to protein chain {protein_id} residue {residue_position}",
            )

            sequences.append(
                {
                    "ligand": {
                        "id": glycan_id,
                        "ccdCodes": ccd_codes,
                        "description": description,
                    }
                }
            )
            bonded_atom_pairs.append(
                [[protein_id, residue_position, "ND2"], [glycan_id, 1, "C1"]]
            )

            for bond in glycan_bonds:
                bonded_atom_pairs.append(
                    [
                        [glycan_id, int(bond["from_res_idx"]), bond["from_atom"]],
                        [glycan_id, int(bond["to_res_idx"]), bond["to_atom"]],
                    ]
                )

    payload = {
        "name": name,
        "modelSeeds": list(model_seeds or [1]),
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": version,
    }
    if bonded_atom_pairs:
        payload["bondedAtomPairs"] = bonded_atom_pairs
    return payload


def make_af3_json(
    name: str,
    chain_sequences: Sequence[str],
    glycan_positions: Optional[Dict[int, List[dict]]] = None,
    dialect: str = "alphafold3",
    **kwargs,
) -> dict | List[dict]:
    """Build either supported AF3 JSON dialect."""
    if dialect == "alphafold3":
        return make_af3_full_json(name, chain_sequences, glycan_positions, **kwargs)
    if dialect == "alphafoldserver":
        allowed = {"unknown_residue"}
        server_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        return make_af3_server_json(
            name,
            chain_sequences,
            glycan_positions,
            **server_kwargs,
        )
    raise ValueError(f"Unknown AF3 dialect: {dialect}")


def write_json(data: dict | List[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
    return path
