#!/usr/bin/env python3
"""Generate AF3 JSON with full glycan branches using CCD + bondedAtomPairs.

This script consumes the topology stored in prep/<PDB_ID>/structure/glycan_trees.json
and emits an AlphaFold 3 local JSON using the "alphafold3" dialect. For older
prep folders that predate the explicit bond schema, it falls back to rebuilding
the glycan topology directly from the raw PDB LINK records.

Usage:
    python generate_full_glycan_af3_json.py --pdb-id 5IFP [--condition full_sequon_fixed]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"
PREP_STEP_DIR = PIPELINE_ROOT / "01_preparation"
if str(PREP_STEP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_STEP_DIR))

from extract_pdb_glycans import extract_glycan_trees  # noqa: E402


def spreadsheet_id(index):
    """Return AF3-style entity IDs: A, B, ..., Z, AA, BA, ..."""
    letters = []
    n = index
    while True:
        letters.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(letters)


def load_stub_sequences(af3_json_path):
    """Load all chain sequences from an existing AlphaFold Server JSON."""
    with open(af3_json_path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected alphafoldserver JSON list in {af3_json_path}")

    sequences = []
    for entry in data[0].get("sequences", []):
        protein_chain = entry.get("proteinChain")
        if protein_chain and protein_chain.get("sequence"):
            sequences.append(protein_chain["sequence"])
    return sequences


def load_chain_order(prep_dir, structural_context_path, n_chains):
    """Load original chain labels in the MPNN chain order."""
    chain_order_path = prep_dir / "sequons" / "mpnn_chain_order.json"
    if chain_order_path.exists():
        with open(chain_order_path) as f:
            data = json.load(f)
        chain_order = list(data.get("chain_order", []))
        if len(chain_order) >= n_chains:
            return chain_order[:n_chains]

    chain_order = []
    with open(structural_context_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["chain"]
            if chain not in chain_order:
                chain_order.append(chain)

    if len(chain_order) >= n_chains:
        return chain_order[:n_chains]

    while len(chain_order) < n_chains:
        chain_order.append(spreadsheet_id(len(chain_order)))
    return chain_order


def load_mpnn_to_pdb_mapping(structural_context_path):
    """Load mapping: (chain_label, mpnn_1idx) -> pdb_resnum."""
    mapping = {}
    with open(structural_context_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["chain"]
            mpnn_pos = int(row["position_1idx"])
            pdb_resnum = int(row["pdb_resnum"])
            mapping[(chain, mpnn_pos)] = pdb_resnum
    return mapping


def find_sequons(sequence):
    """Find all N-X-S/T sequon positions (1-indexed) where X != P."""
    positions = []
    for i in range(len(sequence) - 2):
        if (
            sequence[i] == "N"
            and sequence[i + 1] not in ("P", "X")
            and sequence[i + 2] in ("S", "T")
        ):
            positions.append(i + 1)
    return positions


def _has_explicit_bonds(glycan_trees):
    if not glycan_trees:
        return False
    return all("bonds" in tree for tree in glycan_trees.values())


def load_glycan_trees(prep_dir, pdb_id):
    """Load explicit glycan topology, rebuilding it if the stored JSON is legacy."""
    glycan_path = prep_dir / "structure" / "glycan_trees.json"
    if glycan_path.exists():
        with open(glycan_path) as f:
            glycan_trees = json.load(f)
        if _has_explicit_bonds(glycan_trees):
            return glycan_trees, "stored"

    pdb_path = prep_dir / "structure" / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        pdb_path = prep_dir / "structure" / f"{pdb_id}_protein.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found for {pdb_id}")

    return extract_glycan_trees(pdb_path), "rebuilt_from_pdb"


def generate_full_glycan_json(pdb_id, condition="full_sequon_fixed"):
    """Generate AF3 JSON with full glycan branches for a protein."""
    output_dir = DATA_DIR / "outputs" / f"output_{pdb_id}"
    prep_dir = DATA_DIR / "prep" / pdb_id

    stub_json_path = (
        output_dir
        / "top_designs_for_AF3"
        / f"{pdb_id}_protein_{condition}_top1_AF3_with_glycans.json"
    )
    if not stub_json_path.exists():
        print(f"ERROR: Stub JSON not found: {stub_json_path}")
        return None

    structural_context_path = output_dir / "structural_context.csv"
    if not structural_context_path.exists():
        print(f"ERROR: structural_context.csv not found: {structural_context_path}")
        return None

    chain_sequences = load_stub_sequences(stub_json_path)
    if not chain_sequences:
        print(f"ERROR: No protein sequences found in {stub_json_path}")
        return None

    chain_labels = load_chain_order(prep_dir, structural_context_path, len(chain_sequences))
    mpnn_to_pdb = load_mpnn_to_pdb_mapping(structural_context_path)
    glycan_trees, glycan_source = load_glycan_trees(prep_dir, pdb_id)

    print(f"Protein: {pdb_id}, condition: {condition}")
    print(f"Chains: {len(chain_sequences)} ({', '.join(chain_labels)})")
    print(f"Glycan topology source: {glycan_source}")
    print(f"Glycan trees available: {len(glycan_trees)} sites")

    protein_ids = [spreadsheet_id(i) for i in range(len(chain_sequences))]
    sequences = []
    for protein_id, sequence in zip(protein_ids, chain_sequences):
        sequences.append({
            "protein": {
                "id": protein_id,
                "sequence": sequence,
            }
        })

    bonded_atom_pairs = []
    glycan_count = 0

    for chain_idx, (chain_label, protein_id, sequence) in enumerate(
        zip(chain_labels, protein_ids, chain_sequences)
    ):
        design_sequons = sorted(find_sequons(sequence))
        print(
            f"  Chain {chain_label} -> AF3 {protein_id}: "
            f"{len(sequence)} aa, sequons={design_sequons}"
        )

        for mpnn_pos in design_sequons:
            pdb_resnum = mpnn_to_pdb.get((chain_label, mpnn_pos))
            if pdb_resnum is None:
                # De novo sequon or a chain that was not mapped in structural_context.
                continue

            site_key = f"{chain_label}:{pdb_resnum}"
            tree = glycan_trees.get(site_key)
            glycan_count += 1
            glycan_id = f"G{glycan_count}"

            if tree is None:
                sequences.append({
                    "ligand": {
                        "ccdCodes": ["NAG"],
                        "id": glycan_id,
                    }
                })
                bonded_atom_pairs.append(
                    [[protein_id, mpnn_pos, "ND2"], [glycan_id, 1, "C1"]]
                )
                print(
                    f"    {site_key} (MPNN {mpnn_pos}): no tree found, "
                    f"using single NAG stub -> {glycan_id}"
                )
                continue

            ccd_codes = list(tree.get("residues", []))
            bonds = list(tree.get("bonds", []))
            if not ccd_codes:
                continue

            sequences.append({
                "ligand": {
                    "ccdCodes": ccd_codes,
                    "id": glycan_id,
                }
            })

            # Protein Asn -> first sugar.
            bonded_atom_pairs.append(
                [[protein_id, mpnn_pos, "ND2"], [glycan_id, 1, "C1"]]
            )

            for bond in bonds:
                bonded_atom_pairs.append([
                    [glycan_id, bond["from_res_idx"], bond["from_atom"]],
                    [glycan_id, bond["to_res_idx"], bond["to_atom"]],
                ])

            print(
                f"    {site_key} (MPNN {mpnn_pos}): "
                f"{len(ccd_codes)} sugars, {len(bonds)} glycosidic bonds -> {glycan_id}"
            )

    job_name = f"{pdb_id}_{condition}_full_glycans"
    af3_json = {
        "name": job_name,
        "modelSeeds": [1, 2, 3, 4, 5],
        "sequences": sequences,
        "bondedAtomPairs": bonded_atom_pairs,
        "dialect": "alphafold3",
        "version": 2,
    }

    af3_dir = output_dir / "top_designs_for_AF3"
    af3_dir.mkdir(parents=True, exist_ok=True)
    out_path = af3_dir / f"{pdb_id}_protein_{condition}_top1_AF3_full_glycans.json"
    with open(out_path, "w") as f:
        json.dump(af3_json, f, indent=2)

    total_sugars = sum(
        len(entry["ligand"]["ccdCodes"])
        for entry in sequences
        if "ligand" in entry
    )
    print(f"\nOutput: {out_path}")
    print(f"  Protein entities: {len(chain_sequences)}")
    print(f"  Glycan entities: {len(sequences) - len(chain_sequences)}")
    print(f"  Total sugar residues: {total_sugars}")
    print(f"  Total bondedAtomPairs: {len(bonded_atom_pairs)}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate AF3 JSON with full glycan branches using stored topology."
    )
    parser.add_argument("--pdb-id", required=True, help="PDB ID to process")
    parser.add_argument(
        "--condition",
        default="full_sequon_fixed",
        help="Design condition (default: full_sequon_fixed)",
    )
    args = parser.parse_args()

    generate_full_glycan_json(args.pdb_id, args.condition)


if __name__ == "__main__":
    main()
