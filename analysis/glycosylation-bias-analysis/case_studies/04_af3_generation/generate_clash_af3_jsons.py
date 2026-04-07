#!/usr/bin/env python3
"""Generate AF3 JSON input files for the glycan clash experiment.

Reads clash_sequences.json (produced by generate_clash_sequences.py) and
generates AF3 JSONs for each condition.

Supports two glycan modes:
  --glycan-mode stub   AF3 Server format (alphafoldserver dialect v1).
                       Each glycan is a single NAG specified via the glycans
                       array on the proteinChain.  Use this for the AF3 Server
                       or as a quick pilot before setting up local AF3.

  --glycan-mode full   AF3 local format (alphafold3 dialect v2).
                       Full glycan trees with CCD codes and bondedAtomPairs.
                       Requires local AF3 installation.

Conditions generated:
  1. <PDB>_wt_with_glycan          WT sequence + glycans
  2. <PDB>_mpnn_reintroduced       MPNN design with sequon grafted back + glycans
  3. <PDB>_mpnn_unconstrained      MPNN design as-is, no glycans (reference)

Usage:
    python generate_clash_af3_jsons.py --pdb-ids 1DBN 1J2E --glycan-mode stub
    python generate_clash_af3_jsons.py --pdb-ids 5IFP --glycan-mode full
    python generate_clash_af3_jsons.py --all --glycan-mode stub
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"

# For full glycan mode, we may need extract_pdb_glycans
PREP_STEP_DIR = PIPELINE_ROOT / "01_preparation"
if str(PREP_STEP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_STEP_DIR))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def spreadsheet_id(index):
    """AF3-style entity IDs: A, B, ..., Z, AA, BA, ..."""
    letters = []
    n = index
    while True:
        letters.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(letters)


def load_clash_data(pdb_id):
    """Load clash_sequences.json for a protein."""
    path = DATA_DIR / "outputs" / f"output_{pdb_id}" / "clash_experiment" / "clash_sequences.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stub mode: AF3 Server format (alphafoldserver dialect v1)
# ---------------------------------------------------------------------------

def make_stub_json(name, chain_sequences, glycan_positions=None):
    """Create AF3 Server JSON with optional NAG stubs.

    glycan_positions: dict of chain_idx -> list of 1-indexed positions.

    Handles homodimers/homo-oligomers: the AF3 Server requires identical
    sequences to be merged into a single proteinChain with count > 1.
    When merging, glycan sets are unified (union of all positions across
    identical chains) so each copy gets the same glycan specification.
    """
    # Group chains by sequence to detect homo-oligomers
    from collections import OrderedDict
    seq_groups = OrderedDict()  # sequence -> list of chain indices
    for i, seq in enumerate(chain_sequences):
        seq_groups.setdefault(seq, []).append(i)

    sequences = []
    for seq, chain_indices in seq_groups.items():
        chain_data = {"sequence": seq, "count": len(chain_indices)}

        if glycan_positions:
            # Union of glycan positions across all identical chains
            all_positions = set()
            for ci in chain_indices:
                if ci in glycan_positions:
                    all_positions.update(glycan_positions[ci])
            if all_positions:
                chain_data["glycans"] = sorted(
                    [{"residues": "NAG", "position": p} for p in all_positions],
                    key=lambda g: g["position"],
                )
                if len(chain_indices) > 1:
                    # Check if we had to unify asymmetric glycan sets
                    per_chain = [set(glycan_positions.get(ci, []))
                                 for ci in chain_indices]
                    if len(set(frozenset(s) for s in per_chain)) > 1:
                        print(f"    Note: unified asymmetric glycans across "
                              f"{len(chain_indices)} identical chains "
                              f"(union: {sorted(all_positions)})")

        sequences.append({"proteinChain": chain_data})

    return [{
        "name": name,
        "modelSeeds": [],
        "sequences": sequences,
        "dialect": "alphafoldserver",
        "version": 1,
    }]


# ---------------------------------------------------------------------------
# Full mode: AF3 local format (alphafold3 dialect v2)
# ---------------------------------------------------------------------------

def make_full_glycan_json(name, chain_sequences, glycan_sites):
    """Create AF3 local JSON with full glycan trees and bondedAtomPairs.

    glycan_sites: list of dicts, each with:
      - chain_idx, position_1idx, glycan_residues (list of CCD codes),
        glycan_bonds (list of bond dicts)
    """
    protein_ids = [spreadsheet_id(i) for i in range(len(chain_sequences))]

    sequences = []
    for pid, seq in zip(protein_ids, chain_sequences):
        sequences.append({"protein": {"id": pid, "sequence": seq}})

    bonded_atom_pairs = []
    glycan_count = 0

    for site in glycan_sites:
        chain_idx = site["chain_idx"]
        mpnn_pos = site["position_1idx"]
        ccd_codes = list(site["glycan_residues"])
        bonds = site.get("glycan_bonds", [])
        protein_id = protein_ids[chain_idx]

        glycan_count += 1
        glycan_id = f"G{glycan_count}"

        sequences.append({
            "ligand": {
                "ccdCodes": ccd_codes,
                "id": glycan_id,
            }
        })

        # Protein Asn ND2 -> first sugar C1
        bonded_atom_pairs.append(
            [[protein_id, mpnn_pos, "ND2"], [glycan_id, 1, "C1"]]
        )

        # Intra-glycan bonds
        for bond in bonds:
            bonded_atom_pairs.append([
                [glycan_id, bond["from_res_idx"], bond["from_atom"]],
                [glycan_id, bond["to_res_idx"], bond["to_atom"]],
            ])

    result = {
        "name": name,
        "modelSeeds": [1, 2, 3, 4, 5],
        "sequences": sequences,
        "bondedAtomPairs": bonded_atom_pairs,
        "dialect": "alphafold3",
        "version": 2,
    }
    return result


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_for_protein(pdb_id, glycan_mode, verbose=True):
    """Generate AF3 JSONs for one protein's clash experiment."""
    data = load_clash_data(pdb_id)
    if data is None:
        print(f"  {pdb_id}: No clash_sequences.json found. Run generate_clash_sequences.py first.")
        return []

    sites = data["sites"]
    conditions = data["conditions"]
    chain_order = data["chain_order"]

    # Filter to sites with resolved glycans (all should have them, but be safe)
    glycan_sites = [s for s in sites if s.get("glycan_residues")]

    if not glycan_sites:
        print(f"  {pdb_id}: No glycan sites found.")
        return []

    # In stub mode, filter to single-NAG sites only
    if glycan_mode == "stub":
        usable_sites = [s for s in glycan_sites if s["is_stub"]]
        mode_label = "stub"
        if not usable_sites:
            print(f"  {pdb_id}: No single-NAG stub sites. Use --glycan-mode full for complex glycans.")
            # Fall back to using all sites with just the first NAG
            usable_sites = glycan_sites
            mode_label = "stub_fallback"
            if verbose:
                print(f"  {pdb_id}: Using all {len(usable_sites)} sites with NAG stub (ignoring tree beyond first sugar)")
    else:
        usable_sites = glycan_sites
        mode_label = "full"

    output_dir = DATA_DIR / "outputs" / f"output_{pdb_id}" / "clash_experiment" / "af3_jsons"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []

    # --- Condition 1: WT with glycans ---
    wt_seqs = conditions["wt_with_glycan"]["chain_sequences"]
    name_wt = f"{pdb_id}_clash_wt_with_glycan"

    if glycan_mode == "stub":
        glycan_pos = {}
        for s in usable_sites:
            ci = s["chain_idx"]
            glycan_pos.setdefault(ci, []).append(s["position_1idx"])
        json_data = make_stub_json(name_wt, wt_seqs, glycan_pos)
    else:
        json_data = make_full_glycan_json(name_wt, wt_seqs, usable_sites)

    path = output_dir / f"{name_wt}_AF3.json"
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)
    written.append(path)

    # --- Condition 2: MPNN reintroduced with glycans ---
    reintro_seqs = conditions["mpnn_reintroduced"]["chain_sequences"]
    name_reintro = f"{pdb_id}_clash_mpnn_reintroduced"

    if glycan_mode == "stub":
        # Same glycan positions as WT — the sequon has been grafted back
        json_data = make_stub_json(name_reintro, reintro_seqs, glycan_pos)
    else:
        json_data = make_full_glycan_json(name_reintro, reintro_seqs, usable_sites)

    path = output_dir / f"{name_reintro}_AF3.json"
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)
    written.append(path)

    # --- Condition 3: MPNN unconstrained, no glycans ---
    unc_seqs = conditions["mpnn_unconstrained"]["chain_sequences"]
    name_unc = f"{pdb_id}_clash_mpnn_unconstrained"

    # No glycans for this condition
    if glycan_mode == "stub":
        json_data = make_stub_json(name_unc, unc_seqs)
    else:
        # Still alphafold3 dialect but no glycans
        protein_ids = [spreadsheet_id(i) for i in range(len(unc_seqs))]
        sequences = [{"protein": {"id": pid, "sequence": seq}}
                     for pid, seq in zip(protein_ids, unc_seqs)]
        json_data = {
            "name": name_unc,
            "modelSeeds": [1, 2, 3, 4, 5],
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 2,
        }

    path = output_dir / f"{name_unc}_AF3.json"
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)
    written.append(path)

    # Summary
    n_testable = sum(1 for s in usable_sites if s["testable"])
    if verbose:
        print(f"  {pdb_id}: Wrote {len(written)} AF3 JSONs ({mode_label} mode)")
        print(f"    Glycan sites: {len(usable_sites)} ({n_testable} testable)")
        for s in usable_sites:
            tag = "TESTABLE" if s["testable"] else "kept"
            print(f"      {s['chain']}:{s['position_1idx']} {s['wt_triplet']}→{s['design_triplet']} "
                  f"[{s['glycan_residues_string']}] {tag}")
        for p in written:
            print(f"    {p.name}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AF3 JSONs for the glycan clash experiment."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb-ids", nargs="+", help="PDB IDs to process")
    group.add_argument("--all", action="store_true",
                       help="Process all proteins with clash_sequences.json")

    parser.add_argument(
        "--glycan-mode", choices=["stub", "full"], default="stub",
        help="Glycan representation: 'stub' = single NAG (AF3 Server), "
             "'full' = complete trees (local AF3). Default: stub."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        # Find all proteins with clash_sequences.json
        outputs_dir = DATA_DIR / "outputs"
        pdb_ids = []
        for d in sorted(outputs_dir.iterdir()):
            clash_json = d / "clash_experiment" / "clash_sequences.json"
            if clash_json.exists():
                pdb_ids.append(d.name.replace("output_", ""))
    else:
        pdb_ids = [p.upper() for p in args.pdb_ids]

    if not pdb_ids:
        print("No proteins to process. Run generate_clash_sequences.py first.")
        return

    print(f"Glycan mode: {args.glycan_mode}")
    print(f"Processing {len(pdb_ids)} proteins: {', '.join(pdb_ids)}\n")

    total_files = 0
    for pdb_id in pdb_ids:
        written = generate_for_protein(pdb_id, args.glycan_mode)
        total_files += len(written)

    print(f"\nDone. {total_files} AF3 JSON files written.")


if __name__ == "__main__":
    main()
