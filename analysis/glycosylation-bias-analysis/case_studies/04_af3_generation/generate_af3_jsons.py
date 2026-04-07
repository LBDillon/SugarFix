#!/usr/bin/env python3
"""Generate AlphaFold3 Server JSON input files for the top-scoring
ProteinMPNN design of each protein under unconstrained and
full_sequon_fixed conditions.

For each protein/condition, produces:
  - Plain JSON (no glycans)
  - _with_glycans JSON (glycans at original sequon sites retained in design)
  - _denovo_glycans JSON (unconstrained only: glycans at ALL N-X-S/T sites)

Skips 1EO8 proteins. Produces output in a top_designs_for_AF3/
subdirectory next to each protein's output folder.
"""

import csv
import json
import os
import re
import sys
import glob
import argparse
from collections import defaultdict
from pathlib import Path

# Add mpnn_utils to path (lives in 02_design/)
_DESIGN_DIR = str(Path(__file__).resolve().parent.parent / "02_design")
if _DESIGN_DIR not in sys.path:
    sys.path.insert(0, _DESIGN_DIR)
from mpnn_utils import SEQUON_REGEX

CONDITIONS = ["unconstrained", "full_sequon_fixed"]
SKIP_PREFIXES = ["1EO8"]


def parse_fa_designs(fa_path):
    """Parse .fa file, return list of design dicts."""
    designs = []
    with open(fa_path) as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        header = lines[i].strip().lstrip(">")
        sequence = lines[i + 1].strip()

        score_m = re.search(r"score=([\d.]+)", header)
        sample_m = re.search(r"sample=(\d+)", header)
        recovery_m = re.search(r"seq_recovery=([\d.]+)", header)

        if score_m and sample_m:
            designs.append({
                "sample": int(sample_m.group(1)),
                "score": float(score_m.group(1)),
                "seq_recovery": float(recovery_m.group(1)) if recovery_m else None,
                "sequence": sequence,
            })
    return designs


def find_top_design(fa_path):
    """Return the top-scoring designed sample (lowest score, sample >= 1)."""
    designs = parse_fa_designs(fa_path)
    designed = [d for d in designs if d["sample"] >= 1]
    if not designed:
        return None
    return min(designed, key=lambda d: d["score"])


def find_sequons(sequence):
    """Find all N-X-S/T sequon positions (1-indexed) using canonical SEQUON_REGEX."""
    return [m.start() + 1 for m in SEQUON_REGEX.finditer(sequence)]


def load_original_sequon_positions(structural_context_path):
    """Load original sequon N positions per chain from structural_context.csv.

    Returns three items:
      - positions: dict of chain_index -> list of 1-indexed positions
      - chain_order: list of chain labels in order of appearance
      - evidence: dict of (chain_label, position_1idx) -> evidence_tier str
    """
    positions_by_chain = defaultdict(list)
    chain_order = []
    evidence = {}

    with open(structural_context_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["chain"]
            if chain not in chain_order:
                chain_order.append(chain)
            pos = int(row["position_1idx"])
            positions_by_chain[chain].append(pos)
            tier = row.get("evidence_tier", "motif_only")
            evidence[(chain, pos)] = tier

    # Map to chain index (0-based) matching the order chains appear in the fa file
    result = {}
    for idx, chain in enumerate(chain_order):
        result[idx] = sorted(positions_by_chain[chain])

    return result, chain_order, evidence


def load_pdb_resnum_mapping(structural_context_path):
    """Load mapping from (chain, position_1idx) to PDB resnum.

    Returns dict: (chain_label, position_1idx) -> pdb_resnum.
    """
    mapping = {}
    with open(structural_context_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = row["chain"]
            pos = int(row["position_1idx"])
            pdb_resnum = int(row["pdb_resnum"])
            mapping[(chain, pos)] = pdb_resnum
    return mapping


def load_glycan_trees(glycan_trees_path):
    """Load glycan trees from JSON.

    Returns dict: "chain:pdb_resnum" -> glycan info dict.
    """
    if not os.path.exists(glycan_trees_path):
        return {}
    with open(glycan_trees_path) as f:
        return json.load(f)


def find_glycan_tree(glycan_trees, chain_label, pdb_resnum, tolerance=10):
    """Find glycan tree for a position, allowing fuzzy matching.

    Missing residues can cause offsets between MPNN positions and PDB
    numbering. Try exact match first, then check nearby residues.
    """
    exact_key = f"{chain_label}:{pdb_resnum}"
    if exact_key in glycan_trees:
        return glycan_trees[exact_key]

    # Fuzzy: check nearby PDB residue numbers in same chain
    for offset in range(1, tolerance + 1):
        for delta in [offset, -offset]:
            key = f"{chain_label}:{pdb_resnum + delta}"
            if key in glycan_trees:
                return glycan_trees[key]

    return None


AF3_SERVER_GLYCAN_CCD = "NAG"  # AF3 server accepts a single CCD code per glycan


def make_af3_json(name, chain_sequences, chain_glycans=None):
    """Create an AF3 server JSON structure.

    chain_glycans: optional dict of chain_index -> list of
        {"position": int, "residues": str} dicts.
        The "residues" field must be a single valid CCD code (e.g. "NAG").
        The AF3 server does not accept hyphenated multi-residue strings.
    """
    sequences = []
    for i, seq in enumerate(chain_sequences):
        chain_data = {
            "sequence": seq,
            "count": 1
        }
        if chain_glycans and i in chain_glycans and chain_glycans[i]:
            chain_data["glycans"] = sorted(
                chain_glycans[i], key=lambda g: g["position"]
            )
        sequences.append({"proteinChain": chain_data})

    return [{
        "name": name,
        "modelSeeds": [],
        "sequences": sequences,
        "dialect": "alphafoldserver",
        "version": 1
    }]


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AF3 JSON files from ProteinMPNN FASTAs."
    )
    parser.add_argument(
        "--search-dir",
        default=os.path.dirname(__file__),
        help="Directory to recursively scan for '*/seqs/*.fa' files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    search_dir = os.path.abspath(args.search_dir)

    fa_files = glob.glob(
        os.path.join(search_dir, "**", "seqs", "*.fa"), recursive=True
    )

    for fa_path in sorted(fa_files):
        parts = fa_path.split(os.sep)

        # Determine condition
        condition = None
        for c in CONDITIONS:
            if c in parts:
                condition = c
                break
        if condition is None:
            continue

        protein = os.path.splitext(os.path.basename(fa_path))[0]

        if any(protein.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue

        # Output directory (parent of condition dir)
        condition_idx = parts.index(condition)
        output_dir = os.sep.join(parts[:condition_idx])

        af3_dir = os.path.join(output_dir, "top_designs_for_AF3")
        os.makedirs(af3_dir, exist_ok=True)

        top = find_top_design(fa_path)
        if top is None:
            print(f"  No designs found for {protein} / {condition}")
            continue

        chain_sequences = top["sequence"].split("/")
        job_name = f"{protein}_{condition}_top1"

        # 1) Plain JSON (no glycans)
        plain_path = os.path.join(af3_dir, f"{job_name}_AF3.json")
        write_json(make_af3_json(job_name, chain_sequences), plain_path)

        # Load original sequon positions from structural_context.csv
        struct_ctx_path = os.path.join(output_dir, "structural_context.csv")
        if not os.path.exists(struct_ctx_path):
            print(f"  WARNING: No structural_context.csv for {protein}, skipping glycan JSONs")
            continue

        original_positions, chain_labels, sequon_evidence = load_original_sequon_positions(struct_ctx_path)
        resnum_mapping = load_pdb_resnum_mapping(struct_ctx_path)

        # Load glycan trees from PDB (if available)
        # Search in prep directory - try protein name and PDB ID variants
        glycan_trees = {}
        # Extract likely PDB ID (first 4 chars or before _protein suffix)
        pdb_id = protein.split("_")[0] if "_" in protein else protein
        candidate_names = list(dict.fromkeys([protein, pdb_id]))  # deduplicated
        candidate_dirs = []
        for name in candidate_names:
            candidate_dirs.append(os.path.join(search_dir, "..", "prep", name, "structure"))
            candidate_dirs.append(os.path.join(os.path.dirname(output_dir), "..", "prep", name, "structure"))
        for candidate_dir in candidate_dirs:
            glycan_path_candidate = os.path.join(candidate_dir, "glycan_trees.json")
            if os.path.exists(glycan_path_candidate):
                glycan_trees = load_glycan_trees(glycan_path_candidate)
                break

        # 2) _with_glycans: glycans at original sequon positions that are
        #    retained as N-X-S/T in the designed sequence
        # NOTE: The AF3 server glycans field accepts only a single CCD code
        # per entry (e.g. "NAG"). Full glycan trees from the PDB are stored
        # in prep/<PDB_ID>/structure/glycan_trees.json for reference.
        retained_glycans = {}
        n_validated_retained = 0
        n_motif_only_retained = 0
        for chain_idx, seq in enumerate(chain_sequences):
            design_sequons = set(find_sequons(seq))
            orig = original_positions.get(chain_idx, [])
            chain_label = chain_labels[chain_idx] if chain_idx < len(chain_labels) else str(chain_idx)
            retained_entries = []
            for pos in orig:
                if pos not in design_sequons:
                    continue
                pdb_resnum = resnum_mapping.get((chain_label, pos), pos)
                tree = find_glycan_tree(glycan_trees, chain_label, pdb_resnum)
                tier = sequon_evidence.get((chain_label, pos), "motif_only")
                if not tree and tier == "motif_only":
                    print(f"    Note: No glycan tree in PDB for {chain_label}:{pdb_resnum} (motif_only)")
                if tier in ("experimental", "pdb_evidence"):
                    n_validated_retained += 1
                else:
                    n_motif_only_retained += 1
                retained_entries.append({"residues": AF3_SERVER_GLYCAN_CCD, "position": pos})
            if retained_entries:
                retained_glycans[chain_idx] = retained_entries

        glycan_name = f"{job_name}_with_glycans"
        glycan_out_path = os.path.join(af3_dir, f"{job_name}_AF3_with_glycans.json")
        write_json(
            make_af3_json(glycan_name, chain_sequences, retained_glycans),
            glycan_out_path
        )

        # 3) _denovo_glycans (unconstrained only): glycans at ALL N-X-S/T sites
        if condition == "unconstrained":
            all_glycans = {}
            for chain_idx, seq in enumerate(chain_sequences):
                sequon_positions = find_sequons(seq)
                if not sequon_positions:
                    continue
                entries = []
                for pos in sequon_positions:
                    entries.append({"residues": AF3_SERVER_GLYCAN_CCD, "position": pos})
                all_glycans[chain_idx] = entries

            denovo_name = f"{job_name}_denovo_glycans"
            denovo_path = os.path.join(af3_dir, f"{job_name}_AF3_denovo_glycans.json")
            write_json(
                make_af3_json(denovo_name, chain_sequences, all_glycans),
                denovo_path
            )

        # Summary
        n_retained = sum(len(v) for v in retained_glycans.values())
        extra = ""
        if condition == "unconstrained":
            n_all = sum(len(v) for v in all_glycans.values())
            n_denovo = n_all - n_retained
            extra = f"  denovo={n_denovo}"
        evidence_info = ""
        if n_validated_retained or n_motif_only_retained:
            evidence_info = f"  (validated={n_validated_retained}, motif_only={n_motif_only_retained})"
        print(
            f"{protein:<20} {condition:<22} "
            f"score={top['score']:.4f}  sample={top['sample']:>3}  "
            f"chains={len(chain_sequences)}  retained_glycans={n_retained}{evidence_info}{extra}"
        )


if __name__ == "__main__":
    main()
