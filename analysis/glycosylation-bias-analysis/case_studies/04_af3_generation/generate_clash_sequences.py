#!/usr/bin/env python3
"""Generate sequence variants for the glycan clash experiment.

For each protein, produces three conditions:
  1. wt_with_glycan:      Wild-type sequence, glycans at PDB-resolved sites
  2. mpnn_reintroduced:   Top unconstrained MPNN design with the WT sequon
                          triplet grafted back at each glycosylated position
  3. mpnn_unconstrained:  Top unconstrained MPNN design as-is (reference)

Only sites with a resolved glycan in glycan_trees.json AND where the
unconstrained design destroyed the sequon are "testable" for the clash
hypothesis.  Sites where MPNN kept the sequon are included for completeness
but flagged as non-testable.

Usage:
    python generate_clash_sequences.py --pdb-ids 1DBN 1J2E
    python generate_clash_sequences.py --all          # process all eligible
    python generate_clash_sequences.py --survey        # just print testable site counts
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"

_DESIGN_DIR = str(PIPELINE_ROOT / "02_design")
if _DESIGN_DIR not in sys.path:
    sys.path.insert(0, _DESIGN_DIR)
from mpnn_utils import SEQUON_REGEX  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fasta(fa_path):
    """Return list of (header_str, sequence_str) tuples from a FASTA file."""
    entries = []
    with open(fa_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        entries.append((lines[i].lstrip(">"), lines[i + 1]))
    return entries


def get_wt_and_top_design(fa_path):
    """Return (wt_sequence, top_design_sequence, top_design_score) from FASTA.

    WT is always sample=0 (the first entry).
    Top design is sample >= 1 with the lowest score.
    """
    entries = parse_fasta(fa_path)
    if not entries:
        return None, None, None

    wt_seq = entries[0][1]

    best_score = float("inf")
    best_seq = None
    for header, seq in entries[1:]:
        m = re.search(r"score=([\d.]+)", header)
        if m:
            score = float(m.group(1))
            if score < best_score:
                best_score = score
                best_seq = seq

    return wt_seq, best_seq, best_score


def has_sequon_at(sequence, pos0):
    """Check if there is a valid N-X-S/T sequon at 0-indexed position."""
    if pos0 < 0 or pos0 + 3 > len(sequence):
        return False
    return bool(SEQUON_REGEX.match(sequence[pos0:pos0 + 3]))


def load_chain_order(prep_dir):
    """Load MPNN chain order from prep directory."""
    path = prep_dir / "sequons" / "mpnn_chain_order.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("chain_order", [])
    return []


def load_glycan_trees(prep_dir):
    """Load glycan trees keyed by 'chain:pdb_resnum'."""
    path = prep_dir / "structure" / "glycan_trees.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_structural_context(output_dir):
    """Load structural_context.csv rows as list of dicts."""
    path = output_dir / "structural_context.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_eligible_proteins():
    """Find all proteins with prep data, glycan trees, and unconstrained designs."""
    outputs_dir = DATA_DIR / "outputs"
    prep_dir_root = DATA_DIR / "prep"
    eligible = []

    for output_entry in sorted(outputs_dir.iterdir()):
        if not output_entry.name.startswith("output_"):
            continue
        pdb_id = output_entry.name.replace("output_", "")
        prep = prep_dir_root / pdb_id

        # Must have all required files
        if not (prep / "structure" / "glycan_trees.json").exists():
            continue
        if not (output_entry / "structural_context.csv").exists():
            continue

        fa_pattern = list((output_entry / "unconstrained" / "seqs").glob("*.fa"))
        if not fa_pattern:
            continue

        eligible.append(pdb_id)

    return eligible


def analyse_protein(pdb_id, verbose=True):
    """Analyse one protein and return clash experiment data.

    Returns dict with:
      - pdb_id, chain_order, conditions (wt/mpnn_reintroduced/mpnn_unconstrained)
      - sites: list of per-site dicts with testability info
    """
    output_dir = DATA_DIR / "outputs" / f"output_{pdb_id}"
    prep_dir = DATA_DIR / "prep" / pdb_id

    # Load data sources
    chain_order = load_chain_order(prep_dir)
    glycan_trees = load_glycan_trees(prep_dir)
    sc_rows = load_structural_context(output_dir)

    # Index glycan trees by (chain, pdb_resnum)
    trees_by_pos = {}
    for _key, info in glycan_trees.items():
        trees_by_pos[(info["protein_chain"], info["protein_resnum"])] = info

    # Load sequences
    fa_files = list((output_dir / "unconstrained" / "seqs").glob("*.fa"))
    if not fa_files:
        return None
    wt_full, design_full, design_score = get_wt_and_top_design(fa_files[0])
    if not wt_full or not design_full:
        return None

    wt_chains = wt_full.split("/")
    design_chains = design_full.split("/")

    if not chain_order:
        chain_order = [chr(ord("A") + i) for i in range(len(wt_chains))]

    # Build reintroduced sequence: start from design, paste WT triplet at glycan sites
    reintro_chains = list(design_chains)  # copy

    sites = []
    for row in sc_rows:
        chain = row["chain"]
        pos1 = int(row["position_1idx"])
        pdb_resnum = int(row["pdb_resnum"])
        triplet = row["triplet"]
        tier = row["evidence_tier"]

        # Only process sites with resolved glycans
        tree = trees_by_pos.get((chain, pdb_resnum))
        if tree is None:
            continue

        n_sugars = tree.get("n_sugars", len(tree.get("residues", [])))

        if chain not in chain_order:
            continue
        chain_idx = chain_order.index(chain)
        if chain_idx >= len(wt_chains):
            continue

        pos0 = pos1 - 1  # 0-indexed within chain
        wt_chain = wt_chains[chain_idx]
        des_chain = design_chains[chain_idx]

        if pos0 + 3 > len(wt_chain):
            continue

        wt_triplet = wt_chain[pos0:pos0 + 3]
        des_triplet = des_chain[pos0:pos0 + 3]

        # Validate: WT should have N at this position and match the expected triplet
        if wt_chain[pos0] != "N":
            if verbose:
                print(f"  WARNING: {pdb_id} {chain}:{pos1} WT has '{wt_chain[pos0]}' not 'N' — skipping")
            continue
        if wt_triplet != triplet:
            if verbose:
                print(f"  WARNING: {pdb_id} {chain}:{pos1} WT triplet '{wt_triplet}' != expected '{triplet}' — skipping")
            continue

        sequon_destroyed = not has_sequon_at(des_chain, pos0)

        # Graft WT triplet into the reintroduced sequence
        rc = list(reintro_chains[chain_idx])
        rc[pos0] = wt_triplet[0]
        rc[pos0 + 1] = wt_triplet[1]
        rc[pos0 + 2] = wt_triplet[2]
        reintro_chains[chain_idx] = "".join(rc)

        site_info = {
            "chain": chain,
            "chain_idx": chain_idx,
            "position_1idx": pos1,
            "pdb_resnum": pdb_resnum,
            "wt_triplet": wt_triplet,
            "design_triplet": des_triplet,
            "evidence_tier": tier,
            "n_sugars": n_sugars,
            "is_stub": n_sugars == 1,
            "sequon_destroyed": sequon_destroyed,
            "testable": sequon_destroyed,  # only testable if MPNN actually changed it
            "glycan_residues": tree.get("residues", ["NAG"]),
            "glycan_residues_string": tree.get("residues_string", "NAG"),
            "glycan_bonds": tree.get("bonds", []),
        }
        sites.append(site_info)

    n_testable = sum(1 for s in sites if s["testable"])
    n_testable_stubs = sum(1 for s in sites if s["testable"] and s["is_stub"])

    result = {
        "pdb_id": pdb_id,
        "chain_order": chain_order,
        "design_score": design_score,
        "n_glycan_sites": len(sites),
        "n_testable": n_testable,
        "n_testable_stubs": n_testable_stubs,
        "sites": sites,
        "conditions": {
            "wt_with_glycan": {
                "chain_sequences": wt_chains,
                "description": "Wild-type sequence with glycans at PDB-resolved sites",
            },
            "mpnn_reintroduced": {
                "chain_sequences": reintro_chains,
                "description": "Unconstrained MPNN design with WT sequon triplet grafted back",
            },
            "mpnn_unconstrained": {
                "chain_sequences": list(design_chains),
                "description": "Unconstrained MPNN design as-is (no glycans)",
            },
        },
    }

    if verbose:
        print(f"{pdb_id}: {len(sites)} glycan sites, {n_testable} testable "
              f"({n_testable_stubs} stubs), design score={design_score:.4f}")
        for s in sites:
            flag = "TESTABLE" if s["testable"] else "kept"
            stub = "stub" if s["is_stub"] else f"{s['n_sugars']}sug"
            print(f"  {s['chain']}:{s['position_1idx']} (resnum {s['pdb_resnum']}) "
                  f"{s['wt_triplet']}→{s['design_triplet']} [{s['evidence_tier']}] "
                  f"{stub} {flag}")

    return result


def write_clash_sequences(result, output_dir=None):
    """Write clash_sequences.json to the protein's output directory."""
    pdb_id = result["pdb_id"]
    if output_dir is None:
        output_dir = DATA_DIR / "outputs" / f"output_{pdb_id}" / "clash_experiment"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "clash_sequences.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sequence variants for the glycan clash experiment."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb-ids", nargs="+", help="PDB IDs to process")
    group.add_argument("--all", action="store_true", help="Process all eligible proteins")
    group.add_argument("--survey", action="store_true",
                       help="Print testable site counts without writing files")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.survey or args.all:
        pdb_ids = find_eligible_proteins()
    else:
        pdb_ids = [p.upper() for p in args.pdb_ids]

    if args.survey:
        print(f"\n{'PDB':<6} {'GlySites':<10} {'Testable':<10} {'StubTest':<10} {'Score'}")
        print("-" * 55)
        all_results = []
        for pdb_id in pdb_ids:
            r = analyse_protein(pdb_id, verbose=False)
            if r is None:
                continue
            all_results.append(r)
            print(f"{r['pdb_id']:<6} {r['n_glycan_sites']:<10} {r['n_testable']:<10} "
                  f"{r['n_testable_stubs']:<10} {r['design_score']:.4f}")

        # Rank by testable stubs for the pilot
        print(f"\n--- Best candidates for NAG-stub pilot (sorted by stub testable sites) ---")
        ranked = sorted(all_results, key=lambda r: r["n_testable_stubs"], reverse=True)
        for r in ranked[:10]:
            if r["n_testable_stubs"] == 0:
                break
            print(f"  {r['pdb_id']}: {r['n_testable_stubs']} stub-testable sites, "
                  f"{r['n_testable']} total testable")
        return

    for pdb_id in pdb_ids:
        print(f"\n{'='*60}")
        result = analyse_protein(pdb_id)
        if result is None:
            print(f"  Skipping {pdb_id}: missing data")
            continue
        if result["n_testable"] == 0:
            print(f"  Skipping {pdb_id}: no testable sites (MPNN kept all sequons)")
            continue
        write_clash_sequences(result)


if __name__ == "__main__":
    main()
