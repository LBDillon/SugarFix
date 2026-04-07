#!/usr/bin/env python3
"""
Generate ProteinMPNN designs with sequon constraints for any PDB.

Conditions:
1. unconstrained
2. n_only_fixed
3. full_sequon_fixed
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"
PROJECT_DIR = PIPELINE_ROOT.parent
PROTEINMPNN_DIR = Path(
    os.environ.get("PROTEINMPNN_DIR", str(PIPELINE_ROOT / "ProteinMPNN"))
).resolve()
if not (PROTEINMPNN_DIR / "protein_mpnn_utils.py").exists():
    fallback_dir = PROJECT_DIR / "ProteinMPNN"
    if (fallback_dir / "protein_mpnn_utils.py").exists():
        PROTEINMPNN_DIR = fallback_dir
    env_runner = os.environ.get("PROTEINMPNN_PATH")
    if env_runner:
        candidate = Path(env_runner).resolve().parent
        if (candidate / "protein_mpnn_utils.py").exists():
            PROTEINMPNN_DIR = candidate
sys.path.insert(0, str(PROTEINMPNN_DIR))

from protein_mpnn_utils import parse_PDB

from mpnn_utils import find_sequons as _find_sequons_mpnn_utils, SEQUON_REGEX

try:
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


def verify_indexing(pdb_path: Path, mpnn_seqs: dict[str, str], chain_order: list[str]):
    """Cross-check MPNN sequential indexing against BioPython PDB residue parsing.

    Flags any mismatches between the two parsers so downstream analysis
    (which may use BioPython for RSA/B-factors) stays consistent.
    """
    if not BIOPYTHON_AVAILABLE:
        print("\n  INDEXING CHECK: skipped (BioPython not installed)")
        return True

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    all_ok = True
    print("\n  INDEXING CHECK:")
    for chain_id in chain_order:
        mpnn_seq = mpnn_seqs[chain_id]

        # Extract BioPython sequence for this chain
        bio_residues = []
        for model in structure:
            if chain_id not in [c.id for c in model]:
                continue
            chain = model[chain_id]
            bio_residues = [r for r in chain.get_residues() if is_aa(r)]
            break

        if not bio_residues:
            print(f"    Chain {chain_id}: WARNING - not found in BioPython parse")
            all_ok = False
            continue

        bio_seq = "".join(seq1(r.get_resname()) for r in bio_residues)

        if bio_seq == mpnn_seq:
            pdb_start = bio_residues[0].id[1]
            pdb_end = bio_residues[-1].id[1]
            print(f"    Chain {chain_id}: OK (len={len(mpnn_seq)}, "
                  f"PDB resnum {pdb_start}-{pdb_end})")
        else:
            print(f"    Chain {chain_id}: MISMATCH!")
            print(f"      MPNN  len={len(mpnn_seq)}: {mpnn_seq[:20]}...")
            print(f"      BioPy len={len(bio_seq)}: {bio_seq[:20]}...")
            # Find first difference
            for i, (a, b) in enumerate(zip(mpnn_seq, bio_seq)):
                if a != b:
                    print(f"      First diff at index {i}: MPNN='{a}' vs BioPy='{b}'")
                    break
            all_ok = False

    if all_ok:
        print("    All chains consistent between MPNN and BioPython parsers.")
    else:
        print("    WARNING: Indexing mismatches detected! Fixed positions may be incorrect.")

    return all_ok


def get_mpnn_chain_sequences(pdb_path: Path):
    """Extract chain sequences using ProteinMPNN's parser."""
    pdb_dict_list = parse_PDB(str(pdb_path))
    if not pdb_dict_list:
        raise ValueError(f"Failed to parse PDB: {pdb_path}")

    pdb_dict = pdb_dict_list[0]
    chain_seqs = {}
    chain_order = []

    for key in sorted(pdb_dict.keys()):
        if key.startswith("seq_chain_"):
            chain_id = key.replace("seq_chain_", "")
            seq = pdb_dict[key]
            if seq:
                chain_seqs[chain_id] = seq
                chain_order.append(chain_id)

    return chain_seqs, chain_order


def find_sequons(sequence: str):
    """Find N-X-S/T sequons (0-indexed), excluding N-P-S/T and N-X(unresolved)-S/T.

    Uses canonical SEQUON_REGEX from mpnn_utils.
    """
    return [
        {"position_0idx": s["position_0idx"], "triplet": s["sequon"]}
        for s in _find_sequons_mpnn_utils(sequence)
    ]


def create_fixed_positions_jsonl(pdb_name, chain_positions, all_chain_ids, output_file):
    """Create fixed_positions JSON for ProteinMPNN (1-indexed positions)."""
    payload = {pdb_name: {}}

    for chain_id in all_chain_ids:
        if chain_id in chain_positions and chain_positions[chain_id]:
            payload[pdb_name][chain_id] = sorted([p + 1 for p in chain_positions[chain_id]])
        else:
            payload[pdb_name][chain_id] = []

    with open(output_file, "w") as f:
        json.dump(payload, f)

    return payload


def run_proteinmpnn(pdb_path: Path,
                    output_dir: Path,
                    proteinmpnn_path: Path,
                    fixed_pos_jsonl: Path | None,
                    chains: list[str],
                    num_seqs: int,
                    temp: float,
                    seed: int):
    """Run ProteinMPNN for one condition."""
    cmd = [
        "python",
        str(proteinmpnn_path),
        "--pdb_path", str(pdb_path),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temp),
        "--out_folder", str(output_dir),
        "--seed", str(seed),
    ]
    # Force a single multi-chain complex run (not per-chain independent runs).
    cmd.extend(["--pdb_path_chains", " ".join(chains)])

    if fixed_pos_jsonl:
        cmd.extend(["--fixed_positions_jsonl", str(fixed_pos_jsonl)])

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(proteinmpnn_path.parent),
        )
        return True, None
    except subprocess.CalledProcessError as exc:
        err = exc.stderr[:1000] if exc.stderr else str(exc)
        return False, err


def count_fasta_entries(fasta_path: Path):
    """Count FASTA headers."""
    n = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def default_pdb_path(pipeline_dir: Path, pdb_id: str):
    """Best-effort lookup for an existing prepared protein PDB."""
    candidates = [
        pipeline_dir / "data" / "prep" / pdb_id / "structure" / f"{pdb_id}_protein.pdb",
        pipeline_dir / "data" / "prep" / pdb_id / "structure" / f"{pdb_id}.pdb",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3-condition ProteinMPNN designs for any PDB")
    parser.add_argument("--pdb-id", required=True, help="PDB code, e.g. 1RUZ or 5EQG")
    parser.add_argument(
        "--pdb-path",
        help="Path to input protein PDB (default: prep/<PDB_ID>/structure/<PDB_ID>_protein.pdb)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: case_study_pipeline/output_<PDB_ID>)",
    )
    parser.add_argument("--pdb-label", help="Label/stem used for output files (default: input PDB filename stem)")
    parser.add_argument("--proteinmpnn-path", default=str(PROTEINMPNN_DIR / "protein_mpnn_run.py"), help="Path to protein_mpnn_run.py")
    parser.add_argument("--num-seqs", type=int, default=64, help="Number of sequences per condition")
    parser.add_argument("--sampling-temp", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    pdb_id = args.pdb_id.upper()
    pdb_path = Path(args.pdb_path) if args.pdb_path else default_pdb_path(PIPELINE_ROOT, pdb_id)
    output_dir = Path(args.output_dir) if args.output_dir else (DATA_DIR / "outputs" / f"output_{pdb_id}")
    proteinmpnn_path = Path(args.proteinmpnn_path)

    if not pdb_path.exists():
        print(f"ERROR: input PDB not found: {pdb_path}")
        print(f"Tip: run {PIPELINE_ROOT / '01_preparation' / 'prepare_structure.sh'} {pdb_id}, or provide --pdb-path")
        return 1

    if not proteinmpnn_path.exists():
        print(f"ERROR: ProteinMPNN runner not found: {proteinmpnn_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_label = args.pdb_label or pdb_path.stem
    output_pdb = output_dir / f"{pdb_label}.pdb"
    if pdb_path.resolve() != output_pdb.resolve():
        shutil.copy(pdb_path, output_pdb)

    print("=" * 70)
    print(f"GENERATING DESIGNS FOR {pdb_id}")
    print("=" * 70)
    print(f"Input PDB: {pdb_path}")
    print(f"Working PDB: {output_pdb}")
    print(f"Output dir: {output_dir}")

    chain_seqs, chain_order = get_mpnn_chain_sequences(output_pdb)
    print(f"\nFound {len(chain_order)} chains: {chain_order}")
    print("Design mode: joint multi-chain complex design")

    verify_indexing(output_pdb, chain_seqs, chain_order)

    sequons_by_chain = {}
    total_sequons = 0
    for chain_id in chain_order:
        sequons = find_sequons(chain_seqs[chain_id])
        if sequons:
            sequons_by_chain[chain_id] = sequons
            total_sequons += len(sequons)
            printable = [f"{s['position_0idx'] + 1}:{s['triplet']}" for s in sequons]
            print(f"  Chain {chain_id}: {', '.join(printable)}")

    print(f"Total sequons found: {total_sequons}")

    # Condition 1: unconstrained
    unconstrained_dir = output_dir / "unconstrained"
    unconstrained_dir.mkdir(exist_ok=True)
    ok, err = run_proteinmpnn(
        output_pdb,
        unconstrained_dir,
        proteinmpnn_path,
        fixed_pos_jsonl=None,
        chains=chain_order,
        num_seqs=args.num_seqs,
        temp=args.sampling_temp,
        seed=args.seed,
    )
    if not ok:
        print(f"ERROR (unconstrained): {err}")
        return 1

    # Condition 2: N-only fixed
    n_only_positions = {}
    for chain_id, sequons in sequons_by_chain.items():
        n_only_positions[chain_id] = [s["position_0idx"] for s in sequons]

    n_only_file = output_dir / "fixed_positions_n_only.jsonl"
    create_fixed_positions_jsonl(pdb_label, n_only_positions, chain_order, n_only_file)

    n_only_dir = output_dir / "n_only_fixed"
    n_only_dir.mkdir(exist_ok=True)
    ok, err = run_proteinmpnn(
        output_pdb,
        n_only_dir,
        proteinmpnn_path,
        fixed_pos_jsonl=n_only_file,
        chains=chain_order,
        num_seqs=args.num_seqs,
        temp=args.sampling_temp,
        seed=args.seed,
    )
    if not ok:
        print(f"ERROR (n_only_fixed): {err}")
        return 1

    # Condition 3: full sequon fixed
    full_positions = {}
    for chain_id, sequons in sequons_by_chain.items():
        expanded = []
        for s in sequons:
            p = s["position_0idx"]
            expanded.extend([p, p + 1, p + 2])
        full_positions[chain_id] = sorted(set(expanded))

    full_file = output_dir / "fixed_positions_full_sequon.jsonl"
    create_fixed_positions_jsonl(pdb_label, full_positions, chain_order, full_file)

    full_dir = output_dir / "full_sequon_fixed"
    full_dir.mkdir(exist_ok=True)
    ok, err = run_proteinmpnn(
        output_pdb,
        full_dir,
        proteinmpnn_path,
        fixed_pos_jsonl=full_file,
        chains=chain_order,
        num_seqs=args.num_seqs,
        temp=args.sampling_temp,
        seed=args.seed,
    )
    if not ok:
        print(f"ERROR (full_sequon_fixed): {err}")
        return 1

    print("\n" + "=" * 70)
    print("DESIGN GENERATION COMPLETE")
    print("=" * 70)

    for condition in ["unconstrained", "n_only_fixed", "full_sequon_fixed"]:
        fa_path = output_dir / condition / "seqs" / f"{pdb_label}.fa"
        if fa_path.exists():
            n_entries = count_fasta_entries(fa_path)
            # First entry is WT, remaining are designs.
            print(f"  {condition}: {fa_path} ({max(n_entries - 1, 0)} designs)")
        else:
            print(f"  {condition}: FASTA not found")

    print("\nNext step:")
    print(
        "  python 03_analysis/analyze_case_designs.py "
        f"--pdb-id {pdb_id} --designs-dir {output_dir} --pdb-path {output_pdb}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
