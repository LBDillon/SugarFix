#!/usr/bin/env python3
"""
STEP 2: Sequon Identification

Identifies all N-X-S/T sequons in the wild-type protein sequence using
ProteinMPNN's own PDB parsing to ensure correct indexing.

Usage:
    python 02_identify_sequons.py --pdb_dir ./results/1EO8

Outputs:
    - sequons/sequons.csv
    - sequons/sequons_by_chain.json
    - sequons/figures/sequon_map.png
    - sequons/figures/sequon_positions.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

try:
    from Bio.PDB import MMCIFParser, PDBParser, is_aa
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# mpnn_utils lives in 02_design/
_DESIGN_DIR = str(Path(__file__).resolve().parent.parent / "02_design")
if _DESIGN_DIR not in sys.path:
    sys.path.insert(0, _DESIGN_DIR)

from mpnn_utils import (
    get_mpnn_chain_seqs_and_order,
    find_sequons,
    verify_sequon_positions,
    SEQUON_REGEX
)


# ---------------------------------------------------------------------------
# Evidence tier annotation helpers
# ---------------------------------------------------------------------------

def _as_list(value):
    """Normalize scalar-or-list mmCIF values into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_pdb_dbref(pdb_path):
    """Parse DBREF-style mappings to build UniProt-to-structure numbering.

    Returns dict: chain_id -> list of dicts with keys:
        unp_start, unp_end, pdb_start, pdb_end
    allowing linear interpolation of UniProt pos -> PDB resnum.
    """
    pdb_path = Path(pdb_path)
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        return parse_mmcif_dbref(pdb_path)

    dbref_ranges = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("DBREF ") or line.startswith("DBREF1") or line.startswith("DBREF2")):
                continue
            if line.startswith("DBREF "):
                # Standard DBREF record (PDB v3.3 format)
                try:
                    chain = line[12:13].strip()
                    pdb_start = int(line[14:18].strip())
                    pdb_end = int(line[20:24].strip())
                    db_name = line[26:32].strip()
                    if db_name not in ("UNP", "SWS", "SP", "TR"):
                        continue
                    unp_start = int(line[55:60].strip())
                    unp_end = int(line[62:67].strip())
                    dbref_ranges.setdefault(chain, []).append({
                        "unp_start": unp_start, "unp_end": unp_end,
                        "pdb_start": pdb_start, "pdb_end": pdb_end,
                    })
                except (ValueError, IndexError):
                    continue
    return dbref_ranges


def parse_mmcif_dbref(cif_path):
    """Parse mmCIF _struct_ref_seq mappings into the DBREF-like format we use."""
    if not BIOPYTHON_AVAILABLE:
        return {}

    mmcif = MMCIF2Dict(str(cif_path))
    chains = _as_list(mmcif.get("_struct_ref_seq.pdbx_strand_id"))
    unp_starts = _as_list(mmcif.get("_struct_ref_seq.db_align_beg"))
    unp_ends = _as_list(mmcif.get("_struct_ref_seq.db_align_end"))
    auth_starts = _as_list(mmcif.get("_struct_ref_seq.pdbx_auth_seq_align_beg"))
    auth_ends = _as_list(mmcif.get("_struct_ref_seq.pdbx_auth_seq_align_end"))
    seq_starts = _as_list(mmcif.get("_struct_ref_seq.seq_align_beg"))
    seq_ends = _as_list(mmcif.get("_struct_ref_seq.seq_align_end"))

    n_rows = max(
        len(chains),
        len(unp_starts),
        len(unp_ends),
        len(auth_starts),
        len(auth_ends),
        len(seq_starts),
        len(seq_ends),
    )
    if n_rows == 0:
        return {}

    dbref_ranges = {}
    for i in range(n_rows):
        chain_field = chains[i] if i < len(chains) else None
        if not chain_field or chain_field in {"?", "."}:
            continue

        try:
            unp_start = int(unp_starts[i])
            unp_end = int(unp_ends[i])
            raw_pdb_start = auth_starts[i] if i < len(auth_starts) else None
            raw_pdb_end = auth_ends[i] if i < len(auth_ends) else None
            if raw_pdb_start in {None, "?", "."} or raw_pdb_end in {None, "?", "."}:
                raw_pdb_start = seq_starts[i]
                raw_pdb_end = seq_ends[i]
            pdb_start = int(raw_pdb_start)
            pdb_end = int(raw_pdb_end)
        except (TypeError, ValueError, IndexError):
            continue

        for chain in str(chain_field).split(","):
            chain = chain.strip()
            if not chain:
                continue
            dbref_ranges.setdefault(chain, []).append(
                {
                    "unp_start": unp_start,
                    "unp_end": unp_end,
                    "pdb_start": pdb_start,
                    "pdb_end": pdb_end,
                }
            )

    return dbref_ranges


def uniprot_pos_to_pdb_resnum(unp_pos, dbref_ranges_for_chain):
    """Convert a UniProt position to a PDB residue number using DBREF ranges.

    Returns (pdb_resnum, chain_id_matched) or (None, None) if no mapping found.
    """
    for rng in dbref_ranges_for_chain:
        if rng["unp_start"] <= unp_pos <= rng["unp_end"]:
            offset = unp_pos - rng["unp_start"]
            return rng["pdb_start"] + offset, True
    return None, False


def build_pdb_resnum_to_mpnn_idx(pdb_path, chain_id):
    """Build mapping from PDB residue number -> MPNN 0-indexed position.

    Uses BioPython to iterate amino acid residues in the same order MPNN does.
    """
    if not BIOPYTHON_AVAILABLE:
        return {}
    pdb_path = Path(pdb_path)
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_path))
    else:
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", str(pdb_path))
        except Exception:
            mmcif_path = pdb_path.with_suffix(".cif")
            if not mmcif_path.exists():
                raise
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("protein", str(mmcif_path))
    mapping = {}
    for model in structure:
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        residues = [r for r in chain.get_residues() if is_aa(r)]
        for idx, residue in enumerate(residues):
            mapping[residue.id[1]] = idx
    return mapping


def load_uniprot_evidence(candidates_csv, pdb_id):
    """Load UniProt glycosylation evidence from candidates CSV.

    Returns dict: uniprot_position (int) -> evidence_tier (str).
    Searches for the row whose pdb_ids column contains the given PDB ID.
    """
    if not candidates_csv or not Path(candidates_csv).exists():
        return {}

    evidence = {}
    with open(candidates_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_ids = row.get("pdb_ids", "")
            if pdb_id.upper() not in pdb_ids.upper():
                continue
            # Found matching row — parse glyco_evidence column
            glyco_ev = row.get("glyco_evidence", "")
            if not glyco_ev:
                # Fall back to glyco_positions (old format without tiers)
                glyco_pos = row.get("glyco_positions", "")
                for pos_str in glyco_pos.split(";"):
                    pos_str = pos_str.strip()
                    if pos_str.isdigit():
                        evidence[int(pos_str)] = "experimental"
                break
            for entry in glyco_ev.split(";"):
                entry = entry.strip()
                if ":" not in entry:
                    continue
                parts = entry.split(":")
                try:
                    pos = int(parts[0])
                    tier = parts[1]
                    evidence[pos] = tier
                except (ValueError, IndexError):
                    continue
            break
    return evidence


def load_glycan_trees(glycan_trees_path):
    """Load glycan trees JSON.

    Returns dict: "chain:pdb_resnum" -> glycan info dict.
    """
    if not glycan_trees_path or not Path(glycan_trees_path).exists():
        return {}
    with open(glycan_trees_path) as f:
        return json.load(f)


def annotate_evidence_tiers(sequons_by_chain, chain_seqs, pdb_path,
                            uniprot_evidence, glycan_trees):
    """Annotate each sequon with an evidence tier.

    Tiers (highest to lowest confidence):
      - experimental: UniProt ECO:0000269 literature evidence at this position
      - pdb_evidence: UniProt ECO:0007744 PDB structural evidence OR
                      resolved glycan tree in PDB LINK records
      - curator_inferred: UniProt ECO:0000305
      - motif_only: N-X-S/T motif match with no external validation

    Modifies sequons_by_chain in place, adding 'evidence_tier' to each sequon dict.
    Also returns a summary dict for logging.
    """
    # Build PDB DBREF mappings for UniProt -> PDB resnum conversion
    dbref_ranges = parse_pdb_dbref(pdb_path) if uniprot_evidence else {}

    summary = {"experimental": 0, "pdb_evidence": 0,
               "curator_inferred": 0, "motif_only": 0}

    for chain_id, sequons in sequons_by_chain.items():
        # Build PDB resnum -> MPNN idx mapping for this chain
        pdb_to_mpnn = build_pdb_resnum_to_mpnn_idx(pdb_path, chain_id)
        # Invert: MPNN idx -> PDB resnum
        mpnn_to_pdb = {v: k for k, v in pdb_to_mpnn.items()}

        # Build set of PDB resnums with resolved glycan trees
        glycan_pdb_resnums = set()
        for key, info in glycan_trees.items():
            if info.get("protein_chain") == chain_id:
                glycan_pdb_resnums.add(info["protein_resnum"])

        chain_dbref = dbref_ranges.get(chain_id, [])

        for s in sequons:
            mpnn_idx = s["position_0idx"]
            pdb_resnum = mpnn_to_pdb.get(mpnn_idx)

            tier = "motif_only"

            # Check 1: PDB resolved glycan tree at this position
            if pdb_resnum is not None and pdb_resnum in glycan_pdb_resnums:
                tier = "pdb_evidence"

            # Check 2: UniProt evidence (can upgrade to experimental)
            if uniprot_evidence and pdb_resnum is not None:
                # Try to find the UniProt position corresponding to this PDB resnum
                # by reverse-mapping through DBREF
                for rng in chain_dbref:
                    if rng["pdb_start"] <= pdb_resnum <= rng["pdb_end"]:
                        offset = pdb_resnum - rng["pdb_start"]
                        unp_pos = rng["unp_start"] + offset
                        unp_tier = uniprot_evidence.get(unp_pos)
                        if unp_tier == "experimental":
                            tier = "experimental"
                        elif unp_tier == "pdb_evidence" and tier == "motif_only":
                            tier = "pdb_evidence"
                        elif unp_tier == "curator_inferred" and tier == "motif_only":
                            tier = "curator_inferred"
                        break

            s["evidence_tier"] = tier
            summary[tier] += 1

    return summary


def create_sequon_map(chains_data, sequons_df, output_dir, pdb_id):
    """Create visual map of sequons on protein sequence."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Get unique chains
    chains_with_sequons = sequons_df['chain_id'].unique()
    all_chains = list(chains_data.keys())

    n_chains = len(all_chains)
    fig_height = max(4, n_chains * 1.2)

    fig, ax = plt.subplots(figsize=(14, fig_height))

    colors = sns.color_palette("Set2", n_chains)
    sequon_color = '#E74C3C'  # Red for sequons

    y_positions = {}
    for i, chain_id in enumerate(sorted(all_chains)):
        y_positions[chain_id] = n_chains - i - 1

    # Draw chains as horizontal bars
    for chain_id, chain_info in chains_data.items():
        y = y_positions[chain_id]
        length = chain_info['length']
        color = colors[list(chains_data.keys()).index(chain_id) % len(colors)]

        # Draw chain backbone
        ax.barh(y, length, height=0.6, color=color, alpha=0.3, edgecolor=color)

        # Add chain label
        ax.text(-10, y, f"Chain {chain_id}", ha='right', va='center', fontsize=11, fontweight='bold')
        ax.text(length + 5, y, f"({length} aa)", ha='left', va='center', fontsize=9, color='gray')

    # Mark sequon positions
    for _, row in sequons_df.iterrows():
        chain_id = row['chain_id']
        pos = row['position_0idx']
        y = y_positions[chain_id]

        # Draw sequon marker
        ax.barh(y, 3, left=pos, height=0.6, color=sequon_color, alpha=0.8)

        # Add position label (only if not too crowded)
        if len(sequons_df[sequons_df['chain_id'] == chain_id]) < 10:
            ax.text(pos + 1.5, y + 0.4, f"{row['sequon']}\n({pos})",
                   ha='center', va='bottom', fontsize=7, color=sequon_color)

    # Customize plot
    ax.set_xlim(-50, max(c['length'] for c in chains_data.values()) + 50)
    ax.set_ylim(-0.5, n_chains - 0.5)
    ax.set_xlabel('Residue Position (0-indexed)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'{pdb_id}: N-X-S/T Sequon Map', fontsize=14, fontweight='bold')

    # Add legend
    chain_patch = mpatches.Patch(color='steelblue', alpha=0.3, label='Protein chain')
    sequon_patch = mpatches.Patch(color=sequon_color, alpha=0.8, label='N-X-S/T sequon')
    ax.legend(handles=[chain_patch, sequon_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(fig_dir / "sequon_map.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Sequon positions bar plot
    if len(sequons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        sequon_counts = sequons_df.groupby('chain_id').size().reindex(all_chains, fill_value=0)

        colors = ['#E74C3C' if c > 0 else '#BDC3C7' for c in sequon_counts]
        bars = ax.bar(sequon_counts.index, sequon_counts.values, color=colors)

        ax.set_xlabel('Chain ID', fontsize=12)
        ax.set_ylabel('Number of Sequons', fontsize=12)
        ax.set_title(f'{pdb_id}: Sequons per Chain', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, count in zip(bars, sequon_counts.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "sequon_counts.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 3: Sequon details table
    if len(sequons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(3, len(sequons_df) * 0.4 + 1)))
        ax.axis('off')

        table_data = sequons_df[['chain_id', 'position_0idx', 'sequon']].copy()
        table_data.columns = ['Chain', 'Position (0-idx)', 'Sequon']

        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colColours=['#E74C3C'] * len(table_data.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        for i in range(len(table_data.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'{pdb_id}: Sequon Details', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(fig_dir / "sequon_table.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 2: Identify sequons')
    parser.add_argument('--pdb_dir', required=True, help='Directory from Step 1')
    parser.add_argument('--candidates-csv', default=None,
                        help='Path to candidates.csv from screen_uniprot_candidates.py '
                             '(provides UniProt glycosylation evidence tiers)')
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    structure_dir = pdb_dir / "structure"
    sequons_dir = pdb_dir / "sequons"
    sequons_dir.mkdir(exist_ok=True)

    # Load structure info
    info_path = structure_dir / "structure_info.json"
    if not info_path.exists():
        print(f"ERROR: Run Step 1 first. Missing: {info_path}")
        return

    with open(info_path) as f:
        structure_info = json.load(f)

    pdb_id = structure_info['pdb_id']

    # Get protein-only PDB path (handle stale absolute paths from old runs)
    asymmetric_unit = structure_info['asymmetric_unit']
    pdb_path = Path(asymmetric_unit.get('protein_only_path', asymmetric_unit.get('path')))
    if not pdb_path.exists():
        # Fall back to expected location relative to pdb_dir
        fallback = structure_dir / f"{pdb_id}_protein.pdb"
        if fallback.exists():
            pdb_path = fallback
        else:
            fallback2 = structure_dir / f"{pdb_id}.pdb"
            if fallback2.exists():
                pdb_path = fallback2

    print("=" * 70)
    print(f"STEP 2: SEQUON IDENTIFICATION - {pdb_id}")
    print("=" * 70)

    # CRITICAL: Use ProteinMPNN's parsing for correct indexing
    print(f"\n1. Parsing structure with ProteinMPNN's parser...")
    print(f"   (This ensures sequon positions match ProteinMPNN's internal indexing)")

    chain_seqs, chain_order = get_mpnn_chain_seqs_and_order(pdb_path)

    if not chain_seqs:
        print(f"ERROR: No chains found in {pdb_path}")
        return

    # Build chains_data for visualization compatibility
    chains_data = {}
    for chain_id in chain_order:
        chains_data[chain_id] = {
            'sequence': chain_seqs[chain_id],
            'length': len(chain_seqs[chain_id])
        }

    print(f"   Found {len(chains_data)} chain(s): {', '.join(chain_order)}")

    print(f"\n2. Scanning for N-X-S/T sequons...")

    # Find sequons in each chain
    all_sequons = []
    sequons_by_chain = {}

    for chain_id in chain_order:
        sequence = chain_seqs[chain_id]
        sequons = find_sequons(sequence)

        sequons_by_chain[chain_id] = [
            {'position_0idx': s['position_0idx'], 'sequon': s['sequon']}
            for s in sequons
        ]

        for s in sequons:
            s['chain_id'] = chain_id
            s['position_1idx'] = s['position_0idx'] + 1
            s['n_residue'] = s['sequon'][0]
            s['x_residue'] = s['sequon'][1]
            s['st_residue'] = s['sequon'][2]
            all_sequons.append(s)

        print(f"  Chain {chain_id}: {len(sequons)} sequon(s)")
        for s in sequons:
            print(f"    Position {s['position_0idx']} (1-idx: {s['position_0idx']+1}): {s['sequon']}")

    # CRITICAL: Verify positions are correct
    print(f"\n3. Verifying sequon positions...")
    try:
        verify_sequon_positions(chain_seqs, sequons_by_chain, pdb_id)
        print("   All sequon positions verified successfully")
    except AssertionError as e:
        print(f"   ERROR: Position verification failed: {e}")
        return

    # --- Evidence tier annotation ---
    print(f"\n4. Annotating evidence tiers...")

    # Load UniProt evidence from candidates CSV
    uniprot_evidence = load_uniprot_evidence(args.candidates_csv, pdb_id)
    if uniprot_evidence:
        print(f"   Loaded {len(uniprot_evidence)} UniProt glycosylation site(s)")
    else:
        print(f"   No UniProt evidence available (pass --candidates-csv to enable)")

    # Load PDB glycan trees
    glycan_trees_path = structure_dir / "glycan_trees.json"
    glycan_trees = load_glycan_trees(glycan_trees_path)
    if glycan_trees:
        print(f"   Loaded {len(glycan_trees)} PDB glycan tree(s)")
    else:
        print(f"   No PDB glycan trees found")

    # Use raw PDB for DBREF records (protein-only PDB may lack them)
    raw_pdb_path = structure_dir / f"{pdb_id}.pdb"
    dbref_pdb = raw_pdb_path if raw_pdb_path.exists() else pdb_path

    # Annotate each sequon with evidence tier
    tier_summary = annotate_evidence_tiers(
        sequons_by_chain, chain_seqs, dbref_pdb,
        uniprot_evidence, glycan_trees,
    )

    # Copy evidence_tier back into all_sequons list
    for chain_id, sequons in sequons_by_chain.items():
        for s in sequons:
            # Find matching entry in all_sequons and add the tier
            for a in all_sequons:
                if a.get('chain_id') == chain_id and a['position_0idx'] == s['position_0idx']:
                    a['evidence_tier'] = s.get('evidence_tier', 'motif_only')
                    break

    print(f"   Evidence tier summary:")
    for tier, count in tier_summary.items():
        if count > 0:
            print(f"     {tier}: {count}")

    # Create DataFrame
    sequons_df = pd.DataFrame(all_sequons)

    # Save outputs
    print(f"\n5. Saving outputs...")

    if len(sequons_df) > 0:
        csv_path = sequons_dir / "sequons.csv"
        sequons_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    json_path = sequons_dir / "sequons_by_chain.json"
    with open(json_path, 'w') as f:
        json.dump(sequons_by_chain, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save chain order for downstream scripts
    chain_order_path = sequons_dir / "mpnn_chain_order.json"
    with open(chain_order_path, 'w') as f:
        json.dump({"chain_order": chain_order}, f, indent=2)
    print(f"  Saved: {chain_order_path}")

    # Create visualizations
    print(f"\n6. Creating visualizations...")
    if len(sequons_df) > 0:
        create_sequon_map(chains_data, sequons_df, sequons_dir, pdb_id)
    else:
        print("  No sequons found - skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"\nTotal sequons found: {len(sequons_df)}")

    chains_with_sequons = sequons_df['chain_id'].nunique() if len(sequons_df) > 0 else 0
    print(f"Chains with sequons: {chains_with_sequons}/{len(chains_data)}")

    if 'evidence_tier' in sequons_df.columns:
        print(f"\nEvidence tier breakdown:")
        for tier, count in sequons_df['evidence_tier'].value_counts().items():
            print(f"  {tier}: {count}")
        n_validated = len(sequons_df[sequons_df['evidence_tier'].isin(
            ['experimental', 'pdb_evidence'])])
        print(f"  Validated (experimental + pdb_evidence): {n_validated}/{len(sequons_df)}")

    print(f"\nOutputs in {sequons_dir}/:")
    print(f"  - sequons.csv (with evidence_tier column)")
    print(f"  - sequons_by_chain.json")
    print(f"  - figures/sequon_map.png")
    print(f"  - figures/sequon_counts.png")
    print(f"  - figures/sequon_table.png")

    print(f"\n→ Next: bash 02_design/run_designs.sh <PDB_ID>")


if __name__ == "__main__":
    main()
