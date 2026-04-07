#!/usr/bin/env python3
"""
Extract per-position N retention data for all 81 pipeline proteins.

Generates two CSVs equivalent to the old dataset's:
  - all_n_positions_with_rsa.csv  (per-position, per-condition)
  - all_per_condition_summary.csv (per-protein summary)

For each protein:
  1. Parse WT FASTA to find all N positions (sequon and non-sequon)
  2. Parse design FASTAs to compute retention at each N position
  3. Get RSA from structural_context.csv
  4. Compute secondary structure / phi-psi from PDB via pydssp
  5. Extract MPNN scores per condition
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

PIPE = Path(__file__).resolve().parent.parent
DATA = PIPE / "data"
PREP = DATA / "prep"
OUTPUTS = DATA / "outputs"
CROSS = DATA / "cross_protein_comparison"
OUT = CROSS  # output destination

# The 81 proteins from the summary
SUMMARY = pd.read_csv(CROSS / "protein_comparison_summary.csv")
PDB_IDS = sorted(SUMMARY["PDB"].tolist())

CONDITIONS = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]


def parse_wt_and_designs(pdb_id, condition):
    """Parse FASTA file to get WT sequence and design sequences + scores.

    Returns:
        wt_chains: list of (chain_id, sequence_with_X) per chain
        designs: list of dicts with 'chains' (list of seqs), 'score', 'recovery'
    """
    fa_dir = OUTPUTS / f"output_{pdb_id}" / condition / "seqs"
    if not fa_dir.exists():
        return None, None

    fa_files = sorted(fa_dir.glob("*.fa"))
    if not fa_files:
        return None, None

    # Load chain order
    chain_order_path = PREP / pdb_id / "sequons" / "mpnn_chain_order.json"
    if chain_order_path.exists():
        with open(chain_order_path) as f:
            chain_order = json.load(f)["chain_order"]
    else:
        chain_order = None

    wt_chains = None
    designs = []

    for fa_file in fa_files:
        with open(fa_file) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(">"):
                header = line
                i += 1
                if i < len(lines):
                    seq = lines[i].strip()
                    i += 1
                else:
                    break

                chain_seqs = seq.split("/")

                if wt_chains is None and "sample=" not in header:
                    # This is the WT line
                    if chain_order and len(chain_order) == len(chain_seqs):
                        wt_chains = list(zip(chain_order, chain_seqs))
                    else:
                        wt_chains = [(chr(65 + j), s) for j, s in enumerate(chain_seqs)]
                elif "sample=" in header:
                    # Design line - extract score and recovery
                    score = None
                    recovery = None
                    for part in header.split(","):
                        part = part.strip()
                        if part.startswith("score="):
                            try:
                                score = float(part.replace("score=", ""))
                            except ValueError:
                                pass
                        elif part.startswith("seq_recovery="):
                            try:
                                recovery = float(part.replace("seq_recovery=", ""))
                            except ValueError:
                                pass

                    designs.append({
                        "chains": chain_seqs,
                        "score": score,
                        "recovery": recovery,
                    })
            else:
                i += 1

    return wt_chains, designs


def identify_n_positions(wt_chains, sequons_by_chain):
    """Find all N positions in WT sequence, classify as sequon vs non-sequon.

    Returns list of dicts with:
        chain, chain_idx, pos_in_chain (0-indexed in MPNN seq including X),
        is_sequon, wt_triplet (3-char context if available)
    """
    positions = []

    for chain_idx, (chain_id, seq) in enumerate(wt_chains):
        # Get sequon positions for this chain
        chain_sequons = set()
        if chain_id in sequons_by_chain:
            for sq in sequons_by_chain[chain_id]:
                chain_sequons.add(sq["position_0idx"])

        for pos, aa in enumerate(seq):
            if aa == "N":
                # Check if this is a sequon N
                is_sequon = pos in chain_sequons

                # Get triplet context
                triplet = seq[pos:pos+3] if pos + 2 < len(seq) else seq[pos:]

                # Verify sequon status from sequence
                # A sequon is N-X-S/T where X != P
                seq_is_sequon = False
                if len(triplet) >= 3:
                    x_pos = triplet[1]
                    st_pos = triplet[2]
                    if x_pos != "P" and x_pos != "X" and st_pos in ("S", "T"):
                        seq_is_sequon = True

                # Use the JSON definition as primary, but also check sequence
                is_sequon = is_sequon or seq_is_sequon

                positions.append({
                    "chain": chain_id,
                    "chain_idx": chain_idx,
                    "pos_in_chain": pos,
                    "is_sequon": is_sequon,
                    "wt_triplet": triplet[:3] if len(triplet) >= 3 else triplet,
                })

    return positions


def compute_retention_at_positions(n_positions, wt_chains, designs):
    """For each N position, compute retention stats across designs.

    Returns updated positions with:
        n_retained (count), n_total, n_retention_pct,
        exact_sequon_retained, exact_sequon_pct,
        functional_retained, functional_pct
    For sequon positions also: x_retained, st_retained counts
    """
    results = []

    for npos in n_positions:
        chain_idx = npos["chain_idx"]
        pos = npos["pos_in_chain"]
        is_sequon = npos["is_sequon"]
        wt_triplet = npos["wt_triplet"]

        n_retained = 0
        exact_retained = 0
        functional_retained = 0
        x_retained = 0  # position+1 retained
        st_retained = 0  # position+2 retained
        n_total = 0

        for design in designs:
            if chain_idx >= len(design["chains"]):
                continue

            des_seq = design["chains"][chain_idx]
            if pos >= len(des_seq):
                continue

            n_total += 1

            # N retention
            if des_seq[pos] == "N":
                n_retained += 1

            if is_sequon and len(wt_triplet) >= 3:
                # Exact triplet match
                des_triplet = des_seq[pos:pos+3] if pos + 2 < len(des_seq) else ""
                if len(des_triplet) >= 3:
                    if des_triplet == wt_triplet:
                        exact_retained += 1

                    # Functional sequon: N-X-S/T where X != P
                    if (des_triplet[0] == "N" and
                        des_triplet[1] != "P" and des_triplet[1] != "X" and
                        des_triplet[2] in ("S", "T")):
                        functional_retained += 1

                    # X position retention (position+1)
                    if des_triplet[1] == wt_triplet[1]:
                        x_retained += 1

                    # S/T position retention (position+2)
                    if des_triplet[2] == wt_triplet[2]:
                        st_retained += 1

        result = {
            **npos,
            "n_retained": n_retained,
            "n_total": n_total,
            "n_retention_pct": (n_retained / n_total * 100) if n_total > 0 else np.nan,
        }

        if is_sequon and n_total > 0:
            result["exact_sequon_retained"] = exact_retained
            result["exact_sequon_pct"] = exact_retained / n_total * 100
            result["functional_retained"] = functional_retained
            result["functional_pct"] = functional_retained / n_total * 100
            result["x_retained"] = x_retained
            result["x_retention_pct"] = x_retained / n_total * 100
            result["st_retained"] = st_retained
            result["st_retention_pct"] = st_retained / n_total * 100
        else:
            result["exact_sequon_retained"] = np.nan
            result["exact_sequon_pct"] = np.nan
            result["functional_retained"] = np.nan
            result["functional_pct"] = np.nan
            result["x_retained"] = np.nan
            result["x_retention_pct"] = np.nan
            result["st_retained"] = np.nan
            result["st_retention_pct"] = np.nan

        results.append(result)

    return results


def get_rsa_for_positions(pdb_id, n_positions, wt_chains):
    """Get RSA values from structural_context.csv for sequon positions,
    and compute RSA for non-sequon positions from PDB if possible.
    """
    # Load existing structural context (sequon positions only)
    sc_path = OUTPUTS / f"output_{pdb_id}" / "structural_context.csv"
    sc_df = None
    if sc_path.exists():
        sc_df = pd.read_csv(sc_path)

    # Try to load pooled data for this protein
    pooled_path = CROSS / "pooled_structural_context.csv"
    pooled_df = None
    if pooled_path.exists():
        pooled_full = pd.read_csv(pooled_path)
        pooled_df = pooled_full[pooled_full["pdb_id"] == pdb_id]

    for npos in n_positions:
        npos["rsa"] = np.nan

        # Try structural_context first (has 1-indexed positions)
        if sc_df is not None and npos["is_sequon"]:
            match = sc_df[
                (sc_df["chain"] == npos["chain"]) &
                (sc_df["position_1idx"] == npos["pos_in_chain"] + 1)
            ]
            if len(match) > 0:
                npos["rsa"] = match.iloc[0]["rsa"]
                continue

        # Try pooled structural context
        if pooled_df is not None and len(pooled_df) > 0:
            match = pooled_df[
                (pooled_df["chain"] == npos["chain"]) &
                (pooled_df["position_1idx"] == npos["pos_in_chain"] + 1)
            ]
            if len(match) > 0:
                npos["rsa"] = match.iloc[0]["rsa"]

    return n_positions


def compute_geometry(pdb_id, n_positions, wt_chains):
    """Compute secondary structure and phi/psi for N positions using pydssp."""
    try:
        import pydssp
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import PPBuilder, three_to_index, index_to_one
    except ImportError:
        print("  Warning: pydssp or Biopython not available, skipping geometry")
        for npos in n_positions:
            npos["ss"] = None
            npos["phi"] = None
            npos["psi"] = None
            npos["rama_region"] = None
        return n_positions

    def three_to_one(resname):
        try:
            return index_to_one(three_to_index(resname))
        except Exception:
            return "X"

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    # Find PDB file
    struct_dir = PREP / pdb_id / "structure"
    pdb_path = struct_dir / f"{pdb_id}_protein.pdb"
    if not pdb_path.exists():
        pdb_path = struct_dir / f"{pdb_id}_bioassembly.pdb"
    if not pdb_path.exists():
        pdb_path = struct_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        for npos in n_positions:
            npos["ss"] = None
            npos["phi"] = None
            npos["psi"] = None
            npos["rama_region"] = None
        return n_positions

    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
        model = structure[0]
    except Exception:
        for npos in n_positions:
            npos["ss"] = None
            npos["phi"] = None
            npos["psi"] = None
            npos["rama_region"] = None
        return n_positions

    # Build per-chain data
    chain_data = {}
    for chain_obj in model:
        cid = chain_obj.get_id()
        residues = []
        coords_list = []

        for residue in chain_obj:
            if residue.get_id()[0] != " ":
                continue
            atoms = {}
            for atom in residue:
                if atom.get_name() in ["N", "CA", "C", "O"]:
                    atoms[atom.get_name()] = atom.get_vector().get_array()
            if len(atoms) == 4:
                aa = three_to_one(residue.get_resname())
                residues.append({
                    "resnum": residue.get_id()[1],
                    "resname": residue.get_resname(),
                    "aa": aa,
                })
                coords_list.append([atoms["N"], atoms["CA"], atoms["C"], atoms["O"]])

        if len(coords_list) < 3:
            continue

        coords_arr = np.array(coords_list, dtype=np.float32)
        try:
            ss_arr = pydssp.assign(coords_arr, out_type="c3")
        except Exception:
            ss_arr = None

        # Phi/psi
        phi_psi_map = {}
        for pp in ppb.build_peptides(chain_obj):
            pp_phi_psi = pp.get_phi_psi_list()
            for i, (phi, psi) in enumerate(pp_phi_psi):
                res = pp[i]
                resnum = res.get_id()[1]
                phi_psi_map[resnum] = (phi, psi)

        chain_data[cid] = {
            "residues": residues,
            "ss": ss_arr,
            "phi_psi": phi_psi_map,
        }

    # Map MPNN positions to PDB residues
    # Build mapping: for each chain, map MPNN 0-indexed position to PDB residue index
    for npos in n_positions:
        npos["ss"] = None
        npos["phi"] = None
        npos["psi"] = None
        npos["rama_region"] = None

        chain_id = npos["chain"]
        if chain_id not in chain_data:
            continue

        cd = chain_data[chain_id]
        mpnn_pos = npos["pos_in_chain"]

        # The MPNN sequence includes X for non-standard residues
        # We need to map: MPNN position (including X) -> PDB residue index (excluding X)
        wt_seq = None
        for cid, seq in wt_chains:
            if cid == chain_id:
                wt_seq = seq
                break

        if wt_seq is None:
            continue

        # Count non-X positions up to mpnn_pos
        pdb_idx = 0
        for i in range(mpnn_pos):
            if i < len(wt_seq) and wt_seq[i] != "X":
                pdb_idx += 1

        # pdb_idx is now the index into chain_data residues
        if pdb_idx >= len(cd["residues"]):
            continue

        # Verify amino acid match
        pdb_aa = cd["residues"][pdb_idx]["aa"]
        if pdb_aa != "N":
            # Try direct match
            if mpnn_pos < len(cd["residues"]) and cd["residues"][mpnn_pos]["aa"] == "N":
                pdb_idx = mpnn_pos
            else:
                continue

        # Secondary structure
        if cd["ss"] is not None and pdb_idx < len(cd["ss"]):
            ss_raw = cd["ss"][pdb_idx]
            npos["ss"] = {"H": "Helix", "E": "Sheet", "-": "Coil"}.get(ss_raw, "Coil")

        # Phi/psi
        resnum = cd["residues"][pdb_idx]["resnum"]
        pp_data = cd["phi_psi"].get(resnum)
        if pp_data is not None:
            phi, psi = pp_data
            if phi is not None and psi is not None:
                npos["phi"] = np.degrees(phi)
                npos["psi"] = np.degrees(psi)

                phi_d = npos["phi"]
                psi_d = npos["psi"]
                if -160 < phi_d < -20 and -80 < psi_d < 0:
                    npos["rama_region"] = "Alpha-helix"
                elif -180 < phi_d < -20 and 50 < psi_d < 180:
                    npos["rama_region"] = "Beta-sheet"
                elif 0 < phi_d < 180:
                    npos["rama_region"] = "Left-handed"
                else:
                    npos["rama_region"] = "Other"

    return n_positions


def extract_mpnn_scores(pdb_id, condition, designs):
    """Extract mean MPNN score from designs."""
    scores = [d["score"] for d in designs if d["score"] is not None]
    if scores:
        return np.mean(scores), np.std(scores), len(scores)
    return np.nan, np.nan, 0


def process_protein(pdb_id):
    """Process a single protein: extract all per-position data."""
    # Load sequon definitions
    sequon_path = PREP / pdb_id / "sequons" / "sequons_by_chain.json"
    if not sequon_path.exists():
        return None, None

    with open(sequon_path) as f:
        sequons_by_chain = json.load(f)

    all_position_rows = []
    score_rows = []

    for condition in CONDITIONS:
        wt_chains, designs = parse_wt_and_designs(pdb_id, condition)
        if wt_chains is None or designs is None or len(designs) == 0:
            continue

        # Find all N positions
        n_positions = identify_n_positions(wt_chains, sequons_by_chain)

        # Compute retention
        n_positions = compute_retention_at_positions(n_positions, wt_chains, designs)

        # Get RSA (only need once, but ok to repeat)
        if condition == "unconstrained":
            n_positions = get_rsa_for_positions(pdb_id, n_positions, wt_chains)
            # Compute geometry only for unconstrained (same structure for all conditions)
            n_positions = compute_geometry(pdb_id, n_positions, wt_chains)

        # MPNN scores
        mean_score, sd_score, n_designs = extract_mpnn_scores(pdb_id, condition, designs)
        score_rows.append({
            "pdb_id": pdb_id,
            "condition": condition,
            "mean_score": mean_score,
            "sd_score": sd_score,
            "n_designs": n_designs,
        })

        for npos in n_positions:
            row = {
                "pdb_id": pdb_id,
                "condition": condition,
                "chain": npos["chain"],
                "position": npos["pos_in_chain"],
                "is_sequon": npos["is_sequon"],
                "wt_triplet": npos["wt_triplet"],
                "n_retained": npos.get("n_retained", 0),
                "n_total": npos.get("n_total", 0),
                "n_retention_pct": npos.get("n_retention_pct", np.nan),
                "exact_sequon_pct": npos.get("exact_sequon_pct", np.nan),
                "functional_pct": npos.get("functional_pct", np.nan),
                "x_retention_pct": npos.get("x_retention_pct", np.nan),
                "st_retention_pct": npos.get("st_retention_pct", np.nan),
                "rsa": npos.get("rsa", np.nan),
            }
            # Add geometry only for unconstrained condition
            if condition == "unconstrained":
                row["ss"] = npos.get("ss")
                row["phi"] = npos.get("phi")
                row["psi"] = npos.get("psi")
                row["rama_region"] = npos.get("rama_region")

            all_position_rows.append(row)

    return all_position_rows, score_rows


def build_per_protein_summary(positions_df):
    """Build per-protein summary from per-position data."""
    rows = []

    for (pdb_id, condition), grp in positions_df.groupby(["pdb_id", "condition"]):
        sequon = grp[grp["is_sequon"] == True]
        nonsequon = grp[grp["is_sequon"] == False]

        row = {
            "pdb_id": pdb_id,
            "condition": condition,
            "n_sequon_positions": len(sequon),
            "n_nonsequon_positions": len(nonsequon),
        }

        # Sequon N retention
        if len(sequon) > 0 and sequon["n_retention_pct"].notna().any():
            row["sequon_N_exact_retention_pct"] = sequon["n_retention_pct"].mean()
        else:
            row["sequon_N_exact_retention_pct"] = np.nan

        # Non-sequon N retention
        if len(nonsequon) > 0 and nonsequon["n_retention_pct"].notna().any():
            row["nonsequon_N_exact_retention_pct"] = nonsequon["n_retention_pct"].mean()
        else:
            row["nonsequon_N_exact_retention_pct"] = np.nan

        # Sequon X retention
        if len(sequon) > 0 and sequon["x_retention_pct"].notna().any():
            row["sequon_X_exact_retention_pct"] = sequon["x_retention_pct"].mean()
        else:
            row["sequon_X_exact_retention_pct"] = np.nan

        # Sequon S/T retention
        if len(sequon) > 0 and sequon["st_retention_pct"].notna().any():
            row["sequon_ST_exact_retention_pct"] = sequon["st_retention_pct"].mean()
        else:
            row["sequon_ST_exact_retention_pct"] = np.nan

        # Overall N retention (all positions)
        all_ret = grp["n_retention_pct"].dropna()
        row["overall_retention_mean_pct"] = all_ret.mean() if len(all_ret) > 0 else np.nan

        # Functional sequon retention
        if len(sequon) > 0 and sequon["functional_pct"].notna().any():
            row["sequon_functional_retention_pct"] = sequon["functional_pct"].mean()
        else:
            row["sequon_functional_retention_pct"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    all_positions = []
    all_scores = []

    success = 0
    failed = 0

    for i, pdb_id in enumerate(PDB_IDS):
        print(f"[{i+1}/{len(PDB_IDS)}] Processing {pdb_id}...", end=" ", flush=True)

        try:
            pos_rows, score_rows = process_protein(pdb_id)
            if pos_rows is not None:
                all_positions.extend(pos_rows)
                all_scores.extend(score_rows)
                n_seq = sum(1 for p in pos_rows if p["is_sequon"] and p["condition"] == "unconstrained")
                n_nonseq = sum(1 for p in pos_rows if not p["is_sequon"] and p["condition"] == "unconstrained")
                print(f"OK ({n_seq} sequon, {n_nonseq} non-sequon N positions)")
                success += 1
            else:
                print("SKIP (no data)")
                failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\nProcessed {success} proteins, failed {failed}")

    # Save per-position data
    pos_df = pd.DataFrame(all_positions)
    pos_path = OUT / "all_n_positions_with_rsa.csv"
    pos_df.to_csv(pos_path, index=False)
    print(f"Saved {len(pos_df)} position rows to {pos_path}")

    # Save scores
    score_df = pd.DataFrame(all_scores)
    score_path = OUT / "mpnn_scores_by_condition.csv"
    score_df.to_csv(score_path, index=False)
    print(f"Saved {len(score_df)} score rows to {score_path}")

    # Build and save per-protein summary
    summary_df = build_per_protein_summary(pos_df)
    summary_path = OUT / "all_per_condition_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {len(summary_df)} summary rows to {summary_path}")

    # Print quick stats
    unc = pos_df[pos_df["condition"] == "unconstrained"]
    seq = unc[unc["is_sequon"] == True]
    nonseq = unc[unc["is_sequon"] == False]
    print(f"\nUnconstrained stats:")
    print(f"  Sequon N positions: {len(seq)}, mean retention: {seq['n_retention_pct'].mean():.1f}%")
    print(f"  Non-sequon N positions: {len(nonseq)}, mean retention: {nonseq['n_retention_pct'].mean():.1f}%")

    # Geometry stats
    geo = unc.dropna(subset=["ss"])
    print(f"  Positions with SS data: {len(geo)}")
    if len(geo) > 0:
        for ss in ["Helix", "Sheet", "Coil"]:
            subset = geo[geo["ss"] == ss]
            print(f"    {ss}: {len(subset)} positions")


if __name__ == "__main__":
    main()
