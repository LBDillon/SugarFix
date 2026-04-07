#!/usr/bin/env python3
"""Analyse AF3 clash experiment results.

Tests whether ProteinMPNN actively redesigns glycosite environments to be
glycan-incompatible, or merely removes the sequon motif passively.

Three conditions at each glycosylation site:
  1. WT + glycan           — baseline: glycan in its native protein context
  2. MPNN-reintroduced     — MPNN design with WT sequon grafted back + glycan
  3. MPNN-unconstrained    — MPNN design as-is, no glycan, sequon destroyed

Key comparison (the clash test):
  Condition 2 vs 3 — does adding the glycan back to the MPNN-redesigned
  environment cause EXTRA local distortion beyond what the sequence changes
  alone produce?  If yes → MPNN actively creates glycan-incompatible pockets.
  If no → the bias is purely sequence-level (motif removal).

Supporting comparison:
  Condition 1 vs 3 — baseline: how much does MPNN redesign change local
  geometry at glycosites? (Expected to differ regardless of glycans.)

Usage:
    python analyze_clash_results.py \\
        --results-dir /path/to/folds_download \\
        --pdb-ids 1J2E
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser, PDBParser, ShrakeRupley, Superimposer
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"

SPHERE_RADIUS = 8.0
BACKBONE_ATOMS = ("N", "CA", "C", "O")

CONDITION_LABELS = {
    "wt_with_glycan": "WT + glycan",
    "mpnn_reintroduced": "MPNN reintro + glycan",
    "mpnn_unconstrained": "MPNN unc (no glycan)",
}
CONDITION_COLORS = {
    "wt_with_glycan": "#1b9e77",
    "mpnn_reintroduced": "#d95f02",
    "mpnn_unconstrained": "#7570b3",
}


# ---------------------------------------------------------------------------
# Structure helpers (adapted from analyze_local_glycosites.py)
# ---------------------------------------------------------------------------

def get_standard_residues(chain):
    return [r for r in chain.get_residues() if r.get_id()[0] == " "]


def get_backbone_coords(residues):
    n = len(residues)
    coords = np.full((n, 4, 3), np.nan)
    has_all = np.zeros(n, dtype=bool)
    for i, res in enumerate(residues):
        found = 0
        for j, aname in enumerate(BACKBONE_ATOMS):
            if aname in res:
                coords[i, j] = res[aname].get_vector().get_array()
                found += 1
        has_all[i] = (found == 4)
    return coords, has_all


def global_superimpose(ref_residues, target_residues):
    """Superimpose target onto ref using CA atoms. Returns (rot, tran, global_rmsd)."""
    ref_ca = []
    tgt_ca = []
    for rr, tr in zip(ref_residues, target_residues):
        if "CA" in rr and "CA" in tr:
            ref_ca.append(rr["CA"])
            tgt_ca.append(tr["CA"])
    if len(ref_ca) < 10:
        return None, None, None
    sup = Superimposer()
    sup.set_atoms(ref_ca, tgt_ca)
    return sup.rotran[0], sup.rotran[1], sup.rms


def local_backbone_rmsd(ref_residues, af3_residues, rot, tran, center_idx, radius):
    """Compute backbone RMSD in a sphere around center_idx after global superposition."""
    ref_ca_coords = []
    ca_to_res = []
    for ri, (rr, ar) in enumerate(zip(ref_residues, af3_residues)):
        if "CA" in rr and "CA" in ar:
            ref_ca_coords.append(rr["CA"].get_vector().get_array())
            ca_to_res.append(ri)
    ref_ca_coords = np.array(ref_ca_coords)

    # Find center in CA array
    try:
        ca_center = ca_to_res.index(center_idx)
    except ValueError:
        return np.nan, 0
    center = ref_ca_coords[ca_center]
    dists = np.linalg.norm(ref_ca_coords - center, axis=1)
    local_mask = dists < radius

    ref_bb, ref_bb_mask = get_backbone_coords(ref_residues)
    af3_bb_raw, af3_bb_mask = get_backbone_coords(af3_residues)
    af3_bb = af3_bb_raw.copy()
    for i in range(len(af3_residues)):
        if af3_bb_mask[i]:
            af3_bb[i] = af3_bb_raw[i] @ rot + tran
    pair_mask = ref_bb_mask & af3_bb_mask

    local_ref = []
    local_af3 = []
    for ca_idx in range(len(ref_ca_coords)):
        if local_mask[ca_idx]:
            res_idx = ca_to_res[ca_idx]
            if pair_mask[res_idx]:
                local_ref.append(ref_bb[res_idx])
                local_af3.append(af3_bb[res_idx])

    if len(local_ref) < 3:
        return np.nan, 0

    ref_flat = np.concatenate(local_ref, axis=0)
    af3_flat = np.concatenate(local_af3, axis=0)
    rmsd = np.sqrt(np.mean(np.sum((ref_flat - af3_flat) ** 2, axis=1)))
    return rmsd, len(local_ref)


def measure_nag_asn_distance(af3_model, protein_chain_id, asn_pos_1idx, nag_chains):
    """Measure distance from Asn ND2/CG to nearest NAG C1 atom.

    Returns (min_distance, best_nag_chain_id) or (nan, None).
    """
    try:
        pchain = af3_model[protein_chain_id]
    except KeyError:
        return np.nan, None

    residues = get_standard_residues(pchain)
    idx = asn_pos_1idx - 1
    if idx >= len(residues):
        return np.nan, None

    asn_res = residues[idx]
    # Get Asn sidechain atom for distance measurement
    asn_atom = None
    for aname in ("ND2", "CG", "CA"):
        if aname in asn_res:
            asn_atom = asn_res[aname]
            break
    if asn_atom is None:
        return np.nan, None

    asn_coord = asn_atom.get_vector().get_array()

    best_dist = float("inf")
    best_chain = None
    for nag_chain_id in nag_chains:
        try:
            nag_chain = af3_model[nag_chain_id]
        except KeyError:
            continue
        for res in nag_chain.get_residues():
            if "C1" in res:
                c1_coord = res["C1"].get_vector().get_array()
                d = np.linalg.norm(asn_coord - c1_coord)
                if d < best_dist:
                    best_dist = d
                    best_chain = nag_chain_id

    return (best_dist if best_dist < 100 else np.nan), best_chain


def extract_plddt_around_site(af3_residues, site_pos_0idx, window=5):
    """Extract mean per-residue pLDDT around a glycosite from AF3 CIF structure.

    AF3 stores pLDDT in the B-factor column of CIF files. We average the CA
    B-factor for residues within ±window of the site.
    """
    plddts = []
    start = max(0, site_pos_0idx - window)
    end = min(len(af3_residues), site_pos_0idx + window + 1)
    for i in range(start, end):
        res = af3_residues[i]
        if "CA" in res:
            plddts.append(res["CA"].get_bfactor())
    return np.mean(plddts) if plddts else np.nan


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def find_af3_files(results_dir, pdb_id):
    """Find AF3 result directories for each clash condition."""
    pdb_lower = pdb_id.lower()
    conditions = {}
    for cond_key in ("wt_with_glycan", "mpnn_reintroduced", "mpnn_unconstrained"):
        dirname = f"{pdb_lower}_clash_{cond_key}"
        cond_dir = results_dir / dirname
        if cond_dir.exists():
            conditions[cond_key] = cond_dir
    return conditions


def analyze_protein(pdb_id, results_dir):
    """Run clash analysis for one protein."""
    # Load clash experiment data
    clash_json_path = (DATA_DIR / "outputs" / f"output_{pdb_id}"
                       / "clash_experiment" / "clash_sequences.json")
    if not clash_json_path.exists():
        print(f"No clash_sequences.json for {pdb_id}")
        return None

    with open(clash_json_path) as f:
        clash_data = json.load(f)

    sites = clash_data["sites"]
    chain_order = clash_data["chain_order"]

    # Find crystal structure
    crystal_path = DATA_DIR / "prep" / pdb_id / "structure" / f"{pdb_id}_protein.pdb"
    if not crystal_path.exists():
        crystal_path = DATA_DIR / "prep" / pdb_id / "structure" / f"{pdb_id}.pdb"
    if not crystal_path.exists():
        print(f"Crystal structure not found for {pdb_id}")
        return None

    # Find AF3 results
    af3_conditions = find_af3_files(results_dir, pdb_id)
    if not af3_conditions:
        print(f"No AF3 results found for {pdb_id}")
        return None

    print(f"\n{'='*60}")
    print(f"Clash analysis: {pdb_id}")
    print(f"  Crystal: {crystal_path.name}")
    print(f"  AF3 conditions: {list(af3_conditions.keys())}")
    print(f"  Glycan sites: {len(sites)} ({sum(1 for s in sites if s['testable'])} testable)")

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    crystal = pdb_parser.get_structure(pdb_id, str(crystal_path))
    crystal_model = crystal[0]

    # Compute crystal SASA
    sr = ShrakeRupley()
    sr.compute(crystal_model, level="R")

    crystal_chains = sorted(set(ch.id for ch in crystal_model.get_chains()))

    rows = []

    for cond_key, cond_dir in af3_conditions.items():
        # Use model_0 (top-ranked)
        prefix = f"fold_{pdb_id.lower()}_clash_{cond_key}"
        cif_path = cond_dir / f"{prefix}_model_0.cif"
        full_data_path = cond_dir / f"{prefix}_full_data_0.json"
        confidence_path = cond_dir / f"{prefix}_summary_confidences_0.json"

        if not cif_path.exists():
            print(f"  Missing CIF for {cond_key}")
            continue

        af3_structure = cif_parser.get_structure(f"{pdb_id}_{cond_key}", str(cif_path))
        af3_model = af3_structure[0]

        # Compute AF3 SASA
        sr_af3 = ShrakeRupley()
        sr_af3.compute(af3_model, level="R")

        # Load confidence metrics
        conf = {}
        if confidence_path.exists():
            with open(confidence_path) as f:
                conf = json.load(f)

        # Identify NAG chains in AF3 model
        nag_chains = []
        for ch in af3_model.get_chains():
            residues = list(ch.get_residues())
            if len(residues) == 1 and residues[0].resname == "NAG":
                nag_chains.append(ch.id)

        # Map crystal chains to AF3 protein chains
        af3_protein_chains = []
        for ch in af3_model.get_chains():
            std_res = get_standard_residues(ch)
            if len(std_res) > 10:
                af3_protein_chains.append(ch.id)

        chain_map = {}
        for i, cch in enumerate(crystal_chains):
            if i < len(af3_protein_chains):
                chain_map[cch] = af3_protein_chains[i]

        # Analyse each glycosite
        for site in sites:
            chain_label = site["chain"]
            pos1 = site["position_1idx"]
            af3_chain_id = chain_map.get(chain_label)
            if af3_chain_id is None:
                continue

            chain_idx = chain_order.index(chain_label) if chain_label in chain_order else -1
            if chain_idx < 0:
                continue

            try:
                ref_chain = crystal_model[chain_label]
                af3_chain = af3_model[af3_chain_id]
            except KeyError:
                continue

            ref_residues = get_standard_residues(ref_chain)
            af3_residues = get_standard_residues(af3_chain)

            if len(ref_residues) != len(af3_residues):
                print(f"    Length mismatch {chain_label}: crystal={len(ref_residues)} "
                      f"AF3={len(af3_residues)}")
                continue

            rot, tran, global_rmsd = global_superimpose(ref_residues, af3_residues)
            if rot is None:
                continue

            pos0 = pos1 - 1
            if pos0 >= len(ref_residues):
                continue

            # Local backbone RMSD
            lrmsd, n_local = local_backbone_rmsd(
                ref_residues, af3_residues, rot, tran, pos0, SPHERE_RADIUS
            )

            # SASA at glycosite Asn
            ref_res = ref_residues[pos0]
            af3_res = af3_residues[pos0]
            sasa_crystal = ref_res.sasa if hasattr(ref_res, "sasa") else np.nan
            sasa_af3 = af3_res.sasa if hasattr(af3_res, "sasa") else np.nan

            # NAG-Asn distance (only for glycan conditions)
            nag_dist = np.nan
            nag_chain_match = None
            if nag_chains and cond_key != "mpnn_unconstrained":
                nag_dist, nag_chain_match = measure_nag_asn_distance(
                    af3_model, af3_chain_id, pos1, nag_chains
                )

            # pLDDT around site (from B-factor column of CIF)
            local_plddt = extract_plddt_around_site(af3_residues, pos0, window=5)

            rows.append({
                "pdb_id": pdb_id,
                "chain": chain_label,
                "position_1idx": pos1,
                "pdb_resnum": site["pdb_resnum"],
                "wt_triplet": site["wt_triplet"],
                "design_triplet": site["design_triplet"],
                "testable": site["testable"],
                "is_stub": site["is_stub"],
                "condition": cond_key,
                "local_rmsd_8A": round(lrmsd, 4) if not np.isnan(lrmsd) else np.nan,
                "n_residues_in_sphere": n_local,
                "global_rmsd": round(global_rmsd, 4),
                "sasa_crystal": round(sasa_crystal, 2) if not np.isnan(sasa_crystal) else np.nan,
                "sasa_af3": round(sasa_af3, 2) if not np.isnan(sasa_af3) else np.nan,
                "nag_asn_distance": round(nag_dist, 2) if not np.isnan(nag_dist) else np.nan,
                "nag_chain": nag_chain_match,
                "local_plddt": round(local_plddt, 1) if not np.isnan(local_plddt) else np.nan,
                "ptm": conf.get("ptm"),
                "iptm": conf.get("iptm"),
                "ranking_score": conf.get("ranking_score"),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def print_summary(df, pdb_id):
    """Print head-to-head comparison with correct interpretation."""
    print(f"\n{'='*70}")
    print(f"CLASH EXPERIMENT RESULTS — {pdb_id}")
    print(f"{'='*70}")

    testable = df[df["testable"]]
    n_test = testable[testable["condition"] == "wt_with_glycan"].shape[0]
    print(f"\nTestable sites: {n_test}")

    # Global confidence
    print("\nGlobal confidence (seed 0):")
    for cond in ("wt_with_glycan", "mpnn_reintroduced", "mpnn_unconstrained"):
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        ptm = sub["ptm"].iloc[0]
        iptm = sub["iptm"].iloc[0]
        rs = sub["ranking_score"].iloc[0]
        print(f"  {CONDITION_LABELS[cond]:>25s}: pTM={ptm:.3f}  ipTM={iptm:.3f}  ranking={rs:.3f}")

    # Per-site comparison table
    print(f"\nPer-site metrics (testable sites only):")
    print(f"{'Site':<10} {'WT→Des':<14} "
          f"{'RMSD_WT':>8} {'RMSD_re':>8} {'RMSD_unc':>9} {'Δ(re-unc)':>10} "
          f"{'NAG_WT':>7} {'NAG_re':>7} "
          f"{'pLDDT_WT':>9} {'pLDDT_re':>9} {'pLDDT_unc':>10}")
    print("-" * 120)

    site_keys = testable[["chain", "position_1idx", "wt_triplet",
                           "design_triplet"]].drop_duplicates()

    for _, sk in site_keys.iterrows():
        ch, pos = sk["chain"], sk["position_1idx"]
        wt_t, des_t = sk["wt_triplet"], sk["design_triplet"]

        vals = {}
        for cond in ("wt_with_glycan", "mpnn_reintroduced", "mpnn_unconstrained"):
            row = testable[(testable["chain"] == ch) &
                           (testable["position_1idx"] == pos) &
                           (testable["condition"] == cond)]
            if not row.empty:
                vals[cond] = row.iloc[0]

        def fmt(v, key, prec=3):
            if v is None:
                return "—"
            val = v.get(key, np.nan)
            if pd.isna(val):
                return "—"
            return f"{val:.{prec}f}"

        wt = vals.get("wt_with_glycan")
        re = vals.get("mpnn_reintroduced")
        unc = vals.get("mpnn_unconstrained")

        # Delta: reintroduced - unconstrained (the clash signal)
        delta = "—"
        if re is not None and unc is not None:
            rv = re.get("local_rmsd_8A", np.nan)
            uv = unc.get("local_rmsd_8A", np.nan)
            if not (pd.isna(rv) or pd.isna(uv)):
                d = rv - uv
                delta = f"{d:+.3f}"

        print(f"{ch}:{pos:<7} {wt_t}→{des_t:<8} "
              f"{fmt(wt, 'local_rmsd_8A'):>8} {fmt(re, 'local_rmsd_8A'):>8} "
              f"{fmt(unc, 'local_rmsd_8A'):>9} {delta:>10} "
              f"{fmt(wt, 'nag_asn_distance'):>7} {fmt(re, 'nag_asn_distance'):>7} "
              f"{fmt(wt, 'local_plddt', 1):>9} {fmt(re, 'local_plddt', 1):>9} "
              f"{fmt(unc, 'local_plddt', 1):>10}")

    # Statistical comparisons
    wt = testable[testable["condition"] == "wt_with_glycan"].set_index(
        ["chain", "position_1idx"])
    re = testable[testable["condition"] == "mpnn_reintroduced"].set_index(
        ["chain", "position_1idx"])
    unc = testable[testable["condition"] == "mpnn_unconstrained"].set_index(
        ["chain", "position_1idx"])

    def _paired_test(a, b, metric, label, a_name, b_name):
        common = a.index.intersection(b.index)
        av = a.loc[common, metric].dropna()
        bv = b.loc[common, metric].dropna()
        both = av.index.intersection(bv.index)
        if len(both) < 3:
            return
        w, r = av[both].values, bv[both].values
        if len(w) >= 5:
            _, pval = stats.wilcoxon(w, r)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        else:
            pval, sig = np.nan, f"n={len(w)}"
        d = r - w
        n_pos = (d > 0).sum()
        print(f"  {label:>25s}: {a_name} med={np.median(w):.3f}, "
              f"{b_name} med={np.median(r):.3f}, "
              f"Δ={np.median(d):+.3f}, "
              f"p={pval:.3e} ({sig}), {n_pos}/{len(d)} higher in {b_name}")

    print(f"\n--- CLASH TEST: Reintroduced (+glycan) vs Unconstrained (no glycan) ---")
    print(f"    If glycan causes extra distortion → reintroduced should be worse")
    for metric, label in [("local_rmsd_8A", "Local RMSD"),
                           ("local_plddt", "Local pLDDT"),
                           ("sasa_af3", "SASA at Asn")]:
        _paired_test(unc, re, metric, label, "Unc", "Reintro")

    print(f"\n--- CONTEXT: WT vs Unconstrained (baseline MPNN sequence effect) ---")
    for metric, label in [("local_rmsd_8A", "Local RMSD"),
                           ("local_plddt", "Local pLDDT")]:
        _paired_test(wt, unc, metric, label, "WT", "Unc")

    print(f"\n--- CONTEXT: WT vs Reintroduced (sequence + glycan combined) ---")
    for metric, label in [("local_rmsd_8A", "Local RMSD"),
                           ("nag_asn_distance", "NAG-Asn distance")]:
        _paired_test(wt, re, metric, label, "WT", "Reintro")


def create_figure(df, pdb_id, output_dir):
    """Create clash experiment figure with correct comparisons.

    Panel A: All three conditions side-by-side (local RMSD vs crystal).
             Lines connect the same site across conditions.
    Panel B: The clash test — delta local RMSD (reintroduced minus unconstrained).
             Positive = glycan causes extra distortion.
    Panel C: NAG-Asn distance (zoomed to actual data range).
    Panel D: Local pLDDT comparison.
    """
    testable = df[df["testable"]].copy()
    if testable.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wt = testable[testable["condition"] == "wt_with_glycan"].set_index(
        ["chain", "position_1idx"])
    re = testable[testable["condition"] == "mpnn_reintroduced"].set_index(
        ["chain", "position_1idx"])
    unc = testable[testable["condition"] == "mpnn_unconstrained"].set_index(
        ["chain", "position_1idx"])
    common = wt.index.intersection(re.index).intersection(unc.index)

    # --- Panel A: Three-condition local RMSD ---
    ax = axes[0, 0]
    if len(common) >= 2:
        wv = wt.loc[common, "local_rmsd_8A"].values
        rv = re.loc[common, "local_rmsd_8A"].values
        uv = unc.loc[common, "local_rmsd_8A"].values

        for w, r, u in zip(wv, rv, uv):
            ax.plot([0, 1, 2], [w, u, r], color="grey", alpha=0.3, linewidth=0.7)

        ax.scatter(np.zeros(len(wv)), wv, color=CONDITION_COLORS["wt_with_glycan"],
                   s=40, alpha=0.8, zorder=3)
        ax.scatter(np.ones(len(uv)), uv, color=CONDITION_COLORS["mpnn_unconstrained"],
                   s=40, alpha=0.8, zorder=3)
        ax.scatter(np.full(len(rv), 2), rv, color=CONDITION_COLORS["mpnn_reintroduced"],
                   s=40, alpha=0.8, zorder=3)

        for x, v, c in [(0, wv, "wt_with_glycan"), (1, uv, "mpnn_unconstrained"),
                          (2, rv, "mpnn_reintroduced")]:
            ax.plot([x - 0.15, x + 0.15], [np.median(v)] * 2,
                    color=CONDITION_COLORS[c], linewidth=3, zorder=4)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["WT\n+ glycan", "MPNN unc\n(no glycan)",
                             "MPNN reintro\n+ glycan"], fontsize=9)
        ax.set_title(f"A. Local Backbone RMSD vs Crystal\n(all 3 conditions, n={len(common)} sites)",
                     fontsize=10, fontweight="bold")
    ax.set_ylabel("Local RMSD (8A sphere) [A]", fontsize=9)

    # --- Panel B: The clash test — delta RMSD ---
    ax = axes[0, 1]
    if len(common) >= 2:
        rv = re.loc[common, "local_rmsd_8A"].values
        uv = unc.loc[common, "local_rmsd_8A"].values
        deltas = rv - uv

        colors = [CONDITION_COLORS["mpnn_reintroduced"] if d > 0 else
                  CONDITION_COLORS["wt_with_glycan"] for d in deltas]

        y_pos = np.arange(len(deltas))
        labels = [f"{idx[0]}:{idx[1]}" for idx in common]
        sort_idx = np.argsort(deltas)[::-1]

        ax.barh(y_pos, deltas[sort_idx], color=[colors[i] for i in sort_idx],
                edgecolor="black", linewidth=0.5, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in sort_idx], fontsize=8)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Δ Local RMSD (reintro − unconstrained) [A]", fontsize=9)

        n_pos = (deltas > 0).sum()
        if len(deltas) >= 5:
            _, pval = stats.wilcoxon(deltas)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            title_extra = f"Wilcoxon p={pval:.2e} ({sig})"
        else:
            title_extra = f"n={len(deltas)}"

        ax.set_title(f"B. CLASH TEST: Does glycan cause extra distortion?\n"
                     f"{n_pos}/{len(deltas)} sites worse with glycan. {title_extra}",
                     fontsize=10, fontweight="bold")
        ax.invert_yaxis()

        # Annotation
        ax.text(0.02, 0.02, "← glycan fits fine", transform=ax.transAxes,
                fontsize=8, color=CONDITION_COLORS["wt_with_glycan"], style="italic")
        ax.text(0.98, 0.02, "glycan causes distortion →", transform=ax.transAxes,
                fontsize=8, color=CONDITION_COLORS["mpnn_reintroduced"],
                style="italic", ha="right")

    # --- Panel C: NAG-Asn distance (zoomed) ---
    ax = axes[1, 0]
    glycan_conds = testable[testable["condition"].isin(
        ["wt_with_glycan", "mpnn_reintroduced"])]
    nag_vals = glycan_conds["nag_asn_distance"].dropna()
    if not nag_vals.empty:
        wt_nag = testable[(testable["condition"] == "wt_with_glycan")]["nag_asn_distance"].dropna()
        re_nag = testable[(testable["condition"] == "mpnn_reintroduced")]["nag_asn_distance"].dropna()

        positions = [0, 1]
        bp = ax.boxplot([wt_nag.values, re_nag.values], positions=positions,
                        widths=0.4, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(CONDITION_COLORS["wt_with_glycan"])
        bp["boxes"][1].set_facecolor(CONDITION_COLORS["mpnn_reintroduced"])
        for box in bp["boxes"]:
            box.set_alpha(0.6)

        ax.scatter(np.zeros(len(wt_nag)) + np.random.normal(0, 0.04, len(wt_nag)),
                   wt_nag.values, color="black", alpha=0.5, s=25, zorder=3)
        ax.scatter(np.ones(len(re_nag)) + np.random.normal(0, 0.04, len(re_nag)),
                   re_nag.values, color="black", alpha=0.5, s=25, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(["WT + glycan", "MPNN reintro\n+ glycan"], fontsize=9)
        ax.set_ylabel("NAG C1 — Asn ND2 distance [A]", fontsize=9)

        # Zoom y-axis to data range with padding
        all_vals = np.concatenate([wt_nag.values, re_nag.values])
        ymin = max(0, np.min(all_vals) - 0.2)
        ymax = np.max(all_vals) + 0.2
        ax.set_ylim(ymin, ymax)

    ax.set_title("C. Glycan Placement (NAG–Asn distance)", fontsize=10, fontweight="bold")

    # --- Panel D: Local pLDDT ---
    ax = axes[1, 1]
    has_plddt = not testable["local_plddt"].isna().all()
    if has_plddt:
        plddt_data = []
        plddt_labels = []
        plddt_colors = []
        for cond, label in [("wt_with_glycan", "WT+glyc"),
                             ("mpnn_unconstrained", "MPNN unc"),
                             ("mpnn_reintroduced", "MPNN re+glyc")]:
            sub = testable[testable["condition"] == cond]["local_plddt"].dropna()
            if not sub.empty:
                plddt_data.append(sub.values)
                plddt_labels.append(label)
                plddt_colors.append(CONDITION_COLORS[cond])

        if plddt_data:
            bp = ax.boxplot(plddt_data, positions=range(len(plddt_data)),
                            widths=0.5, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], plddt_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            for i, vals in enumerate(plddt_data):
                jitter = np.random.normal(0, 0.05, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           color="black", alpha=0.5, s=25, zorder=3)

            ax.set_xticks(range(len(plddt_labels)))
            ax.set_xticklabels(plddt_labels, fontsize=9)
            ax.set_ylabel("Mean pLDDT (±5 residues)", fontsize=9)
    ax.set_title("D. AF3 Confidence at Glycosites", fontsize=10, fontweight="bold")

    fig.suptitle(f"Glycan Clash Experiment — {pdb_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"clash_analysis_{pdb_id}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse AF3 clash experiment results.")
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Directory containing AF3 result folders")
    parser.add_argument("--pdb-ids", nargs="+", required=True,
                        help="PDB IDs to analyse")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: clash_experiment/ under each protein)")
    return parser.parse_args()


def main():
    args = parse_args()

    all_dfs = []
    for pdb_id in args.pdb_ids:
        pdb_id = pdb_id.upper()
        df = analyze_protein(pdb_id, args.results_dir)
        if df is None:
            continue

        # Save per-protein CSV
        out_dir = args.output_dir or (DATA_DIR / "outputs" / f"output_{pdb_id}" / "clash_experiment")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"clash_metrics_{pdb_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Metrics saved: {csv_path}")

        print_summary(df, pdb_id)
        create_figure(df, pdb_id, out_dir)
        all_dfs.append(df)

    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_dir = args.output_dir or (DATA_DIR / "outputs" / "clash_experiment_combined")
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_dir / "clash_metrics_all.csv", index=False)
        print(f"\nCombined CSV: {combined_dir / 'clash_metrics_all.csv'}")


if __name__ == "__main__":
    main()
