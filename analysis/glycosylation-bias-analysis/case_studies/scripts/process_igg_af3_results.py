#!/usr/bin/env python3
"""
Process AF3 predictions for IgG Fc case study (3AVE, 3S7G, 1L6X).

1. Organize AF3 outputs into pipeline structure
2. Extract confidence metrics
3. Compute RMSD vs crystal structures
4. Compute local glycosite metrics
"""

import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    from Bio.PDB.PDBIO import PDBIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("WARNING: Biopython not available, RMSD computation will be skipped")

BASE = Path(
    "/Users/lauradillon/PycharmProjects/inverse_fold/Cleaned_research_flow/0_Main_data/Final_Paper_Folder/protein-design-bias"
)
PIPELINE_DATA = BASE / "experiments" / "MPNN_to_AF3_analysis" / "case_study_pipeline" / "data"
FOLDS_DIR = Path("/Users/lauradillon/Downloads/folds_2026_03_12_12_25")

PDBS = ["3AVE", "3S7G", "1L6X"]

# Map folder names to structured conditions
CONDITION_MAP = {
    "full_sequon_fixed_top1": "full_sequon_fixed",
    "full_sequon_fixed_top1_with_glycans": "full_sequon_fixed_with_glycans",
    "unconstrained_top1": "unconstrained",
    "unconstrained_top1_denovo_glycans": "unconstrained_denovo_glycans",
}


def organize_af3_results():
    """Copy best model (model_0) and confidence into pipeline structure."""
    for pdb in PDBS:
        pdb_lower = pdb.lower()
        af3_dir = PIPELINE_DATA / "af3_results" / pdb
        models_dir = af3_dir / "models"
        conf_dir = af3_dir / "confidences"
        models_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        for folder_suffix, condition in CONDITION_MAP.items():
            fold_name = f"{pdb_lower}_protein_{folder_suffix}"
            fold_dir = FOLDS_DIR / fold_name

            if not fold_dir.exists():
                print(f"  WARNING: {fold_dir} not found, skipping")
                continue

            # Copy best model (model_0.cif)
            src_cif = fold_dir / f"fold_{fold_name}_model_0.cif"
            dst_cif = models_dir / f"{pdb}_{condition}.cif"
            if src_cif.exists():
                shutil.copy2(src_cif, dst_cif)

            # Copy confidence (summary_confidences_0.json)
            src_conf = fold_dir / f"fold_{fold_name}_summary_confidences_0.json"
            dst_conf = conf_dir / f"{pdb}_{condition}.json"
            if src_conf.exists():
                shutil.copy2(src_conf, dst_conf)

        # Also copy WT crystal structure as reference
        crystal_pdb = PIPELINE_DATA / "prep" / pdb / "structure" / f"{pdb}_protein.pdb"
        if crystal_pdb.exists():
            shutil.copy2(crystal_pdb, af3_dir / f"{pdb.lower()}.pdb")

        print(f"  Organized {pdb} AF3 results")


def extract_confidence_metrics():
    """Extract pTM, ipTM, ranking_score from confidence JSONs."""
    rows = []
    for pdb in PDBS:
        conf_dir = PIPELINE_DATA / "af3_results" / pdb / "confidences"
        for condition in CONDITION_MAP.values():
            conf_path = conf_dir / f"{pdb}_{condition}.json"
            if not conf_path.exists():
                continue
            with open(conf_path) as f:
                data = json.load(f)
            rows.append({
                "pdb_id": pdb,
                "condition": condition,
                "ptm": data.get("ptm"),
                "iptm": data.get("iptm"),
                "ranking_score": data.get("ranking_score"),
                "fraction_disordered": data.get("fraction_disordered"),
                "has_clash": data.get("has_clash"),
                "chain_ptm": str(data.get("chain_ptm", [])),
            })

    df = pd.DataFrame(rows)
    return df


def compute_rmsd_per_chain(pdb_id):
    """Compute per-chain RMSD between AF3 predictions and crystal structure."""
    if not HAS_BIOPYTHON:
        return []

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    af3_dir = PIPELINE_DATA / "af3_results" / pdb_id
    crystal_path = af3_dir / f"{pdb_id.lower()}.pdb"

    if not crystal_path.exists():
        print(f"  WARNING: Crystal structure not found for {pdb_id}")
        return []

    try:
        ref_structure = pdb_parser.get_structure("ref", str(crystal_path))
        ref_model = ref_structure[0]
    except Exception as e:
        print(f"  ERROR parsing crystal {pdb_id}: {e}")
        return []

    # Get reference CA atoms per chain
    ref_chains = {}
    for chain in ref_model:
        cid = chain.get_id()
        ca_atoms = []
        for res in chain:
            if res.get_id()[0] != " ":
                continue
            if "CA" in res:
                ca_atoms.append(res["CA"])
        if ca_atoms:
            ref_chains[cid] = ca_atoms

    results = []
    models_dir = af3_dir / "models"

    for condition in CONDITION_MAP.values():
        cif_path = models_dir / f"{pdb_id}_{condition}.cif"
        if not cif_path.exists():
            continue

        try:
            target_structure = cif_parser.get_structure("target", str(cif_path))
            target_model = target_structure[0]
        except Exception as e:
            print(f"  ERROR parsing {pdb_id}_{condition}: {e}")
            continue

        # Get target CA atoms per chain
        target_chains = {}
        for chain in target_model:
            cid = chain.get_id()
            ca_atoms = []
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                if "CA" in res:
                    ca_atoms.append(res["CA"])
            if ca_atoms:
                target_chains[cid] = ca_atoms

        # Match chains by letter
        chain_rmsds = []
        for cid in ref_chains:
            if cid not in target_chains:
                continue

            ref_ca = ref_chains[cid]
            tgt_ca = target_chains[cid]

            # Use minimum length
            n = min(len(ref_ca), len(tgt_ca))
            if n < 10:
                continue

            sup = Superimposer()
            sup.set_atoms(ref_ca[:n], tgt_ca[:n])
            rmsd = sup.rms

            results.append({
                "pdb_id": pdb_id,
                "condition": condition,
                "chain": cid,
                "rmsd": rmsd,
                "n_atoms": n,
            })
            chain_rmsds.append(rmsd)

        if chain_rmsds:
            results.append({
                "pdb_id": pdb_id,
                "condition": condition,
                "chain": "mean",
                "rmsd": np.mean(chain_rmsds),
                "n_atoms": None,
            })

    return results


def compute_local_glycosite_metrics(pdb_id):
    """Compute local RMSD around glycosite and SASA."""
    if not HAS_BIOPYTHON:
        return []

    from Bio.PDB import PDBParser, MMCIFParser, Superimposer, NeighborSearch

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)

    af3_dir = PIPELINE_DATA / "af3_results" / pdb_id
    crystal_path = af3_dir / f"{pdb_id.lower()}.pdb"

    if not crystal_path.exists():
        return []

    # Load sequon positions
    sequon_path = PIPELINE_DATA / "prep" / pdb_id / "sequons" / "sequons_by_chain.json"
    if not sequon_path.exists():
        return []

    with open(sequon_path) as f:
        sequons_by_chain = json.load(f)

    # Get chain summary for residue numbering
    chain_csv = PIPELINE_DATA / "prep" / pdb_id / "structure" / "chain_summary.csv"
    chain_info = pd.read_csv(chain_csv).drop_duplicates("chain_id")

    try:
        ref_structure = pdb_parser.get_structure("ref", str(crystal_path))
        ref_model = ref_structure[0]
    except Exception:
        return []

    results = []
    models_dir = af3_dir / "models"

    for condition in CONDITION_MAP.values():
        cif_path = models_dir / f"{pdb_id}_{condition}.cif"
        if not cif_path.exists():
            continue

        try:
            tgt_structure = cif_parser.get_structure("tgt", str(cif_path))
            tgt_model = tgt_structure[0]
        except Exception:
            continue

        for chain_id, sequon_list in sequons_by_chain.items():
            if not sequon_list:
                continue

            # Get first_residue for this chain
            chain_row = chain_info[chain_info["chain_id"] == chain_id]
            if len(chain_row) == 0:
                continue
            first_res = chain_row.iloc[0]["first_residue"]

            for sq in sequon_list:
                pos_0idx = sq["position_0idx"]
                resnum = first_res + pos_0idx  # PDB residue number

                # Get CA atoms within 8A of glycosite in crystal
                if chain_id not in ref_model:
                    continue
                ref_chain = ref_model[chain_id]

                # Find the glycosite residue
                glyco_res = None
                for res in ref_chain:
                    if res.get_id()[1] == resnum and res.get_id()[0] == " ":
                        glyco_res = res
                        break

                if glyco_res is None or "CA" not in glyco_res:
                    continue

                glyco_ca = glyco_res["CA"].get_vector().get_array()

                # Get all CA atoms within 8A in reference
                ref_local_ca = []
                ref_local_resnums = []
                for res in ref_chain:
                    if res.get_id()[0] != " " or "CA" not in res:
                        continue
                    ca = res["CA"]
                    dist = np.linalg.norm(ca.get_vector().get_array() - glyco_ca)
                    if dist <= 8.0:
                        ref_local_ca.append(ca)
                        ref_local_resnums.append(res.get_id()[1])

                if len(ref_local_ca) < 5:
                    continue

                # Get matching atoms in target
                if chain_id not in tgt_model:
                    continue
                tgt_chain = tgt_model[chain_id]

                tgt_local_ca = []
                matched_ref_ca = []
                for rn, rca in zip(ref_local_resnums, ref_local_ca):
                    for tres in tgt_chain:
                        if tres.get_id()[1] == rn and tres.get_id()[0] == " " and "CA" in tres:
                            tgt_local_ca.append(tres["CA"])
                            matched_ref_ca.append(rca)
                            break

                if len(matched_ref_ca) < 5:
                    continue

                # Local RMSD
                sup = Superimposer()
                sup.set_atoms(matched_ref_ca, tgt_local_ca)
                local_rmsd = sup.rms

                # Global RMSD for this chain
                ref_all_ca = [res["CA"] for res in ref_chain
                              if res.get_id()[0] == " " and "CA" in res]
                tgt_all_ca = [res["CA"] for res in tgt_chain
                              if res.get_id()[0] == " " and "CA" in res]
                n_min = min(len(ref_all_ca), len(tgt_all_ca))
                if n_min >= 10:
                    sup2 = Superimposer()
                    sup2.set_atoms(ref_all_ca[:n_min], tgt_all_ca[:n_min])
                    global_rmsd = sup2.rms
                else:
                    global_rmsd = None

                results.append({
                    "pdb_id": pdb_id,
                    "chain": chain_id,
                    "resnum": resnum,
                    "condition": condition,
                    "local_rmsd_8A": local_rmsd,
                    "global_rmsd": global_rmsd,
                })

    return results


def main():
    print("1. Organizing AF3 results...")
    organize_af3_results()

    print("\n2. Extracting confidence metrics...")
    conf_df = extract_confidence_metrics()
    print(conf_df.to_string())

    print("\n3. Computing RMSD...")
    all_rmsd = []
    for pdb in PDBS:
        print(f"  Processing {pdb}...")
        rmsd_results = compute_rmsd_per_chain(pdb)
        all_rmsd.extend(rmsd_results)

    rmsd_df = pd.DataFrame(all_rmsd)
    if len(rmsd_df) > 0:
        print("\nRMSD results (mean per structure):")
        means = rmsd_df[rmsd_df["chain"] == "mean"][["pdb_id", "condition", "rmsd"]]
        print(means.to_string())

    print("\n4. Computing local glycosite metrics...")
    all_local = []
    for pdb in PDBS:
        print(f"  Processing {pdb}...")
        local_results = compute_local_glycosite_metrics(pdb)
        all_local.extend(local_results)

    local_df = pd.DataFrame(all_local)
    if len(local_df) > 0:
        print("\nLocal glycosite metrics:")
        print(local_df.to_string())

    # Save all results
    out_dir = PIPELINE_DATA / "af3_results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge confidence + RMSD into combined CSV
    if len(rmsd_df) > 0:
        rmsd_means = rmsd_df[rmsd_df["chain"] == "mean"][["pdb_id", "condition", "rmsd"]].rename(
            columns={"rmsd": "mean_rmsd"})
        combined = conf_df.merge(rmsd_means, on=["pdb_id", "condition"], how="left")
    else:
        combined = conf_df
        combined["mean_rmsd"] = np.nan

    # Save IgG-specific results
    igg_combined_path = out_dir / "igg_fc_af3_combined.csv"
    combined.to_csv(igg_combined_path, index=False)
    print(f"\nSaved combined results to {igg_combined_path}")

    if len(rmsd_df) > 0:
        igg_rmsd_path = out_dir / "igg_fc_rmsd_details.csv"
        rmsd_df.to_csv(igg_rmsd_path, index=False)
        print(f"Saved RMSD details to {igg_rmsd_path}")

    if len(local_df) > 0:
        igg_local_path = out_dir / "igg_fc_local_glycosite.csv"
        local_df.to_csv(igg_local_path, index=False)
        print(f"Saved local glycosite metrics to {igg_local_path}")

    # Per-structure summary
    for pdb in PDBS:
        pdb_rmsd = rmsd_df[(rmsd_df["pdb_id"] == pdb) & (rmsd_df["chain"] != "mean")]
        pdb_rmsd_path = PIPELINE_DATA / "af3_results" / pdb / "rmsd_result.csv"
        if len(pdb_rmsd) > 0:
            # Add mean rows
            means = rmsd_df[(rmsd_df["pdb_id"] == pdb) & (rmsd_df["chain"] == "mean")]
            full = pd.concat([pdb_rmsd, means])
            full["structure"] = pdb + "_" + full["condition"]
            full["ref_chain"] = full["chain"]
            full["target_chain"] = full["chain"]
            full.to_csv(pdb_rmsd_path, index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
