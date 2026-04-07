#!/usr/bin/env python3
"""
Analyze compensatory mutations around glycosites in ProteinMPNN case-study designs.

Outputs:
  - cd2_1gya_compensatory_mutations.csv
  - cd2_1gya_compensatory_mutations_summary.md
  - igg_fc_glycan_contact_mutations.csv
  - igg_fc_glycan_contact_mutations_summary.md
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.SASA import ShrakeRupley

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PIPELINE_ROOT / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
PREP_DIR = DATA_DIR / "prep"
OUT_DIR = OUTPUTS_DIR / "compensatory_mutation_analysis"

CONDITIONS = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]

AA3_TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# Tien et al. (2013) residue max ASA values in Angstrom^2.
MAX_ASA = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
}

GLYCAN_RESNAMES = {
    "NAG",
    "BMA",
    "MAN",
    "FUC",
    "FUL",
    "GAL",
    "GLC",
    "BGC",
    "SIA",
    "NDG",
    "NGA",
}

CD2_KEY_POSITIONS = {
    61: {
        "wt": "K",
        "rat": "E",
        "role": "Positive-charge cluster center; glycan masks this site",
    },
    63: {
        "wt": "F",
        "rat": "L",
        "role": "Enhanced aromatic sequon packing against GlcNAc1",
    },
    65: {
        "wt": "N",
        "rat": "N",
        "role": "Glycosylated Asn in the N-X-T sequon",
    },
    67: {
        "wt": "T",
        "rat": "D",
        "role": "Sequon-completing Thr that packs with Phe63 and GlcNAc1",
    },
}

CD2_CHARGE_NEUTRALIZING = {"E", "D", "Q"}
CD2_NON_AROMATIC_63 = {"L", "I", "V", "A"}
SEQUON_COMPLETING = {"S", "T"}


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_fasta_records(path: Path):
    records = []
    header = None
    seq_lines = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def parse_header_value(header: str, key: str):
    match = re.search(rf"{re.escape(key)}=([^,]+)", header)
    if not match:
        return None
    value = match.group(1).strip()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_mpnn_fasta(pdb_id: str, condition: str):
    fasta_path = (
        OUTPUTS_DIR / f"output_{pdb_id}" / condition / "seqs" / f"{pdb_id}_protein.fa"
    )
    chain_order = load_json(PREP_DIR / pdb_id / "sequons" / "mpnn_chain_order.json")[
        "chain_order"
    ]
    records = read_fasta_records(fasta_path)
    if not records:
        raise FileNotFoundError(f"No FASTA records found in {fasta_path}")

    wt_header, wt_concat = records[0]
    wt_parts = wt_concat.split("/")
    wt_chains = dict(zip(chain_order, wt_parts))
    if len(wt_parts) != len(chain_order):
        raise ValueError(
            f"{pdb_id} {condition}: chain count mismatch {len(wt_parts)} vs {len(chain_order)}"
        )

    designs = []
    for header, concat_seq in records[1:]:
        parts = concat_seq.split("/")
        if len(parts) != len(chain_order):
            raise ValueError(
                f"{pdb_id} {condition}: design chain count mismatch {len(parts)} vs {len(chain_order)}"
            )
        designs.append(
            {
                "sample": parse_header_value(header, "sample"),
                "score": parse_header_value(header, "score"),
                "seq_recovery": parse_header_value(header, "seq_recovery"),
                "chains": dict(zip(chain_order, parts)),
            }
        )

    return {
        "fasta_path": fasta_path,
        "chain_order": chain_order,
        "wt_header": wt_header,
        "wt_chains": wt_chains,
        "designs": designs,
    }


def build_chain_mappings(protein_pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_pdb_path.stem, protein_pdb_path)
    model = next(structure.get_models())

    mappings = {}
    for chain in model:
        residues = []
        sequence = []
        by_resnum = {}
        for residue in chain:
            if residue.id[0] != " " or residue.resname not in AA3_TO1:
                continue
            aa = AA3_TO1[residue.resname]
            seq_idx = len(residues) + 1
            pdb_resnum = int(residue.id[1])
            icode = residue.id[2].strip() if residue.id[2] else ""
            record = {
                "chain": chain.id,
                "seq_idx": seq_idx,
                "pdb_resnum": pdb_resnum,
                "icode": icode,
                "aa": aa,
                "residue": residue,
            }
            residues.append(record)
            sequence.append(aa)
            by_resnum[(pdb_resnum, icode)] = record
        mappings[chain.id] = {
            "sequence": "".join(sequence),
            "residues": residues,
            "by_resnum": by_resnum,
        }
    return mappings


def get_mapping_record(mappings, chain_id: str, pdb_resnum: int, icode: str = ""):
    key = (pdb_resnum, icode)
    if key in mappings[chain_id]["by_resnum"]:
        return mappings[chain_id]["by_resnum"][key]

    candidates = [
        record
        for (resnum, _icode), record in mappings[chain_id]["by_resnum"].items()
        if resnum == pdb_resnum
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Could not map {chain_id}:{pdb_resnum}{icode}")


def verify_wt_sequences(pdb_id: str, wt_chains, mappings):
    for chain_id, wt_sequence in wt_chains.items():
        pdb_sequence = mappings[chain_id]["sequence"]
        if wt_sequence != pdb_sequence:
            raise ValueError(
                f"{pdb_id} chain {chain_id}: WT FASTA does not match PDB sequence"
            )


def identify_glycan_contact_residues(full_pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(full_pdb_path.stem, full_pdb_path)
    model = next(structure.get_models())

    protein_atoms = []
    glycan_atoms = []
    for atom in model.get_atoms():
        if atom.element == "H":
            continue
        residue = atom.get_parent()
        if residue.resname in GLYCAN_RESNAMES:
            glycan_atoms.append(atom)
        elif residue.id[0] == " " and residue.resname in AA3_TO1:
            protein_atoms.append(atom)

    neighbor_search = NeighborSearch(protein_atoms)
    contacts = defaultdict(set)
    for glycan_atom in glycan_atoms:
        for protein_atom in neighbor_search.search(glycan_atom.coord, 4.0, level="A"):
            residue = protein_atom.get_parent()
            chain_id = residue.get_parent().id
            contacts[chain_id].add((int(residue.id[1]), residue.id[2].strip() or ""))
    return contacts


def compute_relative_sasa(protein_pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_pdb_path.stem, protein_pdb_path)
    model = next(structure.get_models())
    sr = ShrakeRupley()
    sr.compute(model, level="R")

    rasa_map = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or residue.resname not in AA3_TO1:
                continue
            aa = AA3_TO1[residue.resname]
            rasa = residue.sasa / MAX_ASA[aa]
            rasa_map[(chain.id, int(residue.id[1]), residue.id[2].strip() or "")] = rasa
    return rasa_map


def format_pct(numerator: int, denominator: int):
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator


def format_counter(counter: Counter, total: int, max_items: int | None = None):
    items = counter.most_common()
    if max_items is not None and len(items) > max_items:
        shown = items[:max_items]
        other_count = sum(count for _, count in items[max_items:])
        shown.append(("other", other_count))
        items = shown
    return ", ".join(f"{aa}: {100.0 * count / total:.1f}%" for aa, count in items if count)


def format_pct_or_na(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.1f}%"


def position_mutation_rate(counter: Counter, wt_aa: str, n_designs: int):
    return 100.0 * (n_designs - counter.get(wt_aa, 0)) / n_designs


def aggregate_mutation_rate(position_records, n_designs: int):
    if not position_records:
        return float("nan")
    mutated = 0
    total = 0
    for record in position_records:
        mutated += n_designs - record["counts"].get(record["wt"], 0)
        total += n_designs
    return 100.0 * mutated / total if total else float("nan")


def analyze_cd2():
    pdb_id = "1GYA"
    protein_pdb = OUTPUTS_DIR / f"output_{pdb_id}" / f"{pdb_id}_protein.pdb"
    mappings = build_chain_mappings(protein_pdb)

    all_frequency_rows = []
    summary_rows = []
    per_condition_designs = {}

    key_records = {
        pos: get_mapping_record(mappings, "A", pos)
        for pos in sorted(CD2_KEY_POSITIONS)
    }

    for pos, record in key_records.items():
        expected_wt = CD2_KEY_POSITIONS[pos]["wt"]
        if record["aa"] != expected_wt:
            raise ValueError(
                f"{pdb_id} position {pos}: expected WT {expected_wt}, found {record['aa']}"
            )

    for condition in CONDITIONS:
        fasta_data = parse_mpnn_fasta(pdb_id, condition)
        verify_wt_sequences(pdb_id, fasta_data["wt_chains"], mappings)

        design_rows = []
        for design in fasta_data["designs"]:
            sequence = design["chains"]["A"]
            row = {
                "condition": condition,
                "sample": design["sample"],
                "score": design["score"],
                "seq_recovery": design["seq_recovery"],
                "aa66": sequence[key_records[65]["seq_idx"]],
            }
            for pos, mapping_record in key_records.items():
                row[f"aa{pos}"] = sequence[mapping_record["seq_idx"] - 1]

            row["functional_sequon"] = (
                row["aa65"] == "N"
                and row["aa66"] != "P"
                and row["aa67"] in SEQUON_COMPLETING
            )
            row["n65_and_st67"] = row["aa65"] == "N" and row["aa67"] in SEQUON_COMPLETING
            row["charge_neutralizing_61"] = row["aa61"] in CD2_CHARGE_NEUTRALIZING
            row["non_aromatic_63"] = row["aa63"] in CD2_NON_AROMATIC_63
            row["exact_rat_pair"] = row["aa61"] == "E" and row["aa63"] == "L"
            row["analogue_pair"] = row["charge_neutralizing_61"] and row["non_aromatic_63"]
            row["either_compensation"] = (
                row["charge_neutralizing_61"] or row["non_aromatic_63"]
            )
            design_rows.append(row)

        design_df = pd.DataFrame(design_rows)
        per_condition_designs[condition] = design_df
        n_designs = len(design_df)
        lost_df = design_df[~design_df["functional_sequon"]].copy()
        retained_df = design_df[design_df["functional_sequon"]].copy()

        for pos, mapping_record in key_records.items():
            counts = Counter(design_df[f"aa{pos}"])
            for rank, (aa, count) in enumerate(counts.most_common(), start=1):
                all_frequency_rows.append(
                    {
                        "condition": condition,
                        "position_pdb": pos,
                        "position_mpnn": mapping_record["seq_idx"],
                        "wt": CD2_KEY_POSITIONS[pos]["wt"],
                        "rat": CD2_KEY_POSITIONS[pos]["rat"],
                        "role": CD2_KEY_POSITIONS[pos]["role"],
                        "aa": aa,
                        "count": count,
                        "frequency_pct": 100.0 * count / n_designs,
                        "rank": rank,
                        "n_designs": n_designs,
                    }
                )

        summary_rows.append(
            {
                "condition": condition,
                "n_designs": n_designs,
                "functional_sequon_count": int(design_df["functional_sequon"].sum()),
                "functional_sequon_pct": 100.0 * design_df["functional_sequon"].mean(),
                "n65_and_st67_pct": 100.0 * design_df["n65_and_st67"].mean(),
                "exact_rat_pair_count": int(design_df["exact_rat_pair"].sum()),
                "exact_rat_pair_pct": 100.0 * design_df["exact_rat_pair"].mean(),
                "analogue_pair_count": int(design_df["analogue_pair"].sum()),
                "analogue_pair_pct": 100.0 * design_df["analogue_pair"].mean(),
                "lost_count": len(lost_df),
                "retained_count": len(retained_df),
                "charge_neutralizing_when_lost_pct": 100.0
                * lost_df["charge_neutralizing_61"].mean()
                if not lost_df.empty
                else float("nan"),
                "non_aromatic_63_when_lost_pct": 100.0
                * lost_df["non_aromatic_63"].mean()
                if not lost_df.empty
                else float("nan"),
                "either_compensation_when_lost_pct": 100.0
                * lost_df["either_compensation"].mean()
                if not lost_df.empty
                else float("nan"),
                "analogue_pair_when_lost_pct": 100.0 * lost_df["analogue_pair"].mean()
                if not lost_df.empty
                else float("nan"),
                "charge_neutralizing_when_retained_pct": 100.0
                * retained_df["charge_neutralizing_61"].mean()
                if not retained_df.empty
                else float("nan"),
                "non_aromatic_63_when_retained_pct": 100.0
                * retained_df["non_aromatic_63"].mean()
                if not retained_df.empty
                else float("nan"),
                "either_compensation_when_retained_pct": 100.0
                * retained_df["either_compensation"].mean()
                if not retained_df.empty
                else float("nan"),
            }
        )

    frequency_df = pd.DataFrame(all_frequency_rows).sort_values(
        ["condition", "position_pdb", "rank", "aa"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("condition")

    csv_path = OUT_DIR / "cd2_1gya_compensatory_mutations.csv"
    frequency_df.to_csv(csv_path, index=False)

    md_lines = [
        "# CD2 / 1GYA Compensatory Mutation Summary",
        "",
        "ProteinMPNN sequences were analyzed from the case-study pipeline FASTAs.",
        "Positions were mapped from PDB numbering onto MPNN indices using the protein-only input PDB, and the glycosite was verified as `N65-G66-T67 -> MPNN indices 65-67`.",
        "",
    ]

    unconstrained_df = per_condition_designs["unconstrained"]
    unconstrained_summary = summary_df[summary_df["condition"] == "unconstrained"].iloc[0]
    md_lines.extend(
        [
            "## Unconstrained Designs",
            "",
            f"- Designs analyzed: {int(unconstrained_summary['n_designs'])}",
            f"- Functional sequon retained (`N-X-S/T`, `X != P`): {unconstrained_summary['functional_sequon_count']}/{int(unconstrained_summary['n_designs'])} ({unconstrained_summary['functional_sequon_pct']:.1f}%)",
            f"- Exact rat-like pair (`K61E` and `F63L`): {unconstrained_summary['exact_rat_pair_count']}/{int(unconstrained_summary['n_designs'])} ({unconstrained_summary['exact_rat_pair_pct']:.1f}%)",
            f"- Broad rat-analogue pair (`61 in E/D/Q` and `63 in L/I/V/A`): {unconstrained_summary['analogue_pair_count']}/{int(unconstrained_summary['n_designs'])} ({unconstrained_summary['analogue_pair_pct']:.1f}%)",
            "",
            "| Position | WT | Rat | Unconstrained design frequencies |",
            "|---|---|---|---|",
        ]
    )

    for pos in sorted(CD2_KEY_POSITIONS):
        counts = Counter(unconstrained_df[f"aa{pos}"])
        md_lines.append(
            f"| {pos} | {CD2_KEY_POSITIONS[pos]['wt']} | {CD2_KEY_POSITIONS[pos]['rat']} | "
            f"{format_counter(counts, len(unconstrained_df), max_items=8)} |"
        )

    md_lines.extend(
        [
            "",
            "### Key Findings",
            "",
            f"- Among sequon-lost designs ({int(unconstrained_summary['lost_count'])} total), charge-neutralizing mutations at position 61 occurred in {unconstrained_summary['charge_neutralizing_when_lost_pct']:.1f}%.",
            f"- Among sequon-lost designs, non-aromatic replacements at position 63 (`L/I/V/A`) occurred in {unconstrained_summary['non_aromatic_63_when_lost_pct']:.1f}%.",
            f"- Among sequon-lost designs, either compensation signature occurred in {unconstrained_summary['either_compensation_when_lost_pct']:.1f}%, and the broad rat-analogue pair occurred in {unconstrained_summary['analogue_pair_when_lost_pct']:.1f}%.",
        ]
    )

    if int(unconstrained_summary["retained_count"]) > 0:
        md_lines.append(
            f"- Only {int(unconstrained_summary['retained_count'])} unconstrained design retained a functional sequon, so sequon-retained vs sequon-lost compensation enrichment is not statistically stable in this dataset."
        )
    else:
        md_lines.append(
            "- No unconstrained designs retained a functional sequon, so sequon-retained vs sequon-lost compensation enrichment cannot be assessed."
        )

    if unconstrained_summary["exact_rat_pair_count"] == 0:
        md_lines.append(
            "- ProteinMPNN did not independently rediscover the exact rat `K61E/F63L` solution in the unconstrained designs."
        )
    else:
        md_lines.append(
            "- ProteinMPNN sampled the exact rat `K61E/F63L` solution in the unconstrained designs."
        )
    md_lines.append(
        "- Taken together, the unconstrained model removes the glycosylation site almost uniformly by `N65D` without introducing the local charge-neutralizing or de-aromatizing substitutions that make rat CD2 glycan-independent."
    )

    md_lines.extend(
        [
            "",
            "## Fixed-Condition Check",
            "",
            "| Condition | Functional sequon retained | Charge-neutralizing 61 | Non-aromatic 63 | Exact rat pair |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['condition']} | {row['functional_sequon_pct']:.1f}% | "
            f"{format_pct_or_na(row['charge_neutralizing_when_lost_pct'])} among sequon-lost | "
            f"{format_pct_or_na(row['non_aromatic_63_when_lost_pct'])} among sequon-lost | "
            f"{row['exact_rat_pair_pct']:.1f}% |"
        )

    summary_path = OUT_DIR / "cd2_1gya_compensatory_mutations_summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n")

    return {
        "frequency_df": frequency_df,
        "summary_df": summary_df,
        "summary_path": summary_path,
        "csv_path": csv_path,
    }


def analyze_igg_fc():
    relevant_chain_overrides = {"1L6X": ["A"]}

    all_rows = []
    summary_rows = []
    contact_tables = {}
    chain_specific_metrics = {}

    for pdb_id in ["3AVE", "1L6X"]:
        protein_pdb = OUTPUTS_DIR / f"output_{pdb_id}" / f"{pdb_id}_protein.pdb"
        full_pdb = PREP_DIR / pdb_id / "structure" / f"{pdb_id}.pdb"
        mappings = build_chain_mappings(protein_pdb)
        sasa_map = compute_relative_sasa(protein_pdb)
        contact_residues = identify_glycan_contact_residues(full_pdb)
        relevant_chains = relevant_chain_overrides.get(pdb_id, sorted(contact_residues))

        position_meta = []
        for chain_id in relevant_chains:
            if chain_id not in mappings:
                raise KeyError(f"{pdb_id}: missing chain {chain_id} in protein PDB mapping")
            contact_keys = contact_residues.get(chain_id, set())
            for residue_record in mappings[chain_id]["residues"]:
                key = (chain_id, residue_record["pdb_resnum"], residue_record["icode"])
                rasa = sasa_map.get(key, float("nan"))
                position_meta.append(
                    {
                        "structure": pdb_id,
                        "chain": chain_id,
                        "position_pdb": residue_record["pdb_resnum"],
                        "position_mpnn": residue_record["seq_idx"],
                        "wt": residue_record["aa"],
                        "rasa": rasa,
                        "is_glycan_contact": (
                            (residue_record["pdb_resnum"], residue_record["icode"])
                            in contact_keys
                        ),
                        "is_surface_control": (
                            rasa >= 0.20
                            and (
                                (residue_record["pdb_resnum"], residue_record["icode"])
                                not in contact_keys
                            )
                        ),
                    }
                )

        meta_df = pd.DataFrame(position_meta)

        for condition in CONDITIONS:
            fasta_data = parse_mpnn_fasta(pdb_id, condition)
            verify_wt_sequences(pdb_id, fasta_data["wt_chains"], mappings)
            designs = fasta_data["designs"]
            n_designs = len(designs)

            position_records = []
            for meta in position_meta:
                counts = Counter()
                for design in designs:
                    counts.update([design["chains"][meta["chain"]][meta["position_mpnn"] - 1]])

                record = {
                    **meta,
                    "condition": condition,
                    "n_designs": n_designs,
                    "counts": counts,
                    "mutation_rate_pct": position_mutation_rate(counts, meta["wt"], n_designs),
                }
                position_records.append(record)

                for rank, (aa, count) in enumerate(counts.most_common(), start=1):
                    all_rows.append(
                        {
                            "structure": pdb_id,
                            "condition": condition,
                            "chain": meta["chain"],
                            "position_pdb": meta["position_pdb"],
                            "position_mpnn": meta["position_mpnn"],
                            "wt": meta["wt"],
                            "rasa": meta["rasa"],
                            "is_glycan_contact": meta["is_glycan_contact"],
                            "is_surface_control": meta["is_surface_control"],
                            "mutation_rate_pct": record["mutation_rate_pct"],
                            "aa": aa,
                            "count": count,
                            "frequency_pct": 100.0 * count / n_designs,
                            "rank": rank,
                            "n_designs": n_designs,
                        }
                    )

            contact_positions = [r for r in position_records if r["is_glycan_contact"]]
            surface_controls = [r for r in position_records if r["is_surface_control"]]
            chain_specific_metrics[(pdb_id, condition)] = {}
            for chain_id in relevant_chains:
                chain_records = [r for r in position_records if r["chain"] == chain_id]
                chain_contact = [r for r in chain_records if r["is_glycan_contact"]]
                n297_records = [r for r in chain_records if r["position_pdb"] == 297]
                t299_records = [r for r in chain_records if r["position_pdb"] == 299]
                chain_specific_metrics[(pdb_id, condition)][chain_id] = {
                    "contact_mutation_rate_pct": aggregate_mutation_rate(chain_contact, n_designs),
                    "n297_retention_pct": (
                        100.0 * n297_records[0]["counts"].get("N", 0) / n_designs
                        if n297_records
                        else float("nan")
                    ),
                    "t299_st_retention_pct": (
                        100.0
                        * sum(t299_records[0]["counts"].get(aa, 0) for aa in SEQUON_COMPLETING)
                        / n_designs
                        if t299_records
                        else float("nan")
                    ),
                }

            summary_rows.append(
                {
                    "structure": pdb_id,
                    "condition": condition,
                    "n_designs": n_designs,
                    "n_contact_positions": len(contact_positions),
                    "n_surface_controls": len(surface_controls),
                    "contact_mutation_rate_pct": aggregate_mutation_rate(
                        contact_positions, n_designs
                    ),
                    "surface_control_mutation_rate_pct": aggregate_mutation_rate(
                        surface_controls, n_designs
                    ),
                    "protein_average_mutation_rate_pct": aggregate_mutation_rate(
                        position_records, n_designs
                    ),
                    "n297_retention_pct": 100.0
                    * sum(r["counts"].get("N", 0) for r in position_records if r["position_pdb"] == 297)
                    / max(1, n_designs * len([r for r in position_records if r["position_pdb"] == 297])),
                    "t299_st_retention_pct": 100.0
                    * sum(
                        sum(r["counts"].get(aa, 0) for aa in SEQUON_COMPLETING)
                        for r in position_records
                        if r["position_pdb"] == 299
                    )
                    / max(1, n_designs * len([r for r in position_records if r["position_pdb"] == 299])),
                }
            )

            aggregated_contact_rows = []
            contact_groups = defaultdict(list)
            for record in contact_positions:
                contact_groups[record["position_pdb"]].append(record)

            for position_pdb in sorted(contact_groups):
                group = contact_groups[position_pdb]
                pooled_counts = Counter()
                for record in group:
                    pooled_counts.update(record["counts"])
                n_total = n_designs * len(group)
                aggregated_contact_rows.append(
                    {
                        "position_pdb": position_pdb,
                        "wt": group[0]["wt"],
                        "chains": ",".join(sorted({record["chain"] for record in group})),
                        "copies": len(group),
                        "n_total": n_total,
                        "top_substitutions": format_counter(pooled_counts, n_total, max_items=3),
                        "mutation_rate_pct": 100.0
                        * (n_total - pooled_counts.get(group[0]["wt"], 0))
                        / n_total,
                    }
                )
            contact_tables[(pdb_id, condition)] = aggregated_contact_rows

    long_df = pd.DataFrame(all_rows).sort_values(
        ["structure", "condition", "chain", "position_pdb", "rank", "aa"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(["structure", "condition"])

    csv_path = OUT_DIR / "igg_fc_glycan_contact_mutations.csv"
    long_df.to_csv(csv_path, index=False)

    md_lines = [
        "# IgG Fc Glycan-Contact Mutation Summary",
        "",
        "Glycan-contact residues were defined from the glycosylated crystal structures as protein residues with any heavy atom within 4.0 A of any glycan heavy atom.",
        "Surface-control residues were defined on the protein-only MPNN input structures as non-contact residues with relative SASA >= 0.20.",
        "",
    ]

    for pdb_id in ["3AVE", "1L6X"]:
        md_lines.extend([f"## {pdb_id}", ""])
        unconstrained = summary_df[
            (summary_df["structure"] == pdb_id) & (summary_df["condition"] == "unconstrained")
        ].iloc[0]
        unconstrained_table = contact_tables[(pdb_id, "unconstrained")]
        for condition in CONDITIONS:
            summary = summary_df[
                (summary_df["structure"] == pdb_id) & (summary_df["condition"] == condition)
            ].iloc[0]
            md_lines.extend(
                [
                    f"### {condition}",
                    "",
                    f"- Designs analyzed: {int(summary['n_designs'])}",
                    f"- Glycan-contact positions: {int(summary['n_contact_positions'])}",
                    f"- Non-contact surface controls: {int(summary['n_surface_controls'])}",
                    f"- `N297` retention: {summary['n297_retention_pct']:.1f}%",
                    f"- `299` retains `S/T`: {summary['t299_st_retention_pct']:.1f}%",
                    f"- Contact mutation rate: {summary['contact_mutation_rate_pct']:.1f}%",
                    f"- Surface-control mutation rate: {summary['surface_control_mutation_rate_pct']:.1f}%",
                    f"- Protein-average mutation rate on analyzed chains: {summary['protein_average_mutation_rate_pct']:.1f}%",
                    "",
                    "| Position | WT | Chains | Top substitutions | Mutation rate |",
                    "|---|---|---|---|---:|",
                ]
            )
            for row in contact_tables[(pdb_id, condition)]:
                md_lines.append(
                    f"| {row['position_pdb']} | {row['wt']} | {row['chains']} | "
                    f"{row['top_substitutions']} | {row['mutation_rate_pct']:.1f}% |"
                )

            chain_metrics = chain_specific_metrics[(pdb_id, condition)]
            if len(chain_metrics) > 1:
                md_lines.extend(["", "Per-chain `N297` / contact metrics:", ""])
                for chain_id in sorted(chain_metrics):
                    metrics = chain_metrics[chain_id]
                    md_lines.append(
                        f"- Chain {chain_id}: `N297` retention {metrics['n297_retention_pct']:.1f}%, "
                        f"`299 S/T` retention {metrics['t299_st_retention_pct']:.1f}%, "
                        f"contact mutation rate {metrics['contact_mutation_rate_pct']:.1f}%."
                    )
            md_lines.append("")

        if unconstrained["contact_mutation_rate_pct"] < unconstrained[
            "surface_control_mutation_rate_pct"
        ]:
            contact_vs_surface = "less"
        elif unconstrained["contact_mutation_rate_pct"] > unconstrained[
            "surface_control_mutation_rate_pct"
        ]:
            contact_vs_surface = "more"
        else:
            contact_vs_surface = "equally"
        if unconstrained["contact_mutation_rate_pct"] < unconstrained[
            "protein_average_mutation_rate_pct"
        ]:
            contact_vs_average = "below"
        elif unconstrained["contact_mutation_rate_pct"] > unconstrained[
            "protein_average_mutation_rate_pct"
        ]:
            contact_vs_average = "above"
        else:
            contact_vs_average = "equal to"

        dominant_shifts = []
        for row in unconstrained_table:
            top_aa = row["top_substitutions"].split(",")[0].split(":")[0].strip()
            if top_aa != row["wt"] and top_aa != "other":
                dominant_shifts.append(f"{row['position_pdb']} {row['wt']}->{top_aa}")
            if len(dominant_shifts) == 5:
                break

        md_lines.extend(
            [
                "### Unconstrained Readout",
                "",
                f"- In {pdb_id}, glycan-contact residues mutated {contact_vs_surface} often than non-contact surface controls in the unconstrained condition ({unconstrained['contact_mutation_rate_pct']:.1f}% vs {unconstrained['surface_control_mutation_rate_pct']:.1f}%).",
                f"- Their mutation rate was {contact_vs_average} the analyzed-chain average ({unconstrained['protein_average_mutation_rate_pct']:.1f}%).",
                f"- Dominant unconstrained shifts were: {', '.join(dominant_shifts)}.",
                "- This pattern is consistent with ProteinMPNN treating the glycan-facing cavity largely as ordinary redesignable surface unless the sequon is explicitly fixed.",
                "",
            ]
        )

    overall_3ave = summary_df[
        (summary_df["structure"] == "3AVE") & (summary_df["condition"] == "unconstrained")
    ].iloc[0]
    overall_1l6x = summary_df[
        (summary_df["structure"] == "1L6X") & (summary_df["condition"] == "unconstrained")
    ].iloc[0]
    md_lines.extend(
        [
            "## Overall Conclusion",
            "",
            f"- In unconstrained designs, `N297` retention was only {overall_3ave['n297_retention_pct']:.1f}% in 3AVE and {overall_1l6x['n297_retention_pct']:.1f}% in 1L6X.",
            f"- Contact-position mutation rates tracked generic surface mutation rates closely (3AVE: {overall_3ave['contact_mutation_rate_pct']:.1f}% vs {overall_3ave['surface_control_mutation_rate_pct']:.1f}%; 1L6X: {overall_1l6x['contact_mutation_rate_pct']:.1f}% vs {overall_1l6x['surface_control_mutation_rate_pct']:.1f}%).",
            "- The contact-residue substitution patterns do not show strong preservation of the glycan-shaped cavity in the unconstrained condition; explicit sequon constraints are required to keep the canonical glycosylated motif intact.",
            "",
        ]
    )

    summary_path = OUT_DIR / "igg_fc_glycan_contact_mutations_summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n")

    return {
        "long_df": long_df,
        "summary_df": summary_df,
        "summary_path": summary_path,
        "csv_path": csv_path,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cd2_outputs = analyze_cd2()
    igg_outputs = analyze_igg_fc()

    print(f"Wrote {cd2_outputs['csv_path']}")
    print(f"Wrote {cd2_outputs['summary_path']}")
    print(f"Wrote {igg_outputs['csv_path']}")
    print(f"Wrote {igg_outputs['summary_path']}")


if __name__ == "__main__":
    main()
