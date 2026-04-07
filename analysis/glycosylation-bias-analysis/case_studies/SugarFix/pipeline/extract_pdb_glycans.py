#!/usr/bin/env python3
"""Extract glycan trees from PDB LINK records.

Parses LINK and HETATM records to reconstruct the glycan chains
attached to protein ASN residues (N-linked glycosylation).

Usage:
    python extract_pdb_glycans.py --pdb_path structure.pdb --output glycan_trees.json

Output JSON maps each glycosylation site to its observed glycan tree:
{
  "A:312": {
    "protein_chain": "A",
    "protein_resnum": 312,
    "protein_resname": "ASN",
    "glycan_chain": "D",
    "residues": ["NAG", "NAG"],
    "residues_string": "NAG-NAG",
    "n_sugars": 2,
    "bonds": [
      {"from_res_idx": 1, "from_atom": "O4", "to_res_idx": 2, "to_atom": "C1"}
    ]
  },
  ...
}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError:
    MMCIF2Dict = None


def _as_list(value):
    """Normalize scalar-or-list mmCIF values into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_link_records(pdb_path):
    """Parse covalent link records from a PDB or mmCIF file.

    Returns list of dicts with source and destination atom info.
    """
    pdb_path = Path(pdb_path)
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        return parse_struct_conn_records_mmcif(pdb_path)

    links = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("LINK"):
                continue
            # LINK record format (PDB v3.3):
            # Columns 13-16: Atom name 1
            # Columns 17-17: Alternate location 1
            # Columns 18-20: Residue name 1
            # Columns 22-22: Chain ID 1
            # Columns 23-26: Residue sequence number 1
            # Columns 27-27: Insertion code 1
            # Columns 43-46: Atom name 2
            # Columns 47-47: Alternate location 2
            # Columns 48-50: Residue name 2
            # Columns 52-52: Chain ID 2
            # Columns 53-56: Residue sequence number 2
            # Columns 57-57: Insertion code 2
            try:
                atom1 = line[12:16].strip()
                resname1 = line[17:20].strip()
                chain1 = line[21:22].strip()
                resnum1 = int(line[22:26].strip())

                atom2 = line[42:46].strip()
                resname2 = line[47:50].strip()
                chain2 = line[51:52].strip()
                resnum2 = int(line[52:56].strip())

                links.append({
                    "atom1": atom1, "resname1": resname1,
                    "chain1": chain1, "resnum1": resnum1,
                    "atom2": atom2, "resname2": resname2,
                    "chain2": chain2, "resnum2": resnum2,
                })
            except (ValueError, IndexError):
                continue

    return links


def parse_struct_conn_records_mmcif(cif_path):
    """Parse _struct_conn records from an mmCIF file into LINK-like dicts."""
    if MMCIF2Dict is None:
        return []

    mmcif = MMCIF2Dict(str(cif_path))
    comp1 = _as_list(mmcif.get("_struct_conn.ptnr1_auth_comp_id"))
    comp2 = _as_list(mmcif.get("_struct_conn.ptnr2_auth_comp_id"))
    atom1 = _as_list(mmcif.get("_struct_conn.ptnr1_label_atom_id"))
    atom2 = _as_list(mmcif.get("_struct_conn.ptnr2_label_atom_id"))
    chain1 = _as_list(mmcif.get("_struct_conn.ptnr1_auth_asym_id"))
    chain2 = _as_list(mmcif.get("_struct_conn.ptnr2_auth_asym_id"))
    seq1 = _as_list(mmcif.get("_struct_conn.ptnr1_auth_seq_id"))
    seq2 = _as_list(mmcif.get("_struct_conn.ptnr2_auth_seq_id"))

    n_rows = max(
        len(comp1), len(comp2), len(atom1), len(atom2),
        len(chain1), len(chain2), len(seq1), len(seq2)
    )
    links = []
    for i in range(n_rows):
        try:
            resnum1 = int(seq1[i])
            resnum2 = int(seq2[i])
        except (TypeError, ValueError, IndexError):
            continue

        links.append(
            {
                "atom1": atom1[i] if i < len(atom1) else "",
                "resname1": comp1[i] if i < len(comp1) else "",
                "chain1": chain1[i] if i < len(chain1) else "",
                "resnum1": resnum1,
                "atom2": atom2[i] if i < len(atom2) else "",
                "resname2": comp2[i] if i < len(comp2) else "",
                "chain2": chain2[i] if i < len(chain2) else "",
                "resnum2": resnum2,
            }
        )

    return links


# Common N-linked glycan sugar CCD codes
GLYCAN_RESIDUES = {
    "NAG",  # N-acetylglucosamine (GlcNAc)
    "BMA",  # Beta-D-mannose
    "MAN",  # Alpha-D-mannose
    "FUC",  # Fucose
    "GAL",  # Galactose
    "SIA",  # Sialic acid
    "NGA",  # N-acetylgalactosamine (GalNAc)
    "BGC",  # Beta-D-glucose
    "GLC",  # Alpha-D-glucose
    "XYS",  # Xylose
    "A2G",  # N-acetyl-D-galactosamine
    "NDG",  # 2-(acetylamino)-2-deoxy-alpha-D-glucopyranose
}


def extract_glycan_trees(pdb_path):
    """Extract glycan trees attached to protein ASN residues.

    Follows LINK records from ASN -> NAG -> subsequent sugars to build
    the complete glycan tree.

    Returns dict mapping "chain:resnum" -> glycan info dict.
    """
    links = parse_link_records(pdb_path)

    # Step 1: Find protein ASN ND2 -> NAG C1 links (root of N-linked glycans)
    asn_nag_links = []
    for link in links:
        if (
            link["resname1"] == "ASN"
            and link["atom1"] == "ND2"
            and link["resname2"] == "NAG"
            and link["atom2"] == "C1"
        ):
            asn_nag_links.append(link)
        elif (
            link["resname2"] == "ASN"
            and link["atom2"] == "ND2"
            and link["resname1"] == "NAG"
            and link["atom1"] == "C1"
        ):
            asn_nag_links.append({
                "atom1": link["atom2"], "resname1": link["resname2"],
                "chain1": link["chain2"], "resnum1": link["resnum2"],
                "atom2": link["atom1"], "resname2": link["resname1"],
                "chain2": link["chain1"], "resnum2": link["resnum1"],
            })

    # Step 2: Build directed parent -> child edges for sugar-sugar links.
    #
    # Glycosidic bonds are rooted at the child's anomeric carbon (C1). We
    # normalize each LINK record so the stored direction is always:
    #   parent(from_atom) -> child(C1)
    # This guarantees that a BFS from the root NAG yields parent-first order.
    sugar_children = defaultdict(list)
    for link in links:
        is_sugar1 = link["resname1"] in GLYCAN_RESIDUES
        is_sugar2 = link["resname2"] in GLYCAN_RESIDUES
        if is_sugar1 and is_sugar2:
            key1 = (link["chain1"], link["resnum1"], link["resname1"])
            key2 = (link["chain2"], link["resnum2"], link["resname2"])

            if link["atom2"] == "C1" and link["atom1"] != "C1":
                sugar_children[key1].append({
                    "child": key2,
                    "from_atom": link["atom1"],
                    "to_atom": link["atom2"],
                })
            elif link["atom1"] == "C1" and link["atom2"] != "C1":
                sugar_children[key2].append({
                    "child": key1,
                    "from_atom": link["atom2"],
                    "to_atom": link["atom1"],
                })

    # Step 3: For each ASN-NAG root, traverse the glycan tree
    glycan_trees = {}
    for root_link in asn_nag_links:
        protein_chain = root_link["chain1"]
        protein_resnum = root_link["resnum1"]
        nag_chain = root_link["chain2"]
        nag_resnum = root_link["resnum2"]

        site_key = f"{protein_chain}:{protein_resnum}"

        # BFS to collect all sugars in this tree
        root_sugar = (nag_chain, nag_resnum, "NAG")
        visited = set()
        queue = [root_sugar]
        ordered_residues = []
        tree_edges = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ordered_residues.append(current)

            for edge in sugar_children.get(current, []):
                child = edge["child"]
                if child in visited:
                    continue
                tree_edges.append({
                    "parent": current,
                    "child": child,
                    "from_atom": edge["from_atom"],
                    "to_atom": edge["to_atom"],
                })
                queue.append(child)

        # Build residue list (CCD codes in BFS order from root)
        residue_codes = [r[2] for r in ordered_residues]
        residue_indices = {
            residue: idx for idx, residue in enumerate(ordered_residues, start=1)
        }
        bonds = []
        for edge in tree_edges:
            parent_idx = residue_indices.get(edge["parent"])
            child_idx = residue_indices.get(edge["child"])
            if parent_idx is None or child_idx is None:
                continue
            bonds.append({
                "from_res_idx": parent_idx,
                "from_atom": edge["from_atom"],
                "to_res_idx": child_idx,
                "to_atom": edge["to_atom"],
            })

        # Build a human-readable string: NAG-NAG-BMA-MAN-MAN...
        residues_string = "-".join(residue_codes)

        glycan_trees[site_key] = {
            "protein_chain": protein_chain,
            "protein_resnum": protein_resnum,
            "protein_resname": "ASN",
            "glycan_chain": nag_chain,
            "residues": residue_codes,
            "residues_string": residues_string,
            "n_sugars": len(residue_codes),
            "bonds": bonds,
        }

    return glycan_trees


def main():
    parser = argparse.ArgumentParser(
        description="Extract glycan trees from PDB LINK records."
    )
    parser.add_argument("--pdb_path", required=True, help="Path to PDB file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    pdb_path = Path(args.pdb_path)
    if not pdb_path.exists():
        print(f"ERROR: PDB file not found: {pdb_path}")
        return

    glycan_trees = extract_glycan_trees(pdb_path)

    if not glycan_trees:
        print(f"  No N-linked glycan trees found in {pdb_path.name}")
    else:
        print(f"  Found {len(glycan_trees)} glycosylation site(s):")
        for site_key, info in sorted(glycan_trees.items()):
            print(f"    {site_key}: {info['residues_string']} ({info['n_sugars']} sugars)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(glycan_trees, f, indent=2)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
