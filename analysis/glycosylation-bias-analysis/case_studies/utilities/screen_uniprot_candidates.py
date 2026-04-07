#!/usr/bin/env python3
"""Screen UniProt glycoprotein TSV for suitable pipeline candidates.

Filters for proteins with:
- Experimentally validated N-linked glycosylation (ECO:0000269 or ECO:0007744)
- Available PDB structures
- Checks PDB quality (missing residues, completeness)

Usage:
    python 00_screen_uniprot_candidates.py \
        --tsv /path/to/uniprotkb_glycoproteins.tsv \
        --output candidates.csv \
        [--check-pdb]  # downloads PDBs to check missing residues (slower)
        [--max-proteins 100]  # limit for PDB checking
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve


def parse_glycosylation_column(glyco_text):
    """Parse the UniProt Glycosylation column.

    Returns list of dicts with keys: position, glyco_type, evidence_codes.
    """
    if not glyco_text or glyco_text.strip() == "":
        return []

    sites = []
    # Split on "CARBOHYD" entries
    entries = re.split(r"(?=CARBOHYD\s)", glyco_text)

    for entry in entries:
        entry = entry.strip()
        if not entry.startswith("CARBOHYD"):
            continue

        # Extract position
        pos_match = re.match(r"CARBOHYD\s+(\d+)", entry)
        if not pos_match:
            continue
        position = int(pos_match.group(1))

        # Determine glycosylation type
        glyco_type = "unknown"
        if "N-linked" in entry:
            glyco_type = "N-linked"
        elif "O-linked" in entry:
            glyco_type = "O-linked"
        elif "C-linked" in entry:
            glyco_type = "C-linked"
        elif "S-linked" in entry:
            glyco_type = "S-linked"

        # Extract evidence codes
        evidence_codes = set()
        for eco_match in re.finditer(r"ECO:(\d+)", entry):
            evidence_codes.add(f"ECO:{eco_match.group(1)}")

        sites.append({
            "position": position,
            "glyco_type": glyco_type,
            "evidence_codes": evidence_codes,
        })

    return sites


def parse_pdb_column(pdb_text):
    """Parse PDB IDs from the PDB column.

    Returns list of PDB IDs (4 characters).
    """
    if not pdb_text or pdb_text.strip() == "":
        return []
    return [p.strip() for p in pdb_text.split(";") if len(p.strip()) == 4]


def parse_missing_residues_from_pdb(pdb_path):
    """Parse REMARK 465 to count missing residues per chain."""
    missing = {}
    in_remark465 = False
    header_seen = False

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("REMARK 465"):
                if in_remark465:
                    break
                continue
            in_remark465 = True
            content = line[10:].strip()
            if not content or content.startswith("THE FOLLOWING") or content.startswith("EXPERIMENT"):
                continue
            if content.startswith("M RES"):
                header_seen = True
                continue
            if not header_seen:
                continue
            parts = content.split()
            if len(parts) >= 3:
                try:
                    if parts[0].isdigit():
                        chain, resnum = parts[2], int(parts[3])
                    else:
                        chain, resnum = parts[1], int(parts[2])
                    missing.setdefault(chain, []).append(resnum)
                except (ValueError, IndexError):
                    continue

    return missing


def get_sequence_length_from_pdb(pdb_path):
    """Get total resolved protein residues from a PDB file."""
    try:
        from Bio.PDB import PDBParser, is_aa
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        total = 0
        for model in structure:
            for chain in model:
                residues = [r for r in chain.get_residues() if is_aa(r)]
                total += len(residues)
            break  # first model only
        return total
    except Exception:
        return 0


def parse_seqres_length(pdb_path):
    """Parse SEQRES records to get expected chain lengths."""
    chain_lengths = defaultdict(int)
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("SEQRES"):
                chain = line[11:12].strip()
                num_res = int(line[13:17].strip())
                chain_lengths[chain] = num_res
    return dict(chain_lengths)


def check_missing_near_sequons(missing_residues, glyco_sites, tolerance=5):
    """Check if any missing residues are near glycosylation sites.

    glyco_sites: list of position ints (1-indexed, from UniProt).
    missing_residues: dict chain -> list of resnum ints.
    """
    all_missing = set()
    for chain_missing in missing_residues.values():
        all_missing.update(chain_missing)

    for site_pos in glyco_sites:
        for offset in range(-tolerance, tolerance + 1):
            if site_pos + offset in all_missing:
                return True
    return False


def download_pdb_temp(pdb_id):
    """Download PDB to a temp file. Returns path or None."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        urlretrieve(url, tmp.name)
        return tmp.name
    except Exception:
        return None


def screen_candidates(tsv_path, check_pdb=False, max_proteins=None):
    """Screen UniProt TSV for pipeline candidates.

    Returns list of candidate dicts.
    """
    candidates = []

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row_idx, row in enumerate(reader):
            uniprot_id = row.get("Entry", "")
            protein_name = row.get("Protein names", "")
            organism = row.get("Organism", "")
            length = int(row.get("Length", 0))
            glyco_text = row.get("Glycosylation", "")
            pdb_text = row.get("PDB", "")

            # Parse glycosylation sites
            glyco_sites = parse_glycosylation_column(glyco_text)

            # Filter: must have N-linked sites
            n_linked = [s for s in glyco_sites if s["glyco_type"] == "N-linked"]
            if not n_linked:
                continue

            # Filter: must have experimental evidence
            EXPERIMENTAL_CODES = {"ECO:0000269", "ECO:0007744", "ECO:0000305"}
            experimental_sites = [
                s for s in n_linked
                if s["evidence_codes"] & EXPERIMENTAL_CODES
            ]
            if not experimental_sites:
                continue

            # Filter: must have PDB structures
            pdb_ids = parse_pdb_column(pdb_text)
            if not pdb_ids:
                continue

            # Count evidence quality
            pdb_evidence_sites = [
                s for s in n_linked
                if "ECO:0007744" in s["evidence_codes"]
            ]

            glyco_positions = [s["position"] for s in n_linked]

            # Build per-site evidence detail: "pos:tier" pairs
            # Tiers: experimental (ECO:0000269 lit), pdb_evidence (ECO:0007744),
            #        curator_inferred (ECO:0000305)
            glyco_evidence_details = []
            for s in n_linked:
                codes = s["evidence_codes"]
                if "ECO:0000269" in codes:
                    tier = "experimental"
                elif "ECO:0007744" in codes:
                    tier = "pdb_evidence"
                elif "ECO:0000305" in codes:
                    tier = "curator_inferred"
                else:
                    tier = "predicted"
                glyco_evidence_details.append(f"{s['position']}:{tier}")

            candidate = {
                "uniprot_id": uniprot_id,
                "protein_name": protein_name[:80],
                "organism": organism,
                "length": length,
                "n_glyco_sites_total": len(n_linked),
                "n_sites_experimental": len(experimental_sites),
                "n_sites_pdb_evidence": len(pdb_evidence_sites),
                "glyco_positions": ";".join(str(p) for p in glyco_positions),
                "glyco_evidence": ";".join(glyco_evidence_details),
                "pdb_ids": ";".join(pdb_ids[:5]),  # first 5
                "n_pdb_structures": len(pdb_ids),
                "best_pdb": pdb_ids[0] if pdb_ids else "",
            }

            # PDB quality check (optional, slower)
            if check_pdb:
                if max_proteins and len(candidates) >= max_proteins:
                    candidate["pdb_checked"] = False
                    candidate["n_missing_residues"] = -1
                    candidate["missing_near_sequons"] = "unchecked"
                    candidate["completeness_pct"] = -1
                    candidate["recommended"] = False
                else:
                    pdb_path = download_pdb_temp(pdb_ids[0])
                    if pdb_path:
                        try:
                            missing = parse_missing_residues_from_pdb(pdb_path)
                            total_missing = sum(len(v) for v in missing.values())
                            near_sequons = check_missing_near_sequons(
                                missing, glyco_positions
                            )
                            seqres = parse_seqres_length(pdb_path)
                            expected_total = sum(seqres.values()) if seqres else length
                            resolved = get_sequence_length_from_pdb(pdb_path)
                            completeness = (resolved / expected_total * 100) if expected_total > 0 else 0

                            candidate["pdb_checked"] = True
                            candidate["n_missing_residues"] = total_missing
                            candidate["missing_near_sequons"] = near_sequons
                            candidate["completeness_pct"] = round(completeness, 1)
                            candidate["recommended"] = (
                                not near_sequons
                                and completeness > 95
                                and len(experimental_sites) >= 1
                            )
                        except Exception as e:
                            candidate["pdb_checked"] = False
                            candidate["n_missing_residues"] = -1
                            candidate["missing_near_sequons"] = f"error: {e}"
                            candidate["completeness_pct"] = -1
                            candidate["recommended"] = False
                        finally:
                            os.unlink(pdb_path)
                    else:
                        candidate["pdb_checked"] = False
                        candidate["n_missing_residues"] = -1
                        candidate["missing_near_sequons"] = "download_failed"
                        candidate["completeness_pct"] = -1
                        candidate["recommended"] = False
            else:
                # Without PDB check, recommend based on evidence quality
                candidate["pdb_checked"] = False
                candidate["recommended"] = len(pdb_evidence_sites) >= 1

            candidates.append(candidate)

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Screen UniProt glycoprotein TSV for pipeline candidates."
    )
    parser.add_argument("--tsv", required=True, help="UniProt TSV file path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--check-pdb", action="store_true",
        help="Download PDBs to check for missing residues (slower)"
    )
    parser.add_argument(
        "--max-proteins", type=int, default=None,
        help="Max proteins to PDB-check (for --check-pdb mode)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SCREENING UNIPROT GLYCOPROTEINS FOR PIPELINE CANDIDATES")
    print("=" * 70)

    print(f"\nInput: {args.tsv}")
    print(f"PDB checking: {'enabled' if args.check_pdb else 'disabled (fast mode)'}")

    candidates = screen_candidates(
        args.tsv,
        check_pdb=args.check_pdb,
        max_proteins=args.max_proteins,
    )

    # Sort by quality: PDB evidence sites (desc), then total experimental sites
    candidates.sort(
        key=lambda c: (c["n_sites_pdb_evidence"], c["n_sites_experimental"]),
        reverse=True,
    )

    # Write output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "uniprot_id", "protein_name", "organism", "length",
        "n_glyco_sites_total", "n_sites_experimental", "n_sites_pdb_evidence",
        "glyco_positions", "glyco_evidence",
        "pdb_ids", "n_pdb_structures", "best_pdb",
        "recommended",
    ]
    if args.check_pdb:
        fieldnames.extend([
            "pdb_checked", "n_missing_residues",
            "missing_near_sequons", "completeness_pct",
        ])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    n_recommended = sum(1 for c in candidates if c.get("recommended"))
    print(f"\nResults:")
    print(f"  Total candidates with experimental N-linked glycosylation + PDB: {len(candidates)}")
    print(f"  Recommended: {n_recommended}")
    print(f"  Output: {output_path}")

    # Print top 20 recommendations
    top = [c for c in candidates if c.get("recommended")][:20]
    if top:
        print(f"\nTop {len(top)} recommended proteins:")
        print(f"  {'UniProt':<15} {'PDB':<8} {'Sites(exp)':<12} {'Sites(PDB)':<12} {'Length':<8} Name")
        print(f"  {'-'*15} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*40}")
        for c in top:
            print(f"  {c['uniprot_id']:<15} {c['best_pdb']:<8} "
                  f"{c['n_sites_experimental']:<12} {c['n_sites_pdb_evidence']:<12} "
                  f"{c['length']:<8} {c['protein_name'][:40]}")


if __name__ == "__main__":
    main()
