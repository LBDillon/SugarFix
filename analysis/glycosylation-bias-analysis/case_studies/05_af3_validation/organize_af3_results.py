#!/usr/bin/env python3
"""Organize AF3 Server output folders into a clean per-protein structure
for PyMOL RMSD analysis and confidence comparison.

Usage:
    python organize_af3_results.py /path/to/folds_download [--output-dir af3_results]

What it does:
  1. Scans the download directory for AF3 output folders
  2. Groups them by PDB ID and condition
  3. For each protein, copies the top-ranked model (.cif) into a flat structure
  4. Extracts confidence metrics (pTM, ipTM, ranking_score) into a summary CSV
  5. Generates a PyMOL .pml script per protein that loads the crystal structure
     and all AF3 models, ready for align_designs

Output structure:
    af3_results/
    ├── confidence_summary.csv          # All proteins, all conditions
    ├── 4ARN/
    │   ├── models/                     # Top-ranked .cif files (clean names)
    │   │   ├── 4ARN_unconstrained.cif
    │   │   ├── 4ARN_full_sequon_fixed.cif
    │   │   ├── 4ARN_unconstrained_with_glycans.cif
    │   │   ├── 4ARN_full_sequon_fixed_with_glycans.cif
    │   │   └── 4ARN_unconstrained_denovo_glycans.cif
    │   ├── confidences/                # All 5 seed confidence JSONs
    │   ├── load_in_pymol.pml           # PyMOL script to load + align
    │   └── confidence.csv              # Per-protein confidence table
    ├── 1DBN/
    │   └── ...
    └── ...
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"

# Map folder name patterns to clean condition labels
CONDITION_PATTERNS = [
    # Order matters: more specific patterns first
    ("full_sequon_fixed_top1_full_glycans", "full_sequon_fixed_full_glycans"),
    ("full_sequon_fixed_full_glycans", "full_sequon_fixed_full_glycans"),
    ("unconstrained_full_glycans", "unconstrained_full_glycans"),
    ("unconstrained_top1_denovo_glycans", "unconstrained_denovo_glycans"),
    ("unconstrained_top1_with_glycans", "unconstrained_with_glycans"),
    ("full_sequon_fixed_top1_with_glycans", "full_sequon_fixed_with_glycans"),
    ("full_sequon_fixed_top1", "full_sequon_fixed"),
    ("unconstrained_top1", "unconstrained"),
    # Handle 1ruz-style naming (no _protein_ prefix)
    ("unconstrained_top1_denovo_glycans", "unconstrained_denovo_glycans"),
    ("full_sequon_fixed_top1_with_glycans", "full_sequon_fixed_with_glycans"),
    ("full_sequon_fixed_top1", "full_sequon_fixed"),
    ("unconstrained_top1", "unconstrained"),
]


def parse_folder_name(folder_name):
    """Extract PDB ID and condition from an AF3 output folder name.

    Returns (pdb_id_upper, condition_label) or (None, None) if unrecognized.
    """
    name = folder_name.lower()

    # Skip non-fold directories
    if name in ("msas", "templates", "terms_of_use.md"):
        return None, None

    # Try standard pattern: {pdb}_protein_{condition}
    # or 1ruz-style: {pdb}_{condition} (no _protein_)
    timestamp_suffix = r"(?:_\d{8}_\d{6})?"
    for suffix, label in CONDITION_PATTERNS:
        # With _protein_ prefix
        m = re.match(
            r"^([a-z0-9]{4})_protein_" + re.escape(suffix) + timestamp_suffix + r"$",
            name,
        )
        if m:
            return m.group(1).upper(), label
        # Without _protein_ prefix (e.g., 1ruz_unconstrained_top1)
        m = re.match(
            r"^([a-z0-9]{4})_" + re.escape(suffix) + timestamp_suffix + r"$",
            name,
        )
        if m:
            return m.group(1).upper(), label

    # Handle special cases like "1c1z_unfixed_rank03_score0_785291"
    m = re.match(r"^([a-z0-9]{4})_unfixed_", name)
    if m:
        return m.group(1).upper(), "unconstrained_custom"

    # Handle with_glycans_2 style (e.g., 1ruz_full_sequon_fixed_top1_with_glycans_2)
    m = re.match(r"^([a-z0-9]{4})(?:_protein)?_full_sequon_fixed_top1_with_glycans_\d+$", name)
    if m:
        return m.group(1).upper(), "full_sequon_fixed_with_glycans"

    return None, None


def find_best_model(folder_path):
    """Find the top-ranked model .cif file in an AF3 output folder."""
    folder = Path(folder_path)

    # Open-source AF3 writes the top-ranked model at the root of the job dir.
    root_models = sorted(folder.glob("*_model.cif"))
    if root_models:
        return root_models[0], "top-ranked"

    # AF3 server outputs model_0 through model_4, where 0 is best
    for i in range(5):
        candidates = list(folder.glob(f"*_model_{i}.cif"))
        if candidates:
            return candidates[0], i
    return None, None


def extract_confidences(folder_path):
    """Extract confidence metrics from server or local AF3 outputs."""
    folder = Path(folder_path)
    confidences = []

    # Open-source AF3 writes one summary JSON per seed/sample directory.
    sample_dirs = sorted(folder.glob("seed-*_sample-*"))
    if sample_dirs:
        for sample_dir in sample_dirs:
            summary_files = sorted(sample_dir.glob("*_summary_confidences.json"))
            if not summary_files:
                continue

            with open(summary_files[0]) as f:
                data = json.load(f)

            match = re.match(r"seed-(\d+)_sample-(\d+)$", sample_dir.name)
            seed = int(match.group(1)) if match else None
            sample = int(match.group(2)) if match else None
            confidences.append({
                "seed": seed,
                "sample": sample,
                "ptm": data.get("ptm"),
                "iptm": data.get("iptm"),
                "ranking_score": data.get("ranking_score"),
                "fraction_disordered": data.get("fraction_disordered"),
                "has_clash": data.get("has_clash"),
                "chain_ptm": data.get("chain_ptm", []),
            })
        root_summaries = sorted(folder.glob("*_summary_confidences.json"))
        if root_summaries:
            with open(root_summaries[0]) as f:
                data = json.load(f)
            confidences.append({
                "seed": None,
                "sample": None,
                "ptm": data.get("ptm"),
                "iptm": data.get("iptm"),
                "ranking_score": data.get("ranking_score"),
                "fraction_disordered": data.get("fraction_disordered"),
                "has_clash": data.get("has_clash"),
                "chain_ptm": data.get("chain_ptm", []),
            })
        return confidences

    # AF3 server outputs summary_confidences_0..4.json at the folder root.
    for i in range(5):
        candidates = list(folder.glob(f"*_summary_confidences_{i}.json"))
        if not candidates:
            continue
        with open(candidates[0]) as f:
            data = json.load(f)
        confidences.append({
            "seed": i,
            "sample": None,
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
            "chain_ptm": data.get("chain_ptm", []),
        })

    # Fallback: top-ranked local summary at the root only.
    if not confidences:
        candidates = sorted(folder.glob("*_summary_confidences.json"))
        if candidates:
            with open(candidates[0]) as f:
                data = json.load(f)
            confidences.append({
                "seed": None,
                "sample": None,
                "ptm": data.get("ptm"),
                "iptm": data.get("iptm"),
                "ranking_score": data.get("ranking_score"),
                "fraction_disordered": data.get("fraction_disordered"),
                "has_clash": data.get("has_clash"),
                "chain_ptm": data.get("chain_ptm", []),
            })
    return confidences


def organize(download_dir, output_dir):
    """Main organization routine."""
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)

    if not download_dir.exists():
        print(f"ERROR: Download directory not found: {download_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all AF3 output folders
    proteins = {}  # {PDB_ID: {condition: folder_path}}
    skipped = []

    for item in sorted(download_dir.iterdir()):
        if not item.is_dir():
            continue
        pdb_id, condition = parse_folder_name(item.name)
        if pdb_id is None:
            skipped.append(item.name)
            continue

        if pdb_id not in proteins:
            proteins[pdb_id] = {}
        proteins[pdb_id][condition] = item

    print(f"Found {len(proteins)} proteins across {sum(len(v) for v in proteins.values())} AF3 predictions")
    if skipped:
        print(f"  Skipped {len(skipped)} unrecognized folders: {', '.join(skipped)}")

    # Process each protein
    all_confidence_rows = []

    for pdb_id in sorted(proteins.keys()):
        conditions = proteins[pdb_id]
        pdb_dir = output_dir / pdb_id
        models_dir = pdb_dir / "models"
        conf_dir = pdb_dir / "confidences"
        models_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {pdb_id}: {len(conditions)} conditions")
        protein_conf_rows = []

        for condition in sorted(conditions.keys()):
            src_folder = conditions[condition]

            # Copy best model
            best_cif, seed_idx = find_best_model(src_folder)
            if best_cif:
                dest_name = f"{pdb_id}_{condition}.cif"
                dest_path = models_dir / dest_name
                shutil.copy2(best_cif, dest_path)
                print(f"    {condition}: {dest_name} ({seed_idx})")
            else:
                print(f"    {condition}: WARNING - no .cif model found!")

            # Copy confidence JSONs from both server and local AF3 layouts.
            copied_conf = False
            for conf_json in sorted(src_folder.glob("*_summary_confidences_*.json")):
                m = re.search(r"_(\d+)\.json$", conf_json.name)
                if m:
                    dest_conf = conf_dir / f"{pdb_id}_{condition}_seed{m.group(1)}.json"
                else:
                    dest_conf = conf_dir / conf_json.name
                shutil.copy2(conf_json, dest_conf)
                copied_conf = True

            for sample_dir in sorted(src_folder.glob("seed-*_sample-*")):
                match = re.match(r"seed-(\d+)_sample-(\d+)$", sample_dir.name)
                for conf_json in sorted(sample_dir.glob("*_summary_confidences.json")):
                    if match:
                        dest_conf = conf_dir / (
                            f"{pdb_id}_{condition}_seed{match.group(1)}_"
                            f"sample{match.group(2)}.json"
                        )
                    else:
                        dest_conf = conf_dir / conf_json.name
                    shutil.copy2(conf_json, dest_conf)
                    copied_conf = True

            for conf_json in sorted(src_folder.glob("*_summary_confidences.json")):
                dest_conf = conf_dir / f"{pdb_id}_{condition}_best.json"
                shutil.copy2(conf_json, dest_conf)
                copied_conf = True
                break

            if not copied_conf:
                print(f"    {condition}: WARNING - no confidence JSONs found!")

            # Extract confidences
            confidences = extract_confidences(src_folder)
            for conf in confidences:
                row = {
                    "pdb_id": pdb_id,
                    "condition": condition,
                    "seed": conf["seed"],
                    "sample": conf.get("sample"),
                    "ptm": conf["ptm"],
                    "iptm": conf["iptm"],
                    "ranking_score": conf["ranking_score"],
                    "fraction_disordered": conf["fraction_disordered"],
                    "has_clash": conf["has_clash"],
                }
                # Add per-chain pTM
                for i, cptm in enumerate(conf.get("chain_ptm", [])):
                    row[f"chain_{chr(65+i)}_ptm"] = cptm
                all_confidence_rows.append(row)
                protein_conf_rows.append(row)

        # Write per-protein confidence CSV
        if protein_conf_rows:
            _write_confidence_csv(pdb_dir / "confidence.csv", protein_conf_rows)

        # Generate PyMOL load script
        _write_pymol_script(pdb_dir, pdb_id, list(conditions.keys()))

    # Write global confidence summary
    if all_confidence_rows:
        _write_confidence_csv(output_dir / "confidence_summary.csv", all_confidence_rows)

    # Write aggregate comparison (best seed per condition)
    _write_aggregate_summary(output_dir, all_confidence_rows)

    print(f"\n{'=' * 60}")
    print(f"Organized {len(proteins)} proteins into: {output_dir}")
    print(f"  - confidence_summary.csv  (all seeds)")
    print(f"  - aggregate_summary.csv   (best seed per condition)")
    print(f"  - Per-protein: models/, confidences/, load_in_pymol.pml")


def _write_confidence_csv(path, rows):
    """Write confidence rows to CSV."""
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Add any extra chain columns from other rows
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_aggregate_summary(output_dir, all_rows):
    """Write a summary with the best seed (highest ranking_score) per protein/condition."""
    import csv

    # Group by (pdb_id, condition), pick best ranking_score
    best = {}
    for row in all_rows:
        key = (row["pdb_id"], row["condition"])
        score = row.get("ranking_score") or 0
        if key not in best or score > (best[key].get("ranking_score") or 0):
            best[key] = row

    rows = sorted(best.values(), key=lambda r: (r["pdb_id"], r["condition"]))

    if not rows:
        return

    path = output_dir / "aggregate_summary.csv"
    fieldnames = ["pdb_id", "condition", "seed", "sample", "ptm", "iptm", "ranking_score",
                  "fraction_disordered", "has_clash"]
    # Add chain columns
    chain_cols = sorted(set(k for r in rows for k in r if k.startswith("chain_")))
    fieldnames.extend(chain_cols)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print a nice summary table
    print(f"\n{'=' * 60}")
    print("CONFIDENCE SUMMARY (best seed per condition)")
    print(f"{'=' * 60}")
    print(f"  {'PDB':<6} {'Condition':<35} {'pTM':>5} {'ipTM':>5} {'Rank':>5}")
    print(f"  {'-' * 58}")

    current_pdb = None
    for row in rows:
        if row["pdb_id"] != current_pdb:
            if current_pdb is not None:
                print()
            current_pdb = row["pdb_id"]
        ptm = f"{row['ptm']:.2f}" if row.get("ptm") is not None else "N/A"
        iptm = f"{row['iptm']:.2f}" if row.get("iptm") is not None else "N/A"
        rank = f"{row['ranking_score']:.2f}" if row.get("ranking_score") is not None else "N/A"
        print(f"  {row['pdb_id']:<6} {row['condition']:<35} {ptm:>5} {iptm:>5} {rank:>5}")


def _write_pymol_script(pdb_dir, pdb_id, conditions):
    """Generate a PyMOL .pml script that loads crystal + all AF3 models."""
    models_dir = pdb_dir / "models"
    pml_lines = [
        f"# PyMOL script for {pdb_id} AF3 validation",
        f"# Load this in PyMOL:  @{pdb_dir / 'load_in_pymol.pml'}",
        f"#   or: File > Run Script > load_in_pymol.pml",
        "",
        "# --- Load crystal structure ---",
        f"fetch {pdb_id}, type=pdb",
        f"color gray80, {pdb_id}",
        f"show cartoon, {pdb_id}",
        "",
        "# --- Load AF3 models ---",
    ]

    # Color scheme per condition type
    colors = {
        "unconstrained": "salmon",
        "full_sequon_fixed": "palegreen",
        "unconstrained_with_glycans": "lightorange",
        "full_sequon_fixed_with_glycans": "palecyan",
        "full_sequon_fixed_full_glycans": "aquamarine",
        "unconstrained_denovo_glycans": "lightpink",
        "unconstrained_full_glycans": "paleturquoise",
        "unconstrained_custom": "wheat",
    }

    for condition in sorted(conditions):
        cif_path = models_dir / f"{pdb_id}_{condition}.cif"
        if cif_path.exists():
            obj_name = f"{pdb_id}_{condition}"
            color = colors.get(condition, "white")
            pml_lines.extend([
                f"load {cif_path}, {obj_name}",
                f"color {color}, {obj_name}",
                f"show cartoon, {obj_name}",
            ])

    # Load the chain-by-chain alignment script and run it
    align_script = PIPELINE_ROOT / "utilities" / "pymol_align_designs.py"
    log_path = pdb_dir / "rmsd_results"
    pml_lines.extend([
        "",
        "# --- Per-chain alignment with RMSD logging ---",
        f"run {align_script}",
        f"align_designs {pdb_id}, chain=each, log={log_path}",
        "",
        "# --- Display settings ---",
        "bg_color white",
        "set ray_shadow, 0",
        "set cartoon_fancy_helices, 1",
        f"center {pdb_id}",
        "zoom",
    ])

    pml_path = pdb_dir / "load_in_pymol.pml"
    with open(pml_path, "w") as f:
        f.write("\n".join(pml_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Organize AF3 results for PyMOL analysis")
    parser.add_argument("download_dir", help="Path to AF3 download folder")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: af3_results/ in pipeline dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or str(DATA_DIR / "af3_results")

    print("=" * 60)
    print("ORGANIZE AF3 RESULTS")
    print("=" * 60)
    print(f"  Source:  {args.download_dir}")
    print(f"  Output:  {output_dir}")

    organize(args.download_dir, output_dir)


if __name__ == "__main__":
    main()
