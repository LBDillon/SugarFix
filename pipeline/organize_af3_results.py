"""Organize AF3 Server output folders for SugarFix designs.

Adapted from the case-study pipeline's `organize_af3_results.py`.

Recognises SugarFix's AF3 folder naming convention:
    {pdb}_designer_selected[_glycans]
    {pdb}_soft_filter[_glycans]

Usage:
    python -m pipeline.organize_af3_results /path/to/folds_download \
        --pdb-id 2DH2 --crystal-pdb data/structures/2DH2.pdb \
        --output-dir data/outputs/2DH2/af3_organized

Output structure:
    af3_organized/
        2DH2/
            models/                 # top-ranked .cif per condition
            confidences/            # all seed JSONs
            confidence.csv
            load_in_pymol.pml       # opens crystal + all models, runs RMSD
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIGN_SCRIPT_SRC = REPO_ROOT / "utilities" / "pymol_align_designs.py"

CONDITION_COLORS = {
    "designer_selected": "palegreen",
    "designer_selected_with_glycans": "aquamarine",
    "soft_filter": "salmon",
    "soft_filter_with_glycans": "lightorange",
}


def parse_folder_name(folder_name: str, pdb_id: str):
    """Return condition label or None if folder is not recognised."""
    name = folder_name.lower()
    pid = pdb_id.lower()
    if not name.startswith(pid + "_"):
        return None
    suffix = name[len(pid) + 1 :]
    # strip optional trailing _AF3 / timestamp
    suffix = re.sub(r"(_af3)?(_\d{8}_\d{6})?$", "", suffix)
    has_glycans = suffix.endswith("_glycans") or suffix.endswith("_with_glycans")
    base = re.sub(r"(_with)?_glycans$", "", suffix)
    if base not in ("designer_selected", "soft_filter"):
        return None
    return f"{base}_with_glycans" if has_glycans else base


def find_best_model(folder: Path):
    for i in range(5):
        cands = list(folder.glob(f"*_model_{i}.cif"))
        if cands:
            return cands[0], i
    cands = sorted(folder.glob("*_model.cif"))
    return (cands[0], "top") if cands else (None, None)


def extract_confidences(folder: Path):
    rows = []
    for i in range(5):
        cands = list(folder.glob(f"*_summary_confidences_{i}.json"))
        if not cands:
            continue
        data = json.load(open(cands[0]))
        rows.append({
            "seed": i,
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
        })
    return rows


def write_pymol_script(pdb_dir: Path, pdb_id: str, crystal_pdb: Path,
                       conditions: list[str]) -> Path:
    """Write a self-contained PyMOL script using paths relative to pdb_dir.

    Copies the crystal PDB and the align script next to the .pml so the whole
    folder is portable (e.g. download from Colab and run on a local machine).
    """
    # Copy crystal PDB next to the .pml so paths stay portable.
    local_crystal = pdb_dir / f"{pdb_id}_crystal.pdb"
    shutil.copy2(crystal_pdb, local_crystal)

    # Copy the align script next to the .pml.
    local_align = pdb_dir / "pymol_align_designs.py"
    if ALIGN_SCRIPT_SRC.exists():
        shutil.copy2(ALIGN_SCRIPT_SRC, local_align)

    lines = [
        f"# PyMOL script for {pdb_id} AF3 validation (SugarFix)",
        f"# Run from this directory:  pymol load_in_pymol.pml",
        "",
        "# Make sure relative paths resolve next to this script.",
        "import os",
        "from pymol import cmd",
        "cmd.cd(os.path.dirname(__script__))",
        "",
        "# --- Reference crystal structure ---",
        f"load {local_crystal.name}, {pdb_id}",
        f"color gray80, {pdb_id}",
        f"show cartoon, {pdb_id}",
        "",
        "# --- AF3 models ---",
    ]
    for condition in sorted(conditions):
        cif = pdb_dir / "models" / f"{pdb_id}_{condition}.cif"
        if not cif.exists():
            continue
        obj = f"{pdb_id}_{condition}"
        color = CONDITION_COLORS.get(condition, "white")
        rel = f"models/{cif.name}"
        lines += [
            f"load {rel}, {obj}",
            f"color {color}, {obj}",
            f"show cartoon, {obj}",
        ]
    lines += [
        "",
        "# --- Per-chain alignment with RMSD logging ---",
        f"run {local_align.name}",
        f"align_designs {pdb_id}, chain=each, log=rmsd_results",
        "",
        "bg_color white",
        "set cartoon_fancy_helices, 1",
        f"center {pdb_id}",
        "zoom",
    ]
    pml = pdb_dir / "load_in_pymol.pml"
    pml.write_text("\n".join(lines) + "\n")
    return pml


def organize(download_dir: Path, output_dir: Path, pdb_id: str, crystal_pdb: Path):
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)
    crystal_pdb = Path(crystal_pdb).resolve()
    pdb_id = pdb_id.upper()

    if not download_dir.exists():
        sys.exit(f"ERROR: download dir not found: {download_dir}")
    if not crystal_pdb.exists():
        sys.exit(f"ERROR: crystal PDB not found: {crystal_pdb}")

    pdb_dir = output_dir / pdb_id
    models_dir = pdb_dir / "models"
    conf_dir = pdb_dir / "confidences"
    models_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    found = {}
    skipped = []
    for entry in sorted(download_dir.iterdir()):
        if not entry.is_dir():
            continue
        condition = parse_folder_name(entry.name, pdb_id)
        if condition is None:
            skipped.append(entry.name)
            continue
        found[condition] = entry

    if not found:
        sys.exit(
            f"ERROR: no SugarFix AF3 folders matched {pdb_id} in {download_dir}\n"
            f"  Skipped: {skipped}"
        )

    print(f"Found {len(found)} condition(s) for {pdb_id}: {sorted(found)}")
    if skipped:
        print(f"  Skipped {len(skipped)} unrecognized: {skipped}")

    all_rows = []
    for condition, src in sorted(found.items()):
        best_cif, idx = find_best_model(src)
        if best_cif:
            dest = models_dir / f"{pdb_id}_{condition}.cif"
            shutil.copy2(best_cif, dest)
            print(f"  {condition}: copied {best_cif.name} (model {idx})")
        else:
            print(f"  {condition}: WARNING - no model .cif found")

        for cf in sorted(src.glob("*_summary_confidences_*.json")):
            m = re.search(r"_(\d+)\.json$", cf.name)
            seed = m.group(1) if m else "x"
            shutil.copy2(cf, conf_dir / f"{pdb_id}_{condition}_seed{seed}.json")

        for row in extract_confidences(src):
            row = {"pdb_id": pdb_id, "condition": condition, **row}
            all_rows.append(row)

    if all_rows:
        cols = list(all_rows[0].keys())
        with open(pdb_dir / "confidence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)

    pml = write_pymol_script(pdb_dir, pdb_id, crystal_pdb, list(found.keys()))
    print(f"\nDone. Open in PyMOL:\n  pymol {pml}")
    print(f"  RMSD output: {pdb_dir / 'rmsd_results.txt'}  + .csv")
    return pdb_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("download_dir", help="Path to AF3 download folder")
    p.add_argument("--pdb-id", required=True)
    p.add_argument("--crystal-pdb", required=True,
                   help="Path to reference crystal .pdb (e.g. data/structures/2DH2.pdb)")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    organize(args.download_dir, args.output_dir, args.pdb_id, args.crystal_pdb)


if __name__ == "__main__":
    main()
