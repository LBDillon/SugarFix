#!/usr/bin/env python3
"""PyMOL visualization for the glycan clash experiment.

Loads the crystal structure and the three AF3-predicted conditions for a
protein, aligns them, and highlights glycosylation sites to illustrate
that the MPNN->AF3 pipeline is glycan-blind: the local backbone at
glycosites is essentially identical with or without the glycan present.

Usage (inside PyMOL):
    run pymol_clash_visualization.py
    visualize_clash("1J2E")          # single protein, best testable site
    visualize_clash("5IFP", site=156) # specific site by PDB resnum
    visualize_all()                   # panel for all 4 proteins

Or from the command line:
    pymol -cq pymol_clash_visualization.py -- --pdb-id 1J2E
"""

from pymol import cmd, stored
import os
import glob
import csv

# ---------------------------------------------------------------------------
# Path configuration — adjust these to your local layout
# ---------------------------------------------------------------------------
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PIPELINE_ROOT, "data")

# AF3 results — update this path to wherever AF3 outputs were downloaded
AF3_RESULTS_DIRS = [
    os.path.expanduser("~/Downloads/AF3_67_03"),
    os.path.expanduser("~/Downloads/folds_2026_03_27_19_06"),
]

# Tested proteins
PROTEINS = ["1J2E", "1DBN", "5IFP", "2Z64"]

# Condition colors
COLOR_WT = "marine"           # blue — WT + glycan
COLOR_REINTRO = "salmon"      # red/pink — MPNN reintroduced + glycan
COLOR_UNCON = "splitpea"      # olive/green — MPNN unconstrained, no glycan
COLOR_CRYSTAL = "gray70"      # crystal structure reference
COLOR_GLYCAN_WT = "cyan"      # glycan in WT condition
COLOR_GLYCAN_REINTRO = "tv_red"  # glycan in reintroduced condition
COLOR_SEQUON = "yellow"       # sequon highlight
COLOR_SPHERE = "gray90"       # local RMSD sphere indicator


def _find_af3_cif(pdb_id, condition, model_idx=0):
    """Find the AF3 CIF file for a given protein and condition."""
    pdb_lower = pdb_id.lower()
    cif_name = f"fold_{pdb_lower}_clash_{condition}_model_{model_idx}.cif"

    for base_dir in AF3_RESULTS_DIRS:
        if not os.path.isdir(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            if cif_name in files:
                return os.path.join(root, cif_name)
    return None


def _find_crystal_pdb(pdb_id):
    """Find the crystal structure PDB for a protein."""
    prep_dir = os.path.join(DATA_DIR, "prep", pdb_id, "structure")
    # Prefer the full PDB (with glycans) over protein-only
    for suffix in [f"{pdb_id}.pdb", f"{pdb_id}_bioassembly.pdb", f"{pdb_id}_protein.pdb"]:
        path = os.path.join(prep_dir, suffix)
        if os.path.exists(path):
            return path
    return None


def _load_testable_sites(pdb_id):
    """Load testable glycosite info from clash_metrics CSV."""
    csv_path = os.path.join(
        DATA_DIR, "outputs", "clash_experiment_combined",
        f"clash_metrics_{pdb_id}.csv"
    )
    if not os.path.exists(csv_path):
        # Try the combined CSV
        csv_path = os.path.join(
            DATA_DIR, "outputs", "clash_experiment_combined",
            "clash_metrics_all.csv"
        )

    sites = []
    if not os.path.exists(csv_path):
        return sites

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            if row["pdb_id"] != pdb_id:
                continue
            if row["testable"] != "True":
                continue
            if row["condition"] != "wt_with_glycan":
                continue
            key = (row["chain"], row["pdb_resnum"])
            if key in seen:
                continue
            seen.add(key)
            sites.append({
                "chain": row["chain"],
                "pdb_resnum": int(row["pdb_resnum"]),
                "position_1idx": int(row["position_1idx"]),
                "wt_triplet": row["wt_triplet"],
                "design_triplet": row["design_triplet"],
                "is_stub": row["is_stub"] == "True",
            })
    return sites


def visualize_clash(pdb_id, site=None, sphere_radius=8.0, model_idx=0):
    """Main visualization for one protein's clash experiment.

    Args:
        pdb_id: PDB identifier (e.g. "1J2E")
        site: PDB resnum of the glycosite to focus on. If None, picks
              the first testable stub site.
        sphere_radius: Radius (Å) for the local environment sphere.
        model_idx: Which AF3 model to use (0-4).
    """
    pdb_id = pdb_id.upper()
    cmd.reinitialize()

    # --- Load structures ---
    crystal_path = _find_crystal_pdb(pdb_id)
    if crystal_path:
        cmd.load(crystal_path, f"{pdb_id}_crystal")
        print(f"Loaded crystal: {crystal_path}")
    else:
        print(f"WARNING: Crystal structure not found for {pdb_id}")

    conditions = ["wt_with_glycan", "mpnn_reintroduced", "mpnn_unconstrained"]
    cond_names = {
        "wt_with_glycan": f"{pdb_id}_wt",
        "mpnn_reintroduced": f"{pdb_id}_reintro",
        "mpnn_unconstrained": f"{pdb_id}_uncon",
    }

    for cond in conditions:
        cif_path = _find_af3_cif(pdb_id, cond, model_idx)
        if cif_path:
            cmd.load(cif_path, cond_names[cond])
            print(f"Loaded {cond}: {cif_path}")
        else:
            print(f"WARNING: AF3 CIF not found for {pdb_id} {cond}")

    # --- Pick site ---
    sites = _load_testable_sites(pdb_id)
    if not sites:
        print(f"No testable sites found for {pdb_id}. Using manual site if provided.")
        if site is None:
            print("Specify site= parameter with a PDB resnum.")
            return

    if site is not None:
        target_site = next((s for s in sites if s["pdb_resnum"] == site), None)
        if target_site is None:
            # Allow manual override even if not in CSV
            target_site = {"chain": "A", "pdb_resnum": site,
                           "wt_triplet": "???", "design_triplet": "???"}
    else:
        # Pick first stub site, or first site
        stub_sites = [s for s in sites if s["is_stub"]]
        target_site = stub_sites[0] if stub_sites else sites[0]

    chain = target_site["chain"]
    resnum = target_site["pdb_resnum"]

    print(f"\nFocusing on site {chain}:{resnum}")
    print(f"  WT triplet: {target_site['wt_triplet']}")
    print(f"  Design triplet: {target_site['design_triplet']}")

    # --- Align all AF3 models to crystal structure ---
    ref = f"{pdb_id}_crystal"
    if crystal_path:
        for cond in conditions:
            name = cond_names[cond]
            try:
                cmd.align(f"{name} and polymer.protein", f"{ref} and polymer.protein")
            except Exception:
                print(f"  Alignment failed for {name}, trying super...")
                try:
                    cmd.super(f"{name} and polymer.protein", f"{ref} and polymer.protein")
                except Exception:
                    print(f"  super also failed for {name}")
    else:
        # Align to WT AF3 model if no crystal
        ref = cond_names["wt_with_glycan"]
        for cond in ["mpnn_reintroduced", "mpnn_unconstrained"]:
            name = cond_names[cond]
            try:
                cmd.align(f"{name} and polymer.protein", f"{ref} and polymer.protein")
            except Exception:
                pass

    # --- Coloring ---
    # Start with everything as cartoon, muted
    cmd.hide("everything")

    if crystal_path:
        cmd.show("cartoon", f"{pdb_id}_crystal")
        cmd.color(COLOR_CRYSTAL, f"{pdb_id}_crystal")
        cmd.set("cartoon_transparency", 0.7, f"{pdb_id}_crystal")

    cmd.show("cartoon", cond_names["wt_with_glycan"])
    cmd.color(COLOR_WT, cond_names["wt_with_glycan"])

    cmd.show("cartoon", cond_names["mpnn_reintroduced"])
    cmd.color(COLOR_REINTRO, cond_names["mpnn_reintroduced"])

    cmd.show("cartoon", cond_names["mpnn_unconstrained"])
    cmd.color(COLOR_UNCON, cond_names["mpnn_unconstrained"])

    # --- Local environment: show sticks around the glycosite ---
    local_sel = f"(byres all within {sphere_radius} of (chain {chain} and resi {resnum} and name CA))"

    for cond in conditions:
        name = cond_names[cond]
        cmd.show("sticks", f"{name} and polymer.protein and {local_sel}")

    # Highlight the sequon residues (Asn + 2 flanking)
    sequon_resis = f"{resnum}-{resnum+2}"
    for cond in conditions:
        name = cond_names[cond]
        sel_name = f"sequon_{cond.split('_')[0]}"
        cmd.select(sel_name, f"{name} and chain {chain} and resi {sequon_resis}")
        cmd.show("sticks", sel_name)
        cmd.color(COLOR_SEQUON, f"{sel_name} and elem C")
        cmd.set("stick_radius", 0.2, sel_name)

    # --- Show glycans (NAG/carbohydrates) ---
    for cond in ["wt_with_glycan", "mpnn_reintroduced"]:
        name = cond_names[cond]
        glycan_sel = f"{name} and (not polymer.protein)"
        # Show glycan as sticks
        cmd.show("sticks", glycan_sel)
        color = COLOR_GLYCAN_WT if cond == "wt_with_glycan" else COLOR_GLYCAN_REINTRO
        cmd.color(color, f"{glycan_sel} and elem C")

    # --- Show crystal glycans if present ---
    if crystal_path:
        crystal_glycan = f"{pdb_id}_crystal and resn NAG+BMA+MAN+FUC+GAL+SIA+NDG"
        cmd.show("sticks", crystal_glycan)
        cmd.color("white", f"({crystal_glycan}) and elem C")

    # --- Sphere indicator for local RMSD region ---
    # Create a pseudoatom at the glycosite CA to show the measurement sphere
    cmd.pseudoatom("sphere_center", pos=None,
                   selection=f"{cond_names['wt_with_glycan']} and chain {chain} and resi {resnum} and name CA")
    cmd.show("sphere", "sphere_center")
    cmd.set("sphere_scale", sphere_radius, "sphere_center")
    cmd.set("sphere_transparency", 0.85, "sphere_center")
    cmd.color(COLOR_SPHERE, "sphere_center")

    # --- Camera ---
    cmd.center(f"{cond_names['wt_with_glycan']} and chain {chain} and resi {resnum}")
    cmd.zoom(f"(byres all within {sphere_radius + 4} of ({cond_names['wt_with_glycan']} and chain {chain} and resi {resnum} and name CA))")

    # --- Display settings ---
    cmd.set("cartoon_transparency", 0.5, cond_names["wt_with_glycan"])
    cmd.set("cartoon_transparency", 0.5, cond_names["mpnn_reintroduced"])
    cmd.set("cartoon_transparency", 0.5, cond_names["mpnn_unconstrained"])
    cmd.set("stick_radius", 0.15)
    cmd.set("ray_shadow", 0)
    cmd.bg_color("white")
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("spec_reflect", 0.3)

    # --- Labels ---
    # Label the Asn residue in each condition
    cmd.label(f"{cond_names['wt_with_glycan']} and chain {chain} and resi {resnum} and name CA",
              "'WT+glycan'")
    cmd.set("label_color", COLOR_WT, cond_names["wt_with_glycan"])

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Clash Experiment Visualization: {pdb_id} site {chain}:{resnum:<4}         ║
╠══════════════════════════════════════════════════════════════╣
║  Colors:                                                     ║
║    Blue   (marine)   = WT + glycan                           ║
║    Red    (salmon)   = MPNN reintroduced + glycan            ║
║    Green  (splitpea) = MPNN unconstrained, no glycan         ║
║    Gray              = Crystal structure                      ║
║    Yellow            = Sequon residues (N-X-S/T)             ║
║    Cyan / Red sticks = Glycan (NAG) in WT / reintroduced     ║
║  Transparent sphere  = 8 Å local RMSD measurement region     ║
║                                                              ║
║  Key finding: red & green backbones overlap (no glycan clash) ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Create useful selections for the user
    cmd.select("glycosite", f"chain {chain} and resi {resnum}")
    cmd.select("local_env", local_sel)
    cmd.deselect()


def side_by_side(pdb_id, site=None, model_idx=0):
    """Create a three-panel view: WT | Reintroduced | Unconstrained.

    Duplicates each condition into separate objects shown in three
    adjacent positions for a cleaner comparison.
    """
    pdb_id = pdb_id.upper()
    cmd.reinitialize()

    sites = _load_testable_sites(pdb_id)
    if site is None and sites:
        stub_sites = [s for s in sites if s["is_stub"]]
        target = stub_sites[0] if stub_sites else sites[0]
    elif site is not None:
        target = next((s for s in sites if s["pdb_resnum"] == site),
                      {"chain": "A", "pdb_resnum": site})
    else:
        print("No sites found. Specify site= parameter.")
        return

    chain = target["chain"]
    resnum = target["pdb_resnum"]

    conditions = {
        "wt_with_glycan": (COLOR_WT, COLOR_GLYCAN_WT, "WT + Glycan"),
        "mpnn_reintroduced": (COLOR_REINTRO, COLOR_GLYCAN_REINTRO, "MPNN + Glycan Grafted"),
        "mpnn_unconstrained": (COLOR_UNCON, None, "MPNN (no glycan)"),
    }

    # Load crystal as alignment reference
    crystal_path = _find_crystal_pdb(pdb_id)
    if crystal_path:
        cmd.load(crystal_path, "crystal_ref")

    offset = 0
    spacing = 40  # Å between panels

    for cond, (color, glycan_color, label) in conditions.items():
        cif_path = _find_af3_cif(pdb_id, cond, model_idx)
        if not cif_path:
            print(f"Missing: {cond}")
            continue

        obj_name = f"panel_{cond}"
        cmd.load(cif_path, obj_name)

        # Align to crystal
        if crystal_path:
            try:
                cmd.align(f"{obj_name} and polymer.protein",
                          "crystal_ref and polymer.protein")
            except Exception:
                pass

        # Translate for side-by-side layout
        cmd.translate([offset, 0, 0], obj_name)

        # Style
        cmd.hide("everything", obj_name)
        cmd.show("cartoon", f"{obj_name} and polymer.protein")
        cmd.color(color, f"{obj_name} and polymer.protein")
        cmd.set("cartoon_transparency", 0.6, obj_name)

        # Show local sticks
        local = f"(byres {obj_name} within 8 of ({obj_name} and chain {chain} and resi {resnum} and name CA))"
        cmd.show("sticks", f"{obj_name} and polymer.protein and {local}")
        cmd.set("cartoon_transparency", 0.0,
                f"{obj_name} and polymer.protein and {local}")

        # Sequon in yellow
        cmd.color(COLOR_SEQUON,
                  f"{obj_name} and chain {chain} and resi {resnum}-{resnum+2} and elem C")

        # Glycans
        if glycan_color:
            glycan_sel = f"{obj_name} and not polymer.protein"
            cmd.show("sticks", glycan_sel)
            cmd.color(glycan_color, f"({glycan_sel}) and elem C")

        # Panel label
        cmd.pseudoatom(f"label_{cond}",
                       pos=[offset, -15, 0],
                       label=label)
        cmd.set("label_size", 20, f"label_{cond}")
        cmd.show("label", f"label_{cond}")

        offset += spacing

    # Clean up reference
    if crystal_path:
        cmd.delete("crystal_ref")

    cmd.zoom("all")
    cmd.turn("x", -10)
    cmd.bg_color("white")
    cmd.set("ray_shadow", 0)
    cmd.set("antialias", 2)

    print(f"Side-by-side view for {pdb_id} site {chain}:{resnum}")
    print("Use `ray 2400, 1200` then `png clash_sidebyside.png` to save.")


def visualize_all():
    """Load the best testable site for each of the 4 proteins as scenes.

    After running, use `cmd.scene('1J2E')` to switch between proteins,
    or use the scene buttons in the PyMOL GUI.
    """
    cmd.reinitialize()

    # Best site per protein (first testable stub from CSV)
    best_sites = {}
    for pdb_id in PROTEINS:
        sites = _load_testable_sites(pdb_id)
        stub_sites = [s for s in sites if s["is_stub"]]
        if stub_sites:
            best_sites[pdb_id] = stub_sites[0]["pdb_resnum"]
        elif sites:
            best_sites[pdb_id] = sites[0]["pdb_resnum"]

    for pdb_id, resnum in best_sites.items():
        visualize_clash(pdb_id, site=resnum)
        cmd.scene(pdb_id, action="store")
        print(f"Stored scene: {pdb_id} (site {resnum})")

    if best_sites:
        first = list(best_sites.keys())[0]
        cmd.scene(first, action="recall")
    print(f"\nUse cmd.scene('<PDB_ID>') to switch between proteins.")


def save_figure(pdb_id, site=None, width=2400, height=1800, dpi=300):
    """Render and save a publication-quality figure."""
    output_dir = os.path.join(DATA_DIR, "outputs", "clash_experiment_combined")
    os.makedirs(output_dir, exist_ok=True)

    if site:
        filename = f"pymol_clash_{pdb_id}_{site}.png"
    else:
        filename = f"pymol_clash_{pdb_id}.png"

    out_path = os.path.join(output_dir, filename)
    cmd.ray(width, height)
    cmd.png(out_path, dpi=dpi)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point (pymol -cq script.py -- --pdb-id 1J2E)
# ---------------------------------------------------------------------------
if __name__ == "pymol" or __name__ == "__main__":
    import sys
    args = [a for a in sys.argv if not a.startswith("-") and a != __file__]

    # Check for --pdb-id in pymol CLI args
    if "--pdb-id" in sys.argv:
        idx = sys.argv.index("--pdb-id")
        if idx + 1 < len(sys.argv):
            pdb_id = sys.argv[idx + 1].upper()
            site_arg = None
            if "--site" in sys.argv:
                si = sys.argv.index("--site")
                if si + 1 < len(sys.argv):
                    site_arg = int(sys.argv[si + 1])
            visualize_clash(pdb_id, site=site_arg)
