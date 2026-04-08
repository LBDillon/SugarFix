# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.12.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SugarFix Designer Walkthrough
#
# An interactive decision console for glycosite-aware protein design.
#
# **Workflow:**
# 1. Configure your target protein and design parameters
# 2. Prepare the structure and detect glycosylation sites
# 3. **Choose how to protect each site** (the key decision)
# 4. Run ProteinMPNN with your constraints
# 5. Score designs and compare against an unconstrained baseline
# 6. Export AF3 JSONs and save outputs
# 7. *(Optional)* Analyze AF3 validation results after running AF3 externally
#
#

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    def display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)

# ---------- Detect environment, install deps if on Colab ----------
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_ROOT = Path("/content/SugarFix")
    if not REPO_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/LBDillon/SugarFix.git", str(REPO_ROOT)],
            check=True,
        )
    os.chdir(REPO_ROOT)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", str(REPO_ROOT / "requirements.txt")],
        check=True,
    )
    # mkdssp powers the secondary-structure / SASA overlay in Step 4.
    subprocess.run(["apt-get", "install", "-y", "-q", "dssp"], check=False)
    try:
        from google.colab import output as colab_output
        colab_output.enable_custom_widget_manager()
    except Exception:
        pass
else:
    REPO_ROOT = Path.cwd()

# ---------- ProteinMPNN: find or auto-clone ----------
PMPNN_DIR = None
_candidates = [
    os.environ.get("PROTEINMPNN_DIR", ""),
    str(REPO_ROOT / "ProteinMPNN"),
    str(REPO_ROOT.parent / "ProteinMPNN"),
]
for _c in _candidates:
    if _c and Path(_c).is_dir() and (Path(_c) / "protein_mpnn_utils.py").exists():
        PMPNN_DIR = Path(_c).resolve()
        break

if PMPNN_DIR is None:
    print("ProteinMPNN not found \u2014 cloning from GitHub...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/dauparas/ProteinMPNN.git",
         str(REPO_ROOT / "ProteinMPNN")],
        check=True,
    )
    PMPNN_DIR = (REPO_ROOT / "ProteinMPNN").resolve()

os.environ["PROTEINMPNN_DIR"] = str(PMPNN_DIR)
PROTEINMPNN_RUN = PMPNN_DIR / "protein_mpnn_run.py"

# ---------- Pipeline imports ----------
# Add the pipeline package to the path so its modules can import each other
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PMPNN_DIR) not in sys.path:
    sys.path.insert(0, str(PMPNN_DIR))

from pipeline.mpnn_utils import (
    find_sequons as _find_sequons_raw,
    get_mpnn_chain_seqs_and_order,
    is_functional_sequon,
    read_fasta_sequences,
    split_mpnn_concat_seq,
    verify_sequon_positions,
    write_fixed_positions_jsonl,
)
from pipeline.identify_sequons import (
    annotate_evidence_tiers,
    build_pdb_resnum_to_mpnn_idx,
    extract_uniprot_accessions,
    fetch_uniprot_glycosylation,
    load_uniprot_evidence,
    parse_pdb_dbref,
)
from pipeline.extract_pdb_glycans import extract_glycan_trees
from pipeline.prepare_structure import (
    create_protein_only_pdb,
    download_pdb,
    parse_missing_residues,
    parse_structure,
)
from sugarfix_helpers import (
    CONDITION_LABELS,
    DesignSession,
    GlycoSite,
    POLICY_DEFAULTS,
    WIDGETS_AVAILABLE,
    build_condition_manifest,
    build_constraint_bundle,
    build_decision_df,
    build_glyco_site,
    build_site_policy_map,
    snap_to_nearest_sequon,
    clear_output,
    display_df,
    infer_expected_tier,
    make_af3_server_json,
    pdb_resnum_to_uniprot,
    plot_design_dashboard,
    plot_site_strategy_overview,
    policy_requirement_met,
    score_designs,
    select_top_design,
    summarize_glycoprotein_status,
    widgets,
)

print(f"Repository root: {REPO_ROOT}")
print(f"ProteinMPNN:     {PMPNN_DIR}")
print(f"Widgets:         {'available' if WIDGETS_AVAILABLE else 'not available'}")
if IN_COLAB:
    print("Environment:     Google Colab")

# %% [markdown]
# ## Configuration
#
# Adjust the controls below (or edit the defaults in the code cell) and then run the next cells.
# The `session` object carries all state forward — no need to track separate variables.

# %%
session = DesignSession(
    pdb_id="2DH2",
    run_label="",
    num_seqs=64,
    sampling_temp=0.1,
    seed=42,
    pipeline_root=REPO_ROOT,
)
session.setup_paths()

# Optional: AlphaFold DB input. If non-empty, the notebook fetches
# AF-{accession}-F1-model_v4.pdb from https://alphafold.ebi.ac.uk and uses
# that as the design target. The accession also seeds the UniProt evidence
# lookup directly (AF models have no DBREF block).
ALPHAFOLD_UNIPROT = ""

# ---------- Optional: Google Drive output ----------
SAVE_TO_DRIVE = False
if IN_COLAB:
    try:
        _drive_w = widgets.Checkbox(value=False, description="Save outputs to Google Drive", indent=False)
    except Exception:
        _drive_w = None

CANDIDATES_CSV = session.data_dir / "candidates.csv"
if not CANDIDATES_CSV.exists():
    CANDIDATES_CSV = None

if WIDGETS_AVAILABLE:
    _pdb_w = widgets.Text(value=session.pdb_id, description="PDB ID:",
                          layout=widgets.Layout(width="280px"))
    _af_w = widgets.Text(value="", description="AF UniProt:",
                         placeholder="optional, e.g. P08195",
                         layout=widgets.Layout(width="320px"))
    _label_w = widgets.Text(value=session.run_label, description="Run label:",
                            placeholder="optional subfolder",
                            layout=widgets.Layout(width="380px"))
    _nseqs_w = widgets.IntSlider(value=session.num_seqs, min=1, max=64, step=1,
                                  description="Num seqs:")
    _temp_w = widgets.FloatSlider(value=session.sampling_temp, min=0.01, max=1.0,
                                   step=0.01, description="Temp:", readout_format=".2f")
    _seed_w = widgets.IntText(value=session.seed, description="Seed:",
                               layout=widgets.Layout(width="220px"))
    _config_status = widgets.HTML()

    def _sync_config(change=None):
        global ALPHAFOLD_UNIPROT
        session.pdb_id = _pdb_w.value.strip().upper()
        _af_val = _af_w.value.strip().upper()
        if _af_val.startswith("AF-"):
            _af_val = _af_val[3:]
        if _af_val.endswith("-F1"):
            _af_val = _af_val[:-3]
        ALPHAFOLD_UNIPROT = _af_val
        session.run_label = _label_w.value.strip()
        session.num_seqs = _nseqs_w.value
        session.sampling_temp = _temp_w.value
        session.seed = _seed_w.value
        session.setup_paths()
        _config_status.value = (
            f"<div style='padding:4px 0; color:#555;'>"
            f"Output dir: <code>{session.run_dir}</code></div>"
        )

    for w in [_pdb_w, _af_w, _label_w, _nseqs_w, _temp_w, _seed_w]:
        w.observe(_sync_config, names="value")

    _box_children = [
        widgets.HTML("<h4 style='margin:0;'>Design parameters</h4>"),
        widgets.HBox([_pdb_w, _af_w, _label_w]),
        widgets.HBox([_nseqs_w, _temp_w, _seed_w]),
        _config_status,
    ]
    if IN_COLAB and _drive_w is not None:
        _box_children.insert(1, _drive_w)
    display(widgets.VBox(_box_children))
    _sync_config()
else:
    print(f"Protein:     {session.pdb_id}")
    print(f"Run label:   {session.run_label or '(none)'}")
    print(f"Num seqs:    {session.num_seqs}")
    print(f"Temp:        {session.sampling_temp}")
    print(f"Seed:        {session.seed}")
    print(f"Output root: {session.output_root}")

# %% [markdown]
# ---
# ## Step 1 — Prepare Structure
#

# %%
# Mount Google Drive if user opted in
if IN_COLAB:
    try:
        SAVE_TO_DRIVE = _drive_w.value
    except Exception:
        pass
    if SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        # Redirect ALL outputs to Drive directly so nothing lands in the
        # cloned repo folder during the run.
        session.data_dir = Path("/content/drive/MyDrive/SugarFix_outputs")
        session.structure_dir = session.data_dir / "prep" / session.pdb_id / "structure"
        session.output_root = session.data_dir / "outputs" / session.pdb_id
        session.run_dir = (
            session.output_root / session.run_label
            if session.run_label else session.output_root
        )
        session.figure_dir = session.run_dir / "figures"
        session.af3_dir = session.run_dir / "af3"
        for _d in (session.structure_dir, session.run_dir,
                   session.figure_dir, session.af3_dir):
            _d.mkdir(parents=True, exist_ok=True)
        _drive_out = session.run_dir
        print(f"Google Drive output (live): {_drive_out}")

session.structure_dir.mkdir(parents=True, exist_ok=True)

pdb_path = session.structure_dir / f"{session.pdb_id}.pdb"
protein_pdb_path = session.structure_dir / f"{session.pdb_id}_protein.pdb"
annotation_structure_path = pdb_path

USING_ALPHAFOLD = bool(ALPHAFOLD_UNIPROT)
ALPHAFOLD_FORCED_ACCESSIONS = {}

if USING_ALPHAFOLD:
    import urllib.request, json as _json
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{ALPHAFOLD_UNIPROT}"
    print(f"Querying AlphaFold API: {api_url}")
    try:
        with urllib.request.urlopen(api_url) as r:
            meta = _json.loads(r.read())
        af_url = meta[0]["pdbUrl"]
        print(f"Fetching AlphaFold model: {af_url}")
        urllib.request.urlretrieve(af_url, str(pdb_path))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not fetch AlphaFold model for {ALPHAFOLD_UNIPROT}: {exc}"
        )
    print(f"Saved AF model to: {pdb_path.name}")
    # AF models have no DBREF block. Inject the accession so the UniProt
    # cross-check still runs (Step 2 will pick this up as
    # `chain_accessions`).
    ALPHAFOLD_FORCED_ACCESSIONS = {"A": ALPHAFOLD_UNIPROT}
elif not pdb_path.exists():
    print(f"Downloading {session.pdb_id} from RCSB...")
    download_ok = download_pdb(session.pdb_id, pdb_path)
    if not download_ok or not pdb_path.exists():
        raise FileNotFoundError(
            f"Could not download {session.pdb_id} from RCSB."
        )
else:
    print(f"Already present: {pdb_path.name}")

mmcif_path = pdb_path.with_suffix('.cif')
if mmcif_path.exists():
    annotation_structure_path = mmcif_path
    print(f"Using {mmcif_path.name} as the metadata/glycan source.")

chains_info, n_models = parse_structure(pdb_path)
print()
print(f"Structure summary: {len(chains_info)} chain(s), {n_models} model(s)")
for info in chains_info:
    print(
        f"  Chain {info['chain_id']}: {info['length']} residues "
        f"(PDB {info['first_residue']}-{info['last_residue']})"
    )

missing = parse_missing_residues(pdb_path)
if missing:
    total_missing = sum(len(v) for v in missing.values())
    print(f"\nWarning: {total_missing} missing residues reported in REMARK 465")

create_protein_only_pdb(pdb_path, protein_pdb_path, chains_info)
print(f"Refreshed protein-only PDB: {protein_pdb_path.name}")

glycan_trees = extract_glycan_trees(annotation_structure_path)
if glycan_trees:
    glycan_trees_path = session.structure_dir / "glycan_trees.json"
    with open(glycan_trees_path, "w") as handle:
        json.dump(glycan_trees, handle, indent=2)
    print(f"\nResolved glycan trees: {len(glycan_trees)} site(s)")
    for key, payload in sorted(glycan_trees.items()):
        print(f"  {key}: {payload['residues_string']} ({payload['n_sugars']} sugars)")
else:
    print(f"\nNo glycan trees found in {annotation_structure_path.name}")

# %% [markdown]
# ## Step 2 — Detect Sequons And Audit Evidence Tiers
#
# Scans for N-X-S/T motifs, cross-references UniProt annotations and PDB glycan structures,
# assigns evidence tiers, and flags any mismatches.
#

# %%
chain_seqs, chain_order = get_mpnn_chain_seqs_and_order(protein_pdb_path)
session.chain_seqs = chain_seqs
session.chain_order = chain_order

print(f"MPNN chain order: {chain_order}")
for chain_id in chain_order:
    print(f"  Chain {chain_id}: {len(chain_seqs[chain_id])} residues")

# ---------- 1. Detect motif sequons in the chain sequences ----------
sequons_by_chain = {}
for chain_id in chain_order:
    raw_sites = _find_sequons_raw(chain_seqs[chain_id])
    sequons_by_chain[chain_id] = [
        {"position_0idx": record["position_0idx"], "sequon": record["sequon"]}
        for record in raw_sites
    ]

verify_sequon_positions(chain_seqs, sequons_by_chain, session.pdb_id)
print("\nSequon position verification: PASSED")

# ---------- 2. Load evidence: CSV (if available) + UniProt REST API ----------
uniprot_evidence = load_uniprot_evidence(CANDIDATES_CSV, session.pdb_id)
dbref_ranges = parse_pdb_dbref(annotation_structure_path)

# Auto-fetch UniProt glycosylation features for each chain's accession
chain_accessions = extract_uniprot_accessions(annotation_structure_path)
if not chain_accessions and ALPHAFOLD_FORCED_ACCESSIONS:
    chain_accessions = dict(ALPHAFOLD_FORCED_ACCESSIONS)
    print(f"  Using forced AlphaFold accession(s): {chain_accessions}")
fetched_evidence_by_chain = {}
if chain_accessions:
    print(f"\nFetching UniProt glycosylation features:")
    for chain_id, acc in chain_accessions.items():
        print(f"  Chain {chain_id} -> UniProt {acc}")
        fetched = fetch_uniprot_glycosylation(acc)
        if fetched:
            fetched_evidence_by_chain[chain_id] = fetched
            print(f"    Found {len(fetched)} N-linked site(s)")
            # Merge into the global evidence dict (highest tier wins)
            rank = {"experimental": 3, "pdb_evidence": 2, "curator_inferred": 1}
            for pos, tier in fetched.items():
                if rank.get(tier, 0) > rank.get(uniprot_evidence.get(pos), 0):
                    uniprot_evidence[pos] = tier
else:
    print("\nNo UniProt accessions found in DBREF records.")

tier_summary = annotate_evidence_tiers(
    sequons_by_chain, chain_seqs, annotation_structure_path,
    uniprot_evidence, glycan_trees,
)

# ---------- 3. Build GlycoSite objects from motif hits ----------
pdb_to_mpnn_by_chain = {
    chain_id: build_pdb_resnum_to_mpnn_idx(protein_pdb_path, chain_id)
    for chain_id in chain_order
}

glyco_sites = []
covered_keys = set()  # (chain, mpnn_idx)
for chain_id in chain_order:
    for record in sequons_by_chain[chain_id]:
        site = build_glyco_site(
            chain_id, record["position_0idx"], chain_seqs,
            pdb_to_mpnn_by_chain, dbref_ranges, uniprot_evidence,
            glycan_trees, source="motif",
        )
        if site is not None:
            glyco_sites.append(site)
            covered_keys.add((chain_id, record["position_0idx"]))

# ---------- 4. Add UniProt-only sites the regex missed ----------
uniprot_only_added = []
for chain_id, fetched in fetched_evidence_by_chain.items():
    pdb_to_mpnn = pdb_to_mpnn_by_chain.get(chain_id, {})
    chain_dbref = dbref_ranges.get(chain_id, [])
    for unp_pos, tier in fetched.items():
        # Map UniProt pos -> PDB resnum -> MPNN idx
        pdb_resnum = None
        for rng in chain_dbref:
            if rng["unp_start"] <= unp_pos <= rng["unp_end"]:
                pdb_resnum = rng["pdb_start"] + (unp_pos - rng["unp_start"])
                break
        if pdb_resnum is None:
            continue
        mpnn_idx = pdb_to_mpnn.get(pdb_resnum)
        if mpnn_idx is None:
            continue  # residue not resolved in structure
        # Verify the mapped position is actually N-X-S/T; if not (UniProt<->PDB
        # numbering drift), snap to the nearest N-X-S/T motif within +/-50 res.
        triplet = chain_seqs[chain_id][mpnn_idx:mpnn_idx + 3]
        if not (len(triplet) == 3 and triplet[0] == "N" and triplet[1] != "P" and triplet[2] in "ST"):
            snapped = snap_to_nearest_sequon(chain_seqs[chain_id], mpnn_idx, window=50)
            if snapped is None:
                print(f"  WARNING: UniProt {chain_id}/{unp_pos} -> PDB {pdb_resnum} -> "
                      f"MPNN idx {mpnn_idx} ({triplet!r}) is not N-X-S/T and no nearby motif; skipping")
                continue
            print(f"  Note: UniProt {chain_id}/{unp_pos} -> MPNN idx {mpnn_idx} ({triplet!r}) "
                  f"is not N-X-S/T; snapped to nearest motif at MPNN idx {snapped} "
                  f"({chain_seqs[chain_id][snapped:snapped+3]!r})")
            mpnn_idx = snapped
        if (chain_id, mpnn_idx) in covered_keys:
            continue  # already represented (now covered by snap)
        site = build_glyco_site(
            chain_id, mpnn_idx, chain_seqs,
            pdb_to_mpnn_by_chain, dbref_ranges, uniprot_evidence,
            glycan_trees, source="uniprot_only",
        )
        if site is not None:
            # Preserve the original UniProt tier (the snapped PDB resnum may
            # not have its own CARBOHYD entry, but the site is still backed
            # by the UniProt annotation we started from).
            site.uniprot_position = unp_pos
            site.uniprot_tier = tier
            site.evidence_tier = tier
            site.default_policy = POLICY_DEFAULTS[tier]
            site.expected_tier = tier
            site.evidence_ok = True
            site.evidence_reasons = [
                f"UniProt {tier} at position {unp_pos}",
                f"snapped to nearest motif at MPNN idx {mpnn_idx}",
            ]
            glyco_sites.append(site)
            covered_keys.add((chain_id, mpnn_idx))
            uniprot_only_added.append(site.label)

if uniprot_only_added:
    print(f"\nAdded {len(uniprot_only_added)} UniProt-only site(s) "
          f"missed by motif detection: {', '.join(uniprot_only_added)}")

glyco_sites = sorted(glyco_sites, key=lambda s: (s.chain, s.position_0idx))
session.glyco_sites = glyco_sites
session.assessment = summarize_glycoprotein_status(glyco_sites)

site_inventory_df = pd.DataFrame([
    {"site_label": s.label, "chain": s.chain, "pos_1idx": s.position_1idx,
     "pdb_resnum": s.pdb_resnum, "motif": s.motif,
     "evidence_tier": s.evidence_tier, "resolved_glycan": s.glycan_tree is not None,
     "default_policy": s.default_policy,
     "why_this_site_matters": "; ".join(s.evidence_reasons)}
    for s in glyco_sites
])

evidence_audit_df = pd.DataFrame([
    {"site_label": s.label, "assigned_tier": s.evidence_tier,
     "expected_tier": s.expected_tier, "tier_ok": s.evidence_ok,
     "uniprot_position": s.uniprot_position, "uniprot_tier": s.uniprot_tier,
     "has_glycan_tree": s.glycan_tree is not None}
    for s in glyco_sites
])

print()
print(session.assessment["headline"])
print(f"  {session.assessment['n_sites']} sequon(s), "
      f"{session.assessment['validated_sites']} validated, "
      f"{session.assessment['glycan_tree_sites']} with resolved glycans")
print()
print("Site inventory")
display_df(site_inventory_df)

mismatches = evidence_audit_df.loc[~evidence_audit_df["tier_ok"]] if not evidence_audit_df.empty else pd.DataFrame()
if not mismatches.empty:
    print(f"\nWARNING: {len(mismatches)} evidence tier mismatch(es)")
    display_df(mismatches)

# %% [markdown]
# ### Optional — add manually specified sites
#
# If you know of glycosylation positions that were not picked up by motif detection or UniProt (e.g. from the literature, or non-standard contexts), enter them here as comma-separated `chain:pdb_resnum` (PDB numbering) or `chain:Mmpnn_idx` (1-indexed MPNN position with leading `M`). Leave blank to skip.
#

# %%
# Enter extra sites to preserve, e.g. "A:405, A:512" (PDB numbering)
# or "A:M297" for 1-indexed MPNN position. Leave blank to skip.
EXTRA_SITES = ""

def _parse_extra_sites(spec, chain_order, pdb_to_mpnn_by_chain, chain_seqs):
    out = []
    if not spec.strip():
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        chain, pos_str = token.split(":", 1)
        chain = chain.strip()
        pos_str = pos_str.strip()
        if chain not in chain_order:
            print(f"  WARNING: chain {chain!r} not in {chain_order}")
            continue
        if pos_str.upper().startswith("M"):
            try:
                mpnn_idx = int(pos_str[1:]) - 1
            except ValueError:
                print(f"  WARNING: bad MPNN position {pos_str!r}")
                continue
        else:
            try:
                pdb_resnum = int(pos_str)
            except ValueError:
                print(f"  WARNING: bad PDB resnum {pos_str!r}")
                continue
            mpnn_idx = pdb_to_mpnn_by_chain.get(chain, {}).get(pdb_resnum)
            if mpnn_idx is None:
                print(f"  WARNING: PDB resnum {chain}:{pdb_resnum} not resolved in structure")
                continue
        out.append((chain, mpnn_idx))
    return out

extra_specs = _parse_extra_sites(EXTRA_SITES, chain_order, pdb_to_mpnn_by_chain, chain_seqs)
added_manual = []
for chain_id, mpnn_idx in extra_specs:
    if any(s.chain == chain_id and s.position_0idx == mpnn_idx for s in session.glyco_sites):
        print(f"  Already in inventory: {chain_id}:{mpnn_idx + 1}")
        continue
    site = build_glyco_site(
        chain_id, mpnn_idx, chain_seqs, pdb_to_mpnn_by_chain,
        dbref_ranges, uniprot_evidence, glycan_trees,
        source="user_specified",
    )
    if site is None:
        print(f"  Could not build site at {chain_id}:{mpnn_idx + 1}")
        continue
    # User-specified sites default to functional_preserve unless evidence overrides
    if site.evidence_tier == "motif_only":
        site.default_policy = "functional_preserve"
    session.glyco_sites.append(site)
    added_manual.append(site.label)

if added_manual:
    session.glyco_sites = sorted(session.glyco_sites, key=lambda s: (s.chain, s.position_0idx))
    session.assessment = summarize_glycoprotein_status(session.glyco_sites)
    print(f"Added {len(added_manual)} manual site(s): {', '.join(added_manual)}")
    site_inventory_df = pd.DataFrame([
        {"site_label": s.label, "chain": s.chain, "pos_1idx": s.position_1idx,
         "pdb_resnum": s.pdb_resnum, "motif": s.motif,
         "evidence_tier": s.evidence_tier, "resolved_glycan": s.glycan_tree is not None,
         "default_policy": s.default_policy,
         "why_this_site_matters": "; ".join(s.evidence_reasons)}
        for s in session.glyco_sites
    ])
    display_df(site_inventory_df)
else:
    print("No manual sites added.")

# %% [markdown]
# ## Step 3 — Choose Your Preservation Strategy
#
# **This is the key design decision.** Use the controls below to decide how strictly
# each glycosylation site should be protected during redesign.
#
# You can:
# - Pick a **global strategy** that applies the same policy to all sites
# - Or **customise per site** by changing individual site dropdowns
#   (changing any site dropdown automatically switches to "Mixed site-by-site" mode)
#
# Then run the **Confirm plan** cell below to lock in your choices and see the site-strategy figure.
#

# %%
ROLE_OPTIONS = [
    ("glycoprotein", "Treat as a glycoprotein and preserve important sites"),
    ("uncertain", "Treat as uncertain and decide site-by-site"),
    ("not_glycoprotein", "Do not preserve glycosites by default"),
]
STRATEGY_OPTIONS = [
    ("evidence_aware", "Evidence-aware defaults (recommended)"),
    ("full_sequon", "Full sequon everywhere"),
    ("functional_preserve", "Functional preserve everywhere"),
    ("soft_filter", "Soft filter everywhere"),
    ("mixed_custom", "Mixed site-by-site plan"),
    ("ignore", "Ignore glycosites for now"),
]
SITE_POLICY_OPTIONS = [
    ("Full sequon", "full_sequon"),
    ("Functional preserve", "functional_preserve"),
    ("Soft filter", "soft_filter"),
    ("Ignore", "ignore"),
]
_role_label = dict(ROLE_OPTIONS)
_strategy_label = dict(STRATEGY_OPTIONS)

recommended = session.assessment.get("recommended_role", "uncertain")
if recommended == "glycoprotein":
    _default_strategy = "evidence_aware"
elif recommended == "uncertain":
    _default_strategy = "mixed_custom"
else:
    _default_strategy = "ignore"

if WIDGETS_AVAILABLE:
    _role_w = widgets.RadioButtons(
        options=[(label, key) for key, label in ROLE_OPTIONS],
        value=recommended,
        description="Target role:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="540px"),
    )
    _strategy_w = widgets.Dropdown(
        options=[(label, key) for key, label in STRATEGY_OPTIONS],
        value=_default_strategy,
        description="Strategy:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px", border="2px solid #005f73",
                              padding="4px", border_radius="6px"),
    )
    _baseline_w = widgets.Checkbox(
        value=True,
        description="Run soft-filter baseline (unconstrained comparison)",
        indent=False,
    )
    _reset_btn = widgets.Button(
        description="Reset sites to strategy defaults",
        button_style="warning",
        layout=widgets.Layout(width="280px"),
    )
    _status_html = widgets.HTML()

    _override_widgets = {}
    _override_rows = []
    for site in session.glyco_sites:
        dd = widgets.Dropdown(
            options=SITE_POLICY_OPTIONS,
            value=site.default_policy,
            description="",
            layout=widgets.Layout(width="220px"),
        )
        _override_widgets[site.label] = dd
        site_html = widgets.HTML(
            value=(
                f"<div style='min-width:520px;'>"
                f"<b>{site.label} {site.motif}</b>"
                f" &nbsp;<span style='color:#555;'>{site.evidence_tier}</span>"
                f"<br><span style='font-size:0.9em; color:#666;'>"
                f"{'; '.join(site.evidence_reasons)}</span></div>"
            )
        )
        _override_rows.append(widgets.HBox([site_html, dd]))

    _programmatic_update = False

    def _apply_strategy_to_dropdowns(strategy):
        """Set all per-site dropdowns to match the chosen strategy."""
        global _programmatic_update
        _programmatic_update = True
        policy_map = build_site_policy_map(session.glyco_sites, strategy, interactive=False)
        for site in session.glyco_sites:
            _override_widgets[site.label].value = policy_map[site.label]
        _programmatic_update = False

    def _on_strategy_change(change=None):
        strategy = _strategy_w.value
        if strategy != "mixed_custom":
            _apply_strategy_to_dropdowns(strategy)
        _update_status()

    def _on_site_override(change=None):
        if _programmatic_update:
            return
        if _strategy_w.value != "mixed_custom":
            _strategy_w.unobserve(_on_strategy_change, names="value")
            _strategy_w.value = "mixed_custom"
            _strategy_w.observe(_on_strategy_change, names="value")
        _update_status()

    def _on_role_change(change=None):
        if _role_w.value == "not_glycoprotein":
            if _strategy_w.value != "ignore":
                _strategy_w.value = "ignore"
                return
            _strategy_w.disabled = True
            _baseline_w.disabled = True
        else:
            _strategy_w.disabled = False
            _baseline_w.disabled = False
        _update_status()

    def _on_reset(btn):
        strategy = _strategy_w.value
        if strategy == "mixed_custom":
            strategy = "evidence_aware"
            _strategy_w.unobserve(_on_strategy_change, names="value")
            _strategy_w.value = strategy
            _strategy_w.observe(_on_strategy_change, names="value")
        _apply_strategy_to_dropdowns(strategy)
        _update_status()

    def _update_status():
        _status_html.value = (
            f"<div style='padding:6px 0;'>"
            f"<b>Target role:</b> {_role_label.get(_role_w.value, _role_w.value)}"
            f"<br><b>Strategy:</b> {_strategy_label.get(_strategy_w.value, _strategy_w.value)}"
            f"</div>"
        )

    _role_w.observe(_on_role_change, names="value")
    _strategy_w.observe(_on_strategy_change, names="value")
    for dd in _override_widgets.values():
        dd.observe(_on_site_override, names="value")
    _reset_btn.on_click(_on_reset)

    display(widgets.VBox([
        widgets.HTML("<h4 style='margin:0;'>SugarFix Design Controls</h4>"),
        _role_w,
        widgets.HTML("<h4 style='margin:8px 0 4px 0;'>Preservation strategy</h4>"),
        _strategy_w,
        _baseline_w,
        _status_html,
        widgets.HTML("<h4 style='margin:8px 0 0 0;'>Per-site policies</h4>"),
        widgets.HTML("<span style='color:#666;'>Change any site to automatically switch to Mixed mode.</span>"),
        _reset_btn,
        widgets.VBox(_override_rows),
    ]))
    _on_strategy_change()
    _update_status()
    print("Adjust the controls above, then run the next cell to confirm your plan.")

else:
    print(session.assessment.get("headline", ""))
    print()
    print("Available target roles:")
    for i, (key, label) in enumerate(ROLE_OPTIONS, 1):
        tag = " (recommended)" if key == recommended else ""
        print(f"  {i}. {label}{tag}")
    print()
    print(f"Using recommended defaults: role={recommended}, strategy={_default_strategy}")
    print("To customise, set session.target_role / session.selected_strategy before the next cell.")

# %% [markdown]
# ### Confirm plan and preview
#
# Run this cell to lock in your choices, build the decision table, and render the site-strategy figure.
# You can go back and re-run the cell above to change settings, then re-run this cell to update.
#

# %%
# Read current widget values (or fall back to defaults for non-widget mode)
if WIDGETS_AVAILABLE and '_role_w' in dir():
    _target_role = _role_w.value
    _selected_strategy = _strategy_w.value
    _run_baselines = _baseline_w.value
    _site_policies = {label: dd.value for label, dd in _override_widgets.items()}
else:
    _target_role = session.target_role or recommended
    _selected_strategy = session.selected_strategy or _default_strategy
    _run_baselines = session.run_baselines
    _site_policies = build_site_policy_map(
        session.glyco_sites, _selected_strategy, interactive=False
    )

session.apply_decisions(_target_role, _selected_strategy, _run_baselines, _site_policies)

print("Plan confirmed")
print("-" * 50)
print(f"  Target role:       {session.target_role}")
print(f"  Strategy:          {session.selected_strategy}")
print(f"  Run baselines:     {session.run_baselines}")
print(f"  Sites in plan:     {len(session.site_order)}")
print(f"  Output directory:  {session.run_dir}")
print()

display_df(session.decision_df)

# Render the site-strategy overview figure
session.figure_dir.mkdir(parents=True, exist_ok=True)
site_strategy_figure = session.figure_dir / f"{session.pdb_id}_site_strategy_overview.png"

plot_site_strategy_overview(
    session.decision_df,
    session.chain_seqs,
    session.assessment,
    output_path=site_strategy_figure,
    pdb_path=protein_pdb_path,
)

# %% [markdown]
# ## Step 4 — Build Comparison Conditions And Run ProteinMPNN
#
# This reads the confirmed plan from `session` and runs ProteinMPNN for each condition.
#

# %%
assert session.confirmed, "Run the 'Confirm plan' cell first (Step 3)."

condition_site_policy_maps = {"designer_selected": dict(session.selected_site_policies)}
condition_descriptions = {
    "designer_selected": f"Designer-selected plan ({session.selected_strategy})",
}

# Only soft_filter baseline — constrained baselines (full_sequon, functional_preserve)
# always show 100% retention by construction, so they are not informative.
if session.run_baselines and session.glyco_sites:
    condition_site_policy_maps["soft_filter"] = {
        site.label: "soft_filter" for site in session.glyco_sites
    }
    condition_descriptions["soft_filter"] = (
        "Baseline: no fixed-position constraints (soft-filter only)"
    )

constraints_by_condition = {
    condition_name: build_constraint_bundle(
        session.glyco_sites, site_policy_map,
        condition_name, condition_descriptions[condition_name],
    )
    for condition_name, site_policy_map in condition_site_policy_maps.items()
}

condition_manifest_df = build_condition_manifest(session.glyco_sites, constraints_by_condition)
print("Condition manifest")
display_df(condition_manifest_df)


def run_mpnn_for_condition(condition_name, bundle):
    condition_dir = session.run_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    working_pdb = session.run_dir / f"{pdb_label}.pdb"
    if not working_pdb.exists():
        shutil.copy(protein_pdb_path, working_pdb)

    fixed_jsonl = None
    if bundle.fixed_positions_0idx:
        fixed_jsonl = session.run_dir / f"fixed_positions_{condition_name}.jsonl"
        write_fixed_positions_jsonl(
            pdb_label, bundle.fixed_positions_0idx,
            session.chain_order, fixed_jsonl,
        )

    cmd = [
        sys.executable, str(PROTEINMPNN_RUN),
        "--pdb_path", str(working_pdb),
        "--num_seq_per_target", str(session.num_seqs),
        "--sampling_temp", str(session.sampling_temp),
        "--out_folder", str(condition_dir),
        "--seed", str(session.seed),
        "--pdb_path_chains", " ".join(session.chain_order),
    ]
    if fixed_jsonl is not None:
        cmd.extend(["--fixed_positions_jsonl", str(fixed_jsonl)])

    print(f"Running ProteinMPNN for {condition_name}...")
    try:
        subprocess.run(cmd, cwd=str(PMPNN_DIR), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr[:600] if exc.stderr else str(exc)
        print(f"  ERROR: {stderr}")
        return None

    fasta_path = condition_dir / "seqs" / f"{pdb_label}.fa"
    if not fasta_path.exists():
        print(f"  WARNING: FASTA not found at {fasta_path}")
        return None

    n_records = sum(1 for line in fasta_path.read_text().splitlines() if line.startswith(">"))
    print(f"  OK: {max(0, n_records - 1)} design(s) written")
    return fasta_path


session.run_dir.mkdir(parents=True, exist_ok=True)
pdb_label = protein_pdb_path.stem
fasta_paths = {}
for condition_name, bundle in constraints_by_condition.items():
    fasta_paths[condition_name] = run_mpnn_for_condition(condition_name, bundle)

print(f"\nProteinMPNN runs complete. Outputs in {session.run_dir}")

# %% [markdown]
# ## Step 5 — Score Designs Against The Selected Plan
#

# %%
all_results = {}
for condition_name, fasta_path in fasta_paths.items():
    if fasta_path is None:
        print(f"Skipping {condition_name}: no FASTA")
        continue
    results = score_designs(
        fasta_path, session.glyco_sites, session.chain_order,
        session.selected_site_policies, condition_name,
        read_fasta_sequences, split_mpnn_concat_seq, is_functional_sequon,
    )
    all_results[condition_name] = results
    if results:
        mean_plan = np.mean([r.plan_satisfaction_rate for r in results])
        pass_rate = np.mean([r.passes_selected_plan for r in results])
        print(
            f"{condition_name}: {len(results)} design(s), "
            f"mean plan satisfaction {mean_plan*100:.1f}%, "
            f"pass rate {pass_rate*100:.1f}%"
        )

retention_rows = []
for condition_name, results in all_results.items():
    for result in results:
        for status in result.site_statuses:
            retention_rows.append({
                "design_condition": condition_name,
                "design_id": result.design_id,
                "sample_idx": result.sample_idx,
                "mpnn_score": result.mpnn_score,
                "site_label": status.site.label,
                "chain": status.site.chain,
                "pos_1idx": status.site.position_1idx,
                "wt_motif": status.site.motif,
                "design_motif": status.design_triplet,
                "evidence_tier": status.site.evidence_tier,
                "selected_policy": status.selected_policy,
                "required_for_plan": status.required_for_plan,
                "n_retained": status.n_retained,
                "exact_match": status.exact_match,
                "functional": status.functional,
                "meets_selected_policy": status.meets_selected_policy,
            })

retention_df = pd.DataFrame(retention_rows)

summary_rows = []
top_designs = {}
top_rows = []
ordered_conditions = list(condition_site_policy_maps)
for condition_name in ordered_conditions:
    results = all_results.get(condition_name, [])
    if not results:
        continue
    top_design = select_top_design(results)
    top_designs[condition_name] = top_design
    if top_design is not None:
        top_rows.append({
            "design_condition": condition_name,
            "top_design_id": top_design.design_id,
            "mpnn_score": top_design.mpnn_score,
            "required_sites": top_design.n_sites_required,
            "sites_satisfied": top_design.n_sites_satisfied,
            "passes_selected_plan": top_design.passes_selected_plan,
        })
    condition_rows = retention_df[retention_df["design_condition"] == condition_name]
    required_rows = condition_rows[condition_rows["required_for_plan"]]
    summary_rows.append({
        "design_condition": condition_name,
        "label": CONDITION_LABELS.get(condition_name, condition_name),
        "n_designs": len(results),
        "mean_exact_retention": condition_rows["exact_match"].mean(),
        "mean_functional_retention": condition_rows["functional"].mean(),
        "mean_plan_satisfaction": required_rows["meets_selected_policy"].mean() if not required_rows.empty else 1.0,
        "pass_rate": np.mean([r.passes_selected_plan for r in results]),
        "mean_mpnn_score": np.mean([r.mpnn_score for r in results]),
    })

condition_summary_df = pd.DataFrame(summary_rows)
top_designs_df = pd.DataFrame(top_rows)

if not retention_df.empty:
    site_summary_df = (
        retention_df
        .groupby(["design_condition", "site_label", "wt_motif", "evidence_tier", "selected_policy"])
        .agg(
            exact_retention=("exact_match", "mean"),
            functional_retention=("functional", "mean"),
            plan_satisfaction=("meets_selected_policy", "mean"),
        )
        .reset_index()
        .sort_values(["site_label", "design_condition"])
    )
else:
    site_summary_df = pd.DataFrame()

print()
print("Condition summary")
display_df(condition_summary_df)

print()
print("Top designs")
display_df(top_designs_df)

if not site_summary_df.empty:
    print()
    print("Per-site summary")
    display_df(site_summary_df)

dashboard_figure = session.figure_dir / f"{session.pdb_id}_design_dashboard.png"
plot_design_dashboard(
    condition_summary_df, all_results, top_designs,
    session.site_order, output_path=dashboard_figure,
)
print(f"\nDesign dashboard saved to: {dashboard_figure}")

# %% [markdown]
# ## Step 6 — Export AF3 JSONs And Save Outputs
#

# %%
session.af3_dir.mkdir(parents=True, exist_ok=True)
af3_outputs = {}

for condition_name, top_design in top_designs.items():
    if top_design is None:
        continue
    chain_sequences = [top_design.chain_sequences[cid] for cid in session.chain_order]
    job_name = f"{session.pdb_id}_{condition_name}"

    plain_json = make_af3_server_json(job_name, chain_sequences)
    plain_path = session.af3_dir / f"{job_name}_AF3.json"
    with open(plain_path, "w") as handle:
        json.dump(plain_json, handle, indent=2)

    glycan_positions = {}
    for status in top_design.site_statuses:
        if status.selected_policy not in ("full_sequon", "functional_preserve"):
            continue
        # Emit a stub for every preserved site, including UniProt-recovered
        # sites whose middle residue is unresolved (motif parses as NXT). The
        # `functional` flag requires a strict N-X-S/T match, which excludes
        # those sites even though the design intent is to glycosylate them.
        chain_idx = session.chain_order.index(status.site.chain)
        glycan_positions.setdefault(chain_idx, []).append(
            {"residues": "NAG", "position": status.site.position_1idx}
        )

    glycan_json = make_af3_server_json(
        f"{job_name}_glycans", chain_sequences, glycan_positions,
    )
    glycan_path = session.af3_dir / f"{job_name}_AF3_with_glycans.json"
    with open(glycan_path, "w") as handle:
        json.dump(glycan_json, handle, indent=2)

    af3_outputs[condition_name] = {
        "plain": str(plain_path),
        "glycans": str(glycan_path),
        "n_glycan_stubs": int(sum(len(v) for v in glycan_positions.values())),
    }

# Save all CSVs and session JSON
site_inventory_path = session.run_dir / f"{session.pdb_id}_site_inventory.csv"
evidence_audit_path = session.run_dir / f"{session.pdb_id}_evidence_audit.csv"
decisions_path = session.run_dir / f"{session.pdb_id}_site_decisions.csv"
manifest_path = session.run_dir / f"{session.pdb_id}_condition_manifest.csv"
retention_path = session.run_dir / f"{session.pdb_id}_retention.csv"
site_summary_path = session.run_dir / f"{session.pdb_id}_site_summary.csv"
condition_summary_path = session.run_dir / f"{session.pdb_id}_condition_summary.csv"
top_designs_path = session.run_dir / f"{session.pdb_id}_top_designs.csv"
session_path = session.run_dir / f"{session.pdb_id}_designer_session.json"

figure_outputs = {
    "site_strategy_overview": str(site_strategy_figure),
    "design_dashboard": str(dashboard_figure),
}

site_inventory_df.to_csv(site_inventory_path, index=False)
evidence_audit_df.to_csv(evidence_audit_path, index=False)
session.decision_df.to_csv(decisions_path, index=False)
condition_manifest_df.to_csv(manifest_path, index=False)
retention_df.to_csv(retention_path, index=False)
site_summary_df.to_csv(site_summary_path, index=False)
condition_summary_df.to_csv(condition_summary_path, index=False)
top_designs_df.to_csv(top_designs_path, index=False)

# %% [markdown]
# ## Poster figures
#
# Generate the scaling poster visuals (per-site retention, design×site
# heatmap, central-residue logo, substitution stack). SugarFix panels are
# automatically dropped when retention is 100% (uninformative).

# %%
from pipeline import figures as poster_figures

poster_dir = session.figure_dir / "poster"
poster_paths = poster_figures.make_all(retention_path, poster_dir, palette="C")
for name, p in poster_paths.items():
    print(f"  {name}: {p}")

constraint_payload = {}
for condition_name, bundle in constraints_by_condition.items():
    constraint_payload[condition_name] = {
        "description": bundle.description,
        "site_policies": bundle.site_policies,
        "fixed_positions_1idx": {
            chain_id: [pos + 1 for pos in positions]
            for chain_id, positions in bundle.fixed_positions_0idx.items()
        },
    }

session_payload = {
    **session.decision_payload,
    "figure_outputs": figure_outputs,
    "af3_outputs": af3_outputs,
    "constraint_bundles": constraint_payload,
    "assessment": session.assessment,
}
with open(session_path, "w") as handle:
    json.dump(session_payload, handle, indent=2)

# Copy to Google Drive if opted in
if IN_COLAB and SAVE_TO_DRIVE:
    # Outputs already wrote directly to Drive (session paths were rebased
    # in Step 1). Nothing to copy.
    print(f"Outputs were saved live to Google Drive: {session.run_dir}")

# Final summary
mismatches = evidence_audit_df.loc[~evidence_audit_df["tier_ok"]]
print("=" * 70)
print(f"SUGARFIX DESIGNER SUMMARY - {session.pdb_id}")
print("=" * 70)
print(f"Target role:            {session.target_role}")
print(f"Selected strategy:      {session.selected_strategy}")
print(f"Sites detected:         {session.assessment['n_sites']}")
print(f"Validated sites:        {session.assessment['validated_sites']}")
print(f"Resolved glycan trees:  {session.assessment['glycan_tree_sites']}")
print(f"Evidence audit status:  {'REVIEW' if not mismatches.empty else 'PASS'}")

print()
print("Design-condition summary")
for row in condition_summary_df.itertuples(index=False):
    print(
        f"- {row.label}: functional retention {row.mean_functional_retention*100:.1f}% | "
        f"exact retention {row.mean_exact_retention*100:.1f}% | "
        f"MPNN score {row.mean_mpnn_score:.4f}"
    )

if not top_designs_df.empty:
    print()
    print("Top design per condition")
    for row in top_designs_df.itertuples(index=False):
        print(
            f"- {CONDITION_LABELS.get(row.design_condition, row.design_condition)}: "
            f"{row.top_design_id} | score={row.mpnn_score:.4f} | "
            f"sites {row.sites_satisfied}/{row.required_sites}"
        )

print()
print("Saved outputs")
for item in [site_inventory_path, evidence_audit_path, decisions_path,
             manifest_path, retention_path, site_summary_path,
             condition_summary_path, top_designs_path, session_path]:
    print(f"  {item.name}")
print()
print("AF3 exports")
for condition_name, payload in af3_outputs.items():
    print(
        f"  {CONDITION_LABELS.get(condition_name, condition_name)}: "
        f"{payload['n_glycan_stubs']} glycan stub(s)"
    )
print()
print(f"Run directory: {session.run_dir}")
print("=" * 70)

# %% [markdown]
# ---
# ## Step 7 — Analyze AF3 Validation Results *(Optional)*
#
# After submitting the AF3 JSONs from Step 6 to the AlphaFold 3 Server (or running AF3 locally),
# point this cell at the output directory to extract confidence metrics.
#
# **Expected AF3 output structure:**
# ```
# af3_results/
#   fold_{pdb_id}_{condition}_AF3/
#     *_summary_confidences_0.json
#     *_summary_confidences_1.json
#     ...
# ```
#
# Set `AF3_RESULTS_DIR` below to the folder containing your AF3 outputs.
#

# %%
from pipeline.validate_af3_results import collect_af3_metrics, plot_af3_summary

AF3_RESULTS_DIR = session.run_dir / "af3_results"
# Change this ^^^ to wherever your AF3 output folders are.
# e.g. AF3_RESULTS_DIR = Path("/content/drive/MyDrive/AF3_outputs/1ZXQ")

if AF3_RESULTS_DIR.exists() and any(AF3_RESULTS_DIR.iterdir()):
    af3_metrics_df = collect_af3_metrics(AF3_RESULTS_DIR)

    if not af3_metrics_df.empty:
        print(f"Collected metrics for {len(af3_metrics_df)} AF3 prediction(s)\n")
        display_df(af3_metrics_df)

        af3_fig_path = session.figure_dir / f"{session.pdb_id}_af3_validation.png"
        plot_af3_summary(af3_metrics_df, output_path=af3_fig_path)
        print(f"\nAF3 validation figure: {af3_fig_path}")

        af3_csv_path = session.run_dir / f"{session.pdb_id}_af3_validation.csv"
        af3_metrics_df.to_csv(af3_csv_path, index=False)
        print(f"AF3 metrics saved: {af3_csv_path}")
    else:
        print("No AF3 confidence files found in the provided directory.")
else:
    print(f"AF3 results directory not found: {AF3_RESULTS_DIR}")
    print("Set AF3_RESULTS_DIR to your AF3 output folder and re-run this cell.")

# %% [markdown]
# ## Step 8 — Organize AF3 outputs for PyMOL RMSD analysis *(Optional)*
#
# Takes the same `AF3_RESULTS_DIR` as Step 7 and produces a clean per-protein
# folder with:
#
# * `models/` — top-ranked `.cif` per condition (designer_selected, soft_filter, …)
# * `confidences/` — all seed confidence JSONs
# * `confidence.csv` — per-seed confidence table
# * `load_in_pymol.pml` — opens the crystal structure + all AF3 models in PyMOL
#   and runs per-chain alignment, writing `rmsd_results.txt` and `rmsd_results.csv`.
#
# Open the resulting `.pml` in PyMOL (`pymol load_in_pymol.pml`) to inspect global
# and per-chain RMSD between the crystal and each AF3 design.

# %%
from pipeline.organize_af3_results import organize as organize_af3_for_pymol

if AF3_RESULTS_DIR.exists() and any(AF3_RESULTS_DIR.iterdir()):
    af3_organized_dir = session.run_dir / "af3_organized"
    organized_pdb_dir = organize_af3_for_pymol(
        download_dir=AF3_RESULTS_DIR,
        output_dir=af3_organized_dir,
        pdb_id=session.pdb_id,
        crystal_pdb=protein_pdb_path,
    )
    pml_path = organized_pdb_dir / "load_in_pymol.pml"
    print(f"\nOpen in PyMOL:\n  pymol {pml_path}")
else:
    print(f"AF3 results directory not found: {AF3_RESULTS_DIR}")
    print("Run Step 7 first or set AF3_RESULTS_DIR before running this cell.")

# %% [markdown]
# ## Step 9 — Align AF3 Models in Colab *(Optional)*
#
# PyMOL is still the best local interactive viewer, but Colab can do the
# structural alignment directly in Python. This cell uses the organized AF3
# models from Step 8, aligns them to the prepared reference protein, writes
# aligned PDBs, saves a Colab RMSD table, and opens the best aligned model in
# an inline `py3Dmol` viewer.

# %%
from pipeline.align_af3_structures import align_organized_af3_models

try:
    import py3Dmol
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py3Dmol"], check=True)
    import py3Dmol

if "organized_pdb_dir" not in globals():
    organized_pdb_dir = session.run_dir / "af3_organized" / session.pdb_id

if organized_pdb_dir.exists() and (organized_pdb_dir / "models").exists():
    colab_rmsd_df = align_organized_af3_models(
        organized_pdb_dir=organized_pdb_dir,
        reference_path=protein_pdb_path,
        pdb_id=session.pdb_id,
    )
    display_df(colab_rmsd_df)
    print(f"\nColab alignment RMSD table: {organized_pdb_dir / 'colab_alignment_rmsd.csv'}")
    print(f"Aligned models: {organized_pdb_dir / 'aligned_models'}")

    viewable = colab_rmsd_df.dropna(subset=["rmsd_ca"]).reset_index(drop=True)
    if not viewable.empty:
        SUGAR_RESNS = [
            "NAG", "MAN", "BMA", "FUC", "GAL", "GLC",
            "SIA", "NDG", "FUL", "BGC", "XYS", "RIB",
        ]
        # Identify the model-name column (varies by helper version).
        name_col = next(
            (c for c in ("model_name", "model", "condition", "aligned_model")
             if c in viewable.columns),
            viewable.columns[0],
        )
        labels = viewable[name_col].astype(str).tolist()
        # Default to a glycan-containing model if available.
        default_idx = next(
            (i for i, lbl in enumerate(labels) if "glycan" in lbl.lower()),
            0,
        )

        def _show_model(selected_label):
            row = viewable[viewable[name_col].astype(str) == selected_label].iloc[0]
            model_path = Path(row["aligned_model"])
            view = py3Dmol.view(width=900, height=650)
            view.addModel(Path(protein_pdb_path).read_text(), "pdb")
            view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})
            view.addModel(model_path.read_text(), "pdb")
            view.setStyle({"model": 1}, {"cartoon": {"color": "palegreen"}})
            view.setStyle(
                {"model": 1, "resn": SUGAR_RESNS},
                {"stick": {"colorscheme": "magentaCarbon", "radius": 0.25}},
            )
            view.zoomTo()
            view.show()

        try:
            import ipywidgets as widgets
            from IPython.display import display
            dropdown = widgets.Dropdown(
                options=labels,
                value=labels[default_idx],
                description="Model:",
                layout=widgets.Layout(width="500px"),
            )
            out = widgets.Output()

            def _on_change(change):
                if change["name"] == "value":
                    out.clear_output(wait=True)
                    with out:
                        _show_model(change["new"])

            dropdown.observe(_on_change)
            display(dropdown, out)
            with out:
                _show_model(labels[default_idx])
        except ImportError:
            _show_model(labels[default_idx])
else:
    print(f"Organized AF3 model directory not found: {organized_pdb_dir / 'models'}")
    print("Run Step 8 first, then re-run this cell.")
