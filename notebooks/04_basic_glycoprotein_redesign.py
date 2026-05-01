# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 04: Basic Glycoprotein Redesign
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/04_basic_glycoprotein_redesign.ipynb)
#
# **Research question:** Can we redesign a glycoprotein without disrupting its
# existing glycosylation sites?
#
# This tutorial reuses Tutorial 01 to identify and validate sequons, then uses
# those sites to build ProteinMPNN constraints. The AF3 export defaults to the
# standalone AlphaFold 3 JSON dialect so glycan ligands and covalent bonds can
# be passed to the full AF3 codebase. Set `AF3_EXPORT_MODE = "alphafoldserver"`
# when you specifically want server-compatible single-NAG glycan stubs.
# Logic lives in `pipeline/tutorial_workflows.py`; this notebook only sets
# parameters and displays results.

# %% [markdown]
# ## Setup
#
# On Google Colab this cell clones the SugarFix repository, installs Python
# dependencies, installs `mkdssp` (for confidence reports), and ensures
# ProteinMPNN will be cloned on first use. Locally it is a no-op as long as
# the notebook is launched from a SugarFix checkout. ProteinMPNN runs faster
# with a GPU runtime.

# %%
import os
import shutil
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/LBDillon/SugarFix.git"
REPO_DIR_NAME = "SugarFix"

if IN_COLAB:
    if not Path(REPO_DIR_NAME).exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR_NAME], check=True
        )
    os.chdir(REPO_DIR_NAME)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
    if shutil.which("mkdssp") is None and shutil.which("dssp") is None:
        subprocess.run(["apt-get", "install", "-yqq", "dssp"], check=False)
    os.environ.setdefault("SUGARFIX_AUTO_CLONE_PROTEINMPNN", "1")

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

from pipeline.tutorial_workflows import (
    build_design_conditions,
    choose_preservation_strategy,
    new_design_session,
    prepare_structure_for_tutorial,
    run_proteinmpnn_designs,
    run_sequon_mapping,
    score_and_export_designs,
    setup_tutorial_environment,
    write_design_outputs,
    write_tutorial01_outputs,
)

# %% [markdown]
# ## Configure the target and design run

# %%
PDB_ID = os.environ.get("SUGARFIX_PDB_ID", "2DH2")
ALPHAFOLD_UNIPROT = os.environ.get("SUGARFIX_ALPHAFOLD_UNIPROT", "")
RUN_LABEL = os.environ.get("SUGARFIX_RUN_LABEL", "tutorial_04")
EXTRA_SITES = os.environ.get("SUGARFIX_EXTRA_SITES", "")
CANDIDATES_CSV = os.environ.get("SUGARFIX_CANDIDATES_CSV") or None

NUM_SEQS = int(os.environ.get("SUGARFIX_NUM_SEQS", "64"))
SAMPLING_TEMP = float(os.environ.get("SUGARFIX_SAMPLING_TEMP", "0.1"))
SEED = int(os.environ.get("SUGARFIX_SEED", "42"))
PRESERVATION_STRATEGY = os.environ.get("SUGARFIX_PRESERVATION_STRATEGY", "evidence_aware")
AF3_EXPORT_MODE = os.environ.get("SUGARFIX_AF3_EXPORT_MODE", "alphafold3")
AUTO_CLONE_PROTEINMPNN = os.environ.get("SUGARFIX_AUTO_CLONE_PROTEINMPNN", "0") == "1"

env = setup_tutorial_environment(auto_clone_proteinmpnn=AUTO_CLONE_PROTEINMPNN)
session = new_design_session(
    PDB_ID,
    repo_root=env["repo_root"],
    run_label=RUN_LABEL,
    num_seqs=NUM_SEQS,
    sampling_temp=SAMPLING_TEMP,
    seed=SEED,
)

print(f"Repository: {env['repo_root']}")
print(f"ProteinMPNN: {env['proteinmpnn_dir']}")
print(f"Run directory: {session.run_dir}")

# %% [markdown]
# ## Identify glycosylation sites

# %%
structure = prepare_structure_for_tutorial(
    session,
    alphafold_uniprot=ALPHAFOLD_UNIPROT,
)
mapping = run_sequon_mapping(
    session,
    protein_pdb_path=structure.protein_pdb_path,
    annotation_structure_path=structure.annotation_structure_path,
    glycan_trees=structure.glycan_trees,
    candidates_csv=Path(CANDIDATES_CSV) if CANDIDATES_CSV else None,
    extra_sites=EXTRA_SITES,
)

display(mapping.site_inventory_df)

# %% [markdown]
# ## Choose sequon-preservation constraints

# %%
decision_df = choose_preservation_strategy(
    session,
    strategy=PRESERVATION_STRATEGY,
    run_baselines=True,
)
constraints = build_design_conditions(session)

display(decision_df)

# %% [markdown]
# ## Run ProteinMPNN

# %%
condition_manifest_df = run_proteinmpnn_designs(
    session,
    protein_pdb_path=structure.protein_pdb_path,
    proteinmpnn_dir=env["proteinmpnn_dir"],
    constraints_by_condition=constraints,
)

display(condition_manifest_df)

# %% [markdown]
# ## Score designs and export AF3 inputs

# %%
design = score_and_export_designs(
    session,
    constraints_by_condition=constraints,
    condition_manifest_df=condition_manifest_df,
    af3_export_mode=AF3_EXPORT_MODE,
    model_seeds=[SEED],
)

display(design.condition_summary_df)
display(design.top_designs_df)

# %% [markdown]
# ## Write tutorial outputs

# %%
tutorial01_outputs = write_tutorial01_outputs(
    mapping,
    structure_path=structure.annotation_structure_path,
)
design_outputs = write_design_outputs(session, mapping, design)

for label, path in {**tutorial01_outputs, **design_outputs}.items():
    print(f"{label}: {path}")

print("AF3 exports")
for condition, payload in design.af3_outputs.items():
    print(condition, payload)
