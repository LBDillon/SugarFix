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
# # Tutorial 01: Sequon Mapping & Validation
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/01_sequon_mapping_validation.ipynb)
#
# **Research question:** Given a glycoprotein structure, where are all the
# N-X-S/T sequons, and how confident are we that each is actually glycosylated?
#
# This tutorial combines motif detection, PDB-resolved glycan evidence,
# UniProt glycosylation annotations, and numbering/remapping checks into one
# site-confidence report. Logic lives in `pipeline/tutorial_workflows.py`;
# this notebook only sets parameters and displays results.

# %% [markdown]
# ## Setup
#
# On Google Colab this cell clones the SugarFix repository, installs Python
# dependencies, installs `mkdssp` (for confidence reports), and ensures
# ProteinMPNN will be cloned on first use. Locally it is a no-op as long as
# the notebook is launched from a SugarFix checkout.

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
    new_design_session,
    prepare_structure_for_tutorial,
    run_sequon_mapping,
    setup_tutorial_environment,
    write_tutorial01_outputs,
)

# %% [markdown]
# ## Configure the target

# %%
PDB_ID = os.environ.get("SUGARFIX_PDB_ID", "2DH2")
ALPHAFOLD_UNIPROT = os.environ.get("SUGARFIX_ALPHAFOLD_UNIPROT", "")
RUN_LABEL = os.environ.get("SUGARFIX_RUN_LABEL", "tutorial_01")
EXTRA_SITES = os.environ.get("SUGARFIX_EXTRA_SITES", "")
CANDIDATES_CSV = os.environ.get("SUGARFIX_CANDIDATES_CSV") or None
AUTO_CLONE_PROTEINMPNN = os.environ.get("SUGARFIX_AUTO_CLONE_PROTEINMPNN", "0") == "1"

env = setup_tutorial_environment(auto_clone_proteinmpnn=AUTO_CLONE_PROTEINMPNN)
session = new_design_session(
    PDB_ID,
    repo_root=env["repo_root"],
    run_label=RUN_LABEL,
)

print(f"Repository: {env['repo_root']}")
print(f"ProteinMPNN: {env['proteinmpnn_dir']}")
print(f"Run directory: {session.run_dir}")

# %% [markdown]
# ## Prepare the structure

# %%
structure = prepare_structure_for_tutorial(
    session,
    alphafold_uniprot=ALPHAFOLD_UNIPROT,
)

display(structure.chains_info)
print(f"Protein-only PDB: {structure.protein_pdb_path}")
print(f"Resolved glycan trees: {len(structure.glycan_trees)}")

# %% [markdown]
# ## Map sequons and evidence

# %%
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
# ## Audit numbering and annotation consistency

# %%
display(mapping.evidence_audit_df)
display(mapping.remapped_annotations_df)
display(mapping.conflict_df)

# %% [markdown]
# ## Write tutorial outputs

# %%
outputs = write_tutorial01_outputs(
    mapping,
    structure_path=structure.annotation_structure_path,
)

for label, path in outputs.items():
    print(f"{label}: {path}")
