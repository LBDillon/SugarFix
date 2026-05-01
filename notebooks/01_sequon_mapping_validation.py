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
# **Research question:** Given a glycoprotein structure, where are all the
# N-X-S/T sequons, and how confident are we that each is actually glycosylated?
#
# This tutorial combines motif detection, PDB-resolved glycan evidence,
# UniProt glycosylation annotations, and numbering/remapping checks into one
# site-confidence report.

# %%
import os
import sys
from pathlib import Path

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

try:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    _REPO_ROOT = Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
