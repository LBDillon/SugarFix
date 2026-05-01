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
# # SugarFix Tutorials
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/00_tutorial_index.ipynb)
#
# SugarFix tutorials are organised around glycoprotein research questions.
# The notebooks are deliberately small: they configure a target, call the
# script-backed workflow functions in `pipeline/tutorial_workflows.py`, and
# display the resulting tables. New analysis logic should be added to the
# pipeline modules, not to the notebooks.
#
# ## Tier 1 - Core structural bioinformatics
#
# **Tutorial 01: Sequon Mapping & Validation**
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/01_sequon_mapping_validation.ipynb)
#
# Research question: given a glycoprotein structure, where are all N-X-S/T
# sequons, and how confident are we that each is actually glycosylated?
#
# ## Tier 2 - Sequon-preserving design
#
# **Tutorial 04: Basic Glycoprotein Redesign**
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/04_basic_glycoprotein_redesign.ipynb)
#
# Research question: can we redesign a glycoprotein without disrupting its
# existing glycosylation sites?
#
# ## Later tutorials
#
# Tutorials 05 onward need a little more modelling judgement before they should
# become executable notebooks. The roadmap in `docs/TUTORIAL_ROADMAP.md`
# records the intended research questions, outputs, and missing script-level
# capabilities.

# %%
from pathlib import Path

NOTEBOOK_DIR = Path.cwd() / "notebooks"
if not NOTEBOOK_DIR.exists():
    NOTEBOOK_DIR = Path.cwd()
for path in sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")):
    print(path.name)
