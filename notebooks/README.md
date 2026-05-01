# SugarFix Tutorial Notebooks

These notebooks are intentionally thin: cells set parameters, call functions
from `pipeline/tutorial_workflows.py`, and display the returned tables/paths.
Analysis and export logic stays in scripts, not notebooks.

Each notebook is paired (jupytext) as both `.py` and `.ipynb`. Edit the `.py`
form and regenerate the `.ipynb` with `jupytext --to ipynb --update <file.py>`.

## Current executable tutorials

- `00_tutorial_index.{py,ipynb}` - overview and tutorial map.
- `01_sequon_mapping_validation.{py,ipynb}` - sequon mapping, UniProt/PDB
  evidence, numbering checks, confidence report, and PyMOL annotation script.
- `04_basic_glycoprotein_redesign.{py,ipynb}` - evidence-aware sequon
  preservation, ProteinMPNN design, scoring, and AF3 JSON export.

## Open in Colab

Each notebook starts with an "Open In Colab" badge. The first cell detects
Colab, clones the SugarFix repo, installs Python dependencies, installs
`mkdssp`, and (for Tutorial 04) clones ProteinMPNN on first use. Locally the
cell is a no-op as long as the notebook is launched from a SugarFix checkout.

## Local quick smoke test

ProteinMPNN is expected at `./ProteinMPNN` or via `PROTEINMPNN_DIR`.
For a fast terminal check:

```bash
SUGARFIX_RUN_LABEL=smoke_tutorial_04 \
SUGARFIX_NUM_SEQS=1 \
SUGARFIX_AF3_EXPORT_MODE=both \
python notebooks/04_basic_glycoprotein_redesign.py
```

The output will be written under `data/outputs/<PDB_ID>/<RUN_LABEL>/`.

## Planned tutorials

- Tutorial 05: glycan shield analysis.
- Tutorial 06: multi-domain glycoprotein design.
- Tutorial 07: de novo glycan placement.
- Tutorial 08: glycan removal with compensation.
- Tutorial 09: immunogenicity reduction via glycan placement.

See `docs/TUTORIAL_ROADMAP.md` for research questions and missing
script-level capabilities.
