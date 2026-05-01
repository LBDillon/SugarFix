# SugarFix Tutorial Notebooks

These notebooks are intentionally thin: tutorial cells set parameters, call
functions from `pipeline/tutorial_workflows.py`, and display the returned
tables/paths. Analysis and export logic should stay in scripts, not notebooks.

## Current executable tutorials

- `00_tutorial_index.py` - overview and tutorial map.
- `01_sequon_mapping_validation.py` - sequon mapping, UniProt/PDB evidence,
  numbering checks, confidence report, and PyMOL annotation script.
- `04_basic_glycoprotein_redesign.py` - evidence-aware sequon preservation,
  ProteinMPNN design, scoring, and AF3 JSON export.

## Quick smoke test

ProteinMPNN is expected at `./ProteinMPNN` or via `PROTEINMPNN_DIR`.
For a fast terminal check, run:

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
