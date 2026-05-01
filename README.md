# SugarFix

Glycosylation-aware protein redesign with ProteinMPNN.

ProteinMPNN systematically mutates away N-linked glycosylation sites: across
84 glycoproteins analysed, unconstrained designs retain only ~4% of sequons.
SugarFix detects these sites, lets you choose how strictly to protect each
one, runs ProteinMPNN with those constraints, and exports the designs for
AlphaFold 3 validation.

The package is organised as a small set of tutorial notebooks that wrap a
script-level pipeline. Logic lives in `pipeline/`; the notebooks are thin
researcher-facing wrappers.

## Tutorials

| Tutorial | Topic | Open in Colab |
|---|---|---|
| 00 | Tutorial index | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/00_tutorial_index.ipynb) |
| 01 | Sequon mapping & validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/01_sequon_mapping_validation.ipynb) |
| 04 | Basic glycoprotein redesign | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LBDillon/SugarFix/blob/main/notebooks/04_basic_glycoprotein_redesign.ipynb) |

Tutorials 05-09 are described in `docs/TUTORIAL_ROADMAP.md`.

## Run on Colab

Open any tutorial via the badge above and run the cells top to bottom. The
first setup cell clones the repo, installs Python dependencies, installs
`mkdssp`, and (when needed) clones ProteinMPNN. A GPU runtime is recommended
for Tutorial 04.

## Run locally

```bash
git clone https://github.com/LBDillon/SugarFix.git
cd SugarFix
bash setup.sh        # installs deps + clones ProteinMPNN
jupyter lab notebooks/
```

For a fast terminal smoke test of the design tutorial:

```bash
SUGARFIX_RUN_LABEL=smoke_tutorial_04 \
SUGARFIX_NUM_SEQS=1 \
SUGARFIX_AF3_EXPORT_MODE=both \
python notebooks/04_basic_glycoprotein_redesign.py
```

Outputs are written under `data/outputs/<PDB_ID>/<RUN_LABEL>/`.

## Input options

- **PDB ID** — set `SUGARFIX_PDB_ID` (default `2DH2`); structure is fetched
  from RCSB.
- **AlphaFold model** — set `SUGARFIX_ALPHAFOLD_UNIPROT` to a UniProt
  accession (e.g. `P08195`). The pipeline fetches the current AlphaFold DB
  model via the EBI API and uses the accession directly to seed UniProt
  glycosylation evidence.

## Preservation strategies (Tutorial 04)

| Strategy | What it fixes | Use when |
|----------|--------------|----------|
| Full sequon | N, X, and S/T | You need the exact wild-type glycosylation motif |
| Functional preserve | N and S/T (X free) | You need a functional sequon but allow middle-position variation |
| Soft filter | Nothing during design; checks after | You want maximum design freedom and will filter post-hoc |
| Evidence-aware | Per-site based on evidence tier | Default: strict for validated sites, relaxed for motif-only |
| Ignore | Nothing | Site is not important for this design |

Strategy is selected via `SUGARFIX_PRESERVATION_STRATEGY`.

## Repository layout

```
sugarfix/
  notebooks/
    00_tutorial_index.{py,ipynb}
    01_sequon_mapping_validation.{py,ipynb}
    04_basic_glycoprotein_redesign.{py,ipynb}
  pipeline/
    tutorial_workflows.py     # high-level workflows used by notebooks
    prepare_structure.py      # PDB download and parsing
    identify_sequons.py       # Sequon detection + evidence tiers
    extract_pdb_glycans.py    # Glycan tree extraction from LINK records
    mpnn_utils.py             # ProteinMPNN utilities
    af3_json.py               # AF3 JSON builders
    generate_af3_jsons.py     # AF3 JSON CLI helpers
    organize_af3_results.py   # Organise AF3 download folders
    validate_af3_results.py   # Post-AF3 confidence metric analysis
    align_af3_structures.py   # Align AF3 results to reference
    figures.py                # Figure plotting helpers
  sugarfix_helpers.py         # DesignSession, scoring, in-memory tables
  docs/
    TUTORIAL_ROADMAP.md
  requirements.txt
  setup.sh
  README.md
```

## Output structure

```
data/outputs/<PDB_ID>/<RUN_LABEL>/
  <PDB_ID>_site_inventory.csv
  <PDB_ID>_evidence_audit.csv
  <PDB_ID>_site_decisions.csv
  <PDB_ID>_condition_manifest.csv
  <PDB_ID>_retention.csv
  <PDB_ID>_site_summary.csv
  <PDB_ID>_condition_summary.csv
  <PDB_ID>_top_designs.csv
  <PDB_ID>_designer_session.json
  af3/                          # AF3 input JSONs (Tutorial 04)
  designer_selected/seqs/...    # ProteinMPNN FASTAs
  soft_filter/seqs/...
  tutorial_01_sequon_mapping/   # Tutorial 01 confidence report + tables
```

## Requirements

- Python 3.9+
- ProteinMPNN (auto-cloned by `setup.sh` or by the notebook setup cell)
- Python packages: see [requirements.txt](requirements.txt)
- Optional: `mkdssp` for full confidence reports
