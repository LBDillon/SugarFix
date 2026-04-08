# SugarFix

Glycosylation-aware protein redesign with ProteinMPNN.

ProteinMPNN systematically mutates away N-linked glycosylation sites: across 84 glycoproteins analysed, unconstrained designs retain only ~4% of sequons. SugarFix detects these sites, lets you choose how strictly to protect each one, runs ProteinMPNN with those constraints, and exports the designs for AlphaFold 3 validation.

## Quick start

```bash
git clone https://github.com/LBDillon/SugarFix.git
cd SugarFix
bash setup.sh        # installs deps + clones ProteinMPNN
jupyter lab sugarfix_notebook.ipynb
```

Or on Google Colab: open `sugarfix_notebook.ipynb` and run cells top to bottom. The first cell auto-clones the repo, installs dependencies, ProteinMPNN, and `mkdssp`.

## Input options

- **PDB ID** — enter in the config widget; structure is fetched from RCSB.
- **AlphaFold model** — leave the PDB ID as a label and put a UniProt accession (e.g. `P08195`) in the *AF UniProt* widget. The notebook fetches the current AlphaFold DB model via the EBI API and uses the accession directly to seed UniProt glycosylation evidence (AF models have no DBREF block).

## What the notebook does

1. **Configure** target structure (PDB or AlphaFold) and design parameters
2. **Prepare** the structure (download/fetch, extract protein-only PDB, find glycan trees)
3. **Detect** N-X-S/T sequons and assign evidence tiers (experimental, PDB, curator, motif-only)
4. **Choose** a preservation strategy per site (full sequon, functional preserve, soft filter, ignore)
5. **Run ProteinMPNN** with your constraints + an unconstrained baseline for comparison
6. **Score** designs, visualise retention dashboard, and generate poster figures
7. **Export** AF3 Server JSONs (plain + with glycan stubs)
8. *(Optional)* **Analyse** AF3 predictions after running them externally

## Preservation strategies

| Strategy | What it fixes | Use when |
|----------|--------------|----------|
| Full sequon | N, X, and S/T | You need the exact wild-type glycosylation motif |
| Functional preserve | N and S/T (X free) | You need a functional sequon but allow middle-position variation |
| Soft filter | Nothing during design; checks after | You want maximum design freedom and will filter post-hoc |
| Evidence-aware | Per-site based on evidence tier | Default: strict for validated sites, relaxed for motif-only |
| Ignore | Nothing | Site is not important for this design |

## Requirements

- Python 3.9+
- ProteinMPNN (auto-cloned by setup.sh or by the notebook)
- Python packages: see [requirements.txt](requirements.txt)
- For interactive widgets: `ipywidgets` (falls back to text defaults without it)

## Repository structure

```
sugarfix/
  sugarfix_notebook.ipynb       # main notebook (paired with .py via jupytext)
  sugarfix_walkthrough.py       # source of truth for the notebook
  sugarfix_helpers.py           # Dataclasses, in-notebook plotting, scoring
  pipeline/
    prepare_structure.py        # PDB download and parsing
    identify_sequons.py         # Sequon detection + evidence tiers
    extract_pdb_glycans.py      # Glycan tree extraction from LINK records
    mpnn_utils.py               # ProteinMPNN utilities
    generate_af3_jsons.py       # AF3 Server JSON format
    organize_af3_results.py     # Organise AF3 download folders
    validate_af3_results.py     # Post-AF3 confidence metric analysis
    figures.py                  # Scaling poster figures (palettes A/B/C)
  requirements.txt
  setup.sh
  EXPLAINED.md                  # explaination
  README.md                     # This file
```

## Output structure

By default, each protein gets one output folder under `data/outputs/{PDB_ID}/`.
If you enter an optional run label in the notebook, outputs go under
`data/outputs/{PDB_ID}/{run_label}/` instead.

```
data/outputs/1ZXQ/
  1ZXQ_site_inventory.csv
  1ZXQ_evidence_audit.csv
  1ZXQ_site_decisions.csv
  1ZXQ_condition_manifest.csv
  1ZXQ_retention.csv
  1ZXQ_site_summary.csv
  1ZXQ_condition_summary.csv
  1ZXQ_top_designs.csv
  1ZXQ_designer_session.json
  figures/
    1ZXQ_site_strategy_overview.png
    1ZXQ_design_dashboard.png
  af3/
    1ZXQ_designer_selected_AF3.json
    1ZXQ_designer_selected_AF3_with_glycans.json
    ...
  designer_selected/seqs/1ZXQ_protein.fa
  soft_filter/seqs/1ZXQ_protein.fa
```
