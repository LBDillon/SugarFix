# Glycosylation Bias Analysis

A reproducible pipeline for measuring how inverse-folding models (ProteinMPNN) handle N-linked glycosylation sequons, and validating constrained-design solutions with AlphaFold 3.

This repository contains the analysis code behind the finding that **ProteinMPNN systematically destroys ~96% of N-X-S/T glycosylation sequons** in unconstrained designs, and that fixing sequon positions preserves them without compromising design quality.

For the interactive single-protein design tool that applies these findings, see [SugarFix](https://github.com/YOUR_USERNAME/sugarfix).

## Key findings this pipeline reproduces

1. **~96% sequon loss** in unconstrained ProteinMPNN designs (N retention ~25% vs ~56% at non-sequon asparagines)
2. **Motif-driven bias** — validated and unvalidated sites are equally affected (Wilcoxon p=0.52)
3. **Full-sequon constraints** restore ~98% retention with **no quality penalty** (paired t-test p=0.535)
4. **AF3 structural validation** — constrained designs fold correctly (98% < 2A RMSD to native)

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/glycosylation-bias-analysis.git
cd glycosylation-bias-analysis
bash setup.sh
```

Need to publish this folder as its own repository first? See [UPLOAD_TO_NEW_GITHUB_REPO.md](UPLOAD_TO_NEW_GITHUB_REPO.md).

## Reproducing with your own proteins

### 1. Prepare a candidate list

Edit `data/candidates_template.csv` with your proteins:

```csv
pdb_id,chain_id,uniprot_id,gene_name,organism,glyco_type,n_glycosites,resolution,notes
1RUZ,A,P03437,HA,Influenza A,N-linked,7,1.90,Your notes here
```

Or use `case_studies/utilities/screen_uniprot_candidates.py` to screen UniProt for suitable glycoproteins automatically.

### 2. Run the benchmark pipeline

The experiments are numbered in execution order:

| Step | Folder | What it does |
|------|--------|-------------|
| 1 | `experiments/01_benchmark_creation/` | Validate glycosites, download PDBs, scan for sequons |
| 2 | `experiments/02_propensity_matching/` | Create feature-matched control set |
| 3 | `experiments/03_design_generation/` | Run ProteinMPNN under 3 conditions (unconstrained, N-only, full sequon) |
| 4 | `experiments/04_sequon_retention_analysis/` | Quantify per-site sequon retention |
| 5 | `experiments/05_score_analysis/` | Compare MPNN scores across conditions |
| 6 | `experiments/06_publication_visualizations/` | Generate publication figures |
| 7 | `experiments/07_baseline_recovery_analysis/` | Per-residue baseline recovery rates |
| 8 | `experiments/08_structural_predictors_retention/` | RSA, B-factor, DSSP vs retention |
| 9 | `experiments/09_de_novo_sequon_generation/` | De novo sequon emergence rates |

### 3. Run case studies (per-protein deep dive)

For detailed per-protein analysis with AF3 validation:

```bash
# Single protein
cd case_studies
bash runners/run_case_study_end_to_end.sh 1RUZ

# Batch processing
bash runners/run_batch_pipeline.sh
```

The case study pipeline runs these steps per protein:

| Step | Scripts | What it does |
|------|---------|-------------|
| Prep | `01_preparation/` | Download PDB, extract protein-only, find glycan trees, identify sequons with evidence tiers |
| Design | `02_design/` | Run ProteinMPNN under unconstrained + constrained conditions |
| Analyze | `03_analysis/` | Score retention, calculate RSA/B-factors, find de novo sequons |
| AF3 gen | `04_af3_generation/` | Export AF3 JSONs (plain, with glycans, clash detection) |
| AF3 val | `05_af3_validation/` | Analyze AF3 predictions (RMSD, pTM, clash analysis) |
| Compare | `06_cross_protein/` | Pool results across all proteins |

### 4. Supplementary experiments

**Clash analysis** — test whether redesigned surfaces create steric conflicts with glycans:
- `case_studies/04_af3_generation/generate_clash_af3_jsons.py` — generate clash-detection AF3 inputs
- `case_studies/05_af3_validation/analyze_clash_results.py` — analyze clash results

**Corrected per-protein analysis** (alternative pipeline):
- `sequon_analysis_pipeline/` — independent analysis system with corrected methodology
- Includes RSA-stratified analysis, baseline AA retention, structural feature extraction

## Repository structure

```
glycosylation-bias-analysis/
  experiments/
    01_benchmark_creation/       # Build glycoprotein dataset
    02_propensity_matching/      # Feature-matched controls
    03_design_generation/        # ProteinMPNN designs
    04_sequon_retention_analysis/# Retention quantification
    05_score_analysis/           # Design quality assessment
    06_publication_visualizations/# Publication figures
    07_baseline_recovery_analysis/# Per-residue baselines
    08_structural_predictors_retention/# Structural context
    09_de_novo_sequon_generation/# De novo sequon rates
  case_studies/
    01_preparation/              # Structure prep + sequon ID
    02_design/                   # ProteinMPNN with constraints
    03_analysis/                 # Design scoring
    04_af3_generation/           # AF3 JSON export + clash detection
    05_af3_validation/           # AF3 output analysis
    06_cross_protein/            # Multi-protein comparison
    utilities/                   # UniProt screening
    runners/                     # Shell wrappers for batch execution
    scripts/                     # Publication figures + specialized analyses
  sequon_analysis_pipeline/      # Corrected independent analysis
  data/
    candidates_template.csv      # Template for your protein list
    glyco_benchmark/             # Placeholder data directories
  requirements.txt
  setup.sh
```

## Requirements

- Python 3.9+
- ProteinMPNN (auto-cloned by setup.sh)
- For AF3 validation: access to AlphaFold 3 Server or local AF3
- For structural alignment: PyMOL (optional, for RMSD calculations)
- Python packages: see [requirements.txt](requirements.txt)

## Citation

If you use this analysis pipeline in your work, please cite the associated publication (forthcoming).
