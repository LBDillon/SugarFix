# File Map: Sequon Analysis Pipeline

## Current Directory Structure

```
sequon_analysis_pipeline/
├── RESULTS_AND_METHODS.md           # Main results document
├── FILE_MAP.md                      # This file
├── corrected_sequon_analysis.csv    # KEY: Per-protein sequon retention (15 gly proteins)
├── structural_features_glycosylated_only.csv  # Structural features (34 gly sites)
│
├── scripts/                         # Analysis scripts
│   ├── 00_baseline_aa_retention.py  # Baseline AA retention
│   ├── 01_prepare_structure.py      # Prepare PDB structures
│   ├── 02_identify_sequons.py       # Find N-X-S/T motifs
│   ├── 03_run_proteinmpnn.py        # Run ProteinMPNN designs
│   ├── 04_analyze_retention.py      # Analyze sequon retention
│   ├── 05_analyze_denovo.py         # Find de novo sequons
│   ├── 06_structural_context.py     # Extract structural features
│   ├── corrected_sequon_analysis.py # **KEY**: Corrected gly-only analysis
│   ├── structural_features_glycosylated_only.py  # Gly-only structural analysis
│   ├── supplementary_analyses.py    # Functional retention + substitution patterns
│   ├── solvent_accessibility_analysis.py  # **NEW**: RSA-stratified analysis
│   └── create_structural_figures.py # Publication figures
│
├── results_HA_case_study/           # HA (1RUZ) detailed case study
│   ├── 1RUZ/
│   │   ├── structure/               # Chain info, summaries
│   │   ├── sequons/                 # Sequon positions
│   │   ├── designs/                 # MPNN designs (3 conditions)
│   │   │   ├── unconstrained/
│   │   │   ├── n_only_fixed/
│   │   │   └── full_sequon_fixed/
│   │   └── analysis/                # Retention analysis
│   └── figures/                     # Publication figures (Fig 1-5)
│
├── results_full_gly/                # 15 glycosylated protein results
│   └── {pdb_id}/                    # Per-protein: 5gv0, 5vst, etc.
│       ├── structure/
│       ├── sequons/
│       ├── designs/unconstrained/
│       └── analysis/
│
├── results_full_non_gly/            # 52 non-glycosylated results (Question 1 only)
│
├── supplementary_figures/           # Figure for Finding 2.5
│   ├── fig_functional_retention.png    # Sequon fate analysis (Fig S1)
│   └── functional_sequon_analysis.csv  # Site-level functional retention data
│
├── solvent_accessibility_analysis/  # **NEW**: Finding 2.8 RSA analysis
│   ├── rsa_stratified_summary.csv      # Summary table
│   ├── main_dataset_n_rsa.csv          # Per-position RSA for main dataset
│   ├── main_dataset_retention.csv      # Per-position retention for main dataset
│   ├── ha_n_rsa.csv                    # HA RSA data
│   ├── ha_retention.csv                # HA retention data
│   ├── fig_main_n_retention.png        # Figure S2a: N retention by RSA
│   ├── fig_main_sequon_retention.png   # Figure S2b: Sequon retention by RSA
│   ├── fig_ha_n_retention.png          # Figure S3a: HA N retention by RSA
│   └── fig_ha_sequon_retention.png     # Figure S3b: HA sequon retention by RSA
│
├── aggregate_comparison/            # Gly vs non-gly AA comparison
│   ├── dataset_comparison.csv
│   └── comparison_figures/
│
├── PDBs_gly/                        # Input PDB files (glycosylated)
├── PDBs_non_gly/                    # Input PDB files (non-glycosylated)
├── configs/                         # Configuration files
│
└── archive/                         # Deprecated/outdated files
    ├── deprecated_mixed_dataset_analysis/
    │   ├── systematic_sequon_analysis.csv      # Mixed gly+non-gly (invalid)
    │   ├── sequon_structural_features.csv      # Mixed dataset features
    │   ├── propensity_matched_pairs.csv        # Gly vs non-gly matching
    │   ├── protein_features_clean.csv          # Protein-level features
    │   └── structural_feature_analysis.py      # Mixed dataset script
    ├── outdated_docs/
    └── test_outputs/
```

---

## Key Files for Each Finding

| Finding | Key Data File | Script |
|---------|---------------|--------|
| 2.1: Sequon-N vs Non-Sequon-N | `corrected_sequon_analysis.csv` | `corrected_sequon_analysis.py` |
| 2.2: Null Model Test | `corrected_sequon_analysis.csv` | `corrected_sequon_analysis.py` |
| 2.3: Per-Protein Heterogeneity | `corrected_sequon_analysis.csv` | `corrected_sequon_analysis.py` |
| 2.4: HA Case Study | `results_HA_case_study/1RUZ/analysis/` | `04_analyze_retention.py` |
| 2.5: Functional Sequon Retention | `supplementary_figures/fig_functional_retention.png` | `supplementary_analyses.py` |
| 2.6: Substitution Patterns | Text/table in RESULTS_AND_METHODS.md | `supplementary_analyses.py` |
| 2.7: Site-Level Heterogeneity | Text/table in RESULTS_AND_METHODS.md | `supplementary_analyses.py` |
| **2.8: RSA-Stratified Analysis** | `solvent_accessibility_analysis/rsa_stratified_summary.csv` | `solvent_accessibility_analysis.py` |
| Structural Features | `structural_features_glycosylated_only.csv` | `structural_features_glycosylated_only.py` |

---

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Glycosylated proteins (successful) | 15 | `corrected_sequon_analysis.csv` |
| Non-glycosylated proteins | 52 | `results_full_non_gly/` |
| Glycosylated sequon sites | 34 | `structural_features_glycosylated_only.csv` |
| Sequon-N retention | 19.96% | Finding 2.1 |
| Non-sequon-N retention | 42.74% | Finding 2.1 |
| Observed sequon retention | 2.49% | Finding 2.2 |
| Expected sequon retention | 3.34% | Finding 2.2 |
| HA unconstrained retention | 26.7% | Finding 2.4 |
| **RSA: Buried sequon-N vs non-sequon-N diff** | -30.1% | Finding 2.8 |
| **RSA: Intermediate sequon-N vs non-sequon-N diff** | -24.0% | Finding 2.8 |
| **RSA: Exposed sequon-N vs non-sequon-N diff** | +12.9% | Finding 2.8 |

---

## What's Archived (and why)

| File | Reason Archived |
|------|-----------------|
| `systematic_sequon_analysis.csv` | Mixed gly + non-gly comparison is methodologically invalid |
| `sequon_structural_features.csv` | Structural features from mixed dataset |
| `propensity_matched_pairs.csv` | Gly vs non-gly matching no longer used |
| `structural_feature_analysis.py` | Script that ran on mixed dataset |
