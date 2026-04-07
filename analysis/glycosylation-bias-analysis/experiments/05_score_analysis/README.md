# ProteinMPNN Score Analysis

**Status:** Complete

## Overview

This experiment analyzes ProteinMPNN confidence scores across design conditions to determine if fixing sequon positions compromises design quality.

## Purpose

Answer the critical question: **Does fixing glycosylation sites reduce design quality?**

By comparing ProteinMPNN scores (negative log-likelihood) between conditions, we can assess whether position constraints impact model confidence.

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| [11_analyze_proteinmpnn_scores.py](scripts/11_analyze_proteinmpnn_scores.py) | Analyzes score distributions | ✓ |
| [12_analyze_mpnn_scores_and_sequons.py](scripts/12_analyze_mpnn_scores_and_sequons.py) | Correlates scores with sequon retention | ✓ |
| [13_compare_all_condition_scores.py](scripts/13_compare_all_condition_scores.py) | Compares all conditions | ✓ |
| [13b_compare_scores_MATCHED.py](scripts/13b_compare_scores_MATCHED.py) | **Latest - matched protein analysis** | ✓ |

## Key Analyses

### Score Distributions
- Compare mean/median scores across conditions
- Violin plots showing full distributions
- Outlier detection

### Statistical Testing
- Paired t-tests (same proteins across conditions)
- ANOVA for multi-condition comparison
- Effect size calculations (Cohen's d)

### Correlation Analysis
- Score vs sequon retention correlation
- Per-protein effects
- Position-level analysis

## Key Results

### Main Finding: **No Quality Penalty**

| Comparison | p-value | Cohen's d | Interpretation |
|------------|---------|-----------|----------------|
| Unconstrained vs Multifix | p=0.535 | d=-0.073 | No significant difference |
| Single-fix vs Multifix | p>0.05 | d<0.2 | No significant difference |

### Score Statistics

| Condition | Mean Score | Std Dev |
|-----------|------------|---------|
| Unconstrained | 0.891 | 0.068 |
| Multifix | 0.885 | 0.073 |
| Difference | -0.005 | 0.010 |

**Interpretation:** Lower scores = better/higher confidence. No meaningful difference detected.

## Visualizations

- Score distribution violin plots
- Paired comparison plots
- Score vs retention scatter plots
- Per-protein score comparisons

## Usage

### Analyze Score Distributions

```bash
cd experiments/05_score_analysis

python scripts/11_analyze_proteinmpnn_scores.py
```

### Compare Conditions

```bash
python scripts/13_compare_all_condition_scores.py
```

### Matched Protein Analysis (Latest)

```bash
python scripts/13b_compare_scores_MATCHED.py
```

## Output

- `results/score_analysis/` - Analysis results
- Statistical test results
- Comparison figures (PNG/PDF)
- Score distribution tables

## Implications

1. **Fixing sequons does NOT compromise design quality**
2. **Multifix is a viable approach** for preserving glycosylation sites
3. **ProteinMPNN is robust** to position constraints
4. **No trade-off** between sequon retention and model confidence

## Critical Insight

This analysis validates the multifix approach:
- ✓ High sequon retention (97.8%)
- ✓ No quality penalty (p=0.535)
- ✓ Maintains structural integrity
- ✓ Suitable for experimental validation

## Next Steps

After score analysis:
1. Create publication figures ([06_publication_visualizations](../06_publication_visualizations/))
2. Write results section for thesis/paper
3. Consider experimental validation of designs

## Related Experiments

- [vanilla_multifix](../vanilla_multifix/) - Multifix implementation
- [04_sequon_retention_analysis](../04_sequon_retention_analysis/) - Retention rates
- [06_publication_visualizations](../06_publication_visualizations/) - Final figures

## Latest Update (13b_compare_scores_MATCHED.py)

**Date:** Jan 21, 2026 15:31

Most recent analysis with improvements:
- Uses propensity-matched protein set
- More rigorous statistical controls
- Improved visualization
- Comprehensive result reporting

**Last Updated:** 2026-01-23
