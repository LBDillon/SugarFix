# Sequon Retention Analysis

**Status:** Complete

## Overview

This experiment analyzes how well different design conditions preserve N-X-S/T glycosylation sequons compared to native sequences.

## Purpose

Quantify sequon retention rates across conditions:
- **Unconstrained** - Natural bias of ProteinMPNN
- **Single-fix** - Fixing N position only
- **Multifix** - Fixing full N-X-S/T triplet

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| [04_process_design_results.py](scripts/04_process_design_results.py) | Processes raw design outputs | ✓ |
| [07_corrected_sequon_retention_analysis.py](scripts/07_corrected_sequon_retention_analysis.py) | Analyzes sequon retention rates | ✓ |
| [10_visualize_threeway_comparison.py](scripts/10_visualize_threeway_comparison.py) | Creates three-way comparison plots | ✓ |

## Key Analyses

### Retention Rate Calculation
- Count N-X-S/T triplets in designed vs native sequences
- Per-protein and per-position retention rates
- Statistical significance testing

### Three-Way Comparison
- Unconstrained vs Single-fix vs Multifix
- Violin plots showing distributions
- Statistical tests (ANOVA, post-hoc)

### Position-Specific Analysis
- Which positions are most/least retained
- Edge effects (N-terminus, C-terminus)
- Structural context effects

## Key Results

| Condition | Mean Retention | Std Dev |
|-----------|----------------|---------|
| Unconstrained | ~15-20% | Variable |
| Single-fix (N) | ~40-50% | Variable |
| Multifix (N-X-S/T) | **97.8%** | Very low |

## Visualizations

- Retention rate distributions (violin plots)
- Per-protein retention heatmaps
- Position-specific retention curves
- Statistical comparison plots

## Usage

### Process Design Results

```bash
cd experiments/04_sequon_retention_analysis

python scripts/04_process_design_results.py
```

### Analyze Retention

```bash
python scripts/07_corrected_sequon_retention_analysis.py
```

### Create Comparison Plots

```bash
python scripts/10_visualize_threeway_comparison.py
```

## Output

- `results/sequon_retention/` - Analysis results
- Retention rate statistics
- Comparison figures (PNG/PDF)
- Per-protein detailed results

## Key Findings

1. **Unconstrained designs** show low sequon retention (~15-20%)
2. **Single-fix** improves retention but incomplete (~40-50%)
3. **Multifix** achieves near-perfect retention (97.8%)
4. No significant difference in ProteinMPNN scores between conditions

## Implications

- ProteinMPNN has inherent bias against N-X-S/T sequons
- Explicit constraints required for glycosylation site preservation
- Multifix approach maintains design quality while preserving sequons

## Next Steps

After retention analysis:
1. Compare ProteinMPNN scores ([05_score_analysis](../05_score_analysis/))
2. Create publication figures ([06_publication_visualizations](../06_publication_visualizations/))
3. Write up findings for thesis/paper

## Related Experiments

- [vanilla_multifix](../vanilla_multifix/) - Multifix implementation
- [03_design_generation](../03_design_generation/) - Design generation
- [05_score_analysis](../05_score_analysis/) - Score comparisons

**Last Updated:** 2026-01-23
