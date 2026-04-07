# Publication Visualizations

**Status:** Complete

## Overview

This experiment creates publication-quality figures for thesis/paper, combining results from all previous experiments into clear, compelling visualizations.

## Purpose

Generate final figures that communicate:
1. **Sequon retention rates** across conditions
2. **ProteinMPNN score comparisons** showing no quality penalty
3. **Statistical validation** of multifix approach
4. **Overall experimental workflow** and findings

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| [08_create_publication_visualizations.py](scripts/08_create_publication_visualizations.py) | Creates initial publication figures | ✓ |
| [14_create_summary_figure.py](scripts/14_create_summary_figure.py) | Creates summary figure | ✓ |
| [14b_create_summary_figure_CORRECTED.py](scripts/14b_create_summary_figure_CORRECTED.py) | **Latest - corrected summary figure** | ✓ |

## Generated Figures

### Main Figures

1. **Sequon Retention Comparison** (3-panel)
   - Unconstrained vs Single-fix vs Multifix
   - Violin plots with statistical annotations
   - Per-protein retention rates

2. **ProteinMPNN Score Analysis** (2-panel)
   - Score distributions across conditions
   - Paired comparison plots with p-values

3. **Summary Figure** (Multi-panel)
   - Experimental workflow
   - Key results at a glance
   - Statistical validation

### Supplementary Figures

- Per-protein detailed analyses
- Position-specific retention curves
- Feature comparison (propensity matching)
- Design example visualizations

## Figure Specifications

- **Format:** PNG (high-res) and PDF (vector)
- **Resolution:** 300+ DPI for publication
- **Color scheme:** Colorblind-friendly palettes
- **Font:** Arial/Helvetica (standard for journals)
- **Size:** Standard figure dimensions (single/double column)

## Usage

### Create Initial Figures

```bash
cd experiments/06_publication_visualizations

python scripts/08_create_publication_visualizations.py
```

### Generate Summary Figure (Latest)

```bash
python scripts/14b_create_summary_figure_CORRECTED.py
```

## Output

- `figures/` - All publication figures
  - `figure_1_sequon_retention.png/pdf`
  - `figure_2_score_comparison.png/pdf`
  - `figure_3_summary.png/pdf`
  - `supplementary_*.png/pdf`

## Figure Contents

### Figure 1: Sequon Retention

**Panels:**
- A) Unconstrained design retention (~15-20%)
- B) Single-fix design retention (~40-50%)
- C) Multifix design retention (97.8%)
- Statistical annotations (p-values, effect sizes)

### Figure 2: Score Comparison

**Panels:**
- A) Score distributions (violin plots)
- B) Paired protein comparisons
- Statistical results (p=0.535, no significant difference)

### Figure 3: Summary Figure

**Panels:**
- A) Experimental design workflow
- B) Benchmark dataset composition
- C) Key results table
- D) Main conclusion

## Key Messages

1. **ProteinMPNN has inherent bias** against N-X-S/T sequons
2. **Multifix approach solves the problem** (97.8% retention)
3. **No quality penalty** (p=0.535)
4. **Validated approach** for glycoprotein design

## Latest Update (14b_create_summary_figure_CORRECTED.py)

**Date:** Jan 21, 2026 16:10

Most recent figure generation with:
- Corrected statistical annotations
- Improved color scheme
- Better layout
- Publication-ready quality

## Usage in Publications

These figures are ready for inclusion in:
- Master's thesis
- Journal manuscript
- Conference presentations
- Grant applications

## Next Steps

After creating figures:
1. Incorporate into thesis/paper
2. Get feedback from advisors
3. Refine based on reviewer comments
4. Submit for publication

## Related Experiments

All figures integrate results from:
- [01_benchmark_creation](../01_benchmark_creation/)
- [02_propensity_matching](../02_propensity_matching/)
- [03_design_generation](../03_design_generation/)
- [04_sequon_retention_analysis](../04_sequon_retention_analysis/)
- [05_score_analysis](../05_score_analysis/)
- [vanilla_multifix](../vanilla_multifix/)

**Last Updated:** 2026-01-23
