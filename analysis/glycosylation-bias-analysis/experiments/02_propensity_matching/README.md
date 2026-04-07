# Propensity Score Matching

**Status:** Complete

## Overview

This experiment performs feature comparison and propensity score matching to create a balanced control set of non-glycoproteins matched to glycoproteins on key biophysical properties.

## Purpose

To ensure that any observed differences between glycoproteins and controls are due to glycosylation sites rather than confounding factors like:
- Protein length
- Secondary structure composition
- Solvent accessibility
- Thermal stability
- Expression system

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| [01_feature_comparison_analysis.py](scripts/01_feature_comparison_analysis.py) | Compares features between glycoproteins and candidate controls | ✓ |
| [02_propensity_score_matching.py](scripts/02_propensity_score_matching.py) | Performs propensity score matching | ✓ |

## Method

### Feature Comparison
- Extracts biophysical features from PDB structures
- Compares distributions between glycoproteins and controls
- Identifies potential confounding variables

### Propensity Score Matching
- Calculates propensity scores based on multiple features
- Matches each glycoprotein to similar non-glycoproteins
- Creates balanced control set

## Key Features

Features used for matching:
- Sequence length
- Alpha helix content
- Beta sheet content
- Solvent-accessible surface area
- Resolution
- Expression system

## Results

- **Matched control set** with similar biophysical properties
- **Balanced dataset** for unbiased comparison
- **Statistical validation** of matching quality

## Usage

### Run Feature Comparison

```bash
cd experiments/02_propensity_matching

python scripts/01_feature_comparison_analysis.py
```

### Perform Matching

```bash
python scripts/02_propensity_score_matching.py
```

## Output

- Feature comparison visualizations
- Matched control set manifest
- Matching quality statistics
- Balance diagnostics

## Next Steps

After propensity matching:
1. Generate designs on matched dataset ([03_design_generation](../03_design_generation/))
2. Compare glycoproteins vs matched controls
3. Analyze sequon retention patterns

## Notes

- Propensity score matching reduces selection bias
- Ensures fair comparison between glycoproteins and controls
- Critical for valid statistical inference

**Last Updated:** 2026-01-23
