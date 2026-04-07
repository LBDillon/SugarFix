# Protein Design Generation

**Status:** Complete

## Overview

This experiment generates protein sequences using ProteinMPNN under various conditions:
1. **Unconstrained** - No position constraints
2. **Single-fix** - Fix N position only
3. **Multifix** - Fix full N-X-S/T triplets (see [vanilla_multifix](../vanilla_multifix/))

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| [03_mpnn_design_experiment.py](scripts/03_mpnn_design_experiment.py) | Main design generation pipeline | ✓ |
| [03b_mpnn_design_missing_proteins.py](scripts/03b_mpnn_design_missing_proteins.py) | Generates designs for missing proteins | ✓ |
| [03b_mpnn_design_expanded_dataset.py](scripts/03b_mpnn_design_expanded_dataset.py) | Expands dataset with additional designs | ✓ |
| [03d_generate_missing_designs.py](scripts/03d_generate_missing_designs.py) | Fills in missing designs | ✓ |
| [merge_proteinmpnn_results.py](scripts/merge_proteinmpnn_results.py) | Merges design results from multiple runs | ✓ |

## Design Conditions

### 1. Unconstrained Design
- No fixed positions
- ProteinMPNN freely redesigns all residues
- Baseline for comparison

### 2. Single-Fix (N only)
- Fix asparagine (N) position only
- Allow X and S/T to vary
- Tests if fixing N is sufficient

### 3. Multifix (N-X-S/T)
- Fix all three positions of sequon
- Separate experiment: [vanilla_multifix](../vanilla_multifix/)
- Achieves ~98% sequon retention

## Generated Data

- `data/glyco_benchmark/designs/` - Design outputs
- FASTA files with designed sequences
- ProteinMPNN scores for each design
- Fixed position specifications

## Usage

### Generate Unconstrained Designs

```bash
cd experiments/03_design_generation

python scripts/03_mpnn_design_experiment.py
```

### Generate Missing Designs

```bash
python scripts/03b_mpnn_design_missing_proteins.py
python scripts/03d_generate_missing_designs.py
```

### Merge Results

```bash
python scripts/merge_proteinmpnn_results.py
```

## Key Parameters

- **Number of designs per protein:** 100
- **Temperature:** 0.1 (default)
- **Backbone noise:** 0.0 (no noise)
- **Model:** vanilla ProteinMPNN v1.0

## Next Steps

After design generation:
1. Process results ([04_sequon_retention_analysis](../04_sequon_retention_analysis/))
2. Analyze sequon retention rates
3. Compare ProteinMPNN scores across conditions
4. Create publication figures

## Related Experiments

- [vanilla_multifix](../vanilla_multifix/) - Complete multifix implementation
- [04_sequon_retention_analysis](../04_sequon_retention_analysis/) - Retention analysis
- [05_score_analysis](../05_score_analysis/) - Score comparisons

## Notes

- Designs are generated using vanilla ProteinMPNN for consistency
- Multiple conditions allow comparison of constraint strategies
- Results inform optimal approach for preserving glycosylation sites

**Last Updated:** 2026-01-23
