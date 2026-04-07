# Benchmark Dataset Creation

**Status:** Complete

## Overview

This experiment creates the glycoprotein benchmark dataset by:
1. Validating glycosylation sites from Glycosite database
2. Finding additional glycoproteins from PDB
3. Scanning all PDBs for N-X-S/T sequons
4. Quality control and file preparation

## Scripts

### Core Pipeline

| Script | Description | Status |
|--------|-------------|--------|
| [00_validate_glycosite_manifest.py](scripts/00_validate_glycosite_manifest.py) | Validates glycosylation sites from Glycosite database | ✓ |
| [01_find_additional_glycoproteins.py](scripts/01_find_additional_glycoproteins.py) | Searches PDB for additional glycoproteins | ✓ |
| [02_scan_all_pdbs_for_sequons.py](scripts/02_scan_all_pdbs_for_sequons.py) | Scans all structures for sequon motifs | ✓ |

### Utilities

| Script | Description |
|--------|-------------|
| [check_pdb_files.py](scripts/check_pdb_files.py) | Validates PDB file integrity |
| [fix_multichain_pdbs.py](scripts/fix_multichain_pdbs.py) | Repairs multichain structure issues |
| [GlycoDownload.py](scripts/GlycoDownload.py) | Downloads glycoprotein structures |

## Data Generated

- `data/glyco_benchmark/raw/glycoproteins/` - PDB files
- `data/glyco_benchmark/manifests/` - Curated protein lists
- Glycosite validation results
- Sequon scanning results

## Key Results

- **22 glycoproteins** curated with validated glycosylation sites
- **Multiple sequons per protein** identified and mapped
- Quality-controlled dataset for design experiments

## Usage

### Run Full Pipeline

```bash
cd experiments/01_benchmark_creation

# Step 1: Validate Glycosite data
python scripts/00_validate_glycosite_manifest.py

# Step 2: Find additional proteins
python scripts/01_find_additional_glycoproteins.py

# Step 3: Scan for sequons
python scripts/02_scan_all_pdbs_for_sequons.py
```

### Quality Control

```bash
# Check PDB files
python scripts/check_pdb_files.py

# Fix multichain issues
python scripts/fix_multichain_pdbs.py
```

## Next Steps

After creating the benchmark:
1. Run propensity matching ([02_propensity_matching](../02_propensity_matching/))
2. Generate designs ([03_design_generation](../03_design_generation/))
3. Analyze results ([04_sequon_retention_analysis](../04_sequon_retention_analysis/))

## Notes

- This dataset forms the foundation for all downstream experiments
- Glycosylation sites are experimentally validated from Glycosite database
- Quality control ensures all structures are suitable for design experiments

**Last Updated:** 2026-01-23
