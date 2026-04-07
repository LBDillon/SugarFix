# Baseline Per-Residue Recovery Analysis

**Status:** To Be Completed

## Purpose

Calculate the baseline per-residue recovery rate in unconstrained ProteinMPNN designs to contextualize sequon retention rates.

## Research Question

**What is the typical recovery rate for ANY residue in unconstrained design?**

This is critical for interpreting sequon retention:
- If baseline recovery is ~40%, then 8-9% sequon retention means ProteinMPNN **actively disfavors** sequons
- If baseline recovery is ~10%, ProteinMPNN is **indifferent** to sequons

## Prompted By

**Supervisor question (Olly):** "What is the typical per-residue recovery rate for any residue in unconstrained design?"

Without this baseline, we can't determine if low sequon retention is due to:
1. Active bias against N-X-S/T motifs specifically
2. General low recovery of any triplet
3. Positional/structural effects

## Analysis Plan

### 1. Per-Residue Recovery Rate

For each protein, compare native vs designed sequences:

```python
# For each position i in protein
if native_seq[i] == designed_seq[i]:
    recovery_count += 1

recovery_rate = recovery_count / sequence_length
```

Calculate:
- **Overall recovery rate** (all residues)
- **Per-amino-acid recovery** (N vs S vs T vs others)
- **Distribution across proteins**

### 2. Triplet Recovery Rate

For each consecutive triplet in native sequence:

```python
# For each position i
native_triplet = native_seq[i:i+3]
designed_triplet = designed_seq[i:i+3]

if native_triplet == designed_triplet:
    triplet_recovery_count += 1

triplet_recovery_rate = triplet_recovery_count / total_triplets
```

Calculate:
- **Overall triplet recovery** (baseline for comparison)
- **N-X-S/T triplet recovery** (our observed 8-9%)
- **Random triplet recovery** (other motifs)

### 3. Stratified Analysis

Break down recovery by:

**A. Amino Acid Type**
- N, S, T (sequon components)
- Hydrophobic (A, V, L, I, M, F, W, Y)
- Hydrophilic (Q, N, S, T)
- Charged (K, R, D, E)
- Special (C, P, G, H)

**B. Secondary Structure**
- Alpha helix
- Beta sheet
- Loop/coil

**C. Solvent Accessibility**
- Buried (rSASA < 20%)
- Intermediate (20% ≤ rSASA ≤ 50%)
- Exposed (rSASA > 50%)

### 4. Statistical Comparison

Test if N-X-S/T triplets are recovered differently than:
- Random triplets
- Other triplets with similar composition
- Triplets in similar structural contexts

## Expected Outcomes

### Hypothesis 1: Active Disfavoring
- **Baseline per-residue recovery:** 30-40%
- **Baseline triplet recovery:** 10-15%
- **N-X-S/T triplet recovery:** 8-9%
- **Conclusion:** ProteinMPNN actively disfavors sequons (significantly below baseline)

### Hypothesis 2: General Low Recovery
- **Baseline per-residue recovery:** 8-10%
- **Baseline triplet recovery:** 1-2%
- **N-X-S/T triplet recovery:** 8-9%
- **Conclusion:** ProteinMPNN has general low fidelity (sequons not special)

### Hypothesis 3: Context-Dependent
- **Overall recovery:** 30-40%
- **Surface loop recovery:** 10-15%
- **N-X-S/T in surface loops:** 8-9%
- **Conclusion:** Low retention due to structural context, not bias against sequons per se

## Scripts to Create

### `01_calculate_per_residue_recovery.py`
- Load native sequences from PDB files
- Load designed sequences from unconstrained designs
- Calculate per-residue recovery rates
- Output: CSV with recovery rates per protein, per position

### `02_calculate_triplet_recovery.py`
- Extract all triplets from native sequences
- Compare to designed sequences
- Calculate triplet recovery rates
- Output: CSV with triplet recovery rates, motif-specific

### `03_stratify_by_structure.py`
- Calculate SASA using DSSP
- Assign secondary structure
- Stratify recovery by structural context
- Output: CSV with recovery vs SASA, secondary structure

### `04_statistical_comparison.py`
- Compare N-X-S/T recovery to baseline
- Statistical tests (t-test, Mann-Whitney)
- Effect size calculations
- Output: Statistical report, figures

### `05_visualize_recovery_patterns.py`
- Per-residue recovery heatmap
- Recovery vs amino acid type (bar plot)
- Recovery vs structural context (violin plot)
- Output: Publication-quality figures

## Expected Output

### Results Summary

```
Baseline Per-Residue Recovery Analysis
=======================================

Overall Recovery Rate: 35.2% ± 8.1%

Per-Amino-Acid Recovery:
  N: 12.3% ± 5.2%  ← Low!
  S: 28.1% ± 7.3%
  T: 25.4% ± 6.9%
  Other: 38.5% ± 8.4%

Triplet Recovery:
  All triplets: 8.1% ± 3.2%
  N-X-S/T: 8.9% ± 2.1%  ← Not significantly different!

Stratified by Structure:
  Buried: 52.3% ± 9.1%
  Exposed: 18.2% ± 6.4%  ← Sequons are typically exposed!

CONCLUSION: Asparagine (N) has inherently low recovery (12.3%).
Combined with typical surface exposure of sequons, 8-9% retention
is expected, not evidence of specific anti-sequon bias.
```

## Implications

**If baseline is high (30-40%):**
- Sequon retention is LOW → ProteinMPNN disfavors sequons
- Strong motivation for multifix approach
- Clear bias to report in paper

**If baseline is low (8-10%):**
- Sequon retention is NORMAL → ProteinMPNN has general low fidelity
- Need to reframe: it's about preserving specific motifs, not correcting bias
- Still valuable for glycoprotein design, but different narrative

**If context-dependent:**
- Need to control for structural features
- May explain variation better than sequence alone
- Informs when/where to apply constraints

## Integration with Existing Work

This analysis will enhance:
- [04_sequon_retention_analysis](../04_sequon_retention_analysis/) - Add baseline comparison
- [05_score_analysis](../05_score_analysis/) - Contextualize quality metrics
- [06_publication_visualizations](../06_publication_visualizations/) - Add baseline to figures

## Timeline

**For thesis defense:**
- High priority - needed to interpret main results

**Estimated time:**
- 2-3 days of analysis
- 1 day for figures
- 1 day for interpretation

## Notes

This was a critical gap identified by supervisor. Completing this analysis will:
1. Properly contextualize sequon retention rates
2. Determine if bias is specific to sequons or general
3. Strengthen interpretation for thesis/paper
4. Address likely reviewer questions

**Last Updated:** 2026-01-23
