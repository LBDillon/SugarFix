# Structural Predictors of Sequon Retention

**Status:** To Be Completed

## Purpose

Quantify structural features that predict whether a glycosylation sequon will be retained in unconstrained ProteinMPNN designs.

## Research Question

**What structural features predict sequon retention?**

## Prompted By

**Supervisor observation (Olly):** "You noted exposed/flexible sites retained better. Quantifying this (SASA, B-factor, secondary structure) would strengthen the analysis."

## Hypothesis

**Exposed, flexible sequons in loops are more likely retained than buried, rigid sequons in helices.**

Rationale:
- **Exposed sites** have fewer packing constraints → easier to accommodate polar residues
- **Flexible sites** (high B-factor) are more tolerant of sequence variation
- **Loop regions** lack rigid secondary structure requirements

## Structural Features to Calculate

### 1. Solvent Accessible Surface Area (SASA)

**Metric:** Relative SASA (rSASA) for each sequon position

```python
# Using DSSP or Biopython
for sequon_position in sequons:
    rSASA_N = SASA(N) / SASA_max(N)
    rSASA_X = SASA(X) / SASA_max(X)
    rSASA_ST = SASA(S/T) / SASA_max(S/T)

    avg_rSASA = mean([rSASA_N, rSASA_X, rSASA_ST])
```

**Categories:**
- Buried: rSASA < 20%
- Intermediate: 20% ≤ rSASA ≤ 50%
- Exposed: rSASA > 50%

**Hypothesis:** Exposed sequons (rSASA > 50%) have higher retention rates.

---

### 2. B-factors (Thermal Flexibility)

**Metric:** Average B-factor for N-X-S/T positions

```python
for sequon_position in sequons:
    B_N = get_b_factor(N_position)
    B_X = get_b_factor(X_position)
    B_ST = get_b_factor(ST_position)

    avg_B_factor = mean([B_N, B_X, B_ST])
```

**Categories:**
- Rigid: B < 30 Ų
- Intermediate: 30 ≤ B ≤ 60 Ų
- Flexible: B > 60 Ų

**Hypothesis:** Flexible sequons (B > 60) have higher retention rates.

---

### 3. Secondary Structure

**Metric:** DSSP secondary structure assignment

```python
for sequon_position in sequons:
    ss_N = get_secondary_structure(N_position)
    # H = helix, E = sheet, C = coil/loop

    if ss_N in ['H']:
        category = 'helix'
    elif ss_N in ['E']:
        category = 'sheet'
    else:
        category = 'loop'
```

**Categories:**
- Alpha helix (H)
- Beta sheet (E)
- Loop/coil (C, T, S, G, B)

**Hypothesis:** Loop sequons have higher retention than helix/sheet sequons.

---

### 4. Burial Depth

**Metric:** Distance to protein surface

```python
# Calculate depth as minimum distance to any surface atom
for sequon_position in sequons:
    depth = min_distance_to_surface(N_position)
```

**Categories:**
- Surface: depth < 3 Å
- Intermediate: 3 ≤ depth < 6 Å
- Core: depth ≥ 6 Å

**Hypothesis:** Surface sequons (depth < 3 Å) have higher retention.

---

### 5. Local Structural Context

**Additional features:**
- **Hydrogen bonding:** Number of H-bonds involving sequon residues
- **Phi/Psi angles:** Backbone dihedral angles (Ramachandran)
- **Neighboring residues:** Composition of residues within 5 Å sphere
- **Structural motif:** Turn, bend, irregular regions

---

## Analysis Plan

### Phase 1: Calculate Structural Features

**Scripts:**
1. `01_calculate_sasa.py` - Calculate SASA using DSSP
2. `02_extract_b_factors.py` - Extract B-factors from PDB files
3. `03_assign_secondary_structure.py` - DSSP secondary structure
4. `04_calculate_burial_depth.py` - Distance to surface

**Output:** CSV file with all structural features per sequon

```csv
protein,chain,position,sequon,retained,rSASA,B_factor,ss,depth
1CF3,A,41,NVS,True,0.65,42.3,C,2.1
1CF3,A,87,NKT,False,0.15,28.7,H,8.3
...
```

### Phase 2: Statistical Analysis

**Scripts:**
5. `05_retention_vs_sasa.py` - Test retention vs SASA
6. `06_retention_vs_bfactor.py` - Test retention vs B-factor
7. `07_retention_vs_ss.py` - Test retention vs secondary structure
8. `08_multivariate_model.py` - Logistic regression with all features

**Statistical Tests:**
- Mann-Whitney U (retained vs not retained)
- Logistic regression (predict retention from features)
- ROC analysis (predictive power)

### Phase 3: Visualization

**Scripts:**
9. `09_create_feature_plots.py` - Violin/box plots for each feature
10. `10_create_heatmap.py` - Retention vs combined features
11. `11_create_pdb_visualization.py` - Pymol/NGLview colored by retention

**Figures:**
- Retention rate vs rSASA (scatter/violin)
- Retention rate vs B-factor (scatter/violin)
- Retention rate vs secondary structure (bar plot)
- ROC curve for logistic regression model

---

## Expected Results

### Example Output

```
Structural Predictors of Sequon Retention
==========================================

Feature Analysis (n=135 sequons across 22 proteins)

1. Solvent Accessibility (rSASA)
   Buried (rSASA<20%):        8.2% retention (n=38)
   Intermediate (20-50%):    15.3% retention (n=51)
   Exposed (rSASA>50%):      28.7% retention (n=46)  ← Significant!

   Mann-Whitney U: p < 0.001
   Effect size: r = 0.42 (medium-large)

2. B-factor (Flexibility)
   Rigid (B<30):     10.1% retention (n=42)
   Intermediate:     18.2% retention (n=58)
   Flexible (B>60):  24.5% retention (n=35)  ← Marginally significant

   Mann-Whitney U: p = 0.023
   Effect size: r = 0.21 (small-medium)

3. Secondary Structure
   Alpha helix:   6.5% retention (n=31)  ← Lowest!
   Beta sheet:   12.3% retention (n=41)
   Loop/coil:    22.8% retention (n=63)  ← Highest!

   Chi-square: p < 0.001

4. Logistic Regression Model
   Features: rSASA + B-factor + SS

   AUC = 0.73 (moderate predictive power)

   Coefficients:
     rSASA:     β = 2.8, p < 0.001  ← Strongest predictor
     B-factor:  β = 0.9, p = 0.042
     SS (loop): β = 1.5, p = 0.008

CONCLUSION: Solvent exposure is the strongest predictor of sequon
retention, followed by secondary structure. Flexible, surface-exposed
sequons in loops are 3.5× more likely to be retained than buried,
rigid sequons in helices.
```

---

## Biological Interpretation

### Why This Matters

**Surface-exposed sequons in loops are:**
1. **The natural location** for N-glycosylation sites
2. **Accessible** to oligosaccharyltransferase (OST) enzyme
3. **Flexible** enough to accommodate bulky glycans

**ProteinMPNN preserves these preferentially because:**
1. Fewer packing constraints at surface
2. Loop regions have less stringent sequence requirements
3. Model may have learned this from training data

**Buried/rigid sequons are:**
1. **Unlikely to be real glycosylation sites** (inaccessible to OST)
2. **Structurally constrained** (must maintain packing)
3. **Better optimized away** to improve stability

### Implication

**Low sequon retention may be APPROPRIATE for buried sites but PROBLEMATIC for exposed sites.**

This analysis will reveal if ProteinMPNN:
- ✓ Appropriately removes non-functional buried sequons
- ✗ Inappropriately removes functional surface sequons

---

## Integration with Existing Work

### Enhances:

1. **Sequon Retention Analysis** ([04_sequon_retention_analysis](../04_sequon_retention_analysis/))
   - Add structural stratification
   - Explain retention variance

2. **Score Analysis** ([05_score_analysis](../05_score_analysis/))
   - Control for structural context
   - Test if quality penalty varies by sequon location

3. **Publication Figures** ([06_publication_visualizations](../06_publication_visualizations/))
   - Add structural feature panels
   - Show retention vs SASA/B-factor

### New Insights:

- **Contextualize retention rates** - Low retention for buried sites may be appropriate
- **Target constraints** - Only fix surface-exposed sequons?
- **Improve design** - Prioritize constraints on functional (exposed) sites

---

## Timeline

**For thesis defense:**
- High priority - strengthens interpretation

**Estimated time:**
- 3 days for feature calculation (DSSP, parsing)
- 2 days for statistical analysis
- 1 day for visualization
- 1 day for interpretation

**Total: ~1 week**

---

## Tools Required

- **DSSP** - Secondary structure and SASA calculation
- **Biopython** - PDB parsing, B-factor extraction
- **BioPandas** - Alternative PDB parsing
- **Scipy/Statsmodels** - Statistical tests, logistic regression
- **Matplotlib/Seaborn** - Visualization
- **Pymol/NGLview** - 3D structural visualization (optional)

---

## Notes

This was specifically requested by supervisor to strengthen the analysis. Quantifying structural predictors will:

1. Explain observed variation in retention rates
2. Distinguish appropriate (buried) vs inappropriate (exposed) sequon removal
3. Inform targeted constraint strategies
4. Address reviewer questions about mechanism
5. Provide biological context for findings

**Last Updated:** 2026-01-23
