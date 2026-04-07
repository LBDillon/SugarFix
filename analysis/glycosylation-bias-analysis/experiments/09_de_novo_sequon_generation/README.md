# De Novo Sequon Generation Analysis

**Status:** To Be Completed

## Purpose

Test whether ProteinMPNN ever **introduces** N-X-S/T sequons where they didn't exist in the native structure.

## Research Question

**Does ProteinMPNN actively avoid creating sequons, or is it indifferent?**

## Prompted By

**Supervisor suggestion (Olly):** "You haven't systematically tested whether ProteinMPNN ever introduces sequons. Worth checking across your 32×11 designs."

## Hypotheses

### H1: Active Avoidance
ProteinMPNN actively **avoids** creating sequons.

**Prediction:** De novo sequon generation rate < random expectation

**Evidence:** Would indicate bias against N-X-S/T motifs in training data or architecture.

---

### H2: Indifference
ProteinMPNN is **indifferent** to sequons.

**Prediction:** De novo sequon generation rate ≈ random expectation

**Evidence:** Would suggest no specific bias, just general low retention.

---

### H3: Context-Dependent Generation
ProteinMPNN occasionally **creates** sequons in specific structural contexts.

**Prediction:** De novo rate varies by position (higher in loops, surface sites)

**Evidence:** Would indicate model learns appropriate sequon placement.

---

## Random Expectation

### Calculation

Assuming uniform amino acid distribution:

```python
P(N) = 1/20 = 0.05
P(X) = 18/20 = 0.90  # Any residue except Pro
P(S or T) = 2/20 = 0.10

P(N-X-S/T) = P(N) × P(X) × P(S or T)
           = 0.05 × 0.90 × 0.10
           = 0.0045
           = 0.45% per triplet position
```

### Expected Counts

For a protein of length L:
```python
num_triplet_positions = L - 2
expected_sequons = num_triplet_positions × 0.0045

# Example: 300 residue protein
expected_sequons = 298 × 0.0045 = 1.34 sequons
```

For dataset of 32 proteins, ~100 designs each:
```python
total_designs = 32 × 100 = 3,200
avg_length = 250 residues
total_triplet_positions = 3,200 × 248 = 793,600

expected_total_sequons = 793,600 × 0.0045 ≈ 3,571 sequons

# If ProteinMPNN is indifferent:
# Observed ≈ 3,571 ± sqrt(3,571) ≈ 3,571 ± 60
```

---

## Analysis Plan

### Phase 1: Identify De Novo Sequons

**Script:** `01_identify_de_novo_sequons.py`

```python
# For each design:
# 1. Scan designed sequence for all N-X-S/T motifs
# 2. Check if sequon exists at that position in native
# 3. If not, it's a de novo sequon

de_novo_sequons = []

for design in designs:
    for i in range(len(design) - 2):
        triplet = design[i:i+3]

        if is_sequon(triplet):  # N-X-S/T where X != P
            native_triplet = native_seq[i:i+3]

            if not is_sequon(native_triplet):
                de_novo_sequons.append({
                    'protein': protein_id,
                    'position': i,
                    'native_triplet': native_triplet,
                    'designed_triplet': triplet
                })
```

**Output:** CSV of all de novo sequons

```csv
protein,design_id,position,native_triplet,designed_triplet,rSASA,ss,B_factor
1CF3,design_001,123,KVD,NVS,0.62,C,45.2
2YP7,design_015,87,SAT,NKT,0.45,H,32.1
...
```

---

### Phase 2: Calculate Generation Rates

**Script:** `02_calculate_generation_rates.py`

```python
# Overall generation rate
total_positions = sum(len(d) - 2 for d in all_designs)
total_de_novo = len(de_novo_sequons)

observed_rate = total_de_novo / total_positions
expected_rate = 0.0045

rate_ratio = observed_rate / expected_rate
```

**Metrics:**
- **Observed rate:** De novo sequons per triplet position
- **Expected rate:** 0.45% (random)
- **Rate ratio:** Observed / Expected
- **Statistical test:** Binomial test (p-value)

**Interpretation:**
- Rate ratio < 1.0 → Active avoidance (H1)
- Rate ratio ≈ 1.0 → Indifference (H2)
- Rate ratio > 1.0 → Occasional generation (H3)

---

### Phase 3: Stratify by Structural Context

**Script:** `03_stratify_by_structure.py`

Analyze de novo generation rate by:

1. **Solvent Accessibility**
   - Buried vs exposed positions

2. **Secondary Structure**
   - Helix vs sheet vs loop

3. **B-factor**
   - Rigid vs flexible regions

4. **Existing Sequons Nearby**
   - Distance to nearest native sequon

**Hypothesis:** De novo sequons preferentially appear in:
- Surface-exposed positions
- Loop regions
- Flexible areas
- Similar structural contexts to native sequons

---

### Phase 4: Compare to Native Distribution

**Script:** `04_compare_to_native_distribution.py`

Compare de novo sequons to native sequon distribution:

```python
# Feature comparison
features = ['rSASA', 'B_factor', 'ss', 'depth']

for feature in features:
    native_values = get_feature_values(native_sequons, feature)
    de_novo_values = get_feature_values(de_novo_sequons, feature)

    # Statistical test
    p_value = mannwhitneyu(native_values, de_novo_values)
```

**Question:** Are de novo sequons structurally similar to native sequons?

**Interpretation:**
- Similar → Model learns appropriate placement (good!)
- Different → Model creates sequons in inappropriate contexts (bad!)

---

### Phase 5: Mutation Analysis

**Script:** `05_analyze_mutation_patterns.py`

For de novo sequons, analyze what changed:

```python
# What mutations created the sequon?
native_triplet = "KVD"
designed_triplet = "NVS"

mutations = [
    ("K", "N", position),  # K→N created asparagine
    ("V", "V", position),  # V unchanged (X position)
    ("D", "S", position),  # D→S created serine
]
```

**Questions:**
1. How many mutations required to create sequon?
2. What are the most common mutation patterns?
3. Are mutations conservative or radical?

---

## Expected Results

### Scenario A: Active Avoidance (H1)

```
De Novo Sequon Generation Analysis
===================================

Dataset: 32 proteins, 100 designs each, 3,200 total designs

Total triplet positions analyzed: 793,600
Expected de novo sequons (random): 3,571 ± 60
Observed de novo sequons: 1,245

Observed rate: 0.157% per position
Expected rate: 0.450% per position
Rate ratio: 0.35 (active avoidance)

Binomial test: p < 0.0001

CONCLUSION: ProteinMPNN actively AVOIDS creating N-X-S/T sequons,
generating them at only 35% of the random expectation rate.
```

**Interpretation:** Confirms bias against sequons.

---

### Scenario B: Indifference (H2)

```
De Novo Sequon Generation Analysis
===================================

Observed de novo sequons: 3,542
Expected (random): 3,571 ± 60

Observed rate: 0.446% per position
Expected rate: 0.450% per position
Rate ratio: 0.99 (indifferent)

Binomial test: p = 0.62

CONCLUSION: ProteinMPPN generates sequons at the random expectation
rate, indicating no specific bias against (or for) sequons.
```

**Interpretation:** No specific anti-sequon bias, just low general recovery.

---

### Scenario C: Context-Dependent (H3)

```
De Novo Sequon Generation Analysis
===================================

Overall rate ratio: 0.65 (below random)

Stratified by context:
  Surface-exposed loops:  Rate ratio = 1.23 (above random!) ✓
  Buried helices:         Rate ratio = 0.12 (strongly avoided)

De novo sequons vs native sequons:
  rSASA:     p = 0.42 (not significantly different)
  B-factor:  p = 0.28 (not significantly different)
  SS:        p = 0.15 (not significantly different)

CONCLUSION: ProteinMPPN generates sequons in structurally appropriate
contexts (surface, loops) similar to native sequons, but avoids them
in buried/rigid regions.
```

**Interpretation:** Model learns appropriate placement from training data.

---

## Biological Interpretation

### If de novo generation is LOW (<< random):

**Interpretation:** ProteinMPNN actively disfavors N-X-S/T motifs

**Implications:**
- Confirms bias
- Justifies constraint-based approach
- Explains low retention rates

---

### If de novo generation is RANDOM (≈ expected):

**Interpretation:** No specific bias, just general low sequence fidelity

**Implications:**
- Need to reframe narrative
- Focus on precision preservation, not bias correction
- Suggests training data balance is fine

---

### If de novo generation is CONTEXT-DEPENDENT:

**Interpretation:** Model learns appropriate sequon placement

**Implications:**
- Sophisticated understanding of glycosylation
- Could inform which sites need constraints
- May not need to fix ALL sequons

---

## Integration with Existing Work

### Complements:

1. **Sequon Retention Analysis** ([04_sequon_retention_analysis](../04_sequon_retention_analysis/))
   - Removal rate + generation rate = complete picture

2. **Structural Predictors** ([08_structural_predictors_retention](../08_structural_predictors_retention/))
   - Compare de novo vs retained sequon contexts

3. **Publication Figures** ([06_publication_visualizations](../06_publication_visualizations/))
   - Add de novo generation panel

### New Insights:

- **Complete bias profile:** Both removal and (lack of) generation
- **Validate hypothesis:** Is bias specific to sequons or general?
- **Inform strategy:** Should we encourage de novo generation in some cases?

---

## Timeline

**For thesis:**
- Medium priority (strengthens but not critical)

**For paper:**
- High priority (reviewer will ask!)

**Estimated time:**
- 2 days for sequon scanning
- 1 day for statistical analysis
- 1 day for structural stratification
- 1 day for figures

**Total: ~1 week**

---

## Tools Required

- **Biopython** - Sequence parsing
- **Pandas** - Data manipulation
- **Scipy** - Binomial test, Mann-Whitney U
- **Matplotlib/Seaborn** - Visualization

---

## Notes

This analysis completes the picture:

**Without de novo analysis:**
- "ProteinMPNN has low sequon retention" ✓
- But why? Specific bias or general low recovery? ❓

**With de novo analysis:**
- "ProteinMPNN has low sequon retention" ✓
- "AND rarely generates sequons de novo" ✓
- "Therefore, specific bias against N-X-S/T motifs" ✓✓

Essential for:
1. Confirming bias hypothesis
2. Addressing reviewer questions
3. Complete characterization
4. Publication-quality story

**Last Updated:** 2026-01-23
