# ProteinMPNN Glycosylation Sequon Analysis: Results and Methods

## Research Questions

1. **Does ProteinMPNN redesign glycosylated proteins more or less aggressively overall?**
   - Comparison of amino acid retention between glycosylated and non-glycosylated protein datasets

2. **Does ProteinMPNN preserve glycosylation sequons in glycosylated proteins?**
   - Within glycosylated proteins only: comparison of N retention at sequon vs non-sequon sites
   - Null model test: observed vs expected sequon retention

---

## Methods

### 1. Dataset Curation

#### 1.1 Glycosylated Protein Dataset
- **Source**: PDB structures with confirmed N-linked glycosylation annotations
- **Final count**: 52 proteins attempted, 15 with successful designs
- **Selection criteria**:
  - Presence of N-linked glycan modifications in PDB
  - Resolution sufficient for backbone modeling
  - Structures processed to remove glycan atoms, retaining protein-only coordinates

#### 1.2 Non-Glycosylated Protein Dataset
- **Source**: PDB structures without glycosylation annotations
- **Final count**: 75 proteins
- **Selection criteria**:
  - No glycosylation modifications in PDB metadata
  - Matched for general structural quality criteria

> **[UNCERTAINTY FLAG - Dataset Composition]**
> The datasets were not rigorously matched for confounding variables such as:
> - Protein size distribution
> - Secondary structure composition
> - Solvent accessibility profiles
> - Taxonomic origin
> - Functional class
>
> These differences may contribute to observed retention differences independent of glycosylation status.

### 2. ProteinMPNN Design Generation

#### 2.1 Design Parameters
- **Model**: ProteinMPNN v_48_020
- **Sampling temperature**: 0.1 (low temperature for more deterministic sampling)
- **Designs per protein**: 32 sequences
- **Seed**: 42 (for reproducibility)
- **Fixed chains**: None (all chains designed)

#### 2.2 HA Case Study Design Conditions
Three conditions were tested on influenza hemagglutinin (HA, PDB: 1RUZ):

| Condition | Description | Fixed Positions |
|-----------|-------------|-----------------|
| Unconstrained | All positions designable | None |
| N-only fixed | Asparagine at sequon sites fixed | N position only |
| Full sequon fixed | Entire N-X-S/T motif fixed | N, X, and S/T positions |

### 3. Sequon Detection

N-linked glycosylation sequons were identified using the canonical motif:
```
N-X-S/T where X ≠ P
```

- **N**: Asparagine (glycan attachment site)
- **X**: Any amino acid except proline
- **S/T**: Serine or threonine

Proline at the X position is excluded because its cyclic structure prevents the conformational change required for oligosaccharyltransferase recognition.

### 4. Retention Metrics

#### 4.1 Amino Acid Retention
For each amino acid type:
```
Retention(AA) = (Count of AA preserved in designs) / (Count of AA in wild-type) × 100%
```

#### 4.2 Sequon Retention (Exact Match)
```
Sequon Retention = (Count of N-X-S/T motifs fully preserved) / (Total sequon sites × designs) × 100%
```

A sequon is considered "retained" only if ALL three positions match the wild-type.

#### 4.3 Functional Sequon Retention
```
Functional Retention = (Count of ANY valid N-X-S/T at original position) / (Total sequon sites × designs) × 100%
```

A "functional" sequon is any valid N-X-S/T motif (X≠P), regardless of whether it matches the original sequence.

#### 4.4 N Retention at Sequon Sites
```
N Retention = (Count of N preserved at sequon N-positions) / (Total N-positions × designs) × 100%
```

---

## Results

### Question 1: Does ProteinMPNN redesign glycosylated proteins more or less aggressively overall?

#### Finding 1.1: Overall design aggressiveness is similar

| Dataset | Mean Retention | Std Dev |
|---------|---------------|---------|
| Glycosylated (n=15) | 40.7% | 24.8% |
| Non-Glycosylated (n=52) | 41.3% | 21.2% |

> **[UNCERTAINTY FLAG - Sample Size Imbalance]**
> Only 15 glycosylated proteins yielded successful ProteinMPNN designs vs. 52 non-glycosylated. This 3:1 imbalance reduces statistical power for detecting differences.

#### Finding 1.2: Specific amino acids show differential retention

Key differences (Glycosylated - Non-Glycosylated):

| Amino Acid | Gly % | Non-Gly % | Difference |
|------------|-------|-----------|------------|
| Cysteine (C) | 87.1% | 56.5% | **+30.6%** |
| Histidine (H) | 4.7% | 24.2% | **-19.5%** |
| Asparagine (N) | 36.6% | 28.2% | **+8.4%** |
| Serine (S) | 36.9% | 29.1% | **+7.8%** |
| Phenylalanine (F) | 39.4% | 55.3% | **-15.9%** |

> **[UNCERTAINTY FLAG - Multiple Testing]**
> With 20 amino acids compared, some differences may be significant by chance alone. No multiple testing correction was applied.

#### Finding 1.3: Sequon-relevant residues (N, S) are better conserved in glycosylated proteins

- **Asparagine (N)**: +8.4% (36.6% vs 28.2%)
- **Serine (S)**: +7.8% (36.9% vs 29.1%)
- **Threonine (T)**: -2.5% (43.3% vs 45.8%)

---

### Question 2: Does ProteinMPNN preserve glycosylation sequons in glycosylated proteins?

> **[METHODOLOGICAL NOTE]**
> This analysis focuses **only on glycosylated proteins** where N-X-S/T motifs represent actual functional glycosylation sites.

#### Finding 2.1: N is retained at lower rates at sequon sites

| Position Type | N Retained | Total | Retention Rate |
|--------------|-----------|-------|----------------|
| N at sequon sites | 281 | 1,408 | **19.96%** |
| N at non-sequon sites | 1,778 | 4,160 | **42.74%** |
| **Difference** | | | **-22.78%** |

#### Finding 2.2: Sequon retention is lower than null expectation

| Component | Retention Rate |
|-----------|---------------|
| P(N retained at sequon) | 19.96% |
| P(X retained at sequon) | 47.23% |
| P(S/T retained at sequon) | 35.44% |
| **Expected** (if independent) | 19.96% × 47.23% × 35.44% = **3.34%** |
| **Observed** | **2.49%** |
| **Ratio (Observed/Expected)** | **0.74** |

#### Finding 2.3: Per-protein heterogeneity

**Distribution of per-protein sequon retention (n=14 proteins with sequons):**

| Retention | Count | Proteins |
|-----------|-------|----------|
| 0% | 9 | 7xcb, 7abv, 6aex, 5zo1, 1i9e, 1l6z, 1iko, 6c10, 3mj6 |
| 0.1-5% | 3 | 5vst (0.8%), 5l74 (0.8%), 5hrt (0.8%) |
| >5% | 2 | 5gv0 (9.7%), 5jyj (12.5%) |

**5gv0 dominates the statistics**: This single protein accounts for **28/35 (80%)** of all retained sequons.

**Sequon-N vs Non-Sequon-N per protein:**

| PDB | Sequons | Seq-N% | Non-Seq-N% | Diff | Full Seq% |
|-----|---------|--------|------------|------|-----------|
| 5jyj | 1 | 96.9% | 47.6% | **+49.3%** | 12.5% |
| 5gv0 | 9 | 26.0% | 24.1% | **+1.9%** | 9.7% |
| 1i9e | 1 | 34.4% | 48.4% | -14.1% | 0.0% |
| 5vst | 4 | 34.4% | 45.3% | -10.9% | 0.8% |
| 7xcb | 4 | 7.8% | 42.7% | -34.9% | 0.0% |
| 6aex | 3 | 2.1% | 68.8% | -66.7% | 0.0% |
| 3mj6 | 3 | 9.4% | 60.0% | -50.6% | 0.0% |

- 12/14 proteins: sequon-N retained WORSE than non-sequon-N
- 2/14 proteins: sequon-N retained BETTER (5gv0, 5jyj)

#### Finding 2.4: HA Case Study

> **Important**: The HA (1RUZ) case study was analyzed **separately** and is NOT included in the 15 glycosylated proteins above.

##### 2.4.1: Site-specific variation in HA (exact match retention)

| Position | Sequon | Exact Retention | N Retention |
|----------|--------|-----------------|-------------|
| 9 | NNS | **52.1%** | 100% |
| 22 | NVT | 44.8% | 96.9% |
| 87 | NGT | **0%** | 6.2% |
| 153 | NGT | 36.5% | 43.6% |
| 287 | NSS | **0%** | 28.1% |
| **Total** | | **26.7%** | 55.2% |

##### 2.4.2: Effect of fixing N position

| Condition | Sequon Retention | N Retention |
|-----------|-----------------|-------------|
| Unconstrained | 128/480 (26.7%) | 265/480 (55.2%) |
| N-only fixed | 181/480 (37.7%) | 480/480 (100%) |
| Full sequon fixed | 480/480 (100%) | 480/480 (100%) |

##### 2.4.3: Structurally resistant sites

Even with N fixed at 100%, positions 87 and 287 showed **0% sequon retention**:

**Position 87 (NGT)**:
- With N fixed: NGT → NGL (53%) or NGM (47%)
- The threonine is replaced with hydrophobic leucine or methionine

**Position 287 (NSS)**:
- With N fixed: NSS → NTT (81%), NTD (16%), NTS (3%)
- Both serines are replaced, primarily with threonine

##### 2.4.4: HA Functional sequon retention

| Position | WT | Exact | Functional | New Sequons Created |
|----------|-----|-------|------------|---------------------|
| 9 | NNS | 52.1% | **100%** | NSS (26), NDS (20) |
| 22 | NVT | 44.8% | 61.5% | NIT (16) |
| 87 | NGT | 0% | 0% | — |
| 153 | NGT | 36.5% | 36.5% | — |
| 287 | NSS | 0% | **22.9%** | NTT (22) |
| **Total** | | **26.7%** | **44.2%** | 84 |

**HA functional retention ratio**: 44.2% / 26.7% = **1.66×**

Position 9 shows **100% functional retention** despite only 52% exact match—every design has a valid sequon (NNS, NSS, or NDS).

Position 287 shows **0% exact but 23% functional**—NTT is a valid sequon created by MPNN.

#### Finding 2.5: Functional sequon retention (main dataset)

**Fate of 1,088 sequon-positions (34 sites × 32 designs):**

| Outcome | Count | Percentage |
|---------|-------|------------|
| Lost function (no valid sequon) | 1,026 | 94.3% |
| Retained function | 62 | **5.7%** |
| — Exact match preserved | 30 | 2.8% |
| — Shuffled to new valid sequon | 32 | 2.9% |

**Functional retention ratio**: 5.7% / 2.8% = **2.07×**

> See Figure S1: `supplementary_figures/fig_functional_retention.png`

> **Sample size caveat**: These percentages are based on 34 sequon sites across 15 proteins. The confidence intervals on these estimates are wide.

> Data: `supplementary_figures/functional_sequon_analysis.csv`

#### Finding 2.6: Substitution patterns at sequon-N

| When N is replaced, it becomes: | Frequency |
|---------------------------------|-----------|
| **D** (Aspartate) | 21.6% |
| **G** (Glycine) | 19.2% |
| **S** (Serine) | 14.1% |
| **T** (Threonine) | 7.0% |
| Other | 38.1% |
| **N retained** | 19.2% |

**S/T position**: Retained 47.9% of the time (as S or T).

#### Finding 2.7: Site-level heterogeneity

| Statistic | Value |
|-----------|-------|
| Sites with 0% retention | 27/34 (79%) |
| Sites with 1-5% retention | 6/34 (18%) |
| Sites with >5% retention | 1/34 (3%) |
| Maximum site retention | 84.4% (5jyj, single sequon) |

#### Finding 2.8: RSA-Stratified Analysis (Controlling for Solvent Accessibility)

This analysis addresses the main methodological limitation: whether the lower sequon-N retention is simply due to sequon sites being more surface-exposed.

##### 2.8.1: RSA Distribution of N Residues

**Main Dataset (15 proteins):**

| RSA Bin | Sequon-N | Non-Sequon-N |
|---------|----------|--------------|
| Buried (<20%) | 19 (43%) | 66 (51%) |
| Intermediate (20-50%) | 18 (41%) | 47 (36%) |
| Exposed (>50%) | 7 (16%) | 17 (13%) |

**Observation**: Sequon-N and non-sequon-N have **similar RSA distributions**. Sequon-N are NOT preferentially surface-exposed in this dataset.

##### 2.8.2: N Retention Within RSA Bins (Main Dataset)

| RSA Bin | Sequon-N | Non-Sequon-N | Difference |
|---------|----------|--------------|------------|
| Buried | 25.5% ± 35.2 (n=19) | 55.6% ± 44.9 (n=66) | **-30.1%** |
| Intermediate | 7.1% ± 21.2 (n=18) | 31.1% ± 36.9 (n=47) | **-24.0%** |
| Exposed | 37.9% ± 44.0 (n=7) | 25.0% ± 37.6 (n=17) | **+12.9%** |

**Key finding**: Even when controlling for solvent accessibility:
- In **buried** and **intermediate** positions, sequon-N is retained **less** than non-sequon-N
- In **exposed** positions, sequon-N is actually retained **more** than non-sequon-N

##### 2.8.3: HA Case Study RSA-Stratified Analysis

| RSA Bin | Sequon-N | Non-Sequon-N | Difference |
|---------|----------|--------------|------------|
| Buried | 64.2% ± 42.4 (n=9) | 49.1% ± 45.9 (n=60) | **+15.2%** |
| Intermediate | 38.1% ± 13.5 (n=5) | 23.1% ± 38.6 (n=36) | **+15.0%** |
| Exposed | 53.9% ± 27.3 (n=4) | 24.0% ± 12.6 (n=3) | **+29.9%** |

**Key finding**: In HA, sequon-N is retained **MORE** than non-sequon-N in **all RSA bins**. This is opposite to the main dataset pattern.

##### 2.8.4: Interpretation of RSA Analysis

The RSA-stratified analysis reveals:

1. **The structural confound does NOT explain the lower sequon-N retention** in the main dataset. Even comparing N residues at similar burial levels, sequon-N is retained less in buried and intermediate positions.

2. **The effect is protein-specific**: HA shows the opposite pattern (sequon-N favored) compared to the main dataset (sequon-N disfavored).

3. **Surface exposure actually favors sequon-N**: In exposed positions, sequon-N retention exceeds non-sequon-N in both datasets.

> **Data**: `solvent_accessibility_analysis/rsa_stratified_summary.csv`
> **Figures**: `solvent_accessibility_analysis/fig_main_retention_comparison.png`, `fig_ha_retention_comparison.png`

---

## Exploratory Analysis: Structural Features (Glycosylated Only)

> **Note**: This analysis examined only the 34 sequon sites in glycosylated proteins.

### Middle Position (X) Residue Effect

| Middle AA | Count | Mean Retention |
|-----------|-------|----------------|
| **G (Glycine)** | 5 | **17.5%** |
| **I (Isoleucine)** | 3 | 1.0% |
| **S (Serine)** | 4 | 0.8% |
| All others | 22 | ~0% |

### B-factor Analysis

B-factor showed **no correlation** with sequon retention (r = -0.030, n=19 sites).

> **Note**: An earlier analysis on mixed (glycosylated + non-glycosylated) data showed r = 0.272. The disappearance of this correlation when restricted to glycosylated proteins suggests the earlier signal was driven by dataset composition.

### Positional Analysis

Position in chain showed no clear pattern (sample sizes too small for reliable conclusions).

---

## Limitations

### 1. Small Sample Size
- Only 15 glycosylated proteins with successful designs
- Results dominated by 1-2 proteins (5gv0 accounts for 80% of retained sequons)
- Insufficient power for robust statistical inference

### 2. Structural Confound (Now Addressed)
- **Original concern**: Sequon-N residues might be in more surface-exposed positions, and surface residues are inherently less conserved
- **RSA-stratified analysis** (Finding 2.8) shows:
  - Sequon-N and non-sequon-N have **similar RSA distributions** (not preferentially surface-exposed)
  - **Within RSA bins**, sequon-N is still retained less in buried/intermediate positions
  - The structural confound does **not explain** the lower sequon-N retention

### 3. High Heterogeneity
- Per-protein sequon retention ranges from 0% to 97%
- Aggregate statistics may not represent typical protein behavior
- Individual proteins may have idiosyncratic patterns

### 4. Dataset Composition
- Glycosylated and non-glycosylated datasets not matched for confounders
- Question 1 results may reflect compositional differences rather than glycosylation effects

### 5. HA Case Study Generalizability
- HA is a single viral protein with unique structure (trimer of heterodimers)
- Position 9 (NNS) is exceptionally well-retained—unusual among all sequons
- May not represent typical glycoprotein behavior

---

## Interpretation

### Does MPNN "destroy" glycosylation sites?

The data show that MPNN does not specifically protect sequons—sequon-N is retained at lower rates (20%) than non-sequon-N (43%), and observed sequon retention (2.5%) is below null expectation (3.3%).

However, the **functional retention analysis** provides important nuance:

1. **Functional retention is ~2× exact match retention** (5.7% vs 2.8% in main dataset; 44% vs 27% in HA)
2. **MPNN creates new valid sequons about as often as it preserves original ones**
3. **Some sites show 100% functional retention** (HA position 9) despite <100% exact match

This suggests MPNN is **indifferent to sequon identity rather than actively hostile**. The high "destruction" rate likely reflects that:
- Sequon sites are surface-exposed, and surface residues are generally less conserved
- MPNN optimizes for structural fit, not glycosylation biology
- When MPNN changes a sequon, it sometimes creates a different valid sequon by chance

### Substitution patterns support "indifference" interpretation

When N is replaced, the top substitutions (D, G, S) are all **small, hydrophilic residues**:
- **Aspartate (D)** is structurally nearly identical to asparagine
- **Glycine (G)** maximizes backbone flexibility
- **Serine (S)** is small and polar

MPNN wants something small and hydrophilic at these surface positions—it's not anti-asparagine, it's just not pro-asparagine either.

### The structural confound (resolved)

We initially hypothesized that lower sequon-N retention could simply reflect surface exposure—glycosylation sites are typically in surface-exposed loops, and surface residues are inherently less conserved.

The RSA-stratified analysis (Finding 2.8) **rules out this explanation**:
- Sequon-N and non-sequon-N have **similar RSA distributions** in our dataset
- **Within each RSA bin**, sequon-N shows different retention patterns than non-sequon-N
- In buried/intermediate positions: sequon-N retained **less** (main dataset)
- In exposed positions: sequon-N retained **more** (both datasets)

This suggests MPNN's treatment of sequon-N is **context-dependent**—not simply a function of surface exposure.

### Protein-specific effects

The striking difference between the main dataset (sequon-N disfavored) and HA (sequon-N favored in all bins) suggests that local structural context, not just the presence of a sequon motif, determines retention. Some proteins may have sequons in structurally favorable contexts while others do not.

### Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| Does MPNN redesign glycosylated proteins more aggressively? | No systematic difference detected | MODERATE |
| Does MPNN protect sequons? | No evidence of specific protection | MODERATE |
| Does MPNN actively destroy sequons? | Not clearly—functional retention is 2× exact match | MODERATE |

**Bottom line**: MPNN appears **indifferent** to glycosylation sequons. It doesn't protect them, but it also doesn't specifically avoid creating them. The ~95% "destruction" rate reflects surface exposure and structural preferences, not anti-glycosylation bias.

---

## Implications for Glycosylation-Aware Protein Design

**For applications requiring preserved glycosylation**:

1. **Fixing N positions** alone increases retention from 26.7% → 37.7% (HA case study)
2. **Fixing full sequons** guarantees 100% retention but may compromise other design objectives
3. **Consider glycosylation-aware fine-tuning** of MPNN to explicitly preserve functional sequons

A glycosylation-aware model may be beneficial for applications where glycan attachment is critical, but the evidence does not suggest MPNN has an inherent anti-glycosylation bias that must be corrected.

> **[CAVEAT]**
> These implications are based on a limited dataset (15 proteins) with high heterogeneity. Larger, better-controlled studies are needed.

---

## Future Work

1. **Expand the dataset**: More glycosylated proteins with diverse structures
2. **Per-site structural analysis**: Characterize why some sites (5gv0 NGT, HA NNS) show exceptional retention
3. **Validate with experimental data**: Test whether designs with preserved sequons maintain glycosylation when expressed
4. **AlphaFold validation**: Assess structural quality of designs with preserved vs disrupted sequons
5. **Investigate protein-specific effects**: Understand why HA shows opposite pattern (sequon-N favored) vs main dataset

---

## Data Availability

- **Corrected sequon analysis** (glycosylated only): `corrected_sequon_analysis.csv`
- **Structural features** (glycosylated only): `structural_features_glycosylated_only.csv`
- **Functional retention data**: `supplementary_figures/functional_sequon_analysis.csv`
- **RSA-stratified analysis**: `solvent_accessibility_analysis/`
  - `rsa_stratified_summary.csv` - Summary of RSA-binned retention comparison
  - `main_dataset_n_rsa.csv` - Per-position RSA data for main dataset
  - `main_dataset_retention.csv` - Per-position retention data for main dataset
  - `ha_n_rsa.csv` - HA RSA data
  - `ha_retention.csv` - HA retention data
- HA case study: `results_HA_case_study/`
- Glycosylated results: `results_full_gly/`
- Non-glycosylated results: `results_full_non_gly/`

---

## Figures

### Main Figures (HA Case Study)
1. **Figure 1**: Amino acid retention comparison between glycosylated and non-glycosylated proteins
2. **Figure 2**: Sequon-relevant residue (N, S, T) retention comparison
3. **Figure 3**: HA case study - sequon retention by position under different constraints
4. **Figure 4**: HA case study - overall retention summary
5. **Figure 5**: HA case study - amino acid substitutions at structurally resistant positions

Location: `results_HA_case_study/figures/`

### Supplementary Figures

6. **Figure S1**: Functional sequon retention analysis
   - Left panel: Stacked bar showing fate of sequon-positions (destroyed vs shuffled vs exact match)
   - Right panel: Breakdown of functional retention showing ~50/50 split between exact matches and new valid sequons

Location: `supplementary_figures/`
- `fig_functional_retention.png` (Figure S1)
- `functional_sequon_analysis.csv` (site-level data)

7. **Figure S2a**: N retention analysis (main dataset)
   - Panel A: Overall N retention (sequon-N vs non-sequon-N)
   - Panel B: N retention by RSA bin (controlling for solvent accessibility)

8. **Figure S2b**: Sequon retention analysis (main dataset)
   - Panel A: Overall sequon retention (exact N vs functional sequon)
   - Panel B: Sequon retention by RSA bin

9. **Figure S3a**: N retention analysis (HA case study)
   - Same layout as Figure S2a, showing opposite pattern (sequon-N favored)

10. **Figure S3b**: Sequon retention analysis (HA case study)
    - Same layout as Figure S2b

Location: `solvent_accessibility_analysis/`
- `fig_main_n_retention.png` (Figure S2a)
- `fig_main_sequon_retention.png` (Figure S2b)
- `fig_ha_n_retention.png` (Figure S3a)
- `fig_ha_sequon_retention.png` (Figure S3b)
