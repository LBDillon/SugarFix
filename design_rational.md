# SugarFix: Design Rationale

*A glycosylation-aware wrapper for deep-learning protein design pipelines*

ProteinMPNN designs sequences by optimising amino acid identity for a given backbone, but it has no explicit representation of glycans. Trained exclusively on backbone atom coordinates from PDB structures, where glycans are routinely absent or stripped, the model treats the N-X-S/T N-linked glycosylation sequon as ordinary residues available for optimisation. Preliminary analysis across a set of glycoproteins suggests that unconstrained ProteinMPNN designs retain only a small fraction of functional N-linked glycosylation sequons, and that the model appears to actively disfavour the sequon motif relative to asparagine in other sequence contexts. When these designs are passed to AlphaFold 3, which is commonly used alongside ProteinMPNN to check whether designed sequences are predicted to adopt the target fold, the glycan-stripped sequences are not flagged as misfolded or otherwise deleterious, and would not be filtered out on the basis of standard structural metrics. This creates a silent failure mode. A designed glycoprotein can pass computational quality checks while lacking glycosylation sites that may be required for correct folding, stability, receptor engagement, or pharmacokinetic behaviour, depending on the protein. 

The proposed tool SugarFix wraps around the inverse folding step. It would operate both before design (analysing structures, generating glycosylation-aware constraints) and after design (checking retention, filtering, exporting) to increase the ability for glycosylation handling at the point in the pipeline where it is currently absent.

## 1\. Why glycosylation matters for designed proteins

### 1.1 Glycosylation as a folding quality control signal

N-linked glycosylation occurs at asparagine residues within Asn-X-Ser/Thr sequons (where X is any amino acid except proline). THe first reason why glycosylation in protein designs matters is (talk about why the basic need of being able to accurately design glycoproteins regardless of any additional quality control. Then there are several other ways in which glycosylation will affect the folding process that we would likely want to account for when designing. 

Glycosylation is one of several co-translational quality control mechanisms in eukaryotic cells, but a particularly consequential one because it couples protein folding surveillance to a covalent chemical modification (the glycosylation) that also affects stability, trafficking, and function.

**\[Figure 1 — ER quality control pathway\]** Oligosaccharyltransferase (OST) transfers a 14-sugar precursor (Glc₃Man₉GlcNAc₂) onto the asparagine side chain. Sequential glucose trimming (by what? OST?) produces intermediates that are read by lectin chaperones calnexin and calreticulin as a folding progress signal. UGGT acts as a conformational sensor, re-glucosylating incompletely folded glycoproteins to re-enter the cycle. Terminally misfolded proteins are targeted to endoplasmic-reticulum-associated protein degradation (ERAD) via mannose trimming. (Varki et al., *Essentials of Glycobiology*, 4th ed., 2022; Hebert & Bhatt, PMC4805476.)


**\[Sidebar — Thermodynamic mechanism\]** Shental-Bechor & Levy (2008) showed via coarse-grained simulations that glycosylation stabilises folded proteins primarily by destabilising the unfolded state: bulky glycan chains force the unfolded ensemble into extended conformations, raising its free energy. The effect is enthalpic, proportional to the number of glycan chains, and largely independent of glycan size beyond the core trisaccharide (Hanson et al., 2009). Petrescu et al. (2004) established the structural basis, surveying 2,592 sequons across 506 glycoproteins and finding that glycosylation alters asparagine torsion angle distributions, reduces flexibility, and shields hydrophobic surface patches. A 2024 study in *Science Advances* showed that eukaryotic secretory proteins have evolved to flank their aggregation-prone regions with N-glycans, establishing glycosylation as a systematic anti-aggregation strategy.

For protein design, if a design disrupts glycosylation sites on a protein that requires them, the protein may misfold, aggregate, or be degraded by ER quality control. The standard metrics used to evaluate designed proteins include backbone RMSD against a reference structure, per-residue confidence (pLDDT), predicted alignment error (PAE), and template modelling scores (pTM, ipTM). All measure geometric and confidence properties of the predicted fold. None encodes whether the designed sequence would be recognised by OST, enter the calnexin cycle, or avoid ERAD. These metrics evaluate structural plausibility, not biological viability in a glycosylation-competent expression system, which means a design can score well on every available computational metric while being fundamentally incompatible with the cellular machinery that processes it.

### 1.2 Four ways glycosylation matters for protein design

**\[Figure 2 — Four-panel figure: why glycosylation matters for designed proteins\]**

**Panel B: Glycan quantity determines dosing regimen (EPO).** EPO is \~40% carbohydrate by weight. Terminal sialic acid on N-glycans masks galactose that would otherwise trigger hepatic clearance. Darbepoetin alfa, engineered with two additional N-glycosylation sites, achieves a three-fold longer half-life, enabling weekly rather than thrice-weekly dosing (Egrie et al., 2003).

**Panel C: Glycan presence determines whether the drug works (IFN-β).** IFN-β-1a (glycosylated) is 10-fold more active than IFN-β-1b (non-glycosylated) in antiviral assays. Runkel et al. (1998) showed that of all structural differences between the two forms, only glycosylation affected activity. The non-glycosylated form aggregates and is more immunogenic.

**Panel D: Glycan shielding determines immune visibility (viral glycoproteins).** SARS-CoV-2 Spike encodes 22 N-linked glycan sequons per protomer; simulations estimate glycans shield \~40% of the protein surface from antibody recognition (Grant et al., 2020). HIV-1 gp120 is \>50% carbohydrate by weight. H3N2 influenza has accumulated glycosylation sites on the HA head domain over \~50 years as a direct immune-evasion mechanism. For vaccine design, glycosylation awareness is a functional requirement.

## 2\. SugarFix Context

### 2.1 Prediction, annotation, and database tools 

The glycoinformatics landscape is well-developed, but disconnected from the design pipeline. Here I outline three distinct categories of tools.

**Glycosylation site predictors** 

Sequence-context predictors, using local amino acid patterns around candidate sites:  
NetNGlyc, Gupta & Brunak 2002;   
NetOGlyc 4.0, Steentoft et al. 2013 

Structure-aware predictors incorporate AlphaFold-derived structural features alongside sequence: DeepMVP, *Nature Methods* 2025; Ertelt et al. 2024, 

Protein language model approaches:embed PTM tokens directly into language model vocabularies, representing modifications as part of the sequence rather than as post-hoc annotations.  
PTM-Mamba, *Nature Methods* 2025;   
PTMGPT2, *Nature Communications* 2024\. 

They predict whether a site will *be* glycosylated, not whether a designed sequence has preserved or disrupted an existing site.

**Glycosylation databases**   
provide curated, experimentally grounded annotation. The GlySpace Alliance is a group of Glycosylation knowledge bases:

GlyGen (Georgetown/UGA/GWU; Kahsay et al. 2020). GlyGen integrates data from UniProtKB, GlyTouCan, UniCarbKB, PDB, and other sources into a knowledgebase with RESTful API and SPARQL access, keyed by UniProtKB accessions. 

Glycomics@ExPASy/GlyConnect (SIB; Alocci et al. 2019), GlyConnect curates glycoprotein–glycosite–glycan structure relationships. 

GlyCosmos (Soka University) uniquely includes glycoprotein data from IGOT-MS and lectin microarray experiments.

SugarFix could query GlyGen's API per protein accession to retrieve site-specific glycosylation evidence, glycan structures, and references. Would strengthen evidence tier classification.

**Glycan structural modelling tools**

GlycoShape/Re-Glyco (Ives et al. 2024\) grafts glycan conformer libraries onto protein structures and predicts N-glycosylation occupancy with 93% agreement with experiment. could provide glycan conformers for AF3 input preparation?

GlycoSHIELD (*Cell*, 2024\) models glycan shielding effects. could assess shielding coverage of preserved glycosites.

None of these tools sits within the protein design pipeline. They operate upstream (predicting sites), in parallel (cataloguing data), or downstream (modelling glycan structures on solved proteins). (make into a figure) 

None can enforce or restore glycosylation in a designed sequence.

### 2.2 

Ertelt et al. (2024) used PTM prediction with Rosetta to design proteins that maximise or minimise predicted glycosylation probability at specific sites (did glycoengineering of influenza HA for vaccine design). The key distinction from SugarFix is that their approach designs *new* glycosylation sites rather than checking or restoring *existing* ones that have been disrupted by an external design tool. Rosetta's GlycanTreeModeler and CreateGlycanSequonMover (Gray Lab; *PLoS Comput Biol*, 2024\) similarly operate within the Rosetta design loop and are not designed to interface with deep-learning inverse folding models.

Glycans in the PDB are typically partially resolved, highly dynamic, and absent from most design workflows where backbones are generated de novo.

### 2.4 Where does SugarFix fit?

The standard pipeline of backbone generation → inverse folding → structure prediction is currently glycosylation-agnostic at each step. The prediction tools, databases, and Rosetta modules described above are complementary to what we are proposing with a wrapper: predictors could feed SugarFix's evidence classification, databases for enriching the annotation layer, and glycan modelling tools could improve its AF3 export step. Integration is the key contribution in the inverse folding design loop. 

| Category | Examples | Relationship to SugarFix |
| :---- | :---- | :---- |
| Glycosite predictors | NetNGlyc, DeepMVP | Could supplement evidence tier annotation |
| Glycoinformatics DBs | GlyGen, GlyConnect | Could replace bespoke UniProt parsing for evidence data |
| Glycan modelling | GlycoShape, GlycoSHIELD | Could improve AF3 glycan input preparation |
| Rosetta glycan design | Ertelt et al., GlycanTreeModeler | Operates in different design paradigm |
| DL inverse folding | ProteinMPNN | The models SugarFix wraps around |
| Structure prediction | AlphaFold 3 | Predicts fold geometry; does not assess whether a designed sequence would be glycosylated by the cell of course |

## 3\. ProteinMPNN and AlphaFold 3 limitations specific to glycosylation

### 3.1 Glycan-free representation

ProteinMPNN (Dauparas et al., *Science*, 2022\) was trained on PDB structures deposited before August 2021, clustered at 30% sequence identity into 25,361 clusters. Its input representation consists exclusively of distances between five backbone atoms (N, Cα, C, O, virtual Cβ) for each residue in a k-nearest-neighbour graph of 48 Cα neighbours. 

The key consideration here is that all non-protein atoms (ligands, cofactors, glycans, waters, metal ions) are stripped during structure-to-graph conversion, and model outputs only the standard 20 amino acids. Therefore the \~4% of PDB crystallographic structures containing glycan information (Jo & Lee, *Sci Rep*, 2015), are likely invisible to ProteinMPNN's input pipeline. The model learned to design proteins in a representation where glycosylation does not factor.

### 3.2 Sequon preservation 

- Unconstrained designs retain only a small fraction of functional sequons, with the majority of proteins showing less than 25% retention.  
- Asparagine within sequon motifs is retained at a significantly lower rate than asparagine outside sequons, suggesting the model disfavours the N-X-S/T motif specifically rather than asparagine in general.  
- The bias appears motif-driven rather than glycosylation-driven: retention rates are similar for experimentally validated glycosylation sites and motif-only sites, consistent with a pattern-level effect rather than a response to glycan-related structural features.  
- Fixing only the asparagine is insufficient: flanking mutations still destroy roughly half of sequon motifs in preliminary data. A "functional preserve" approach  (fixing N and enforcing X \!= Pro while allowing S/T variation) is likely more useful than N-only constraint.  
- Full triplet constraint (N, X, and S/T all fixed) achieves complete retention by construction, with no detectable cost to overall design quality as measured by ProteinMPNN scores.

These findings are from a limited analysis and the specific numbers should be treated as indicative rather than definitive. However, the direction of the effect showing systematic sequon loss under unconstrained design is consistent across the dataset.

### 3.3 Structure prediction does not assess glycosylation competence

AlphaFold 3 excels at its trained objective: predicting 3D structure from sequence and molecular component inputs. When designs with destroyed glycosylation sequons are passed to AF3, the predicted structures show no significant deviation from designs that preserve sequons — most fall within 1–2 Å RMSD of the crystal structure. AF3's metrics (pLDDT, PAE, pTM, ipTM) report high confidence because the backbone geometry is sound.

The problem is not that AF3 performs poorly, but that high structural confidence is commonly interpreted as evidence that a design is "good" — and this interpretation does not hold for glycosylation. AF3 was trained to predict molecular structure, not to assess whether a protein would be recognised by oligosaccharyltransferase, enter the calnexin/calreticulin folding cycle, avoid ER-associated degradation, or be correctly processed through the Golgi. These are cellular processes that depend on sequence motifs, enzyme specificity, and expression system context — none of which are part of AF3's training objective or input representation. A design that looks structurally identical to the wild type but lacks N-X-S/T sequons at critical positions will score identically on all AF3 metrics while being incompatible with the glycosylation machinery.

The most comprehensive assessment of AF3's glycan-specific capabilities (Huang, Kannan & Moremen, *Glycobiology*, 2025\) reinforces this point: even when glycans are explicitly provided as input, only about half of single-glycan structures are correctly predicted, and AF3's scoring metrics fail to capture glycan-specific conformational errors. Glycans with incorrect linkages or stereochemistry can still receive high confidence scores.

### 3.4 Stress tests: two proteins where glycosylation is central

To illustrate the compound failure, we ran two proteins where glycosylation is central to fold or function through the standard ProteinMPNN → AF3 pipeline.

**CD2 — glycosylation stabilises the fold itself.**

The adhesion domain of human CD2 is a small protein with a single glycosylation site at position 65\. The glycan stabilises the fold by several kilocalories per mole — enough that removing it causes the domain to substantially unfold (Hanson et al., 2009; Wyss et al., 1995; Recny et al., 1992). The first GlcNAc accounts for roughly two thirds of native-state stabilisation and the entire acceleration of folding, packing directly against a phenylalanine and a threonine that flank the glycosylation site (Wyss et al., 1995). These residues, together with the glycan, form an enhanced aromatic sequon. The glycan also masks a cluster of positive charges centred on a nearby lysine that would otherwise destabilise the local structure.

There are homologs where the glycosylation site is absent but fold and function are maintained. In rat CD2, compensatory substitutions (K61E and F63L) neutralise the local energetic penalty that the glycan normally alleviates. ProteinMPNN, redesigning human CD2 without constraints, retains a functional glycosylation sequon in approximately one design out of sixty-four. It removes the asparagine at position 65 in nearly every case, replacing it with aspartate. No design analysed contained residues at the equivalent positions that would compensate in the way the rat ortholog does. The model converges on removing the glycosylation site and changing the surrounding residues, producing a sequence that has lost the glycan without designing the structural compensation that would make glycan-independence viable. When passed to AlphaFold 3, these designs are predicted to fold to within approximately 1.4 Å of the crystal structure — despite this being a case where the designs almost certainly cannot fold stably.

The biophysical problem that the glycan solves — a destabilising charge cluster masked by a carbohydrate — is not legible from backbone coordinates alone, highlighting the gap between what ProteinMPNN can access in its input representation and the level of glycosylation awareness needed for proteins like CD2.

**IgG1 Fc — glycosylation determines function without altering fold.**

The IgG1 Fc glycan at position 297 sits between two CH2 domains, where it stabilises a loop conformation and holds the domains in the open arrangement required for binding immune receptors. If the glycan is removed, the Fc fragment still folds into a biophysically viable structure, but the domains drift closer together and receptor engagement is lost (Subedi and Barb, 2015).

When running the open-conformation glycosylated crystal structure through ProteinMPNN, the Asn297 sequon is removed in roughly two thirds of designs. If the model were implicitly preserving a glycan-shaped cavity — recognising that this surface patch differs from an ordinary solvent-exposed surface even without explicit glycan information — one would expect the residues that directly contact the glycan in the crystal structure to be retained at a higher rate than generic surface positions. They are not: glycan-contacting residues are redesigned at the same rate as non-contact surface controls. The contact network that would hold the glycan between the two domains is treated as a typical solvent-exposed surface.

This means that editing the sequon back into a ProteinMPNN design would probably not restore the glycan's structural environment, because the surrounding residues have been redesigned for a context in which the glycan does not exist. AlphaFold 3 predicts these designs to fold within approximately one angstrom of the reference structure. The pipeline presents a well-folded Fc fragment with no indication that the glycan-supported domain arrangement required for receptor engagement is absent. If proceeding to experimental validation on this basis, the problem would surface only at the functional assay, after the protein had been produced.

**\[Figure 3 — Stress test panel\]** Left: CD2 unconstrained design structure prediction (coloured) overlaid on crystal structure (grey), highlighting the glycosite region. Right: IgG1 Fc unconstrained design showing the conformational shift at the inter-domain glycan site compared to the glycosylated crystal structure.

---

## 4\. What SugarFix does

SugarFix is a glycosylation-aware wrapper for the inverse folding design step. It operates in two phases: **pre-design** (analysing the input structure, detecting and classifying glycosylation sites, generating constraints) and **post-design** (checking sequon retention in output sequences, filtering, ranking, and exporting AF3-ready files). It has two modes.

### Mode 1: Preserve

**Goal:** Keep glycosylation sites intact in designed sequences so the protein can be correctly glycosylated when expressed in a glycosylation-competent cell.

The pipeline:

1. **Detect** N-X-S/T sequons in the input structure.  
2. **Classify** each site by evidence strength: experimental (UniProt ECO evidence) \> PDB evidence (glycan resolved in structure) \> curator-inferred \> motif-only.  
3. **Constrain** ProteinMPNN design: fix the appropriate residues at validated sites according to the user's chosen policy.  
4. **Filter** output sequences by sequon retention.  
5. **Export** AF3-ready JSON files with glycan specifications at preserved sites.

The user controls the trade-off between glycosylation preservation and design freedom:

| Policy | What is fixed | Use case |
| :---- | :---- | :---- |
| Full sequon | N, X, and S/T all fixed | Wild-type-like; minimal change near glycosites |
| Functional preserve | N fixed; X ≠ Pro enforced; S/T allowed to vary | Moderate design freedom while maintaining glycan competence |
| Soft filter | Nothing fixed; post-hoc selection for designs that retain sequons | Maximum freedom; glycan-awareness as a selection criterion |

Evidence-aware defaults: validated sites (experimental, PDB evidence) get full sequon fixing; curator-inferred sites get functional preserve; motif-only sites get soft filtering or are skipped. Users can override per site.

### Mode 2: Compensate (future)

**Goal:** When a glycan is intentionally removed (e.g., for bacterial expression), ensure the design does not leave behind an energetically unstable surface.

The primary strategy is homolog rescue: for each glycosylated site, retrieve homologous sequences that naturally lack glycosylation at that position, align, and identify substitutions that wild-type evolution uses to stabilise the glycan-free surface. The CD2/rat-CD2 example demonstrates this principle. Mode 2 is harder and more speculative; the homolog rescue approach is the tractable starting point.

---

## 5\. Architecture and MVP scope

*Architectural diagrams, data structures, configuration YAML, and pseudocode are provided in the accompanying reference files.*

### MVP delivers Mode 1 only

**Input:** PDB file (or PDB ID). **Output:** Top-N designed sequences with sequon preservation report \+ AF3-ready JSON files.

MVP includes: sequon detection, evidence classification (UniProt annotation or fallback to motif-only), constraint generation for ProteinMPNN at both full-sequon and functional-preserve levels, sequon retention analysis, AF3 server JSON export with NAG stubs.

MVP does not include: Mode 2 (Compensate), homolog retrieval, local energetics scoring, signal peptide handling, multi-model support, GUI.

**Test case:** 1J2E (erythropoietin receptor complex) — 2 chains, 18 sequon sites, mix of evidence tiers, full pipeline data already available for validation.

### Success criteria

- Detects all sequon sites correctly (validated against existing sequons.csv for 84 proteins).  
- Evidence tiers match existing annotations.  
- Full-sequon-fixed designs retain 100% of sequons (sanity check).  
- Unconstrained designs show \~4% retention (matches known baseline).  
- AF3 JSONs are valid and uploadable to AF3 Server.  
- Run completes in \<5 min on a laptop.

---

## 6\. Open questions

Several questions remain unresolved:

**How does the cell determine glycan branch composition?** After the initial 14-sugar precursor is transferred, glycan processing in the ER and Golgi produces diverse branch structures (high-mannose, complex, hybrid). The determinants are partially understood — local protein structure, accessibility to processing enzymes, cell-type-specific enzyme expression — but not well enough to predict branch composition for a designed protein. For the MVP, SugarFix does not attempt to specify branch composition beyond what is resolved in the input PDB structure.

**What is the retention rate under functional preserve constraints?** The functional preserve policy (fix N, enforce X ≠ Pro, allow S/T variation) has not been empirically tested. Measuring its retention rate across the existing dataset is a concrete next experiment that would determine whether this intermediate policy provides meaningful design freedom while maintaining glycan competence.

**How should heavily glycosylated proteins be handled?** For glycan shields (HIV gp120, influenza HA, SARS-CoV-2 Spike), preserving every site may be unnecessary if the biological role is collective shielding rather than site-specific. Allowing the user to specify a target preservation fraction or key sites is the likely solution.

**Can SugarFix generalise beyond ProteinMPNN?** The architecture is designed to be model-agnostic — any inverse folding model that accepts fixed-position constraints can slot in. ESM-IF is the obvious second target. RFdiffusion is a harder case because it generates backbones (upstream of sequence design), but backbone generation with glycosite-aware constraints is a natural extension.

---

## References

Abramson J et al. (2024) Accurate structure prediction of biomolecular interactions with AlphaFold 3\. *Nature* 630:493–500.

Alocci D et al. (2019) GlyConnect: glycoproteomics goes visual, interactive, and analytical. *J Proteome Res* 18:753–766.

Aricescu AR et al. (2007) Glycoprotein structural genomics: solving the glycosylation problem. *Structure* 15:1–6.

Bagdonas H et al. (2021) The case for post-predictional modifications in the AlphaFold Protein Structure Database. *Nat Struct Mol Biol* 28:869–870.

Dauparas J et al. (2022) Robust deep learning–based protein sequence design using ProteinMPNN. *Science* 378:49–56.

Dauparas J et al. (2025) Atomic context-conditioned protein sequence design using LigandMPNN. *Nature Methods* 22:717–723.

Egrie JC et al. (2003) Darbepoetin alfa has a longer circulating half-life and greater in vivo potency than recombinant human erythropoietin. *Exp Hematol* 31:290–299.

Ertelt M et al. (2024) Combining machine learning with structure-based protein design to predict and engineer post-translational modifications of proteins. *PLoS Comput Biol* 20:e1011939.

Hanson SR et al. (2009) The core trisaccharide of an N-linked glycoprotein intrinsically accelerates folding and enhances stability. *PNAS* 106:3131–3136.

Huang W, Kannan N, Moremen KW (2025) Modeling glycans with AlphaFold 3: capabilities, caveats, and limitations. *Glycobiology* 35:cwaf048.

Ives CM et al. (2024) Restoring protein glycosylation with GlycoShape. *Nature Methods* 21:2117–2127.

Kahsay R et al. (2020) GlyGen data model and processing workflow. *Bioinformatics* 36:3941–3943.

Petrescu AJ et al. (2004) Statistical analysis of the protein environment of N-glycosylation sites: implications for occupancy, structure, and folding. *Glycobiology* 14:103–114.

Recny MA et al. (1992) Structural and functional role of glycosylation in the adhesion molecule CD2. *J Biol Chem* 267:22428–22434.

Runkel L et al. (1998) Structural and functional differences between glycosylated and non-glycosylated forms of human interferon-β. *Pharm Res* 15:641–649.

Shental-Bechor D & Levy Y (2008) Effect of glycosylation on protein folding: a close look at thermodynamic stabilization. *PNAS* 105:8256–8261.

Shental-Bechor D & Levy Y (2009) Folding of glycoproteins: toward understanding the biophysics of the glycosylation code. *Curr Opin Struct Biol* 19:524–533.

Shields RL et al. (2002) Lack of fucose on human IgG1 N-linked oligosaccharide improves binding to human FcγRIII and antibody-dependent cellular toxicity. *J Biol Chem* 277:26733–26740.

Steentoft C et al. (2013) Precision mapping of the human O-GalNAc glycoproteome through SimpleCell technology. *EMBO J* 32:1478–1488.

Subedi GP & Barb AW (2015) The structural role of antibody N-glycosylation in receptor interactions. *Structure* 23:1573–1583.

Varki A et al. (2022) *Essentials of Glycobiology*, 4th ed. Cold Spring Harbor Laboratory Press.

Wang D et al. (2020) MusiteDeep: a deep-learning-based webserver for protein post-translational modification site prediction and visualization. *Nucleic Acids Res* 48:W140–W146.

Wyss DF et al. (1995) Conformation and function of the N-linked glycan in the adhesion domain of human CD2. *Science* 269:1273–1278.  

