# What the SugarFix Wrapper Does

ProteinMPNN is a tool that redesigns protein sequences to fold into a given backbone shape. It's very good at this, but it has a blind spot: it tends to mutates away N-linked glycosylation sites for glycoproteins where they are required for fold and function. These sites follow a short sequence pattern called a sequon (N-X-S/T, where X is any amino acid except proline). Across 84 glycoproteins and over 700 sites, unconstrained ProteinMPNN designs retain only about 4% of the original sequons. This matters because glycosylation is essential for the folding, stability, and function of many proteins. If you redesign a glycoprotein with ProteinMPNN and don't account for this, you will have no indication in the pipeline if you end up with a protein that can't be glycosylated properly without knowing to check.

## Why it may happen

The bias is likely driven by the sequence motif itself, not by the structural context of actual glycosylation. ProteinMPNN avoids the N-X-S/T pattern equally at experimentally validated glycosylation sites and at positions that just happen to match the motif by coincidence (25.0% vs 25.6% asparagine retention; no significant difference, p=0.52). Compare this to non-sequon asparagines, which are retained about 56% of the time. The model has learned that the N-X-S/T pattern is somehow undesirable for protein designs, and applies this preference everywhere.

## What SugarFix does

SugarFix is a glycosylation-aware wrapper around ProteinMPNN. The walkthrough notebook implements its "Preserve" mode: it detects glycosylation sites, lets you decide how strictly to protect each one, runs ProteinMPNN with those constraints, scores the resulting designs, and exports them for structural validation with AlphaFold 3.

## What each step does

### Step 1 — Prepare the structure

The notebook downloads the PDB file for your target protein (or uses one you already have), strips out everything except the protein chains (removing water, ligands, and glycan sugars), and checks for issues like missing residues. It also extracts any resolved glycan trees from the structure, which provide strong evidence that a particular site is infact glycosylated.ProteinMPNN needs a clean protein-only PDB as input. If a glycan was resolved in the crystal, that site is almost certainly a real glycosylation site, not just a coincidental sequence match which we may not want to preserve. 

### Step 2 — Detect sequons and audit evidence

Not all sequons are equally important to protect. An experimentally validated glycosylation site on a therapeutic antibody is worth strict protection. A motif-only match in an internal loop might not need any special treatment. The evidence tiers let you make informed decisions rather than treating every N-X-S/T the same.

The notebook scans the protein sequence for N-X-S/T motifs, then cross-references each one against three sources of evidence:

- **UniProt annotations** — does the UniProt entry for this protein list this position as glycosylated, and with what confidence? Some annotations are backed by published experiments (strongest evidence), while others are inferred by curators from sequence similarity (weaker).
- **PDB glycan structures** — is there a sugar molecule covalently bonded to this asparagine in the crystal structure?
- **Motif matching** — the site matches the N-X-S/T pattern, but there's no external evidence it's actually glycosylated.

Each site gets assigned an evidence tier: **Experimental** (published proof), **PDB evidence** (structural proof), **Curator inferred** (bioinformatic inference), or **Motif only** (sequence pattern match with no validation).

### Step 3 — Choose a preservation strategy

The notebook presents three choices:

1. **Target role** — Is this protein a glycoprotein that needs its glycans, or are you unsure, or do you know it's not glycosylated? This sets the overall framing.

2. **Strategy** — How strictly should sites be protected during redesign?
   - *Evidence-aware defaults* — each site gets a policy matched to its evidence tier. Experimentally validated sites get full protection; motif-only sites get lighter treatment.
   - *Full sequon everywhere* — fix all three positions (N, X, and S/T) at every site. Maximum preservation but most restrictive for the design algorithm.
   - *Functional preserve* — fix the asparagine and the serine/threonine but let the middle position vary. Keeps the site functional but gives ProteinMPNN more freedom.
   - *Soft filter* — don't fix anything during design, but check afterward whether a functional sequon survived. This lets ProteinMPNN design freely and then filters for designs that happened to retain the sites.
   - *Mixed site-by-site* — set a different policy for each individual site.

3. **Baseline comparisons** — optionally run the full-sequon, functional-preserve, and soft-filter strategies as baselines alongside your chosen plan, so you can see how your custom strategy compares.

There's a trade-off between sequon protection and design freedom. Fixing more positions guarantees glycosylation sites survive but constrains ProteinMPNN more heavily, which may reduce the design score. The right balance depends on your specific protein and what you plan to do with the design.

### Step 4 — Run ProteinMPNN

For each design condition (your chosen plan plus any baselines), the notebook:

1. Translates the site-level policies into a set of fixed positions for ProteinMPNN
2. Runs ProteinMPNN with those constraints
3. Collects the designed sequences

This produces multiple candidate sequences per condition. The default is 8 sequences at a sampling temperature of 0.1, but you can adjust both parameters.

### Step 5 — Score designs against your plan

Each designed sequence is evaluated at every glycosylation site:

- Did the exact wild-type triplet survive? (Exact match)
- Is there still a functional N-X-S/T sequon? (Functional retention)
- Does the site meet the policy you selected? (Plan satisfaction)

Designs are ranked by plan satisfaction first (how many required sites passed their policy), then by ProteinMPNN score (lower is better — it means the model is more confident the sequence will fold correctly). The top design from each condition is selected for AF3 validation.

The notebook also produces a dashboard figure showing how each condition performed: retention rates, plan satisfaction, and a heatmap of which sites passed or failed in each top design.

### Step 6 — Export for AF3 validation and save everything

The top design from each condition is exported as an AlphaFold 3 Server input JSON in two versions:

- **Plain** — just the designed protein sequence, for predicting whether the new sequence folds correctly.
- **With glycans** — the protein sequence plus NAG (N-acetylglucosamine) stubs at every site that retained a functional sequon, for predicting whether glycans would be compatible with the designed structure.

All intermediate data (site inventories, evidence audits, decision tables, retention scores, condition summaries) are saved as CSVs, and the full session configuration is saved as a JSON file so you can reproduce or review the run later.
