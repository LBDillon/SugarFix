# SugarFix Tutorial Roadmap

The tutorial notebooks should remain researcher-facing wrappers around script
APIs. If a workflow needs branching, parsing, scoring, plotting, or export
schema knowledge, that logic belongs in `pipeline/` first.

## Tutorial 01 - Sequon Mapping & Validation

Status: executable in `notebooks/01_sequon_mapping_validation.py`.

Research question: given a glycoprotein structure, where are all N-X-S/T
sequons, and how confident are we that each is actually glycosylated?

Implemented outputs:

- Sequon inventory CSV.
- Evidence audit CSV.
- UniProt-to-PDB remapped annotation CSV.
- Numbering/conflict CSV.
- HTML confidence report.
- PyMOL annotation script.

Next script-level improvements:

- Add full UniProt canonical sequence to PDB SEQRES alignment rather than
  relying primarily on DBREF/mmCIF mappings.
- Add RCSB-wide deposited-structure search per accession.
- Add bibliography-ready citation extraction from UniProt evidence references.

## Tutorial 04 - Basic Glycoprotein Redesign

Status: executable in `notebooks/04_basic_glycoprotein_redesign.py`.

Research question: can we redesign a glycoprotein without disrupting existing
glycosylation sites?

Implemented outputs:

- Design FASTA from ProteinMPNN.
- Constraint and decision tables.
- Per-site sequon retention table.
- Condition summary and top-design table.
- Standalone AlphaFold 3 JSON by default.
- AlphaFold Server glycan-stub JSON when `AF3_EXPORT_MODE = "alphafoldserver"`
  or `"both"`.

## Tutorial 05 - Glycan Shield Analysis

Status: design needed before executable notebook.

Research question: for heavily glycosylated proteins, how do we handle high
sequon density and steric constraints between adjacent glycans?

Needed script capabilities:

- Pairwise sequon spacing in sequence and 3D.
- Attachment-cone or glycan-envelope clash model.
- Solvent-accessible glycan shield coverage estimate.
- Density-aware ProteinMPNN constraint policy.
- Before/after shield coverage comparison.

## Tutorial 06 - Multi-domain Glycoprotein Design

Status: design needed before executable notebook.

Research question: for proteins where glycans mediate inter-domain
interactions, how do we design without disrupting functional glycan contacts?

Needed script capabilities:

- Glycan-protein and glycan-mediated inter-domain contact extraction.
- Contact classification into direct H-bonds, packing, and spacer/linker roles.
- Automatic fixation of contact-forming residues near resolved glycans.
- AF3 validation metrics for preserved inter-domain geometry.

## Tier 3 - Advanced Glycoengineering

Tutorials 07-09 should wait until the core structural evidence and
sequon-preserving design workflows are stable.

Planned script capabilities:

- De novo sequon placement with local sequence edit enumeration.
- Glycan removal with stability/interaction compensation.
- Epitope masking by glycan placement and manufacturability checks.
