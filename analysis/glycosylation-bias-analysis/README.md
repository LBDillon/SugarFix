# Glycosylation bias analysis

Modular inverse-folding design pipeline for measuring how different models handle N-X-S/T glycosylation sequons. The package wraps each model (ProteinMPNN, ESM-IF, TriFlow, Caliby, …) behind a common `DesignModel` interface so we can compare sequon retention across models from a single notebook.

## Layout

```
src/glyco_design/
  base.py          DesignModel ABC + DesignResult dataclass
  io.py            manifest loading, PDB lookup, FASTA writer
  sequon.py        N-X-S/T classifier + PDB→model position mapping
  pipeline.py      run_unconstrained_experiment(model, manifest, ...)
  models/
    proteinmpnn.py   ColabDesign wrapper
    esm_if.py        facebookresearch/esm inverse-folding wrapper
    triflow.py       jzhoulab/TriFlow wrapper (calls aa_sample directly)
    caliby.py        [TODO] ProteinDesignLab/caliby wrapper

notebooks/
  unconstrained_design.ipynb    Colab-ready: setup + one section per model

data/glyco_benchmark/
  manifests/       benchmark CSVs (pdb_id, chain_id, glycosite_positions, …)
  raw/             glycoproteins/*.pdb + controls/*.pdb
  designs/         written by pipeline: <model>/<pdb>_<chain>_unconstrained.fasta

experiments/       downstream analysis (04–09: retention, scores, figures)
case_studies/      per-protein AF3 validation
```

## Running a model

```python
from glyco_design.models.proteinmpnn import ProteinMPNNDesignModel
from glyco_design.pipeline import run_unconstrained_experiment

model = ProteinMPNNDesignModel()
model.load()

df = run_unconstrained_experiment(
    model=model,
    manifest_path='data/glyco_benchmark/manifests/benchmark_manifest_simple.csv',
    pdb_root='data/glyco_benchmark/raw',
    output_dir='data/glyco_benchmark/designs/proteinmpnn',
    num_seqs=32,
    temperature=0.1,
    protein_class='glycoprotein',
)
```

The returned DataFrame has one row per (design, glycosite) with columns:
`model, pdb_id, chain_id, condition, design_idx, glycosite_pdb, glycosite_model, sequon_status, design_score, seqid`.

## Adding a new model

1. Write `src/glyco_design/models/<model>.py` subclassing `DesignModel` — implement `load()` and `generate(pdb_path, chain, num_seqs, temperature, fix_pos=None)` returning a `DesignResult`.
2. Add a section in `notebooks/unconstrained_design.ipynb` that instantiates the adapter and calls `run_unconstrained_experiment`.

That's it — nothing downstream needs to change as long as FASTAs and the retention CSV match the shared format.

## Colab usage

`notebooks/unconstrained_design.ipynb` is the primary entrypoint. It clones this repo, `pip install -e`s the package, and runs a section per model with heavy deps installed lazily so you only pay the install cost for models you actually run.

## Data

Manifests and PDBs under `data/glyco_benchmark/` were lifted from the Bias_Paper project. The canonical manifest for this analysis is `benchmark_manifest_simple.csv` (18 glycoproteins + 16 controls; 32 unique PDB IDs). `expanded_manifest_validated.csv` has an additional 22 proteins.
