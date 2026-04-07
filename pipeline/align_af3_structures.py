"""Colab-friendly structural alignment for organized AF3 outputs.

This module mirrors the lightweight validation role of the PyMOL script, but
uses Biopython so it can run inside a notebook/Colab runtime. It performs
sequence-aware C-alpha matching before fitting, which avoids residue-numbering
offsets between the crystal/PDB reference and AF3 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Superimposer


AA3_TO_1 = {key.upper(): value for key, value in protein_letters_3to1_extended.items()}
AA3_TO_1["MSE"] = "M"


def _read_structure(path: Path, structure_id: str):
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, str(path))


def _ca_items_by_chain(structure) -> dict[str, list[tuple[str, object]]]:
    """Return chain -> [(one-letter residue, CA atom), ...] for model 1."""
    chains = {}
    model = next(structure.get_models())
    for chain in model:
        items = []
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            aa = AA3_TO_1.get(residue.resname.upper(), "X")
            items.append((aa, residue["CA"]))
        if items:
            chains[chain.id] = items
    return chains


def _chain_pairs(
    ref_chains: dict[str, list[tuple[str, object]]],
    mob_chains: dict[str, list[tuple[str, object]]],
) -> list[tuple[str, str]]:
    """Map reference chains to mobile chains by ID first, then by order."""
    ref_ids = list(ref_chains)
    mob_ids = list(mob_chains)
    pairs = []
    for idx, ref_chain in enumerate(ref_ids):
        if ref_chain in mob_chains:
            mob_chain = ref_chain
        elif idx < len(mob_ids):
            mob_chain = mob_ids[idx]
        else:
            continue
        pairs.append((ref_chain, mob_chain))
    return pairs


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.1
    return aligner


def _sequence_matched_ca_pairs(
    ref_items: list[tuple[str, object]],
    mob_items: list[tuple[str, object]],
    aligner: PairwiseAligner,
) -> list[tuple[object, object]]:
    """Match C-alpha atoms after sequence alignment."""
    ref_seq = "".join(aa for aa, _ in ref_items)
    mob_seq = "".join(aa for aa, _ in mob_items)
    alignment = aligner.align(ref_seq, mob_seq)[0]

    pairs = []
    for (ref_start, ref_end), (mob_start, mob_end) in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        block_len = min(ref_end - ref_start, mob_end - mob_start)
        for offset in range(block_len):
            pairs.append(
                (
                    ref_items[ref_start + offset][1],
                    mob_items[mob_start + offset][1],
                )
            )
    return pairs


def _iterative_fit(
    ca_pairs: list[tuple[object, object]],
    cycles: int = 5,
    cutoff: float = 2.0,
) -> tuple[Superimposer, int]:
    """Fit mobile atoms onto reference atoms with PyMOL-like outlier trimming."""
    if len(ca_pairs) < 3:
        raise ValueError("Need at least 3 matched CA atoms to superimpose structures.")

    fixed_atoms = [pair[0] for pair in ca_pairs]
    moving_atoms = [pair[1] for pair in ca_pairs]
    keep = np.arange(len(ca_pairs))

    superimposer = Superimposer()
    for _ in range(cycles + 1):
        superimposer.set_atoms(
            [fixed_atoms[i] for i in keep],
            [moving_atoms[i] for i in keep],
        )
        rotation, translation = superimposer.rotran
        moved = np.array([moving_atoms[i].coord @ rotation + translation for i in keep])
        fixed = np.array([fixed_atoms[i].coord for i in keep])
        distances = np.sqrt(((moved - fixed) ** 2).sum(axis=1))
        next_keep = keep[distances <= cutoff]

        if len(next_keep) == len(keep) or len(next_keep) < 3:
            break
        keep = next_keep

    return superimposer, len(keep)


def _resolve_reference_path(
    organized_pdb_dir: Path,
    pdb_id: str,
    reference_path: Optional[Path],
) -> Path:
    """Find the reference PDB in either the organized bundle or data/prep."""
    if reference_path is not None:
        reference_path = Path(reference_path)
        if reference_path.exists():
            return reference_path
        raise FileNotFoundError(f"Reference structure not found: {reference_path}")

    candidates = [
        organized_pdb_dir / f"{pdb_id}_crystal.pdb",
        organized_pdb_dir / f"{pdb_id}_protein.pdb",
    ]

    for parent in [organized_pdb_dir, *organized_pdb_dir.parents]:
        if parent.name == "data":
            candidates.append(parent / "prep" / pdb_id / "structure" / f"{pdb_id}_protein.pdb")
            candidates.append(parent / "prep" / pdb_id / "structure" / f"{pdb_id}.pdb")
            break

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a reference structure. Tried:\n  {tried}")


def align_organized_af3_models(
    organized_pdb_dir: Path,
    reference_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    pdb_id: Optional[str] = None,
    model_glob: str = "*.cif",
    cycles: int = 5,
    cutoff: float = 2.0,
) -> pd.DataFrame:
    """Align organized AF3 models to a reference structure.

    Parameters
    ----------
    organized_pdb_dir:
        Directory produced by ``organize_af3_results`` for one PDB ID.
    reference_path:
        Optional explicit reference structure path. In Colab this is usually
        ``data/prep/<PDB_ID>/structure/<PDB_ID>_protein.pdb``.
    output_dir:
        Where aligned model PDBs and the RMSD CSV should be written.
    pdb_id:
        PDB ID. Defaults to ``organized_pdb_dir.name``.
    model_glob:
        Glob pattern for AF3 models inside ``organized_pdb_dir/models``.
    cycles, cutoff:
        Iterative outlier-trimming settings. Defaults mimic PyMOL's align
        defaults closely enough for notebook QC.
    """
    organized_pdb_dir = Path(organized_pdb_dir)
    pdb_id = (pdb_id or organized_pdb_dir.name).upper()
    reference_path = _resolve_reference_path(organized_pdb_dir, pdb_id, reference_path)
    models_dir = organized_pdb_dir / "models"
    output_dir = Path(output_dir) if output_dir is not None else organized_pdb_dir / "aligned_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = sorted(models_dir.glob(model_glob))
    if not model_paths:
        raise FileNotFoundError(f"No AF3 model files matched {models_dir / model_glob}")

    aligner = _make_aligner()
    ref_structure = _read_structure(reference_path, "reference")
    ref_chains = _ca_items_by_chain(ref_structure)
    rows = []

    for model_path in model_paths:
        model_name = model_path.stem
        mob_structure = _read_structure(model_path, model_name)
        mob_chains = _ca_items_by_chain(mob_structure)
        chain_pairs = _chain_pairs(ref_chains, mob_chains)

        all_ca_pairs = []
        chain_map_parts = []
        for ref_chain, mob_chain in chain_pairs:
            ca_pairs = _sequence_matched_ca_pairs(
                ref_chains[ref_chain],
                mob_chains[mob_chain],
                aligner,
            )
            all_ca_pairs.extend(ca_pairs)
            chain_map_parts.append(f"{ref_chain}->{mob_chain}:{len(ca_pairs)} CA")

        if len(all_ca_pairs) < 3:
            rows.append(
                {
                    "model": model_name,
                    "rmsd_ca": np.nan,
                    "n_ca_initial": len(all_ca_pairs),
                    "n_ca_used": 0,
                    "chain_map": "; ".join(chain_map_parts),
                    "aligned_model": "",
                }
            )
            continue

        superimposer, n_ca_used = _iterative_fit(all_ca_pairs, cycles=cycles, cutoff=cutoff)
        superimposer.apply(list(mob_structure.get_atoms()))

        aligned_path = output_dir / f"{model_name}_aligned.pdb"
        writer = PDBIO()
        writer.set_structure(mob_structure)
        writer.save(str(aligned_path))

        rows.append(
            {
                "model": model_name,
                "rmsd_ca": float(superimposer.rms),
                "n_ca_initial": len(all_ca_pairs),
                "n_ca_used": int(n_ca_used),
                "chain_map": "; ".join(chain_map_parts),
                "aligned_model": str(aligned_path),
            }
        )

    df = pd.DataFrame(rows).sort_values("rmsd_ca", na_position="last")
    df.to_csv(organized_pdb_dir / "colab_alignment_rmsd.csv", index=False)
    return df


def read_text_for_view(path: Path) -> str:
    """Read a model file as text for py3Dmol."""
    return Path(path).read_text()
