from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from glyco_design.base import DesignModel
from glyco_design.io import (
    find_pdb_file,
    load_designs_fasta,
    load_manifest,
    parse_glycosites,
    save_designs_fasta,
)
from glyco_design.sequon import (
    check_sequon,
    first_chain_with_residues,
    pdb_to_model_position,
)

RETENTION_COLUMNS = [
    "model",
    "pdb_id",
    "chain_id",
    "condition",
    "design_idx",
    "glycosite_pdb",
    "glycosite_model",
    "sequon_status",
    "design_score",
    "seqid",
]

STATUS_COLUMNS = [
    "model",
    "pdb_id",
    "chain_id",
    "status",
    "n_designs",
    "fasta_path",
    "source_path",
    "error",
]


def run_unconstrained_experiment(
    model: DesignModel,
    manifest_path: str | Path,
    pdb_root: str | Path,
    output_dir: str | Path,
    num_seqs: int = 32,
    temperature: float = 0.1,
    protein_class: Optional[str] = None,
    limit: Optional[int] = None,
    cache_existing: bool = True,
    overwrite: bool = False,
    existing_fasta_dirs: Optional[Iterable[str | Path]] = None,
    checkpoint: bool = True,
) -> pd.DataFrame:
    """Run unconstrained design for every protein in the manifest.

    Writes one FASTA per protein to `output_dir/<pdb>_<chain>_unconstrained.fasta`,
    returns a per-site retention DataFrame (rows: one per (design, glycosite)).

    Parameters
    ----------
    protein_class
        Filter manifest rows by 'protein_class' column (e.g. 'glycoprotein').
        None = use all rows.
    limit
        Stop after this many proteins (useful for quick sanity checks).
    cache_existing
        Reuse existing FASTA files instead of regenerating them.
    overwrite
        Regenerate designs even when a matching FASTA already exists.
    existing_fasta_dirs
        Extra directories to search for previously downloaded FASTA files.
        Matching files are copied into `output_dir` before scoring.
    checkpoint
        Rewrite the retention/status CSVs after every protein so interrupted
        notebook runs still leave usable results behind.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sequon_retention_unconstrained.csv"
    status_path = output_dir / "run_status_unconstrained.csv"

    proteins = load_manifest(manifest_path, protein_class=protein_class)
    if limit is not None:
        proteins = proteins[:limit]
    print(f"[{model.name}] {len(proteins)} proteins from {manifest_path}")

    rows: List[dict] = []
    status_rows: List[dict] = []
    ok, skipped = 0, 0

    for i, protein in enumerate(proteins):
        pdb_id = protein["pdb_id"]
        chain_id = protein["chain_id"]
        glycosites_pdb = parse_glycosites(protein.get("glycosite_positions", ""))
        print(f"\n[{i + 1}/{len(proteins)}] {pdb_id} chain {chain_id} "
              f"({len(glycosites_pdb)} glycosites)")

        pdb_path = find_pdb_file(pdb_id, chain_id, pdb_root)
        if pdb_path is None:
            print(f"  skip: PDB not found under {pdb_root}")
            skipped += 1
            status_rows.append(
                _status_row(
                    model.name, pdb_id, chain_id, "skipped", 0, "", "", "PDB not found"
                )
            )
            if checkpoint:
                _write_checkpoint(rows, summary_path)
                _write_status(status_rows, status_path)
            continue

        actual_chain = first_chain_with_residues(pdb_path)
        if actual_chain is None:
            print(f"  skip: no chain with amino acids")
            skipped += 1
            status_rows.append(
                _status_row(
                    model.name,
                    pdb_id,
                    chain_id,
                    "skipped",
                    0,
                    "",
                    "",
                    "no chain with amino acids",
                )
            )
            if checkpoint:
                _write_checkpoint(rows, summary_path)
                _write_status(status_rows, status_path)
            continue
        if actual_chain != chain_id:
            print(f"  note: chain mismatch — manifest {chain_id!r} → using {actual_chain!r}")
            chain_id = actual_chain

        glycosite_pairs: List[tuple[int, int]] = []
        for pdb_pos in glycosites_pdb:
            mpos = pdb_to_model_position(pdb_pos, pdb_path, chain_id)
            if mpos is None:
                print(f"  warn: could not map glycosite PDB:{pdb_pos}")
                continue
            glycosite_pairs.append((pdb_pos, mpos))

        fasta_path = output_dir / f"{pdb_id}_{chain_id}_unconstrained.fasta"
        result = None
        source_path = ""
        status = "generated"
        error = ""

        if cache_existing and not overwrite:
            cached_path = _find_cached_fasta(
                pdb_id=pdb_id,
                chain_id=chain_id,
                output_dir=output_dir,
                existing_fasta_dirs=existing_fasta_dirs,
            )
            if cached_path is not None:
                try:
                    result = load_designs_fasta(cached_path, max_seqs=num_seqs)
                    if len(result) == 0:
                        print(f"  cache ignored: no sequences in {cached_path}")
                        result = None
                    else:
                        source_path = str(cached_path)
                        status = "cached"
                        if cached_path.resolve() != fasta_path.resolve():
                            shutil.copy2(cached_path, fasta_path)
                        print(f"  reused {len(result)} cached designs ← {cached_path.name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  cache ignored: could not read {cached_path}: {exc}")
                    result = None

        if result is None:
            try:
                result = model.generate(
                    pdb_path=pdb_path,
                    chain=chain_id,
                    num_seqs=num_seqs,
                    temperature=temperature,
                    fix_pos=None,
                )
            except Exception as exc:  # noqa: BLE001 — adapter-specific failures are logged, not raised
                error = str(exc)
                print(f"  error generating: {exc}")
                skipped += 1
                status_rows.append(
                    _status_row(
                        model.name, pdb_id, chain_id, "error", 0, "", source_path, error
                    )
                )
                if checkpoint:
                    _write_checkpoint(rows, summary_path)
                    _write_status(status_rows, status_path)
                _cleanup_accelerator_memory()
                continue

            save_designs_fasta(result, fasta_path, pdb_id=pdb_id,
                               condition="unconstrained", model_name=model.name)
            print(f"  saved {len(result)} designs -> {fasta_path.name}")

        if result is None:
            skipped += 1
            continue

        rows.extend(_retention_rows(model.name, pdb_id, chain_id, result, glycosite_pairs))
        status_rows.append(
            _status_row(
                model.name,
                pdb_id,
                chain_id,
                status,
                len(result),
                str(fasta_path),
                source_path,
                error,
            )
        )
        ok += 1
        if checkpoint:
            _write_checkpoint(rows, summary_path)
            _write_status(status_rows, status_path)
        _cleanup_accelerator_memory()

    df = _write_checkpoint(rows, summary_path)
    _write_status(status_rows, status_path)
    print(f"\n[{model.name}] done: {ok} ok, {skipped} skipped → {summary_path}")

    if not df.empty:
        preserved = df["sequon_status"].isin(["NXS", "NXT"]).sum()
        total = len(df)
        print(f"[{model.name}] sequons preserved: {preserved}/{total} ({100 * preserved / total:.1f}%)")

    return df


def summarize_unconstrained_cache(
    manifest_path: str | Path,
    pdb_root: str | Path,
    fasta_dirs: Iterable[str | Path],
    protein_class: Optional[str] = None,
    limit: Optional[int] = None,
    num_seqs: Optional[int] = None,
) -> pd.DataFrame:
    """Report which manifest entries already have reusable unconstrained FASTAs."""
    proteins = load_manifest(manifest_path, protein_class=protein_class)
    if limit is not None:
        proteins = proteins[:limit]

    rows: List[dict] = []
    for protein in proteins:
        pdb_id = protein["pdb_id"]
        manifest_chain = protein["chain_id"]
        chain_id = manifest_chain
        pdb_path = find_pdb_file(pdb_id, chain_id, pdb_root)
        if pdb_path is not None:
            actual_chain = first_chain_with_residues(pdb_path)
            if actual_chain is not None:
                chain_id = actual_chain
        cached_path = _find_cached_fasta(
            pdb_id=pdb_id,
            chain_id=chain_id,
            output_dir=None,
            existing_fasta_dirs=fasta_dirs,
        )
        n_designs = 0
        if cached_path is not None:
            try:
                n_designs = len(load_designs_fasta(cached_path, max_seqs=num_seqs))
            except Exception:  # noqa: BLE001
                n_designs = 0
        rows.append({
            "pdb_id": pdb_id,
            "manifest_chain": manifest_chain,
            "chain_id": chain_id,
            "has_fasta": cached_path is not None and n_designs > 0,
            "n_designs": n_designs,
            "fasta_path": str(cached_path) if cached_path is not None else "",
        })

    return pd.DataFrame(rows)


def _retention_rows(
    model_name: str,
    pdb_id: str,
    chain_id: str,
    result,
    glycosite_pairs: List[tuple[int, int]],
) -> List[dict]:
    rows: List[dict] = []
    for seq_idx, seq in enumerate(result.sequences):
        score = result.scores[seq_idx] if seq_idx < len(result.scores) else float("nan")
        seqid = result.seqid[seq_idx] if seq_idx < len(result.seqid) else float("nan")
        for glycosite_pdb, mpos in glycosite_pairs:
            rows.append({
                "model": model_name,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "condition": "unconstrained",
                "design_idx": seq_idx,
                "glycosite_pdb": glycosite_pdb,
                "glycosite_model": mpos,
                "sequon_status": check_sequon(seq, mpos - 1),
                "design_score": score,
                "seqid": seqid,
            })
    return rows


def _find_cached_fasta(
    pdb_id: str,
    chain_id: str,
    output_dir: Optional[Path],
    existing_fasta_dirs: Optional[Iterable[str | Path]],
) -> Optional[Path]:
    filename = f"{pdb_id}_{chain_id}_unconstrained.fasta"
    dirs: List[Path] = []
    if output_dir is not None:
        dirs.append(Path(output_dir))
    if existing_fasta_dirs is not None:
        dirs.extend(Path(path) for path in existing_fasta_dirs if path)

    seen = set()
    for directory in dirs:
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _status_row(
    model_name: str,
    pdb_id: str,
    chain_id: str,
    status: str,
    n_designs: int,
    fasta_path: str,
    source_path: str,
    error: str,
) -> dict:
    return {
        "model": model_name,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "status": status,
        "n_designs": n_designs,
        "fasta_path": fasta_path,
        "source_path": source_path,
        "error": error,
    }


def _write_checkpoint(rows: List[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RETENTION_COLUMNS)
    df.to_csv(path, index=False)
    return df


def _write_status(rows: List[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    df.to_csv(path, index=False)
    return df


def _cleanup_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
