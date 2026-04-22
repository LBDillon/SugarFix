from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from glyco_design.base import DesignModel
from glyco_design.io import (
    find_pdb_file,
    load_manifest,
    parse_glycosites,
    save_designs_fasta,
)
from glyco_design.sequon import (
    check_sequon,
    first_chain_with_residues,
    pdb_to_model_position,
)


def run_unconstrained_experiment(
    model: DesignModel,
    manifest_path: str | Path,
    pdb_root: str | Path,
    output_dir: str | Path,
    num_seqs: int = 32,
    temperature: float = 0.1,
    protein_class: Optional[str] = None,
    limit: Optional[int] = None,
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
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proteins = load_manifest(manifest_path, protein_class=protein_class)
    if limit is not None:
        proteins = proteins[:limit]
    print(f"[{model.name}] {len(proteins)} proteins from {manifest_path}")

    rows: List[dict] = []
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
            continue

        actual_chain = first_chain_with_residues(pdb_path)
        if actual_chain is None:
            print(f"  skip: no chain with amino acids")
            skipped += 1
            continue
        if actual_chain != chain_id:
            print(f"  note: chain mismatch — manifest {chain_id!r} → using {actual_chain!r}")
            chain_id = actual_chain

        glycosites_model: List[int] = []
        for pdb_pos in glycosites_pdb:
            mpos = pdb_to_model_position(pdb_pos, pdb_path, chain_id)
            if mpos is None:
                print(f"  warn: could not map glycosite PDB:{pdb_pos}")
                continue
            glycosites_model.append(mpos)

        try:
            result = model.generate(
                pdb_path=pdb_path,
                chain=chain_id,
                num_seqs=num_seqs,
                temperature=temperature,
                fix_pos=None,
            )
        except Exception as exc:  # noqa: BLE001 — adapter-specific failures are logged, not raised
            print(f"  error generating: {exc}")
            skipped += 1
            continue

        fasta_path = output_dir / f"{pdb_id}_{chain_id}_unconstrained.fasta"
        save_designs_fasta(result, fasta_path, pdb_id=pdb_id,
                           condition="unconstrained", model_name=model.name)
        print(f"  saved {len(result)} designs → {fasta_path.name}")

        for seq_idx, seq in enumerate(result.sequences):
            score = result.scores[seq_idx] if seq_idx < len(result.scores) else float("nan")
            seqid = result.seqid[seq_idx] if seq_idx < len(result.seqid) else float("nan")
            for site_idx, mpos in enumerate(glycosites_model):
                rows.append({
                    "model": model.name,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "condition": "unconstrained",
                    "design_idx": seq_idx,
                    "glycosite_pdb": glycosites_pdb[site_idx] if site_idx < len(glycosites_pdb) else None,
                    "glycosite_model": mpos,
                    "sequon_status": check_sequon(seq, mpos - 1),
                    "design_score": score,
                    "seqid": seqid,
                })
        ok += 1

    df = pd.DataFrame(rows)
    summary_path = output_dir / "sequon_retention_unconstrained.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n[{model.name}] done: {ok} ok, {skipped} skipped → {summary_path}")

    if not df.empty:
        preserved = df["sequon_status"].isin(["NXS", "NXT"]).sum()
        total = len(df)
        print(f"[{model.name}] sequons preserved: {preserved}/{total} ({100 * preserved / total:.1f}%)")

    return df
