from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from glyco_design.base import DesignResult


def load_manifest(path: str | Path, protein_class: Optional[str] = None) -> List[dict]:
    """Load a benchmark manifest CSV.

    If `protein_class` is given (e.g. 'glycoprotein' or 'control'), only rows
    matching that class are returned.
    """
    rows: List[dict] = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if protein_class is None or row.get("protein_class") == protein_class:
                rows.append(row)
    return rows


def parse_glycosites(glycosite_string: str) -> List[int]:
    """Parse 'N1,N2,N3'-style glycosite strings into a list of PDB residue ints."""
    if not glycosite_string:
        return []
    out: List[int] = []
    for token in glycosite_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def find_pdb_file(pdb_id: str, chain_id: str, pdb_root: str | Path) -> Optional[str]:
    """Find a PDB file under `pdb_root/{glycoproteins,controls}/` by id+chain."""
    pdb_root = Path(pdb_root)
    patterns = [f"{pdb_id}_{chain_id}.pdb", f"{pdb_id}.pdb"]
    for subdir in ("glycoproteins", "controls", ""):
        base = pdb_root / subdir if subdir else pdb_root
        for pattern in patterns:
            candidate = base / pattern
            if candidate.exists():
                return str(candidate)
    return None


def save_designs_fasta(
    result: DesignResult,
    output_path: str | Path,
    pdb_id: str,
    condition: str,
    model_name: str,
) -> None:
    """Write a FASTA with one record per designed sequence.

    Header format: >{pdb_id}_{model}_{condition}_design{NN}|score={s}|seqid={i}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seqids = result.seqid if result.seqid else [float("nan")] * len(result)
    with open(output_path, "w") as f:
        for i, (seq, score, seqid) in enumerate(zip(result.sequences, result.scores, seqids)):
            header = (
                f">{pdb_id}_{model_name}_{condition}_design{i:02d}"
                f"|score={score:.4f}|seqid={seqid:.4f}"
            )
            f.write(f"{header}\n{seq}\n")


def load_designs_fasta(path: str | Path, max_seqs: Optional[int] = None) -> DesignResult:
    """Load a design FASTA written by `save_designs_fasta`.

    The parser is intentionally permissive so downloaded or hand-collected FASTA
    files can be reused even if scores/seqids are absent from the headers.
    """
    path = Path(path)
    sequences: List[str] = []
    scores: List[float] = []
    seqids: List[float] = []
    header: Optional[str] = None
    chunks: List[str] = []

    def flush_record() -> None:
        nonlocal header, chunks
        if header is None:
            return
        seq = "".join(chunks).replace(" ", "").strip()
        if not seq:
            header = None
            chunks = []
            return
        sequences.append(seq)
        meta = _parse_fasta_header_meta(header)
        scores.append(meta.get("score", float("nan")))
        seqids.append(meta.get("seqid", float("nan")))
        header = None
        chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                if max_seqs is not None and len(sequences) >= max_seqs:
                    break
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if max_seqs is None or len(sequences) < max_seqs:
            flush_record()

    if max_seqs is not None:
        sequences = sequences[:max_seqs]
        scores = scores[:max_seqs]
        seqids = seqids[:max_seqs]

    return DesignResult(sequences=sequences, scores=scores, seqid=seqids)


def _parse_fasta_header_meta(header: str) -> dict:
    meta = {}
    for part in header.split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            meta[key] = float(value)
        except ValueError:
            continue
    return meta
