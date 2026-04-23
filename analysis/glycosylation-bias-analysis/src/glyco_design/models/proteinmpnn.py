"""ProteinMPNN adapter using the official dauparas/ProteinMPNN CLI."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from glyco_design.base import DesignModel, DesignResult


class ProteinMPNNDesignModel(DesignModel):
    name = "proteinmpnn"

    def __init__(
        self,
        proteinmpnn_dir: Optional[str | Path] = None,
        model_name: Optional[str] = None,
        seed: int = 42,
        weights: Optional[str] = None,
    ):
        # `weights` is kept as a backwards-compatible alias for `model_name`.
        if model_name is None and weights is not None:
            model_name = weights
        self.proteinmpnn_dir = (
            Path(proteinmpnn_dir).resolve() if proteinmpnn_dir else None
        )
        self.model_name = model_name
        self.seed = seed
        self._run_script: Optional[Path] = None

    def load(self, **kwargs) -> None:
        self.proteinmpnn_dir = self._find_proteinmpnn_dir(self.proteinmpnn_dir)
        self._run_script = self.proteinmpnn_dir / "protein_mpnn_run.py"
        if str(self.proteinmpnn_dir) not in sys.path:
            sys.path.insert(0, str(self.proteinmpnn_dir))
        model_desc = self.model_name if self.model_name is not None else "CLI default"
        print(f"[proteinmpnn] using official ProteinMPNN at {self.proteinmpnn_dir}")
        print(f"[proteinmpnn] model: {model_desc}")

    def generate(
        self,
        pdb_path: str,
        chain: str,
        num_seqs: int = 32,
        temperature: float = 0.1,
        fix_pos: Optional[List[int]] = None,
    ) -> DesignResult:
        if self._run_script is None or self.proteinmpnn_dir is None:
            raise RuntimeError("Call .load() before .generate()")

        with tempfile.TemporaryDirectory(prefix="glyco_design_proteinmpnn_") as tmp:
            tmpdir = Path(tmp)
            working_pdb = tmpdir / Path(pdb_path).name
            shutil.copy2(pdb_path, working_pdb)

            cmd = [
                sys.executable,
                str(self._run_script),
                "--pdb_path",
                str(working_pdb),
                "--num_seq_per_target",
                str(num_seqs),
                "--sampling_temp",
                str(temperature),
                "--out_folder",
                str(tmpdir),
                "--seed",
                str(self.seed),
                "--pdb_path_chains",
                chain,
            ]
            if self.model_name is not None:
                cmd.extend(["--model_name", self.model_name])
            if fix_pos:
                fixed_path = tmpdir / "fixed_positions.jsonl"
                self._write_fixed_positions_jsonl(
                    target_name=working_pdb.stem,
                    chain=chain,
                    fix_pos_1idx=fix_pos,
                    output_path=fixed_path,
                )
                cmd.extend(["--fixed_positions_jsonl", str(fixed_path)])

            proc = subprocess.run(
                cmd,
                cwd=str(self.proteinmpnn_dir),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.strip() or proc.stdout.strip()
                raise RuntimeError(f"official ProteinMPNN failed: {stderr[-1200:]}")

            fasta_path = tmpdir / "seqs" / f"{working_pdb.stem}.fa"
            if not fasta_path.exists():
                candidates = sorted((tmpdir / "seqs").glob("*.fa"))
                if candidates:
                    fasta_path = candidates[0]
                else:
                    raise FileNotFoundError(f"ProteinMPNN FASTA not found under {tmpdir}")

            return self._parse_proteinmpnn_fasta(fasta_path, pdb_path, chain, num_seqs)

    @staticmethod
    def _find_proteinmpnn_dir(proteinmpnn_dir: Optional[Path]) -> Path:
        candidates = [
            proteinmpnn_dir,
            Path(os.environ["PROTEINMPNN_DIR"]) if os.environ.get("PROTEINMPNN_DIR") else None,
            Path(os.environ["PROTEINMPNN_PATH"]) if os.environ.get("PROTEINMPNN_PATH") else None,
            Path.cwd() / "ProteinMPNN",
            Path.cwd().parent / "ProteinMPNN",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = candidate.resolve()
            if (candidate / "protein_mpnn_run.py").exists() and (
                candidate / "protein_mpnn_utils.py"
            ).exists():
                return candidate
        raise FileNotFoundError(
            "Official ProteinMPNN not found. Clone "
            "https://github.com/dauparas/ProteinMPNN and set PROTEINMPNN_DIR, "
            "or pass proteinmpnn_dir=..."
        )

    @staticmethod
    def _write_fixed_positions_jsonl(
        target_name: str,
        chain: str,
        fix_pos_1idx: List[int],
        output_path: Path,
    ) -> None:
        payload = {target_name: {chain: sorted(set(fix_pos_1idx))}}
        output_path.write_text(json.dumps(payload))

    def _parse_proteinmpnn_fasta(
        self,
        fasta_path: Path,
        pdb_path: str,
        chain: str,
        num_seqs: int,
    ) -> DesignResult:
        chain_order = self._proteinmpnn_chain_order(pdb_path)
        sequences: List[str] = []
        scores: List[float] = []
        seqids: List[float] = []

        for header, seq in self._read_fasta_records(fasta_path):
            meta = self._parse_header(header)
            if meta["sample"] is None:
                continue
            sequences.append(self._extract_chain_sequence(seq, chain, chain_order))
            scores.append(meta["score"])
            seqids.append(meta["seqid"])
            if len(sequences) >= num_seqs:
                break

        return DesignResult(
            sequences=sequences,
            scores=scores,
            seqid=seqids,
            meta={
                "implementation": "official_dauparas_proteinmpnn_cli",
                "model_name": self.model_name or "cli_default",
                "seed": self.seed,
            },
        )

    def _proteinmpnn_chain_order(self, pdb_path: str) -> List[str]:
        try:
            from protein_mpnn_utils import parse_PDB
        except ImportError:
            return []
        parsed = parse_PDB(str(pdb_path))
        if not parsed:
            return []
        return sorted(k.split("_")[-1] for k in parsed[0] if k.startswith("seq_chain_"))

    @staticmethod
    def _read_fasta_records(fasta_path: Path) -> List[tuple[str, str]]:
        records: List[tuple[str, str]] = []
        header: Optional[str] = None
        chunks: List[str] = []
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(chunks)))
                    header = line[1:]
                    chunks = []
                else:
                    chunks.append(line)
        if header is not None:
            records.append((header, "".join(chunks)))
        return records

    @staticmethod
    def _parse_header(header: str) -> dict:
        sample_match = re.search(r"(?:^|[, ])sample=(\d+)", header)
        score_match = re.search(r"(?:^|[, ])score=([-+0-9.eE]+)", header)
        seqid_match = re.search(r"(?:^|[, ])seq_recovery=([-+0-9.eE]+)", header)
        return {
            "sample": int(sample_match.group(1)) if sample_match else None,
            "score": float(score_match.group(1)) if score_match else float("nan"),
            "seqid": float(seqid_match.group(1)) if seqid_match else float("nan"),
        }

    @staticmethod
    def _extract_chain_sequence(seq: str, chain: str, chain_order: List[str]) -> str:
        if "/" not in seq:
            return seq
        parts = seq.split("/")
        if chain in chain_order:
            idx = chain_order.index(chain)
            if idx < len(parts):
                return parts[idx]
        return "".join(parts)
