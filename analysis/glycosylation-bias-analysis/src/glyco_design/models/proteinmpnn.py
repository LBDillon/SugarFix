"""ProteinMPNN adapter (via ColabDesign's mk_mpnn_model wrapper)."""
from __future__ import annotations

from typing import List, Optional

from glyco_design.base import DesignModel, DesignResult


class ProteinMPNNDesignModel(DesignModel):
    name = "proteinmpnn"

    DEFAULT_WEIGHTS = "v_48_030"  # vanilla ProteinMPNN v1.0, 30-epoch

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        self.weights = weights
        self._model = None

    def load(self, **kwargs) -> None:
        from colabdesign.mpnn import mk_mpnn_model

        self._model = mk_mpnn_model(self.weights)
        print(f"[proteinmpnn] loaded weights: {self.weights}")

    def generate(
        self,
        pdb_path: str,
        chain: str,
        num_seqs: int = 32,
        temperature: float = 0.1,
        fix_pos: Optional[List[int]] = None,
    ) -> DesignResult:
        if self._model is None:
            raise RuntimeError("Call .load() before .generate()")

        fix_pos_str = ",".join(str(p) for p in fix_pos) if fix_pos else None
        self._model.prep_inputs(
            pdb_filename=pdb_path,
            chain=chain,
            fix_pos=fix_pos_str,
            verbose=False,
        )
        # ColabDesign's `num` is outer batches; `batch` is inner parallel samples
        # per forward pass. Total designs = num * batch (capped at num_seqs).
        num_outer = max(1, num_seqs // 32)
        batch = min(32, num_seqs)
        out = self._model.sample(num=num_outer, batch=batch, temperature=temperature)

        return DesignResult(
            sequences=list(out["seq"]),
            scores=list(out["score"]),
            seqid=list(out.get("seqid", [])),
            meta={"weights": self.weights, "temperature": temperature},
        )
