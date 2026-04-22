"""ProteinMPNN adapter (via ColabDesign's mk_mpnn_model wrapper)."""
from __future__ import annotations

from math import ceil
from typing import List, Optional

from glyco_design.base import DesignModel, DesignResult


class ProteinMPNNDesignModel(DesignModel):
    name = "proteinmpnn"

    DEFAULT_WEIGHTS = "v_48_030"  # vanilla ProteinMPNN v1.0, 30-epoch

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        self.weights = weights
        self._model = None

    @staticmethod
    def _patch_colabdesign_runtime() -> None:
        """Keep ColabDesign 1.1.0 working with newer NumPy/JAX releases."""
        import numpy as np

        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]

        import jax

        if not hasattr(jax, "tree_map"):
            jax.tree_map = jax.tree_util.tree_map  # type: ignore[attr-defined]

    def load(self, **kwargs) -> None:
        self._patch_colabdesign_runtime()
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
        # per forward pass. Total designs = num * batch, then we trim to request.
        batch = min(32, num_seqs)
        num_outer = max(1, ceil(num_seqs / batch))
        out = self._model.sample(num=num_outer, batch=batch, temperature=temperature)

        return DesignResult(
            sequences=list(out["seq"])[:num_seqs],
            scores=list(out["score"])[:num_seqs],
            seqid=list(out.get("seqid", []))[:num_seqs],
            meta={"weights": self.weights, "temperature": temperature},
        )
