"""ESM-IF (ESM inverse folding) adapter.

Uses facebookresearch/esm's `esm.inverse_folding` module with weights
`esm_if1_gvp4_t16_142M_UR50`. Each `model.sample(coords, temperature=T)`
call returns one string — we loop to produce `num_seqs` samples. The
per-sequence score is the model's own avg log-likelihood of the sampled
sequence (higher = better), matching how the scoring notebook in
decoding-design-bias reports ESM-IF scores.
"""
from __future__ import annotations

from typing import List, Optional

from glyco_design.base import DesignModel, DesignResult


class ESMIFDesignModel(DesignModel):
    name = "esm_if"

    DEFAULT_WEIGHTS = "esm_if1_gvp4_t16_142M_UR50"

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str = "cuda"):
        self.weights = weights
        self.device = device
        self._model = None
        self._alphabet = None

    def load(self, **kwargs) -> None:
        import esm  # noqa: F401  (namespace trigger)
        import esm.inverse_folding  # noqa: F401
        from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

        model, alphabet = esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(self.device)
        self._model = model
        self._alphabet = alphabet
        print(f"[esm_if] loaded {self.weights} on {self.device}")

    def _load_coords(self, pdb_path: str, chain: str):
        from esm.inverse_folding.util import (
            extract_coords_from_structure,
            load_structure,
        )

        structure = load_structure(pdb_path, chain)
        coords, native_seq = extract_coords_from_structure(structure)
        return coords, native_seq

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

        from esm.inverse_folding.util import score_sequence

        coords, native_seq = self._load_coords(pdb_path, chain)

        partial_seq = None
        if fix_pos:
            # Mask every position; fix the requested ones to their native AA.
            chars = ["<mask>"] * len(native_seq)
            for p in fix_pos:
                i = p - 1
                if 0 <= i < len(native_seq):
                    chars[i] = native_seq[i]
            partial_seq = chars

        sequences: List[str] = []
        scores: List[float] = []
        seqids: List[float] = []

        for _ in range(num_seqs):
            seq = self._model.sample(
                coords,
                temperature=temperature,
                device=self.device,
                partial_seq=partial_seq,
            )
            _, avg_ll, _ = score_sequence(self._model, self._alphabet, coords, seq)
            scores.append(float(avg_ll))
            sequences.append(seq)
            n = min(len(seq), len(native_seq))
            if n:
                matches = sum(1 for i in range(n) if seq[i] == native_seq[i])
                seqids.append(matches / n)
            else:
                seqids.append(float("nan"))

        return DesignResult(
            sequences=sequences,
            scores=scores,
            seqid=seqids,
            meta={"weights": self.weights, "temperature": temperature},
        )
