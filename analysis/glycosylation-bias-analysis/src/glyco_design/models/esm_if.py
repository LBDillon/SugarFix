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

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: Optional[str] = None):
        self.weights = weights
        self.device = device
        self._model = None
        self._alphabet = None

    @staticmethod
    def _patch_biotite_for_esm() -> None:
        """Keep fair-esm 2.0.0 imports working with newer Biotite releases."""
        import biotite.structure as bs

        if not hasattr(bs, "filter_backbone") and hasattr(bs, "filter_peptide_backbone"):
            bs.filter_backbone = bs.filter_peptide_backbone  # type: ignore[attr-defined]

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        import torch

        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[esm_if] requested CUDA but it is unavailable; using CPU")
            return "cpu"
        return device

    def load(self, **kwargs) -> None:
        if self.weights != self.DEFAULT_WEIGHTS:
            raise ValueError(f"Unsupported ESM-IF weights: {self.weights!r}")

        self._patch_biotite_for_esm()
        import esm  # noqa: F401  (namespace trigger)
        import esm.inverse_folding  # noqa: F401
        from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

        self.device = self._resolve_device(self.device)
        model, alphabet = esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(self.device)
        self._model = model
        self._alphabet = alphabet
        print(f"[esm_if] loaded {self.weights} on {self.device}")

    def _load_coords(self, pdb_path: str, chain: str):
        self._patch_biotite_for_esm()
        from esm.inverse_folding.util import (
            extract_coords_from_structure,
            load_structure,
        )

        structure = load_structure(pdb_path, chain)
        coords, native_seq = extract_coords_from_structure(structure)
        return coords, native_seq

    def _sample_sequence(
        self,
        coords,
        temperature: float,
        partial_seq: Optional[List[str]] = None,
    ) -> str:
        import torch
        import torch.nn.functional as F
        from esm.inverse_folding.util import CoordBatchConverter

        dictionary = self._model.decoder.dictionary
        batch_converter = CoordBatchConverter(dictionary)
        batch_coords, confidence, _, _, padding_mask = batch_converter(
            [(coords, None, None)]
        )
        batch_coords = batch_coords.to(self.device)
        confidence = confidence.to(self.device)
        padding_mask = padding_mask.to(self.device)

        length = len(coords)
        mask_idx = dictionary.get_idx("<mask>")
        sampled_tokens = torch.full(
            (1, 1 + length), mask_idx, dtype=torch.long, device=self.device
        )
        sampled_tokens[0, 0] = dictionary.get_idx("<cath>")
        if partial_seq is not None:
            for i, token in enumerate(partial_seq):
                sampled_tokens[0, i + 1] = dictionary.get_idx(token)

        incremental_state = {}
        with torch.no_grad():
            encoder_out = self._model.encoder(batch_coords, padding_mask, confidence)
            for i in range(1, length + 1):
                if sampled_tokens[0, i] != mask_idx:
                    continue
                logits, _ = self._model.decoder(
                    sampled_tokens[:, :i],
                    encoder_out,
                    incremental_state=incremental_state,
                )
                logits = logits[0].transpose(0, 1)
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        sampled_seq = sampled_tokens[0, 1:].detach().cpu().tolist()
        return "".join(dictionary.get_tok(token) for token in sampled_seq)

    def _score_sequence(self, coords, seq: str) -> float:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from esm.inverse_folding.util import CoordBatchConverter

        batch_converter = CoordBatchConverter(self._alphabet)
        batch_coords, confidence, _, tokens, padding_mask = batch_converter(
            [(coords, None, seq)]
        )
        batch_coords = batch_coords.to(self.device)
        confidence = confidence.to(self.device)
        padding_mask = padding_mask.to(self.device)
        tokens = tokens.to(self.device)

        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]

        with torch.no_grad():
            logits, _ = self._model.forward(
                batch_coords, padding_mask, confidence, prev_output_tokens
            )
            loss = F.cross_entropy(logits, target, reduction="none")

        loss = loss[0].detach().cpu().numpy()
        coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
        if not coord_mask.any():
            return float("nan")
        return float(-np.sum(loss * coord_mask) / np.sum(coord_mask))

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
            seq = self._sample_sequence(
                coords, temperature=temperature, partial_seq=partial_seq
            )
            avg_ll = self._score_sequence(coords, seq)
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
