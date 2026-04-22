"""TriFlow adapter (discrete flow-matching inverse folding).

TriFlow is distributed as a GitHub repo (not a pip package), so the user must
clone https://github.com/jzhoulab/TriFlow first and pass `triflow_dir`. The
repo contains weights under `weights/afdb_dataset/afdb_weights.pt` and loads
them via paths relative to its own root — we `os.chdir(triflow_dir)` inside
`load()` to match.

We bypass the file-writing `predict()` method and call the underlying
`interpolant.aa_sample(...)` directly (same approach as the scoring notebook
in decoding-design-bias), so sampling stays in-memory and each sample yields
its own log-probability score via the terminal-step `unmasked_probs`.
"""
from __future__ import annotations

import os
import sys
import gc
from pathlib import Path
from typing import List, Optional

from glyco_design.base import DesignModel, DesignResult


class TriFlowDesignModel(DesignModel):
    name = "triflow"

    def __init__(
        self,
        triflow_dir: str,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.triflow_dir = str(Path(triflow_dir).resolve())
        self.ckpt_path = ckpt_path or str(
            Path(self.triflow_dir) / "weights" / "afdb_dataset" / "afdb_weights.pt"
        )
        self.device = device
        self._predictor = None
        self._aatype_to_str = None

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        import torch

        if device in (None, "auto"):
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[triflow] requested CUDA but it is unavailable; using CPU")
            return "cpu"
        return device

    def load(self, **kwargs) -> None:
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"TriFlow weights not found at {self.ckpt_path}")
        if self.triflow_dir not in sys.path:
            sys.path.insert(0, self.triflow_dir)
        os.chdir(self.triflow_dir)

        from sample import TriFoldPredictor

        self.device = self._resolve_device(self.device)
        self._predictor = TriFoldPredictor(ckpt_path=self.ckpt_path, device=self.device)

        # TriFlow stores the AA-index → letter decoder in a few possible places
        # across versions; try the known ones.
        try:
            from triflow.data.data_transforms import _aatype_to_str_sequence
        except ImportError:
            try:
                from openfold.data.data_transforms import _aatype_to_str_sequence
            except ImportError:
                from sample import _aatype_to_str_sequence  # re-exported in sample.py
        self._aatype_to_str = _aatype_to_str_sequence
        print(f"[triflow] loaded {self.ckpt_path} on {self.device}")

    def generate(
        self,
        pdb_path: str,
        chain: str,
        num_seqs: int = 32,
        temperature: float = 0.1,
        fix_pos: Optional[List[int]] = None,
    ) -> DesignResult:
        if self._predictor is None:
            raise RuntimeError("Call .load() before .generate()")

        import torch
        from triflow.utils.loss import scale_trans
        from triflow.utils.rigid_utils import Rigid
        from triflow.utils.tensor_utils import tensor_tree_map

        predictor = self._predictor

        data, seq_prior = predictor.process_pdb(
            pdb_path=pdb_path,
            chain_condition=chain if fix_pos else None,
            res_condition=fix_pos if fix_pos else None,
        )
        data = tensor_tree_map(lambda x: x.clone().to(predictor.device), data)
        seq_prior_base = seq_prior.clone().to(predictor.device)

        rigid_frames = (
            Rigid.from_tensor_4x4(data["backbone_rigid_tensor"][..., -1])
            .to_tensor_7()
            .to(predictor.device)
        )
        rigid_frames = scale_trans(rigid_frames, 0.1)
        data["noise_label"] = torch.tensor([0.0], device=predictor.device)[None]
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        native_seq_ids = torch.argmax(data["target_feat"][..., -1], dim=-1)[0]  # (L,)
        native_seq = self._aatype_to_str(native_seq_ids.cpu())

        sequences: List[str] = []
        scores: List[float] = []
        seqids: List[float] = []

        with torch.no_grad():
            for _ in range(num_seqs):
                prot_traj, _conf, unmasked_probs = predictor.interpolant.aa_sample(
                    data,
                    predictor.model,
                    rigid_frames,
                    aa_init=seq_prior_base.clone(),
                    temp=temperature,
                    omit_AA=None,
                    tied_weights=False,
                    sample_priority=False,
                    run_cfg=False,
                    sample_purity=False,
                )
                sampled_ids = prot_traj[-1][0]  # (L,) final AA tokens
                seq = self._aatype_to_str(sampled_ids.cpu())

                probs = unmasked_probs[0]  # (L, 21)
                idx = torch.arange(sampled_ids.shape[0], device=probs.device)
                chosen = probs[idx, sampled_ids].clamp(min=1e-30)
                avg_log_prob = torch.log(chosen).mean().item()

                sequences.append(seq)
                scores.append(avg_log_prob)
                n = min(len(seq), len(native_seq))
                seqids.append(
                    sum(1 for i in range(n) if seq[i] == native_seq[i]) / n if n else float("nan")
                )
                del prot_traj, _conf, unmasked_probs, probs, chosen, idx, sampled_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return DesignResult(
            sequences=sequences,
            scores=scores,
            seqid=seqids,
            meta={"ckpt_path": self.ckpt_path, "temperature": temperature},
        )
