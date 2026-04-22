from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DesignResult:
    sequences: List[str]
    scores: List[float]
    seqid: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequences)


class DesignModel(ABC):
    """Common interface every inverse-folding adapter implements.

    Positions use 1-indexed coordinates along the designable chain's residue
    ordering (i.e. the ordering ProteinMPNN/ESM-IF see after the PDB is
    parsed), not raw PDB residue numbers. The pipeline handles the PDB-to-
    model-position mapping before calling `generate`.
    """

    name: str = "base"

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load weights and move to device. Call once."""

    @abstractmethod
    def generate(
        self,
        pdb_path: str,
        chain: str,
        num_seqs: int = 32,
        temperature: float = 0.1,
        fix_pos: Optional[List[int]] = None,
    ) -> DesignResult:
        """Sample `num_seqs` sequences from the model.

        `fix_pos` is accepted for interface symmetry but is not used by the
        unconstrained notebook. Adapters may raise NotImplementedError on
        fix_pos until the fixed-sequon notebook is written.
        """
