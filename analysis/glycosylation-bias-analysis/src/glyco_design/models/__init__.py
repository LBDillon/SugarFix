"""Adapters that wrap each inverse-folding model behind the common DesignModel API.

Each adapter is in its own module so its heavy dependencies (torch, jax,
model-specific packages) are only imported when that model is used.

    from glyco_design.models.proteinmpnn import ProteinMPNNDesignModel
    from glyco_design.models.esm_if import ESMIFDesignModel
    from glyco_design.models.triflow import TriFlowDesignModel
    # from glyco_design.models.caliby import CalibyDesignModel   # TODO
"""
