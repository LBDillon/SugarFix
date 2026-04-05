"""Collect and visualize AF3 confidence metrics for SugarFix designs.

Adapted from the case-study pipeline's ``collect_af3_metrics.py`` for
single-protein, interactive use from the walkthrough notebook.

Expected AF3 output layout (from the AF3 Server download)::

    af3_results/
        fold_{pdb}_{condition}_AF3/
            *_summary_confidences_0.json
            *_summary_confidences_1.json
            ...
        fold_{pdb}_{condition}_AF3_with_glycans/
            ...

Each ``*_summary_confidences_*.json`` contains at minimum::

    {"ptm": float, "iptm": float, "ranking_score": float,
     "fraction_disordered": float, "has_clash": bool}
"""

from __future__ import annotations

import glob
import json
import re
import statistics
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_condition_from_folder(folder_name: str) -> str:
    """Extract a human-readable condition label from an AF3 folder name."""
    name = folder_name.lower()
    if "with_glycan" in name:
        return "with_glycans"
    if "glycan" in name:
        return "with_glycans"
    if "designer" in name or "selected" in name:
        return "designer_selected"
    if "soft_filter" in name:
        return "soft_filter"
    # Fall back to the folder name itself
    return folder_name


def _extract_metrics_from_folder(folder: Path) -> Optional[dict]:
    """Extract confidence metrics from all seed JSONs in one AF3 folder."""
    conf_files = sorted(glob.glob(str(folder / "*summary_confidences*.json")))
    if not conf_files:
        return None

    all_metrics = []
    for cf in conf_files:
        with open(cf) as f:
            data = json.load(f)
        all_metrics.append({
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
        })

    if not all_metrics:
        return None

    # Use the first seed (model 0) as the primary, with cross-seed stats
    primary = all_metrics[0]
    ptm_vals = [m["ptm"] for m in all_metrics if m["ptm"] is not None]
    iptm_vals = [m["iptm"] for m in all_metrics if m["iptm"] is not None]
    rank_vals = [m["ranking_score"] for m in all_metrics if m["ranking_score"] is not None]

    return {
        "n_seeds": len(all_metrics),
        "ptm": primary["ptm"],
        "iptm": primary["iptm"],
        "ranking_score": primary["ranking_score"],
        "fraction_disordered": primary["fraction_disordered"],
        "has_clash": primary["has_clash"],
        "ptm_mean": statistics.mean(ptm_vals) if ptm_vals else None,
        "ptm_std": statistics.stdev(ptm_vals) if len(ptm_vals) > 1 else None,
        "ranking_score_mean": statistics.mean(rank_vals) if rank_vals else None,
    }


def collect_af3_metrics(af3_dir: Path) -> pd.DataFrame:
    """Scan *af3_dir* for AF3 output folders and return a metrics DataFrame.

    Parameters
    ----------
    af3_dir : Path
        Directory containing AF3 output sub-folders.

    Returns
    -------
    pd.DataFrame
        One row per AF3 prediction folder with columns:
        folder, condition, n_seeds, ptm, iptm, ranking_score,
        fraction_disordered, has_clash, ptm_mean, ptm_std,
        ranking_score_mean.
    """
    rows = []
    for entry in sorted(af3_dir.iterdir()):
        if not entry.is_dir():
            continue
        metrics = _extract_metrics_from_folder(entry)
        if metrics is None:
            continue
        condition = _parse_condition_from_folder(entry.name)
        rows.append({"folder": entry.name, "condition": condition, **metrics})

    return pd.DataFrame(rows)


def plot_af3_summary(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """Two-panel figure: pTM + ranking score by condition."""
    if df.empty:
        print("No AF3 metrics to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    conditions = df["condition"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(conditions), 3)))
    color_map = dict(zip(conditions, colors))

    for _, row in df.iterrows():
        c = color_map.get(row["condition"], "#888888")
        ax1.bar(row["folder"], row["ptm"] or 0, color=c, label=row["condition"])
        ax2.bar(row["folder"], row["ranking_score"] or 0, color=c)

    ax1.set_ylabel("pTM")
    ax1.set_title("Predicted TM-score")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    ax2.set_ylabel("Ranking score")
    ax2.set_title("AF3 ranking score")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)

    # De-duplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    if unique:
        ax1.legend(*zip(*unique), frameon=False, fontsize=8)

    fig.suptitle("AF3 validation summary", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")

    plt.show()
