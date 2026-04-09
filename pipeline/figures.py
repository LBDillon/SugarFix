"""Visualisations for SugarFix vs vanilla ProteinMPNN.

All plots scale to any number of designs (8, 32, 64, 128) and any number of
sites (1 -> 12+). Use `make_all(retention_csv, out_dir)` to produce the full
set, or call individual builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----- Palettes ------------------------------------------------------------

PALETTE = {
    "designer": "#5A9E8C",
    "soft":     "#D88766",
    "glycan":   "#C9A14A",
    "polar":    "#C7D9E8",
    "accent":   "#444444",
    "bg":       "#FAFAFA",
    "grid":     "#DCDCDC",
    "lost":     "#D88766",
    "kept":     "#5A9E8C",
}
# Backward-compat: some callers still pass palette_key="C".
PALETTES = {"C": PALETTE}

CONDITION_LABEL = {
    "designer_selected": "SugarFix (evidence-aware)",
    "soft_filter":       "ProteinMPNN baseline",
}

# Categorise the 20 amino acids for the substitution-aggregate panel.
AA_GROUPS = {
    "N (kept)":  list("N"),
    "Acidic":    list("DE"),
    "Basic":     list("KRH"),
    "Polar":     list("STQ"),
    "Aromatic":  list("FYW"),
    "Hydrophobic": list("AVILMC"),
    "Special":   list("GP"),
}


def apply_style(palette_key=None):
    """Apply the SugarFix matplotlib style. Safe to call repeatedly."""
    pal = PALETTE
    plt.rcParams.update({
        "figure.facecolor": pal["bg"],
        "axes.facecolor":   pal["bg"],
        "axes.edgecolor":   pal["accent"],
        "axes.labelcolor":  pal["accent"],
        "xtick.color":      pal["accent"],
        "ytick.color":      pal["accent"],
        "text.color":       pal["accent"],
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })


_apply_style = apply_style  # backward-compat alias used internally


def _finalize(fig, out_path: Optional[Path]) -> Optional[Path]:
    """Save (if path given) and display the figure exactly once."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.show()
    return out_path


def _conditions_to_show(df: pd.DataFrame) -> list:
    """Return which conditions are worth plotting.

    Drops `designer_selected` when it perfectly preserves every site (100% exact
    retention) — that panel is redundant when SugarFix is hitting its plan.
    """
    conds = []
    for cond in ("designer_selected", "soft_filter"):
        sub = df[df["design_condition"] == cond]
        if sub.empty:
            continue
        if cond == "designer_selected" and sub["exact_match"].all():
            continue
        conds.append(cond)
    return conds


# ----- Data prep -----------------------------------------------------------

def load_retention(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"design_condition", "site_label", "wt_motif", "design_motif",
              "exact_match", "functional", "design_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"retention csv missing columns: {missing}")
    return df


def per_site_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of designs (any N) where the central N is retained per site."""
    # exact_match captures wt-motif preservation; we report N-position retention.
    grouped = df.groupby(["design_condition", "site_label"]).agg(
        n_designs=("design_id", "nunique"),
        n_kept=("exact_match", "sum"),
    ).reset_index()
    grouped["retention"] = grouped["n_kept"] / grouped["n_designs"]
    return grouped


def substitution_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of central-position residues per condition, aggregated over all sites."""
    df = df.copy()
    df["central"] = df["design_motif"].str[0]
    counts = df.groupby(["design_condition", "central"]).size().reset_index(name="n")
    totals = counts.groupby("design_condition")["n"].transform("sum")
    counts["frac"] = counts["n"] / totals
    return counts


# ----- Individual visuals --------------------------------------------------

def plot_per_site_retention(df: pd.DataFrame,
                            out_path: Optional[Path] = None, ax=None):
    pal = PALETTE
    apply_style()
    summary = per_site_retention(df)

    sites = sorted(summary["site_label"].unique())
    conds = _conditions_to_show(df) or ["designer_selected", "soft_filter"]
    width = 0.38
    x = np.arange(len(sites))

    embedded = ax is not None
    if not embedded:
        w = max(5, min(14, 0.9 * len(sites) + 2.5))
        fig, ax = plt.subplots(figsize=(w, 4))
    else:
        fig = ax.figure

    for i, cond in enumerate(conds):
        sub = summary[summary["design_condition"] == cond].set_index("site_label")
        vals = [sub.loc[s, "retention"] if s in sub.index else 0 for s in sites]
        color = pal["designer"] if cond == "designer_selected" else pal["soft"]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color,
               label=CONDITION_LABEL[cond], edgecolor=pal["accent"], linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=30 if len(sites) <= 8 else 60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sequon retention rate")
    ax.set_title("Sequon retention per site")
    ax.axhline(0, color=pal["accent"], linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              frameon=False, fontsize=9)
    if embedded:
        return fig
    fig.tight_layout()
    _finalize(fig, out_path)
    return out_path


def plot_substitution_stack(df: pd.DataFrame, out_path: Path):
    """Aggregate stacked bar: what does each condition put at the central position?"""
    pal = PALETTE
    apply_style()
    counts = substitution_distribution(df)

    conds = _conditions_to_show(df)
    if not conds:
        return out_path
    group_colors = {
        "N (kept)":   pal["kept"],
        "Acidic":     pal["soft"],
        "Basic":      pal["glycan"],
        "Polar":      pal["polar"],
        "Aromatic":   "#B89B7A",
        "Hydrophobic": "#9B9B9B",
        "Special":    "#C2B8A3",
    }

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bottoms = {c: 0.0 for c in conds}
    for group, members in AA_GROUPS.items():
        fracs = []
        for c in conds:
            sub = counts[(counts["design_condition"] == c)
                         & (counts["central"].isin(members))]
            fracs.append(sub["frac"].sum())
        ax.barh([CONDITION_LABEL[c] for c in conds],
                fracs, left=[bottoms[c] for c in conds],
                color=group_colors[group], edgecolor=pal["accent"],
                linewidth=0.6, label=group)
        for i, c in enumerate(conds):
            bottoms[c] += fracs[i]

    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of central-position residues across all sites & designs")
    ax.set_title("What replaces the glycan asparagine?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    return _finalize(fig, out_path)


def plot_design_x_site_heatmap(df: pd.DataFrame, out_path: Path):
    """Per-protein heatmap: rows = designs, cols = sites, cell = kept/lost.

    Auto-sizes to design count (8/32/64/128) and site count (1..N).
    Drops the SugarFix panel if it is 100% retained (uninformative).
    """
    pal = PALETTE
    apply_style()
    sites = sorted(df["site_label"].unique())
    conds = _conditions_to_show(df)
    if not conds:
        return out_path

    fig_w = max(5, min(16, 0.5 * len(sites) * len(conds) + 3))
    n_designs = max(df["design_id"].nunique() // 2, 8)
    fig_h = max(3, min(14, 0.18 * n_designs + 2))
    fig, axes = plt.subplots(1, len(conds), figsize=(fig_w, fig_h), sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, cond in zip(axes, conds):
        sub = df[df["design_condition"] == cond]
        designs = sorted(sub["design_id"].unique(),
                         key=lambda d: int(d.rsplit("sample", 1)[-1])
                         if "sample" in d else 0)
        mat = np.zeros((len(designs), len(sites)))
        for i, d in enumerate(designs):
            for j, s in enumerate(sites):
                row = sub[(sub["design_id"] == d) & (sub["site_label"] == s)]
                if not row.empty:
                    mat[i, j] = 1.0 if bool(row["exact_match"].iloc[0]) else 0.0

        cmap = plt.matplotlib.colors.ListedColormap([pal["lost"], pal["kept"]])
        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(sites,
                           rotation=30 if len(sites) <= 8 else 60, ha="right")
        # Sparse y-ticks for large N
        step = max(1, len(designs) // 12)
        ax.set_yticks(range(0, len(designs), step))
        ax.set_yticklabels([str(i + 1) for i in range(0, len(designs), step)],
                           fontsize=7)
        ax.set_title(CONDITION_LABEL[cond])
        ax.set_xlabel("Site")

    axes[0].set_ylabel("Design index")
    handles = [
        plt.matplotlib.patches.Patch(color=pal["kept"], label="N retained"),
        plt.matplotlib.patches.Patch(color=pal["lost"], label="N replaced"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    fig.suptitle("Per-design site retention", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _finalize(fig, out_path)


def plot_central_residue_logo(df: pd.DataFrame, out_path: Path):
    """Stacked-frequency strip per site (works for any N_sites and N_designs)."""
    pal = PALETTE
    apply_style()
    df = df.copy()
    df["central"] = df["design_motif"].str[0]
    sites = sorted(df["site_label"].unique())
    conds = _conditions_to_show(df)
    if not conds:
        return out_path

    fig_w = max(5, min(14, 0.7 * len(sites) + 3))
    fig, axes = plt.subplots(1, len(conds), figsize=(fig_w, 4), sharey=True,
                             squeeze=False)
    axes = axes[0]

    group_colors = {
        "N (kept)":  pal["kept"],
        "Acidic":    pal["soft"],
        "Basic":     pal["glycan"],
        "Polar":     pal["polar"],
        "Aromatic":  "#B89B7A",
        "Hydrophobic": "#9B9B9B",
        "Special":   "#C2B8A3",
    }

    def aa_to_group(aa):
        for g, members in AA_GROUPS.items():
            if aa in members:
                return g
        return "Special"

    for ax, cond in zip(axes, conds):
        sub = df[df["design_condition"] == cond]
        x = np.arange(len(sites))
        bottoms = np.zeros(len(sites))
        # stack groups in fixed order
        for group in AA_GROUPS:
            heights = []
            for s in sites:
                site_rows = sub[sub["site_label"] == s]
                total = max(1, len(site_rows))
                hits = site_rows["central"].apply(aa_to_group).eq(group).sum()
                heights.append(hits / total)
            ax.bar(x, heights, bottom=bottoms, color=group_colors[group],
                   edgecolor=pal["accent"], linewidth=0.4,
                   label=group if ax is axes[0] else None)
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels(sites,
                           rotation=30 if len(sites) <= 8 else 60, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(CONDITION_LABEL[cond])

    axes[0].set_ylabel("Fraction of designs")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Central-position residue distribution per site",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return _finalize(fig, out_path)


# ----- Driver --------------------------------------------------------------

def make_all(retention_csv: Path, out_dir: Path, **_ignored) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_retention(Path(retention_csv))

    return {
        "per_site_retention": plot_per_site_retention(
            df, out_dir / "per_site_retention.png"),
        "substitution_stack": plot_substitution_stack(
            df, out_dir / "substitution_stack.png"),
        "design_x_site": plot_design_x_site_heatmap(
            df, out_dir / "design_x_site.png"),
        "central_logo": plot_central_residue_logo(
            df, out_dir / "central_logo.png"),
    }
