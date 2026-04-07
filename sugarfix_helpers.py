from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

try:
    from IPython.display import HTML, clear_output, display
except Exception:
    HTML = None

    def clear_output(wait: bool = False) -> None:
        return None

    def display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)

try:
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except Exception:
    widgets = None
    WIDGETS_AVAILABLE = False


EVIDENCE_ORDER = ["experimental", "pdb_evidence", "curator_inferred", "motif_only"]
EVIDENCE_LABELS = {
    "experimental": "Experimental",
    "pdb_evidence": "PDB evidence",
    "curator_inferred": "Curator inferred",
    "motif_only": "Motif only",
}
EVIDENCE_COLORS = {
    "experimental": "#0b6e4f",
    "pdb_evidence": "#2a9d8f",
    "curator_inferred": "#e9c46a",
    "motif_only": "#bdbdbd",
}

POLICY_ORDER = ["full_sequon", "functional_preserve", "soft_filter", "ignore"]
POLICY_LABELS = {
    "full_sequon": "Full sequon",
    "functional_preserve": "Functional preserve",
    "soft_filter": "Soft filter",
    "ignore": "Ignore",
}
POLICY_COLORS = {
    "full_sequon": "#264653",
    "functional_preserve": "#287271",
    "soft_filter": "#f4a261",
    "ignore": "#bdbdbd",
}

# Only designer_selected + soft_filter baseline (constrained baselines removed
# because they always show 100% retention by construction).
CONDITION_LABELS = {
    "designer_selected": "Your plan",
    "soft_filter": "Baseline: no constraints",
}
CONDITION_COLORS = {
    "designer_selected": "#005f73",
    "soft_filter": "#bb3e03",
}

POLICY_DEFAULTS = {
    "experimental": "full_sequon",
    "pdb_evidence": "full_sequon",
    "curator_inferred": "functional_preserve",
    "motif_only": "soft_filter",
}


@dataclass
class GlycoSite:
    chain: str
    position_0idx: int
    position_1idx: int
    motif: str
    evidence_tier: str = "motif_only"
    pdb_resnum: Optional[int] = None
    glycan_tree: Optional[dict] = None
    default_policy: str = "soft_filter"
    evidence_reasons: List[str] = field(default_factory=list)
    uniprot_position: Optional[int] = None
    uniprot_tier: Optional[str] = None
    expected_tier: Optional[str] = None
    evidence_ok: bool = True

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.position_1idx}"


@dataclass
class SiteStatus:
    site: GlycoSite
    design_triplet: str
    n_retained: bool
    exact_match: bool
    functional: bool
    selected_policy: str
    required_for_plan: bool
    meets_selected_policy: bool


@dataclass
class DesignResult:
    design_condition: str
    design_id: str
    sequence: str
    chain_sequences: Dict[str, str] = field(default_factory=dict)
    mpnn_score: float = 0.0
    sample_idx: int = 0
    site_statuses: List[SiteStatus] = field(default_factory=list)
    passes_selected_plan: bool = True
    n_sites_satisfied: int = 0
    n_sites_required: int = 0
    plan_satisfaction_rate: float = 0.0


@dataclass
class ConstraintBundle:
    condition_name: str
    description: str
    fixed_positions_0idx: Dict[str, List[int]] = field(default_factory=dict)
    site_policies: Dict[str, str] = field(default_factory=dict)


@dataclass
class DesignSession:
    """Single mutable object that carries state between notebook cells."""

    pdb_id: str = ""
    run_label: str = "interactive_designer"
    num_seqs: int = 8
    sampling_temp: float = 0.1
    seed: int = 42

    # Paths
    pipeline_root: Optional[Path] = None
    data_dir: Optional[Path] = None
    structure_dir: Optional[Path] = None
    output_root: Optional[Path] = None
    run_dir: Optional[Path] = None
    figure_dir: Optional[Path] = None
    af3_dir: Optional[Path] = None

    # Designer decisions
    target_role: str = ""
    selected_strategy: str = ""
    run_baselines: bool = True
    selected_site_policies: Dict[str, str] = field(default_factory=dict)
    confirmed: bool = False

    # Computed state
    glyco_sites: List[GlycoSite] = field(default_factory=list)
    chain_seqs: Dict[str, str] = field(default_factory=dict)
    chain_order: List[str] = field(default_factory=list)
    assessment: dict = field(default_factory=dict)
    decision_df: Optional[pd.DataFrame] = None
    site_order: List[str] = field(default_factory=list)

    def setup_paths(self) -> None:
        root = self.pipeline_root
        self.data_dir = root / "data"
        self.structure_dir = self.data_dir / "prep" / self.pdb_id / "structure"
        self.output_root = self.data_dir / "outputs" / f"output_{self.pdb_id}"
        self.run_dir = self.output_root / self.run_label
        self.figure_dir = self.run_dir / "figures"
        self.af3_dir = self.run_dir / "af3"

    def apply_decisions(
        self,
        target_role: str,
        selected_strategy: str,
        run_baselines: bool,
        site_policies: Dict[str, str],
    ) -> None:
        self.target_role = target_role
        self.selected_strategy = selected_strategy
        self.run_baselines = run_baselines
        self.selected_site_policies = dict(site_policies)
        self.decision_df = build_decision_df(self.glyco_sites, self.selected_site_policies)
        self.site_order = (
            list(self.decision_df["site_label"]) if not self.decision_df.empty else []
        )
        self.confirmed = True

    @property
    def decision_payload(self) -> dict:
        return {
            "pdb_id": self.pdb_id,
            "target_role": self.target_role,
            "selected_strategy": self.selected_strategy,
            "run_baselines": self.run_baselines,
            "site_policies": dict(self.selected_site_policies),
        }


def display_df(df: pd.DataFrame, n: Optional[int] = None) -> None:
    if df is None:
        print("None")
        return
    if n is not None and len(df) > n:
        display(df.head(n))
        print(f"Showing first {n} of {len(df)} rows")
    else:
        display(df)


def policy_fixed_positions(site: GlycoSite, policy: str) -> List[int]:
    if policy == "full_sequon":
        return [site.position_0idx, site.position_0idx + 1, site.position_0idx + 2]
    if policy == "functional_preserve":
        return [site.position_0idx, site.position_0idx + 2]
    return []


def policy_requirement_met(exact_match: bool, functional: bool, policy: str) -> bool:
    if policy == "full_sequon":
        return exact_match
    if policy in {"functional_preserve", "soft_filter"}:
        return functional
    return True


def pdb_resnum_to_uniprot(
    chain_id: str,
    pdb_resnum: Optional[int],
    dbref_ranges: dict,
) -> Optional[int]:
    if pdb_resnum is None:
        return None
    for rng in dbref_ranges.get(chain_id, []):
        if rng["pdb_start"] <= pdb_resnum <= rng["pdb_end"]:
            return rng["unp_start"] + (pdb_resnum - rng["pdb_start"])
    return None


def snap_to_nearest_sequon(
    chain_seq: str,
    mpnn_idx: int,
    window: int = 50,
) -> Optional[int]:
    """Search for the closest N-X-S/T motif within +/- `window` of mpnn_idx.

    Returns the MPNN 0-indexed position of the N, or None if no motif found.
    Used to recover from UniProt<->PDB numbering drift.
    """
    # Permissive: allow X (unresolved residue) in the middle position. The
    # strict find_sequons regex excludes X, but for UniProt-only recovery we
    # want to catch sites where the middle residue is missing from the
    # structure (e.g. PDB gap inside the sequon).
    pattern = re.compile(r"N[^P][ST]")
    best = None
    best_dist = window + 1
    lo = max(0, mpnn_idx - window)
    hi = min(len(chain_seq), mpnn_idx + window + 3)
    for m in pattern.finditer(chain_seq, lo, hi):
        dist = abs(m.start() - mpnn_idx)
        if dist < best_dist:
            best_dist = dist
            best = m.start()
    return best


def build_glyco_site(
    chain: str,
    mpnn_idx: int,
    chain_seqs: Dict[str, str],
    pdb_to_mpnn_by_chain: Dict[str, Dict[int, int]],
    dbref_ranges: dict,
    uniprot_evidence: Dict[int, str],
    glycan_trees: dict,
    source: str = "motif",
) -> Optional["GlycoSite"]:
    """Construct a GlycoSite for a given chain + MPNN 0-indexed position.

    `source` is one of: "motif" (regex hit), "uniprot_only" (UniProt CARBOHYD
    that the regex missed), or "user_specified" (manual entry).
    """
    chain_seq = chain_seqs.get(chain, "")
    if mpnn_idx < 0 or mpnn_idx + 3 > len(chain_seq):
        return None

    triplet = chain_seq[mpnn_idx:mpnn_idx + 3]
    pdb_to_mpnn = pdb_to_mpnn_by_chain.get(chain, {})
    mpnn_to_pdb = {v: k for k, v in pdb_to_mpnn.items()}
    pdb_resnum = mpnn_to_pdb.get(mpnn_idx)

    tree = glycan_trees.get(f"{chain}:{pdb_resnum}") if pdb_resnum is not None else None
    uniprot_position = pdb_resnum_to_uniprot(chain, pdb_resnum, dbref_ranges)
    uniprot_tier = uniprot_evidence.get(uniprot_position) if uniprot_position is not None else None
    expected_tier = infer_expected_tier(uniprot_tier, tree is not None)

    # Tier assignment for this site
    if uniprot_tier is not None:
        evidence_tier = uniprot_tier
    elif tree is not None:
        evidence_tier = "pdb_evidence"
    else:
        evidence_tier = "motif_only"

    reasons = []
    if tree is not None:
        reasons.append(f"resolved glycan tree ({tree.get('n_sugars', '?')} sugars)")
    if uniprot_tier is not None:
        reasons.append(f"UniProt {uniprot_tier} at position {uniprot_position}")
    if source == "uniprot_only":
        reasons.append("not detected as N-X-S/T motif in chain sequence")
    elif source == "user_specified":
        reasons.append("user-specified site")
    if not reasons:
        reasons.append("motif-only regex match")

    return GlycoSite(
        chain=chain,
        position_0idx=mpnn_idx,
        position_1idx=mpnn_idx + 1,
        motif=triplet,
        evidence_tier=evidence_tier,
        pdb_resnum=pdb_resnum,
        glycan_tree=tree,
        default_policy=POLICY_DEFAULTS[evidence_tier],
        evidence_reasons=reasons,
        uniprot_position=uniprot_position,
        uniprot_tier=uniprot_tier,
        expected_tier=expected_tier,
        evidence_ok=(expected_tier == evidence_tier),
    )


def infer_expected_tier(uniprot_tier: Optional[str], has_glycan_tree: bool) -> str:
    if uniprot_tier == "experimental":
        return "experimental"
    if has_glycan_tree or uniprot_tier == "pdb_evidence":
        return "pdb_evidence"
    if uniprot_tier == "curator_inferred":
        return "curator_inferred"
    return "motif_only"


def summarize_glycoprotein_status(sites: List[GlycoSite]) -> dict:
    n_sites = len(sites)
    validated_sites = sum(site.evidence_tier != "motif_only" for site in sites)
    glycan_tree_sites = sum(site.glycan_tree is not None for site in sites)
    experimental_sites = sum(site.evidence_tier == "experimental" for site in sites)
    motif_only_sites = sum(site.evidence_tier == "motif_only" for site in sites)

    if glycan_tree_sites > 0:
        headline = (
            "Resolved glycans are present in the structure, so this is strong "
            "glycoprotein evidence."
        )
        recommended_role = "glycoprotein"
        confidence = "strong"
    elif validated_sites > 0:
        headline = (
            "External annotations support glycosylation even without resolved "
            "glycans in the PDB."
        )
        recommended_role = "glycoprotein"
        confidence = "strong"
    elif n_sites > 0:
        headline = (
            "Only motif-level evidence is available, so treat glycosylation as "
            "a design hypothesis to confirm."
        )
        recommended_role = "uncertain"
        confidence = "possible"
    else:
        headline = "No N-linked sequons were found in the parsed protein chains."
        recommended_role = "not_glycoprotein"
        confidence = "unlikely"

    return {
        "n_sites": n_sites,
        "validated_sites": validated_sites,
        "glycan_tree_sites": glycan_tree_sites,
        "experimental_sites": experimental_sites,
        "motif_only_sites": motif_only_sites,
        "headline": headline,
        "recommended_role": recommended_role,
        "confidence": confidence,
    }


def build_site_policy_map(
    sites: List[GlycoSite],
    strategy: str,
    interactive: bool = True,
) -> Dict[str, str]:
    if strategy == "evidence_aware":
        return {site.label: site.default_policy for site in sites}
    if strategy in {"full_sequon", "functional_preserve", "soft_filter", "ignore"}:
        return {site.label: strategy for site in sites}
    if strategy == "mixed_custom":
        return {site.label: site.default_policy for site in sites}
    raise ValueError(f"Unknown strategy: {strategy}")


def build_constraint_bundle(
    sites: List[GlycoSite],
    site_policies: Dict[str, str],
    condition_name: str,
    description: str,
) -> ConstraintBundle:
    fixed_positions = {}
    for site in sites:
        policy = site_policies.get(site.label, site.default_policy)
        positions = policy_fixed_positions(site, policy)
        if positions:
            fixed_positions.setdefault(site.chain, []).extend(positions)

    for chain_id in fixed_positions:
        fixed_positions[chain_id] = sorted(set(fixed_positions[chain_id]))

    return ConstraintBundle(
        condition_name=condition_name,
        description=description,
        fixed_positions_0idx=fixed_positions,
        site_policies=dict(site_policies),
    )


def build_condition_manifest(
    sites: List[GlycoSite],
    bundles: Dict[str, ConstraintBundle],
) -> pd.DataFrame:
    rows = []
    for condition_name, bundle in bundles.items():
        policy_counts = {policy: 0 for policy in POLICY_ORDER}
        for site in sites:
            selected = bundle.site_policies.get(site.label, site.default_policy)
            policy_counts[selected] += 1

        rows.append(
            {
                "design_condition": condition_name,
                "label": CONDITION_LABELS.get(condition_name, condition_name),
                "description": bundle.description,
                "required_sites": sum(
                    count for policy, count in policy_counts.items() if policy != "ignore"
                ),
                "full_sequon_sites": policy_counts["full_sequon"],
                "functional_preserve_sites": policy_counts["functional_preserve"],
                "soft_filter_sites": policy_counts["soft_filter"],
                "ignored_sites": policy_counts["ignore"],
                "n_fixed_positions": sum(
                    len(positions) for positions in bundle.fixed_positions_0idx.values()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_decision_df(
    sites: List[GlycoSite],
    selected_site_policies: Dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_label": site.label,
                "chain": site.chain,
                "pos_1idx": site.position_1idx,
                "motif": site.motif,
                "evidence_tier": site.evidence_tier,
                "has_glycan_tree": site.glycan_tree is not None,
                "default_policy": site.default_policy,
                "selected_policy": selected_site_policies.get(
                    site.label, site.default_policy
                ),
                "required_for_plan": selected_site_policies.get(
                    site.label, site.default_policy
                )
                != "ignore",
                "reason": "; ".join(site.evidence_reasons),
            }
            for site in sites
        ]
    )


def parse_mpnn_score(header: str) -> Tuple[int, float]:
    sample_match = re.search(r"sample=(\d+)", header)
    score_match = re.search(r"score=([\d.]+)", header)
    sample = int(sample_match.group(1)) if sample_match else 0
    score = float(score_match.group(1)) if score_match else np.nan
    return sample, score


def score_designs(
    fa_path: Path,
    sites: List[GlycoSite],
    chain_order: List[str],
    selected_site_policies: Dict[str, str],
    design_condition: str,
    read_fasta_sequences,
    split_mpnn_concat_seq,
    is_functional_sequon,
) -> List[DesignResult]:
    records = read_fasta_sequences(fa_path)
    if len(records) < 2:
        return []

    results = []
    for header, seq in records[1:]:
        sample_idx, score = parse_mpnn_score(header)
        chain_sequences = split_mpnn_concat_seq(seq, chain_order)
        site_statuses = []

        for site in sites:
            chain_seq = chain_sequences.get(site.chain, "")
            if site.position_0idx + 3 > len(chain_seq):
                continue

            design_triplet = chain_seq[site.position_0idx:site.position_0idx + 3]
            n_retained = design_triplet[0] == "N"
            exact_match = design_triplet == site.motif
            functional = is_functional_sequon(design_triplet)
            selected_policy = selected_site_policies.get(site.label, site.default_policy)
            required_for_plan = selected_policy != "ignore"
            meets_selected_policy = policy_requirement_met(
                exact_match,
                functional,
                selected_policy,
            )

            site_statuses.append(
                SiteStatus(
                    site=site,
                    design_triplet=design_triplet,
                    n_retained=n_retained,
                    exact_match=exact_match,
                    functional=functional,
                    selected_policy=selected_policy,
                    required_for_plan=required_for_plan,
                    meets_selected_policy=meets_selected_policy,
                )
            )

        required_statuses = [status for status in site_statuses if status.required_for_plan]
        n_sites_required = len(required_statuses)
        n_sites_satisfied = sum(status.meets_selected_policy for status in required_statuses)
        passes_selected_plan = all(status.meets_selected_policy for status in required_statuses)
        plan_satisfaction_rate = (
            n_sites_satisfied / n_sites_required if n_sites_required else 1.0
        )

        results.append(
            DesignResult(
                design_condition=design_condition,
                design_id=f"{design_condition}_sample{sample_idx}",
                sequence=seq,
                chain_sequences=chain_sequences,
                mpnn_score=score,
                sample_idx=sample_idx,
                site_statuses=site_statuses,
                passes_selected_plan=passes_selected_plan,
                n_sites_satisfied=n_sites_satisfied,
                n_sites_required=n_sites_required,
                plan_satisfaction_rate=plan_satisfaction_rate,
            )
        )

    return results


def select_top_design(results: List[DesignResult]) -> Optional[DesignResult]:
    if not results:
        return None
    passing = [result for result in results if result.passes_selected_plan]
    if passing:
        return min(passing, key=lambda result: result.mpnn_score)
    return sorted(
        results,
        key=lambda result: (-result.n_sites_satisfied, result.mpnn_score),
    )[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_site_strategy_overview(
    decision_df: pd.DataFrame,
    chain_seqs: Dict[str, str],
    assessment: dict,
    output_path: Optional[Path] = None,
) -> None:
    if decision_df.empty:
        print("No glycosites found, so there is no site strategy plot to draw.")
        return

    ordered_df = decision_df.sort_values(["chain", "pos_1idx"]).reset_index(drop=True)
    chains = list(chain_seqs)

    fig = plt.figure(figsize=(15, max(5, 1.0 * len(ordered_df) + 2)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_track = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])

    for idx, chain_id in enumerate(chains):
        y = len(chains) - idx
        chain_len = len(chain_seqs[chain_id])
        ax_track.hlines(y, 1, chain_len, color="#d9d9d9", linewidth=6)
        ax_track.text(0, y, f"Chain {chain_id}", ha="right", va="center", fontsize=10)

        sub = ordered_df[ordered_df["chain"] == chain_id]
        for _, row in sub.iterrows():
            ax_track.scatter(
                row["pos_1idx"],
                y,
                s=180 if row["has_glycan_tree"] else 120,
                c=EVIDENCE_COLORS[row["evidence_tier"]],
                edgecolors=POLICY_COLORS[row["selected_policy"]],
                linewidths=2.5,
                zorder=3,
            )
            ax_track.text(
                row["pos_1idx"],
                y + 0.16,
                row["site_label"],
                fontsize=8,
                rotation=45,
                ha="left",
                va="bottom",
            )

    ax_track.set_title("Where the sequons are on the target")
    ax_track.set_xlabel("Residue index within chain")
    ax_track.set_yticks([])
    ax_track.set_ylim(0.4, len(chains) + 0.8)
    ax_track.spines[["left", "right", "top"]].set_visible(False)

    evidence_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=9,
            markerfacecolor=EVIDENCE_COLORS[key], markeredgecolor="black",
            label=EVIDENCE_LABELS[key],
        )
        for key in EVIDENCE_ORDER
    ]
    policy_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=9,
            markerfacecolor="white", markeredgewidth=2.5,
            markeredgecolor=POLICY_COLORS[key], label=POLICY_LABELS[key],
        )
        for key in POLICY_ORDER
    ]
    ax_track.legend(
        handles=evidence_handles + policy_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, frameon=False, fontsize=9,
    )

    matrix_y = np.arange(len(ordered_df))[::-1]
    ax_matrix.scatter(
        np.zeros(len(ordered_df)), matrix_y, s=180,
        c=[EVIDENCE_COLORS[val] for val in ordered_df["evidence_tier"]],
        edgecolors="black", linewidths=0.6,
    )
    ax_matrix.scatter(
        np.ones(len(ordered_df)), matrix_y, s=180,
        c=[POLICY_COLORS[val] for val in ordered_df["selected_policy"]],
        edgecolors="black", linewidths=0.6,
    )

    glycan_markers = ["s" if flag else "x" for flag in ordered_df["has_glycan_tree"]]
    for y, marker in zip(matrix_y, glycan_markers):
        ax_matrix.scatter(2, y, s=100, c="#404040", marker=marker)

    ax_matrix.set_xlim(-0.7, 2.5)
    ax_matrix.set_xticks([0, 1, 2])
    ax_matrix.set_xticklabels(["Evidence", "Selected\npolicy", "Resolved\nglycan"])
    ax_matrix.set_yticks(matrix_y)
    ax_matrix.set_yticklabels(
        [f"{row.site_label} {row.motif}" for row in ordered_df.itertuples()],
        fontsize=9,
    )
    ax_matrix.set_title("Why SugarFix is protecting each site")
    ax_matrix.grid(axis="x", alpha=0.2)
    ax_matrix.spines[["right", "top"]].set_visible(False)

    fig.suptitle(
        f"SugarFix site guide - {assessment['headline']}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()


def plot_design_dashboard(
    condition_summary_df: pd.DataFrame,
    all_results: Dict[str, List[DesignResult]],
    top_designs: Dict[str, Optional[DesignResult]],
    site_order: List[str],
    output_path: Optional[Path] = None,
) -> None:
    if condition_summary_df.empty:
        print("No design summary available yet.")
        return

    condition_order = list(condition_summary_df["design_condition"])
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[0, 2])

    # --- Left panel: retention metrics (these vary meaningfully) ---
    x = np.arange(len(condition_order))
    width = 0.3
    ax_bar.bar(
        x - width / 2,
        condition_summary_df["mean_functional_retention"] * 100,
        width, color="#90caf9", label="Functional sequon retained",
    )
    ax_bar.bar(
        x + width / 2,
        condition_summary_df["mean_exact_retention"] * 100,
        width, color="#66bb6a", label="Exact wild-type triplet",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [CONDITION_LABELS.get(name, name) for name in condition_order],
        rotation=20, ha="right",
    )
    ax_bar.set_ylabel("Retention (%)")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title("Glycosylation site retention")
    ax_bar.legend(frameon=False, fontsize=9)

    # --- Middle panel: score vs sites satisfied ---
    for condition in condition_order:
        results = all_results.get(condition, [])
        if not results:
            continue
        ax_scatter.scatter(
            [result.mpnn_score for result in results],
            [result.n_sites_satisfied for result in results],
            s=[80 if result.passes_selected_plan else 45 for result in results],
            alpha=0.75,
            color=CONDITION_COLORS.get(condition, "#666666"),
            edgecolors="black", linewidths=0.4,
            label=CONDITION_LABELS.get(condition, condition),
        )
    ax_scatter.set_xlabel("MPNN score (lower is better)")
    ax_scatter.set_ylabel("Required sites satisfied")
    ax_scatter.set_title("Design quality vs site preservation")
    ax_scatter.legend(frameon=False, fontsize=8)

    # --- Right panel: per-site heatmap of top designs ---
    if site_order:
        matrix = np.full((len(condition_order), len(site_order)), np.nan)
        annotations = np.empty((len(condition_order), len(site_order)), dtype=object)
        annotations[:] = ""

        for row_idx, condition in enumerate(condition_order):
            top_design = top_designs.get(condition)
            if top_design is None:
                continue
            status_map = {status.site.label: status for status in top_design.site_statuses}
            for col_idx, site_label in enumerate(site_order):
                status = status_map.get(site_label)
                if status is None:
                    continue
                if status.functional:
                    matrix[row_idx, col_idx] = 1.0
                elif status.n_retained:
                    matrix[row_idx, col_idx] = 0.5
                else:
                    matrix[row_idx, col_idx] = 0.0
                annotations[row_idx, col_idx] = status.design_triplet

        cmap = ListedColormap(["#c62828", "#ffb74d", "#2e7d32"])
        norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.1], cmap.N)
        sns.heatmap(
            matrix, ax=ax_heat, cmap=cmap, norm=norm,
            annot=annotations, fmt="", cbar=False,
            linewidths=0.5, linecolor="white",
            xticklabels=site_order,
            yticklabels=[CONDITION_LABELS.get(name, name) for name in condition_order],
        )
        legend_handles = [
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#2e7d32", label="Functional sequon"),
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#ffb74d", label="N retained only"),
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#c62828", label="Sequon lost"),
        ]
        ax_heat.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8,
        )
        ax_heat.set_title("Per-site outcome (top design)")
        ax_heat.set_xlabel("Site")
        ax_heat.set_ylabel("")
    else:
        ax_heat.text(0.5, 0.5, "No glycosites to display", ha="center", va="center")
        ax_heat.axis("off")

    fig.suptitle("SugarFix design dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()


def make_af3_server_json(
    name: str,
    chain_sequences: List[str],
    glycan_positions: Optional[Dict[int, List[dict]]] = None,
) -> List[dict]:
    sequences = []
    for chain_idx, sequence in enumerate(chain_sequences):
        if "X" in sequence:
            x_positions = [i + 1 for i, aa in enumerate(sequence) if aa == "X"]
            print(
                f"  Note: chain {chain_idx} has {len(x_positions)} unresolved "
                f"residue(s) at position(s) {x_positions}; substituting 'G' "
                f"so AF3 server accepts the sequence."
            )
            sequence = sequence.replace("X", "G")
        chain_payload = {"sequence": sequence, "count": 1}
        if glycan_positions and glycan_positions.get(chain_idx):
            chain_payload["glycans"] = sorted(
                glycan_positions[chain_idx],
                key=lambda record: record["position"],
            )
        sequences.append({"proteinChain": chain_payload})

    return [
        {
            "name": name,
            "modelSeeds": [],
            "sequences": sequences,
            "dialect": "alphafoldserver",
            "version": 1,
        }
    ]
