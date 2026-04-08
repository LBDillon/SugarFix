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
# Palette C — "Okabe-Ito soft" — desaturated, colorblind-safe.
EVIDENCE_COLORS = {
    "experimental":     "#5A9E8C",  # sage green
    "pdb_evidence":     "#7BB6A6",  # lighter sage
    "curator_inferred": "#C9A14A",  # warm gold
    "motif_only":       "#B8B8B8",  # neutral gray
}

POLICY_ORDER = ["full_sequon", "functional_preserve", "soft_filter", "ignore"]
POLICY_LABELS = {
    "full_sequon": "Full sequon",
    "functional_preserve": "Functional preserve",
    "soft_filter": "Soft filter",
    "ignore": "Ignore",
}
POLICY_COLORS = {
    "full_sequon":         "#444444",  # accent dark
    "functional_preserve": "#5A9E8C",
    "soft_filter":         "#D88766",
    "ignore":              "#B8B8B8",
}

CONDITION_LABELS = {
    "designer_selected": "SugarFix (evidence-aware)",
    "soft_filter": "ProteinMPNN baseline",
}
CONDITION_COLORS = {
    "designer_selected": "#5A9E8C",
    "soft_filter":       "#D88766",
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
    run_label: str = ""
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
        self.output_root = self.data_dir / "outputs" / self.pdb_id
        self.run_dir = self.output_root / self.run_label if self.run_label else self.output_root
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

def _compute_dssp_per_chain(pdb_path: Path):
    """Return {chain_id: {seq_index_1based: (ss_letter, rel_sasa)}}.

    Tries Bio.PDB.DSSP first (needs the `mkdssp`/`dssp` binary). On failure
    returns {} so callers can render the plot without the SS/SASA overlay.
    """
    if pdb_path is None or not Path(pdb_path).exists():
        return {}
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except Exception:
        return {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", str(pdb_path))
        model = next(structure.get_models())
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        try:
            dssp = DSSP(model, str(pdb_path), dssp="dssp")
        except Exception:
            return {}
    out: Dict[str, Dict[int, tuple]] = {}
    for key in dssp.keys():
        chain_id, res_id = key
        record = dssp[key]
        ss = record[2] if len(record) > 2 else "-"
        sasa = record[3] if len(record) > 3 else None
        try:
            sasa_val = float(sasa) if sasa is not None else None
        except (TypeError, ValueError):
            sasa_val = None
        # res_id is (' ', resnum, icode); we key by resnum.
        resnum = res_id[1]
        out.setdefault(chain_id, {})[int(resnum)] = (ss, sasa_val)
    return out


def _ss_to_color(ss: str) -> str:
    if ss in ("H", "G", "I"):
        return "#5A9E8C"  # helix - sage
    if ss in ("E", "B"):
        return "#C9A14A"  # strand - gold
    return "#D0D0D0"      # loop / coil


def plot_site_strategy_overview(
    decision_df: pd.DataFrame,
    chain_seqs: Dict[str, str],
    assessment: dict,
    output_path: Optional[Path] = None,
    pdb_path: Optional[Path] = None,
) -> None:
    """Per-chain sequence track with SS / SASA overlay and sequon callouts.

    Chains with no detected sequons are still shown as muted "no sequons" rows
    so the user has a complete chain inventory.
    """
    if decision_df.empty:
        print("No glycosites found, so there is no site strategy plot to draw.")
        return

    ordered_df = decision_df.sort_values(["chain", "pos_1idx"]).reset_index(drop=True)
    chains = list(chain_seqs)
    if not chains:
        chains = list(ordered_df["chain"].unique())

    dssp_data = _compute_dssp_per_chain(Path(pdb_path)) if pdb_path else {}
    has_dssp = bool(dssp_data)

    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n_chains = len(chains)
    # Each chain gets ~1.6 inches of vertical space (track + label callouts).
    fig_h = max(3.0, 1.6 * n_chains + 1.6)
    fig_w = 14
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FAFAFA")
    gs = fig.add_gridspec(n_chains, 1, hspace=0.9)

    pos_to_resnum = {}  # chain -> {pos_1idx (mpnn) -> pdb resnum}
    for _, r in ordered_df.iterrows():
        if pd.notna(r.get("pdb_resnum")):
            pos_to_resnum.setdefault(r["chain"], {})[int(r["pos_1idx"])] = int(r["pdb_resnum"])

    sasa_max = 1.0
    if has_dssp:
        all_sasa = [v[1] for d in dssp_data.values() for v in d.values()
                    if v[1] is not None]
        if all_sasa:
            sasa_max = max(0.4, min(1.0, max(all_sasa)))

    cmap = plt.cm.get_cmap("YlGnBu_r")  # buried -> exposed
    sites_in_chain = {c: ordered_df[ordered_df["chain"] == c] for c in chains}

    for row_idx, chain_id in enumerate(chains):
        ax = fig.add_subplot(gs[row_idx, 0])
        chain_len = len(chain_seqs.get(chain_id, "")) or (
            int(ordered_df[ordered_df["chain"] == chain_id]["pos_1idx"].max()) + 5
            if not ordered_df[ordered_df["chain"] == chain_id].empty else 100
        )
        sub = sites_in_chain[chain_id]

        # SASA strip (under the SS bar)
        if has_dssp and chain_id in dssp_data:
            xs, colors = [], []
            for resnum in range(1, chain_len + 1):
                hit = dssp_data[chain_id].get(resnum)
                xs.append(resnum)
                if hit and hit[1] is not None:
                    colors.append(cmap(min(1.0, hit[1] / sasa_max)))
                else:
                    colors.append("#EEEEEE")
            # Draw SASA as thin colored bars
            ax.bar(xs, [0.25] * len(xs), bottom=-0.15, width=1.0,
                   color=colors, linewidth=0)
            # Draw SS as colored ribbon on top
            for resnum in range(1, chain_len + 1):
                hit = dssp_data[chain_id].get(resnum)
                ss_color = _ss_to_color(hit[0]) if hit else "#D0D0D0"
                ax.bar(resnum, 0.18, bottom=0.12, width=1.0,
                       color=ss_color, linewidth=0)
        else:
            # Fallback: flat gray ribbon
            ax.hlines(0.2, 1, chain_len, color="#D0D0D0", linewidth=6)

        # Sequon glyphs + staggered labels
        if not sub.empty:
            ys_glyph = 0.21
            label_rows = [0.55, 0.85]  # two-row stagger
            for i, (_, site) in enumerate(sub.sort_values("pos_1idx").iterrows()):
                x = site["pos_1idx"]
                ax.scatter([x], [ys_glyph], s=85,
                           c=EVIDENCE_COLORS.get(site["evidence_tier"], "#888"),
                           edgecolors="#222222", linewidths=0.6, zorder=5)
                yl = label_rows[i % 2]
                ax.plot([x, x], [ys_glyph + 0.03, yl - 0.02],
                        color="#666666", linewidth=0.5, zorder=4)
                ax.text(x, yl, f"{site['site_label']}\n{site['motif']}",
                        fontsize=7, ha="center", va="bottom",
                        color="#222222", zorder=6)
        else:
            ax.text(chain_len / 2, 0.55, "no sequons detected",
                    ha="center", va="center", fontsize=9, style="italic",
                    color="#999999")

        ax.set_xlim(-chain_len * 0.02, chain_len * 1.02)
        ax.set_ylim(-0.2, 1.15)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"Chain {chain_id}  ({chain_len} aa)",
                     loc="left", fontsize=10, fontweight="bold", pad=4,
                     color="#444444")
        if row_idx == n_chains - 1:
            ax.set_xlabel("Residue index", fontsize=9)

    # Legend (bottom)
    evidence_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=EVIDENCE_COLORS[key],
               markeredgecolor="#222222",
               label=EVIDENCE_LABELS[key])
        for key in EVIDENCE_ORDER
    ]
    ss_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor="#5A9E8C", markeredgecolor="none",
               label="α-helix"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor="#C9A14A", markeredgecolor="none",
               label="β-strand"),
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor="#D0D0D0", markeredgecolor="none",
               label="loop / coil"),
    ]
    legend_handles = evidence_handles + ss_handles
    legend_labels = [h.get_label() for h in legend_handles]
    fig.legend(legend_handles, legend_labels,
               loc="lower center", ncol=min(7, len(legend_handles)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    if has_dssp:
        # SASA colorbar in top-right
        cax = fig.add_axes([0.86, 0.93, 0.11, 0.015])
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=sasa_max))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Relative SASA (buried → exposed)", fontsize=7,
                     color="#444444")

    headline = assessment.get("headline", "") if isinstance(assessment, dict) else ""
    title = "Sequon map across chains"
    if headline:
        title += f"  ·  {headline}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
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
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(18, 6.0), facecolor="#FAFAFA")
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
        width, color="#7BB6A6", edgecolor="#444444", linewidth=0.6,
        label="Functional sequon retained",
    )
    ax_bar.bar(
        x + width / 2,
        condition_summary_df["mean_exact_retention"] * 100,
        width, color="#5A9E8C", edgecolor="#444444", linewidth=0.6,
        label="Exact wild-type triplet",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [CONDITION_LABELS.get(name, name) for name in condition_order],
        rotation=20, ha="right",
    )
    ax_bar.set_ylabel("Retention (%)")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title("Glycosylation site retention")
    ax_bar.legend(frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2)

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
    ax_scatter.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.22), ncol=1)

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

        cmap = ListedColormap(["#D88766", "#C9A14A", "#5A9E8C"])
        norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.1], cmap.N)
        sns.heatmap(
            matrix, ax=ax_heat, cmap=cmap, norm=norm,
            annot=annotations, fmt="", cbar=False,
            linewidths=0.6, linecolor="#FAFAFA",
            xticklabels=site_order,
            yticklabels=[CONDITION_LABELS.get(name, name) for name in condition_order],
            annot_kws={"color": "#222222", "fontsize": 9, "fontweight": "bold"},
        )
        # Rotate site labels harder if there are many sites.
        rot = 30 if len(site_order) <= 8 else 60
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=rot, ha="right")
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
        legend_handles = [
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#5A9E8C", markeredgecolor="#444444",
                   label="Functional sequon"),
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#C9A14A", markeredgecolor="#444444",
                   label="N retained only"),
            Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor="#D88766", markeredgecolor="#444444",
                   label="Sequon lost"),
        ]
        ax_heat.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8,
        )
        ax_heat.set_title("Per-site outcome (top design)")
        ax_heat.set_xlabel("Site")
        ax_heat.set_ylabel("")
    else:
        ax_heat.text(0.5, 0.5, "No glycosites to display", ha="center", va="center")
        ax_heat.axis("off")

    fig.suptitle("SugarFix design dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

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
