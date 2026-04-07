#!/usr/bin/env python3
"""
Analyze 3-condition ProteinMPNN design outputs for any PDB.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT / "data"
PROJECT_DIR = PIPELINE_ROOT.parent
PROTEINMPNN_DIR = Path(
    os.environ.get("PROTEINMPNN_DIR", str(PIPELINE_ROOT / "ProteinMPNN"))
).resolve()
if not (PROTEINMPNN_DIR / "protein_mpnn_utils.py").exists():
    fallback_dir = PROJECT_DIR / "ProteinMPNN"
    if (fallback_dir / "protein_mpnn_utils.py").exists():
        PROTEINMPNN_DIR = fallback_dir
    env_runner = os.environ.get("PROTEINMPNN_PATH")
    if env_runner:
        candidate = Path(env_runner).resolve().parent
        if (candidate / "protein_mpnn_utils.py").exists():
            PROTEINMPNN_DIR = candidate
sys.path.insert(0, str(PROTEINMPNN_DIR))

from protein_mpnn_utils import parse_PDB

# Add mpnn_utils to path (lives in 02_design/)
_DESIGN_DIR = str(Path(__file__).resolve().parent.parent / "02_design")
if _DESIGN_DIR not in sys.path:
    sys.path.insert(0, _DESIGN_DIR)
from mpnn_utils import SEQUON_REGEX

try:
    from Bio.PDB import PDBParser, ShrakeRupley, is_aa
    from Bio.SeqUtils import seq1
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

CONDITIONS = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]


def verify_indexing(pdb_path: Path, mpnn_seqs: dict[str, str], chain_order: list[str]):
    """Cross-check MPNN sequential indexing against BioPython PDB residue parsing."""
    if not BIOPYTHON_AVAILABLE:
        print("  INDEXING CHECK: skipped (BioPython not installed)")
        return True

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    all_ok = True
    print("\n  INDEXING CHECK:")
    for chain_id in chain_order:
        mpnn_seq = mpnn_seqs[chain_id]
        bio_residues = []
        for model in structure:
            if chain_id not in [c.id for c in model]:
                continue
            chain = model[chain_id]
            bio_residues = [r for r in chain.get_residues() if is_aa(r)]
            break

        if not bio_residues:
            print(f"    Chain {chain_id}: WARNING - not found in BioPython parse")
            all_ok = False
            continue

        bio_seq = "".join(seq1(r.get_resname()) for r in bio_residues)
        if bio_seq == mpnn_seq:
            pdb_start = bio_residues[0].id[1]
            pdb_end = bio_residues[-1].id[1]
            print(f"    Chain {chain_id}: OK (len={len(mpnn_seq)}, "
                  f"PDB resnum {pdb_start}-{pdb_end})")
        else:
            print(f"    Chain {chain_id}: MISMATCH!")
            print(f"      MPNN  len={len(mpnn_seq)}: {mpnn_seq[:20]}...")
            print(f"      BioPy len={len(bio_seq)}: {bio_seq[:20]}...")
            for i, (a, b) in enumerate(zip(mpnn_seq, bio_seq)):
                if a != b:
                    print(f"      First diff at index {i}: MPNN='{a}' vs BioPy='{b}'")
                    break
            all_ok = False

    if all_ok:
        print("    All chains consistent between MPNN and BioPython parsers.")
    else:
        print("    WARNING: Indexing mismatches detected! Structural metrics may be offset.")
    return all_ok

MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "E": 223, "Q": 225, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}


def get_mpnn_chain_sequences(pdb_path: Path):
    pdb_dict_list = parse_PDB(str(pdb_path))
    if not pdb_dict_list:
        raise ValueError(f"Failed to parse PDB: {pdb_path}")

    pdb_dict = pdb_dict_list[0]
    chain_seqs = {}
    chain_order = []

    for key in sorted(pdb_dict.keys()):
        if key.startswith("seq_chain_"):
            chain_id = key.replace("seq_chain_", "")
            seq = pdb_dict[key]
            if seq:
                chain_seqs[chain_id] = seq
                chain_order.append(chain_id)

    return chain_seqs, chain_order


def find_sequons(sequence: str):
    """Find N-X-S/T sequons using canonical SEQUON_REGEX from mpnn_utils."""
    sequons = []
    for match in SEQUON_REGEX.finditer(sequence):
        sequons.append(
            {
                "position_0idx": match.start(),
                "position_1idx": match.start() + 1,
                "triplet": match.group(),
            }
        )
    return sequons


def read_fasta(fasta_path: Path):
    records = []
    header = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        records.append((header, "".join(seq_lines)))

    return records


def calculate_rsa(pdb_path: Path):
    if not BIOPYTHON_AVAILABLE:
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    rsa_data = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            rsa_data.setdefault(chain_id, {})
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                res_num = residue.id[1]
                try:
                    aa = seq1(residue.get_resname())
                    asa = residue.sasa if hasattr(residue, "sasa") else 0
                    rsa = (asa / MAX_ASA[aa]) * 100 if aa in MAX_ASA and MAX_ASA[aa] > 0 else None
                    rsa_data[chain_id][res_num] = {"aa": aa, "asa": asa, "rsa": rsa}
                except Exception:
                    continue

    return rsa_data


def get_bfactors(pdb_path: Path):
    if not BIOPYTHON_AVAILABLE:
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    bfactor_data = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            bfactor_data.setdefault(chain_id, {})
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                res_num = residue.id[1]
                vals = [atom.get_bfactor() for atom in residue]
                if vals:
                    bfactor_data[chain_id][res_num] = float(np.mean(vals))

    return bfactor_data


def map_mpnn_to_pdb_positions(pdb_path: Path, chain_id: str):
    if not BIOPYTHON_AVAILABLE:
        return {}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    mapping = {}
    for model in structure:
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        residues = [r for r in chain.get_residues() if is_aa(r)]
        for idx, residue in enumerate(residues):
            mapping[idx] = residue.id[1]

    return mapping


def resolve_fasta_path(seqs_dir: Path, fasta_stem: str | None):
    if fasta_stem:
        candidate = seqs_dir / f"{fasta_stem}.fa"
        if candidate.exists():
            return candidate

    files = sorted(seqs_dir.glob("*.fa"))
    if len(files) == 1:
        return files[0]
    if len(files) > 1 and fasta_stem:
        for item in files:
            if item.stem == fasta_stem:
                return item
    return None


def parse_chain_groups(chain_groups_arg: str | None, chain_order: list[str]):
    if not chain_groups_arg:
        return {chain: [chain] for chain in chain_order}

    groups = {}
    for group_spec in chain_groups_arg.split(";"):
        group_spec = group_spec.strip()
        if not group_spec:
            continue
        if ":" not in group_spec:
            raise ValueError(f"Invalid chain group spec: {group_spec}")
        name, chains_csv = group_spec.split(":", 1)
        chains = [c.strip() for c in chains_csv.split(",") if c.strip()]
        if not chains:
            raise ValueError(f"No chains provided for group: {name}")
        groups[name.strip()] = chains

    return groups


def chain_to_group_map(chain_groups: dict[str, list[str]], chain_order: list[str]):
    mapping = {chain: chain for chain in chain_order}
    for group_name, chains in chain_groups.items():
        for chain in chains:
            mapping[chain] = group_name
    return mapping


def analyze_condition(condition,
                      wt_seqs,
                      chain_order,
                      sequons_by_chain,
                      designs_dir,
                      fasta_stem,
                      evidence_tiers=None):
    seqs_dir = designs_dir / condition / "seqs"
    fa_path = resolve_fasta_path(seqs_dir, fasta_stem)
    if fa_path is None:
        print(f"  {condition}: FASTA not found in {seqs_dir}")
        return None

    records = read_fasta(fa_path)
    if len(records) <= 1:
        print(f"  {condition}: no design records in {fa_path}")
        return None

    print(f"  {condition}: {len(records) - 1} designs ({fa_path.name})")

    chain_starts = {}
    pos = 0
    for chain_id in chain_order:
        chain_starts[chain_id] = pos
        pos += len(wt_seqs[chain_id])

    wt_concat = "".join(wt_seqs[c] for c in chain_order)
    rows = []

    for header, seq in records[1:]:  # skip WT
        design_seq = seq.replace("/", "")
        if len(design_seq) != len(wt_concat):
            continue

        score = None
        recovery = None
        for part in header.split(","):
            part = part.strip()
            if part.startswith("score="):
                try:
                    score = float(part.split("=", 1)[1])
                except ValueError:
                    pass
            if part.startswith("seq_recovery="):
                try:
                    recovery = float(part.split("=", 1)[1])
                except ValueError:
                    pass

        for chain_id, sequons in sequons_by_chain.items():
            start = chain_starts[chain_id]
            for seq_info in sequons:
                pos_0idx = seq_info["position_0idx"]
                global_pos = start + pos_0idx
                wt_triplet = seq_info["triplet"]
                if global_pos + 3 > len(design_seq):
                    continue

                design_triplet = design_seq[global_pos:global_pos + 3]
                contains_x = "X" in design_triplet
                n_retained = design_triplet[0] == "N" and not contains_x
                exact_match = design_triplet == wt_triplet
                functional = (
                    design_triplet[0] == "N"
                    and design_triplet[1] not in ("P", "X")
                    and design_triplet[2] in "ST"
                    and not contains_x
                )

                tier = (evidence_tiers or {}).get((chain_id, pos_0idx), "motif_only")

                rows.append(
                    {
                        "condition": condition,
                        "chain": chain_id,
                        "position_0idx": pos_0idx,
                        "position_1idx": pos_0idx + 1,
                        "wt_triplet": wt_triplet,
                        "design_triplet": design_triplet,
                        "evidence_tier": tier,
                        "n_retained": n_retained,
                        "exact_match": exact_match,
                        "functional": functional,
                        "contains_x": contains_x,
                        "score": score,
                        "recovery": recovery,
                    }
                )

    return pd.DataFrame(rows)


def analyze_denovo_sequons(condition, wt_seqs, chain_order, sequons_by_chain,
                           designs_dir, fasta_stem):
    """Scan designed sequences for de novo sequons (not present in wild-type)."""
    seqs_dir = designs_dir / condition / "seqs"
    fa_path = resolve_fasta_path(seqs_dir, fasta_stem)
    if fa_path is None:
        return None

    records = read_fasta(fa_path)
    if len(records) <= 1:
        return None

    # Collect native sequon positions per chain (0-indexed)
    native_positions = {}
    for chain_id in chain_order:
        native_positions[chain_id] = set()
    for chain_id, sequons in sequons_by_chain.items():
        for s in sequons:
            native_positions[chain_id].add(s["position_0idx"])

    # Build chain start offsets for the concatenated sequence
    chain_starts = {}
    pos = 0
    for chain_id in chain_order:
        chain_starts[chain_id] = pos
        pos += len(wt_seqs[chain_id])

    wt_concat = "".join(wt_seqs[c] for c in chain_order)
    rows = []

    for design_idx, (header, seq) in enumerate(records[1:]):
        design_seq = seq.replace("/", "")
        if len(design_seq) != len(wt_concat):
            continue

        for chain_id in chain_order:
            start = chain_starts[chain_id]
            chain_len = len(wt_seqs[chain_id])
            chain_seq = design_seq[start:start + chain_len]

            for match in SEQUON_REGEX.finditer(chain_seq):
                pos_0idx = match.start()
                if pos_0idx not in native_positions.get(chain_id, set()):
                    rows.append({
                        "condition": condition,
                        "design": design_idx + 1,
                        "chain": chain_id,
                        "position_0idx": pos_0idx,
                        "position_1idx": pos_0idx + 1,
                        "wt_triplet": wt_seqs[chain_id][pos_0idx:pos_0idx + 3],
                        "denovo_triplet": match.group(),
                    })

    return pd.DataFrame(rows)


def create_denovo_figures(pdb_id, denovo_df, chain_order, wt_seqs, output_dir):
    """Create de novo sequon analysis figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    condition_order = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]

    # --- Figure 1: De novo sequon frequency by condition ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Average de novo sequons per design
    ax = axes[0]
    n_designs_per_cond = denovo_df.groupby("condition")["design"].nunique()
    total_per_design = (denovo_df.groupby(["condition", "design"]).size()
                        .groupby("condition").mean())

    # Include conditions with zero de novo (they won't appear in denovo_df)
    avg_counts = []
    for cond in condition_order:
        if cond in total_per_design.index:
            avg_counts.append(total_per_design[cond])
        else:
            avg_counts.append(0.0)

    colors = ["#d95f02", "#1b9e77", "#7570b3"]
    bars = ax.bar(condition_order, avg_counts, color=colors)
    for bar, val in zip(bars, avg_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg De Novo Sequons per Design")
    ax.set_title(f"{pdb_id}: De Novo Sequon Generation")
    ax.tick_params(axis="x", rotation=20)

    # Panel B: Top hotspot positions (unconstrained)
    ax = axes[1]
    unconstrained = denovo_df[denovo_df["condition"] == "unconstrained"]
    if not unconstrained.empty:
        n_designs = unconstrained["design"].nunique()
        pos_counts = unconstrained.groupby(["chain", "position_1idx"]).size()
        pos_pct = (pos_counts / n_designs * 100).sort_values(ascending=True)
        top = pos_pct.tail(15)

        labels = [f"{c}:{p}" for c, p in top.index]
        bar_colors = ["#E74C3C" if v >= 75 else "#F39C12" if v >= 50 else "#3498DB"
                      for v in top.values]
        ax.barh(labels, top.values, color=bar_colors)
        ax.set_xlabel("Occurrence (%)")
        ax.set_title("Top De Novo Hotspots (Unconstrained)")
        ax.set_xlim(0, 105)
    else:
        ax.text(0.5, 0.5, "No de novo sequons in unconstrained", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Top De Novo Hotspots (Unconstrained)")

    plt.suptitle(f"{pdb_id}: De Novo Sequon Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    freq_path = fig_dir / f"{pdb_id}_denovo_sequon_frequency.png"
    plt.savefig(freq_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {freq_path}")

    # --- Figure 2: Sequon map with de novo positions ---
    unconstrained = denovo_df[denovo_df["condition"] == "unconstrained"]
    if not unconstrained.empty:
        n_designs = unconstrained["design"].nunique()

        fig, axes_map = plt.subplots(len(chain_order), 1,
                                     figsize=(16, max(3, 2 * len(chain_order))),
                                     squeeze=False)
        for i, chain_id in enumerate(chain_order):
            ax = axes_map[i, 0]
            chain_len = len(wt_seqs[chain_id])
            chain_data = unconstrained[unconstrained["chain"] == chain_id]

            freq = np.zeros(chain_len)
            for _, row in chain_data.iterrows():
                p = row["position_0idx"]
                if p < chain_len:
                    freq[p] += 1
            freq_pct = freq / n_designs * 100

            ax.bar(range(chain_len), freq_pct, width=1.0, color="#3498DB", alpha=0.7)
            ax.set_ylabel("Freq (%)")
            ax.set_title(f"Chain {chain_id} (len={chain_len})")
            ax.set_xlim(0, chain_len)
            ax.set_ylim(0, max(105, freq_pct.max() + 5) if freq_pct.max() > 0 else 10)

        axes_map[-1, 0].set_xlabel("Residue Position (0-indexed)")
        plt.suptitle(f"{pdb_id}: De Novo Sequon Map (Unconstrained)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        map_path = fig_dir / f"{pdb_id}_sequon_map_with_denovo.png"
        plt.savefig(map_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {map_path}")


def create_figures(pdb_id, all_results, structural_df, denovo_df, output_dir):
    """Create adaptive comprehensive figures.

    Layout adapts based on whether the protein has meaningful retention
    variation or not (e.g. 5EQG with 0% unconstrained retention).
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    condition_order = ["unconstrained", "n_only_fixed", "full_sequon_fixed"]
    cond_colors = ["#d95f02", "#1b9e77", "#7570b3"]
    cond_labels = ["Unconstrained", "N-only Fixed", "Full Sequon\nFixed"]

    unconstrained = all_results[all_results["condition"] == "unconstrained"]
    unconstrained_ret = unconstrained["n_retained"].mean() * 100 if not unconstrained.empty else 0
    has_retention_variation = unconstrained_ret > 5  # meaningful variation threshold
    has_multi_chain = len(all_results["chain"].unique()) > 1

    # Decide layout: 2x3 for data-rich proteins, 2x3 adapted for low-retention
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Panel A: Retention by condition (always useful) ---
    ax = axes[0, 0]
    retention_types = []
    for cond in condition_order:
        cond_data = all_results[all_results["condition"] == cond]
        if not cond_data.empty:
            retention_types.append({
                "condition": cond,
                "N Retained": cond_data["n_retained"].mean() * 100,
                "Exact Match": cond_data["exact_match"].mean() * 100,
                "Functional": cond_data["functional"].mean() * 100,
            })
    if retention_types:
        ret_df = pd.DataFrame(retention_types).set_index("condition").reindex(condition_order)
        ret_df.plot(kind="bar", ax=ax, color=["#3498db", "#9b59b6", "#2ecc71"])
        ax.set_xticklabels(cond_labels, rotation=0, fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("A. Sequon Retention by Condition", fontsize=11, fontweight="bold")
    ax.set_ylabel("Retention (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("")

    # --- Panel B: Adapts based on data richness ---
    ax = axes[0, 1]
    if has_retention_variation and has_multi_chain:
        # Multi-chain with variation: chain group retention
        chain_ret = unconstrained.groupby("chain_type")["n_retained"].mean() * 100
        chain_ret.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#4c78a8")
        ax.set_title("B. Unconstrained Retention by Chain Group", fontsize=11, fontweight="bold")
        ax.set_ylabel("N Retention (%)")
        ax.set_ylim(0, 105)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    else:
        # Low retention: show what MPNN designed at sequon positions
        ax.set_title("B. Designed AA at Sequon N-Positions", fontsize=11, fontweight="bold")
        if not unconstrained.empty:
            n_pos_aa = unconstrained["design_triplet"].str[0].value_counts(normalize=True) * 100
            top_aa = n_pos_aa.head(10)
            bars = ax.bar(range(len(top_aa)), top_aa.values, color="#e74c3c")
            ax.set_xticks(range(len(top_aa)))
            ax.set_xticklabels(top_aa.index, fontsize=10)
            ax.set_ylabel("Frequency (%)")
            ax.set_xlabel("Amino Acid (at N position)")
            for bar, val in zip(bars, top_aa.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    # --- Panel C: MPNN score comparison (always informative) ---
    ax = axes[0, 2]
    score_data = []
    for cond in condition_order:
        cond_data = all_results[all_results["condition"] == cond]
        if not cond_data.empty and cond_data["score"].notna().any():
            scores = cond_data.groupby("score").size().index.tolist()
            score_data.append({"condition": cond, "scores": cond_data["score"].dropna().unique()})

    if score_data:
        box_data = [d["scores"] for d in score_data]
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=cond_labels)
        for patch, color in zip(bp["boxes"], cond_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("ProteinMPNN Score")
        # Add median labels
        for i, d in enumerate(box_data):
            median = np.median(d)
            ax.text(i + 1, median, f"{median:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
    ax.set_title("C. Design Quality (MPNN Score)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    # --- Panel D: RSA context (adapted) ---
    ax = axes[1, 0]
    if structural_df is not None and not structural_df.empty:
        valid = structural_df.dropna(subset=["rsa"])
        if not valid.empty:
            if has_retention_variation and valid["n_retention"].notna().any():
                # Scatter plot with trend
                sns.scatterplot(data=valid, x="rsa", y="n_retention", hue="chain_type",
                                s=100, ax=ax, legend=True)
                if len(valid) >= 3 and valid["n_retention"].std() > 0:
                    rho, pval = stats.spearmanr(valid["rsa"], valid["n_retention"])
                    z = np.polyfit(valid["rsa"], valid["n_retention"], 1)
                    x_line = np.linspace(valid["rsa"].min(), valid["rsa"].max(), 100)
                    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
                    ax.text(0.02, 0.98, f"rho={rho:.2f}, p={pval:.2e}",
                            transform=ax.transAxes, va="top", fontsize=9)
                ax.set_ylabel("N Retention (%)")
                ax.set_title("D. RSA vs N Retention", fontsize=11, fontweight="bold")
            else:
                # No variation: show RSA context as a bar chart with retention annotated
                bar_labels = [f"{r['chain']}:{r['position_1idx']}\n{r['triplet']}"
                              for _, r in valid.iterrows()]
                colors_rsa = ["#3498db" if r["rsa_category"] == "Buried"
                              else "#f1c40f" if r["rsa_category"] == "Intermediate"
                              else "#e74c3c" for _, r in valid.iterrows()]
                n_bars = len(valid)
                bar_width = 0.6 if n_bars <= 6 else 0.8
                bars = ax.bar(range(n_bars), valid["rsa"].values,
                              color=colors_rsa, width=bar_width, edgecolor="gray",
                              linewidth=0.5)
                ax.set_xticks(range(n_bars))
                rotation = 45 if n_bars > 4 else 0
                ha = "right" if rotation else "center"
                ax.set_xticklabels(bar_labels, fontsize=9, rotation=rotation, ha=ha)
                ax.set_ylabel("RSA (%)")
                ax.yaxis.grid(True, alpha=0.3)
                # Annotate retention above each bar with padding
                max_rsa = valid["rsa"].max() if not valid["rsa"].empty else 50
                pad = max_rsa * 0.05 + 1
                for bar, (_, row) in zip(bars, valid.iterrows()):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + pad,
                            f"{row['n_retention']:.1f}%",
                            ha="center", va="bottom",
                            fontsize=10, fontweight="bold", color="#333")
                ax.set_ylim(0, max_rsa * 1.25 + 5)
                ax.set_title("D. Sequon Structural Context (RSA)", fontsize=11, fontweight="bold")
                # Legend for RSA categories
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor="#3498db", label="Buried (<20%)"),
                                   Patch(facecolor="#f1c40f", label="Intermediate (20-50%)"),
                                   Patch(facecolor="#e74c3c", label="Exposed (>50%)")]
                ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
        else:
            ax.text(0.5, 0.5, "No RSA data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("RSA (%)" if has_retention_variation else "")

    # --- Panel E: Per-position heatmap or designed triplet table ---
    ax = axes[1, 1]
    if has_retention_variation and has_multi_chain:
        # Rich heatmap
        pos_ret = unconstrained.groupby(["chain", "position_1idx"])["n_retained"].mean() * 100
        pos_df = pos_ret.reset_index()
        if not pos_df.empty:
            pivot = pos_df.pivot(index="chain", columns="position_1idx", values="n_retained")
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax)
            ax.set_xlabel("Position (1-indexed)")
            ax.set_ylabel("Chain")
        ax.set_title("E. Per-Position Retention (Unconstrained)", fontsize=11, fontweight="bold")
    else:
        # Show what the top designed triplets are at each sequon site
        ax.set_title("E. Top Designed Triplets at Sequon Sites", fontsize=11, fontweight="bold")
        if not unconstrained.empty:
            table_data = []
            for (chain, pos), grp in unconstrained.groupby(["chain", "position_1idx"]):
                wt = grp["wt_triplet"].iloc[0]
                top_triplets = grp["design_triplet"].value_counts(normalize=True).head(3)
                top_str = ", ".join(f"{t} ({v*100:.0f}%)" for t, v in top_triplets.items())
                table_data.append([f"{chain}:{pos}", wt, top_str])

            if table_data:
                ax.axis("off")
                table = ax.table(cellText=table_data,
                                 colLabels=["Position", "WT Triplet", "Top Designs"],
                                 loc="center", cellLoc="left")
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")

    # --- Panel F: De novo or retention types ---
    ax = axes[1, 2]
    if denovo_df is not None and not denovo_df.empty:
        # De novo summary across conditions
        avg_counts = []
        for cond in condition_order:
            cond_data = denovo_df[denovo_df["condition"] == cond]
            if cond_data.empty:
                avg_counts.append(0.0)
            else:
                n_designs = cond_data["design"].nunique()
                avg_counts.append(len(cond_data) / n_designs)

        bars = ax.bar(cond_labels, avg_counts, color=cond_colors)
        for bar, val in zip(bars, avg_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Avg De Novo Sequons / Design")
        ax.set_title("F. De Novo Sequon Generation", fontsize=11, fontweight="bold")
    else:
        # No de novo data: show retention types
        if retention_types:
            ret_df2 = pd.DataFrame(retention_types).set_index("condition").reindex(condition_order)
            ret_df2[["N Retained", "Functional"]].plot(kind="bar", ax=ax,
                                                        color=["#3498db", "#2ecc71"])
            ax.set_xticklabels(cond_labels, rotation=0, fontsize=9)
            ax.set_ylabel("Retention (%)")
            ax.set_ylim(0, 105)
        ax.set_title("F. Retention Types by Condition", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    plt.suptitle(f"{pdb_id}: Comprehensive Design Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = fig_dir / f"{pdb_id}_comprehensive_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved figure: {out_path}")
    return out_path


def default_pdb_candidates(pipeline_dir: Path, pdb_id: str):
    return [
        pipeline_dir / "data" / "prep" / pdb_id / "structure" / f"{pdb_id}_protein.pdb",
        pipeline_dir / "data" / "prep" / pdb_id / "structure" / f"{pdb_id}.pdb",
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze modular case-design outputs for any PDB")
    parser.add_argument("--pdb-id", required=True, help="PDB code")
    parser.add_argument(
        "--pdb-path",
        help="Path to structure PDB (default: data/prep/<PDB_ID>/structure/<PDB_ID>_protein.pdb)",
    )
    parser.add_argument(
        "--designs-dir",
        help="Directory containing unconstrained/n_only_fixed/full_sequon_fixed (default: data/outputs/output_<PDB_ID>)",
    )
    parser.add_argument("--output-dir", help="Analysis output directory (default: designs-dir)")
    parser.add_argument("--fasta-stem", help="FASTA stem if ambiguous (e.g., 5EQG_protein)")
    parser.add_argument(
        "--chain-groups",
        help="Optional mapping like 'HA1:H,J,L;HA2:I,K,M'. Default: each chain is its own group",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pdb_id = args.pdb_id.upper()

    if args.pdb_path:
        pdb_path = Path(args.pdb_path)
    else:
        pdb_path = None
        for candidate in default_pdb_candidates(PIPELINE_ROOT, pdb_id):
            if candidate.exists():
                pdb_path = candidate
                break
        if pdb_path is None:
            pdb_path = default_pdb_candidates(PIPELINE_ROOT, pdb_id)[0]

    designs_dir = Path(args.designs_dir) if args.designs_dir else (DATA_DIR / "outputs" / f"output_{pdb_id}")
    output_dir = Path(args.output_dir) if args.output_dir else designs_dir

    if not pdb_path.exists():
        print(f"ERROR: PDB not found: {pdb_path}")
        return 1

    if not designs_dir.exists():
        print(f"ERROR: designs directory not found: {designs_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ANALYZING DESIGNS FOR {pdb_id}")
    print("=" * 70)
    print(f"PDB: {pdb_path}")
    print(f"Designs: {designs_dir}")
    print(f"Output: {output_dir}")

    wt_seqs, chain_order = get_mpnn_chain_sequences(pdb_path)
    verify_indexing(pdb_path, wt_seqs, chain_order)
    chain_groups = parse_chain_groups(args.chain_groups, chain_order)
    chain_type_map = chain_to_group_map(chain_groups, chain_order)

    sequons_by_chain = {}
    total_sequons = 0
    for chain_id in chain_order:
        sequons = find_sequons(wt_seqs[chain_id])
        if sequons:
            sequons_by_chain[chain_id] = sequons
            total_sequons += len(sequons)

    print(f"Found {len(chain_order)} chains and {total_sequons} total sequons")

    # Load evidence tiers from sequons.csv if available
    evidence_tiers = {}  # (chain_id, position_0idx) -> tier
    sequons_csv_candidates = [
        PIPELINE_ROOT / "data" / "prep" / pdb_id / "sequons" / "sequons.csv",
    ]
    for csv_candidate in sequons_csv_candidates:
        if csv_candidate.exists():
            import csv as _csv
            with open(csv_candidate) as _f:
                reader = _csv.DictReader(_f)
                for row in reader:
                    if "evidence_tier" in row and row["evidence_tier"]:
                        evidence_tiers[(row["chain_id"], int(row["position_0idx"]))] = row["evidence_tier"]
            if evidence_tiers:
                print(f"Loaded evidence tiers for {len(evidence_tiers)} sequon(s) from {csv_candidate}")
            break

    rsa_data = calculate_rsa(pdb_path)
    bfactor_data = get_bfactors(pdb_path)

    all_frames = []
    for condition in CONDITIONS:
        df = analyze_condition(
            condition,
            wt_seqs,
            chain_order,
            sequons_by_chain,
            designs_dir,
            args.fasta_stem,
            evidence_tiers=evidence_tiers,
        )
        if df is not None and not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("ERROR: No analyzable design outputs found")
        return 1

    all_results = pd.concat(all_frames, ignore_index=True)
    all_results["chain_type"] = all_results["chain"].map(chain_type_map).fillna(all_results["chain"])

    structural_rows = []
    unconstrained = all_results[all_results["condition"] == "unconstrained"]

    for chain_id, sequons in sequons_by_chain.items():
        pos_map = map_mpnn_to_pdb_positions(pdb_path, chain_id)
        for seq_info in sequons:
            pos_0idx = seq_info["position_0idx"]
            pos_1idx = seq_info["position_1idx"]
            pdb_resnum = pos_map.get(pos_0idx, pos_1idx)

            rsa = None
            if rsa_data and chain_id in rsa_data and pdb_resnum in rsa_data[chain_id]:
                rsa = rsa_data[chain_id][pdb_resnum].get("rsa")

            bfactor = None
            if bfactor_data and chain_id in bfactor_data and pdb_resnum in bfactor_data[chain_id]:
                bfactor = bfactor_data[chain_id][pdb_resnum]

            pos_df = unconstrained[
                (unconstrained["chain"] == chain_id)
                & (unconstrained["position_1idx"] == pos_1idx)
            ]

            n_retention = pos_df["n_retained"].mean() * 100 if not pos_df.empty else np.nan
            functional = pos_df["functional"].mean() * 100 if not pos_df.empty else np.nan

            if pd.isna(rsa):
                rsa_cat = "Unknown"
            elif rsa < 20:
                rsa_cat = "Buried"
            elif rsa < 50:
                rsa_cat = "Intermediate"
            else:
                rsa_cat = "Exposed"

            tier = evidence_tiers.get((chain_id, pos_0idx), "motif_only")

            structural_rows.append(
                {
                    "chain": chain_id,
                    "chain_type": chain_type_map.get(chain_id, chain_id),
                    "position_1idx": pos_1idx,
                    "pdb_resnum": pdb_resnum,
                    "triplet": seq_info["triplet"],
                    "evidence_tier": tier,
                    "rsa": rsa,
                    "rsa_category": rsa_cat,
                    "bfactor": bfactor,
                    "n_retention": n_retention,
                    "functional_retention": functional,
                }
            )

    structural_df = pd.DataFrame(structural_rows)

    # De novo sequon analysis (run before figures so we can include it)
    print("\n--- De Novo Sequon Analysis ---")
    denovo_frames = []
    for condition in CONDITIONS:
        df = analyze_denovo_sequons(
            condition, wt_seqs, chain_order, sequons_by_chain,
            designs_dir, args.fasta_stem,
        )
        if df is not None and not df.empty:
            denovo_frames.append(df)

    denovo_df = pd.concat(denovo_frames, ignore_index=True) if denovo_frames else pd.DataFrame()

    if not denovo_df.empty:
        denovo_path = output_dir / "denovo_sequons.csv"
        denovo_df.to_csv(denovo_path, index=False)
        print(f"  Found {len(denovo_df)} de novo sequon instances across all conditions")
        print(f"  Saved: {denovo_path}")

        # Per-condition summary
        for condition in CONDITIONS:
            cond_data = denovo_df[denovo_df["condition"] == condition]
            if cond_data.empty:
                print(f"  {condition}: 0 de novo sequons")
            else:
                n_designs = cond_data["design"].nunique()
                avg_per_design = len(cond_data) / n_designs
                unique_positions = cond_data.groupby(["chain", "position_0idx"]).ngroups
                print(f"  {condition}: {avg_per_design:.1f} avg/design, {unique_positions} unique positions")

        create_denovo_figures(pdb_id, denovo_df, chain_order, wt_seqs, output_dir)
    else:
        print("  No de novo sequons found in any condition")

    # Comprehensive figures (includes de novo data)
    create_figures(pdb_id, all_results, structural_df, denovo_df, output_dir)

    all_results_path = output_dir / "all_conditions_retention.csv"
    structural_path = output_dir / "structural_context.csv"
    report_path = output_dir / f"{pdb_id}_CASE_STUDY_REPORT.md"

    all_results.to_csv(all_results_path, index=False)
    structural_df.to_csv(structural_path, index=False)

    with open(report_path, "w") as f:
        f.write(f"# {pdb_id} Case Study\n\n")
        f.write(f"- Chains analyzed: {len(chain_order)}\n")
        f.write(f"- Total sequons: {total_sequons}\n")

        # Evidence tier summary
        if "evidence_tier" in structural_df.columns:
            tier_counts = structural_df["evidence_tier"].value_counts()
            f.write(f"\n### Evidence Tiers\n\n")
            f.write("| Tier | Count |\n|---|---:|\n")
            for tier_name in ["experimental", "pdb_evidence", "curator_inferred", "motif_only"]:
                if tier_name in tier_counts.index:
                    f.write(f"| {tier_name} | {tier_counts[tier_name]} |\n")
            n_validated = tier_counts.get("experimental", 0) + tier_counts.get("pdb_evidence", 0)
            f.write(f"\nValidated sites: {n_validated}/{total_sequons}\n")

        f.write("\n## N Retention By Condition\n\n")
        f.write("| Condition | N Retention (%) | Exact Match (%) | Functional (%) |\n")
        f.write("|---|---:|---:|---:|\n")
        for condition in CONDITIONS:
            cond = all_results[all_results["condition"] == condition]
            if cond.empty:
                continue
            f.write(
                f"| {condition} | "
                f"{100 * cond['n_retained'].mean():.1f} | "
                f"{100 * cond['exact_match'].mean():.1f} | "
                f"{100 * cond['functional'].mean():.1f} |\n"
            )

        # Retention by evidence tier (unconstrained condition)
        if "evidence_tier" in all_results.columns and evidence_tiers:
            f.write("\n## N Retention By Evidence Tier (Unconstrained)\n\n")
            f.write("| Evidence Tier | N Retention (%) | Functional (%) | Count |\n")
            f.write("|---|---:|---:|---:|\n")
            unconstrained_et = all_results[all_results["condition"] == "unconstrained"]
            if not unconstrained_et.empty:
                for tier_name in ["experimental", "pdb_evidence", "curator_inferred", "motif_only"]:
                    tier_data = unconstrained_et[unconstrained_et["evidence_tier"] == tier_name]
                    if not tier_data.empty:
                        n_ret = tier_data["n_retained"].mean() * 100
                        func_ret = tier_data["functional"].mean() * 100
                        n_sites = tier_data.groupby(["chain", "position_0idx"]).ngroups
                        f.write(f"| {tier_name} | {n_ret:.1f} | {func_ret:.1f} | {n_sites} |\n")

        if not structural_df.empty and structural_df["rsa"].notna().sum() >= 3:
            valid = structural_df.dropna(subset=["rsa", "n_retention"])
            rho, pval = stats.spearmanr(valid["rsa"], valid["n_retention"])
            f.write("\n## RSA Correlation\n\n")
            f.write(f"Spearman rho = {rho:.3f}, p = {pval:.3e}\n")

        # De novo sequon section
        if not denovo_df.empty:
            f.write("\n## De Novo Sequon Analysis\n\n")
            f.write("| Condition | Avg De Novo / Design | Unique Positions |\n")
            f.write("|---|---:|---:|\n")
            for condition in CONDITIONS:
                cond_data = denovo_df[denovo_df["condition"] == condition]
                if cond_data.empty:
                    f.write(f"| {condition} | 0.0 | 0 |\n")
                else:
                    n_designs = cond_data["design"].nunique()
                    avg = len(cond_data) / n_designs
                    unique = cond_data.groupby(["chain", "position_0idx"]).ngroups
                    f.write(f"| {condition} | {avg:.1f} | {unique} |\n")

            # Top hotspots
            unconstrained_dn = denovo_df[denovo_df["condition"] == "unconstrained"]
            if not unconstrained_dn.empty:
                n_designs = unconstrained_dn["design"].nunique()
                pos_counts = unconstrained_dn.groupby(
                    ["chain", "position_1idx"]).size().sort_values(ascending=False)
                f.write("\n### Top De Novo Hotspots (Unconstrained)\n\n")
                f.write("| Position | Frequency (%) | WT Triplet | De Novo Triplet(s) |\n")
                f.write("|---|---:|---|---|\n")
                for (chain_id, pos), count in pos_counts.head(10).items():
                    pct = count / n_designs * 100
                    subset = unconstrained_dn[
                        (unconstrained_dn["chain"] == chain_id)
                        & (unconstrained_dn["position_1idx"] == pos)
                    ]
                    wt_tri = subset["wt_triplet"].iloc[0]
                    dn_tris = ", ".join(sorted(subset["denovo_triplet"].unique()))
                    f.write(f"| {chain_id}:{pos} | {pct:.1f} | {wt_tri} | {dn_tris} |\n")

    print("\nOutputs:")
    print(f"  {all_results_path}")
    print(f"  {structural_path}")
    print(f"  {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
