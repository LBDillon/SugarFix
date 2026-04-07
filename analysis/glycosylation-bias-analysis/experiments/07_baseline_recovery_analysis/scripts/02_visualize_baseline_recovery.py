#!/usr/bin/env python3
"""
Visualize baseline recovery analysis results.

Creates publication-quality figures comparing sequon retention to baseline recovery.

Author: Laura Dillon
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "experiments/07_baseline_recovery_analysis/results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300

def load_results():
    """Load analysis results."""
    per_residue = pd.read_csv(RESULTS_DIR / "per_residue_recovery_detailed.csv")
    triplets = pd.read_csv(RESULTS_DIR / "triplet_recovery_detailed.csv")
    summary = pd.read_csv(RESULTS_DIR / "protein_recovery_summary.csv")

    with open(RESULTS_DIR / "summary_statistics.json", 'r') as f:
        stats = json.load(f)

    return per_residue, triplets, summary, stats

def plot_recovery_comparison(triplets, stats):
    """Compare sequon vs non-sequon triplet recovery."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Prepare data
    sequon_data = triplets[triplets['is_sequon']]['recovery_rate']
    non_sequon_data = triplets[~triplets['is_sequon']]['recovery_rate']

    data = [non_sequon_data, sequon_data]
    labels = ['Non-sequon\nTriplets', 'Sequon\n(N-X-S/T)\nTriplets']
    colors = ['#3498db', '#e74c3c']

    # Create violin plot
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Add individual points (sampled if too many)
    for i, (d, label) in enumerate(zip(data, labels)):
        y = np.random.normal(i, 0.04, size=min(len(d), 200))
        sample = np.random.choice(d, size=min(len(d), 200), replace=False)
        ax.scatter(y, sample, alpha=0.3, s=2, color=colors[i])

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Triplet Recovery Rate (%)', fontsize=12)
    ax.set_title('Sequon vs Non-Sequon Triplet Recovery', fontsize=13, fontweight='bold')

    # Add statistics
    p_val = stats['mannwhitney_p_value']
    cohens_d = stats['cohens_d']

    ax.text(0.5, 0.95, f"p = {p_val:.4f}\nCohen's d = {cohens_d:.3f}",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sequon_vs_nonsequon_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "sequon_vs_nonsequon_comparison.pdf", bbox_inches='tight')
    print(f"Saved: sequon_vs_nonsequon_comparison.png/pdf")
    plt.close()

def plot_per_aa_recovery(per_residue):
    """Plot per-amino-acid recovery rates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate mean recovery per AA
    aa_recovery = per_residue.groupby('native_aa')['recovery_rate'].agg(['mean', 'std', 'count'])
    aa_recovery = aa_recovery.sort_values('mean', ascending=False)

    # Highlight sequon-relevant AAs
    colors = ['#e74c3c' if aa in ['N', 'S', 'T'] else '#3498db'
              for aa in aa_recovery.index]

    # Bar plot
    x = np.arange(len(aa_recovery))
    ax.bar(x, aa_recovery['mean'], yerr=aa_recovery['std'], color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(aa_recovery.index, fontsize=11)
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Recovery Rate (%)', fontsize=12)
    ax.set_title('Per-Amino-Acid Recovery Rates', fontsize=13, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Sequon components (N, S, T)'),
        Patch(facecolor='#3498db', alpha=0.7, label='Other amino acids')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_aa_recovery.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "per_aa_recovery.pdf", bbox_inches='tight')
    print(f"Saved: per_aa_recovery.png/pdf")
    plt.close()

def plot_recovery_distribution(per_residue, triplets, stats):
    """Plot distribution of recovery rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-residue recovery distribution
    ax = axes[0]
    ax.hist(per_residue['recovery_rate'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(stats['overall_per_residue_recovery_mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean = {stats['overall_per_residue_recovery_mean']:.1f}%")
    ax.set_xlabel('Recovery Rate (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Per-Residue Recovery Distribution', fontsize=13, fontweight='bold')
    ax.legend()

    # Triplet recovery distribution
    ax = axes[1]
    ax.hist(triplets[~triplets['is_sequon']]['recovery_rate'], bins=50, color='#3498db',
            alpha=0.6, edgecolor='black', label='Non-sequon triplets')
    ax.hist(triplets[triplets['is_sequon']]['recovery_rate'], bins=20, color='#e74c3c',
            alpha=0.6, edgecolor='black', label='Sequon triplets')
    ax.set_xlabel('Recovery Rate (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Triplet Recovery Distribution', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recovery_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "recovery_distributions.pdf", bbox_inches='tight')
    print(f"Saved: recovery_distributions.png/pdf")
    plt.close()

def plot_summary_figure(stats):
    """Create summary figure with key statistics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # Title
    title_text = "Baseline Recovery Analysis Summary"
    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Key statistics
    stats_text = f"""
Per-Residue Recovery:
  Mean: {stats['overall_per_residue_recovery_mean']:.1f}% ± {stats['overall_per_residue_recovery_std']:.1f}%

Triplet Recovery:
  All triplets: {stats['overall_triplet_recovery_mean']:.1f}% ± {stats['overall_triplet_recovery_std']:.1f}%
  Sequon triplets: {stats['sequon_triplet_recovery_mean']:.1f}% ± {stats['sequon_triplet_recovery_std']:.1f}%
  Non-sequon triplets: {stats['non_sequon_triplet_recovery_mean']:.1f}% ± {stats['non_sequon_triplet_recovery_std']:.1f}%

Statistical Comparison (Sequon vs Non-Sequon):
  Mann-Whitney U test: p = {stats['mannwhitney_p_value']:.4f}
  Cohen's d: {stats['cohens_d']:.3f}

Dataset:
  {stats['n_proteins']} proteins
  {stats['total_residues']} total residues
  {stats['n_sequon_triplets']} sequon triplets
  {stats['n_non_sequon_triplets']} non-sequon triplets
    """

    ax.text(0.1, 0.85, stats_text, ha='left', va='top', fontsize=11,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Interpretation
    if stats['overall_per_residue_recovery_mean'] > 30:
        interp = "INTERPRETATION: High baseline recovery suggests ProteinMPNN\nACTIVELY DISFAVORS N-X-S/T sequons."
        color = '#e74c3c'
    elif stats['overall_per_residue_recovery_mean'] < 15:
        interp = "INTERPRETATION: Low baseline recovery suggests GENERAL LOW\nFIDELITY, not specific anti-sequon bias."
        color = '#f39c12'
    else:
        interp = "INTERPRETATION: Moderate baseline recovery. Further analysis\nneeded to determine significance."
        color = '#3498db'

    ax.text(0.5, 0.1, interp, ha='center', va='bottom', fontsize=12, fontweight='bold',
            transform=ax.transAxes, color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_figure.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "summary_figure.pdf", bbox_inches='tight')
    print(f"Saved: summary_figure.png/pdf")
    plt.close()

def main():
    """Main visualization pipeline."""
    print("="*70)
    print("BASELINE RECOVERY VISUALIZATION")
    print("="*70)
    print()

    # Load results
    print("Loading results...")
    per_residue, triplets, summary, stats = load_results()
    print(f"Loaded data for {len(summary)} proteins\n")

    # Create figures
    print("Creating figures...")
    plot_recovery_comparison(triplets, stats)
    plot_per_aa_recovery(per_residue)
    plot_recovery_distribution(per_residue, triplets, stats)
    plot_summary_figure(stats)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
