#!/usr/bin/env python3
"""
Visualize de novo sequon generation results.

Creates publication-quality figures showing ProteinMPNN's sequon generation patterns.

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
RESULTS_DIR = BASE_DIR / "experiments/09_de_novo_sequon_generation/results"
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
    try:
        de_novo = pd.read_csv(RESULTS_DIR / "de_novo_sequons_detailed.csv")
    except:
        de_novo = pd.DataFrame()

    summary = pd.read_csv(RESULTS_DIR / "protein_de_novo_summary.csv")

    with open(RESULTS_DIR / "summary_statistics.json", 'r') as f:
        stats = json.load(f)

    return de_novo, summary, stats

def plot_observed_vs_expected(summary, stats):
    """Plot observed vs expected de novo sequons."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    observed = stats['observed_de_novo_sequons']
    expected = stats['expected_de_novo_sequons']
    ratio = stats['observed_over_expected_ratio']
    p_val = stats['binomial_p_value']

    # Bar plot
    x = ['Expected\n(Random)', 'Observed\n(ProteinMPNN)']
    y = [expected, observed]
    colors = ['#95a5a6', '#e74c3c' if ratio < 0.9 else '#3498db']

    bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Error bars (Poisson error for expected)
    ax.errorbar([0], [expected], yerr=[np.sqrt(expected)], fmt='none',
                ecolor='black', capsize=5, capthick=2)

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, y)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_ylabel('Number of De Novo Sequons', fontsize=12)
    ax.set_title('De Novo Sequon Generation: Observed vs Expected', fontsize=13, fontweight='bold')

    # Add statistics
    stats_text = f"Ratio: {ratio:.3f}\np = {p_val:.2e}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=11)

    # Interpretation
    if ratio < 0.5:
        interp = "ProteinMPNN STRONGLY AVOIDS\ngenerating sequons"
        color = '#e74c3c'
    elif ratio < 0.9:
        interp = "ProteinMPNN MODERATELY AVOIDS\ngenerating sequons"
        color = '#f39c12'
    else:
        interp = "No evidence of avoidance"
        color = '#2ecc71'

    ax.text(0.5, 0.5, interp, transform=ax.transAxes, ha='center', va='center',
            fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "observed_vs_expected.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "observed_vs_expected.pdf", bbox_inches='tight')
    print(f"Saved: observed_vs_expected.png/pdf")
    plt.close()

def plot_per_protein_generation(summary):
    """Plot per-protein de novo generation rates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    summary_sorted = summary.sort_values('observed_over_expected', ascending=True)

    # Bar plot
    x = np.arange(len(summary_sorted))
    y = summary_sorted['observed_over_expected']
    colors = ['#e74c3c' if val < 0.9 else '#3498db' for val in y]

    ax.barh(x, y, color=colors, alpha=0.7, edgecolor='black')

    # Reference line at 1.0 (random expectation)
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Random expectation')

    # Formatting
    ax.set_yticks(x)
    ax.set_yticklabels(summary_sorted['protein'], fontsize=9)
    ax.set_xlabel('Observed / Expected Ratio', fontsize=12)
    ax.set_title('Per-Protein De Novo Sequon Generation', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_protein_generation.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "per_protein_generation.pdf", bbox_inches='tight')
    print(f"Saved: per_protein_generation.png/pdf")
    plt.close()

def plot_generation_rate_comparison(stats):
    """Compare generation rate to random expectation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    observed_rate = stats['observed_rate_percent']
    expected_rate = stats['expected_rate_percent']
    ratio = stats['rate_ratio']

    # Bar plot
    x = ['Expected\n(Random)', 'Observed\n(ProteinMPNN)']
    y = [expected_rate, observed_rate]
    colors = ['#95a5a6', '#e74c3c' if ratio < 0.9 else '#3498db']

    bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add values on bars
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_ylabel('De Novo Sequon Generation Rate (%)', fontsize=12)
    ax.set_title('Sequon Generation Rate per Triplet Position', fontsize=13, fontweight='bold')

    # Add statistics
    stats_text = f"Rate ratio: {ratio:.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "generation_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "generation_rate_comparison.pdf", bbox_inches='tight')
    print(f"Saved: generation_rate_comparison.png/pdf")
    plt.close()

def plot_summary_figure(stats):
    """Create summary figure with key statistics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # Title
    title_text = "De Novo Sequon Generation Analysis Summary"
    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Key statistics
    stats_text = f"""
Dataset:
  {stats['total_proteins']} proteins
  {stats['total_designs']} designs
  {stats['total_positions_scanned']:,} triplet positions scanned

De Novo Sequon Generation:
  Observed: {stats['observed_de_novo_sequons']} sequons
  Expected (random): {stats['expected_de_novo_sequons']:.1f} sequons
  Ratio (observed/expected): {stats['observed_over_expected_ratio']:.3f}

Generation Rates:
  Observed: {stats['observed_rate_percent']:.3f}% per position
  Expected: {stats['expected_rate_percent']:.3f}% per position
  Rate ratio: {stats['rate_ratio']:.3f}

Statistical Test (Binomial):
  p-value: {stats['binomial_p_value']:.2e}
    """

    ax.text(0.1, 0.85, stats_text, ha='left', va='top', fontsize=11,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Interpretation
    ratio = stats['observed_over_expected_ratio']

    if ratio < 0.5:
        interp = "INTERPRETATION: ProteinMPNN STRONGLY AVOIDS generating\nN-X-S/T sequons de novo. Combined with low retention, this\nconfirms ACTIVE BIAS against sequon motifs."
        color = '#e74c3c'
    elif ratio < 0.9:
        interp = "INTERPRETATION: ProteinMPNN MODERATELY AVOIDS generating\nsequons. Suggests bias against sequon motifs."
        color = '#f39c12'
    elif ratio > 1.1:
        interp = "INTERPRETATION: ProteinMPNN generates sequons MORE than\nexpected. Unexpected result requiring investigation."
        color = '#9b59b6'
    else:
        interp = "INTERPRETATION: ProteinMPNN generates sequons at random\nexpectation. No evidence of bias in de novo generation."
        color = '#2ecc71'

    ax.text(0.5, 0.1, interp, ha='center', va='bottom', fontsize=11, fontweight='bold',
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
    print("DE NOVO SEQUON GENERATION VISUALIZATION")
    print("="*70)
    print()

    # Load results
    print("Loading results...")
    de_novo, summary, stats = load_results()
    print(f"Loaded data for {len(summary)} proteins\n")

    # Create figures
    print("Creating figures...")
    plot_observed_vs_expected(summary, stats)
    plot_generation_rate_comparison(stats)
    if len(summary) > 0:
        plot_per_protein_generation(summary)
    plot_summary_figure(stats)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
