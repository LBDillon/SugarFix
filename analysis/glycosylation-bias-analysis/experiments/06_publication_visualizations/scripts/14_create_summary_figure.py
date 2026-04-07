#!/usr/bin/env python3
"""
Create Comprehensive Summary Figure: Retention + Scores

Creates a publication-ready figure showing both sequon retention
and MPNN scores across all three design conditions.

Author: Claude Code
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
OUTPUT_DIR = DATA_DIR / "analysis" / "figures"

# Input files
RETENTION_FILE = DATA_DIR / "analysis" / "sequon_retention_threeway_detailed.csv"
SCORES_FILE = DATA_DIR / "analysis" / "proteinmpnn_scores_all_conditions.csv"

# =============================================================================
# CREATE FIGURE
# =============================================================================

def create_summary_figure():
    """Create comprehensive summary figure."""

    print("Loading data...")
    retention_df = pd.read_csv(RETENTION_FILE)
    scores_df = pd.read_csv(SCORES_FILE)

    # Calculate retention rates
    retention_summary = retention_df.groupby('condition').agg({
        'sequon_preserved': lambda x: (x.sum() / len(x)) * 100
    }).reset_index()
    retention_summary.columns = ['condition', 'retention_pct']

    # Calculate mean scores
    score_summary = scores_df.groupby('condition')['score'].agg(['mean', 'std', 'sem']).reset_index()

    # Map condition names
    condition_map = {
        'unconstrained': 'Unconstrained\n(no fixes)',
        'fixed': 'Single-fix\n(N only)',
        'single_fix': 'Single-fix\n(N only)',
        'multifix': 'Multi-fix\n(N-X-S/T)',
        'multi_fix': 'Multi-fix\n(N-X-S/T)'
    }

    retention_summary['condition_label'] = retention_summary['condition'].map(condition_map)
    score_summary['condition_label'] = score_summary['condition'].map(condition_map)

    # Order conditions
    condition_order = ['Unconstrained\n(no fixes)', 'Single-fix\n(N only)', 'Multi-fix\n(N-X-S/T)']
    retention_summary = retention_summary.set_index('condition_label').loc[condition_order].reset_index()
    score_summary = score_summary.set_index('condition_label').loc[condition_order].reset_index()

    # Create figure
    print("Creating figure...")
    fig = plt.figure(figsize=(14, 6))

    # Define colors
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green

    # Panel A: Sequon Retention
    ax1 = plt.subplot(1, 2, 1)

    bars = ax1.bar(range(len(retention_summary)), retention_summary['retention_pct'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Sequon Retention (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Design Condition', fontsize=14, fontweight='bold')
    ax1.set_title('A. N-Glycosylation Sequon Retention', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(retention_summary)))
    ax1.set_xticklabels(retention_summary['condition_label'], fontsize=11)
    ax1.set_ylim(0, 55)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, retention_summary['retention_pct'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Add fold-change annotations
    baseline = retention_summary.iloc[0]['retention_pct']
    for i in range(1, len(retention_summary)):
        val = retention_summary.iloc[i]['retention_pct']
        fold_change = val / baseline
        ax1.text(i, val/2, f'{fold_change:.1f}x',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

    # Panel B: MPNN Scores
    ax2 = plt.subplot(1, 2, 2)

    bars = ax2.bar(range(len(score_summary)), score_summary['mean'],
                   yerr=score_summary['sem'], capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax2.set_ylabel('Mean ProteinMPNN Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Design Condition', fontsize=14, fontweight='bold')
    ax2.set_title('B. ProteinMPNN Scoring Preference', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(score_summary)))
    ax2.set_xticklabels(score_summary['condition_label'], fontsize=11)
    ax2.set_ylim(0.90, 0.98)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, score_summary['mean'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Add significance annotations
    # Single-fix is significantly lower
    ax2.plot([0, 1], [0.975, 0.975], 'k-', linewidth=1.5)
    ax2.text(0.5, 0.976, '**', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Multi-fix vs unconstrained: ns
    ax2.plot([0, 2], [0.970, 0.970], 'k-', linewidth=1.5)
    ax2.text(1, 0.971, 'ns', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add overall title
    fig.suptitle('ProteinMPNN Shows U-Shaped Scoring Pattern: Multi-Fix Achieves High Retention Without Penalty',
                 fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file = OUTPUT_DIR / "sequon_retention_and_scores_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    # Create version for presentation (higher contrast)
    fig.patch.set_facecolor('white')
    output_file2 = OUTPUT_DIR / "sequon_retention_and_scores_summary_presentation.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file2}")

    plt.close()

def main():
    print("="*80)
    print("CREATING SUMMARY FIGURE")
    print("="*80)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    create_summary_figure()

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
