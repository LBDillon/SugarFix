#!/usr/bin/env python3
"""
Analyze Relationship Between ProteinMPNN Scores and Sequon Retention

This script analyzes:
1. How MPNN scores relate to sequon retention
2. Whether designs that preserve sequons have different scores
3. Score differences across design conditions

Author: Claude Code
Date: 2026-01-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
OUTPUT_DIR = DATA_DIR / "analysis"

# Load data
SCORES_FILE = OUTPUT_DIR / "proteinmpnn_design_scores_detailed.csv"
SEQUON_FILE = OUTPUT_DIR / "sequon_retention_threeway_detailed.csv"

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_scores_and_sequons():
    """Analyze relationship between MPNN scores and sequon retention."""
    print("="*80)
    print("ANALYZING PROTEINMPNN SCORES AND SEQUON RETENTION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    scores_df = pd.read_csv(SCORES_FILE)
    sequon_df = pd.read_csv(SEQUON_FILE)

    print(f"  Scores: {len(scores_df)} design scores")
    print(f"  Sequons: {len(sequon_df)} sequon retention tests")
    print()

    # Merge datasets
    # Group sequon data by design to get per-design retention rate
    sequon_summary = sequon_df.groupby(['pdb_id', 'chain_id', 'condition', 'design_idx']).agg({
        'sequon_preserved': 'mean',  # Proportion of sequons preserved
        'n_preserved': 'sum'  # Number preserved
    }).reset_index()
    sequon_summary.rename(columns={'sequon_preserved': 'sequon_retention_rate'}, inplace=True)

    # Scores are per-protein-condition, not per-design
    # We need to assign scores to individual designs
    # Group scores by protein and condition
    score_means = scores_df.groupby(['pdb_id', 'chain_id', 'condition'])['score'].agg(['mean', 'std', 'count']).reset_index()

    # For now, let's analyze at the aggregated level
    # Compare mean scores across conditions with sequon retention rates

    print("="*80)
    print("1. SCORE STATISTICS BY CONDITION")
    print("="*80)
    print()

    for condition in sorted(scores_df['condition'].unique()):
        cond_scores = scores_df[scores_df['condition'] == condition]['score']
        print(f"{condition.upper()}:")
        print(f"  N designs: {len(cond_scores)}")
        print(f"  Mean score: {cond_scores.mean():.4f}")
        print(f"  Std: {cond_scores.std():.4f}")
        print()

    # Statistical comparison between conditions
    print("="*80)
    print("2. STATISTICAL COMPARISON BETWEEN CONDITIONS")
    print("="*80)
    print()

    conditions = sorted(scores_df['condition'].unique())
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            cond1, cond2 = conditions[i], conditions[j]
            scores1 = scores_df[scores_df['condition'] == cond1]['score']
            scores2 = scores_df[scores_df['condition'] == cond2]['score']

            if len(scores1) > 0 and len(scores2) > 0:
                stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                mean_diff = scores1.mean() - scores2.mean()

                print(f"{cond1.upper()} vs {cond2.upper()}:")
                print(f"  Mean difference: {mean_diff:.4f}")
                print(f"  p-value: {p_value:.6f}")

                if p_value < 0.001:
                    print("  Significance: *** (p < 0.001)")
                elif p_value < 0.01:
                    print("  Significance: ** (p < 0.01)")
                elif p_value < 0.05:
                    print("  Significance: * (p < 0.05)")
                else:
                    print("  Significance: ns")
                print()

    # Analyze per-protein patterns
    print("="*80)
    print("3. PER-PROTEIN SCORE ANALYSIS")
    print("="*80)
    print()

    protein_stats = scores_df.groupby(['pdb_id', 'condition']).agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    protein_stats.columns = ['pdb_id', 'condition', 'mean_score', 'std_score', 'n_designs']

    for pdb_id in sorted(protein_stats['pdb_id'].unique()):
        protein_data = protein_stats[protein_stats['pdb_id'] == pdb_id]
        print(f"\n{pdb_id}:")

        for _, row in protein_data.iterrows():
            print(f"  {row['condition']:15s}: {row['mean_score']:.4f} ± {row['std_score']:.4f} (n={int(row['n_designs'])})")

    return scores_df, sequon_df, sequon_summary

def create_visualizations(scores_df):
    """Create visualizations."""
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print("="*80)
    print()

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # Figure 1: Score distribution by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    ax = axes[0]
    conditions = sorted(scores_df['condition'].unique())
    data_for_plot = [scores_df[scores_df['condition'] == cond]['score'] for cond in conditions]

    bp = ax.boxplot(data_for_plot, labels=[c.replace('_', '-').title() for c in conditions],
                   patch_artist=True, widths=0.6)

    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('ProteinMPNN Score', fontsize=12)
    ax.set_title('ProteinMPNN Scores by Design Condition', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Violin plot
    ax = axes[1]
    positions = list(range(1, len(conditions) + 1))
    parts = ax.violinplot(data_for_plot, positions=positions, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([c.replace('_', '-').title() for c in conditions])
    ax.set_ylabel('ProteinMPNN Score', fontsize=12)
    ax.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "figures" / "proteinmpnn_scores_by_condition_detailed.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Figure 2: Per-protein comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    protein_stats = scores_df.groupby(['pdb_id', 'condition'])['score'].mean().unstack(fill_value=np.nan)

    x = np.arange(len(protein_stats))
    width = 0.25
    conditions = protein_stats.columns

    for i, condition in enumerate(conditions):
        offset = (i - len(conditions)/2 + 0.5) * width
        values = protein_stats[condition]
        ax.bar(x + offset, values, width, label=condition.replace('_', '-').title(),
              alpha=0.8)

    ax.set_xlabel('Protein', fontsize=12)
    ax.set_ylabel('Mean ProteinMPNN Score', fontsize=12)
    ax.set_title('Per-Protein ProteinMPNN Scores by Condition', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(protein_stats.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "figures" / "proteinmpnn_scores_per_protein_by_condition.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_report(scores_df):
    """Generate analysis report."""
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print("="*80)
    print()

    report_lines = []

    report_lines.append("="*80)
    report_lines.append("PROTEINMPNN SCORE ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Date: 2026-01-20")
    report_lines.append("")

    # Overall statistics
    report_lines.append("1. OVERALL STATISTICS")
    report_lines.append("-"*80)
    report_lines.append("")
    report_lines.append(f"Total designs: {len(scores_df)}")
    report_lines.append(f"Proteins: {scores_df['pdb_id'].nunique()}")
    report_lines.append(f"Mean score: {scores_df['score'].mean():.4f}")
    report_lines.append(f"Std: {scores_df['score'].std():.4f}")
    report_lines.append("")

    # By condition
    report_lines.append("2. SCORES BY DESIGN CONDITION")
    report_lines.append("-"*80)
    report_lines.append("")

    for condition in sorted(scores_df['condition'].unique()):
        cond_scores = scores_df[scores_df['condition'] == condition]['score']
        report_lines.append(f"{condition.upper()}:")
        report_lines.append(f"  N designs: {len(cond_scores)}")
        report_lines.append(f"  Mean: {cond_scores.mean():.4f}")
        report_lines.append(f"  Median: {cond_scores.median():.4f}")
        report_lines.append(f"  Std: {cond_scores.std():.4f}")
        report_lines.append(f"  Min: {cond_scores.min():.4f}")
        report_lines.append(f"  Max: {cond_scores.max():.4f}")
        report_lines.append("")

    # Statistical comparisons
    report_lines.append("3. STATISTICAL COMPARISONS")
    report_lines.append("-"*80)
    report_lines.append("")

    conditions = sorted(scores_df['condition'].unique())
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            cond1, cond2 = conditions[i], conditions[j]
            scores1 = scores_df[scores_df['condition'] == cond1]['score']
            scores2 = scores_df[scores_df['condition'] == cond2]['score']

            if len(scores1) > 0 and len(scores2) > 0:
                stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                mean_diff = scores1.mean() - scores2.mean()

                report_lines.append(f"{cond1.upper()} vs {cond2.upper()}:")
                report_lines.append(f"  Mean scores: {scores1.mean():.4f} vs {scores2.mean():.4f}")
                report_lines.append(f"  Difference: {mean_diff:.4f}")
                report_lines.append(f"  Mann-Whitney U: {stat:.2f}")
                report_lines.append(f"  p-value: {p_value:.6e}")

                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                report_lines.append(f"  Significance: {sig}")
                report_lines.append("")

    # Save report
    output_file = OUTPUT_DIR / "proteinmpnn_scores_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved report: {output_file}")

def main():
    scores_df, sequon_df, sequon_summary = analyze_scores_and_sequons()
    create_visualizations(scores_df)
    generate_report(scores_df)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()

if __name__ == '__main__':
    main()
