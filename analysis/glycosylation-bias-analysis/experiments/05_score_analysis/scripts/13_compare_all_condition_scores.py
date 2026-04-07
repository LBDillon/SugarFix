#!/usr/bin/env python3
"""
Compare ProteinMPNN Scores Across All Three Design Conditions

Analyzes how MPNN scores differ between:
1. Unconstrained designs (no positions fixed)
2. Single-fix designs (only N positions fixed)
3. Multi-fix designs (all N-X-S/T positions fixed)

Author: Claude Code
Date: 2026-01-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
DESIGNS_DIR = DATA_DIR / "designs"
OUTPUT_DIR = DATA_DIR / "analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

MANIFEST_FILE = DATA_DIR / "manifests" / "expanded_manifest_validated.csv"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_manifest():
    """Load manifest with glycosite information."""
    df = pd.read_csv(MANIFEST_FILE)
    return df

def extract_scores_from_fasta(fasta_path):
    """Extract scores from FASTA headers."""
    scores = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Look for score=X.XXX pattern
                match = re.search(r'score[=_](\d+\.\d+)', line)
                if match:
                    scores.append(float(match.group(1)))
    return scores

def collect_all_scores():
    """Collect scores from all design files across all three conditions."""
    manifest = load_manifest()

    all_scores = []

    for _, row in manifest.iterrows():
        pdb_id = row['pdb_id']
        chain_id = row['chain_id']
        n_glycosites = row['n_glycosites']

        # Check for each condition
        conditions = {
            'unconstrained': f"{pdb_id}_{chain_id}_unconstrained.fasta",
            'single_fix': f"{pdb_id}_{chain_id}_fixed.fasta",
            'multi_fix': f"{pdb_id}_{chain_id}_multifix.fasta"
        }

        for condition, filename in conditions.items():
            fasta_path = DESIGNS_DIR / filename
            if fasta_path.exists():
                scores = extract_scores_from_fasta(fasta_path)
                for score in scores:
                    all_scores.append({
                        'pdb_id': pdb_id,
                        'chain_id': chain_id,
                        'condition': condition,
                        'n_glycosites': n_glycosites,
                        'score': score
                    })

    return pd.DataFrame(all_scores)

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_scores(df):
    """Analyze score distributions across conditions."""
    print("="*80)
    print("PROTEINMPNN SCORES ACROSS ALL THREE CONDITIONS")
    print("="*80)
    print()

    print("Dataset Summary:")
    print(f"  Total scores: {len(df)}")
    print(f"  Proteins: {df['pdb_id'].nunique()}")
    print(f"  Conditions: {', '.join(sorted(df['condition'].unique()))}")
    print()

    print("="*80)
    print("1. OVERALL SCORE STATISTICS BY CONDITION")
    print("="*80)
    print()

    condition_order = ['unconstrained', 'single_fix', 'multi_fix']
    condition_labels = {
        'unconstrained': 'Unconstrained (no fixes)',
        'single_fix': 'Single-fix (N only)',
        'multi_fix': 'Multi-fix (N-X-S/T)'
    }

    stats_summary = []

    for condition in condition_order:
        if condition in df['condition'].values:
            cond_data = df[df['condition'] == condition]['score']
            print(f"{condition_labels[condition]}:")
            print(f"  N designs: {len(cond_data)}")
            print(f"  Mean: {cond_data.mean():.4f}")
            print(f"  Median: {cond_data.median():.4f}")
            print(f"  Std: {cond_data.std():.4f}")
            print(f"  Min: {cond_data.min():.4f}")
            print(f"  Max: {cond_data.max():.4f}")
            print()

            stats_summary.append({
                'condition': condition,
                'label': condition_labels[condition],
                'n': len(cond_data),
                'mean': cond_data.mean(),
                'std': cond_data.std()
            })

    # Statistical comparisons
    print("="*80)
    print("2. PAIRWISE STATISTICAL COMPARISONS")
    print("="*80)
    print()

    comparisons = []

    for i in range(len(condition_order)):
        for j in range(i+1, len(condition_order)):
            cond1, cond2 = condition_order[i], condition_order[j]

            if cond1 in df['condition'].values and cond2 in df['condition'].values:
                scores1 = df[df['condition'] == cond1]['score']
                scores2 = df[df['condition'] == cond2]['score']

                stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                mean_diff = scores1.mean() - scores2.mean()

                print(f"{condition_labels[cond1]}")
                print(f"  vs")
                print(f"{condition_labels[cond2]}:")
                print(f"  Mean scores: {scores1.mean():.4f} vs {scores2.mean():.4f}")
                print(f"  Difference: {mean_diff:+.4f}")
                print(f"  Mann-Whitney U: {stat:.2f}")
                print(f"  p-value: {p_value:.6e}")

                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                print(f"  Significance: {sig}")
                print()

                comparisons.append({
                    'cond1': cond1,
                    'cond2': cond2,
                    'mean_diff': mean_diff,
                    'p_value': p_value,
                    'sig': sig
                })

    # Per-protein analysis
    print("="*80)
    print("3. PER-PROTEIN SCORE PATTERNS")
    print("="*80)
    print()

    protein_stats = df.groupby(['pdb_id', 'condition'])['score'].agg(['mean', 'std', 'count']).reset_index()

    for pdb_id in sorted(df['pdb_id'].unique()):
        protein_data = protein_stats[protein_stats['pdb_id'] == pdb_id]
        print(f"\n{pdb_id}:")

        for condition in condition_order:
            cond_data = protein_data[protein_data['condition'] == condition]
            if len(cond_data) > 0:
                row = cond_data.iloc[0]
                print(f"  {condition_labels[condition]:30s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

    return stats_summary, comparisons

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df):
    """Create comprehensive visualizations."""
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print("="*80)
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    condition_order = ['unconstrained', 'single_fix', 'multi_fix']
    condition_labels = ['Unconstrained\n(no fixes)', 'Single-fix\n(N only)', 'Multi-fix\n(N-X-S/T)']
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green

    # Figure 1: Box + Violin plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    ax = axes[0]
    data_for_plot = [df[df['condition'] == cond]['score'] for cond in condition_order]

    bp = ax.boxplot(data_for_plot, labels=condition_labels,
                   patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('ProteinMPNN Score', fontsize=13, fontweight='bold')
    ax.set_title('ProteinMPNN Scores Across Design Conditions', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean values as text
    for i, cond in enumerate(condition_order):
        mean_val = df[df['condition'] == cond]['score'].mean()
        ax.text(i+1, mean_val, f'{mean_val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Violin plot
    ax = axes[1]
    positions = list(range(1, len(condition_order) + 1))
    parts = ax.violinplot(data_for_plot, positions=positions, widths=0.7,
                         showmeans=True, showextrema=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('ProteinMPNN Score', fontsize=13, fontweight='bold')
    ax.set_title('Score Distribution by Condition', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = FIGURES_DIR / "mpnn_scores_all_three_conditions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Figure 2: Per-protein heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    protein_means = df.groupby(['pdb_id', 'condition'])['score'].mean().unstack(fill_value=np.nan)
    protein_means = protein_means[condition_order]  # Reorder columns

    sns.heatmap(protein_means, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.7, vmax=1.0, cbar_kws={'label': 'Mean MPNN Score'},
                ax=ax, linewidths=0.5)

    ax.set_xlabel('Design Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protein', fontsize=12, fontweight='bold')
    ax.set_title('Mean ProteinMPNN Scores by Protein and Condition',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticklabels(['Unconstrained', 'Single-fix', 'Multi-fix'], rotation=45, ha='right')

    plt.tight_layout()
    output_file = FIGURES_DIR / "mpnn_scores_heatmap_all_conditions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Figure 3: Score change progression
    fig, ax = plt.subplots(figsize=(12, 7))

    for pdb_id in sorted(df['pdb_id'].unique()):
        protein_data = df[df['pdb_id'] == pdb_id]
        means = []
        for condition in condition_order:
            cond_data = protein_data[protein_data['condition'] == condition]['score']
            if len(cond_data) > 0:
                means.append(cond_data.mean())
            else:
                means.append(np.nan)

        if not all(np.isnan(means)):
            ax.plot(range(len(condition_order)), means, marker='o', linewidth=2,
                   markersize=8, label=pdb_id, alpha=0.7)

    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('Mean ProteinMPNN Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Design Condition', fontsize=13, fontweight='bold')
    ax.set_title('Score Progression Across Constraint Conditions',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Protein')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = FIGURES_DIR / "mpnn_scores_progression.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df, stats_summary, comparisons):
    """Generate comprehensive report."""
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print("="*80)
    print()

    report_lines = []

    report_lines.append("="*80)
    report_lines.append("PROTEINMPNN SCORE COMPARISON: ALL THREE CONDITIONS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Date: 2026-01-21")
    report_lines.append("")
    report_lines.append("Conditions analyzed:")
    report_lines.append("  1. Unconstrained: No positions fixed")
    report_lines.append("  2. Single-fix: Only asparagine (N) positions fixed")
    report_lines.append("  3. Multi-fix: All three N-X-S/T positions fixed")
    report_lines.append("")

    # Summary statistics
    report_lines.append("="*80)
    report_lines.append("1. SCORE STATISTICS BY CONDITION")
    report_lines.append("="*80)
    report_lines.append("")

    for stat in stats_summary:
        report_lines.append(f"{stat['label']}:")
        report_lines.append(f"  N designs: {stat['n']}")
        report_lines.append(f"  Mean score: {stat['mean']:.4f}")
        report_lines.append(f"  Std: {stat['std']:.4f}")
        report_lines.append("")

    # Statistical comparisons
    report_lines.append("="*80)
    report_lines.append("2. STATISTICAL COMPARISONS")
    report_lines.append("="*80)
    report_lines.append("")

    condition_labels = {
        'unconstrained': 'Unconstrained',
        'single_fix': 'Single-fix',
        'multi_fix': 'Multi-fix'
    }

    for comp in comparisons:
        report_lines.append(f"{condition_labels[comp['cond1']]} vs {condition_labels[comp['cond2']]}:")
        report_lines.append(f"  Mean difference: {comp['mean_diff']:+.4f}")
        report_lines.append(f"  p-value: {comp['p_value']:.6e}")
        report_lines.append(f"  Significance: {comp['sig']}")
        report_lines.append("")

    # Key findings
    report_lines.append("="*80)
    report_lines.append("3. KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")

    unconstrained_mean = df[df['condition'] == 'unconstrained']['score'].mean()
    single_fix_mean = df[df['condition'] == 'single_fix']['score'].mean()
    multi_fix_mean = df[df['condition'] == 'multi_fix']['score'].mean()

    report_lines.append(f"Score progression:")
    report_lines.append(f"  Unconstrained:  {unconstrained_mean:.4f} (baseline)")
    report_lines.append(f"  Single-fix:     {single_fix_mean:.4f} ({single_fix_mean - unconstrained_mean:+.4f})")
    report_lines.append(f"  Multi-fix:      {multi_fix_mean:.4f} ({multi_fix_mean - unconstrained_mean:+.4f})")
    report_lines.append("")
    report_lines.append("INTERPRETATION:")
    report_lines.append("")

    if multi_fix_mean < single_fix_mean < unconstrained_mean:
        report_lines.append("ProteinMPNN scores DECREASE as more positions are constrained:")
        report_lines.append("  - Each additional constraint reduces the model's score")
        report_lines.append("  - The model actively prefers sequences WITHOUT fixed positions")
        report_lines.append("  - This confirms ProteinMPNN's anti-glycosylation bias")
        report_lines.append("")
        report_lines.append("The scoring pattern (unconstrained > single-fix > multi-fix) demonstrates")
        report_lines.append("that the model has learned to disfavor glycosylation sequons. Even when")
        report_lines.append("forced to preserve these motifs, the model assigns lower scores,")
        report_lines.append("indicating it views them as suboptimal sequence features.")

    # Save report
    output_file = OUTPUT_DIR / "mpnn_scores_three_conditions_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved: {output_file}")

    # Also save CSV
    df.to_csv(OUTPUT_DIR / "proteinmpnn_scores_all_conditions.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'proteinmpnn_scores_all_conditions.csv'}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ANALYZING PROTEINMPNN SCORES: ALL THREE CONDITIONS")
    print("="*80)
    print()

    # Collect scores
    print("Collecting scores from all design files...")
    df = collect_all_scores()
    print(f"✓ Collected {len(df)} scores from {df['pdb_id'].nunique()} proteins")
    print()

    # Analyze
    stats_summary, comparisons = analyze_scores(df)

    # Visualize
    create_visualizations(df)

    # Report
    generate_report(df, stats_summary, comparisons)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Key Result:")
    unconstrained_mean = df[df['condition'] == 'unconstrained']['score'].mean()
    single_fix_mean = df[df['condition'] == 'single_fix']['score'].mean()
    multi_fix_mean = df[df['condition'] == 'multi_fix']['score'].mean()

    print(f"  Unconstrained: {unconstrained_mean:.4f}")
    print(f"  Single-fix:    {single_fix_mean:.4f} ({single_fix_mean - unconstrained_mean:+.4f})")
    print(f"  Multi-fix:     {multi_fix_mean:.4f} ({multi_fix_mean - unconstrained_mean:+.4f})")
    print()

if __name__ == '__main__':
    main()
