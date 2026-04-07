#!/usr/bin/env python3
"""
Create Visualizations for Three-Way Sequon Retention Comparison

Generates figures showing:
1. Overall retention rates across three conditions
2. Per-protein heatmap
3. Score vs retention tradeoff

Author: Claude Code
Date: 2026-01-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = ANALYSIS_DIR / "sequon_retention_threeway_summary.csv"
DETAILED_FILE = ANALYSIS_DIR / "sequon_retention_threeway_detailed.csv"

def create_threeway_bar_chart(summary_df):
    """Figure 1: Three-way comparison bar chart"""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate overall rates with confidence intervals
    conditions = []
    rates = []
    errors = []
    colors_map = {
        'unconstrained': 'steelblue',
        'fixed': 'coral',
        'multifix': 'forestgreen'
    }
    colors = []

    from statsmodels.stats.proportion import proportion_confint

    for condition in ['unconstrained', 'fixed', 'multifix']:
        cond_data = summary_df[summary_df['condition'] == condition]

        if len(cond_data) > 0:
            total = cond_data['sequons_tested'].sum()
            preserved = cond_data['sequons_preserved'].sum()
            rate = preserved / total if total > 0 else 0

            ci_low, ci_high = proportion_confint(preserved, total, method='wilson')
            error = rate - ci_low

            conditions.append(condition.replace('multifix', 'Multi-fix').replace('fixed', 'Single-fix').capitalize())
            rates.append(rate * 100)
            errors.append(error * 100)
            colors.append(colors_map[condition])

    # Create bar chart
    bars = ax.bar(conditions, rates, yerr=errors, capsize=12,
                   color=colors, edgecolor='black', linewidth=2, alpha=0.85)

    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

    # Add significance brackets
    from scipy.stats import fisher_exact

    # Load detailed data for statistical tests
    detailed_df = pd.read_csv(DETAILED_FILE)

    # Unconstrained vs Single-fix
    data_unc = detailed_df[detailed_df['condition'] == 'unconstrained']
    data_fix = detailed_df[detailed_df['condition'] == 'fixed']

    if len(data_unc) > 0 and len(data_fix) > 0:
        contingency = [[data_unc['full_motif_preserved'].sum(), len(data_unc) - data_unc['full_motif_preserved'].sum()],
                      [data_fix['full_motif_preserved'].sum(), len(data_fix) - data_fix['full_motif_preserved'].sum()]]
        _, p1 = fisher_exact(contingency)

        y_bracket1 = max(rates[:2]) + 8
        ax.plot([0, 1], [y_bracket1, y_bracket1], 'k-', linewidth=1.5)
        sig_text = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'ns'
        ax.text(0.5, y_bracket1 + 1.5, sig_text, ha='center', fontsize=13, fontweight='bold')

    # Unconstrained vs Multi-fix
    data_multi = detailed_df[detailed_df['condition'] == 'multifix']

    if len(data_unc) > 0 and len(data_multi) > 0:
        contingency = [[data_unc['full_motif_preserved'].sum(), len(data_unc) - data_unc['full_motif_preserved'].sum()],
                      [data_multi['full_motif_preserved'].sum(), len(data_multi) - data_multi['full_motif_preserved'].sum()]]
        _, p2 = fisher_exact(contingency)

        y_bracket2 = max(rates) + 15
        ax.plot([0, 2], [y_bracket2, y_bracket2], 'k-', linewidth=1.5)
        sig_text = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
        ax.text(1, y_bracket2 + 1.5, sig_text, ha='center', fontsize=13, fontweight='bold')

    ax.set_ylabel('Sequon Retention Rate (%)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Constraint Condition', fontsize=15, fontweight='bold')
    ax.set_title('N-X-S/T Sequon Retention: Effect of Position Constraints',
                 fontsize=16, fontweight='bold', pad=25)
    ax.set_ylim([0, max(rates) + 25])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_threeway_overall_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_per_protein_heatmap(summary_df):
    """Figure 2: Per-protein heatmap across three conditions"""

    # Pivot data
    pivot = summary_df.pivot_table(
        index='pdb_id',
        columns='condition',
        values='retention_rate',
        aggfunc='mean'
    )

    # Reorder columns
    column_order = ['unconstrained', 'fixed', 'multifix']
    pivot = pivot[[c for c in column_order if c in pivot.columns]]

    # Rename columns for display
    pivot.columns = [c.replace('multifix', 'Multi-fix').replace('fixed', 'Single-fix').capitalize()
                     for c in pivot.columns]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(7, len(pivot) * 0.6)))

    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'},
                linewidths=1, linecolor='gray', ax=ax, annot_kws={'fontsize': 11})

    ax.set_xlabel('Constraint Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Protein', fontsize=14, fontweight='bold')
    ax.set_title('Sequon Retention by Protein and Constraint Condition',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_threeway_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_score_comparison(detailed_df):
    """Figure 3: Compare design scores across conditions"""

    # This requires parsing scores from FASTA headers
    # For now, create a placeholder figure showing retention rates

    fig, ax = plt.subplots(figsize=(12, 7))

    conditions_order = ['unconstrained', 'fixed', 'multifix']
    condition_labels = ['Unconstrained', 'Single-fix', 'Multi-fix']
    colors = ['steelblue', 'coral', 'forestgreen']

    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []

    for condition, label, color in zip(conditions_order, condition_labels, colors):
        cond_data = detailed_df[detailed_df['condition'] == condition]
        if len(cond_data) > 0:
            # Get retention rates per design
            design_rates = cond_data.groupby('design_idx')['full_motif_preserved'].mean()
            data_to_plot.append(design_rates.values)
            labels_to_plot.append(label)
            colors_to_plot.append(color)

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1.5, edgecolor='black'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='darkred'),
                        meanprops=dict(linewidth=2, color='blue', linestyle='--'))

        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Per-Design Sequon Retention Rate', fontsize=14, fontweight='bold')
        ax.set_xlabel('Constraint Condition', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Sequon Retention Across Individual Designs',
                     fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.grid(axis='y', alpha=0.3)

        # Add median values as text
        medians = [np.median(d) for d in data_to_plot]
        for i, (label, median) in enumerate(zip(labels_to_plot, medians)):
            ax.text(i + 1, median + 0.05, f'{median*100:.1f}%',
                   ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_threeway_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_statistical_report(summary_df, detailed_df):
    """Generate updated statistical report"""

    report_file = ANALYSIS_DIR / "statistical_report_threeway.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS: THREE-WAY SEQUON RETENTION COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write("CONDITIONS COMPARED:\n")
        f.write("  1. Unconstrained: No positions fixed\n")
        f.write("  2. Single-fix: Only asparagine (N) fixed\n")
        f.write("  3. Multi-fix: All three N-X-S/T positions fixed\n\n")

        # Overall rates
        f.write("1. OVERALL RETENTION RATES\n")
        f.write("-"*80 + "\n\n")

        from statsmodels.stats.proportion import proportion_confint

        for condition in ['unconstrained', 'fixed', 'multifix']:
            cond_data = summary_df[summary_df['condition'] == condition]

            if len(cond_data) > 0:
                total = cond_data['sequons_tested'].sum()
                preserved = cond_data['sequons_preserved'].sum()
                rate = preserved / total if total > 0 else 0

                ci_low, ci_high = proportion_confint(preserved, total, method='wilson')

                condition_name = condition.upper().replace('MULTIFIX', 'MULTI-FIX').replace('FIXED', 'SINGLE-FIX')

                f.write(f"{condition_name}:\n")
                f.write(f"  Sequons tested: {total}\n")
                f.write(f"  Sequons preserved: {preserved}\n")
                f.write(f"  Retention rate: {rate*100:.2f}%\n")
                f.write(f"  95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]\n\n")

        # Statistical comparisons
        f.write("\n2. PAIRWISE STATISTICAL COMPARISONS\n")
        f.write("-"*80 + "\n\n")

        from scipy.stats import fisher_exact

        pairs = [
            ('unconstrained', 'fixed', 'Unconstrained vs Single-fix'),
            ('unconstrained', 'multifix', 'Unconstrained vs Multi-fix'),
            ('fixed', 'multifix', 'Single-fix vs Multi-fix')
        ]

        for cond1, cond2, label in pairs:
            data1 = detailed_df[detailed_df['condition'] == cond1]
            data2 = detailed_df[detailed_df['condition'] == cond2]

            if len(data1) > 0 and len(data2) > 0:
                preserved1 = data1['full_motif_preserved'].sum()
                total1 = len(data1)
                preserved2 = data2['full_motif_preserved'].sum()
                total2 = len(data2)

                contingency = [[preserved1, total1 - preserved1],
                              [preserved2, total2 - preserved2]]
                odds_ratio, p_value = fisher_exact(contingency)

                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'

                rate1 = preserved1 / total1 * 100
                rate2 = preserved2 / total2 * 100
                effect_size = abs(rate2 - rate1)

                f.write(f"{label}:\n")
                f.write(f"  Fisher's exact test p-value: {p_value:.4e} ({sig})\n")
                f.write(f"  Odds ratio: {odds_ratio:.3f}\n")
                f.write(f"  Effect size: {effect_size:.1f} percentage points\n")
                f.write(f"  {cond1}: {rate1:.1f}% → {cond2}: {rate2:.1f}%\n\n")

        # Interpretation
        f.write("\n3. INTERPRETATION\n")
        f.write("-"*80 + "\n\n")

        # Get rates
        unc_data = summary_df[summary_df['condition'] == 'unconstrained']
        fix_data = summary_df[summary_df['condition'] == 'fixed']
        multi_data = summary_df[summary_df['condition'] == 'multifix']

        unc_rate = (unc_data['sequons_preserved'].sum() / unc_data['sequons_tested'].sum() * 100) if len(unc_data) > 0 else 0
        fix_rate = (fix_data['sequons_preserved'].sum() / fix_data['sequons_tested'].sum() * 100) if len(fix_data) > 0 else 0
        multi_rate = (multi_data['sequons_preserved'].sum() / multi_data['sequons_tested'].sum() * 100) if len(multi_data) > 0 else 0

        f.write(f"ProteinMPNN sequon retention rates:\n")
        f.write(f"  - Unconstrained: {unc_rate:.1f}%\n")
        f.write(f"  - Single-fix (N only): {fix_rate:.1f}%\n")
        f.write(f"  - Multi-fix (N-X-S/T): {multi_rate:.1f}%\n\n")

        if multi_rate > 90:
            f.write("CONCLUSION: Multi-position fixing achieves near-complete sequon retention.\n")
            f.write("This confirms that MPNN's sequence optimization is what breaks sequons,\n")
            f.write("not technical limitations. Constraining all three positions resolves the issue.\n\n")

            f.write("PRACTICAL RECOMMENDATION:\n")
            f.write("For glycoprotein design, fix all three N-X-S/T positions at known glycosites.\n")
            f.write("This maintains glycosylation potential while allowing MPNN to optimize elsewhere.\n")
        elif multi_rate > fix_rate + 20:
            f.write("CONCLUSION: Multi-position fixing significantly improves sequon retention.\n")
            f.write("Fixing all three positions is substantially better than fixing N alone.\n\n")

            f.write("PRACTICAL RECOMMENDATION:\n")
            f.write("Use multi-position constraints for critical glycosylation sites.\n")
        else:
            f.write("CONCLUSION: Multi-position fixing shows limited improvement.\n")
            f.write("Additional factors beyond position fixing may be affecting sequon retention.\n")

    print(f"✓ Saved: {report_file}")

def main():
    """Generate all visualizations."""

    print("="*80)
    print("CREATING THREE-WAY COMPARISON VISUALIZATIONS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    summary_df = pd.read_csv(SUMMARY_FILE)
    detailed_df = pd.read_csv(DETAILED_FILE)
    print(f"  Summary: {len(summary_df)} rows")
    print(f"  Detailed: {len(detailed_df)} rows")

    # Check which conditions are present
    conditions_present = detailed_df['condition'].unique()
    print(f"\n  Conditions present: {', '.join(conditions_present)}")

    # Generate figures
    print(f"\nGenerating figures...")

    create_threeway_bar_chart(summary_df)
    create_per_protein_heatmap(summary_df)
    create_score_comparison(detailed_df)

    # Generate report
    print(f"\nGenerating statistical report...")
    generate_statistical_report(summary_df, detailed_df)

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
