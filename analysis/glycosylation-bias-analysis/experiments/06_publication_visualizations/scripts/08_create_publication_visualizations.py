#!/usr/bin/env python3
"""
Create Publication-Quality Visualizations for Sequon Retention Analysis

Generates figures showing:
1. Overall retention rates (unconstrained vs fixed)
2. Per-protein breakdown
3. Statistical comparisons

Author: Claude Code
Date: 2026-01-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "glyco_benchmark"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
SUMMARY_FILE = ANALYSIS_DIR / "sequon_retention_corrected_summary.csv"
DETAILED_FILE = ANALYSIS_DIR / "sequon_retention_corrected_detailed.csv"

def load_data():
    """Load analysis results."""
    summary = pd.read_csv(SUMMARY_FILE)
    detailed = pd.read_csv(DETAILED_FILE)
    return summary, detailed

def create_overall_comparison(summary_df):
    """
    Figure 1: Overall sequon retention rates
    Bar chart comparing unconstrained vs fixed
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate overall rates
    overall_rates = []
    conditions = []
    errors = []

    for condition in ['unconstrained', 'fixed']:
        cond_data = summary_df[summary_df['condition'] == condition]

        # Aggregate across proteins
        total_preserved = cond_data['sequons_preserved'].sum()
        total_tested = cond_data['total_sequons'].sum()
        rate = total_preserved / total_tested if total_tested > 0 else 0

        # Calculate 95% CI (Wilson score interval)
        from statsmodels.stats.proportion import proportion_confint
        ci_low, ci_high = proportion_confint(total_preserved, total_tested, method='wilson')
        error = rate - ci_low

        overall_rates.append(rate * 100)
        conditions.append(condition.capitalize())
        errors.append(error * 100)

    # Create bar chart
    colors = ['steelblue', 'coral']
    bars = ax.bar(conditions, overall_rates, yerr=errors, capsize=10,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars, overall_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Sequon Retention Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
    ax.set_title('Overall N-X-S/T Sequon Retention in ProteinMPNN Designs',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([0, max(overall_rates) + 15])

    # Add significance indicator if appropriate
    if len(overall_rates) == 2:
        # Fisher's exact test
        unc_data = summary_df[summary_df['condition'] == 'unconstrained']
        fix_data = summary_df[summary_df['condition'] == 'fixed']

        unc_preserved = unc_data['sequons_preserved'].sum()
        unc_total = unc_data['total_sequons'].sum()
        fix_preserved = fix_data['sequons_preserved'].sum()
        fix_total = fix_data['total_sequons'].sum()

        from scipy.stats import fisher_exact
        contingency = [[unc_preserved, unc_total - unc_preserved],
                      [fix_preserved, fix_total - fix_preserved]]
        odds_ratio, p_value = fisher_exact(contingency)

        # Add significance annotation
        y_max = max(overall_rates) + 5
        ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(0.5, y_max + 1, sig_text, ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig1_overall_retention_rates.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_per_protein_heatmap(summary_df):
    """
    Figure 2: Per-protein retention rates heatmap
    """
    # Pivot data
    pivot = summary_df.pivot_table(
        index='pdb_id',
        columns='condition',
        values='retention_rate',
        aggfunc='mean'
    )

    # Reorder columns
    pivot = pivot[['unconstrained', 'fixed']] if 'fixed' in pivot.columns else pivot

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.5)))

    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Protein', fontsize=14, fontweight='bold')
    ax.set_title('Sequon Retention by Protein and Condition',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig2_per_protein_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_violin_plot(detailed_df):
    """
    Figure 3: Distribution of sequon preservation across designs
    """
    # Calculate per-design retention rates
    design_rates = detailed_df.groupby(['pdb_id', 'condition', 'design_idx']).agg({
        'full_motif_preserved': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create violin plot
    sns.violinplot(data=design_rates, x='condition', y='full_motif_preserved',
                   ax=ax, palette=['steelblue', 'coral'], inner='box')

    ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sequon Retention Rate', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Sequon Retention Across Designs',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig3_distribution_violin.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_protein_comparison_bars(summary_df):
    """
    Figure 4: Per-protein comparison bars
    """
    # Filter to proteins with both conditions
    proteins_both = summary_df.groupby('pdb_id')['condition'].nunique()
    proteins_both = proteins_both[proteins_both == 2].index

    if len(proteins_both) == 0:
        print("  Skipping protein comparison (no proteins with both conditions)")
        return

    data_both = summary_df[summary_df['pdb_id'].isin(proteins_both)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for grouped bar chart
    pivot = data_both.pivot_table(
        index='pdb_id',
        columns='condition',
        values='retention_rate'
    )

    pivot = pivot * 100  # Convert to percentage

    # Create grouped bar chart
    x = np.arange(len(pivot))
    width = 0.35

    bars1 = ax.bar(x - width/2, pivot['unconstrained'], width,
                   label='Unconstrained', color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot['fixed'], width,
                   label='Fixed', color='coral', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Protein', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sequon Retention Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Sequon Retention by Protein: Unconstrained vs Fixed',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig4_protein_comparison_bars.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_statistical_report(summary_df, detailed_df):
    """
    Generate statistical analysis report
    """
    report_file = ANALYSIS_DIR / "statistical_report.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT: SEQUON RETENTION IN PROTEINMPNN\n")
        f.write("="*80 + "\n\n")

        # Overall comparison
        f.write("1. OVERALL RETENTION RATES\n")
        f.write("-"*80 + "\n\n")

        for condition in ['unconstrained', 'fixed']:
            cond_data = summary_df[summary_df['condition'] == condition]
            total_preserved = cond_data['sequons_preserved'].sum()
            total_tested = cond_data['total_sequons'].sum()
            rate = total_preserved / total_tested if total_tested > 0 else 0

            from statsmodels.stats.proportion import proportion_confint
            ci_low, ci_high = proportion_confint(total_preserved, total_tested, method='wilson')

            f.write(f"{condition.upper()}:\n")
            f.write(f"  Sequons tested: {total_tested}\n")
            f.write(f"  Sequons preserved: {total_preserved}\n")
            f.write(f"  Retention rate: {rate*100:.2f}%\n")
            f.write(f"  95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]\n\n")

        # Fisher's exact test
        f.write("\n2. STATISTICAL COMPARISON\n")
        f.write("-"*80 + "\n\n")

        unc_data = summary_df[summary_df['condition'] == 'unconstrained']
        fix_data = summary_df[summary_df['condition'] == 'fixed']

        unc_preserved = unc_data['sequons_preserved'].sum()
        unc_total = unc_data['total_sequons'].sum()
        fix_preserved = fix_data['sequons_preserved'].sum()
        fix_total = fix_data['total_sequons'].sum()

        from scipy.stats import fisher_exact
        contingency = [[unc_preserved, unc_total - unc_preserved],
                      [fix_preserved, fix_total - fix_preserved]]
        odds_ratio, p_value = fisher_exact(contingency)

        f.write("Fisher's Exact Test (Unconstrained vs Fixed):\n")
        f.write(f"  Odds ratio: {odds_ratio:.3f}\n")
        f.write(f"  P-value: {p_value:.4e}\n")
        f.write(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}\n\n")

        # Interpretation
        f.write("\n3. INTERPRETATION\n")
        f.write("-"*80 + "\n\n")

        unc_rate = unc_preserved / unc_total * 100
        fix_rate = fix_preserved / fix_total * 100

        f.write(f"ProteinMPNN preserves N-X-S/T sequons at the original position in:\n")
        f.write(f"  - {unc_rate:.1f}% of unconstrained designs\n")
        f.write(f"  - {fix_rate:.1f}% of designs with fixed asparagine\n\n")

        if unc_rate < 20:
            f.write("CONCLUSION: ProteinMPNN ACTIVELY DESTROYS glycosylation sequons.\n")
            f.write("This indicates a strong need for glyco-aware protein design methods.\n")
        elif unc_rate < 50:
            f.write("CONCLUSION: ProteinMPNN MODERATELY preserves glycosylation sequons.\n")
            f.write("Glyco-aware methods may improve sequon retention.\n")
        else:
            f.write("CONCLUSION: ProteinMPNN LARGELY preserves glycosylation sequons.\n")
            f.write("Existing tools may be sufficient for glycoprotein design.\n")

        # Effect size
        f.write(f"\nEffect size (absolute difference): {fix_rate - unc_rate:.1f} percentage points\n")

    print(f"✓ Saved: {report_file}")

def main():
    """Generate all visualizations and reports."""
    print("="*80)
    print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("="*80)

    # Load data
    print("\nLoading data...")
    summary_df, detailed_df = load_data()
    print(f"  Summary: {len(summary_df)} rows")
    print(f"  Detailed: {len(detailed_df)} rows")

    # Generate figures
    print(f"\nGenerating figures...")
    print(f"  Output directory: {OUTPUT_DIR}")

    create_overall_comparison(summary_df)
    create_per_protein_heatmap(summary_df)
    create_violin_plot(detailed_df)
    create_protein_comparison_bars(summary_df)

    # Generate statistical report
    print(f"\nGenerating statistical report...")
    generate_statistical_report(summary_df, detailed_df)

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"Statistical report: {ANALYSIS_DIR / 'statistical_report.txt'}")

if __name__ == '__main__':
    main()
