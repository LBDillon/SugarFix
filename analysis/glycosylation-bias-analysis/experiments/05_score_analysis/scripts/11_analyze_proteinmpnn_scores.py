#!/usr/bin/env python3
"""
Analyze ProteinMPNN Scores: Glycoproteins vs Controls

This script analyzes:
1. ProteinMPNN scores for wild-type (WT) sequences
2. ProteinMPNN scores for designed sequences
3. Comparison between glycoproteins and controls

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
DESIGNS_DIR = DATA_DIR / "designs"
MANIFEST_PATH = DATA_DIR / "manifests" / "expanded_manifest_validated.csv"
OUTPUT_DIR = DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_fasta_scores(fasta_file):
    """
    Parse FASTA files and extract ProteinMPNN scores from headers.
    Returns list of scores.
    """
    scores = []

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Example header: >design_0_unconstrained_score_-2.345
                # or: >T=0.1, sample=0, score=-2.345, global_score=-2.345, fixed_chains=[''], ...

                # Try different score extraction patterns
                if '_score_' in line:
                    # Format: design_0_unconstrained_score_-2.345
                    try:
                        score_part = line.split('_score_')[1].split()[0]
                        score = float(score_part)
                        scores.append(score)
                    except:
                        pass
                elif 'score=' in line:
                    # Format: T=0.1, sample=0, score=-2.345
                    try:
                        score_part = line.split('score=')[1].split(',')[0]
                        score = float(score_part)
                        scores.append(score)
                    except:
                        pass

    return scores

def load_manifest():
    """Load the manifest with protein information."""
    return pd.read_csv(MANIFEST_PATH)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_design_scores():
    """
    Analyze ProteinMPNN scores from designed sequences.
    Compare glycoproteins vs controls and different design conditions.
    """
    print("="*80)
    print("ANALYZING PROTEINMPNN SCORES FOR DESIGNED SEQUENCES")
    print("="*80)
    print()

    manifest = load_manifest()

    results = []

    # Iterate through all design files
    for design_file in DESIGNS_DIR.glob("*.fasta"):
        filename = design_file.name

        # Parse filename: PDBID_CHAIN_CONDITION.fasta
        parts = filename.replace('.fasta', '').split('_')
        if len(parts) < 3:
            continue

        pdb_id = parts[0]
        chain_id = parts[1]
        condition = '_'.join(parts[2:])  # Handle multifix, unconstrained, fixed

        # Get protein class from manifest
        protein_info = manifest[(manifest['pdb_id'] == pdb_id) &
                               (manifest['chain_id'] == chain_id)]

        if len(protein_info) == 0:
            continue

        # Determine if it's a glycoprotein (has sequons in manifest)
        n_glycosites = int(protein_info['n_glycosites'].iloc[0])
        is_glycoprotein = n_glycosites > 0
        protein_class = "glycoprotein" if is_glycoprotein else "control"

        # Parse scores
        scores = parse_fasta_scores(design_file)

        if len(scores) > 0:
            for score in scores:
                results.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'condition': condition,
                    'protein_class': protein_class,
                    'n_glycosites': n_glycosites,
                    'score': score,
                    'is_glycoprotein': is_glycoprotein
                })

    if len(results) == 0:
        print("No scores found in design files!")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    output_file = OUTPUT_DIR / "proteinmpnn_design_scores_detailed.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved detailed scores: {output_file}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total designs analyzed: {len(df)}")
    print(f"Unique proteins: {df['pdb_id'].nunique()}")
    print(f"Glycoproteins: {df[df['is_glycoprotein']]['pdb_id'].nunique()}")
    print(f"Controls: {df[~df['is_glycoprotein']]['pdb_id'].nunique()}")
    print()

    # Overall statistics
    print("Overall Score Statistics:")
    print(f"  Mean: {df['score'].mean():.3f}")
    print(f"  Median: {df['score'].median():.3f}")
    print(f"  Std: {df['score'].std():.3f}")
    print()

    # By protein class
    print("By Protein Class:")
    for protein_class in ['glycoprotein', 'control']:
        class_df = df[df['protein_class'] == protein_class]
        if len(class_df) > 0:
            print(f"\n  {protein_class.upper()}:")
            print(f"    N designs: {len(class_df)}")
            print(f"    Mean score: {class_df['score'].mean():.3f}")
            print(f"    Median score: {class_df['score'].median():.3f}")
            print(f"    Std: {class_df['score'].std():.3f}")

    # By condition
    print("\nBy Design Condition:")
    for condition in df['condition'].unique():
        cond_df = df[df['condition'] == condition]
        print(f"\n  {condition.upper()}:")
        print(f"    N designs: {len(cond_df)}")
        print(f"    Mean score: {cond_df['score'].mean():.3f}")

        # Compare glycoproteins vs controls for this condition
        glyco_scores = cond_df[cond_df['is_glycoprotein']]['score']
        control_scores = cond_df[~cond_df['is_glycoprotein']]['score']

        if len(glyco_scores) > 0 and len(control_scores) > 0:
            print(f"    Glycoproteins: {glyco_scores.mean():.3f} (n={len(glyco_scores)})")
            print(f"    Controls: {control_scores.mean():.3f} (n={len(control_scores)})")

            # Statistical test
            stat, p_value = stats.mannwhitneyu(glyco_scores, control_scores, alternative='two-sided')
            print(f"    Mann-Whitney U test: p={p_value:.4f}")

    # Statistical comparison: glycoproteins vs controls
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON: GLYCOPROTEINS VS CONTROLS")
    print(f"{'='*80}\n")

    glyco_scores = df[df['is_glycoprotein']]['score']
    control_scores = df[~df['is_glycoprotein']]['score']

    if len(glyco_scores) > 0 and len(control_scores) > 0:
        print(f"Glycoproteins: n={len(glyco_scores)}, mean={glyco_scores.mean():.3f}, std={glyco_scores.std():.3f}")
        print(f"Controls: n={len(control_scores)}, mean={control_scores.mean():.3f}, std={control_scores.std():.3f}")
        print()

        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(glyco_scores, control_scores, alternative='two-sided')
        print(f"Mann-Whitney U test: U={stat:.2f}, p={p_value:.6f}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((glyco_scores.std()**2 + control_scores.std()**2) / 2)
        cohens_d = (glyco_scores.mean() - control_scores.mean()) / pooled_std
        print(f"Cohen's d (effect size): {cohens_d:.3f}")

        if p_value < 0.001:
            print("Result: Highly significant difference (***)")
        elif p_value < 0.01:
            print("Result: Very significant difference (**)")
        elif p_value < 0.05:
            print("Result: Significant difference (*)")
        else:
            print("Result: No significant difference (ns)")

    return df

def create_visualizations(df):
    """Create visualizations comparing scores."""
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # Figure 1: Overall comparison - Glycoproteins vs Controls
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    ax = axes[0]
    data_for_plot = [
        df[df['is_glycoprotein']]['score'],
        df[~df['is_glycoprotein']]['score']
    ]
    bp = ax.boxplot(data_for_plot, labels=['Glycoproteins', 'Controls'],
                    patch_artist=True, widths=0.6)

    # Color boxes
    colors = ['#ff9999', '#66b3ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('ProteinMPNN Score', fontsize=12)
    ax.set_title('ProteinMPNN Scores: Glycoproteins vs Controls', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    glyco_mean = df[df['is_glycoprotein']]['score'].mean()
    control_mean = df[~df['is_glycoprotein']]['score'].mean()
    ax.text(0.02, 0.98, f'Glycoprotein mean: {glyco_mean:.3f}\nControl mean: {control_mean:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Violin plot
    ax = axes[1]
    parts = ax.violinplot([df[df['is_glycoprotein']]['score'],
                           df[~df['is_glycoprotein']]['score']],
                          positions=[1, 2], widths=0.7, showmeans=True,
                          showextrema=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Glycoproteins', 'Controls'])
    ax.set_ylabel('ProteinMPNN Score', fontsize=12)
    ax.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "figures" / "proteinmpnn_scores_glyco_vs_control.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # Figure 2: By condition
    conditions = df['condition'].unique()
    n_conditions = len(conditions)

    if n_conditions > 0:
        fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 5))
        if n_conditions == 1:
            axes = [axes]

        for ax, condition in zip(axes, conditions):
            cond_df = df[df['condition'] == condition]

            data_for_plot = [
                cond_df[cond_df['is_glycoprotein']]['score'],
                cond_df[~cond_df['is_glycoprotein']]['score']
            ]

            bp = ax.boxplot(data_for_plot, labels=['Glyco', 'Control'],
                           patch_artist=True, widths=0.6)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('ProteinMPNN Score', fontsize=12)
            ax.set_title(f'{condition.replace("_", "-").title()}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add sample sizes
            n_glyco = len(data_for_plot[0])
            n_control = len(data_for_plot[1])
            ax.text(0.02, 0.98, f'n_glyco={n_glyco}\nn_control={n_control}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_file = OUTPUT_DIR / "figures" / "proteinmpnn_scores_by_condition.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()

    # Figure 3: Per-protein comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    # Group by protein and protein class
    protein_stats = df.groupby(['pdb_id', 'protein_class'])['score'].agg(['mean', 'std', 'count']).reset_index()
    protein_stats = protein_stats.sort_values('mean')

    # Separate glycoproteins and controls
    glyco_stats = protein_stats[protein_stats['protein_class'] == 'glycoprotein']
    control_stats = protein_stats[protein_stats['protein_class'] == 'control']

    x_pos = np.arange(len(protein_stats))

    colors_per_protein = ['#ff9999' if pc == 'glycoprotein' else '#66b3ff'
                         for pc in protein_stats['protein_class']]

    ax.bar(x_pos, protein_stats['mean'], yerr=protein_stats['std'],
           color=colors_per_protein, alpha=0.7, capsize=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(protein_stats['pdb_id'], rotation=45, ha='right')
    ax.set_ylabel('Mean ProteinMPNN Score', fontsize=12)
    ax.set_title('Per-Protein ProteinMPNN Scores', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff9999', alpha=0.7, label='Glycoprotein'),
                      Patch(facecolor='#66b3ff', alpha=0.7, label='Control')]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    output_file = OUTPUT_DIR / "figures" / "proteinmpnn_scores_per_protein.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def generate_report(df):
    """Generate text report with statistical analysis."""
    print(f"\n{'='*80}")
    print("GENERATING STATISTICAL REPORT")
    print(f"{'='*80}\n")

    report_lines = []

    report_lines.append("="*80)
    report_lines.append("PROTEINMPNN SCORE ANALYSIS: GLYCOPROTEINS VS CONTROLS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Analysis Date: 2026-01-20")
    report_lines.append(f"Total designs analyzed: {len(df)}")
    report_lines.append(f"Unique proteins: {df['pdb_id'].nunique()}")
    report_lines.append("")

    # Overall statistics
    report_lines.append("1. OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for protein_class in ['glycoprotein', 'control']:
        class_df = df[df['protein_class'] == protein_class]
        if len(class_df) > 0:
            report_lines.append(f"{protein_class.upper()}:")
            report_lines.append(f"  N designs: {len(class_df)}")
            report_lines.append(f"  N proteins: {class_df['pdb_id'].nunique()}")
            report_lines.append(f"  Mean score: {class_df['score'].mean():.4f}")
            report_lines.append(f"  Median score: {class_df['score'].median():.4f}")
            report_lines.append(f"  Std deviation: {class_df['score'].std():.4f}")
            report_lines.append(f"  Min score: {class_df['score'].min():.4f}")
            report_lines.append(f"  Max score: {class_df['score'].max():.4f}")
            report_lines.append("")

    # Statistical comparison
    report_lines.append("2. STATISTICAL COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")

    glyco_scores = df[df['is_glycoprotein']]['score']
    control_scores = df[~df['is_glycoprotein']]['score']

    if len(glyco_scores) > 0 and len(control_scores) > 0:
        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(glyco_scores, control_scores, alternative='two-sided')

        report_lines.append("Mann-Whitney U Test:")
        report_lines.append(f"  U-statistic: {stat:.2f}")
        report_lines.append(f"  p-value: {p_value:.6e}")

        if p_value < 0.001:
            report_lines.append("  Significance: *** (p < 0.001)")
        elif p_value < 0.01:
            report_lines.append("  Significance: ** (p < 0.01)")
        elif p_value < 0.05:
            report_lines.append("  Significance: * (p < 0.05)")
        else:
            report_lines.append("  Significance: ns (not significant)")

        report_lines.append("")

        # Effect size
        pooled_std = np.sqrt((glyco_scores.std()**2 + control_scores.std()**2) / 2)
        cohens_d = (glyco_scores.mean() - control_scores.mean()) / pooled_std

        report_lines.append("Effect Size (Cohen's d):")
        report_lines.append(f"  d = {cohens_d:.4f}")

        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        report_lines.append(f"  Interpretation: {effect_interpretation} effect")
        report_lines.append("")

    # By condition analysis
    report_lines.append("3. ANALYSIS BY DESIGN CONDITION")
    report_lines.append("-" * 80)
    report_lines.append("")

    for condition in sorted(df['condition'].unique()):
        cond_df = df[df['condition'] == condition]
        report_lines.append(f"{condition.upper()}:")

        glyco_cond = cond_df[cond_df['is_glycoprotein']]['score']
        control_cond = cond_df[~cond_df['is_glycoprotein']]['score']

        if len(glyco_cond) > 0:
            report_lines.append(f"  Glycoproteins: n={len(glyco_cond)}, mean={glyco_cond.mean():.4f}, std={glyco_cond.std():.4f}")
        if len(control_cond) > 0:
            report_lines.append(f"  Controls: n={len(control_cond)}, mean={control_cond.mean():.4f}, std={control_cond.std():.4f}")

        if len(glyco_cond) > 0 and len(control_cond) > 0:
            stat, p_value = stats.mannwhitneyu(glyco_cond, control_cond, alternative='two-sided')
            report_lines.append(f"  p-value: {p_value:.6f}")

        report_lines.append("")

    # Write report
    output_file = OUTPUT_DIR / "proteinmpnn_score_analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Saved report: {output_file}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*80)
    print("PROTEINMPNN SCORE ANALYSIS")
    print("="*80)
    print()

    # Analyze design scores
    df = analyze_design_scores()

    if df is not None and len(df) > 0:
        # Create visualizations
        create_visualizations(df)

        # Generate report
        generate_report(df)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
        print("Files generated:")
        print("  - proteinmpnn_design_scores_detailed.csv")
        print("  - proteinmpnn_score_analysis_report.txt")
        print("  - figures/proteinmpnn_scores_glyco_vs_control.png")
        print("  - figures/proteinmpnn_scores_by_condition.png")
        print("  - figures/proteinmpnn_scores_per_protein.png")
    else:
        print("No data to analyze!")

if __name__ == '__main__':
    main()
