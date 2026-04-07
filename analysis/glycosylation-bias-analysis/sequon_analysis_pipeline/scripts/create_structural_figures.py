#!/usr/bin/env python3
"""
Create visualization figures for structural feature analysis using the
glycosylated-only structural features dataset.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def create_structural_analysis_figures(df, output_dir):
    """Create comprehensive figures for structural analysis."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Middle position (X) analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Middle AA and retention
    df['middle_aa'] = df['wt_sequon'].apply(lambda x: x[1] if len(x) > 1 else 'X')
    middle_stats = df.groupby('middle_aa').agg({
        'sequon_retention_pct': ['mean', 'count']
    }).round(2)
    middle_stats.columns = ['mean_retention', 'count']
    middle_stats = middle_stats.sort_values('count', ascending=False).head(10)

    colors = ['#2ecc71' if ret > 5 else '#e74c3c' if ret == 0 else '#f39c12'
              for ret in middle_stats['mean_retention']]

    bars = axes[0].bar(range(len(middle_stats)), middle_stats['mean_retention'],
                       color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(middle_stats)))
    axes[0].set_xticklabels([f"{aa}\n(n={int(middle_stats.loc[aa, 'count'])})"
                            for aa in middle_stats.index])
    axes[0].set_xlabel('Middle Position (X in N-X-S/T)')
    axes[0].set_ylabel('Mean Sequon Retention (%)')
    axes[0].set_title('A. Sequon Retention by Middle Position Residue')
    axes[0].axhline(y=df['sequon_retention_pct'].mean(), color='gray',
                    linestyle='--', label=f"Overall mean: {df['sequon_retention_pct'].mean():.1f}%")
    axes[0].legend()

    # Add value labels
    for i, (idx, row) in enumerate(middle_stats.iterrows()):
        axes[0].text(i, row['mean_retention'] + 0.5, f"{row['mean_retention']:.1f}%",
                    ha='center', fontsize=9)

    # Panel B: Dataset comparison (or glycosylated-only summary)
    dataset_colors = {'glycosylated': '#3498db', 'non_glycosylated': '#9b59b6'}
    categories = ['Sequon\nRetention', 'N\nRetention', 'S/T\nRetention']

    x = np.arange(len(categories))

    gly_data = df[df['dataset'] == 'glycosylated']
    nongly_data = df[df['dataset'] == 'non_glycosylated']

    gly_means = [gly_data['sequon_retention_pct'].mean(),
                 gly_data['n_retention_pct'].mean(),
                 gly_data['st_retention_pct'].mean()]
    if len(nongly_data) > 0:
        width = 0.35
        nongly_means = [nongly_data['sequon_retention_pct'].mean(),
                        nongly_data['n_retention_pct'].mean(),
                        nongly_data['st_retention_pct'].mean()]

        bars1 = axes[1].bar(x - width/2, gly_means, width,
                            label=f'Glycosylated (n={len(gly_data)})',
                            color=dataset_colors['glycosylated'], edgecolor='black')
        bars2 = axes[1].bar(x + width/2, nongly_means, width,
                            label=f'Non-glycosylated (n={len(nongly_data)})',
                            color=dataset_colors['non_glycosylated'], edgecolor='black')

        axes[1].set_title('B. Retention Comparison by Dataset')
        axes[1].legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{height:.1f}%', ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{height:.1f}%', ha='center', fontsize=9)
    else:
        width = 0.6
        bars1 = axes[1].bar(x, gly_means, width,
                            label=f'Glycosylated (n={len(gly_data)})',
                            color=dataset_colors['glycosylated'], edgecolor='black')
        axes[1].set_title('B. Retention (Glycosylated Only)')
        axes[1].legend()
        axes[1].text(0.5, 0.9, 'Non-glycosylated dataset not included',
                     transform=axes[1].transAxes, ha='center', fontsize=9)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{height:.1f}%', ha='center', fontsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('Retention (%)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_sequence_context_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_sequence_context_analysis.pdf', bbox_inches='tight')
    plt.close()

    # Figure 2: Structural features (B-factor and position)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: B-factor vs retention
    df_valid = df[df['b_factor_avg'].notna()].copy()
    if len(df_valid) > 0:
        retained = df_valid[df_valid['sequon_retention_pct'] > 0]
        not_retained = df_valid[df_valid['sequon_retention_pct'] == 0]

        axes[0].scatter(not_retained['b_factor_avg'], not_retained['sequon_retention_pct'],
                       alpha=0.6, label=f'Not retained (n={len(not_retained)})',
                       color='#e74c3c', s=60)
        axes[0].scatter(retained['b_factor_avg'], retained['sequon_retention_pct'],
                       alpha=0.8, label=f'Retained (n={len(retained)})',
                       color='#2ecc71', s=80, marker='*')

        axes[0].set_xlabel('B-factor (Å²)')
        axes[0].set_ylabel('Sequon Retention (%)')
        axes[0].set_title('A. B-factor vs Sequon Retention')
        axes[0].legend()

        # Add correlation text
        corr = df_valid[['b_factor_avg', 'sequon_retention_pct']].corr().iloc[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Position region analysis
    df_pos = df[df['position_region'].notna() & (df['position_region'] != 'unknown')]
    if len(df_pos) > 0:
        regions = ['N-terminal', 'middle', 'C-terminal']
        region_data = []
        for region in regions:
            subset = df_pos[df_pos['position_region'] == region]
            if len(subset) > 0:
                region_data.append({
                    'region': region,
                    'sequon': subset['sequon_retention_pct'].mean(),
                    'n': subset['n_retention_pct'].mean(),
                    'st': subset['st_retention_pct'].mean(),
                    'count': len(subset)
                })

        region_df = pd.DataFrame(region_data)

        x = np.arange(len(region_df))
        width = 0.25

        axes[1].bar(x - width, region_df['sequon'], width, label='Sequon',
                   color='#3498db', edgecolor='black')
        axes[1].bar(x, region_df['n'], width, label='N',
                   color='#2ecc71', edgecolor='black')
        axes[1].bar(x + width, region_df['st'], width, label='S/T',
                   color='#e74c3c', edgecolor='black')

        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{r['region']}\n(n={r['count']})" for _, r in region_df.iterrows()])
        axes[1].set_ylabel('Retention (%)')
        axes[1].set_title('B. Retention by Position in Chain')
        axes[1].legend()

    # Panel C: Sequon composition (NxT vs NxS)
    df['third_aa'] = df['wt_sequon'].apply(lambda x: x[2] if len(x) > 2 else 'X')
    nxt = df[df['third_aa'] == 'T']
    nxs = df[df['third_aa'] == 'S']

    categories = ['Sequon', 'N', 'S/T (3rd pos)']
    nxt_means = [nxt['sequon_retention_pct'].mean(),
                 nxt['n_retention_pct'].mean(),
                 nxt['st_retention_pct'].mean()]
    nxs_means = [nxs['sequon_retention_pct'].mean(),
                 nxs['n_retention_pct'].mean(),
                 nxs['st_retention_pct'].mean()]

    x = np.arange(len(categories))
    width = 0.35

    axes[2].bar(x - width/2, nxt_means, width, label=f'N-X-T (n={len(nxt)})',
               color='#1abc9c', edgecolor='black')
    axes[2].bar(x + width/2, nxs_means, width, label=f'N-X-S (n={len(nxs)})',
               color='#f1c40f', edgecolor='black')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].set_ylabel('Retention (%)')
    axes[2].set_title('C. Retention by Third Position (T vs S)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_structural_features.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_structural_features.pdf', bbox_inches='tight')
    plt.close()

    # Figure 3: High-retention sites analysis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by retention
    df_sorted = df.sort_values('sequon_retention_pct', ascending=True)

    # Color by dataset
    colors = ['#3498db' if d == 'glycosylated' else '#9b59b6' for d in df_sorted['dataset']]

    y_pos = range(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['sequon_retention_pct'], color=colors, alpha=0.7)

    # Highlight high retention
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['sequon_retention_pct'] > 10:
            bars[i].set_alpha(1.0)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    # Add labels for top sites
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        if row['sequon_retention_pct'] > 10:
            ax.text(row['sequon_retention_pct'] + 1, i,
                   f"{row['pdb_id']}:{row['wt_sequon']} ({row['sequon_retention_pct']:.1f}%)",
                   va='center', fontsize=9)

    ax.set_xlabel('Sequon Retention (%)')
    ax.set_ylabel(f"Sequon Sites (n={len(df_sorted)})")
    ax.set_title('Glycosylated Sequon Sites Ranked by Retention')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Glycosylated'),
                      Patch(facecolor='#9b59b6', label='Non-glycosylated')]
    ax.legend(handles=legend_elements, loc='lower right')

    # Remove y ticks for cleaner look
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_all_sequon_retention_ranked.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig8_all_sequon_retention_ranked.pdf', bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {output_dir}")

    return


def main():
    base_dir = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description='Create structural analysis figures from glycosylated-only features'
    )
    parser.add_argument(
        '--features_file',
        default=str(base_dir / 'structural_features_glycosylated_only.csv'),
        help='Path to the structural features CSV'
    )
    args = parser.parse_args()

    # Load data
    features_file = Path(args.features_file)
    print(f"Loading data from: {features_file}")
    df = pd.read_csv(features_file)
    print(f"Total sequons: {len(df)}")

    # Create figures
    figures_dir = base_dir / 'results_HA_case_study' / 'figures'
    create_structural_analysis_figures(df, figures_dir)


if __name__ == '__main__':
    main()
