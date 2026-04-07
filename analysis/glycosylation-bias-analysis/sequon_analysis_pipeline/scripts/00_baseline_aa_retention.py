#!/usr/bin/env python3
"""
STEP 0: Baseline Amino Acid Retention Analysis

Calculates the baseline retention rate for each amino acid type from WT to designs.
This provides context for understanding sequon-specific effects - is asparagine (N)
particularly well or poorly retained compared to other amino acids?

Can run on:
- A single PDB directory
- A list of PDB directories (from file)
- A folder containing multiple PDB directories
- Compare two folders of PDB directories

Usage:
    # Single PDB (after running steps 1-3)
    python 00_baseline_aa_retention.py --pdb_dir ./results/1EO8

    # Multiple PDBs from list
    python 00_baseline_aa_retention.py --pdb_list pdb_list.txt --output_dir ./results/aggregate

    # Multiple PDBs from folder
    python 00_baseline_aa_retention.py --pdb_folder ./experiments/ --output_dir ./results/folder_analysis

    # Compare two folders
    python 00_baseline_aa_retention.py --compare_folders ./folder1 ./folder2 --folder_labels "Dataset A" "Dataset B" --output_dir ./comparison

    # Where pdb_list.txt contains one PDB directory per line:
    #   ./results/1EO8
    #   ./results/2ABC
    #   ./results/3XYZ

Outputs:
    - analysis/baseline/aa_retention_by_type.csv
    - analysis/baseline/aa_retention_summary.csv
    - analysis/baseline/figures/aa_retention_barplot.png
    - analysis/baseline/figures/aa_retention_heatmap.png
    - analysis/baseline/figures/aa_confusion_matrix.png

For comparisons:
    - dataset_comparison.csv
    - comparison_figures/retention_comparison_barplot.png
    - comparison_figures/retention_differences.png
    - comparison_figures/asparagine_comparison.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Standard amino acid properties for grouping
AA_GROUPS = {
    'Hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'],
    'Polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
    'Charged+': ['K', 'R', 'H'],
    'Charged-': ['D', 'E'],
    'Special': ['G']
}

AA_TO_GROUP = {}
for group, aas in AA_GROUPS.items():
    for aa in aas:
        AA_TO_GROUP[aa] = group

# One-letter codes
ALL_AAS = list('ACDEFGHIKLMNPQRSTVWY')


def read_fasta(path):
    """Read FASTA file."""
    sequences = []
    current_header = None
    current_seq = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_seq:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            sequences.append((current_header, ''.join(current_seq)))
    return sequences


def analyze_aa_retention(fa_path, pdb_id):
    """Analyze amino acid retention for a single FASTA file."""
    if not fa_path.exists():
        return None

    sequences = read_fasta(fa_path)
    if len(sequences) < 2:
        return None

    wt_header, wt_seq = sequences[0]
    designs = sequences[1:]

    # Remove chain separators for analysis
    wt_seq_clean = wt_seq.replace('/', '')

    # Count retention by amino acid type
    aa_counts = defaultdict(lambda: {'total': 0, 'retained': 0, 'substitutions': defaultdict(int)})

    for header, des_seq in designs:
        des_seq_clean = des_seq.replace('/', '')

        # Align WT and design (should be same length)
        min_len = min(len(wt_seq_clean), len(des_seq_clean))

        for i in range(min_len):
            wt_aa = wt_seq_clean[i]
            des_aa = des_seq_clean[i]

            # Skip non-standard amino acids
            if wt_aa not in ALL_AAS or wt_aa == 'X':
                continue

            aa_counts[wt_aa]['total'] += 1

            if des_aa == wt_aa:
                aa_counts[wt_aa]['retained'] += 1
            else:
                if des_aa in ALL_AAS:
                    aa_counts[wt_aa]['substitutions'][des_aa] += 1

    # Calculate retention rates
    results = []
    for aa in ALL_AAS:
        if aa_counts[aa]['total'] > 0:
            retention_rate = aa_counts[aa]['retained'] / aa_counts[aa]['total'] * 100
            results.append({
                'pdb_id': pdb_id,
                'amino_acid': aa,
                'group': AA_TO_GROUP.get(aa, 'Other'),
                'total_occurrences': aa_counts[aa]['total'],
                'retained': aa_counts[aa]['retained'],
                'retention_pct': retention_rate,
                'substitutions': dict(aa_counts[aa]['substitutions'])
            })

    return results, aa_counts


def create_confusion_matrix(aa_counts_list):
    """Create amino acid substitution confusion matrix."""
    # Aggregate substitutions across all analyses
    total_subs = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    for aa_counts in aa_counts_list:
        for wt_aa, data in aa_counts.items():
            total_counts[wt_aa] += data['total']
            for des_aa, count in data['substitutions'].items():
                total_subs[wt_aa][des_aa] += count
            # Add retained as diagonal
            total_subs[wt_aa][wt_aa] += data['retained']

    # Create matrix
    matrix = np.zeros((len(ALL_AAS), len(ALL_AAS)))
    for i, wt_aa in enumerate(ALL_AAS):
        row_total = sum(total_subs[wt_aa].values())
        if row_total > 0:
            for j, des_aa in enumerate(ALL_AAS):
                matrix[i, j] = total_subs[wt_aa][des_aa] / row_total * 100

    return matrix


def collect_pdb_dirs_from_folder(folder_path, condition='unconstrained'):
    """Collect all PDB directories from a folder that contain FASTA files."""
    folder_path = Path(folder_path)
    pdb_dirs = []

    if not folder_path.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return pdb_dirs

    # Look for directories containing seqs folders with FASTA files
    for item in folder_path.iterdir():
        if item.is_dir():
            # Check for FASTA files in various structures
            fa_files = list(item.glob("**/*.fa"))
            if fa_files:
                pdb_dirs.append(item)

    return pdb_dirs


def compare_datasets(retention_df1, retention_df2, aa_counts_list1, aa_counts_list2,
                    output_dir, label1="Dataset 1", label2="Dataset 2"):
    """Compare retention patterns between two datasets."""
    fig_dir = output_dir / "comparison_figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Aggregate data for each dataset
    summary1 = retention_df1.groupby('amino_acid').agg({
        'retention_pct': 'mean',
        'total_occurrences': 'sum'
    }).round(2)
    summary1.columns = ['retention_pct', 'total_occurrences']
    summary1 = summary1.reset_index()

    summary2 = retention_df2.groupby('amino_acid').agg({
        'retention_pct': 'mean',
        'total_occurrences': 'sum'
    }).round(2)
    summary2.columns = ['retention_pct', 'total_occurrences']
    summary2 = summary2.reset_index()

    # Merge for comparison
    comparison = pd.merge(summary1[['amino_acid', 'retention_pct']],
                         summary2[['amino_acid', 'retention_pct']],
                         on='amino_acid', suffixes=(f'_{label1.replace(" ", "_")}', f'_{label2.replace(" ", "_")}'))

    # Figure 1: Side-by-side comparison
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(ALL_AAS))
    width = 0.35

    ret1_col = f'retention_pct_{label1.replace(" ", "_")}'
    ret2_col = f'retention_pct_{label2.replace(" ", "_")}'

    bars1 = ax.bar(x - width/2, comparison[ret1_col], width, label=label1, color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, comparison[ret2_col], width, label=label2, color='#E74C3C', alpha=0.8)

    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Retention Rate (%)', fontsize=12)
    ax.set_title(f'Amino Acid Retention Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['amino_acid'])
    ax.set_ylim(0, 100)
    ax.legend()

    # Highlight differences
    for i, (ret1, ret2) in enumerate(zip(comparison[ret1_col], comparison[ret2_col])):
        diff = ret2 - ret1
        if abs(diff) > 10:  # Significant difference
            color = 'red' if diff > 0 else 'blue'
            ax.text(i, max(ret1, ret2) + 2, f'{diff:+.1f}%', ha='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_dir / "retention_comparison_barplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Difference plot
    fig, ax = plt.subplots(figsize=(14, 6))

    differences = comparison[ret2_col] - comparison[ret1_col]
    colors = ['#E74C3C' if x > 0 else '#3498DB' for x in differences]

    bars = ax.bar(comparison['amino_acid'], differences, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel(f'Retention Difference ({label2} - {label1}) (%)', fontsize=12)
    ax.set_title(f'Retention Rate Differences: {label2} vs {label1}', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        if height != 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + (1 if height > 0 else -3),
                   f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "retention_differences.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Asparagine focus comparison
    if 'N' in comparison['amino_acid'].values:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_data1 = retention_df1[retention_df1['amino_acid'] == 'N']
        n_data2 = retention_df2[retention_df2['amino_acid'] == 'N']

        if len(n_data1) > 0 and len(n_data2) > 0:
            # Left: N retention comparison
            n_ret1 = n_data1['retention_pct'].mean()
            n_ret2 = n_data2['retention_pct'].mean()

            bars = axes[0].bar([label1, label2], [n_ret1, n_ret2], color=['#3498DB', '#E74C3C'])
            axes[0].set_ylabel('Asparagine Retention (%)', fontsize=12)
            axes[0].set_title('N Retention Comparison', fontsize=12)
            axes[0].set_ylim(0, 100)

            for bar, val in zip(bars, [n_ret1, n_ret2]):
                axes[0].text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', fontsize=10)

            # Right: N substitution patterns
            n_subs1 = defaultdict(int)
            n_subs2 = defaultdict(int)

            for aa_counts in aa_counts_list1:
                if 'N' in aa_counts:
                    for des_aa, count in aa_counts['N']['substitutions'].items():
                        n_subs1[des_aa] += count

            for aa_counts in aa_counts_list2:
                if 'N' in aa_counts:
                    for des_aa, count in aa_counts['N']['substitutions'].items():
                        n_subs2[des_aa] += count

            # Get top substitutions
            all_subs = set(list(n_subs1.keys()) + list(n_subs2.keys()))
            sub_comparison = []
            for aa in all_subs:
                sub_comparison.append({
                    'aa': aa,
                    f'{label1}': n_subs1.get(aa, 0),
                    f'{label2}': n_subs2.get(aa, 0)
                })

            if sub_comparison:
                sub_df = pd.DataFrame(sub_comparison).sort_values(f'{label2}', ascending=False).head(8)

                x = np.arange(len(sub_df))
                width = 0.35

                axes[1].bar(x - width/2, sub_df[f'{label1}'], width, label=label1, color='#3498DB', alpha=0.7)
                axes[1].bar(x + width/2, sub_df[f'{label2}'], width, label=label2, color='#E74C3C', alpha=0.7)

                axes[1].set_xlabel('Amino Acid', fontsize=12)
                axes[1].set_ylabel('Substitution Count', fontsize=12)
                axes[1].set_title('N Substitution Patterns', fontsize=12)
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(sub_df['aa'])
                axes[1].legend()

        plt.suptitle(f'Asparagine (N) Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / "asparagine_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Save comparison data
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)

    print(f"  Saved comparison figures to {fig_dir}/")
    print(f"  Saved comparison data to {output_dir}/dataset_comparison.csv")

    return comparison


def create_visualizations(retention_df, aa_counts_list, output_dir, title_prefix=""):
    """Create baseline retention visualizations."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Retention rate by amino acid (bar plot)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Aggregate by amino acid
    aa_summary = retention_df.groupby('amino_acid').agg({
        'retention_pct': 'mean',
        'total_occurrences': 'sum'
    }).reset_index()
    aa_summary = aa_summary.sort_values('retention_pct', ascending=False)

    # Color by group
    colors = []
    group_colors = {
        'Hydrophobic': '#3498DB',
        'Polar': '#2ECC71',
        'Charged+': '#E74C3C',
        'Charged-': '#9B59B6',
        'Special': '#F39C12'
    }
    for aa in aa_summary['amino_acid']:
        group = AA_TO_GROUP.get(aa, 'Other')
        colors.append(group_colors.get(group, '#95A5A6'))

    bars = ax.bar(aa_summary['amino_acid'], aa_summary['retention_pct'], color=colors)

    # Highlight asparagine (N) - the sequon amino acid
    n_idx = list(aa_summary['amino_acid']).index('N') if 'N' in list(aa_summary['amino_acid']) else -1
    if n_idx >= 0:
        bars[n_idx].set_edgecolor('black')
        bars[n_idx].set_linewidth(3)

    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Retention Rate (%)', fontsize=12)
    ax.set_title(f'{title_prefix}Baseline Amino Acid Retention by ProteinMPNN', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=group) for group, color in group_colors.items()]
    legend_elements.append(Patch(facecolor='white', edgecolor='black', linewidth=2, label='N (Asparagine) - Sequon'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "aa_retention_barplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Retention by amino acid group
    fig, ax = plt.subplots(figsize=(10, 6))

    group_summary = retention_df.copy()
    group_summary['group'] = group_summary['amino_acid'].map(AA_TO_GROUP)

    group_means = group_summary.groupby('group')['retention_pct'].mean().sort_values(ascending=False)

    colors = [group_colors.get(g, '#95A5A6') for g in group_means.index]
    bars = ax.bar(group_means.index, group_means.values, color=colors)

    ax.set_xlabel('Amino Acid Group', fontsize=12)
    ax.set_ylabel('Mean Retention Rate (%)', fontsize=12)
    ax.set_title(f'{title_prefix}Retention by Amino Acid Group', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_dir / "aa_retention_by_group.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Substitution confusion matrix
    if aa_counts_list:
        matrix = create_confusion_matrix(aa_counts_list)

        fig, ax = plt.subplots(figsize=(14, 12))

        # Mask diagonal for clearer view of substitutions
        mask = np.eye(len(ALL_AAS), dtype=bool)

        sns.heatmap(
            matrix,
            xticklabels=ALL_AAS,
            yticklabels=ALL_AAS,
            cmap='YlOrRd',
            ax=ax,
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Frequency (%)'},
            annot=True,
            fmt='.0f',
            annot_kws={'size': 7}
        )

        ax.set_xlabel('Designed Amino Acid', fontsize=12)
        ax.set_ylabel('Wild-Type Amino Acid', fontsize=12)
        ax.set_title(f'{title_prefix}Amino Acid Substitution Matrix\n(Diagonal = Retention)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(fig_dir / "aa_substitution_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 4: Asparagine (N) focus - what does it get substituted to?
    if aa_counts_list:
        n_subs = defaultdict(int)
        n_total = 0
        n_retained = 0

        for aa_counts in aa_counts_list:
            if 'N' in aa_counts:
                n_total += aa_counts['N']['total']
                n_retained += aa_counts['N']['retained']
                for des_aa, count in aa_counts['N']['substitutions'].items():
                    n_subs[des_aa] += count

        if n_total > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: What N becomes
            sub_data = pd.DataFrame([
                {'aa': aa, 'count': count, 'pct': count/n_total*100}
                for aa, count in n_subs.items()
            ]).sort_values('pct', ascending=False)

            if len(sub_data) > 0:
                top_subs = sub_data.head(10)
                axes[0].barh(range(len(top_subs)), top_subs['pct'], color='coral')
                axes[0].set_yticks(range(len(top_subs)))
                axes[0].set_yticklabels(top_subs['aa'])
                axes[0].set_xlabel('Frequency (%)', fontsize=12)
                axes[0].set_title('When N is Lost, It Becomes...', fontsize=12)
                axes[0].invert_yaxis()

                for i, pct in enumerate(top_subs['pct']):
                    axes[0].text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=9)

            # Right: N retention vs other polar AAs
            polar_retention = retention_df[retention_df['amino_acid'].isin(['N', 'S', 'T', 'Q', 'Y', 'C'])]
            polar_summary = polar_retention.groupby('amino_acid')['retention_pct'].mean().sort_values(ascending=False)

            colors = ['#E74C3C' if aa == 'N' else '#3498DB' for aa in polar_summary.index]
            axes[1].bar(polar_summary.index, polar_summary.values, color=colors)
            axes[1].set_xlabel('Amino Acid', fontsize=12)
            axes[1].set_ylabel('Retention Rate (%)', fontsize=12)
            axes[1].set_title('N Retention vs Other Polar Amino Acids', fontsize=12)
            axes[1].set_ylim(0, 100)

            for i, (aa, pct) in enumerate(polar_summary.items()):
                axes[1].text(i, pct + 2, f'{pct:.0f}%', ha='center', fontsize=10, fontweight='bold')

            plt.suptitle(f'{title_prefix}Asparagine (N) - Sequon Amino Acid Analysis',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / "asparagine_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Figure 5: Summary table as image
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    table_data = aa_summary[['amino_acid', 'retention_pct', 'total_occurrences']].copy()
    table_data['retention_pct'] = table_data['retention_pct'].round(1)
    table_data.columns = ['AA', 'Retention %', 'Occurrences']

    # Add group column
    table_data['Group'] = table_data['AA'].map(AA_TO_GROUP)
    table_data = table_data[['AA', 'Group', 'Retention %', 'Occurrences']]

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(table_data.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    for i in range(len(table_data.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight N row
    n_row = list(table_data['AA']).index('N') + 1 if 'N' in list(table_data['AA']) else -1
    if n_row > 0:
        for j in range(len(table_data.columns)):
            table[(n_row, j)].set_facecolor('#FFEB9C')

    ax.set_title(f'{title_prefix}Amino Acid Retention Summary\n(Asparagine highlighted)',
                fontsize=14, fontweight='bold', pad=20)
    plt.savefig(fig_dir / "aa_retention_table.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved figures to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Step 0: Baseline amino acid retention analysis')
    parser.add_argument('--pdb_dir', help='Single PDB results directory')
    parser.add_argument('--pdb_list', help='File with list of PDB directories (one per line)')
    parser.add_argument('--pdb_folder', help='Folder containing multiple PDB directories to analyze')
    parser.add_argument('--compare_folders', nargs=2, metavar=('FOLDER1', 'FOLDER2'),
                       help='Compare two folders of PDB directories (provide two folder paths)')
    parser.add_argument('--folder_labels', nargs=2, metavar=('LABEL1', 'LABEL2'),
                       default=['Folder 1', 'Folder 2'],
                       help='Labels for the two folders being compared (default: "Folder 1" "Folder 2")')
    parser.add_argument('--output_dir', help='Output directory for aggregate results')
    parser.add_argument('--condition', default='unconstrained',
                       choices=['unconstrained', 'n_only_fixed', 'full_sequon_fixed'],
                       help='Which design condition to analyze')
    args = parser.parse_args()

    # Validate arguments
    input_methods = sum([bool(args.pdb_dir), bool(args.pdb_list), bool(args.pdb_folder), bool(args.compare_folders)])
    if input_methods == 0:
        print("ERROR: Must specify one of: --pdb_dir, --pdb_list, --pdb_folder, or --compare_folders")
        return
    elif input_methods > 1:
        print("ERROR: Can only specify one input method at a time")
        return

    print("=" * 70)
    print("STEP 0: BASELINE AMINO ACID RETENTION ANALYSIS")
    print("=" * 70)

    # Handle different input methods
    if args.compare_folders:
        # Comparison mode
        folder1, folder2 = args.compare_folders
        label1, label2 = args.folder_labels

        print(f"\nComparing two datasets:")
        print(f"  {label1}: {folder1}")
        print(f"  {label2}: {folder2}")
        print(f"  Condition: {args.condition}")

        # Analyze first folder
        pdb_dirs1 = collect_pdb_dirs_from_folder(folder1, args.condition)
        print(f"\nAnalyzing {label1}: {len(pdb_dirs1)} PDB(s)")

        results1, counts1 = analyze_multiple_pdbs(pdb_dirs1, args.condition)

        # Analyze second folder
        pdb_dirs2 = collect_pdb_dirs_from_folder(folder2, args.condition)
        print(f"\nAnalyzing {label2}: {len(pdb_dirs2)} PDB(s)")

        results2, counts2 = analyze_multiple_pdbs(pdb_dirs2, args.condition)

        if not results1 or not results2:
            print("ERROR: No valid data found in one or both folders")
            return

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("./comparison_results")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison
        print(f"\n{'=' * 70}")
        print("CREATING COMPARISON")
        print("=" * 70)

        retention_df1 = pd.DataFrame(results1)
        retention_df2 = pd.DataFrame(results2)

        comparison = compare_datasets(retention_df1, retention_df2, counts1, counts2,
                                    output_dir, label1, label2)

        # Print comparison summary
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print("=" * 70)

        print(f"\n{label1} vs {label2} - Key Differences:")
        for _, row in comparison.iterrows():
            ret1 = row[f'retention_pct_{label1.replace(" ", "_")}']
            ret2 = row[f'retention_pct_{label2.replace(" ", "_")}']
            diff = ret2 - ret1
            if abs(diff) > 5:  # Show significant differences
                direction = "higher" if diff > 0 else "lower"
                print(f"  {row['amino_acid']}: {abs(diff):.1f}% {direction} in {label2} ({ret1:.1f}% vs {ret2:.1f}%)")

        # Focus on asparagine
        n_comparison = comparison[comparison['amino_acid'] == 'N']
        if len(n_comparison) > 0:
            n_ret1 = n_comparison[f'retention_pct_{label1.replace(" ", "_")}'].values[0]
            n_ret2 = n_comparison[f'retention_pct_{label2.replace(" ", "_")}'].values[0]
            n_diff = n_ret2 - n_ret1

            print(f"\nAsparagine (N) Comparison:")
            print(f"  {label1}: {n_ret1:.1f}%")
            print(f"  {label2}: {n_ret2:.1f}%")
            print(f"  Difference: {n_diff:+.1f}%")

            if abs(n_diff) > 5:
                better = label2 if n_diff > 0 else label1
                print(f"  → N is better conserved in {better}")
            else:
                print("  → N retention is similar between datasets")

        print(f"\nOutputs in {output_dir}/:")
        print("  - dataset_comparison.csv")
        print("  - comparison_figures/retention_comparison_barplot.png")
        print("  - comparison_figures/retention_differences.png")
        print("  - comparison_figures/asparagine_comparison.png")

    else:
        # Single dataset analysis (existing functionality)
        # Collect PDB directories
        pdb_dirs = []
        if args.pdb_dir:
            pdb_dirs.append(Path(args.pdb_dir))
        elif args.pdb_list:
            with open(args.pdb_list) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pdb_dirs.append(Path(line))
        elif args.pdb_folder:
            pdb_dirs = collect_pdb_dirs_from_folder(args.pdb_folder, args.condition)

        print(f"\nAnalyzing {len(pdb_dirs)} PDB(s)...")
        print(f"Condition: {args.condition}")

        all_results, all_aa_counts = analyze_multiple_pdbs(pdb_dirs, args.condition)

        if not all_results:
            print("\nNo results to analyze!")
            return

        # Create DataFrame
        retention_df = pd.DataFrame(all_results)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / "analysis" / "baseline"
        elif len(pdb_dirs) == 1:
            output_dir = pdb_dirs[0] / "analysis" / "baseline"
        else:
            output_dir = Path("./results/aggregate/analysis/baseline")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSVs
        print(f"\n{'=' * 70}")
        print("SAVING OUTPUTS")
        print("=" * 70)

        retention_df.to_csv(output_dir / "aa_retention_by_type.csv", index=False)

        # Create summary
        summary = retention_df.groupby('amino_acid').agg({
            'retention_pct': ['mean', 'std', 'count'],
            'total_occurrences': 'sum'
        }).round(2)
        summary.columns = ['mean_retention_pct', 'std_retention_pct', 'n_pdbs', 'total_occurrences']
        summary = summary.reset_index()
        summary['group'] = summary['amino_acid'].map(AA_TO_GROUP)
        summary = summary.sort_values('mean_retention_pct', ascending=False)
        summary.to_csv(output_dir / "aa_retention_summary.csv", index=False)

        print(f"  Saved: {output_dir}/aa_retention_by_type.csv")
        print(f"  Saved: {output_dir}/aa_retention_summary.csv")

        # Create visualizations
        print(f"\n{'=' * 70}")
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        title_prefix = f"{pdb_dirs[0].name}: " if len(pdb_dirs) == 1 else "Aggregate: "
        create_visualizations(retention_df, all_aa_counts, output_dir, title_prefix)

        # Print summary
        print("\n" + "=" * 70)
        print("STEP 0 COMPLETE")
        print("=" * 70)

        print("\nAmino Acid Retention Summary (sorted by retention):")
        print("-" * 50)
        for _, row in summary.iterrows():
            highlight = " ← ASPARAGINE (sequon)" if row['amino_acid'] == 'N' else ""
            print(f"  {row['amino_acid']} ({row['group']:<12}): {row['mean_retention_pct']:>5.1f}%{highlight}")

        # Key finding about N
        n_row = summary[summary['amino_acid'] == 'N']
        if len(n_row) > 0:
            n_retention = n_row['mean_retention_pct'].values[0]
            n_rank = list(summary['amino_acid']).index('N') + 1

            print(f"\n{'=' * 70}")
            print("KEY FINDING: ASPARAGINE (N) BASELINE")
            print("=" * 70)
            print(f"\n  Asparagine retention: {n_retention:.1f}%")
            print(f"  Rank among all AAs: {n_rank}/20")

            avg_all = summary['mean_retention_pct'].mean()
            print(f"  Average across all AAs: {avg_all:.1f}%")

            if n_retention < avg_all - 5:
                print(f"\n  ⚠️  N is LESS conserved than average by {avg_all - n_retention:.1f}%")
                print(f"     This suggests ProteinMPNN has a bias AGAINST asparagine")
            elif n_retention > avg_all + 5:
                print(f"\n  ✓ N is MORE conserved than average by {n_retention - avg_all:.1f}%")
            else:
                print(f"\n  ~ N retention is similar to average")

        print(f"\nOutputs in {output_dir}/:")
        print(f"  - aa_retention_by_type.csv")
        print(f"  - aa_retention_summary.csv")
        print(f"  - figures/aa_retention_barplot.png")
        print(f"  - figures/aa_retention_by_group.png")
        print(f"  - figures/aa_substitution_matrix.png")
        print(f"  - figures/asparagine_analysis.png")
        print(f"  - figures/aa_retention_table.png")

        print(f"\n→ Next: python 01_prepare_structure.py (or continue with existing pipeline)")

    print("=" * 70)
    print("STEP 0: BASELINE AMINO ACID RETENTION ANALYSIS")
    print("=" * 70)

    # Collect PDB directories
    pdb_dirs = []
    if args.pdb_dir:
        pdb_dirs.append(Path(args.pdb_dir))
    if args.pdb_list:
        with open(args.pdb_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pdb_dirs.append(Path(line))

    print(f"\nAnalyzing {len(pdb_dirs)} PDB(s)...")
    print(f"Condition: {args.condition}")

    all_results = []
    all_aa_counts = []

def analyze_multiple_pdbs(pdb_dirs, condition):
    """Analyze multiple PDB directories and return aggregated results."""
    all_results = []
    all_aa_counts = []

    for pdb_dir in pdb_dirs:
        # Try to find FASTA file - handle different directory structures
        fa_path = None
        pdb_id = None

        # Check if it's the expected pipeline structure
        info_path = pdb_dir / "structure" / "structure_info.json"
        if info_path.exists():
            with open(info_path) as f:
                structure_info = json.load(f)
            pdb_id = structure_info['pdb_id']
            # Look for any .fa file in the seqs directory (handles both {pdb_id}.fa and {pdb_id}_protein.fa)
            seqs_dir = pdb_dir / "designs" / condition / "seqs"
            if seqs_dir.exists():
                fa_files = list(seqs_dir.glob("*.fa"))
                if fa_files:
                    fa_path = fa_files[0]
        else:
            # Handle alternative structures
            seqs_dir = pdb_dir / "seqs"
            if seqs_dir.exists():
                # Look for .fa files in seqs directory
                fa_files = list(seqs_dir.glob("*.fa"))
                if fa_files:
                    fa_path = fa_files[0]
                    # Infer pdb_id from filename (remove .fa extension)
                    pdb_id = fa_path.stem

        if not fa_path or not fa_path.exists():
            print(f"  Skipping {pdb_dir} - no designs found")
            continue

        print(f"  Analyzing {pdb_id}...")
        result = analyze_aa_retention(fa_path, pdb_id)

        if result:
            results, aa_counts = result
            all_results.extend(results)
            all_aa_counts.append(aa_counts)

            # Quick summary
            n_data = next((r for r in results if r['amino_acid'] == 'N'), None)
            if n_data:
                print(f"    Asparagine (N) retention: {n_data['retention_pct']:.1f}%")

    return all_results, all_aa_counts

    if not all_results:
        print("\nNo results to analyze!")
        return

    # Create DataFrame
    retention_df = pd.DataFrame(all_results)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / "analysis" / "baseline"
    elif len(pdb_dirs) == 1:
        output_dir = pdb_dirs[0] / "analysis" / "baseline"
    else:
        output_dir = Path("./results/aggregate/analysis/baseline")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")
    print("=" * 70)

    retention_df.to_csv(output_dir / "aa_retention_by_type.csv", index=False)

    # Create summary
    summary = retention_df.groupby('amino_acid').agg({
        'retention_pct': ['mean', 'std', 'count'],
        'total_occurrences': 'sum'
    }).round(2)
    summary.columns = ['mean_retention_pct', 'std_retention_pct', 'n_pdbs', 'total_occurrences']
    summary = summary.reset_index()
    summary['group'] = summary['amino_acid'].map(AA_TO_GROUP)
    summary = summary.sort_values('mean_retention_pct', ascending=False)
    summary.to_csv(output_dir / "aa_retention_summary.csv", index=False)

    print(f"  Saved: {output_dir}/aa_retention_by_type.csv")
    print(f"  Saved: {output_dir}/aa_retention_summary.csv")

    # Create visualizations
    print(f"\n{'=' * 70}")
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    title_prefix = f"{pdb_dirs[0].name}: " if len(pdb_dirs) == 1 else "Aggregate: "
    create_visualizations(retention_df, all_aa_counts, output_dir, title_prefix)

    # Print summary
    print("\n" + "=" * 70)
    print("STEP 0 COMPLETE")
    print("=" * 70)

    print("\nAmino Acid Retention Summary (sorted by retention):")
    print("-" * 50)
    for _, row in summary.iterrows():
        highlight = " ← ASPARAGINE (sequon)" if row['amino_acid'] == 'N' else ""
        print(f"  {row['amino_acid']} ({row['group']:<12}): {row['mean_retention_pct']:>5.1f}%{highlight}")

    # Key finding about N
    n_row = summary[summary['amino_acid'] == 'N']
    if len(n_row) > 0:
        n_retention = n_row['mean_retention_pct'].values[0]
        n_rank = list(summary['amino_acid']).index('N') + 1

        print(f"\n{'=' * 70}")
        print("KEY FINDING: ASPARAGINE (N) BASELINE")
        print("=" * 70)
        print(f"\n  Asparagine retention: {n_retention:.1f}%")
        print(f"  Rank among all AAs: {n_rank}/20")

        avg_all = summary['mean_retention_pct'].mean()
        print(f"  Average across all AAs: {avg_all:.1f}%")

        if n_retention < avg_all - 5:
            print(f"\n  ⚠️  N is LESS conserved than average by {avg_all - n_retention:.1f}%")
            print(f"     This suggests ProteinMPNN has a bias AGAINST asparagine")
        elif n_retention > avg_all + 5:
            print(f"\n  ✓ N is MORE conserved than average by {n_retention - avg_all:.1f}%")
        else:
            print(f"\n  ~ N retention is similar to average")

    print(f"\nOutputs in {output_dir}/:")
    print(f"  - aa_retention_by_type.csv")
    print(f"  - aa_retention_summary.csv")
    print(f"  - figures/aa_retention_barplot.png")
    print(f"  - figures/aa_retention_by_group.png")
    print(f"  - figures/aa_substitution_matrix.png")
    print(f"  - figures/asparagine_analysis.png")
    print(f"  - figures/aa_retention_table.png")

    print(f"\n→ Next: python 01_prepare_structure.py (or continue with existing pipeline)")


if __name__ == "__main__":
    main()
