#!/usr/bin/env python3
"""
Supplementary Analyses for Sequon Retention Study

1. Per-Protein Heterogeneity Visualization
2. Functional Sequon Reanalysis (any valid N-X-S/T vs exact match)
3. Substitution Flow Analysis (what sequons become)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from collections import defaultdict


def is_valid_sequon(triplet):
    """Check if a triplet is a valid N-X-S/T sequon (X ≠ P)"""
    if len(triplet) != 3:
        return False
    return (triplet[0] == 'N' and
            triplet[1] != 'P' and
            triplet[2] in ['S', 'T'])


def analyze_functional_retention(df_sites):
    """
    Analyze functional sequon retention vs exact match retention.
    A functional sequon is ANY valid N-X-S/T (X≠P), not just the original.
    """
    results = []

    for _, row in df_sites.iterrows():
        pdb_id = row['pdb_id']
        wt_sequon = row['wt_sequon']
        n_designs = row['n_designs']
        exact_retained = row['sequon_retained']

        # Parse substitutions
        subs_str = row['all_substitutions']
        try:
            subs = ast.literal_eval(subs_str)
        except:
            continue

        # Count functional sequons (any valid N-X-S/T)
        functional_retained = 0
        for triplet, count in subs.items():
            if is_valid_sequon(triplet):
                functional_retained += count

        results.append({
            'pdb_id': pdb_id,
            'wt_sequon': wt_sequon,
            'n_designs': n_designs,
            'exact_retained': exact_retained,
            'exact_retention_pct': 100 * exact_retained / n_designs,
            'functional_retained': functional_retained,
            'functional_retention_pct': 100 * functional_retained / n_designs,
            'new_valid_sequons': functional_retained - exact_retained  # Sequons created by shuffling
        })

    return pd.DataFrame(results)


def analyze_substitution_patterns(df_sites):
    """
    Analyze what amino acids replace N, X, and S/T positions.
    """
    n_substitutions = defaultdict(int)  # What replaces N
    x_substitutions = defaultdict(int)  # What replaces X
    st_substitutions = defaultdict(int)  # What replaces S/T

    total_designs = 0

    for _, row in df_sites.iterrows():
        wt_sequon = row['wt_sequon']
        subs_str = row['all_substitutions']

        try:
            subs = ast.literal_eval(subs_str)
        except:
            continue

        for triplet, count in subs.items():
            if len(triplet) != 3:
                continue
            n_substitutions[triplet[0]] += count
            x_substitutions[triplet[1]] += count
            st_substitutions[triplet[2]] += count
            total_designs += count

    return {
        'N_position': dict(n_substitutions),
        'X_position': dict(x_substitutions),
        'ST_position': dict(st_substitutions),
        'total': total_designs
    }


def create_functional_retention_figure(total_designs, exact_retained, functional_retained, output_path):
    """
    Create figure showing fate of sequons: destroyed vs shuffled vs exact match.
    Emphasizes that functional retention is 2x exact match.
    """
    # Calculate categories
    destroyed = total_designs - functional_retained
    shuffled = functional_retained - exact_retained  # New valid sequons

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Stacked bar showing fate of all sequon-positions
    ax1 = axes[0]

    categories = ['Sequon\nFate']
    destroyed_pct = 100 * destroyed / total_designs
    shuffled_pct = 100 * shuffled / total_designs
    exact_pct = 100 * exact_retained / total_designs

    # Stacked horizontal bar
    bar_height = 0.5
    ax1.barh(categories, [destroyed_pct], height=bar_height, color='#d62728', label=f'Lost function ({destroyed_pct:.1f}%)', edgecolor='black')
    ax1.barh(categories, [shuffled_pct], height=bar_height, left=[destroyed_pct], color='#ff7f0e', label=f'Shuffled to new valid ({shuffled_pct:.1f}%)', edgecolor='black')
    ax1.barh(categories, [exact_pct], height=bar_height, left=[destroyed_pct + shuffled_pct], color='#2ca02c', label=f'Exact match retained ({exact_pct:.1f}%)', edgecolor='black')

    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Percentage of designs', fontsize=11)
    ax1.set_title(f'Fate of {total_designs:,} sequon-positions\n(34 sites × 32 designs)', fontsize=12)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    # Right panel: Zoom on "retained function" portion
    ax2 = axes[1]

    functional_pct = 100 * functional_retained / total_designs

    # Pie chart of functional retention breakdown
    sizes = [exact_retained, shuffled]
    labels = [f'Exact match\n({exact_retained} designs)', f'New valid sequon\n({shuffled} designs)']
    colors = ['#2ca02c', '#ff7f0e']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'fontsize': 10})

    ax2.set_title(f'Breakdown of functional retention\n({functional_retained} designs = {functional_pct:.1f}% of total)', fontsize=12)

    # Add annotation about the key insight
    fig.text(0.5, 0.02,
             'Key insight: ~half of functionally retained sequons are NEW valid motifs, not exact matches.\n'
             'This suggests MPNN is indifferent to sequon identity, not actively avoiding it.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_site_swarm_figure(df_sites, output_path):
    """
    Create strip/swarm plot showing individual sequon sites colored by protein.
    """
    df = df_sites.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique proteins and assign colors
    proteins = df['pdb_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(proteins)))
    protein_colors = {p: colors[i] for i, p in enumerate(proteins)}

    # Add jitter for visibility
    np.random.seed(42)
    jitter = np.random.uniform(-0.3, 0.3, len(df))

    # Plot each protein's sites
    for i, protein in enumerate(proteins):
        mask = df['pdb_id'] == protein
        protein_data = df[mask]
        x = np.full(len(protein_data), i) + jitter[mask]
        y = protein_data['sequon_retention_pct']

        ax.scatter(x, y, c=[protein_colors[protein]], s=100, alpha=0.7,
                  edgecolors='black', linewidths=0.5, label=protein)

    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, rotation=45, ha='right')
    ax.set_ylabel('Sequon Retention (%)', fontsize=12)
    ax.set_xlabel('Protein (PDB ID)', fontsize=12)
    ax.set_title('Individual Sequon Site Retention by Protein\n(Each point = one sequon site)', fontsize=14)

    # Add horizontal line at mean
    mean_ret = df['sequon_retention_pct'].mean()
    ax.axhline(y=mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.1f}%')

    ax.set_ylim(-5, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_substitution_flow_figure(sub_patterns, output_path):
    """
    Create bar charts showing substitution patterns for N, X, and S/T positions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    positions = ['N_position', 'X_position', 'ST_position']
    titles = ['N Position Substitutions', 'X Position Substitutions', 'S/T Position Substitutions']
    original = ['N', None, 'S/T']

    for i, (pos, title) in enumerate(zip(positions, titles)):
        ax = axes[i]

        data = sub_patterns[pos]
        total = sum(data.values())

        # Sort by frequency and take top 10
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)[:10]

        aas = [x[0] for x in sorted_data]
        counts = [x[1] for x in sorted_data]
        pcts = [100 * c / total for c in counts]

        # Color original AA differently
        colors = ['green' if aa == 'N' or (pos == 'ST_position' and aa in ['S', 'T']) else 'steelblue'
                 for aa in aas]

        bars = ax.bar(aas, pcts, color=colors, edgecolor='black')
        ax.set_xlabel('Amino Acid', fontsize=11)
        ax.set_ylabel('Frequency (%)', fontsize=11)
        ax.set_title(title, fontsize=12)

        # Add percentage labels on bars
        for bar, pct in zip(bars, pcts):
            ax.annotate(f'{pct:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)

    plt.suptitle('Substitution Patterns at Sequon Positions\n(Green = original/valid glycosylation residue)',
                fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path(__file__).parent.parent

    # Load data
    protein_file = base_dir / 'corrected_sequon_analysis.csv'
    sites_file = base_dir / 'archive' / 'deprecated_mixed_dataset_analysis' / 'systematic_sequon_analysis.csv'

    df_protein = pd.read_csv(protein_file)
    df_sites_all = pd.read_csv(sites_file)

    # Filter to glycosylated only
    df_sites = df_sites_all[df_sites_all['dataset'] == 'glycosylated'].copy()

    print("="*70)
    print("SUPPLEMENTARY ANALYSES - GLYCOSYLATED PROTEINS ONLY")
    print("="*70)

    # Create output directory
    output_dir = base_dir / 'supplementary_figures'
    output_dir.mkdir(exist_ok=True)

    # ===================================================================
    # 1. Functional Sequon Retention Analysis (MAIN FIGURE)
    # ===================================================================
    print("\n1. FUNCTIONAL SEQUON RETENTION")
    print("-"*50)

    # First compute functional retention to get the numbers for the figure
    df_functional = analyze_functional_retention(df_sites)

    total_exact = df_functional['exact_retained'].sum()
    total_functional = df_functional['functional_retained'].sum()
    total_designs = df_functional['n_designs'].sum()

    print(f"Total sequon-positions: {total_designs}")
    print(f"Exact match retained: {total_exact} ({100*total_exact/total_designs:.2f}%)")
    print(f"Functional retained: {total_functional} ({100*total_functional/total_designs:.2f}%)")
    print(f"Shuffled to new valid: {total_functional - total_exact}")
    print(f"Ratio (functional/exact): {total_functional/total_exact:.2f}×")

    # Create the main figure
    create_functional_retention_figure(
        total_designs, total_exact, total_functional,
        output_dir / 'fig_functional_retention.png'
    )

    # Also print per-protein heterogeneity stats (for text)
    print("\n2. PER-PROTEIN HETEROGENEITY (for text)")
    print("-"*50)
    df_with_sequons = df_protein[df_protein['n_sequons'] > 0]
    total_retained = df_with_sequons['sequon_retained'].sum()
    gv0_retained = df_protein[df_protein['pdb_id'] == '5gv0']['sequon_retained'].values[0]

    print(f"Total retained sequons: {total_retained}")
    print(f"5gv0 retained sequons: {gv0_retained}")
    print(f"5gv0 percentage: {100*gv0_retained/total_retained:.1f}%")

    # ===================================================================
    # 3. Sites with Shuffling (detail for text)
    # ===================================================================
    print("\n3. SITES WITH SHUFFLING TO NEW VALID SEQUONS")
    print("-"*50)

    # Show sites where functional > exact (MPNN created different valid sequons)
    df_shuffled = df_functional[df_functional['new_valid_sequons'] > 0].sort_values('new_valid_sequons', ascending=False)

    if len(df_shuffled) > 0:
        print(df_shuffled[['pdb_id', 'wt_sequon', 'exact_retained', 'functional_retained', 'new_valid_sequons']].to_string(index=False))

    # Interpretation
    ratio = total_functional / total_exact if total_exact > 0 else float('inf')
    print(f"\nFunctional / Exact ratio: {ratio:.2f}×")
    print("INTERPRETATION: MPNN is indifferent to sequon identity (shuffles ~as often as preserves)")

    # Save functional analysis
    df_functional.to_csv(output_dir / 'functional_sequon_analysis.csv', index=False)
    print(f"\nSaved: {output_dir / 'functional_sequon_analysis.csv'}")

    # ===================================================================
    # 4. Substitution Flow Analysis (for text)
    # ===================================================================
    print("\n4. SUBSTITUTION FLOW ANALYSIS")
    print("-"*50)

    sub_patterns = analyze_substitution_patterns(df_sites)

    # N position analysis
    print("\nN position → (what N becomes):")
    n_data = sub_patterns['N_position']
    n_total = sum(n_data.values())
    for aa, count in sorted(n_data.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {aa}: {count} ({100*count/n_total:.1f}%)")

    # Check if N is retained
    n_retained = n_data.get('N', 0)
    print(f"\n  N retained: {100*n_retained/n_total:.1f}%")
    print(f"  N replaced: {100*(1-n_retained/n_total):.1f}%")

    # Most common N replacements (excluding N)
    print("\n  Top N replacements (when N is lost):")
    for aa, count in sorted(n_data.items(), key=lambda x: x[1], reverse=True)[:6]:
        if aa != 'N':
            print(f"    → {aa}: {100*count/n_total:.1f}%")

    # S/T position analysis
    print("\nS/T position → (what S/T becomes):")
    st_data = sub_patterns['ST_position']
    st_total = sum(st_data.values())
    st_kept = st_data.get('S', 0) + st_data.get('T', 0)
    print(f"  S or T retained: {100*st_kept/st_total:.1f}%")
    print(f"  S/T replaced: {100*(1-st_kept/st_total):.1f}%")

    # Note: Substitution flow figure removed - converted to text summary (see RESULTS_AND_METHODS.md)
    # create_substitution_flow_figure(sub_patterns, output_dir / 'fig_substitution_patterns.png')

    # ===================================================================
    # Summary for RESULTS_AND_METHODS.md
    # ===================================================================
    print("\n" + "="*70)
    print("SUMMARY FOR RESULTS_AND_METHODS.md")
    print("="*70)

    print(f"""
### Finding 2.5: Functional Sequon Retention Analysis

To distinguish between "MPNN destroys sequons" vs "MPNN shuffles but preserves function,"
we counted designs where ANY valid N-X-S/T motif (X≠P) exists at the original position,
not just exact sequence matches.

| Metric | Value |
|--------|-------|
| Exact match retention | {100*total_exact/total_designs:.2f}% |
| Functional retention (any valid sequon) | {100*total_functional/total_designs:.2f}% |
| Ratio (functional/exact) | {ratio:.2f} |

**Interpretation**: {"MPNN shuffles but largely preserves glycosylatability" if ratio > 1.5 else "MPNN genuinely destroys most glycosylation sites" if ratio < 1.1 else "Mixed - limited shuffling to valid sequons"}

### Finding 2.6: Substitution Patterns

When sequon-N is lost, MPNN preferentially substitutes:
""")

    for aa, count in sorted(n_data.items(), key=lambda x: x[1], reverse=True)[:4]:
        if aa != 'N':
            print(f"- **{aa}** ({100*count/n_total:.1f}%)")

    print(f"""
S/T position is retained {100*st_kept/st_total:.1f}% of the time (as S or T).
""")

    return df_functional, sub_patterns


if __name__ == '__main__':
    df_functional, sub_patterns = main()
