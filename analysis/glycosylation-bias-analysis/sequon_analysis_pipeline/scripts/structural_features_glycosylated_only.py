#!/usr/bin/env python3
"""
Structural Feature Analysis - GLYCOSYLATED PROTEINS ONLY

This corrected analysis examines structural features only for sequons in
glycosylated proteins, where N-X-S/T motifs are biologically meaningful.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent

    # Load the structural features data
    features_file = base_dir / 'sequon_structural_features.csv'
    print(f"Loading data from: {features_file}")
    df = pd.read_csv(features_file)

    # Filter to glycosylated only
    df_gly = df[df['dataset'] == 'glycosylated'].copy()
    print(f"\nTotal sequon sites: {len(df)}")
    print(f"Glycosylated sites: {len(df_gly)}")
    print(f"Non-glycosylated sites: {len(df) - len(df_gly)}")

    print("\n" + "="*70)
    print("STRUCTURAL FEATURE ANALYSIS - GLYCOSYLATED PROTEINS ONLY")
    print("="*70)

    # =================================================================
    # 1. B-factor Analysis (Flexibility proxy)
    # =================================================================
    print("\n1. B-FACTOR ANALYSIS (Flexibility)")
    print("-"*50)

    df_valid_bf = df_gly[df_gly['b_factor_avg'].notna()]
    print(f"Sites with valid B-factor: {len(df_valid_bf)}/{len(df_gly)}")

    if len(df_valid_bf) > 0:
        retained = df_valid_bf[df_valid_bf['sequon_retention_pct'] > 0]
        not_retained = df_valid_bf[df_valid_bf['sequon_retention_pct'] == 0]

        print(f"\nRetained (>0%): n={len(retained)}")
        if len(retained) > 0:
            print(f"  Mean B-factor: {retained['b_factor_avg'].mean():.2f} ± {retained['b_factor_avg'].std():.2f}")

        print(f"\nNot retained (0%): n={len(not_retained)}")
        if len(not_retained) > 0:
            print(f"  Mean B-factor: {not_retained['b_factor_avg'].mean():.2f} ± {not_retained['b_factor_avg'].std():.2f}")

        # Correlation
        if len(df_valid_bf) > 5:
            corr = df_valid_bf['b_factor_avg'].corr(df_valid_bf['sequon_retention_pct'])
            print(f"\nCorrelation (B-factor vs retention): r = {corr:.3f}")

    # =================================================================
    # 2. Position Analysis
    # =================================================================
    print("\n2. POSITIONAL ANALYSIS")
    print("-"*50)

    df_valid_pos = df_gly[df_gly['position_region'] != 'unknown']
    print(f"Sites with valid position: {len(df_valid_pos)}/{len(df_gly)}")

    if len(df_valid_pos) > 0:
        for region in ['N-terminal', 'middle', 'C-terminal']:
            subset = df_valid_pos[df_valid_pos['position_region'] == region]
            if len(subset) > 0:
                print(f"\n{region} (n={len(subset)}):")
                print(f"  Mean sequon retention: {subset['sequon_retention_pct'].mean():.2f}%")
                print(f"  Mean N retention: {subset['n_retention_pct'].mean():.2f}%")

    # =================================================================
    # 3. Middle Position (X) Analysis
    # =================================================================
    print("\n3. MIDDLE POSITION (X) RESIDUE EFFECT")
    print("-"*50)

    df_gly['middle_aa'] = df_gly['wt_sequon'].apply(lambda x: x[1] if len(x) > 1 else 'X')

    print("\nMiddle position (X) in N-X-S/T:")
    middle_counts = df_gly['middle_aa'].value_counts()

    results = []
    for aa in middle_counts.index:
        subset = df_gly[df_gly['middle_aa'] == aa]
        mean_ret = subset['sequon_retention_pct'].mean()
        results.append({
            'AA': aa,
            'Count': len(subset),
            'Mean_Retention': mean_ret
        })

    results_df = pd.DataFrame(results).sort_values('Mean_Retention', ascending=False)
    print("\n" + results_df.to_string(index=False))

    # =================================================================
    # 4. High Retention Sites
    # =================================================================
    print("\n4. HIGH RETENTION SITES (>0%)")
    print("-"*50)

    high_ret = df_gly[df_gly['sequon_retention_pct'] > 0].sort_values('sequon_retention_pct', ascending=False)
    print(f"Sites with >0% retention: {len(high_ret)}/{len(df_gly)}")

    if len(high_ret) > 0:
        print("\n" + high_ret[['pdb_id', 'wt_sequon', 'context', 'sequon_retention_pct',
                              'n_retention_pct', 'position_region']].to_string(index=False))

    # =================================================================
    # 5. Per-Protein Summary
    # =================================================================
    print("\n5. PER-PROTEIN SUMMARY")
    print("-"*50)

    protein_summary = df_gly.groupby('pdb_id').agg({
        'wt_sequon': 'count',
        'sequon_retention_pct': 'mean',
        'n_retention_pct': 'mean'
    }).rename(columns={
        'wt_sequon': 'n_sequons',
        'sequon_retention_pct': 'mean_sequon_ret',
        'n_retention_pct': 'mean_n_ret'
    }).sort_values('mean_sequon_ret', ascending=False)

    print("\n" + protein_summary.to_string())

    # =================================================================
    # Summary Statistics
    # =================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (GLYCOSYLATED ONLY)")
    print("="*70)

    print(f"\nTotal glycosylated sequon sites: {len(df_gly)}")
    print(f"Sites with >0% retention: {len(df_gly[df_gly['sequon_retention_pct'] > 0])}")
    print(f"Sites with 0% retention: {len(df_gly[df_gly['sequon_retention_pct'] == 0])}")

    print(f"\nOverall mean sequon retention: {df_gly['sequon_retention_pct'].mean():.2f}%")
    print(f"Overall mean N retention: {df_gly['n_retention_pct'].mean():.2f}%")
    print(f"Overall mean S/T retention: {df_gly['st_retention_pct'].mean():.2f}%")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Most retained sequon
    max_ret = df_gly.loc[df_gly['sequon_retention_pct'].idxmax()]
    print(f"\n1. Highest retention site: {max_ret['pdb_id']} {max_ret['wt_sequon']} ({max_ret['sequon_retention_pct']:.1f}%)")

    # Glycine effect
    gly_sites = df_gly[df_gly['middle_aa'] == 'G']
    other_sites = df_gly[df_gly['middle_aa'] != 'G']
    print(f"\n2. Glycine at X position effect:")
    print(f"   NGT/NGS sites: {len(gly_sites)} sites, mean retention {gly_sites['sequon_retention_pct'].mean():.1f}%")
    print(f"   Other N-X-S/T: {len(other_sites)} sites, mean retention {other_sites['sequon_retention_pct'].mean():.1f}%")

    # Save results
    output_file = base_dir / 'structural_features_glycosylated_only.csv'
    df_gly.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return df_gly


if __name__ == '__main__':
    result = main()
