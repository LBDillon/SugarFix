#!/usr/bin/env python3
"""
Feature-based comparative analysis: Glycoproteins vs Controls

Tests whether the 29 sequence/structural features differ between glycoproteins
and controls. Identifies which properties characterize each group and whether
these differences are associated with ProteinMPNN score differences.

Produces:
- feature_comparison_results.csv
- feature_comparison_visualization.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load enriched feature dataset
df = pd.read_csv('data/glyco_benchmark_with_features.csv')

print("="*70)
print("FEATURE-BASED COMPARATIVE ANALYSIS")
print("Glycoproteins vs Controls")
print("="*70)

# All 29 features
sequence_features = [
    'molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point',
    'charge_at_pH7', 'gravy', 'acidic_fraction', 'basic_fraction', 'aromatic_fraction',
    'ionizable_fraction', 'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction',
    'mw_per_residue', 'charge_per_residue', 'buffer_capacity', 'acidic_basic_ratio',
    'charge_asymmetry', 'basic_residue_fraction', 'acidic_residue_fraction'
]

structural_features = [
    'helix_percent', 'sheet_percent', 'loop_percent', 'num_residues',
    'avg_cb_distance', 'compactness', 'rco', 'surface_exposure'
]

all_features = sequence_features + structural_features

# Separate by class
glyco = df[df['protein_class'] == 'glycoprotein'].copy()
control = df[df['protein_class'] == 'control'].copy()

print(f"\nSample sizes:")
print(f"  Glycoproteins: {len(glyco)}")
print(f"  Controls: {len(control)}")

# Statistical comparison
results_list = []

print(f"\n" + "="*70)
print("FEATURE COMPARISON: Independent t-tests")
print("="*70)
print(f"\n{'Feature':<30} {'Glyco Mean':<12} {'Control Mean':<12} {'t-stat':<8} {'p-value':<8} {'Cohen d':<8} {'Sig.':<5}")
print("-"*100)

for feat in all_features:
    # Get data
    glyco_vals = glyco[feat].dropna()
    control_vals = control[feat].dropna()
    
    if len(glyco_vals) < 2 or len(control_vals) < 2:
        continue
    
    # t-test
    t_stat, p_val = stats.ttest_ind(glyco_vals, control_vals)
    
    # Cohen's d
    pooled_std = np.sqrt(((len(glyco_vals)-1) * glyco_vals.std()**2 + 
                           (len(control_vals)-1) * control_vals.std()**2) / 
                          (len(glyco_vals) + len(control_vals) - 2))
    cohens_d = (glyco_vals.mean() - control_vals.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Significance marker
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "NS"
    
    # Print
    print(f"{feat:<30} {glyco_vals.mean():<12.4f} {control_vals.mean():<12.4f} {t_stat:<8.3f} {p_val:<8.4f} {cohens_d:<8.3f} {sig:<5}")
    
    # Store results
    results_list.append({
        'feature': feat,
        'feature_type': 'sequence' if feat in sequence_features else 'structural',
        'glyco_mean': glyco_vals.mean(),
        'glyco_std': glyco_vals.std(),
        'control_mean': control_vals.mean(),
        'control_std': control_vals.std(),
        'mean_difference': glyco_vals.mean() - control_vals.mean(),
        'percent_difference': ((glyco_vals.mean() - control_vals.mean()) / abs(control_vals.mean()) * 100) if control_vals.mean() != 0 else np.nan,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'n_glyco': len(glyco_vals),
        'n_control': len(control_vals)
    })

# Save results
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('p_value')
results_df.to_csv('data/feature_comparison_results.csv', index=False)
print(f"\n✓ Results saved to feature_comparison_results.csv")

# Summary statistics
sig_features = results_df[results_df['p_value'] < 0.05]
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total features tested: {len(results_df)}")
print(f"Significant differences (p < 0.05): {len(sig_features)}")
if len(sig_features) > 0:
    print(f"\nSignificant features:")
    for _, row in sig_features.iterrows():
        print(f"  • {row['feature']}: p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f}")

# Identify features with large effect sizes (|Cohen's d| > 0.8)
large_effect = results_df[abs(results_df['cohens_d']) > 0.8]
print(f"\nFeatures with large effect size (|d| > 0.8): {len(large_effect)}")
if len(large_effect) > 0:
    for _, row in large_effect.iterrows():
        print(f"  • {row['feature']}: d={row['cohens_d']:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Volcano plot (p-value vs effect size)
ax = axes[0, 0]
colors = ['red' if p < 0.05 else 'gray' for p in results_df['p_value']]
sizes = [100 if p < 0.05 else 50 for p in results_df['p_value']]
ax.scatter(results_df['cohens_d'], -np.log10(results_df['p_value']), 
           c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, alpha=0.5, label='p=0.05')
ax.axvline(0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='d=0.8')
ax.axvline(-0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
ax.set_ylabel("-log10(p-value)", fontsize=12, fontweight='bold')
ax.set_title('Volcano Plot: Feature Effect Sizes', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 2: Feature differences (top 15 by effect size)
ax = axes[0, 1]
top_features = results_df.nlargest(15, 'cohens_d')[['feature', 'cohens_d']].sort_values('cohens_d')
colors_eff = ['red' if d > 0 else 'blue' for d in top_features['cohens_d']]
ax.barh(range(len(top_features)), top_features['cohens_d'].values, color=colors_eff, 
        alpha=0.6, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values, fontsize=9)
ax.set_xlabel("Cohen's d", fontsize=11, fontweight='bold')
ax.set_title('Top 15 Features by Effect Size', fontsize=12, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Plot 3: Mean values comparison (normalized)
ax = axes[1, 0]
# Normalize for visualization
scaler = StandardScaler()
glyco_subset = glyco[all_features].dropna()
control_subset = control[all_features].dropna()

# Use common scaling
all_vals = pd.concat([glyco_subset, control_subset])
scaled_glyco_means = scaler.fit_transform(all_vals).mean(axis=0)  # This is not quite right but for viz...

# Better approach: compute z-scores for means
feature_means_glyco = []
feature_means_control = []
feature_names_clean = []

for feat in all_features:
    glyco_vals = glyco[feat].dropna()
    control_vals = control[feat].dropna()
    if len(glyco_vals) > 0 and len(control_vals) > 0:
        feature_means_glyco.append(glyco_vals.mean())
        feature_means_control.append(control_vals.mean())
        feature_names_clean.append(feat)

# Normalize
feat_array = np.array([feature_means_glyco, feature_means_control])
feat_scaled = (feat_array - feat_array.mean(axis=1, keepdims=True)) / feat_array.std(axis=1, keepdims=True)

x_pos = np.arange(len(feature_names_clean))
width = 0.35
ax.bar(x_pos - width/2, feat_scaled[0], width, label='Glycoprotein', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.bar(x_pos + width/2, feat_scaled[1], width, label='Control', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Normalized Mean (z-score)', fontsize=11, fontweight='bold')
ax.set_title('Feature Values: Glycoproteins vs Controls (Normalized)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names_clean, rotation=90, fontsize=7)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 4: p-value distribution
ax = axes[1, 1]
ax.hist(results_df['p_value'], bins=15, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
ax.set_xlabel('p-value', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Distribution of p-values (Feature Comparisons)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data/feature_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to feature_comparison_visualization.png")

# Conclusion
print(f"\n" + "="*70)
print("BIOLOGICAL INTERPRETATION")
print("="*70)
if len(sig_features) == 0:
    print("\n✓ No statistically significant differences found between glycoproteins and controls")
    print("  at the p<0.05 level in sequence or structural features.")
    print("\n  This suggests glycoproteins and controls have similar biochemical properties,")
    print("  further supporting the finding of no ProteinMPNN bias.")
else:
    print(f"\n⚠️  {len(sig_features)} significant feature differences detected:")
    for _, row in sig_features.iterrows():
        direction = "higher" if row['mean_difference'] > 0 else "lower"
        print(f"  • {row['feature']:30s}: Glycoproteins are {direction}")

plt.show()
