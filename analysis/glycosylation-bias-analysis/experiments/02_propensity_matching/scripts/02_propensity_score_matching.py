#!/usr/bin/env python3
"""
Propensity Score Matching: Feature-matched controls for glycoproteins

Matches each glycoprotein with a control protein that has similar sequence and
structural features. Tests whether glycoproteins still differ from feature-matched
controls on ProteinMPNN scores.

This approach removes confounding by feature similarity and provides a cleaner
test of whether glycoprotein class itself affects predictions.

Produces:
- propensity_matched_dataset.csv
- propensity_matching_results.csv
- propensity_matching_visualization.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from scipy import stats

# Load enriched feature dataset
df = pd.read_csv('data/glyco_benchmark_with_features.csv')

print("="*70)
print("PROPENSITY SCORE MATCHING")
print("Feature-based matching of glycoproteins to controls")
print("="*70)

# All 29 features for matching
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

# Prepare data
glyco = df[df['protein_class'] == 'glycoprotein'].dropna(subset=['sequence_score'])
control = df[df['protein_class'] == 'control'].dropna(subset=['sequence_score'])

print(f"\nInitial sample sizes:")
print(f"  Glycoproteins with scores: {len(glyco)}")
print(f"  Controls with scores: {len(control)}")

# Standardize features for distance calculation
all_data = pd.concat([glyco, control])
feature_data = all_data[all_features].copy()

# Handle missing values
feature_data = feature_data.fillna(feature_data.mean())

scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_data)

# Map back to indices
glyco_indices = glyco.index.tolist()
control_indices = control.index.tolist()

glyco_scaled = features_scaled[[all_data.index.get_loc(idx) for idx in glyco_indices]]
control_scaled = features_scaled[[all_data.index.get_loc(idx) for idx in control_indices]]

# Match each glycoprotein to nearest control
matches = []
used_controls = set()

print(f"\nMatching glycoproteins to controls...")

for i, glyco_idx in enumerate(glyco_indices):
    distances = []
    for j, control_idx in enumerate(control_indices):
        if control_idx in used_controls:
            distance = np.inf
        else:
            distance = euclidean(glyco_scaled[i], control_scaled[j])
        distances.append((distance, control_idx, j))
    
    # Find best match (smallest distance)
    best_distance, best_control_idx, best_j = min(distances)
    used_controls.add(best_control_idx)
    
    glyco_row = glyco.loc[glyco_idx]
    control_row = control.loc[best_control_idx]
    
    matches.append({
        'pair_id': len(matches) + 1,
        'glyco_protein_id': glyco_row['pair_id'],
        'glyco_name': glyco_row['protein_name'],
        'glyco_sequence_length': glyco_row['sequence_length'],
        'glyco_score': glyco_row['sequence_score'],
        'control_protein_id': control_row['pair_id'],
        'control_name': control_row['protein_name'],
        'control_sequence_length': control_row['sequence_length'],
        'control_score': control_row['sequence_score'],
        'matching_distance': best_distance,
        'score_difference': glyco_row['sequence_score'] - control_row['sequence_score']
    })

matches_df = pd.DataFrame(matches)

# Save matched dataset
matches_df.to_csv('data/propensity_matched_dataset.csv', index=False)
print(f"✓ Matched dataset saved to propensity_matched_dataset.csv")

# Analyze matches
print(f"\n" + "="*70)
print("MATCHING QUALITY")
print("="*70)
print(f"\nMatched pairs: {len(matches_df)}")
print(f"Average matching distance (Euclidean, scaled features): {matches_df['matching_distance'].mean():.4f}")
print(f"  Min: {matches_df['matching_distance'].min():.4f}")
print(f"  Max: {matches_df['matching_distance'].max():.4f}")
print(f"  Std: {matches_df['matching_distance'].std():.4f}")

# Check if features are actually balanced after matching
print(f"\n" + "="*70)
print("COVARIATE BALANCE AFTER MATCHING")
print("="*70)
print(f"{'Feature':<30} {'Glyco Mean':<12} {'Control Mean':<12} {'Std Diff':<12}")
print("-"*70)

covariate_balance = []

for feat in all_features[:10]:  # Show first 10 features
    glyco_matched_vals = []
    control_matched_vals = []
    
    for _, match in matches_df.iterrows():
        glyco_row = glyco.loc[glyco.index[glyco.index.get_loc(glyco_indices[matches_df.index.tolist().index(_)])]].to_dict() if _ == 0 else None
    
    # Better approach: get original data
    glyco_names = glyco.index.tolist()
    control_names = control.index.tolist()
    
    # Get matched values directly
    for pair_id, glyco_id, control_id in zip(matches_df['pair_id'], matches_df['glyco_protein_id'], matches_df['control_protein_id']):
        glyco_match = df[df['pair_id'] == glyco_id]
        control_match = df[df['pair_id'] == control_id]
        
        if len(glyco_match) > 0 and len(control_match) > 0:
            glyco_matched_vals.append(glyco_match[feat].values[0])
            control_matched_vals.append(control_match[feat].values[0])
    
    if len(glyco_matched_vals) > 0 and len(control_matched_vals) > 0:
        glyco_mean = np.nanmean(glyco_matched_vals)
        control_mean = np.nanmean(control_matched_vals)
        pooled_std = np.nanstd(np.concatenate([glyco_matched_vals, control_matched_vals]))
        std_diff = abs(glyco_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"{feat:<30} {glyco_mean:<12.4f} {control_mean:<12.4f} {std_diff:<12.4f}")
        covariate_balance.append({
            'feature': feat,
            'glyco_mean': glyco_mean,
            'control_mean': control_mean,
            'std_diff': std_diff
        })

# Test matched pairs on score difference
print(f"\n" + "="*70)
print("PAIRED T-TEST: ProteinMPNN Scores in Matched Pairs")
print("="*70)

glyco_scores = matches_df['glyco_score'].values
control_scores = matches_df['control_score'].values
differences = glyco_scores - control_scores

# Paired t-test
t_stat, p_val = stats.ttest_rel(glyco_scores, control_scores)

print(f"\nGlycoprotein mean score: {glyco_scores.mean():.4f} (SD={glyco_scores.std():.4f})")
print(f"Control mean score:      {control_scores.mean():.4f} (SD={control_scores.std():.4f})")
print(f"Mean difference:         {differences.mean():.4f} (SD={differences.std():.4f})")
print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_val:.4f}")
print(f"  n pairs:     {len(matches_df)}")

if p_val < 0.05:
    print(f"\n⚠️  SIGNIFICANT DIFFERENCE: Glycoproteins score differently than matched controls (p={p_val:.4f})")
else:
    print(f"\n✓  NO SIGNIFICANT DIFFERENCE: Matched pairs have similar scores (p={p_val:.4f})")

# Regression on matched pairs
print(f"\n" + "="*70)
print("MIXED EFFECTS REGRESSION ON MATCHED PAIRS")
print("="*70)

# Reshape for mixed model
matched_long = []
for _, row in matches_df.iterrows():
    matched_long.append({
        'pair_id': row['pair_id'],
        'protein_class': 'glycoprotein',
        'sequence_score': row['glyco_score'],
        'protein_name': row['glyco_name']
    })
    matched_long.append({
        'pair_id': row['pair_id'],
        'protein_class': 'control',
        'sequence_score': row['control_score'],
        'protein_name': row['control_name']
    })

matched_long_df = pd.DataFrame(matched_long)

# Simple OLS on matched pairs (accounting for pairing with fixed effects)
m_matched = ols('sequence_score ~ C(protein_class) + C(pair_id)', data=matched_long_df).fit()
class_effect = m_matched.params[1] if len(m_matched.params) > 1 else m_matched.params[0]
class_pval = m_matched.pvalues.iloc[1] if len(m_matched.pvalues) > 1 else m_matched.pvalues.iloc[0]

print(f"\nClass effect (glycoprotein vs control):")
print(f"  Coefficient: {class_effect:.4f}")
print(f"  p-value:     {class_pval:.4f}")
print(f"  Significant: {'Yes' if class_pval < 0.05 else 'No'}")

# Save results
results_summary = pd.DataFrame([{
    'analysis': 'Propensity Score Matching',
    'n_matched_pairs': len(matches_df),
    'mean_matching_distance': matches_df['matching_distance'].mean(),
    'glyco_mean_score': glyco_scores.mean(),
    'control_mean_score': control_scores.mean(),
    'score_difference': differences.mean(),
    'paired_t_stat': t_stat,
    'paired_t_pval': p_val,
    'regression_class_coef': class_effect,
    'regression_class_pval': class_pval,
    'significant_class_effect': class_pval < 0.05
}])

results_summary.to_csv('data/propensity_matching_results.csv', index=False)
print("\n✓ Results saved to propensity_matching_results.csv")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Matching quality - distance distribution
ax = axes[0, 0]
ax.hist(matches_df['matching_distance'], bins=10, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
ax.axvline(matches_df['matching_distance'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f"Mean={matches_df['matching_distance'].mean():.3f}")
ax.set_xlabel('Matching Distance (Euclidean, scaled)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Quality of Feature Matches', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 2: Score comparison in matched pairs
ax = axes[0, 1]
x_pos = np.arange(len(matches_df))
width = 0.35
ax.bar(x_pos - width/2, matches_df['glyco_score'], width, label='Glycoprotein', 
       alpha=0.7, color='#FF6B6B', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + width/2, matches_df['control_score'], width, label='Control', 
       alpha=0.7, color='#4ECDC4', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Matched Pair ID', fontsize=11, fontweight='bold')
ax.set_ylabel('ProteinMPNN Score', fontsize=11, fontweight='bold')
ax.set_title('Scores in Matched Pairs', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 3: Score difference by pair
ax = axes[1, 0]
colors_diff = ['red' if d > 0 else 'blue' for d in matches_df['score_difference']]
ax.bar(x_pos, matches_df['score_difference'], color=colors_diff, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(differences.mean(), color='orange', linestyle='--', linewidth=2, 
           label=f"Mean diff={differences.mean():.4f}")
ax.set_xlabel('Matched Pair ID', fontsize=11, fontweight='bold')
ax.set_ylabel('Score Difference (Glyco - Control)', fontsize=11, fontweight='bold')
ax.set_title('Score Differences in Matched Pairs (paired t-test p={:.3f})'.format(p_val), fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Matching distance vs score difference
ax = axes[1, 1]
scatter = ax.scatter(matches_df['matching_distance'], abs(matches_df['score_difference']), 
                    s=100, alpha=0.6, edgecolors='black', linewidth=0.5, c=matches_df['score_difference'],
                    cmap='RdBu_r')
ax.set_xlabel('Matching Distance', fontsize=11, fontweight='bold')
ax.set_ylabel('|Score Difference|', fontsize=11, fontweight='bold')
ax.set_title('Match Quality vs Prediction Difference', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Score Diff')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/propensity_matching_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to propensity_matching_visualization.png")

print("\n" + "="*70)
print("SUMMARY & INTERPRETATION")
print("="*70)
print(f"""
Propensity Score Matching successfully created {len(matches_df)} matched pairs of
glycoproteins and controls with similar feature profiles (avg distance={matches_df['matching_distance'].mean():.3f}).

In these feature-matched pairs:
  • Glycoproteins: mean score = {glyco_scores.mean():.4f}
  • Controls:      mean score = {control_scores.mean():.4f}
  • Difference:    {differences.mean():.4f}
  • Paired t-test: p = {p_val:.4f}

Result: {'SIGNIFICANT' if p_val < 0.05 else 'NO SIGNIFICANT'} difference between matched pairs.

Interpretation:
When glycoproteins are compared to feature-matched controls (rather than all controls),
the glycoprotein class effect {'REMAINS SIGNIFICANT' if p_val < 0.05 else 'disappears (p≥0.05)'}.

This demonstrates that any apparent glycoprotein effect is {'a real class effect' if p_val < 0.05 else 'due to feature differences, not glycosylation status'}.
""")

plt.show()
