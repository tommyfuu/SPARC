"""Paired t-test of per-feature MSE: SPARC-P ensemble vs the random-chance baseline.

Reads the CSV from eval_0_pretrained_mse_vs_chance.py; each time-series feature is one pair.
Usage: python eval_0_mse_pairwise_test.py --input_csv .../mse_results_MSK_CAIA_parsimonious.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# command-line arguments
parser = argparse.ArgumentParser(description='Statistical comparison of MSE between SPARC-P and Random Chance')
parser.add_argument('--input_csv', type=str, required=True,
                    help='Path to MSE results CSV file')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for plots (defaults to same directory as input CSV)')

args = parser.parse_args()

# load data
print("\n" + "="*70)
print("LOADING MSE RESULTS")
print("="*70)

if not os.path.exists(args.input_csv):
    raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

df = pd.read_csv(args.input_csv)
print(f"Loaded {len(df)} rows from {args.input_csv}")

if args.output_dir is None:
    args.output_dir = os.path.dirname(args.input_csv)
os.makedirs(args.output_dir, exist_ok=True)

# Extract dataset and modules from filename for labeling
filename = os.path.basename(args.input_csv)
dataset_info = filename.replace('mse_results_', '').replace('.csv', '')

# filter data
print("\n" + "="*70)
print("FILTERING DATA")
print("="*70)

# Filter to ensemble SPARC-P and baseline Random_Chance
# Exclude aggregate rows (feature_name == 'ALL_TIME_SERIES_FEATURES')
sparc_ensemble = df[(df['model'] == 'SPARC-P') &
                     (df['fold'] == 'ensemble') &
                     (df['feature_name'] != 'ALL_TIME_SERIES_FEATURES')].copy()

random_baseline = df[(df['model'] == 'Random_Chance') &
                      (df['fold'] == 'baseline') &
                      (df['feature_name'] != 'ALL_TIME_SERIES_FEATURES')].copy()

print(f"SPARC-P ensemble: {len(sparc_ensemble)} features")
print(f"Random Chance baseline: {len(random_baseline)} features")

# merge on feature_name so the comparison is paired
merged = pd.merge(
    sparc_ensemble[['feature_name', 'feature_idx', 'mse']],
    random_baseline[['feature_name', 'mse']],
    on='feature_name',
    suffixes=('_sparc', '_baseline')
)

print(f"\nPaired features: {len(merged)}")

# Remove any rows with NaN MSE values
merged_clean = merged.dropna(subset=['mse_sparc', 'mse_baseline'])
print(f"After removing NaN values: {len(merged_clean)}")

if len(merged_clean) < 3:
    print("ERROR: Not enough valid features for statistical test (need at least 3)")
    exit(1)

# statistical tests
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

mse_sparc = merged_clean['mse_sparc'].values
mse_baseline = merged_clean['mse_baseline'].values

mean_sparc = np.mean(mse_sparc)
mean_baseline = np.mean(mse_baseline)
std_sparc = np.std(mse_sparc, ddof=1)
std_baseline = np.std(mse_baseline, ddof=1)

print(f"\nDescriptive Statistics:")
print(f"  SPARC-P Ensemble:")
print(f"    Mean MSE: {mean_sparc:.6f}")
print(f"    Std MSE:  {std_sparc:.6f}")
print(f"    Min MSE:  {np.min(mse_sparc):.6f}")
print(f"    Max MSE:  {np.max(mse_sparc):.6f}")
print(f"\n  Random Chance Baseline:")
print(f"    Mean MSE: {mean_baseline:.6f}")
print(f"    Std MSE:  {std_baseline:.6f}")
print(f"    Min MSE:  {np.min(mse_baseline):.6f}")
print(f"    Max MSE:  {np.max(mse_baseline):.6f}")

# Paired t-test (two-tailed)
t_statistic, p_value = stats.ttest_rel(mse_sparc, mse_baseline)

print(f"\nPaired t-test (two-tailed):")
print(f"  t-statistic: {t_statistic:.4f}")
print(f"  p-value:     {p_value:.2e}")
if p_value < 0.001:
    print(f"  Significance: *** (p < 0.001)")
elif p_value < 0.01:
    print(f"  Significance: ** (p < 0.01)")
elif p_value < 0.05:
    print(f"  Significance: * (p < 0.05)")
else:
    print(f"  Significance: ns (p >= 0.05)")

# Effect size (Cohen's d for paired samples)
# d = mean(difference) / std(difference)
differences = mse_sparc - mse_baseline
cohens_d = np.mean(differences) / np.std(differences, ddof=1)

print(f"\nEffect Size:")
print(f"  Cohen's d:   {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interp = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interp = "small"
elif abs(cohens_d) < 0.8:
    effect_interp = "medium"
else:
    effect_interp = "large"
print(f"  Interpretation: {effect_interp}")

# Percent improvement
mean_improvement = (mean_baseline - mean_sparc) / mean_baseline * 100
print(f"\nMean Improvement:")
print(f"  {mean_improvement:.2f}% reduction in MSE")

# Count how many features improved
n_improved = np.sum(mse_sparc < mse_baseline)
n_total = len(mse_sparc)
pct_improved = n_improved / n_total * 100
print(f"\nFeatures Improved:")
print(f"  {n_improved}/{n_total} ({pct_improved:.1f}%) features have lower MSE with SPARC-P")

# visualization
print("\n" + "="*70)
print("CREATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# plot 1: scatter with identity line
ax1 = axes[0]

ax1.scatter(mse_baseline, mse_sparc, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

# Identity line (y = x)
max_val = max(np.max(mse_baseline), np.max(mse_sparc))
min_val = min(np.min(mse_baseline), np.min(mse_sparc))
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Identity (no improvement)')

# Diagonal shading to show improvement region
ax1.fill_between([min_val, max_val], min_val, max_val, alpha=0.1, color='red', label='SPARC-P worse')
ax1.fill_between([min_val, max_val], [min_val, max_val], min_val, alpha=0.1, color='green', label='SPARC-P better')

ax1.set_xlabel('Random Chance MSE', fontsize=12, fontweight='bold')
ax1.set_ylabel('SPARC-P Ensemble MSE', fontsize=12, fontweight='bold')
ax1.set_title(f'MSE Comparison: SPARC-P vs Random Chance\n{dataset_info}', fontsize=13, fontweight='bold')

# Add statistics text
stats_text = f'Paired t-test: p = {p_value:.2e}\nCohen\'s d = {cohens_d:.3f} ({effect_interp})\n{n_improved}/{n_total} features improved'
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# plot 2: histogram of differences
ax2 = axes[1]

differences = mse_sparc - mse_baseline
ax2.hist(differences, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax2.axvline(np.mean(differences), color='orange', linestyle='-', linewidth=2, label=f'Mean diff = {np.mean(differences):.3f}')

ax2.set_xlabel('MSE Difference (SPARC-P - Baseline)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_title(f'Distribution of MSE Differences\n{dataset_info}', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_file = f'{args.output_dir}/mse_comparison_pairwise_test_{dataset_info}.pdf'
plt.savefig(output_file, bbox_inches='tight', dpi=300)
print(f"Saved plot to: {output_file}")

# also save as PNG
output_file_png = output_file.replace('.pdf', '.png')
plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
print(f"Saved plot to: {output_file_png}")

plt.close()

# save detailed results to csv
print("\n" + "="*70)
print("SAVING DETAILED RESULTS")
print("="*70)

# Add improvement metrics to merged data
merged_clean['mse_difference'] = merged_clean['mse_sparc'] - merged_clean['mse_baseline']
merged_clean['mse_improvement_pct'] = (merged_clean['mse_baseline'] - merged_clean['mse_sparc']) / merged_clean['mse_baseline'] * 100
merged_clean['sparc_is_better'] = merged_clean['mse_sparc'] < merged_clean['mse_baseline']

merged_clean_sorted = merged_clean.sort_values('mse_improvement_pct', ascending=False)

detailed_output_file = f'{args.output_dir}/mse_comparison_detailed_{dataset_info}.csv'
merged_clean_sorted.to_csv(detailed_output_file, index=False)
print(f"Saved detailed comparison to: {detailed_output_file}")

# Print top/bottom features
print(f"\nTop 10 features with greatest improvement:")
print(merged_clean_sorted.head(10)[['feature_name', 'mse_sparc', 'mse_baseline', 'mse_improvement_pct']].to_string(index=False))

print(f"\nBottom 10 features (least improvement or worse):")
print(merged_clean_sorted.tail(10)[['feature_name', 'mse_sparc', 'mse_baseline', 'mse_improvement_pct']].to_string(index=False))

# save summary statistics
summary_stats = {
    'dataset': [dataset_info],
    'n_features': [len(merged_clean)],
    'mean_mse_sparc': [mean_sparc],
    'std_mse_sparc': [std_sparc],
    'mean_mse_baseline': [mean_baseline],
    'std_mse_baseline': [std_baseline],
    't_statistic': [t_statistic],
    'p_value': [p_value],
    'cohens_d': [cohens_d],
    'effect_size_interpretation': [effect_interp],
    'mean_improvement_pct': [mean_improvement],
    'n_features_improved': [n_improved],
    'pct_features_improved': [pct_improved]
}

summary_df = pd.DataFrame(summary_stats)
summary_output_file = f'{args.output_dir}/mse_comparison_summary_{dataset_info}.csv'
summary_df.to_csv(summary_output_file, index=False)
print(f"\nSaved summary statistics to: {summary_output_file}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
