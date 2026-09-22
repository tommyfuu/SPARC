"""Directly-supervised vs pretrained-then-finetuned SPARC across training-data
fractions, one panel per task.

Plots absolute performance each task with
a Spearman rho and p-value quantifying whether the pretraining benefit shrinks
as training data grows. Side-effect tasks use test_auroc_nextinf_weighted,
binary tasks use test_auroc. Fractions are equally spaced.

Reads the CSVs from eval_4_titration_compare_all_fractions.py.
"""

# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# Set font to Arial
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

# Paths
model_path = sparc_paths.SPARC_MODEL_ROOT
results_path = f'{model_path}/results_vis_060726'
csv_dir = f'{results_path}/eval_trainsize'
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--save_dir', type=str, default=None,
                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without touching the published ones.')
args = parser.parse_args()

save_dir = args.save_dir if args.save_dir else csv_dir
Path(save_dir).mkdir(parents=True, exist_ok=True)

dataset = 'MSK_CAIA'  # Default dataset used in evaluation

# Task configurations
side_effect_tasks = {
    'febrile_neutropenia_risk': 'Febrile Neutropenia',
    'neutropenia_risk': 'Grade 3 Neutropenia',
    'anemia_risk': 'Grade 3 Anemia',
    'AKI_risk': 'Acute kidney injury',
    'Liver_toxicity_risk': 'Liver Toxicity'
}
binary_tasks = {
    'binary': '180D Hosp. mort.',
    'binary_icu_pre': '30D ICU mort.'
}

# Fraction ordering (smallest to largest)
side_effect_fractions = ['1/256', '1/128', '1/64', '1/16', '1/4', '1/1']
binary_fractions = ['1/256', '1/128', '1/64', '1/16', '1/4', '1/1']
binary_icu_pre_fractions = ['1/256', '1/128', '1/64', '1/32', '1/16', '1/4', '1/1']

# Color scheme
colors = {
    'SPARC': '#2E86AB',      # Blue
    'SPARC-P': '#A23B72',    # Purple/Magenta
}

markers = {
    'SPARC': 'o',
    'SPARC-P': 's',
}

# Display labels for legend
display_labels = {
    'SPARC': 'SPARC (directly supervised)',
    'SPARC-P': 'SPARC (pretrained-then-finetuned)'
}

def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def compute_spearman_correlation(raw_values, fractions):
    """
    Compute Spearman correlation between training fraction and pretraining benefit.

    A negative correlation indicates that pretraining benefit decreases as
    training data increases (i.e., pretraining helps more with less data).

    Returns:
        dict: {'rho': correlation coefficient, 'p_value': p-value}
    """
    # Map fractions to numeric values
    fraction_numeric = {
        '1/256': 1/256,
        '1/128': 1/128,
        '1/64': 1/64,
        '1/32': 1/32,
        '1/16': 1/16,
        '1/4': 1/4,
        '1/1': 1.0
    }

    fraction_vals = []
    benefit_vals = []

    for fraction in fractions:
        if fraction not in raw_values:
            continue

        sparc_vals = raw_values[fraction].get('SPARC', np.array([]))
        sparc_p_vals = raw_values[fraction].get('SPARC-P', np.array([]))

        if len(sparc_vals) > 0 and len(sparc_p_vals) > 0:
            sparc_mean = sparc_vals.mean()
            sparc_p_mean = sparc_p_vals.mean()

            # Calculate relative benefit: (SPARC-P - SPARC) / SPARC * 100
            if sparc_mean != 0:
                benefit = ((sparc_p_mean - sparc_mean) / sparc_mean) * 100
            else:
                benefit = 0

            fraction_vals.append(fraction_numeric[fraction])
            benefit_vals.append(benefit)

    # Need at least 3 points for correlation
    if len(fraction_vals) >= 3:
        rho, p_value = stats.spearmanr(fraction_vals, benefit_vals)
        return {'rho': rho, 'p_value': p_value}
    else:
        return {'rho': np.nan, 'p_value': np.nan}

def load_data_for_task(task, fractions):
    """Load performance data for a specific task across all fractions with CIs."""
    csv_path = f'{csv_dir}/trainsize_comparison_{task}_{dataset}_all_fractions.csv'

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: File not found: {csv_path}")
        return pd.DataFrame(), {}

    # Determine metric column
    if task in side_effect_tasks:
        metric_col = 'test_auroc_nextinf_weighted'
    else:  # binary tasks
        metric_col = 'auroc'

    if metric_col not in df.columns:
        print(f"Warning: Metric column '{metric_col}' not found in {csv_path}")
        return pd.DataFrame(), {}

    data = []
    raw_values = {}  # Store raw values for t-tests

    for fraction in fractions:
        # Filter for this fraction
        df_frac = df[df['data_fraction'] == fraction]

        if df_frac.empty:
            continue

        raw_values[fraction] = {}

        # Get SPARC and SPARC-P values with confidence intervals
        for model_name, model_label in [('SPARC_direct', 'SPARC'), ('SPARC_finetuned', 'SPARC-P')]:
            model_rows = df_frac[df_frac['model'] == model_name]
            if not model_rows.empty:
                values = model_rows[metric_col].dropna().values
                raw_values[fraction][model_label] = values

                if len(values) > 0:
                    mean_val = values.mean()
                    # Calculate 95% confidence interval
                    if len(values) > 1:
                        ci = stats.t.interval(0.95, len(values)-1,
                                             loc=mean_val,
                                             scale=stats.sem(values))
                        ci_lower, ci_upper = ci
                    else:
                        ci_lower = ci_upper = mean_val

                    data.append({
                        'fraction': fraction,
                        'model_type': model_label,
                        'performance': mean_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })

    return pd.DataFrame(data), raw_values

def compute_significance(raw_values, fractions):
    """
    Compute pairwise t-tests between SPARC and SPARC-P at each fraction
    with Bonferroni correction.

    Returns:
        dict: {fraction: {'p_value': p, 'p_corrected': p_corr, 'stars': stars}}
    """
    results = {}
    n_tests = len(fractions)

    for fraction in fractions:
        if fraction not in raw_values:
            continue

        sparc_vals = raw_values[fraction].get('SPARC', np.array([]))
        sparc_p_vals = raw_values[fraction].get('SPARC-P', np.array([]))

        if len(sparc_vals) > 1 and len(sparc_p_vals) > 1:
            # Perform paired or independent t-test
            # Use independent t-test since these are different models
            t_stat, p_value = stats.ttest_ind(sparc_vals, sparc_p_vals)

            # Apply Bonferroni correction
            p_corrected = min(p_value * n_tests, 1.0)

            results[fraction] = {
                'p_value': p_value,
                'p_corrected': p_corrected,
                'stars': get_significance_stars(p_corrected),
                't_stat': t_stat
            }
        else:
            results[fraction] = {
                'p_value': np.nan,
                'p_corrected': np.nan,
                'stars': 'n/a',
                't_stat': np.nan
            }

    return results

def plot_task_performance(task, task_name, fractions, ax):
    """Plot performance for a single task with Spearman correlation."""
    df, raw_values = load_data_for_task(task, fractions)

    if df.empty:
        ax.text(0.5, 0.5, f'No data for {task_name}',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Compute Spearman correlation
    spearman_result = compute_spearman_correlation(raw_values, fractions)

    # Create x-axis positions (equal spacing)
    x_positions = np.arange(len(fractions))

    # Plot each model type
    for model_type in ['SPARC', 'SPARC-P']:
        model_data = df[df['model_type'] == model_type].copy()
        model_data['x_pos'] = model_data['fraction'].map(
            {frac: i for i, frac in enumerate(fractions)}
        )
        model_data = model_data.sort_values('x_pos')

        # Plot line
        ax.plot(model_data['x_pos'], model_data['performance'],
                color=colors[model_type], marker=markers[model_type],
                label=display_labels[model_type], linewidth=2.5, markersize=8,
                markeredgewidth=1.5, markeredgecolor='white')

        # Add confidence interval shading
        ax.fill_between(model_data['x_pos'],
                        model_data['ci_lower'],
                        model_data['ci_upper'],
                        color=colors[model_type],
                        alpha=0.2)

    # Formatting
    ax.set_xlabel('Training Data Fraction', fontweight='bold')
    ax.set_ylabel('AUROC', fontweight='bold')

    # Create title with task name and Spearman correlation
    rho = spearman_result['rho']
    p_val = spearman_result['p_value']

    if not np.isnan(rho) and not np.isnan(p_val):
        # Format p-value
        if p_val < 0.001:
            p_str = "p<0.001"
        else:
            p_str = f"p={p_val:.3f}"

        # Add significance indicator
        sig_indicator = ""
        if p_val < 0.05:
            sig_indicator = " *"
        if p_val < 0.01:
            sig_indicator = " **"
        if p_val < 0.001:
            sig_indicator = " ***"

        title_text = f"{task_name}\nSpearman ρ={rho:.3f}, {p_str}{sig_indicator}"
    else:
        title_text = task_name

    ax.set_title(title_text, fontweight='bold', pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(fractions, rotation=45, ha='right')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Use custom y-axis limits based on data range
    all_values = df['performance'].values
    if len(all_values) > 0:
        y_min = all_values.min() - 0.02
        y_max = min(1.0, all_values.max() + 0.05)  # Some padding for visibility
        ax.set_ylim([y_min, y_max])

    # Add horizontal line at 0.5 (random) if visible in range
    y_lim = ax.get_ylim()
    if y_lim[0] <= 0.5 <= y_lim[1]:
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Create figure with subplots for all tasks
print("Creating per-task line plots with significance testing...")

# Side effect tasks (5 tasks)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (task, task_name) in enumerate(side_effect_tasks.items()):
    plot_task_performance(task, task_name, side_effect_fractions, axes[idx])

# Hide the last subplot (we have 5 tasks but 6 subplots)
axes[5].axis('off')

plt.tight_layout()
plt.savefig(f'{save_dir}/titration_side_effect_tasks_lineplot_with_significance.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/titration_side_effect_tasks_lineplot_with_significance.png', bbox_inches='tight', dpi=300)
print(f"Saved: {save_dir}/titration_side_effect_tasks_lineplot_with_significance.pdf")
plt.close()

# Binary tasks (2 tasks - note binary_icu_pre has more fractions)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_task_performance('binary', binary_tasks['binary'], binary_fractions, axes[0])
plot_task_performance('binary_icu_pre', binary_tasks['binary_icu_pre'], binary_icu_pre_fractions, axes[1])

plt.tight_layout()
plt.savefig(f'{save_dir}/titration_binary_tasks_lineplot_with_significance.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/titration_binary_tasks_lineplot_with_significance.png', bbox_inches='tight', dpi=300)
print(f"Saved: {save_dir}/titration_binary_tasks_lineplot_with_significance.pdf")
plt.close()

# Create combined figure with all tasks
print("Creating combined figure with all tasks...")
fig = plt.figure(figsize=(20, 12))

# Side effect tasks in top 2 rows
for idx, (task, task_name) in enumerate(side_effect_tasks.items()):
    ax = plt.subplot(3, 3, idx + 1)
    plot_task_performance(task, task_name, side_effect_fractions, ax)

# Binary tasks in bottom row
ax = plt.subplot(3, 3, 7)
plot_task_performance('binary', binary_tasks['binary'], binary_fractions, ax)

ax = plt.subplot(3, 3, 8)
plot_task_performance('binary_icu_pre', binary_tasks['binary_icu_pre'], binary_icu_pre_fractions, ax)

plt.tight_layout()
plt.savefig(f'{save_dir}/titration_all_tasks_lineplot_with_significance.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{save_dir}/titration_all_tasks_lineplot_with_significance.png', bbox_inches='tight', dpi=300)
print(f"Saved: {save_dir}/titration_all_tasks_lineplot_with_significance.pdf")
plt.close()

# Create a summary table of Spearman correlation results
print("\n" + "="*100)
print("SPEARMAN CORRELATION SUMMARY")
print("="*100)
print("Negative correlation = pretraining benefit decreases with more training data")
print("(i.e., pretraining helps more with less data)")
print("-"*100)
print(f"{'Task':<30} {'Spearman ρ':<15} {'p-value':<12} {'Significance':<15} {'Interpretation'}")
print("-"*100)

for task_key, task_name in {**side_effect_tasks, **binary_tasks}.items():
    # Determine which fractions to use
    if task_key in side_effect_tasks:
        task = task_key
        fractions = side_effect_fractions
    elif task_key == 'binary':
        task = task_key
        fractions = binary_fractions
    else:  # binary_icu_pre
        task = task_key
        fractions = binary_icu_pre_fractions

    df, raw_values = load_data_for_task(task, fractions)
    if not df.empty:
        spearman_result = compute_spearman_correlation(raw_values, fractions)
        rho = spearman_result['rho']
        p_val = spearman_result['p_value']

        if not np.isnan(rho) and not np.isnan(p_val):
            # Determine significance
            if p_val < 0.001:
                sig_str = "***"
            elif p_val < 0.01:
                sig_str = "**"
            elif p_val < 0.05:
                sig_str = "*"
            else:
                sig_str = "ns"

            # Interpretation
            if rho < 0:
                interp = "Benefit ↓ with more data"
            else:
                interp = "Benefit ↑ with more data"

            print(f"{task_name:<30} {rho:>14.3f} {p_val:>11.4f} {sig_str:<15} {interp}")

print("="*100)
print("\nVisualization complete!")
print(f"Plots saved to: {save_dir}")
print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
print("\nInterpretation:")
print("  - Negative ρ with p<0.05: Strong evidence that pretraining helps more with less data")
print("  - Negative ρ with p≥0.05: Trend suggesting pretraining helps more with less data (not significant)")
print("  - Positive ρ: Opposite pattern (pretraining benefit increases with more data)")
