"""SPARC-P performance across medication regimens, all tasks in one figure.

One horizontal-bar panel per task, bars are mean +/- std over folds with a red
diamond for the ensemble. --combo draws the 'all' baseline as a bar and the
specific regimens as a forest plot; --fn drops febrile neutropenia for a 2x2.

Reads the CSVs from eval_2_across_medications_comparison_ensemble.py.

Usage:
    python vis_eval_2_med_regimens_ALL_TASKS_bar.py -m parsimonious -ds MSK_CAIA
    python vis_eval_2_med_regimens_ALL_TASKS_bar.py -m parsimonious -ds MSK_CAIA --combo
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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    prog='vis_eval_2_med_regimens_ALL_TASKS_bar',
    description='Bar chart visualization of SPARC-P across medication regimens for all tasks')
parser.add_argument("-m", "--modules", type=str, default='parsimonious',
                    help='Module category')
parser.add_argument("-ds", "--dataset", type=str, default='MSK_CAIA',
                    help='Dataset: MSK_CAIA, MSK_CAIA_timeshift')
parser.add_argument("--min_num_samples", type=int, default=50,
                    help='Minimum number of patients to include regimen (default: 50)')
parser.add_argument("--min_num_pos", type=int, default=10,
                    help='Minimum number of positive cases to include regimen (default: 10)')
parser.add_argument("--metric", type=str, default='auto',
                    help='Metric to visualize. "auto" (default) uses auroc_nextinf_weighted for side effects')
parser.add_argument("--combo", action='store_true',
                    help='Use combo style: bar for "all", forest plot for specific regimens')
parser.add_argument("--fn", action='store_true',
                    help='Exclude febrile neutropenia (results in 2x2 layout with 4 subplots)')

parser.add_argument('--save_dir', type=str, default=None,

                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without touching the published ones.')

args = parser.parse_args()
modules = args.modules
dataset = args.dataset
min_num_samples = args.min_num_samples
min_num_pos = args.min_num_pos
metric_choice = args.metric
combo_style = args.combo
exclude_fn = args.fn

BASE_DIR = sparc_paths.SPARC_MODEL_ROOT + '/results_vis_060726/eval_across_meds'
OUTPUT_DIR = args.save_dir if args.save_dir else os.path.join(BASE_DIR, 'combined_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define task categories
if exclude_fn:
    HEME_TASKS = ['anemia', 'neutropenia']
else:
    HEME_TASKS = ['anemia', 'neutropenia', 'febrile_neutropenia']
AKI_TASKS = ['AKI']
LT_TASKS = ['Liver_toxicity']
ALL_TASKS = HEME_TASKS + AKI_TASKS + LT_TASKS

TASK_DISPLAY = {
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
    'febrile_neutropenia': 'Febrile neutropenia',
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
}

# Load data for all tasks
task_data = {}
task_ensemble_data = {}

for task in ALL_TASKS:
    # Try ensemble CSV first
    ensemble_csv_path = os.path.join(
        BASE_DIR,
        f'model_performance_dict_across_meds_{modules}_{task}_{dataset}_regimen_060726_ensemble.csv'
    )
    regular_csv_path = os.path.join(
        BASE_DIR,
        f'model_performance_dict_across_meds_{modules}_{task}_{dataset}_regimen_060726.csv'
    )

    # Load ensemble data if available
    if os.path.exists(ensemble_csv_path):
        df_ensemble = pd.read_csv(ensemble_csv_path)
        df_ensemble_sparc = df_ensemble[(df_ensemble['model'] == 'SPARC-P_ensemble') &
                                         (df_ensemble['val_fold'] == 'ensemble')].copy()
        if not df_ensemble_sparc.empty:
            task_ensemble_data[task] = df_ensemble_sparc
            print(f"Loaded ensemble data for {task}: {len(df_ensemble_sparc)} rows")

    # Load regular fold data
    csv_path = ensemble_csv_path if os.path.exists(ensemble_csv_path) else regular_csv_path

    if not os.path.exists(csv_path):
        print(f"WARNING: File not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    # Filter to fold-level data only (exclude ensemble rows)
    df = df[(df['model'] == 'SPARC-P') & (df['val_fold'] != 'ensemble')].copy()

    if df.empty:
        print(f"WARNING: No SPARC-P fold data for {task}")
        continue

    task_data[task] = df
    print(f"Loaded {task} fold data: {len(df)} rows")

if len(task_data) == 0:
    print("ERROR: No task data found. Run eval_2_across_medications_comparison_060926_ensemble.py first.")
    exit(1)

print(f"\nLoaded {len(task_data)} tasks: {list(task_data.keys())}")
if task_ensemble_data:
    print(f"Ensemble data available for {len(task_ensemble_data)} tasks: {list(task_ensemble_data.keys())}")

# Determine metric for each task
def get_metric_for_task(task):
    """Use weighted AUROC for all side-effect tasks"""
    if metric_choice != 'auto':
        return metric_choice
    return 'auroc_nextinf_weighted'

# Compute statistics for each task and regimen
all_stats = {}  # {task: {regimen: {metric_mean, metric_std, n_patients, n_positives}}}
all_ensemble_stats = {}  # {task: {regimen: ensemble_metric_value}}

for task, df in task_data.items():
    metric = get_metric_for_task(task)
    all_stats[task] = {}

    for regimen, grp in df.groupby('med_sublist'):
        n_pat = int(grp['unique_num_patients'].dropna().iloc[0]) if not grp['unique_num_patients'].dropna().empty else 0
        n_pos = int(grp['unique_positives'].dropna().iloc[0]) if not grp['unique_positives'].dropna().empty else 0

        vals = grp[metric].dropna()
        mean_val = vals.mean() if len(vals) > 0 else np.nan
        std_val = vals.std(ddof=1) if len(vals) > 1 else np.nan

        all_stats[task][regimen] = {
            'mean': mean_val,
            'std': std_val,
            'n_patients': n_pat,
            'n_positives': n_pos,
        }

# Load ensemble statistics
for task, df_ens in task_ensemble_data.items():
    metric = get_metric_for_task(task)
    all_ensemble_stats[task] = {}

    for regimen, grp in df_ens.groupby('med_sublist'):
        vals = grp[metric].dropna()
        if len(vals) > 0:
            all_ensemble_stats[task][regimen] = vals.iloc[0]

# Get regimens relevant to each category (filter by which tasks they appear in)
def get_regimens_for_tasks(tasks, all_stats):
    """Get regimens that meet threshold in at least one of the given tasks"""
    regimens_in_category = set()
    for task in tasks:
        if task not in all_stats:
            continue
        for regimen, stats in all_stats[task].items():
            n_pat = stats['n_patients']
            n_pos = stats['n_positives']
            # Include if meets threshold OR is 'all' baseline
            if regimen == 'all' or (n_pat >= min_num_samples and n_pos >= min_num_pos):
                regimens_in_category.add(regimen)

    # Order: 'all' first, then by avg patient count across these tasks
    def get_avg_count(reg):
        if reg == 'all':
            return float('inf')
        counts = []
        for task in tasks:
            if task in all_stats and reg in all_stats[task]:
                counts.append(all_stats[task][reg]['n_patients'])
        return np.mean(counts) if counts else 0.0

    return sorted(regimens_in_category, key=get_avg_count, reverse=True)

heme_regimens = get_regimens_for_tasks([t for t in HEME_TASKS if t in task_data], all_stats)
aki_regimens = get_regimens_for_tasks([t for t in AKI_TASKS if t in task_data], all_stats)
lt_regimens = get_regimens_for_tasks([t for t in LT_TASKS if t in task_data], all_stats)

print(f"\nRegimens by category:")
print(f"  Hematological: {len(heme_regimens)} regimens")
print(f"  AKI: {len(aki_regimens)} regimens")
print(f"  Liver toxicity: {len(lt_regimens)} regimens")

# 3-PANEL BAR CHART VISUALIZATION

# Determine figure dimensions
max_regimens = max(len(heme_regimens), len(aki_regimens), len(lt_regimens))
row_height_per_regimen = 0.35
min_height = 8.0
fig_height = max(min_height, max_regimens * row_height_per_regimen)

# Create figure with GridSpec
fig = plt.figure(figsize=(16, fig_height))
heme_tasks_present = [t for t in HEME_TASKS if t in task_data]

if exclude_fn:
    # 2x2 layout: 2 heme tasks on top, AKI and LT on bottom
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.5)

    # Top panel: 2 hematological tasks (each takes 3 columns)
    heme_axes = []
    for i, task in enumerate(heme_tasks_present):
        ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
        heme_axes.append(ax)

    # Bottom panels: AKI (left, 3 columns) and LT (right, 3 columns)
    aki_ax = fig.add_subplot(gs[1, 0:3]) if 'AKI' in task_data else None
    lt_ax = fig.add_subplot(gs[1, 3:6]) if 'Liver_toxicity' in task_data else None
else:
    # Original layout: 3 heme tasks on top, AKI and LT on bottom
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.5)

    # Top panel: Hematological tasks (3 subplots, each takes 2 columns)
    heme_axes = []
    for i, task in enumerate(heme_tasks_present):
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
        heme_axes.append(ax)

    # Bottom panels: AKI (left) and LT (right)
    aki_ax = fig.add_subplot(gs[1, 0:2]) if 'AKI' in task_data else None
    lt_ax = fig.add_subplot(gs[1, 4:6]) if 'Liver_toxicity' in task_data else None

# Helper function to plot a single task
def plot_task_bar(ax, task, regimens_list, show_ylabel=True):
    """Plot bar chart for one task with its relevant regimens"""
    metric = get_metric_for_task(task)
    n_regimens = len(regimens_list)
    y_positions = np.arange(n_regimens)

    # Create y-labels
    y_labels = []
    for regimen in regimens_list:
        if regimen == 'all':
            y_labels.append('All regimens')
        else:
            # Truncate long regimen names
            display_regimen = regimen if len(regimen) <= 30 else regimen[:27] + '...'
            y_labels.append(display_regimen)

    # Create color palette for specific regimens (excluding 'all')
    specific_regimens = [r for r in regimens_list if r != 'all']
    n_specific = len(specific_regimens)
    if n_specific > 0:
        # Use a nice qualitative colormap
        colors = cm.get_cmap('Set2')(np.linspace(0, 1, max(8, n_specific)))
        regimen_colors = {reg: colors[i % len(colors)] for i, reg in enumerate(specific_regimens)}
    else:
        regimen_colors = {}

    # Plot each regimen
    for i, regimen in enumerate(regimens_list):
        if regimen not in all_stats[task]:
            continue

        stats = all_stats[task][regimen]
        mean = stats['mean']
        std = stats['std']

        if np.isnan(mean):
            continue

        xerr = std * 100 if (not np.isnan(std) and std > 0) else None
        is_all = (regimen == 'all')

        # Combo style: bar for 'all', forest plot for specific regimens
        if combo_style:
            if is_all:
                # Bar chart for 'all' baseline
                ax.barh(
                    i,
                    mean * 100,
                    height=0.6,
                    xerr=xerr,
                    facecolor='#3B4992',
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.9,
                    error_kw={'elinewidth': 1.0, 'capsize': 2.5, 'capthick': 1.0}
                )
            else:
                # Forest plot style for specific regimens
                ax.errorbar(
                    mean * 100,
                    i,
                    xerr=xerr,
                    fmt='o',
                    color='#AAAAAA',
                    ecolor='#AAAAAA',
                    markersize=6,
                    capsize=3,
                    linewidth=1.2,
                )
        else:
            # Standard style: colored bars with transparency
            if is_all:
                color = '#3B4992'
                alpha = 0.9
            else:
                color = '#AAAAAA'  # Shallow grey for specific regimens
                alpha = 0.7  # Slightly higher alpha for better visibility

            ax.barh(
                i,
                mean * 100,
                height=0.6,
                xerr=xerr,
                facecolor=color,
                edgecolor='black',
                linewidth=0.8,
                alpha=alpha,
                error_kw={'elinewidth': 1.0, 'capsize': 2.5, 'capthick': 1.0}
            )

    # Add ensemble performance as red diamond markers
    if task in all_ensemble_stats:
        for i, regimen in enumerate(regimens_list):
            if regimen in all_ensemble_stats[task]:
                ensemble_val = all_ensemble_stats[task][regimen]
                if not np.isnan(ensemble_val):
                    # Draw a red diamond marker at the ensemble value
                    ax.plot(ensemble_val * 100, i, marker='D', markersize=6,
                           color='red', markeredgecolor='darkred',
                           markeredgewidth=1.2, zorder=10)

    # Formatting
    ax.axvline(50, linestyle=':', color='gray', alpha=0.3, linewidth=1)
    ax.set_xlabel('Weighted AUROC (%)', fontsize=10)
    ax.set_xlim(40, 100)
    ax.set_ylim(-0.5, n_regimens - 0.5)
    ax.invert_yaxis()
    ax.set_title(TASK_DISPLAY.get(task, task), fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

    # Y-axis labels
    if show_ylabel:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.tick_params(axis='y', which='both', length=4)
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)

    # Add horizontal line after 'all' row
    if 'all' in regimens_list:
        ax.axhline(0.5, color='gray', linewidth=1, linestyle='-', alpha=0.5)

# Plot hematological tasks
for i, (task, ax) in enumerate(zip(heme_tasks_present, heme_axes)):
    show_ylabel = (i == 0)  # Only leftmost shows labels
    plot_task_bar(ax, task, heme_regimens, show_ylabel=show_ylabel)

# Plot AKI
if aki_ax is not None and 'AKI' in task_data:
    plot_task_bar(aki_ax, 'AKI', aki_regimens, show_ylabel=True)

# Plot Liver toxicity
if lt_ax is not None and 'Liver_toxicity' in task_data:
    # In 2x2 layout, LT is on the right so don't show ylabel
    show_lt_ylabel = not exclude_fn
    plot_task_bar(lt_ax, 'Liver_toxicity', lt_regimens, show_ylabel=show_lt_ylabel)

# Add legend
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

if combo_style:
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#3B4992', edgecolor='black', alpha=0.9, label='All regimens'),
        Line2D([0], [0], marker='o', color='#AAAAAA', linestyle='-', markersize=6, label='Specific regimens'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=6, markeredgewidth=1.2,
               label='Ensemble', linestyle='')
    ]
else:
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#3B4992', edgecolor='black', alpha=0.9, label='All regimens'),
        Rectangle((0, 0), 1, 1, facecolor='#AAAAAA', edgecolor='black', alpha=0.7, label='Specific regimens'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=6, markeredgewidth=1.2,
               label='Ensemble', linestyle='')
    ]

plt.suptitle(f'SPARC-P Performance Across Medication Regimens\n'
            f'{dataset.replace("_", "-")} | {modules} | min_samples≥{min_num_samples}, min_pos≥{min_num_pos}',
            fontsize=13, y=0.98)

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96),
          ncol=3, fontsize=10, frameon=True, fancybox=True)

# Save figure
combo_suffix = '_combo' if combo_style else ''
fn_suffix = '_noFN' if exclude_fn else ''
out_pdf = os.path.join(OUTPUT_DIR,
                       f'bar_regimens_all_tasks_{modules}_{dataset}_minS{min_num_samples}_minP{min_num_pos}{combo_suffix}{fn_suffix}.pdf')
out_png = os.path.join(OUTPUT_DIR,
                       f'bar_regimens_all_tasks_{modules}_{dataset}_minS{min_num_samples}_minP{min_num_pos}{combo_suffix}{fn_suffix}.png')

fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved bar chart to:")
print(f"  {out_pdf}")
print(f"  {out_png}")
plt.close()

print("\nDone!")
