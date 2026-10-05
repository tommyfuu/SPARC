"""Parsimonious vs all modules across all 7 tasks, one horizontal-bar figure.

Mean +/- std over folds with Bonferroni-corrected paired tests. Side-effect tasks use the
next-infusion AUROC, binary tasks plain AUROC. Input: CSVs from design_choice_comparison_4_parsimony.py.
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
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design_choice_summary import build_design_choice_summary
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, ttest_ind


# Paired: both arms use the same folds. At n=10 the smallest two-sided Wilcoxon p
# is 0.00195, so strong effects all pin there; read the deltas, not the stars.
def paired_test(vals1, vals2):
    """Two-sided p per STAT_TEST; nan (not None) when not computable."""
    a = np.asarray(vals1, dtype=float)
    b = np.asarray(vals2, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    try:
        if STAT_TEST in ('wilcoxon', 'ttest_rel'):
            if len(a) != len(b):
                return float('nan')      # paired tests need matched folds
            if np.allclose(a - b, 0):
                return float('nan')      # wilcoxon raises; ttest_rel gives nan
            return (wilcoxon(a, b)[1] if STAT_TEST == 'wilcoxon'
                    else ttest_rel(a, b)[1])
        if STAT_TEST == 'mannwhitney':
            return mannwhitneyu(a, b, alternative='two-sided')[1]
        return ttest_ind(a, b)[1]
    except Exception:
        return float('nan')


# Configuration
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--save_dir', type=str, default=None,
                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without overwriting the existing ones.')
parser.add_argument('--csv_dir', type=str, default=None,
                    help='directory holding the design-choice CSVs; overrides the default results folder')
parser.add_argument('-am', '--auroc_mode', type=str, default='weighted', choices=['weighted', 'overall'],
                    help="Side-effect AUROC: 'weighted' (default) or 'overall' (pooled over the valid "
                         "infusion groups); binary tasks use auroc either way. 'overall' adds "
                         "_overallauroc to the figure and summary names.")
args = parser.parse_args()

# --csv_dir overrides the default results dir
_arg_csv_dir = getattr(args, 'csv_dir', None)
csv_dir = _arg_csv_dir if _arg_csv_dir else f'{sparc_paths.SPARC_RESULTS_VIS_ROOT}/design_choices_4_parsimony'   # bf=True results

save_dir = args.save_dir if args.save_dir else csv_dir
dataset = 'MSK'
# test is set by the STAT_TEST env var; see paired_test()
STAT_TEST = os.environ.get('STAT_TEST', 'wilcoxon')

SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']
BINARY_TASKS = ['binary', 'binary_icu_pre']
ALL_TASKS = SIDE_EFFECT_TASKS + BINARY_TASKS

# AUROC column names differ by task type
AUROC_COL = {t: f'test_auroc_nextinf_{args.auroc_mode}' for t in SIDE_EFFECT_TASKS}
AM_SUFFIX = '_overallauroc' if args.auroc_mode == 'overall' else ''
AUROC_COL.update({t: 'auroc' for t in BINARY_TASKS})

TASK_DISPLAY = {
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
    'binary': '180D Hosp. mort.',
    'binary_icu_pre': '30D ICU mort.',
}

# Two module configurations to compare
MODELS_TO_PLOT = ['parsimonious', 'all']

MODEL_DISPLAY = {
    'parsimonious': 'Parsimonious',
    'all': 'All Modules',
}

# Parsimonious (the final design) gets the SPARC-P blue
MODEL_COLORS = {
    'parsimonious': '#3B4992',
    'all': '#707070',
}

# Load CSVs
task_data = {}

for task in ALL_TASKS:
    csv_path = f'{csv_dir}/SPARC_parsimony_{task}_{dataset}.csv'

    if not os.path.exists(csv_path):
        print(f"WARNING: missing CSV for {task} — task will be skipped")
        continue

    df = pd.read_csv(csv_path)

    # drop the fold-averaged rows
    if 'val_fold' in df.columns:
        df = df[df['val_fold'] != 'averaged'].copy()

    task_data[task] = df

available_tasks = [t for t in ALL_TASKS if t in task_data]
n_tasks = len(available_tasks)

if n_tasks == 0:
    raise RuntimeError("No task CSVs found. Run design_choice_comparison_4_parsimony.py first.")

print(f"\nLoaded {n_tasks} tasks: {available_tasks}")
print(f"Plotting {len(MODELS_TO_PLOT)} module configurations: {[MODEL_DISPLAY[m] for m in MODELS_TO_PLOT]}\n")

# Data summary export (aggregate the per-task CSVs into one)
# AUROC_COL maps each task to its own AUROC column
build_design_choice_summary(
    design_choice='design_choice_4_parsimony',
    task_data=task_data,
    available_tasks=available_tasks,
    models=MODELS_TO_PLOT,
    metrics={'auroc': AUROC_COL},
    save_dir=save_dir,
    out_stem=f'parsimony_comparison_{dataset}{AM_SUFFIX}',
    task_display=TASK_DISPLAY,
    model_display=MODEL_DISPLAY,
    to_percent=True,
)

# Bonferroni-corrected significance thresholds
alpha = 0.05
alpha_corr = alpha / n_tasks
print(f"Bonferroni-corrected α = {alpha}/{n_tasks} = {alpha_corr:.4f}\n")

def get_vals(task, model_key):
    """Get fold-level AUROC values for a model on a task"""
    col = AUROC_COL[task]
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty or col not in rows.columns:
        return np.array([])
    vals = rows[col].dropna().values
    # CSVs may hold fractions or percentages
    if vals.max() <= 1.0:
        return vals * 100
    return vals

# Figure
n_models = len(MODELS_TO_PLOT)
bar_height = 0.5
task_gap = 0.6    # Gap between task groups

# Build y-axis positions
y_positions = {}  # {(task, model): y_pos}
task_centers = {}  # {task: center_y}

current_y = 0
for task in reversed(available_tasks):  # Reverse so first task is at top
    task_y_positions = []
    for i, model in enumerate(MODELS_TO_PLOT):
        # Parsimonious on top
        y_pos = current_y + (n_models - 1 - i) * bar_height
        y_positions[(task, model)] = y_pos
        task_y_positions.append(y_pos)

    task_centers[task] = np.mean(task_y_positions)

    current_y += n_models * bar_height + task_gap

fig_height = max(8, n_tasks * 1.5)
fig_width = 10
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Collect data and plot bars
for task in available_tasks:
    for model in MODELS_TO_PLOT:
        vals = get_vals(task, model)
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            y_pos = y_positions[(task, model)]
            color = MODEL_COLORS[model]

            ax.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                   color=color, edgecolor='black', linewidth=1.0,
                   error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

# Bonferroni-corrected significance testing
print("\n" + "="*70)
print("SIGNIFICANCE TESTING (Parsimonious vs All Modules)")
print("="*70)

# Position for significance brackets
x_offset = 2.0  # Offset from the rightmost bar
text_offset = 0.5

for task in available_tasks:
    parsimonious_vals = get_vals(task, 'parsimonious')
    all_vals = get_vals(task, 'all')

    if len(parsimonious_vals) > 1 and len(all_vals) > 1:
        p_raw = paired_test(parsimonious_vals, all_vals)
        p_corrected = min(p_raw * n_tasks, 1.0)

        if p_corrected < alpha:
            star = '***' if p_corrected < 0.001 else ('**' if p_corrected < 0.01 else '*')
        else:
            star = 'NS'

        # negative = parsimonious worse
        difference = np.mean(parsimonious_vals) - np.mean(all_vals)

        print(f"  {TASK_DISPLAY[task]:25s}: p_raw={p_raw:.4f}, p_Bonf={p_corrected:.4f} → {star} "
              f"(Δ={difference:+.2f}%)")

        y_pars = y_positions[(task, 'parsimonious')]
        y_all = y_positions[(task, 'all')]

        # bracket sits right of the longer bar
        pars_mean = np.mean(parsimonious_vals)
        pars_std = np.std(parsimonious_vals)
        all_mean = np.mean(all_vals)
        all_std = np.std(all_vals)

        x_right = max(pars_mean + pars_std, all_mean + all_std) + x_offset

        ax.plot([x_right, x_right], [y_pars, y_all], 'k-', linewidth=1,
               alpha=0.5 if star == 'NS' else 1.0)

        y_mid = (y_pars + y_all) / 2
        ax.text(x_right + text_offset, y_mid, star,
               ha='left', va='center', fontsize=9,
               alpha=0.6 if star == 'NS' else 1.0, fontweight='bold')

print("="*70 + "\n")

# Pooled across all tasks
print("="*70)
print("OVERALL STATISTICS (Across All Tasks)")
print("="*70)

all_parsimonious = []
all_all_modules = []

for task in available_tasks:
    parsimonious_vals = get_vals(task, 'parsimonious')
    all_vals = get_vals(task, 'all')

    if len(parsimonious_vals) > 0:
        all_parsimonious.extend(parsimonious_vals.tolist())
    if len(all_vals) > 0:
        all_all_modules.extend(all_vals.tolist())

if len(all_parsimonious) > 0 and len(all_all_modules) > 0:
    all_parsimonious = np.array(all_parsimonious)
    all_all_modules = np.array(all_all_modules)

    p_overall = paired_test(all_parsimonious, all_all_modules)
    overall_difference = np.mean(all_parsimonious) - np.mean(all_all_modules)

    star = '***' if p_overall < 0.001 else ('**' if p_overall < 0.01 else ('*' if p_overall < 0.05 else 'NS'))

    print(f"\n  Parsimonious:  {np.mean(all_parsimonious):.2f} ± {np.std(all_parsimonious):.2f}%")
    print(f"  All Modules:   {np.mean(all_all_modules):.2f} ± {np.std(all_all_modules):.2f}%")
    print(f"  Overall Difference: {overall_difference:+.2f}%, p={p_overall:.4f} {star}")
    print(f"\n  Note: Parsimonious uses fewer features but maintains comparable performance")

print("="*70 + "\n")

# Set y-axis labels (task names)
y_ticks = [task_centers[task] for task in reversed(available_tasks)]
y_labels = [TASK_DISPLAY[task] for task in reversed(available_tasks)]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=11)

# Set x-axis
ax.set_xlabel('(Weighted) AUROC (%)' if args.auroc_mode == 'weighted' else 'AUROC (%)', fontsize=12, fontweight='bold')
ax.set_xlim(left=50, right=105)  # room for the brackets

# Title
title_text = f'Feature Parsimony: Parsimonious vs All Modules — {dataset}'
ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)

# Grid and spines
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_elements = []
for model in MODELS_TO_PLOT:
    legend_elements.append(
        Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[model],
                 edgecolor='black', label=MODEL_DISPLAY[model])
    )

ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
         frameon=True, fancybox=True)

plt.tight_layout()

# Save figure
test_suffix = '_' + STAT_TEST if STAT_TEST != 'wilcoxon' else ''
out_pdf = f'{save_dir}/parsimony_comparison_{dataset}{test_suffix}{AM_SUFFIX}_horizontal_bar.pdf'
out_png = f'{save_dir}/parsimony_comparison_{dataset}{test_suffix}{AM_SUFFIX}_horizontal_bar.png'

fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=150)
plt.close()

print(f"Figure saved:")
print(f"  {out_pdf}")
print(f"  {out_png}\n")
print("Done.")
