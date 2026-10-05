"""Meds-as-features on/off across the 5 side-effect tasks.

Weighted AUROC and weighted C-index figures: mean +/- std over folds, Bonferroni-corrected
paired tests. Input: CSVs from design_choice_comparison_3_medsasfeatures.py.
Usage: python vis_design_choice_3_medsasfeature_bar_horizontal.py [--dataset MSK_CAIA_timeshift]
"""

# put the repo root, data/ and models/ on sys.path
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
import argparse
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


# Argument parsing
parser = argparse.ArgumentParser(description='Succinct horizontal bar chart: medications as features comparison')
parser.add_argument('--dataset', type=str, default='MSK',
                    help='Dataset name (default: MSK)')
parser.add_argument('--modules', type=str, default='parsimonious',
                    help='Module category (default: parsimonious)')
parser.add_argument('--stat_test', type=str, default='wilcoxon',
                    choices=['wilcoxon', 'ttest_rel', 'mannwhitney', 'ttest_ind'],
                    help="Significance test. 'wilcoxon' (default) and 'ttest_rel' are PAIRED, "
                         "matching the design: both arms use the same folds. 'mannwhitney' and "
                         "'ttest_ind' are the UNPAIRED tests this script used previously, kept "
                         "to reproduce the old figures. A non-default choice is appended to the "
                         "output filename.")
parser.add_argument('--save_dir', type=str, default=None,
                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without overwriting the existing ones.')
parser.add_argument('--csv_dir', type=str, default=None,
                    help='directory holding the design-choice CSVs; overrides the default results folder')
parser.add_argument('-am', '--auroc_mode', type=str, default='weighted', choices=['weighted', 'overall'],
                    help="Side-effect AUROC: 'weighted' (default; per-infusion-group AUROC weighted by "
                         "group size) or 'overall' (one AUROC pooled over the valid groups). 'overall' "
                         "draws only the AUROC figure, with _overallauroc in its filename and summary.")
args = parser.parse_args()
STAT_TEST = args.stat_test

dataset = args.dataset
modules = args.modules

# Configuration
# --csv_dir overrides the default results dir
_arg_csv_dir = getattr(args, 'csv_dir', None)
csv_dir = _arg_csv_dir if _arg_csv_dir else f'{sparc_paths.SPARC_RESULTS_VIS_ROOT}/design_choices_3_medsasfeatures'   # bf=True results
save_dir = args.save_dir if args.save_dir else csv_dir
os.makedirs(save_dir, exist_ok=True)

# side-effect tasks
SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']

TASK_DISPLAY = {
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
}

# models to compare
FEATURE_MODELS = ['med_as_feature_False', 'med_as_feature_True']

FEATURE_DISPLAY = {
    'med_as_feature_False': 'Meds NOT as Features',
    'med_as_feature_True': 'Meds as Features'
}

# meds-as-features highlighted
FEATURE_COLORS = {
    'med_as_feature_False': '#AAAAAA',    # Gray
    'med_as_feature_True': '#3B4992',     # Royal blue (highlighted)
}

# Load CSVs
task_data = {}

for task in SIDE_EFFECT_TASKS:
    csv_path = f'{csv_dir}/SPARC_medsasfeatures_{modules}_{task}_{dataset}.csv'

    if not os.path.exists(csv_path):
        print(f"WARNING: missing CSV for {task}: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    task_data[task] = df

available_tasks = [t for t in SIDE_EFFECT_TASKS if t in task_data]
n_tasks = len(available_tasks)

if n_tasks == 0:
    raise RuntimeError("No task CSVs found. Run design_choice_comparison_3_medsasfeatures.py first.")

print(f"\nLoaded {n_tasks} tasks: {available_tasks}")
print(f"Plotting {len(FEATURE_MODELS)} feature options: {[FEATURE_DISPLAY[m] for m in FEATURE_MODELS]}\n")

# Data summary export (aggregate the per-task CSVs into one)
build_design_choice_summary(
    design_choice='design_choice_3_medsasfeature',
    task_data=task_data,
    available_tasks=available_tasks,
    models=FEATURE_MODELS,
    metrics=({
        'weighted_auroc': 'test_auroc_nextinf_weighted',
        'weighted_cindex': 'test_cindex_weighted',
    } if args.auroc_mode == 'weighted' else {'overall_auroc': 'test_auroc_nextinf_overall'}),
    save_dir=save_dir,
    out_stem=f'medsasfeatures_comparison_{modules}_{dataset}' + ('_overallauroc' if args.auroc_mode == 'overall' else ''),
    task_display=TASK_DISPLAY,
    model_display=FEATURE_DISPLAY,
    to_percent=True,
)

# Bonferroni-corrected significance thresholds
alpha = 0.05
alpha_corr = alpha / n_tasks
print(f"Bonferroni-corrected α = {alpha}/{n_tasks} = {alpha_corr:.4f}\n")

def get_vals(task, model_key, metric_col):
    """Get fold-level metric values for a model on a task"""
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty or metric_col not in rows.columns:
        return np.array([])
    return rows[metric_col].dropna().values * 100   # → percent

def create_horizontal_barchart(metric_col, metric_name, metric_suffix):
    """Grouped horizontal bars for one metric, with significance brackets."""

    n_models = len(FEATURE_MODELS)
    bar_height = 0.45
    task_gap = 0.2     # Minimal gap between task groups

    y_positions = {}  # {(task, model): y_pos}
    task_centers = {}  # {task: center_y}

    current_y = 0
    for task in reversed(available_tasks):  # Reverse so first task is at top
        task_y_positions = []
        for i, model in enumerate(FEATURE_MODELS):
            # first model (NOT as features) on top
            y_pos = current_y + (n_models - 1 - i) * bar_height
            y_positions[(task, model)] = y_pos
            task_y_positions.append(y_pos)

        task_centers[task] = np.mean(task_y_positions)

        current_y += n_models * bar_height + task_gap

    fig_height = max(6, n_tasks * 1.2)  # Compact vertical layout
    fig_width = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for task in available_tasks:
        for model in FEATURE_MODELS:
            vals = get_vals(task, model, metric_col)
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                y_pos = y_positions[(task, model)]
                color = FEATURE_COLORS[model]

                ax.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                       color=color, edgecolor='black', linewidth=1.2,
                       error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

    # Bonferroni-corrected significance testing
    print("\n" + "="*70)
    print(f"SIGNIFICANCE TESTING - {metric_name}")
    print("="*70)

    # Two-way comparison: NOT as Features vs as Features
    for task in available_tasks:
        vals_not = get_vals(task, 'med_as_feature_False', metric_col)
        vals_yes = get_vals(task, 'med_as_feature_True', metric_col)

        if len(vals_not) > 1 and len(vals_yes) > 1:
            p_raw = paired_test(vals_not, vals_yes)
            p_corrected = min(p_raw * n_tasks, 1.0)

            if p_corrected < alpha:
                star = '***' if p_corrected < 0.001 else ('**' if p_corrected < 0.01 else '*')
            else:
                star = 'NS'

            print(f"  {TASK_DISPLAY[task]:25s} NOT vs YES: "
                  f"p_raw={p_raw:.4f}, p_Bonf={p_corrected:.4f} → {star}")

            y1 = y_positions[(task, 'med_as_feature_False')]
            y2 = y_positions[(task, 'med_as_feature_True')]

            # bracket sits right of the longer bar
            mean1 = np.mean(vals_not)
            std1 = np.std(vals_not)
            mean2 = np.mean(vals_yes)
            std2 = np.std(vals_yes)

            x_right = max(mean1 + std1, mean2 + std2) + 1.5

            ax.plot([x_right, x_right], [y1, y2], 'k-', linewidth=1,
                   alpha=0.5 if star == 'NS' else 1.0)

            y_mid = (y1 + y2) / 2
            ax.text(x_right + 0.5, y_mid, star,
                   ha='left', va='center', fontsize=9,
                   alpha=0.6 if star == 'NS' else 1.0)

    print("="*70 + "\n")

    y_ticks = [task_centers[task] for task in reversed(available_tasks)]
    y_labels = [TASK_DISPLAY[task] for task in reversed(available_tasks)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)

    ax.set_xlabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(left=50, right=105)

    dataset_display = dataset.replace('_', '-')
    title_text = f'Medications as Features Comparison — {metric_name} — {dataset_display}'
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)

    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=FEATURE_COLORS['med_as_feature_False'],
                 edgecolor='black', label=FEATURE_DISPLAY['med_as_feature_False'],
                 linewidth=1.2),
        Rectangle((0, 0), 1, 1, facecolor=FEATURE_COLORS['med_as_feature_True'],
                 edgecolor='black', label=FEATURE_DISPLAY['med_as_feature_True'],
                 linewidth=1.2),
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
             frameon=True, fancybox=True)

    plt.tight_layout()

    test_suffix = '_' + STAT_TEST if STAT_TEST != 'wilcoxon' else ''
    out_pdf = f'{save_dir}/medsasfeatures_comparison_{metric_suffix}_{modules}_{dataset}{test_suffix}_bar_horizontal.pdf'
    out_png = f'{save_dir}/medsasfeatures_comparison_{metric_suffix}_{modules}_{dataset}{test_suffix}_bar_horizontal.png'

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFigure saved:")
    print(f"  {out_pdf}")
    print(f"  {out_png}\n")

# Create both figures (AUROC and C-index)

if args.auroc_mode == 'overall':
    create_horizontal_barchart('test_auroc_nextinf_overall', 'Overall AUROC', 'overallauroc')
else:
    create_horizontal_barchart('test_auroc_nextinf_weighted', 'Weighted AUROC', 'auroc')

    create_horizontal_barchart('test_cindex_weighted', 'Weighted C-index', 'cindex')

print("Done.")
