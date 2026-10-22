"""SPARC-P performance across cancer types, one panel per task, from eval_3 CSVs.

Bars are mean +/- std over folds, red dashed line = ensemble. --combo draws 'all' as a bar and
the cancer types as a forest plot; --max_ctypes caps how many are shown.
-ci ensemble: bars = ensemble with 95% bootstrap CI (from the eval's -ci bootstrap CSVs), 'all' in
colour and cancer types grey; files get _ensembleCI.
Usage: python vis_eval_3_cancer_types_ALL_TASKS_bar.py -m parsimonious -ds MSK_CAIA [--combo | -ci ensemble]
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
import os as _os
_CSV_DIR = _os.environ.get('SPARC_VIS_CSV_DIR')  # optional override of the CSV dir
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import argparse
import pickle
from collections import Counter

parser = argparse.ArgumentParser(
    prog='vis_eval_3_cancer_types_ALL_TASKS_bar',
    description='Bar chart visualization of SPARC-P across cancer types for all tasks')
parser.add_argument("-m", "--modules", type=str, default='parsimonious',
                    help='Module category')
parser.add_argument("-ds", "--dataset", type=str, default='MSK_CAIA',
                    help='Dataset: MSK_CAIA, MSK_CAIA_timeshift')
parser.add_argument("--min_num_samples", type=int, default=50,
                    help='Minimum number of patients to include cancer type (default: 50)')
parser.add_argument("--min_num_pos", type=int, default=10,
                    help='Minimum number of positive cases to include cancer type (default: 10)')
parser.add_argument("--max_ctypes", type=int, default=25,
                    help='Maximum number of cancer types to display per task (default: 25)')
parser.add_argument("--metric", type=str, default='auto',
                    help='Metric to visualize. "auto" (default) uses auroc_nextinf_weighted for side effects, auroc for binary')
parser.add_argument("-am", "--auroc_mode", type=str, default='weighted', choices=['weighted', 'overall'],
                    help="Side-effect AUROC used when --metric auto: 'weighted' (default) or 'overall' "
                         "(pooled over the valid infusion groups). 'overall' adds _overallauroc to the "
                         "output filenames so the weighted figures are not overwritten.")
parser.add_argument("--combo", action='store_true',
                    help='Use combo style: bar for "all", forest plot for specific cancer types')

parser.add_argument("-ci", "--ci", type=str, default='folds', choices=['folds', 'ensemble'],
                    help="folds: fold mean +/- SD bars with the ensemble marked (default). ensemble: bars = "
                         "10-fold ensemble, error bars = its 95%% bootstrap CI (eval with -ci bootstrap), "
                         "'all' coloured and every subgroup grey; adds _ensembleCI to the filenames.")
parser.add_argument('--save_dir', type=str, default=None,

                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without overwriting the existing ones.')

args = parser.parse_args()
modules = args.modules
dataset = args.dataset
min_num_samples = args.min_num_samples
min_num_pos = args.min_num_pos
max_ctypes = args.max_ctypes
metric_choice = args.metric
combo_style = args.combo
ci_ensemble = args.ci == 'ensemble'
_ci_csv = 'CI' if ci_ensemble else ''

BASE_DIR = _CSV_DIR or (sparc_paths.SPARC_RESULTS_VIS_ROOT + '/eval_across_ctypes')
OUTPUT_DIR = args.save_dir if args.save_dir else os.path.join(BASE_DIR, 'combined_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patient counts per cancer type, from the dataset
data_source_path = sparc_paths.DATA_ROOT + '/'
ctype_pickle_path = os.path.join(
    data_source_path, dataset,
    f'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_{dataset}_till2021_050726.pickle'
)

print(f"Loading cancer type data from: {ctype_pickle_path}")
with open(ctype_pickle_path, 'rb') as f:
    ctype_dict = pickle.load(f)

def get_first_cancer_type(dmp_id, ctype_dict):
    """Cancer type at the patient's earliest date (first listed if several)."""
    if dmp_id not in ctype_dict:
        return None
    dates = sorted(ctype_dict[dmp_id].keys())
    if not dates:
        return None
    earliest_date = dates[0]
    ctypes = ctype_dict[dmp_id][earliest_date]
    return ctypes[0] if ctypes else None

ctype_patient_counts = Counter()
for dmp_id in ctype_dict.keys():
    first_ctype = get_first_cancer_type(dmp_id, ctype_dict)
    if first_ctype:
        ctype_patient_counts[first_ctype] += 1

print(f"Loaded cancer type data for {len(ctype_dict)} patients")
print(f"Found {len(ctype_patient_counts)} unique cancer types")

SIDE_EFFECT_TASKS = ['anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', 'Liver_toxicity']
BINARY_TASKS = ['binary', 'binary_icu_pre']
ALL_TASKS = SIDE_EFFECT_TASKS + BINARY_TASKS

TASK_DISPLAY = {
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
    'febrile_neutropenia': 'Febrile neutropenia',
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'binary': '180D Hosp. mort.',
    'binary_icu_pre': '30D ICU mort.',
}

# Load per-task CSVs
task_data = {}
task_ensemble_data = {}

for task in ALL_TASKS:
    is_side_effect = task in SIDE_EFFECT_TASKS
    suffix = '_risk' if is_side_effect else ''

    # prefer the ensemble CSV (it also has the fold rows)
    ensemble_csv_path = os.path.join(
        BASE_DIR,
        f'model_performance_dict_across_ctypes_{modules}_{task}{suffix}_{dataset}_060726_ensemble{_ci_csv}.csv'
    )
    regular_csv_path = os.path.join(
        BASE_DIR,
        f'model_performance_dict_across_ctypes_{modules}_{task}{suffix}_{dataset}_060726.csv'
    )

    if os.path.exists(ensemble_csv_path):
        df_ensemble = pd.read_csv(ensemble_csv_path)
        df_ensemble_sparc = df_ensemble[(df_ensemble['model'] == 'SPARC-P_ensemble') &
                                         (df_ensemble['val_fold'] == 'ensemble')].copy()
        if not df_ensemble_sparc.empty:
            task_ensemble_data[task] = df_ensemble_sparc
            print(f"Loaded ensemble data for {task}: {len(df_ensemble_sparc)} rows")

    csv_path = ensemble_csv_path if os.path.exists(ensemble_csv_path) else regular_csv_path

    if not os.path.exists(csv_path):
        print(f"WARNING: File not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    # fold rows only
    df = df[(df['model'] == 'SPARC-P') & (df['val_fold'] != 'ensemble')].copy()

    if df.empty:
        print(f"WARNING: No SPARC-P fold data for {task}")
        continue

    task_data[task] = df
    print(f"Loaded {task} fold data: {len(df)} rows")

if len(task_data) == 0:
    print("ERROR: No task data found. Run eval_3_across_cancer_types_comparison_ensemble.py first.")
    exit(1)

print(f"\nLoaded {len(task_data)} tasks: {list(task_data.keys())}")
if task_ensemble_data:
    print(f"Ensemble data available for {len(task_ensemble_data)} tasks: {list(task_ensemble_data.keys())}")

# Metric per task
def get_metric_for_task(task):
    if metric_choice != 'auto':
        return metric_choice

    if task in SIDE_EFFECT_TASKS:
        return f'auroc_nextinf_{args.auroc_mode}'
    else:
        return 'auroc'

# Per task / cancer type stats
all_stats = {}  # {task: {cancer_type: {mean, std, n_patients, n_positives}}}
all_ensemble_ci = {}  # {task: {subgroup: (lo, hi)}}, -ci ensemble only
all_ensemble_stats = {}  # {task: {cancer_type: ensemble_metric_value}}

for task, df in task_data.items():
    metric = get_metric_for_task(task)
    all_stats[task] = {}

    for ctype, grp in df.groupby('cancer_type'):
        n_pat = int(grp['unique_num_patients'].dropna().iloc[0]) if not grp['unique_num_patients'].dropna().empty else 0
        n_pos = int(grp['unique_positives'].dropna().iloc[0]) if not grp['unique_positives'].dropna().empty else 0

        vals = grp[metric].dropna()
        mean_val = vals.mean() if len(vals) > 0 else np.nan
        std_val = vals.std(ddof=1) if len(vals) > 1 else np.nan

        all_stats[task][ctype] = {
            'mean': mean_val,
            'std': std_val,
            'n_patients': n_pat,
            'n_positives': n_pos,
        }

# Ensemble values
for task, df_ens in task_ensemble_data.items():
    metric = get_metric_for_task(task)
    all_ensemble_stats[task] = {}

    for ctype, grp in df_ens.groupby('cancer_type'):
        vals = grp[metric].dropna()
        if ci_ensemble:
            _cols = (('auroc_weighted_ci_lower', 'auroc_weighted_ci_upper') if metric.endswith('weighted')
                     else ('auroc_ci_lower', 'auroc_ci_upper'))
            if all(c in grp.columns for c in _cols) and grp[list(_cols)].notna().all(axis=1).any():
                _r = grp[grp[list(_cols)].notna().all(axis=1)].iloc[0]
                all_ensemble_ci.setdefault(task, {})[ctype] = (_r[_cols[0]], _r[_cols[1]])
        if len(vals) > 0:
            all_ensemble_stats[task][ctype] = vals.iloc[0]

all_cancer_types = set()
for task_stats in all_stats.values():
    all_cancer_types.update(task_stats.keys())

print(f"\nFound {len(all_cancer_types)} unique cancer types across all tasks")

# keep cancer types that pass the size thresholds in at least one task
def should_keep_ctype(ctype):
    if ctype == 'all':
        return True

    for task_stats in all_stats.values():
        if ctype not in task_stats:
            continue
        stats = task_stats[ctype]
        n_pat = stats['n_patients']
        n_pos = stats['n_positives']

        if n_pat >= min_num_samples and n_pos >= min_num_pos:
            return True

    return False

filtered_ctypes = [c for c in all_cancer_types if should_keep_ctype(c)]
print(f"After filtering: {len(filtered_ctypes)} cancer types")

# 'all' first, then by patient count (descending)
def get_patient_count(ctype):
    if ctype == 'all':
        return float('inf')
    return ctype_patient_counts.get(ctype, 0)

cancer_types_ordered = sorted(filtered_ctypes, key=get_patient_count, reverse=True)

# cap at max_ctypes, always keeping 'all'
if len(cancer_types_ordered) > max_ctypes:
    if 'all' in cancer_types_ordered:
        cancer_types_ordered = ['all'] + [c for c in cancer_types_ordered if c != 'all'][:max_ctypes-1]
    else:
        cancer_types_ordered = cancer_types_ordered[:max_ctypes]
    print(f"Limited to top {len(cancer_types_ordered)} cancer types")

print(f"\nTop cancer types by dataset prevalence:")
for ctype in cancer_types_ordered[:10]:
    count = ctype_patient_counts.get(ctype, 0) if ctype != 'all' else len(ctype_dict)
    print(f"  {ctype}: {count} patients")

# Plot

# tasks with data
tasks_present = [t for t in ALL_TASKS if t in task_data]
n_tasks = len(tasks_present)
n_ctypes = len(cancer_types_ordered)

# Figure dimensions
fig_width = 18
fig_height = max(10, n_ctypes * 0.25 * (n_tasks / 5))

fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height), sharey=False)

if n_tasks == 1:
    axes = [axes]

# y labels
y_labels = []
for ctype in cancer_types_ordered:
    if ctype == 'all':
        y_labels.append('All cancers')
    else:
        # truncate long names
        display_ctype = ctype if len(ctype) <= 40 else ctype[:37] + '...'
        y_labels.append(display_ctype)

y_positions = np.arange(n_ctypes)

# colours for the specific cancer types
specific_ctypes = [c for c in cancer_types_ordered if c != 'all']
n_specific = len(specific_ctypes)
if n_specific > 0:
    colors = cm.get_cmap('Set2')(np.linspace(0, 1, max(8, n_specific)))
    ctype_colors = {ctype: colors[i % len(colors)] for i, ctype in enumerate(specific_ctypes)}
else:
    ctype_colors = {}


ALL_COLOR = '#3B4992'      # 'all' row
SUBGROUP_GREY = '#B5B5B5'  # subgroups in -ci ensemble, as in eval_1 --style simple

# One panel per task
for ax_idx, (task, ax) in enumerate(zip(tasks_present, axes)):
    metric = get_metric_for_task(task)
    panel_min = 40.0

    for i, ctype in enumerate(cancer_types_ordered):
        if ctype not in all_stats[task]:
            continue

        stats = all_stats[task][ctype]
        mean = stats['mean']
        std = stats['std']

        if np.isnan(mean):
            continue

        xerr = std * 100 if (not np.isnan(std) and std > 0) else None
        is_all = (ctype == 'all')

        if ci_ensemble:
            # per-cell size threshold: the row filter only needs one task to pass
            if not is_all and (stats['n_patients'] < min_num_samples or stats['n_positives'] < min_num_pos):
                continue
            ens = all_ensemble_stats.get(task, {}).get(ctype)
            if ens is None or np.isnan(ens):
                continue
            ci = all_ensemble_ci.get(task, {}).get(ctype)
            xerr_e = None if ci is None else [[max(ens - ci[0], 0) * 100], [max(ci[1] - ens, 0) * 100]]
            ax.barh(i, ens * 100, height=0.6, xerr=xerr_e,
                    color=ALL_COLOR if is_all else SUBGROUP_GREY, edgecolor='black', linewidth=0.8,
                    error_kw={'elinewidth': 1.0, 'capsize': 2.5, 'capthick': 1.0})
            panel_min = min(panel_min, (ci[0] if ci is not None else ens) * 100)
            continue
        # combo: bar for 'all', forest plot for the rest
        if combo_style:
            if is_all:
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
                ax.errorbar(
                    mean * 100,
                    i,
                    xerr=xerr,
                    fmt='o',
                    color='black',
                    ecolor='black',
                    markersize=6,
                    capsize=3,
                    linewidth=1.2,
                )
        else:
            # default: coloured bars, faded for specific cancer types
            if is_all:
                color = '#3B4992'
                alpha = 0.9
            else:
                color = ctype_colors.get(ctype, '#808080')
                alpha = 0.5

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

    # ensemble value as a short red dashed line
    if task in all_ensemble_stats and not ci_ensemble:
        for i, ctype in enumerate(cancer_types_ordered):
            if ctype in all_ensemble_stats[task]:
                ensemble_val = all_ensemble_stats[task][ctype]
                if not np.isnan(ensemble_val):
                    ax.plot([ensemble_val * 100, ensemble_val * 100],
                           [i - 0.3, i + 0.3],
                           color='red', linestyle='--', linewidth=2, alpha=0.9, zorder=10)

    ax.axvline(50, linestyle=':', color='gray', alpha=0.3, linewidth=1)
    ax.set_xlabel('AUROC (%)' if args.auroc_mode == 'weighted' else 'Overall AUROC (%)', fontsize=10, fontweight='bold')
    if ci_ensemble:
        _kind = '' if task in BINARY_TASKS else ('Weighted ' if args.auroc_mode == 'weighted' else 'Overall ')
        ax.set_xlabel(_kind + 'AUROC (%)', fontsize=10, fontweight='bold')
    ax.set_xlim(min(40, 10 * np.floor((panel_min - 2) / 10)) if ci_ensemble else 40, 100)
    ax.set_ylim(-0.5, n_ctypes - 0.5)
    ax.invert_yaxis()
    ax.set_title(TASK_DISPLAY.get(task, task), fontsize=11, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)

    ax.set_yticks(y_positions)

    # y labels on the leftmost panel only
    if ax_idx == 0:
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.tick_params(axis='y', which='both', length=4)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)

    # separator below 'all'
    if 'all' in cancer_types_ordered:
        ax.axhline(0.5, color='gray', linewidth=1.5, linestyle='-', alpha=0.4)

# Legend
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

if ci_ensemble:
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=ALL_COLOR, edgecolor='black', label='All cancers'),
        Rectangle((0, 0), 1, 1, facecolor=SUBGROUP_GREY, edgecolor='black', label='Specific cancer types'),
    ]
elif combo_style:
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#3B4992', edgecolor='black', alpha=0.9, label='All cancers'),
        Line2D([0], [0], marker='o', color='black', linestyle='-', markersize=6, label='Specific cancer types'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ensemble performance')
    ]
else:
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#3B4992', edgecolor='black', alpha=0.9, label='All cancers'),
        Rectangle((0, 0), 1, 1, facecolor='#66C2A5', edgecolor='black', alpha=0.5, label='Specific cancer types'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Ensemble performance')
    ]

if ci_ensemble:
    plt.suptitle(f'SPARC-P Performance Across Cancer Types\n'
                f'{dataset.replace("_", "-")} | {modules} | min_samples≥{min_num_samples}, min_pos≥{min_num_pos} | '
                f'bars: 10-fold ensemble, error bars: 95% bootstrap CI',
                fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.025),
              ncol=2, fontsize=10, frameon=False)
else:
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=3, fontsize=10, frameon=True, fancybox=True)

    plt.suptitle(f'SPARC-P Performance Across Cancer Types\n'
                f'{dataset.replace("_", "-")} | {modules} | min_samples≥{min_num_samples}, min_pos≥{min_num_pos}',
                fontsize=12, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
combo_suffix = ('_combo' if combo_style else '') + ('_ensembleCI' if ci_ensemble else '')
am_suffix = '_overallauroc' if args.auroc_mode == 'overall' else ''
out_pdf = os.path.join(OUTPUT_DIR,
                       f'bar_all_tasks_{modules}_{dataset}_minS{min_num_samples}_minP{min_num_pos}{combo_suffix}{am_suffix}.pdf')
out_png = os.path.join(OUTPUT_DIR,
                       f'bar_all_tasks_{modules}_{dataset}_minS{min_num_samples}_minP{min_num_pos}{combo_suffix}{am_suffix}.png')

fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\nSaved bar chart to:")
print(f"  {out_pdf}")
print(f"  {out_png}")
plt.close()

print("\nDone!")
