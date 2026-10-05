"""DC6: time representation ablation -- Fourier encoding (A9) vs raw time (A16/A17) vs no time (A15).

All 7 tasks (mortality tasks only in the AUROC figure), read from <results_root>/design_choices_10_rawtime_bf
(written by design_choice_comparison_6_time_representation.py). -st (default paired Wilcoxon) compares Fourier with each other arm.
Usage: python3 vis_design_choice_6_time_representation_bar.py [-mt cindex]
"""

# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import numpy as np
import pandas as pd
import sparc_paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design_choice_summary import build_design_choice_summary
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("-ar", "--ablation_root", type=str,
                    default='/data1/morrisq/fuc/SPARC_ablation_direct_092326',
                    help='root the comparison script wrote under')
parser.add_argument("-rr", "--results_root", type=str, default=sparc_paths.SPARC_RESULTS_VIS_ROOT,
                    help='where the design_choices_* folders live (default: the bf=True results tree; '
                         'pass <ablation_root>/results for the original copies)')
parser.add_argument("-m", "--modules", type=str, default='parsimonious')
parser.add_argument("-ds", "--dataset", type=str, default='MSK')
parser.add_argument("-mt", "--metric", type=str, default='auroc', choices=['auroc', 'cindex'])
parser.add_argument("-st", "--stat_test", type=str, default='wilcoxon',
                    choices=['wilcoxon', 'ttest_rel', 'mannwhitney'],
                    help="'wilcoxon' (paired signed-rank, the default: both arms are scored on the same "
                         "10 folds, so observations are paired by design, and the rank statistic is not "
                         "distorted by a single extreme fold). 'ttest_rel' is the paired parametric "
                         "alternative -- avoid it here: on binary_icu_pre one fold differs by +11.66 vs "
                         "+0.14..+3.73 for the rest, which triples the SD of the differences and drives p "
                         "from 0.002 to 0.064 even though all 10 folds favour the same arm. "
                         "'mannwhitney' is unpaired -- it reproduces the deprecated "
                         "timeencode figure. "
                         "Note at n=10 the smallest attainable two-sided p is 0.00195, so the strongest "
                         "results all pin there; read the delta column for effect size, not the stars.")
parser.add_argument("-bf", "--bonferroni", type=str, default='False',
                    help="True | False -- Bonferroni-correct the per-task p-values by the number of "
                         "tasks. Off by default: these are pre-specified single comparisons per task, "
                         "not a search over many, so the family-wise correction is not called for.")
parser.add_argument("-am", "--auroc_mode", type=str, default='weighted', choices=['weighted', 'overall'],
                    help="For side-effect tasks only. 'weighted' averages per-infusion-group AUROCs "
                         "weighted by group size (groups with one class present are dropped); 'overall' "
                         "pools all valid samples into a single AUROC. Binary tasks have one AUROC either way.")
parser.add_argument('--save_dir', type=str, default=None,
                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without overwriting the existing ones.')
args = parser.parse_args()

results_root = args.results_root or f'{args.ablation_root}/results'
csv_dir = f'{results_root}/design_choices_10_rawtime_bf'   # A9 / A16-A17 / A15, batch_first=True
save_dir = args.save_dir if args.save_dir else csv_dir
modules = args.modules
dataset = args.dataset
metric = args.metric

SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']
BINARY_TASKS = ['binary', 'binary_icu_pre']
ALL_TASKS = SIDE_EFFECT_TASKS + BINARY_TASKS   # C-index skips the binary tasks

# Source column per task type, per metric. Binary CSVs carry only auroc/auprc.
SRC_COL = {
    'auroc':  {'side_effect': f'test_auroc_nextinf_{args.auroc_mode}', 'binary': 'auroc'},
    'cindex': {'side_effect': ('test_cindex_weighted' if args.auroc_mode == 'weighted'
                               else 'test_cindex_overall_valid'),      'binary': None},
}[metric]

X_LABEL = {
    ('auroc', 'weighted'): '(Weighted) AUROC (%)',
    ('auroc', 'overall'):  'Overall AUROC (%)',
    ('cindex', 'weighted'): 'Weighted C-index (%)',
    ('cindex', 'overall'):  'Overall C-index (%)',
}[(metric, args.auroc_mode)]
X_MIN, X_MAX = 50, 105

UNIFIED_COL = 'metric_plot'

TASK_DISPLAY = {
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
    'binary': '180D Hosp. mort.',
    'binary_icu_pre': '30D ICU mort.',
}

# Arm keys as written by design_choice_comparison_6_time_representation.py.
MODELS_TO_PLOT = ['fourier', 'rawtime', 'notime']
EXCLUDED_ARMS = []

MODEL_DISPLAY = {
    'fourier': 'Continuous-time encoding (Fourier)',
    'rawtime': 'Raw time value',
    'notime':  'No time input',
}

# Same palette as the other design-choice figures (colorblind-checked).
MODEL_COLORS = {
    'fourier': '#3B4992',   # royal blue, matches SPARC-P elsewhere
    'rawtime': '#8FB3E0',   # light blue, the intermediate arm
    'notime':  '#A0A0A0',   # gray, the ablated baseline
}

# Load CSVs
task_data = {}

for task in ALL_TASKS:
    src_col = SRC_COL['side_effect' if task in SIDE_EFFECT_TASKS else 'binary']
    if src_col is None:
        continue

    path = f'{csv_dir}/rawtime_{modules}_{task}_{dataset}.csv'
    if not os.path.exists(path):
        print(f"WARNING: missing {os.path.basename(path)} — {task} skipped")
        continue

    df = pd.read_csv(path)
    if 'val_fold' in df.columns:
        df = df[df['val_fold'] != 'averaged'].copy()
    df = df[~df['model'].isin(EXCLUDED_ARMS)]

    if src_col not in df.columns:
        print(f"WARNING: {os.path.basename(path)} has no '{src_col}' column — {task} skipped")
        continue

    part = df[df['model'].isin(MODELS_TO_PLOT)][['model', 'val_fold', src_col]].copy()
    part = part.rename(columns={src_col: UNIFIED_COL}).dropna(subset=[UNIFIED_COL])
    if part.empty:
        print(f"WARNING: {task}: no rows for {MODELS_TO_PLOT} — skipped")
        continue

    missing = [m for m in MODELS_TO_PLOT if m not in set(part['model'])]
    if missing:
        print(f"WARNING: {task}: arm(s) {missing} absent — bar will be missing")

    task_data[task] = part
    print(f"Loaded {task:22s} <- {os.path.basename(path)} [{src_col}]  "
          f"n={dict(part['model'].value_counts())}")

available_tasks = [t for t in ALL_TASKS if t in task_data]
n_tasks = len(available_tasks)
if n_tasks == 0:
    raise RuntimeError(
        f"No CSVs found in {csv_dir}. "
        "Run design_choice_comparison_6_time_representation.py first.")

print(f"\nLoaded {n_tasks} tasks: {available_tasks}")
print(f"Metric: {metric} ({X_LABEL})\n")

# Data summary export: same aggregation the figure plots from
build_design_choice_summary(
    design_choice='design_choice_10_rawtime',
    task_data=task_data,
    available_tasks=available_tasks,
    models=MODELS_TO_PLOT,
    metrics={metric: UNIFIED_COL},
    save_dir=save_dir,
    out_stem=f'rawtime_{modules}_{dataset}_{metric}_{args.auroc_mode}',
    task_display=TASK_DISPLAY,
    model_display=MODEL_DISPLAY,
    to_percent=True,
)


def get_vals(task, model_key):
    """Fold-level values for one arm, in percent, sorted by fold so the paired test lines up."""
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty:
        return np.array([])
    rows = rows.sort_values('val_fold')
    vals = rows[UNIFIED_COL].dropna().values
    return vals * 100 if len(vals) and vals.max() <= 1.0 else vals


# Figure
n_models = len(MODELS_TO_PLOT)
bar_height = 0.5
task_gap = 0.6

y_positions = {}
task_centers = {}
current_y = 0
for task in reversed(available_tasks):
    ys = []
    for i, model in enumerate(MODELS_TO_PLOT):
        # reversed so 'fourier' sits on top within each group
        y = current_y + (n_models - 1 - i) * bar_height
        y_positions[(task, model)] = y
        ys.append(y)
    task_centers[task] = np.mean(ys)
    current_y += n_models * bar_height + task_gap

fig, ax = plt.subplots(figsize=(10, max(8, n_tasks * 2.0)))

for task in available_tasks:
    for model in MODELS_TO_PLOT:
        vals = get_vals(task, model)
        if len(vals) == 0:
            continue
        ax.barh(y_positions[(task, model)], np.mean(vals), height=bar_height,
                xerr=np.std(vals), color=MODEL_COLORS[model],
                edgecolor='black', linewidth=1.0,
                error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

# Paired significance test (-st) and optional Bonferroni (-bf)
alpha = 0.05
bonferroni = args.bonferroni.lower() == 'true'
print("=" * 78)
TEST_LABEL = {'ttest_rel': 'paired t-test',
              'wilcoxon': 'paired Wilcoxon signed-rank',
              'mannwhitney': 'Mann-Whitney U (unpaired)'}[args.stat_test]
print(f"SIGNIFICANCE (Fourier vs raw time, Fourier vs no time; {TEST_LABEL})")
print(f"Test: {args.stat_test} | Correction: "
      f"{'Bonferroni x %d' % n_tasks if bonferroni else 'none (uncorrected)'} | alpha = {alpha}")
print("=" * 78)

COMPARISONS = [('fourier', 'rawtime'), ('fourier', 'notime')]
x_offset, text_offset, bracket_step = 2.0, 0.5, 4.5
for task in available_tasks:
    x_base = max(np.mean(get_vals(task, m)) + np.std(get_vals(task, m))
                 for m in MODELS_TO_PLOT if len(get_vals(task, m))) + x_offset
    for k, (ma, mb) in enumerate(COMPARISONS):
        a_v, b_v = get_vals(task, ma), get_vals(task, mb)
        if len(a_v) < 2 or len(b_v) < 2:
            continue
        if len(a_v) != len(b_v):
            print(f"  {TASK_DISPLAY[task]:25s} {ma} vs {mb}: unpaired fold counts "
                  f"({len(a_v)} vs {len(b_v)}) - test skipped")
            continue
        diffs = a_v - b_v
        if np.allclose(diffs, 0):
            p_raw = 1.0
        else:
            _, p_raw = ({'ttest_rel':   lambda: ttest_rel(a_v, b_v),
                         'wilcoxon':    lambda: wilcoxon(a_v, b_v),
                         'mannwhitney': lambda: mannwhitneyu(a_v, b_v, alternative='two-sided')
                         }[args.stat_test])()
        p_corr = min(p_raw * n_tasks, 1.0) if bonferroni else p_raw
        star = ('***' if p_corr < 0.001 else '**' if p_corr < 0.01
                else '*' if p_corr < alpha else 'NS')
        print(f"  {TASK_DISPLAY[task]:25s} {ma:>7s} vs {mb:<7s}: p_raw={p_raw:.4f}, p_used={p_corr:.4f} -> {star:3s} "
              f"(delta={np.mean(diffs):+.2f}%, {ma} better in {int((diffs > 0).sum())}/{len(diffs)} folds)")
        y_a, y_b = y_positions[(task, ma)], y_positions[(task, mb)]
        x_right = x_base + k * bracket_step
        ax.plot([x_right, x_right], [y_a, y_b], 'k-', linewidth=1,
                alpha=0.5 if star == 'NS' else 1.0)
        ax.text(x_right + text_offset, (y_a + y_b) / 2, star, ha='left', va='center',
                fontsize=9, alpha=0.6 if star == 'NS' else 1.0, fontweight='bold')

print("=" * 78 + "\n")

# Pooled across tasks (descriptive; folds correlated within task)
print("=" * 78)
print("POOLED ACROSS TASKS (descriptive; folds correlated within task)")
print("=" * 78)
for m in MODELS_TO_PLOT:
    v = np.concatenate([get_vals(t, m) for t in available_tasks])
    print(f"  {MODEL_DISPLAY[m]:36s}: {v.mean():.2f} +/- {v.std():.2f}%")
print("=" * 78 + "\n")

ax.set_yticks([task_centers[t] for t in reversed(available_tasks)])
ax.set_yticklabels([TASK_DISPLAY[t] for t in reversed(available_tasks)], fontsize=11)
ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
ax.set_xlim(left=X_MIN, right=X_MAX)
ax.set_title(f'Time Representation Ablation — {dataset.replace("_", "-")}',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m],
                             edgecolor='black', label=MODEL_DISPLAY[m])
                   for m in MODELS_TO_PLOT],
          loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3,
          fontsize=11, frameon=False)   # below the axis: the bottom-right corner holds 30D ICU bars

plt.tight_layout()
stem = f'{save_dir}/rawtime_{modules}_{dataset}_{metric}_{args.auroc_mode}_horizontal_bar'
fig.savefig(f'{stem}.pdf', bbox_inches='tight')
fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=150)
plt.close()

print("Figure saved:")
print(f"  {stem}.pdf")
print(f"  {stem}.png\n")
