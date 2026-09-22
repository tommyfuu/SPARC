"""
Module-tokenizer ablation (with vs without) across all 7 tasks, one figure.

Bars are mean +/- std over folds. Both arms come from a single CSV per task,
written by design_choice_comparison_5_modular_new_fixed.py, which varies only
the tokenizer. An earlier timeencode ablation confounded this by also varying
model class, continuous-time encoding and weighted BCE; it is in deprecated/.

Metric (-mt): 'auroc' uses test_auroc_nextinf_weighted for the side-effect tasks
and auroc for the binary ones; 'cindex' uses test_cindex_weighted and skips the
binary tasks. -am switches weighted/overall.

Significance (-st) defaults to paired Wilcoxon signed-rank, since both arms use
the same folds; ttest_rel and mannwhitney are selectable. At n=10 the smallest
attainable two-sided p is 0.00195, so the strongest results all pin there -- and
the folds measure stability across re-splits of one dataset, not sampling from
the patient population, so treat the p-values as descriptive.

The AKI CSV carries a third arm ('nomod_bce') that is filtered out here.

Usage:
    python3 vis_design_choice_5_modular_new_fixed_bar.py
    python3 vis_design_choice_5_modular_new_fixed_bar.py -mt cindex
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

import argparse
import numpy as np
import pandas as pd
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
                    default='/data1/morrisq/fuc/SPARC_ablation_direct_091826',
                    help='root the comparison script wrote under')
parser.add_argument("-rr", "--results_root", type=str, default=None,
                    help='where the CSVs live (default: <ablation_root>/results)')
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
                    help='Override the output directory (default: alongside the input CSVs). Use it to regenerate figures without touching the published ones.')
args = parser.parse_args()

results_root = args.results_root or f'{args.ablation_root}/results'
csv_dir = f'{results_root}/design_choices_5_modular_new_fixed'
save_dir = args.save_dir if args.save_dir else csv_dir
modules = args.modules
dataset = args.dataset
metric = args.metric

SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']
BINARY_TASKS = ['binary', 'binary_icu_pre']
ALL_TASKS = SIDE_EFFECT_TASKS + (BINARY_TASKS if metric == 'auroc' else [])

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

# Arm keys as written by design_choice_comparison_5_modular_new_fixed.py.
# 'nomod_bce' is present in the AKI CSV only and is deliberately excluded.
MODELS_TO_PLOT = ['full', 'nomod']
EXCLUDED_ARMS = ['nomod_bce']

MODEL_DISPLAY = {
    'full':  'With Modular',
    'nomod': 'Without Modular',
}

# Same pair as the other design-choice figures. Verified against the six-check
# palette rules: normal-vision dE 29.8, worst case (tritanopia) 18.2, both above
# the 15 / 8 floors.
MODEL_COLORS = {
    'full':  '#3B4992',   # royal blue, matches SPARC-P elsewhere
    'nomod': '#A0A0A0',   # gray, the ablated baseline
}

# Load CSVs
task_data = {}

for task in ALL_TASKS:
    src_col = SRC_COL['side_effect' if task in SIDE_EFFECT_TASKS else 'binary']
    if src_col is None:
        continue

    path = f'{csv_dir}/modular_new_fixed_{modules}_{task}_{dataset}.csv'
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
        "Run design_choice_comparison_5_modular_new_fixed.py first.")

print(f"\nLoaded {n_tasks} tasks: {available_tasks}")
print(f"Metric: {metric} ({X_LABEL})\n")

# Data summary export — same aggregation the figure plots from
build_design_choice_summary(
    design_choice='design_choice_5_modular_new_fixed',
    task_data=task_data,
    available_tasks=available_tasks,
    models=MODELS_TO_PLOT,
    metrics={metric: UNIFIED_COL},
    save_dir=save_dir,
    out_stem=f'modular_new_fixed_{modules}_{dataset}_{metric}_{args.auroc_mode}',
    task_display=TASK_DISPLAY,
    model_display=MODEL_DISPLAY,
    to_percent=True,
)


def get_vals(task, model_key):
    """Fold-level values for one arm on one task, as percentages, fold-ordered.

    Fold order is enforced so the paired test lines up fold i against fold i.
    """
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
        # reversed so 'full' sits on top within each group
        y = current_y + (n_models - 1 - i) * bar_height
        y_positions[(task, model)] = y
        ys.append(y)
    task_centers[task] = np.mean(ys)
    current_y += n_models * bar_height + task_gap

fig, ax = plt.subplots(figsize=(10, max(8, n_tasks * 1.5)))

for task in available_tasks:
    for model in MODELS_TO_PLOT:
        vals = get_vals(task, model)
        if len(vals) == 0:
            continue
        ax.barh(y_positions[(task, model)], np.mean(vals), height=bar_height,
                xerr=np.std(vals), color=MODEL_COLORS[model],
                edgecolor='black', linewidth=1.0,
                error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})
        # direct value label -- two series, so labelling both stays readable

# Paired significance testing -- test and correction are both selectable (-st / -bf)
alpha = 0.05
bonferroni = args.bonferroni.lower() == 'true'
print("=" * 78)
TEST_LABEL = {'ttest_rel': 'paired t-test',
              'wilcoxon': 'paired Wilcoxon signed-rank',
              'mannwhitney': 'Mann-Whitney U (unpaired)'}[args.stat_test]
print(f"SIGNIFICANCE (With vs Without Modular; {TEST_LABEL})")
print(f"Test: {args.stat_test} | Correction: "
      f"{'Bonferroni x %d' % n_tasks if bonferroni else 'none (uncorrected)'} | alpha = {alpha}")
print("=" * 78)

x_offset, text_offset = 2.0, 0.5
for task in available_tasks:
    a, b = get_vals(task, 'full'), get_vals(task, 'nomod')
    if len(a) < 2 or len(b) < 2:
        continue
    if len(a) != len(b):
        print(f"  {TASK_DISPLAY[task]:25s}: unpaired fold counts "
              f"({len(a)} vs {len(b)}) — test skipped")
        continue

    diffs = a - b
    if np.allclose(diffs, 0):
        # wilcoxon raises and ttest_rel returns nan when every difference is zero
        p_raw = 1.0
    else:
        _, p_raw = ({'ttest_rel':   lambda: ttest_rel(a, b),
                     'wilcoxon':    lambda: wilcoxon(a, b),
                     'mannwhitney': lambda: mannwhitneyu(a, b, alternative='two-sided')
                     }[args.stat_test])()
    p_corr = min(p_raw * n_tasks, 1.0) if bonferroni else p_raw
    star = ('***' if p_corr < 0.001 else '**' if p_corr < 0.01
            else '*' if p_corr < alpha else 'NS')

    print(f"  {TASK_DISPLAY[task]:25s}: p_raw={p_raw:.4f}, p_used={p_corr:.4f} -> {star:3s} "
          f"(delta={np.mean(diffs):+.2f}%, with-modular better in "
          f"{int((diffs > 0).sum())}/{len(diffs)} folds)")

    y_a, y_b = y_positions[(task, 'full')], y_positions[(task, 'nomod')]
    x_right = max(np.mean(a) + np.std(a), np.mean(b) + np.std(b)) + x_offset
    ax.plot([x_right, x_right], [y_a, y_b], 'k-', linewidth=1,
            alpha=0.5 if star == 'NS' else 1.0)
    ax.text(x_right + text_offset, (y_a + y_b) / 2, star, ha='left', va='center',
            fontsize=9, alpha=0.6 if star == 'NS' else 1.0, fontweight='bold')

print("=" * 78 + "\n")

# Pooled across tasks. Folds stay paired, but fold values are correlated within a
# task, so this is descriptive -- it is not an independent-sample test.
pooled_a = np.concatenate([get_vals(t, 'full') for t in available_tasks])
pooled_b = np.concatenate([get_vals(t, 'nomod') for t in available_tasks])
if len(pooled_a) == len(pooled_b) and len(pooled_a) > 1:
    print("=" * 78)
    print("POOLED ACROSS TASKS (descriptive; folds correlated within task)")
    print("=" * 78)
    print(f"  With Modular:    {pooled_a.mean():.2f} +/- {pooled_a.std():.2f}%")
    print(f"  Without Modular: {pooled_b.mean():.2f} +/- {pooled_b.std():.2f}%")
    print(f"  Difference:      {(pooled_a - pooled_b).mean():+.2f}%")
    print("=" * 78 + "\n")

ax.set_yticks([task_centers[t] for t in reversed(available_tasks)])
ax.set_yticklabels([TASK_DISPLAY[t] for t in reversed(available_tasks)], fontsize=11)
ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
ax.set_xlim(left=X_MIN, right=X_MAX)
ax.set_title(f'Module Tokenizer Ablation — {dataset.replace("_", "-")}',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m],
                             edgecolor='black', label=MODEL_DISPLAY[m])
                   for m in MODELS_TO_PLOT],
          loc='lower right', fontsize=11, frameon=True, fancybox=True)

plt.tight_layout()
stem = f'{save_dir}/modular_new_fixed_{modules}_{dataset}_{metric}_{args.auroc_mode}_horizontal_bar'
fig.savefig(f'{stem}.pdf', bbox_inches='tight')
fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=150)
plt.close()

print("Figure saved:")
print(f"  {stem}.pdf")
print(f"  {stem}.png\n")
