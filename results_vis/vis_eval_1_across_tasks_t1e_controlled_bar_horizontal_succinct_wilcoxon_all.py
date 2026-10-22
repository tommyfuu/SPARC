"""Horizontal bar grid (one panel per task) of every model in the eval_1 CSVs.

-ci ensemble: ensemble value with its 95% bootstrap CI; -ci folds: fold mean +/- SD.
Stars vs. SPARC-P from --stat_test (ttest_rel / wilcoxon over folds, bootstrap or DeLong on
the ensemble predictions); --correction bonferroni is per task.

Usage: python vis_eval_1_across_tasks_t1e_controlled_bar_horizontal_succinct_wilcoxon_all.py --dataset MSK_CAIA
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
import sparc_paths
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser(
    description='Per-task subplot grid: all models plotted horizontally (y-axis = models)')
parser.add_argument('--dataset', type=str, default='MSK_CAIA',
                     help='Dataset name (e.g. MSK_CAIA, MSK_CAIA_timeshift)')
parser.add_argument('--modules', type=str, default='parsimonious',
                     help='Module category (default: parsimonious)')
parser.add_argument('--zeroshot', type=str, default='True',
                     help='Whether to use zeroshot models for MIMICIV; '
                          'when False, finetuned models are used. Default: True')
parser.add_argument('--fn', action='store_true',
                     help='Exclude febrile neutropenia from visualization')
parser.add_argument('--metric', type=str, default='weighted_auroc',
                     choices=['weighted_auroc', 'overall_auroc', 'weighted_auprc', 'overall_auprc'],
                     help="Metric to plot. 'weighted'/'overall' only differ for side-effect tasks "
                          "(binary tasks have a single AUROC/AUPRC column either way). "
                          "Default: weighted_auroc")
parser.add_argument('-ci', '--ci', type=str, default='ensemble', choices=['ensemble', 'folds'],
                    help="'ensemble' (default): bar = ensemble estimate, error bar = its 95%% bootstrap CI "
                         "(no diamond). 'folds': bar = mean over folds, error bar = SD over folds, plus the "
                         "ensemble diamond (original figure). Significance stars are the fold-paired tests in both.")
parser.add_argument('--ensemble', dest='ensemble', action='store_true', default=True,
                     help='Show ensemble performance as red diamond markers (default: on)')
parser.add_argument('--no-ensemble', dest='ensemble', action='store_false',
                     help='Hide ensemble markers')
parser.add_argument('--highlight', type=str, default='',
                     help="Model to visually highlight (thicker black outline) in every "
                          "subplot. Default is '' (no highlight) — SPARC is no longer "
                          "emphasised by default so all models are drawn identically. "
                          "Use the internal model name, e.g. 'SPARC-P'.")
parser.add_argument('--alpha', type=float, default=0.05,
                     help="Threshold for a single star (default 0.05). ** and *** stay at p<0.01 / "
                          "p<0.001. A non-default value adds _alpha<value> to the filename.")
parser.add_argument('--style', type=str, default='default', choices=['default', 'simple'],
                     help="'simple': SPARC (SPARC-P) keeps its colour and every other method is grey; "
                          "methods are grouped time-series first, then last-timepoint, ordered by "
                          "performance within each group (per --sort_by), with a thin rule between "
                          "the groups. Adds _simple to the filename.")
parser.add_argument('--sort_by', type=str, default='global', choices=['global', 'per_task'],
                     help="'global': models ranked once (by mean performance across all "
                          "available tasks) and that order is kept fixed in every subplot. "
                          "'per_task': each subplot independently ranks its own models. "
                          "Default: global (keeps rows aligned across the panel).")
parser.add_argument('--ncols', type=int, default=3,
                     help='Number of subplot columns in the grid (default: 3)')
parser.add_argument('--model_path', type=str, default=None,
                     help='Root holding the results_vis_* trees. Defaults to '
                          '$SPARC_MODEL_PATH, else the path in DEFAULT_MODEL_PATH.')
parser.add_argument('--results_subdir', type=str, default='results_vis_eval1_bfTrue',
                     help='Which results tree under --model_path to read '
                          '(default: %(default)s). Use e.g. '
                          'results_vis_060726_batch_091426 for a batch rerun.')
parser.add_argument('--eval_subdir', type=str, default='eval_across_models_valthresh_disabled',
                     help='Subdirectory of the results tree holding the CSVs '
                          '(default: %(default)s)')
parser.add_argument('--csv_dir', type=str, default=None,
                     help='Override the CSV input directory outright, ignoring '
                          '--model_path/--results_subdir/--eval_subdir')
parser.add_argument('--save_dir', type=str, default=None,
                     help='Override the output directory (defaults to csv_dir)')
parser.add_argument('--family_style', type=str, default='color',
                     choices=['color', 'underline'],
                     help="How to mark time-series vs. last-timepoint methods on the "
                          "y-axis labels: 'color' (teal vs. amber text) or 'underline' "
                          "(black text, time-series labels underlined). Default: color")
parser.add_argument('--stat_test', type=str, default='ttest_rel',
                     choices=['ttest_rel', 'wilcoxon', 'bootstrap', 'delong'],
                     help="Paired test for SPARC-P vs. each model. Both arms are scored on the "
                          "same CV folds, so the observations are paired either way; this only "
                          "chooses parametric vs. rank-based. "
                          "'ttest_rel' (default) -- paired t-test. Justified here empirically: over "
                          "the 63 SPARC-P-vs-benchmark comparisons on MSK_CAIA, Shapiro-Wilk rejects "
                          "normality of the paired differences in only 6 (10%%, about the chance rate), "
                          "and the two tests disagree at alpha=0.05 in 1 of 63. It also avoids the "
                          "n=10 rank floor below. "
                          "'wilcoxon' -- paired signed-rank. Prefer it when a single fold is a large "
                          "outlier: the t-test puts the spread of the differences in its denominator, "
                          "so one extreme fold AGREEING with the effect inflates the SD and shrinks t. "
                          "Cost: at n=10 the smallest attainable two-sided p is 0.00195, so the "
                          "strongest comparisons all pin there and stop being distinguishable. "
                          "'bootstrap' -- paired sample-level bootstrap of the 10-fold ENSEMBLE "
                          "predictions (bootstrap_stats.py): one shared resample per draw, stratified "
                          "within infusion groups, two-sided p = 2*min(P(diff<0), P(diff>0)), ties "
                          "split. Tests the same quantity the -ci ensemble bars show and has no n=10 "
                          "floor (floor is ~2/n_bootstrap). Falls back to an unpaired bootstrap if a "
                          "model's rows are not aligned with SPARC-P's (printed). "
                          "'delong' -- paired DeLong test on the ensemble predictions (AUROC only): "
                          "plain DeLong on the pooled valid rows for overall AUROC and the mortality "
                          "tasks; for weighted AUROC a stratified DeLong, i.e. the group-size-weighted "
                          "sum of the per-infusion-group AUROC differences with variance "
                          "sum(w_g^2 * Var_g), groups independent. No stars for AUPRC. "
                          "The chosen test is appended to the output filename.")
parser.add_argument('--n_bootstrap', type=int, default=1000,
                     help='Resamples for --stat_test bootstrap (default 1000)')
parser.add_argument('--boot_seed', type=int, default=1024,
                     help='Seed for --stat_test bootstrap (default 1024)')
parser.add_argument('--precompute_only', action='store_true',
                     help='With --stat_test bootstrap: fill bootstrap_cache/ for --task (or all '
                          'tasks) and exit without drawing. Tasks are seeded independently, so '
                          'running them as separate jobs gives the same numbers as one serial run.')
parser.add_argument('--task', type=str, default=None,
                     help='Comma-separated task(s) for --precompute_only (default: all)')
parser.add_argument('--correction', type=str, default='none',
                     choices=['none', 'bonferroni'],
                     help='Multiple testing correction, applied WITHIN each task: '
                          'the family is the set of models compared against SPARC-P '
                          'in that one subplot, not the whole figure. A model '
                          'appearing in several tasks is corrected independently in '
                          'each. Options: none or bonferroni (default: none)')
args = parser.parse_args()

dataset = args.dataset
modules = args.modules
zeroshot = args.zeroshot.lower() == 'true'
exclude_fn = args.fn
metric = args.metric
ci_mode = args.ci
show_ensemble = args.ensemble and ci_mode == 'folds'   # the CI bars already show the ensemble
highlight_model = args.highlight.strip() if args.highlight else None
sort_by = args.sort_by
ncols_req = args.ncols
correction_method = args.correction
stat_test = args.stat_test
family_style = args.family_style

# root holding the results_vis_* trees ($SPARC_BF_MODEL_PATH or --model_path)
DEFAULT_MODEL_PATH = sparc_paths.SPARC_BF_MODEL_ROOT
all_model_path = args.model_path or DEFAULT_MODEL_PATH

# --csv_dir overrides the path built from the parts
default_csv_dir = os.path.join(all_model_path, args.results_subdir, args.eval_subdir)
csv_dir = args.csv_dir if args.csv_dir else default_csv_dir
save_dir = args.save_dir if args.save_dir else csv_dir

if not os.path.isdir(csv_dir):
    raise SystemExit(
        f"ERROR: CSV directory not found:\n  {csv_dir}\n"
        f"Set --model_path / --results_subdir / --eval_subdir, or --csv_dir to "
        f"point at the directory holding "
        f"model_performance_dict_across_models_*.csv"
    )

os.makedirs(save_dir, exist_ok=True)
print(f"Reading CSVs from : {csv_dir}")
print(f"Writing output to : {save_dir}")

SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']
BINARY_TASKS = ['binary', 'binary_icu_pre']

# 3x3 layout, None = empty panel
if exclude_fn:
    TASK_ORDER = [
        'AKI', 'anemia', None,
        'neutropenia', 'Liver_toxicity', None,
        'binary', 'binary_icu_pre', None
    ]
else:
    TASK_ORDER = [
        'AKI', 'anemia', 'febrile_neutropenia',
        'neutropenia', 'Liver_toxicity', None,
        'binary', 'binary_icu_pre', None
    ]

ALL_TASKS = [t for t in TASK_ORDER if t is not None]

METRIC_TO_COLUMN_SIDE_EFFECT = {
    'weighted_auroc': 'test_auroc_nextinf_weighted',
    'overall_auroc':  'test_auroc_nextinf_overall',
    'weighted_auprc': 'test_auprc_nextinf_weighted',
    'overall_auprc':  'test_auprc_nextinf_overall',
}
METRIC_TO_COLUMN_BINARY = {
    'weighted_auroc': 'auroc',
    'overall_auroc':  'auroc',
    'weighted_auprc': 'auprc',
    'overall_auprc':  'auprc',
}
METRIC_COL = {}
for t in SIDE_EFFECT_TASKS:
    METRIC_COL[t] = METRIC_TO_COLUMN_SIDE_EFFECT[metric]
for t in BINARY_TASKS:
    METRIC_COL[t] = METRIC_TO_COLUMN_BINARY[metric]

metric_is_auroc = 'auroc' in metric
metric_is_weighted = metric.startswith('weighted')
metric_label_name = 'AUROC' if metric_is_auroc else 'AUPRC'
metric_label_kind = 'Weighted' if metric_is_weighted else 'Overall'

# ensemble-row bootstrap CI columns from eval_1: auroc_ci_* is the overall AUROC CI,
# auroc_weighted_ci_* the weighted one; weighted AUPRC has no CI
CI_COLS_SIDE_EFFECT = {
    'weighted_auroc': ('auroc_weighted_ci_lower', 'auroc_weighted_ci_upper'),
    'overall_auroc':  ('auroc_ci_lower', 'auroc_ci_upper'),
    'weighted_auprc': None,
    'overall_auprc':  ('auprc_ci_lower', 'auprc_ci_upper'),
}
CI_COLS_BINARY = {
    'weighted_auroc': ('auroc_ci_lower', 'auroc_ci_upper'),
    'overall_auroc':  ('auroc_ci_lower', 'auroc_ci_upper'),
    'weighted_auprc': ('auprc_ci_lower', 'auprc_ci_upper'),
    'overall_auprc':  ('auprc_ci_lower', 'auprc_ci_upper'),
}

TASK_DISPLAY = {
    'AKI':                 'Acute kidney injury',
    'Liver_toxicity':      'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia':              'Severe anemia',
    'neutropenia':         'Neutropenia',
    'binary':              '180D Hosp. mort.',
    'binary_icu_pre':      '30D ICU mort.',
}

# display names; unlisted models keep their raw name
MODEL_DISPLAY = {
    'SPARC-P':          'SPARC',  # published name
    'lasttp_MLP':       'MLP',
    'lasttp_xgboost':   'XGBoost',
    'temporal_xgboost': 'T-XGBoost',
    'temporal_tabpfn':  'TabPFN',
    'rnn':              'RNN',
    'duett':            'DuETT',
    'raindrop':         'Raindrop',
}

# time-series vs. last-timepoint methods
SIMPLE_GREY = '#B5B5B5'   # --style simple: benchmark bars
TEMPORAL_MODELS = {'SPARC-P', 'temporal_xgboost', 'temporal_tabpfn', 'rnn', 'duett', 'raindrop'}
LAST_TP_MODELS = {'lasttp_MLP', 'lasttp_xgboost'}

# family colour of the y-axis labels: Okabe-Ito blue and a darkened vermillion
# (colour-blind safe, >= 4.5:1 contrast on white)
TIMESERIES_COLOR = '#0072B2'
LASTTP_COLOR     = '#B54A00'

# Load CSVs (prefer the _ensemble.csv)
_finetuned_suffix = '_finetuned' if dataset == 'MIMICIV_hosp' and not zeroshot else ''

task_data = {}         # fold rows
task_data_full = {}    # fold + ensemble rows
task_sample_info = {}  # n and n_pos per task

for task in ALL_TASKS:
    ensemble_csv_path = f'{csv_dir}/model_performance_dict_across_models_{modules}_{task}_{dataset}{_finetuned_suffix}_ensemble.csv'
    regular_csv_path = f'{csv_dir}/model_performance_dict_across_models_{modules}_{task}_{dataset}{_finetuned_suffix}.csv'

    csv_path = None
    if os.path.exists(ensemble_csv_path):
        csv_path = ensemble_csv_path
    elif os.path.exists(regular_csv_path):
        csv_path = regular_csv_path
    else:
        print(f"WARNING: missing CSV for {task} — task will be skipped")
        continue

    df_full = pd.read_csv(csv_path)
    task_data_full[task] = df_full

    if 'val_fold' in df_full.columns:
        df = df_full[df_full['val_fold'] != 'ensemble'].copy()
    else:
        df = df_full.copy()
    task_data[task] = df

    # n_pos is next-infusion positives for side-effect tasks, overall positives for binary
    if len(df_full) > 0:
        n_test = df_full['n_test'].iloc[0] if 'n_test' in df_full.columns else None
        if task in SIDE_EFFECT_TASKS and 'n_pos_next_infusion' in df_full.columns:
            n_pos = df_full['n_pos_next_infusion'].iloc[0]
        elif 'n_pos_overall' in df_full.columns:
            n_pos = df_full['n_pos_overall'].iloc[0]
        else:
            n_pos = None
        task_sample_info[task] = {'n': n_test, 'n_pos': n_pos}

available_tasks = ALL_TASKS

if len(available_tasks) == 0:
    raise RuntimeError("No task CSVs found. Run eval_1_across_models_comparison_ensemble_FP_fixed.py first "
                        "(or pass --csv_dir to point at your data).")

print(f"\nCSV directory: {csv_dir}")
print(f"Loaded {len(available_tasks)} tasks: {available_tasks}")
print(f"Metric: {metric} ({metric_label_kind} {metric_label_name})\n")


def get_vals(task, model_key):
    """Fold-level metric values (percent)."""
    col = METRIC_COL[task]
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty or col not in rows.columns:
        return np.array([])
    return rows[col].dropna().values * 100


def get_ensemble_val(task, model_key):
    """Ensemble-row value (percent) or None."""
    col = METRIC_COL[task]
    df = task_data_full.get(task)
    if df is None:
        return None
    ensemble_model_name = f'{model_key}_ensemble'
    rows = df[(df['model'] == ensemble_model_name) & (df['val_fold'] == 'ensemble')]
    if rows.empty or col not in rows.columns:
        return None
    val = rows[col].iloc[0]
    if pd.isna(val):
        return None
    return val * 100


def get_ensemble_ci(task, model_key):
    """(lower, upper) 95% bootstrap CI (percent) of the ensemble row, or None."""
    cols = (CI_COLS_BINARY if task in BINARY_TASKS else CI_COLS_SIDE_EFFECT)[metric]
    df = task_data_full.get(task)
    if cols is None or df is None or not all(c in df.columns for c in cols):
        return None
    rows = df[(df['model'] == f'{model_key}_ensemble') & (df['val_fold'] == 'ensemble')]
    if rows.empty:
        return None
    lo, hi = rows[cols[0]].iloc[0], rows[cols[1]].iloc[0]
    if pd.isna(lo) or pd.isna(hi):
        return None
    return lo * 100, hi * 100


def bar_value(task, model_key):
    """Value the bar shows: ensemble estimate (-ci ensemble) or fold mean (-ci folds)."""
    if ci_mode == 'ensemble':
        e = get_ensemble_val(task, model_key)
        if e is not None:
            return e
    v = get_vals(task, model_key)
    return np.mean(v) if len(v) else -np.inf


def discover_models(task):
    """Base model names in a task's fold-level data, excluding *_ensemble rows and 'SPARC' (non-parsimonious)."""
    df = task_data.get(task)
    if df is None or 'model' not in df.columns:
        return []
    models = [m for m in df['model'].unique()
              if not str(m).endswith('_ensemble') and m != 'SPARC']
    return models


per_task_models = {task: discover_models(task) for task in available_tasks}
all_models_seen = sorted(set(m for models in per_task_models.values() for m in models))


# global ranking: mean bar value over the tasks a model appears in
model_means = {}
for m in all_models_seen:
    vals = []
    for task in available_tasks:
        if m in per_task_models[task]:
            bv = bar_value(task, m)
            if np.isfinite(bv):
                vals.append(bv)
    model_means[m] = np.mean(vals) if vals else -np.inf

global_model_order = sorted(all_models_seen, key=lambda m: model_means[m], reverse=True)

PROFESSIONAL_COLORS = [
    '#2C5F8D',
    '#E07A5F',
    '#81B29A',
    '#F2CC8F',
    '#6A4C93',
    '#D96459',
    '#5A7D7C',
    '#B68D40',
    '#4A6FA5',
    '#A87C7C',
]

# colour fixed per model (MSK_CAIA weighted-AUROC order) so it is stable across figures
FIXED_COLOR_ORDER = ['SPARC-P', 'temporal_tabpfn', 'temporal_xgboost', 'lasttp_xgboost',
                     'rnn', 'lasttp_MLP', 'duett', 'raindrop']
MODEL_COLORS = {m: PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)]
                for i, m in enumerate(FIXED_COLOR_ORDER)}
for m in global_model_order:
    if m not in MODEL_COLORS:
        MODEL_COLORS[m] = PROFESSIONAL_COLORS[len(MODEL_COLORS) % len(PROFESSIONAL_COLORS)]

print("Models discovered (ranked by mean performance across available tasks):")
for m in global_model_order:
    disp = MODEL_DISPLAY.get(m, m)
    mean_val = model_means[m]
    temporal_marker = " [time series]" if m in TEMPORAL_MODELS else " [last timepoint]" if m in LAST_TP_MODELS else ""
    print(f"  {disp:12s} ({m:20s}): mean = {mean_val:.2f}%{temporal_marker}" if np.isfinite(mean_val)
          else f"  {disp:12s} ({m:20s}): mean = n/a{temporal_marker}")
print()

def paired_test(vals1, vals2):
    """Fold-paired t-test or Wilcoxon signed-rank (per --stat_test); None if not computable."""
    if len(vals1) == 0 or len(vals2) == 0:
        return None
    if len(vals1) != len(vals2):
        return None
    try:
        if stat_test == 'wilcoxon':
            import numpy as _np
            if _np.allclose(_np.asarray(vals1) - _np.asarray(vals2), 0):
                return None       # wilcoxon raises if all differences are zero
            stat_v, p_value = stats.wilcoxon(vals1, vals2, alternative='two-sided')
        else:
            stat_v, p_value = stats.ttest_rel(vals1, vals2, alternative='two-sided')
        return p_value
    except Exception:
        return None

_EVAL1_RSEED, _EVAL1_BATCH = 1024, 512   # eval_1's -r default and test batch size

def _eval1_loader_perm(n):
    """Replay eval_1's seeded test-loader shuffle.

    The DL models' prediction rows follow this shuffled order, the tree models' rows are in
    index order, so dl_row[k] == tree_row[perm[k]].
    """
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    pl.seed_everything(_EVAL1_RSEED, workers=False)
    return np.concatenate([b.numpy() for b in
                           DataLoader(list(range(n)), batch_size=_EVAL1_BATCH, shuffle=True)])

def _align_to_reference(preds, ref_key='SPARC-P'):
    """Reorder each model's rows to SPARC-P's order if the replayed shuffle makes labels, times
    and infusion numbers match exactly; otherwise leave it (bootstrap_stats then tests unpaired)."""
    keys = ['y_events', 'time_to_events', 'num_infusions']
    ref = preds[ref_key]
    same = lambda d, idx=None: all(
        np.array_equal(np.asarray(ref[k]).ravel(),
                       (np.asarray(d[k]).ravel() if idx is None else np.asarray(d[k]).ravel()[idx]))
        for k in keys)
    # tree models store (n,) where DL models store (n,1); bootstrap_stats needs equal shapes
    fix_shape = lambda d: {**d, **{k: np.asarray(d[k]).reshape(np.asarray(ref[k]).shape) for k in keys}}
    perm = None
    for m, d in list(preds.items()):
        if m == ref_key:
            continue
        if len(d['y_events']) == len(ref['y_events']) and same(d):
            preds[m] = fix_shape(d)
            continue
        if perm is None:
            perm = _eval1_loader_perm(len(ref['y_events']))
        if len(d['y_events']) == len(ref['y_events']) and same(d, perm):
            preds[m] = fix_shape({k: np.asarray(v).reshape(len(perm), -1)[perm].reshape(np.asarray(v).shape)
                                  for k, v in d.items()})
            print(f"  [bootstrap] {m}: realigned to {ref_key} row order (replayed eval_1 shuffle)")
        else:
            print(f"  [bootstrap] {m}: could NOT realign -- unpaired test for this task")
    return preds

def bootstrap_pvalues(task, models):
    """--stat_test bootstrap: {model: p} for SPARC-P vs each model on this task, from the
    ensemble-prediction pickles eval_1 wrote. Cached per task in <csv_dir>/bootstrap_cache."""
    import pickle
    import bootstrap_stats as bs
    option = f'{task}_risk' if task in SIDE_EFFECT_TASKS else task
    # binary tasks have one AUROC/AUPRC, so test the overall variant
    variant = metric if task in SIDE_EFFECT_TASKS else ('overall_auroc' if metric_is_auroc else 'overall_auprc')
    cache_dir = os.path.join(csv_dir, 'bootstrap_cache')
    summ = bs.load_summary(cache_dir, task, dataset, _finetuned_suffix, modules, args.n_bootstrap, args.boot_seed)
    if summ is None:
        pred_dir = os.path.join(csv_dir, 'ensemble_predictions')
        preds = {}
        for m in ['SPARC-P'] + [x for x in models if x != 'SPARC-P']:
            pth = os.path.join(pred_dir, f'{modules}_{option}_{dataset}{_finetuned_suffix}_{m}.pickle')
            if not os.path.exists(pth):
                print(f"  [bootstrap] missing {os.path.basename(pth)} -- no test for {m}")
                continue
            with open(pth, 'rb') as f:
                preds[m] = pickle.load(f)
        if 'SPARC-P' not in preds:
            print(f"  [bootstrap] {task}: no SPARC-P ensemble pickle -- no tests")
            return {}
        preds = _align_to_reference(preds)
        bs.run_and_cache_task(preds, task=task, option=option, dataset=dataset,
                              finetuned_suffix=_finetuned_suffix, modules=modules,
                              sparc_model='SPARC-P',
                              benchmark_models=[m for m in preds if m != 'SPARC-P'],
                              per_fold_values_by_model={}, fold_mean_by_model={},
                              cache_dir=cache_dir, n_bootstrap=args.n_bootstrap,
                              rseed=args.boot_seed, task_index=ALL_TASKS.index(task),
                              verbose=False)
        summ = bs.load_summary(cache_dir, task, dataset, _finetuned_suffix, modules, args.n_bootstrap, args.boot_seed)
    pv = summ[(summ['row_type'] == 'pvalue') & (summ['metric'] == variant)]
    if len(pv) and (pv['test_type'] != 'paired').any():
        print(f"  [bootstrap] {task}: UNPAIRED fallback used (rows not aligned)")
    out = {r['benchmark']: (None if pd.isna(r['p_value']) else float(r['p_value']))
           for _, r in pv.iterrows()}
    print(f"  [bootstrap] {task} ({variant}): " +
          ", ".join(f"{k}={v:.4g}" if v is not None else f"{k}=n/a" for k, v in out.items()))
    return out

if args.precompute_only:
    if stat_test != 'bootstrap':
        raise SystemExit('--precompute_only needs --stat_test bootstrap')
    _pre = args.task.split(',') if args.task else available_tasks
    for _t in _pre:
        if _t not in per_task_models:
            print(f"  [bootstrap] {_t}: no CSV for {dataset}, skipped")
            continue
        import time as _time; _t0 = _time.time()
        bootstrap_pvalues(_t, per_task_models[_t])
        print(f"  [bootstrap] {_t}: {_time.time() - _t0:.1f}s")
    raise SystemExit(0)

def _delong_components(y, s):
    """DeLong structural components (midrank form) for one score vector.
    Returns (auc, V10 over positives, V01 over negatives)."""
    from scipy.stats import rankdata
    y = np.asarray(y).ravel().astype(bool)
    s = np.asarray(s, dtype=np.float64).ravel()
    x, z = s[y], s[~y]
    m, n = x.size, z.size
    tz = rankdata(np.concatenate([x, z]))
    v10 = (tz[:m] - rankdata(x)) / n
    v01 = 1.0 - (tz[m:] - rankdata(z)) / m
    return v10.mean(), v10, v01


def _delong_diff(y, s1, s2):
    """Paired DeLong: (AUC1 - AUC2, its variance) for two scores on the same rows."""
    a1, p1, q1 = _delong_components(y, s1)
    a2, p2, q2 = _delong_components(y, s2)
    d10, d01 = p1 - p2, q1 - q2
    var = ((np.var(d10, ddof=1) / d10.size) if d10.size > 1 else 0.0) + \
          ((np.var(d01, ddof=1) / d01.size) if d01.size > 1 else 0.0)
    return a1 - a2, var


def delong_pvalues(task, models):
    """Paired DeLong p-values, SPARC-P vs. each model (AUROC only), on the realigned ensemble pickles.

    Weighted AUROC uses DeLong stratified by infusion group; overall AUROC and binary tasks
    use plain DeLong.
    """
    import pickle
    from scipy.stats import norm
    import bootstrap_stats as bs
    if not metric_is_auroc:
        print(f"  [delong] {task}: DeLong is defined for AUROC only -- no stars for {metric}")
        return {}
    option = f'{task}_risk' if task in SIDE_EFFECT_TASKS else task
    weighted = (task in SIDE_EFFECT_TASKS) and metric_is_weighted
    pred_dir = os.path.join(csv_dir, 'ensemble_predictions')
    preds = {}
    for m in ['SPARC-P'] + [x for x in models if x != 'SPARC-P']:
        pth = os.path.join(pred_dir, f'{modules}_{option}_{dataset}{_finetuned_suffix}_{m}.pickle')
        if os.path.exists(pth):
            with open(pth, 'rb') as f:
                preds[m] = pickle.load(f)
        else:
            print(f"  [delong] missing {os.path.basename(pth)} -- no test for {m}")
    if 'SPARC-P' not in preds:
        return {}
    preds = _align_to_reference(preds)
    # two-class infusion groups and their weights, as in the weighted metric
    ctx = bs.prepare_context(preds['SPARC-P'], option)
    if ctx['total_n'] == 0:
        print(f"  [delong] {task}: no usable infusion groups -- no tests")
        return {}
    labels = ctx['labels'].ravel()
    ref = np.asarray(preds['SPARC-P']['y_preds']).ravel()
    out = {}
    for m, d in preds.items():
        if m == 'SPARC-P':
            continue
        other = np.asarray(d['y_preds']).ravel()
        if weighted:
            delta, var = 0.0, 0.0
            for rows, n_g in zip(ctx['group_row_idx'], ctx['group_sizes']):
                w = n_g / ctx['total_n']
                dg, vg = _delong_diff(labels[rows], ref[rows], other[rows])
                delta += w * dg
                var += w * w * vg
        else:
            rows = np.concatenate(ctx['group_row_idx'])
            delta, var = _delong_diff(labels[rows], ref[rows], other[rows])
        out[m] = float(2 * norm.sf(abs(delta) / np.sqrt(var))) if var > 0 else None
    print(f"  [delong] {task} ({'weighted, stratified' if weighted else 'overall'}): " +
          ", ".join(f"{k}={v:.3g}" if v is not None else f"{k}=n/a" for k, v in out.items()))
    return out

def format_pvalue(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < args.alpha:
        return "*"
    else:
        return ""

def apply_correction(pvalues, method='bonferroni'):
    """Bonferroni-correct one task's p-values (None entries kept as None)."""
    if method == 'none':
        return pvalues
    if method != 'bonferroni':
        raise ValueError(f"unsupported correction method: {method!r} "
                         "(expected 'none' or 'bonferroni')")

    valid_pvals = []
    valid_indices = []
    for i, pv in enumerate(pvalues):
        if pv is not None:
            valid_pvals.append(pv)
            valid_indices.append(i)

    if len(valid_pvals) == 0:
        return pvalues

    n = len(valid_pvals)
    corrected = [None] * len(pvalues)
    for i, idx in enumerate(valid_indices):
        corrected[idx] = min(valid_pvals[i] * n, 1.0)

    return corrected

# fixed 3x3 grid
nrows = 3
ncols = 3

max_models_per_task = max(len(per_task_models[t]) for t in available_tasks)
row_height = max(2.2, max_models_per_task * 0.5)
fig_width = 6.2 * ncols
fig_height = row_height * nrows

fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
axes_flat = axes.flatten()

bar_height = 0.6

# --family_style underline: underlines drawn after tight_layout()
axes_needing_underline = []

for idx, task in enumerate(TASK_ORDER):
    ax = axes_flat[idx]

    if task is None:
        ax.axis('off')
        continue

    if task not in task_data:
        ax.axis('off')
        continue

    col_idx = idx % ncols
    is_leftmost = (col_idx == 0)

    if sort_by == 'global':
        task_models = [m for m in global_model_order if m in per_task_models[task]]
    else:
        task_models = sorted(per_task_models[task],
                              key=lambda m: bar_value(task, m))
        task_models = list(reversed(task_models))
    if args.style == 'simple':
        # time-series methods first, then last-timepoint
        _ts = [m for m in task_models if m in TEMPORAL_MODELS]
        task_models = _ts + [m for m in task_models if m not in TEMPORAL_MODELS]
        _n_ts = len(_ts)

    y_positions = np.arange(len(task_models))[::-1]  # best model on top

    # AUROC axis 50-100; AUPRC 0-50 or 50-100 depending on the best fold mean
    if metric_is_auroc:
        xlim_left, xlim_right = 50, 100
    else:
        max_mean = max([np.mean(get_vals(task, m)) for m in task_models
                        if len(get_vals(task, m)) > 0], default=0)
        if max_mean >= 50:
            xlim_left, xlim_right = 50, 100
        else:
            xlim_left, xlim_right = 0, 50

    sig_star_x = xlim_right - 2

    sparc_p_vals = get_vals(task, 'SPARC-P') if 'SPARC-P' in task_models else np.array([])

    # significance vs. SPARC-P
    sig_annotations = {}
    if len(sparc_p_vals) > 0:
        pvalues = []
        model_order = []
        boot_p = (bootstrap_pvalues(task, task_models) if stat_test == 'bootstrap' else
                  delong_pvalues(task, task_models) if stat_test == 'delong' else None)
        for model in task_models:
            if model != 'SPARC-P':
                if boot_p is not None:
                    p_val = boot_p.get(model)
                else:
                    model_vals = get_vals(task, model)
                    p_val = paired_test(sparc_p_vals, model_vals)
                pvalues.append(p_val)
                model_order.append(model)

        corrected_pvalues = apply_correction(pvalues, method=correction_method)

        for model, p_val in zip(model_order, corrected_pvalues):
            sig_annotations[model] = format_pvalue(p_val)

    for y_pos, model in zip(y_positions, task_models):
        vals = get_vals(task, model)
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        xerr_val = std_val
        if ci_mode == 'ensemble':
            ens_val = get_ensemble_val(task, model)
            if ens_val is None:
                print(f"WARNING [{task}] {model}: no ensemble row; drawing fold mean +/- SD instead")
            else:
                mean_val = ens_val
                ci = get_ensemble_ci(task, model)
                if ci is None:
                    print(f"WARNING [{task}] {model}: no ensemble CI for {metric}; bar drawn without error bar")
                    xerr_val = None
                else:
                    xerr_val = [[max(mean_val - ci[0], 0)], [max(ci[1] - mean_val, 0)]]
        color = MODEL_COLORS[model]
        if args.style == 'simple' and model != 'SPARC-P':
            color = SIMPLE_GREY

        is_highlight = (highlight_model is not None and model == highlight_model)
        edge_color = 'black'
        edge_width = 2.2 if is_highlight else 1.0

        ax.barh(y_pos, mean_val, height=bar_height, xerr=xerr_val,
                color=color, edgecolor=edge_color, linewidth=edge_width,
                error_kw={'elinewidth': 1.1, 'capsize': 2.5, 'capthick': 1.1})

        if model in sig_annotations and sig_annotations[model] and sig_annotations[model] != "":
            sig_text = sig_annotations[model]
            ax.text(sig_star_x, y_pos, sig_text,
                    va='center', ha='right', fontsize=9, fontweight='bold', color='black')

        if show_ensemble:
            ens_val = get_ensemble_val(task, model)
            if ens_val is not None:
                ax.plot(ens_val, y_pos, marker='D', markersize=6,
                        color='red', markeredgecolor='darkred',
                        markeredgewidth=1.1, zorder=10)

    if args.style == 'simple' and 0 < _n_ts < len(task_models):
        # rule between the two groups
        ax.axhline((y_positions[_n_ts - 1] + y_positions[_n_ts]) / 2, color='#888888',
                   linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_yticks(y_positions)

    # y labels on the left column only; method family shown by label colour
    if is_leftmost:
        labels = [MODEL_DISPLAY.get(m, m) for m in task_models]
        ax.set_yticklabels(labels, fontsize=10)
        if family_style == 'color':
            for tick_label, m in zip(ax.get_yticklabels(), task_models):
                if m in TEMPORAL_MODELS:
                    tick_label.set_color(TIMESERIES_COLOR)
                elif m in LAST_TP_MODELS:
                    tick_label.set_color(LASTTP_COLOR)
                else:
                    tick_label.set_color('black')
        else:
            # underlines are drawn after the final layout (below)
            for tick_label in ax.get_yticklabels():
                tick_label.set_color('black')
            axes_needing_underline.append((ax, list(task_models)))
    else:
        ax.set_yticklabels([])

    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_xlabel(f'{metric_label_kind} {metric_label_name} (%)', fontsize=10)

    title_str = TASK_DISPLAY.get(task, task)
    if task in task_sample_info:
        info = task_sample_info[task]
        if info['n'] is not None and info['n_pos'] is not None:
            title_str += f"\n(n={int(info['n'])}, n$_{{pos}}$={int(info['n_pos'])})"
    ax.set_title(title_str, fontsize=12, fontweight='bold')

    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

dataset_display = dataset.replace('_', '-')
suptitle_text = (f'Model Performance — {metric_label_kind} {metric_label_name} '
                  f'Across Clinical Tasks ({dataset_display})'
                  + ('\nBars: 10-fold ensemble, error bars: 95% bootstrap CI' if ci_mode == 'ensemble'
                     else '\nBars: mean over 10 folds, error bars: SD across folds'))
fig.suptitle(suptitle_text, fontsize=15, fontweight='bold', y=1.0 + 0.01 * nrows)

# legend: method-family swatches (and the ensemble marker)
if family_style == 'color':
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=TIMESERIES_COLOR,
                   markeredgecolor=TIMESERIES_COLOR, markeredgewidth=0, markersize=9,
                   label='Time series method'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=LASTTP_COLOR,
                   markeredgecolor=LASTTP_COLOR, markeredgewidth=0, markersize=9,
                   label='Last timepoint method'),
    ]
else:
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.4, marker='', markersize=0,
                   label='Time series method (underlined)'),
        plt.Line2D([0], [0], color='white', lw=0, marker='', markersize=0,
                   label='Last timepoint method (plain)'),
    ]

if show_ensemble:
    legend_elements.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markeredgewidth=1.1, markersize=7,
                   label='◆ Ensemble performance')
    )

# legend just below the axes, significance note below it
legend_y_pos = 0.005
_leg = fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, legend_y_pos),
                  ncol=len(legend_elements), fontsize=10, frameon=False)
if family_style == 'color':
    for _txt, _h in zip(_leg.get_texts(), legend_elements):
        _txt.set_color(_h.get_markerfacecolor())

correction_text = ''
if correction_method == 'bonferroni':
    correction_text = '; Bonferroni correction applied within each task'

_test_label = {'ttest_rel': 'paired t-test',
               'wilcoxon': 'paired Wilcoxon signed-rank',
               'bootstrap': f'paired stratified bootstrap of ensemble predictions, {args.n_bootstrap} resamples',
               'delong': ('paired DeLong on ensemble predictions, stratified by infusion group'
                          if metric_is_weighted else 'paired DeLong on ensemble predictions')}[stat_test]
sig_note = f'*p<{args.alpha:g}, **p<0.01, ***p<0.001 ({_test_label} vs. SPARC{correction_text}; non-significant comparisons not shown)'
sig_y_pos = -0.045
fig.text(0.5, sig_y_pos, sig_note,
         ha='center', fontsize=8, style='italic', color='dimgray')

plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.3)

# --family_style underline: draw the rules by hand once label extents are final
if family_style == 'underline' and axes_needing_underline:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ax, _models in axes_needing_underline:
        for _tick_label, _m in zip(_ax.get_yticklabels(), _models):
            if _m not in TEMPORAL_MODELS:
                continue
            bb = _tick_label.get_window_extent(renderer=renderer)
            bb_ax = bb.transformed(_ax.transAxes.inverted())
            # just under the baseline
            y = bb_ax.y0 - 0.006
            _ax.plot([bb_ax.x0, bb_ax.x1], [y, y],
                     transform=_ax.transAxes, color='black',
                     linewidth=0.9, clip_on=False, zorder=5)

fn_suffix = '_noFN' if exclude_fn else ''
ensemble_suffix = '_ensembleCI' if ci_mode == 'ensemble' else ('_with_ensemble' if show_ensemble else '')
highlight_suffix = f'_hl-{highlight_model}' if highlight_model else ''
correction_suffix = f'_{correction_method}' if correction_method != 'none' else ''
# non-default tests are tagged in the filename
test_suffix = f'_{stat_test}' if stat_test != 'ttest_rel' else ''
style_suffix = (('_underline' if family_style == 'underline' else '') + ('_simple' if args.style == 'simple' else '')
                + (f'_alpha{args.alpha:g}' if args.alpha != 0.05 else ''))
out_pdf = (f'{save_dir}/all_models_grid_{modules}_{dataset}{_finetuned_suffix}'
           f'{ensemble_suffix}{fn_suffix}{highlight_suffix}{test_suffix}{correction_suffix}{style_suffix}_{metric}.pdf')
out_png = (f'{save_dir}/all_models_grid_{modules}_{dataset}{_finetuned_suffix}'
           f'{ensemble_suffix}{fn_suffix}{highlight_suffix}{test_suffix}{correction_suffix}{style_suffix}_{metric}.png')

fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=150)
plt.close()

print("Figure saved:")
print(f"  {out_pdf}")
print(f"  {out_png}\n")
print("Done.")