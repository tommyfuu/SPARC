"""Model comparison across tasks: one subplot per task, one horizontal bar per model.

Models are discovered from each task's CSV rather than hard-coded, ranked by
performance, with mean +/- std over folds and an optional ensemble diamond.

Metric (--metric): weighted_auroc | overall_auroc | weighted_auprc | overall_auprc.
Side-effect tasks read test_auroc_nextinf_* / test_auprc_nextinf_*; binary tasks
read auroc / auprc.

Significance (--stat_test) is against SPARC-P, paired across folds; ttest_rel by
default, wilcoxon selectable, --correction bonferroni applies within each task.
At n=10 the smallest attainable two-sided p is 0.00195, so the strongest
comparisons all pin there -- read the effect size, not the stars.

Reads the CSVs from eval_1_across_models_comparison_ensemble_FP_fixed.py.

Usage:
    python vis_eval_1_across_tasks_t1e_controlled_bar_horizontal_succinct_wilcoxon_all.py --dataset MSK_CAIA
    python vis_eval_1_across_tasks_t1e_controlled_bar_horizontal_succinct_wilcoxon_all.py --metric weighted_auprc --stat_test wilcoxon
"""

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

# Argument parsing
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
parser.add_argument('--ensemble', dest='ensemble', action='store_true', default=True,
                     help='Show ensemble performance as red diamond markers (default: on)')
parser.add_argument('--no-ensemble', dest='ensemble', action='store_false',
                     help='Hide ensemble markers')
parser.add_argument('--highlight', type=str, default='',
                     help="Model to visually highlight (thicker black outline) in every "
                          "subplot. Default is '' (no highlight) — SPARC is no longer "
                          "emphasised by default so all models are drawn identically. "
                          "Use the internal model name, e.g. 'SPARC-P'.")
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
parser.add_argument('--results_subdir', type=str, default='results_vis_060726',
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
                     choices=['ttest_rel', 'wilcoxon'],
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
                          "The chosen test is appended to the output filename.")
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
show_ensemble = args.ensemble
highlight_model = args.highlight.strip() if args.highlight else None
sort_by = args.sort_by
ncols_req = args.ncols
correction_method = args.correction
stat_test = args.stat_test
family_style = args.family_style

# Configuration
# Root holding the results_vis_* trees. The original hardcoded value pointed at
# /data/morrisq (the old lilac mount); on hosts that mount it elsewhere set
# --model_path or the SPARC_MODEL_PATH environment variable.
DEFAULT_MODEL_PATH = '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models_060726'

all_model_path = (args.model_path
                  or os.environ.get('SPARC_MODEL_PATH')
                  or DEFAULT_MODEL_PATH)

# --csv_dir wins outright; otherwise assemble it from the parts so a batch
# rerun can be plotted by passing only --results_subdir.
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

# Custom task order for the figure layout
# Row 1: AKI, anemia, febrile_neutropenia
# Row 2: neutropenia, Liver_toxicity
# Row 3: binary, binary_icu_pre
if exclude_fn:
    # If excluding FN, shift layout: Row 1 has AKI, anemia, None
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

TASK_DISPLAY = {
    'AKI':                 'Acute kidney injury',
    'Liver_toxicity':      'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia':              'Severe anemia',
    'neutropenia':         'Neutropenia',
    'binary':              '180D Hosp. mort.',
    'binary_icu_pre':      '30D ICU mort.',
}

# Friendly display names for known models. Any model not listed here falls
# back to its raw name, so new/unexpected models still plot correctly.
MODEL_DISPLAY = {
    'SPARC-P':          'SPARC',  # SPARC-P is now called SPARC for publication
    'lasttp_MLP':       'MLP',
    'lasttp_xgboost':   'XGBoost',
    'temporal_xgboost': 'T-XGBoost',
    'temporal_tabpfn':  'TabPFN',
    'rnn':              'RNN',
    'duett':            'DuETT',
    'raindrop':         'Raindrop',
}

# Differentiate time-series vs. last-timepoint methods.
# (TEMPORAL_MODELS keeps its name for continuity with the rest of the codebase;
# the user-facing wording is "time series".)
TEMPORAL_MODELS = {'SPARC-P', 'temporal_xgboost', 'temporal_tabpfn', 'rnn', 'duett', 'raindrop'}
LAST_TP_MODELS = {'lasttp_MLP', 'lasttp_xgboost'}

# Method family is shown by the colour of the model's y-axis label.
#
# Okabe-Ito blue paired with a darkened vermillion. Simulating the three main
# colour-vision deficiencies, the RGB separation between the two is:
#
#                      normal  protan  deutan  tritan   contrast on white
#   teal/amber (old)    0.660   0.360   0.467   0.581      6.0 / 5.1
#   blue/vermillion     1.008   0.698   0.719   0.780      5.2 / 5.3
#
# The old teal/amber pair degraded to two near-identical greys under
# protanopia (#65656E vs #6D6D13). Vermillion is darkened from the stock
# Okabe-Ito #D55E00 so it clears the WCAG 4.5:1 text-contrast floor (#D55E00
# is only 3.9:1 on white).
TIMESERIES_COLOR = '#0072B2'   # blue       -> time series methods
LASTTP_COLOR     = '#B54A00'   # vermillion -> last-timepoint methods

# Load CSVs
# Prefer _ensemble.csv files when they exist (they have correct DUETT data)
_finetuned_suffix = '_finetuned' if dataset == 'MIMICIV_hosp' and not zeroshot else ''

task_data = {}       # fold-level rows only, per task
task_data_full = {}  # full rows (incl. ensemble), per task
task_sample_info = {}  # n_test and n_pos_next_infusion per task

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

    # Extract n_test and n_pos_next_infusion for this task
    if len(df_full) > 0:
        n_test = df_full['n_test'].iloc[0] if 'n_test' in df_full.columns else None
        # For side-effect tasks, use n_pos_next_infusion; for binary tasks, use n_pos_overall
        if task in SIDE_EFFECT_TASKS and 'n_pos_next_infusion' in df_full.columns:
            n_pos = df_full['n_pos_next_infusion'].iloc[0]
        elif 'n_pos_overall' in df_full.columns:
            n_pos = df_full['n_pos_overall'].iloc[0]
        else:
            n_pos = None
        task_sample_info[task] = {'n': n_test, 'n_pos': n_pos}

available_tasks = ALL_TASKS  # Use the custom order

if len(available_tasks) == 0:
    raise RuntimeError("No task CSVs found. Run eval_1_across_models_comparison.py first "
                        "(or pass --csv_dir to point at your data).")

print(f"\nCSV directory: {csv_dir}")
print(f"Loaded {len(available_tasks)} tasks: {available_tasks}")
print(f"Metric: {metric} ({metric_label_kind} {metric_label_name})\n")


# Helper functions
def get_vals(task, model_key):
    """Fold-level metric values (percent) for a model on a task."""
    col = METRIC_COL[task]
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty or col not in rows.columns:
        return np.array([])
    return rows[col].dropna().values * 100


def get_ensemble_val(task, model_key):
    """Single ensemble performance value (percent), or None if unavailable."""
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


def discover_models(task):
    """All base model names present in a task's fold-level data (excludes
    rows that are themselves ensemble summary rows, e.g. 'SPARC-P_ensemble').
    Also excludes 'SPARC' (non-parsimonious version) per publication requirements."""
    df = task_data.get(task)
    if df is None or 'model' not in df.columns:
        return []
    models = [m for m in df['model'].unique()
              if not str(m).endswith('_ensemble') and m != 'SPARC']
    return models


# Build a master model list + fixed color map (consistent across subplots)
per_task_models = {task: discover_models(task) for task in available_tasks}
all_models_seen = sorted(set(m for models in per_task_models.values() for m in models))


# Rank models globally by their mean value across all tasks where they appear
model_means = {}
for m in all_models_seen:
    vals = []
    for task in available_tasks:
        if m in per_task_models[task]:
            v = get_vals(task, m)
            if len(v) > 0:
                vals.append(np.mean(v))
    model_means[m] = np.mean(vals) if vals else -np.inf

global_model_order = sorted(all_models_seen, key=lambda m: model_means[m], reverse=True)

# Assign a fixed color to every model using a professional color palette
# Professional color palette with distinct, formal colors
PROFESSIONAL_COLORS = [
    '#2C5F8D',  # Deep blue
    '#E07A5F',  # Terracotta
    '#81B29A',  # Sage green
    '#F2CC8F',  # Sand
    '#6A4C93',  # Deep purple
    '#D96459',  # Muted red
    '#5A7D7C',  # Teal
    '#B68D40',  # Amber
    '#4A6FA5',  # Medium blue
    '#A87C7C',  # Mauve
]

# Assign colors to models, cycling through the palette if needed
MODEL_COLORS = {}
for i, m in enumerate(global_model_order):
    MODEL_COLORS[m] = PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)]

print("Models discovered (ranked by mean performance across available tasks):")
for m in global_model_order:
    disp = MODEL_DISPLAY.get(m, m)
    mean_val = model_means[m]
    temporal_marker = " [time series]" if m in TEMPORAL_MODELS else " [last timepoint]" if m in LAST_TP_MODELS else ""
    print(f"  {disp:12s} ({m:20s}): mean = {mean_val:.2f}%{temporal_marker}" if np.isfinite(mean_val)
          else f"  {disp:12s} ({m:20s}): mean = n/a{temporal_marker}")
print()

# Statistical significance testing function
def paired_test(vals1, vals2):
    """
    Paired test between two sets of fold-level values, selected by --stat_test.
    Returns p-value, or None if the test cannot be performed.

    Both tests are PAIRED: vals1[i] and vals2[i] are the same CV fold, so the
    fold-to-fold component is shared and belongs inside the pair, not in the
    between-group spread.
    """
    if len(vals1) == 0 or len(vals2) == 0:
        return None
    if len(vals1) != len(vals2):
        return None
    try:
        if stat_test == 'wilcoxon':
            # signed-rank, the paired form. (Rank-SUM / Mann-Whitney is the
            # unpaired one and would be the wrong test for this data.)
            import numpy as _np
            if _np.allclose(_np.asarray(vals1) - _np.asarray(vals2), 0):
                return None       # wilcoxon raises when every difference is zero
            stat_v, p_value = stats.wilcoxon(vals1, vals2, alternative='two-sided')
        else:
            stat_v, p_value = stats.ttest_rel(vals1, vals2, alternative='two-sided')
        return p_value
    except Exception:
        return None

def format_pvalue(p):
    """Format p-value for display."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""  # Don't show non-significant results

def apply_correction(pvalues, method='bonferroni'):
    """
    Apply multiple testing correction to a list of p-values.

    Scope: the caller passes only the p-values for ONE task (every model in
    that subplot compared against the reference model), so the correction
    family is per-task, not the whole figure. A model appearing in several
    tasks is therefore corrected independently in each.

    Args:
        pvalues: List of p-values (can contain None)
        method: 'none' or 'bonferroni'

    Returns:
        List of corrected p-values (same length as input)
    """
    if method == 'none':
        return pvalues
    if method != 'bonferroni':
        raise ValueError(f"unsupported correction method: {method!r} "
                         "(expected 'none' or 'bonferroni')")

    # Extract non-None p-values and their indices
    valid_pvals = []
    valid_indices = []
    for i, pv in enumerate(pvalues):
        if pv is not None:
            valid_pvals.append(pv)
            valid_indices.append(i)

    if len(valid_pvals) == 0:
        return pvalues

    # Bonferroni: multiply by the number of tests within this task
    n = len(valid_pvals)
    corrected = [None] * len(pvalues)
    for i, idx in enumerate(valid_indices):
        corrected[idx] = min(valid_pvals[i] * n, 1.0)

    return corrected

# Figure layout: fixed 3x3 grid for custom task layout
nrows = 3
ncols = 3

# Height scales with the largest number of models any single subplot needs
max_models_per_task = max(len(per_task_models[t]) for t in available_tasks)
row_height = max(2.2, max_models_per_task * 0.5)
fig_width = 6.2 * ncols
fig_height = row_height * nrows

fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
axes_flat = axes.flatten()

bar_height = 0.6

# Collected when --family_style underline; consumed after tight_layout().
axes_needing_underline = []

for idx, task in enumerate(TASK_ORDER):
    ax = axes_flat[idx]

    # Skip if this is an empty slot
    if task is None:
        ax.axis('off')
        continue

    # Skip if task data not available
    if task not in task_data:
        ax.axis('off')
        continue

    # Determine if this subplot is in the leftmost column
    col_idx = idx % ncols
    is_leftmost = (col_idx == 0)

    if sort_by == 'global':
        task_models = [m for m in global_model_order if m in per_task_models[task]]
    else:
        task_models = sorted(per_task_models[task],
                              key=lambda m: np.mean(get_vals(task, m)) if len(get_vals(task, m)) else -np.inf)
        task_models = list(reversed(task_models))

    y_positions = np.arange(len(task_models))[::-1]  # best model on top

    # Determine x-axis limits first (needed for positioning significance stars)
    # AUROC: always 50-100
    # AUPRC: 0-50 or 50-100 based on max mean value for this task
    if metric_is_auroc:
        xlim_left, xlim_right = 50, 100
    else:
        # For AUPRC, determine range based on max mean
        max_mean = max([np.mean(get_vals(task, m)) for m in task_models
                        if len(get_vals(task, m)) > 0], default=0)
        if max_mean >= 50:
            xlim_left, xlim_right = 50, 100
        else:
            xlim_left, xlim_right = 0, 50

    # Position for significance stars: slightly left of right edge
    sig_star_x = xlim_right - 2

    # Get SPARC-P values for statistical testing
    sparc_p_vals = get_vals(task, 'SPARC-P') if 'SPARC-P' in task_models else np.array([])

    # Calculate statistical significance for SPARC-P vs. all other models
    sig_annotations = {}
    if len(sparc_p_vals) > 0:
        # Collect all p-values first
        pvalues = []
        model_order = []
        for model in task_models:
            if model != 'SPARC-P':
                model_vals = get_vals(task, model)
                p_val = paired_test(sparc_p_vals, model_vals)
                pvalues.append(p_val)
                model_order.append(model)

        # Apply correction if requested
        corrected_pvalues = apply_correction(pvalues, method=correction_method)

        # Format corrected p-values
        for model, p_val in zip(model_order, corrected_pvalues):
            sig_annotations[model] = format_pvalue(p_val)

    for y_pos, model in zip(y_positions, task_models):
        vals = get_vals(task, model)
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        color = MODEL_COLORS[model]

        is_highlight = (highlight_model is not None and model == highlight_model)
        edge_color = 'black'
        edge_width = 2.2 if is_highlight else 1.0

        ax.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                color=color, edgecolor=edge_color, linewidth=edge_width,
                error_kw={'elinewidth': 1.1, 'capsize': 2.5, 'capthick': 1.1})

        # Add statistical significance annotation if available
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

    ax.set_yticks(y_positions)

    # Only show y-axis labels on leftmost column, with temporal/last-tp markers
    if is_leftmost:
        # Method family is encoded in the TEXT COLOUR of the tick label rather
        # than a glyph prefix: it keeps the labels flush-left (no ragged
        # indent from the marker) and reads at a glance.
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
            # 'underline': labels stay black; the rules are drawn after layout
            # is final (see UNDERLINE PASS below), because a text extent is only
            # known once the figure has been rendered.
            for tick_label in ax.get_yticklabels():
                tick_label.set_color('black')
            axes_needing_underline.append((ax, list(task_models)))
    else:
        ax.set_yticklabels([])

    # Set x-axis limits (already calculated earlier)
    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_xlabel(f'{metric_label_kind} {metric_label_name} (%)', fontsize=10)

    # Build title with sample info
    title_str = TASK_DISPLAY.get(task, task)
    if task in task_sample_info:
        info = task_sample_info[task]
        if info['n'] is not None and info['n_pos'] is not None:
            title_str += f"\n(n={int(info['n'])}, n$_{{pos}}$={int(info['n_pos'])})"
    ax.set_title(title_str, fontsize=12, fontweight='bold')

    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Overall title
dataset_display = dataset.replace('_', '-')
suptitle_text = (f'Model Performance — {metric_label_kind} {metric_label_name} '
                  f'Across Clinical Tasks ({dataset_display})')
fig.suptitle(suptitle_text, fontsize=15, fontweight='bold', y=1.0 + 0.01 * nrows)

# Create legend entries for temporal indicators and ensemble
# Colour swatches matching the y-axis label colours (see TIMESERIES_COLOR /
# LASTTP_COLOR): the legend text itself is drawn in the family colour.
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
    # Underline variant: a short rule stands in for the swatch.
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

# Add legend to the figure with better positioning
# Legend sits just below the axes; the significance note goes further below
# it. These were 0.00 / -0.025, which overlapped once the note grew to
# mention the per-task correction.
legend_y_pos = 0.005
_leg = fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, legend_y_pos),
                  ncol=len(legend_elements), fontsize=10, frameon=False)
# Draw each legend label in its own family colour so the legend and the
# y-axis labels use exactly the same encoding.
if family_style == 'color':
    for _txt, _h in zip(_leg.get_texts(), legend_elements):
        _txt.set_color(_h.get_markerfacecolor())

# Add note about statistical significance with better spacing to avoid overlap
correction_text = ''
if correction_method == 'bonferroni':
    correction_text = '; Bonferroni correction applied within each task'

_test_label = {'ttest_rel': 'paired t-test',
               'wilcoxon': 'paired Wilcoxon signed-rank'}[stat_test]
sig_note = f'*p<0.05, **p<0.01, ***p<0.001 ({_test_label} vs. SPARC{correction_text}; non-significant comparisons not shown)'
sig_y_pos = -0.045
fig.text(0.5, sig_y_pos, sig_note,
         ha='center', fontsize=8, style='italic', color='dimgray')

plt.tight_layout()
# Reduce space between subplots
plt.subplots_adjust(wspace=0.15, hspace=0.3)

# UNDERLINE PASS (--family_style underline)
# matplotlib has no text-underline property, so the rule is drawn manually.
# It must happen after tight_layout()/subplots_adjust(), because a label's
# extent is only meaningful once the final layout has been rendered.
if family_style == 'underline' and axes_needing_underline:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ax, _models in axes_needing_underline:
        for _tick_label, _m in zip(_ax.get_yticklabels(), _models):
            if _m not in TEMPORAL_MODELS:
                continue
            bb = _tick_label.get_window_extent(renderer=renderer)
            bb_ax = bb.transformed(_ax.transAxes.inverted())
            # Sit the rule just under the glyph baseline; a little inset on
            # each side so it tracks the text rather than its padding box.
            y = bb_ax.y0 - 0.006
            _ax.plot([bb_ax.x0, bb_ax.x1], [y, y],
                     transform=_ax.transAxes, color='black',
                     linewidth=0.9, clip_on=False, zorder=5)

# Save figure
fn_suffix = '_noFN' if exclude_fn else ''
ensemble_suffix = '_with_ensemble' if show_ensemble else ''
highlight_suffix = f'_hl-{highlight_model}' if highlight_model else ''
correction_suffix = f'_{correction_method}' if correction_method != 'none' else ''
# non-default test is tagged in the filename so the two versions cannot
# overwrite each other; the default keeps the established name.
test_suffix = f'_{stat_test}' if stat_test != 'ttest_rel' else ''
style_suffix = '_underline' if family_style == 'underline' else ''
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