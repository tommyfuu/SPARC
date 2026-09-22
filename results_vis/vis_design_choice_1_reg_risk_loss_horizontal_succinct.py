"""
Loss-function comparison (vanilla / hybrid / cox) across the 5 side-effect tasks.

Two figures, one per metric: weighted AUROC and weighted C-index. Bars are mean
+/- std over folds, with Bonferroni-corrected significance across tasks (--stat_test, default
paired Wilcoxon signed-rank).

Reads the CSVs from design_choice_comparison_1_reg_risk_loss.py.

Usage:
    python vis_design_choice_1_reg_risk_loss_horizontal_succinct.py
    python vis_design_choice_1_reg_risk_loss_horizontal_succinct.py /path/to/design_choices_1_regvscox
"""

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


# Both arms are scored on the same folds, so the comparison is paired; earlier
# versions of this script used an unpaired test. At n=10 the smallest attainable
# two-sided p is 0.00195, so the strongest comparisons all pin there -- read the
# deltas, not the star tiers. The old unpaired tests stay selectable via -st.
def paired_test(vals1, vals2):
    """Two-sided p for vals1 vs vals2 per STAT_TEST; nan when not computable.

    nan rather than None so callers doing `min(p * n, 1.0)` stay safe.
    """
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
parser = argparse.ArgumentParser(description='Succinct horizontal bar chart: loss function comparison')
parser.add_argument('--dataset', type=str, default='MSK',
                    help='Dataset name (default: MSK)')
parser.add_argument('--modules', type=str, default='parsimonious',
                    help='Module category (default: parsimonious)')
parser.add_argument('csv_dir_pos', nargs='?', default=None, metavar='CSV_DIR',
                    help='Directory holding the SPARC_reg_risk_comparison_*.csv '
                         'files. Positional shorthand for --csv_dir.')
parser.add_argument('--model_path', type=str, default=None,
                    help='Root holding the results_vis_* trees. Defaults to '
                         '$SPARC_MODEL_PATH, else the path in DEFAULT_MODEL_PATH.')
parser.add_argument('--results_subdir', type=str, default='results_vis_060726',
                    help='Which results tree under --model_path to read '
                         '(default: %(default)s). Use e.g. '
                         'results_vis_060726_batch_091426 for a batch rerun.')
parser.add_argument('--design_subdir', type=str, default='design_choices_1_regvscox',
                    help='Subdirectory of the results tree holding the CSVs '
                         '(default: %(default)s)')
parser.add_argument('--csv_dir', type=str, default=None,
                    help='Override the CSV input directory outright, ignoring '
                         '--model_path/--results_subdir/--design_subdir')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Override the output directory (defaults to csv_dir)')
parser.add_argument('--stat_test', type=str, default='wilcoxon',
                    choices=['wilcoxon', 'ttest_rel', 'mannwhitney', 'ttest_ind'],
                    help="Significance test. 'wilcoxon' (default) and 'ttest_rel' are PAIRED, "
                         "matching the design: both arms use the same folds. 'mannwhitney' and "
                         "'ttest_ind' are the UNPAIRED tests this script used previously, kept "
                         "to reproduce the old figures. A non-default choice is appended to the "
                         "output filename.")
args = parser.parse_args()
STAT_TEST = args.stat_test

dataset = args.dataset
modules = args.modules

# Configuration
# Root holding the results_vis_* trees. The original hardcoded value pointed at
# /data/morrisq (the old lilac mount); on hosts that mount it elsewhere set
# --model_path or the SPARC_MODEL_PATH environment variable.
DEFAULT_MODEL_PATH = '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models_060726'

all_model_path = (args.model_path
                  or os.environ.get('SPARC_MODEL_PATH')
                  or DEFAULT_MODEL_PATH)

# The positional argument and --csv_dir both win outright; otherwise assemble the
# path from its parts so a batch rerun can be plotted by passing only
# --results_subdir.
default_csv_dir = os.path.join(all_model_path, args.results_subdir, args.design_subdir)
csv_dir = args.csv_dir_pos or args.csv_dir or default_csv_dir
save_dir = args.save_dir if args.save_dir else csv_dir

if not os.path.isdir(csv_dir):
    raise SystemExit(
        f"ERROR: CSV directory not found:\n  {csv_dir}\n"
        f"Pass the directory as the first argument, or set --model_path / "
        f"--results_subdir / --design_subdir, to point at the directory holding "
        f"SPARC_reg_risk_comparison_*.csv"
    )

os.makedirs(save_dir, exist_ok=True)
print(f"Reading CSVs from : {csv_dir}")
print(f"Writing output to : {save_dir}")

# Define the 5 side effect tasks
SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']

TASK_DISPLAY = {
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
}

# Define models to compare
LOSS_MODELS = ['SPARC_vanilla', 'SPARC_hybrid', 'SPARC_cox']

LOSS_DISPLAY = {
    'SPARC_vanilla': 'Vanilla Loss',
    'SPARC_hybrid': 'Hybrid Loss',
    'SPARC_cox': 'Cox Loss'
}

# Encoding: fill = loss function, contour weight = chosen model
# Loss functions form an ordered progression (pure BCE -> BCE+Cox -> pure Cox),
# so they are drawn as one blue ramp light->dark rather than as unrelated hues.
# Steps 250/450/650 of the sequential blue; validated as an ordinal ramp
# (monotone lightness, adjacent dL >= 0.06, light end 2.06:1 on the surface).
#
# A model's colour is keyed to the model, not to its position in the group.
LOSS_COLOR = {
    'SPARC_vanilla': '#86b6ef',   # blue 250 - lightest
    'SPARC_hybrid':  '#2a78d6',   # blue 450
    'SPARC_cox':     '#104281',   # blue 650 - darkest
}

# The chosen model for a task is marked by a heavy contour instead of a separate
# fill colour. The contour reads against the white background at the bar's outer
# edge, so it stays legible even on the darkest step.
CHOSEN_EDGE_COLOR = '#0b0b0b'      # text-primary ink
CHOSEN_EDGE_WIDTH = 2.6
PLAIN_EDGE_COLOR = '#52514e'       # text-secondary ink
PLAIN_EDGE_WIDTH = 0.7

# Task-specific choice:
# AKI and Liver_toxicity: Cox Loss is chosen
# febrile_neutropenia, anemia, neutropenia: Hybrid Loss is chosen
def get_chosen_model(task):
    """Which loss function is the chosen one for this task"""
    if task in ['AKI', 'Liver_toxicity']:
        return 'SPARC_cox'
    else:  # febrile_neutropenia, anemia, neutropenia
        return 'SPARC_hybrid'

def get_bar_style(task, model):
    """Fill colour, edge colour and edge width for one bar"""
    chosen = model == get_chosen_model(task)
    return (LOSS_COLOR[model],
            CHOSEN_EDGE_COLOR if chosen else PLAIN_EDGE_COLOR,
            CHOSEN_EDGE_WIDTH if chosen else PLAIN_EDGE_WIDTH)

# Load CSVs
task_data = {}

for task in SIDE_EFFECT_TASKS:
    csv_path = f'{csv_dir}/SPARC_reg_risk_comparison_{modules}_{task}_{dataset}.csv'

    if not os.path.exists(csv_path):
        print(f"WARNING: missing CSV for {task} — {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    task_data[task] = df

available_tasks = [t for t in SIDE_EFFECT_TASKS if t in task_data]
n_tasks = len(available_tasks)

if n_tasks == 0:
    raise RuntimeError("No task CSVs found. Run design_choice_comparison_1_reg_risk_loss.py first.")

print(f"\nLoaded {n_tasks} tasks: {available_tasks}")
print(f"Plotting {len(LOSS_MODELS)} loss functions: {[LOSS_DISPLAY[m] for m in LOSS_MODELS]}\n")

# Data summary export (aggregate the per-task CSVs into one)
build_design_choice_summary(
    design_choice='design_choice_1_loss',
    task_data=task_data,
    available_tasks=available_tasks,
    models=LOSS_MODELS,
    metrics={
        'weighted_auroc': 'test_auroc_nextinf_weighted',
        'weighted_cindex': 'test_cindex_weighted',
    },
    save_dir=save_dir,
    out_stem=f'loss_comparison_{modules}_{dataset}',
    task_display=TASK_DISPLAY,
    model_display=LOSS_DISPLAY,
    to_percent=True,
    extra_cols=['description'],
)

# Bonferroni-corrected significance thresholds
# α = 0.05 / n_tasks
alpha = 0.05
alpha_corr = alpha / n_tasks
print(f"Bonferroni-corrected α = {alpha}/{n_tasks} = {alpha_corr:.4f}\n")

# Helper functions
def get_vals(task, model_key, metric_col):
    """Get fold-level metric values for a model on a task"""
    df = task_data.get(task)
    if df is None:
        return np.array([])
    rows = df[df['model'] == model_key]
    if rows.empty or metric_col not in rows.columns:
        return np.array([])
    return rows[metric_col].dropna().values * 100   # → percent

# Create figure function for each metric
def create_horizontal_barchart(metric_col, metric_name, metric_suffix):
    """Create a horizontal bar chart for a specific metric"""

    n_models = len(LOSS_MODELS)
    bar_pitch = 0.45   # Centre-to-centre spacing of bars within a task
    bar_height = 0.38  # Drawn thickness; the shortfall is the gap between bars,
                       # which separates the fills so the heavy contour on the
                       # chosen bar reads even on the darkest blue step
    task_gap = 0.2     # Minimal gap between task groups

    # Build y-axis positions
    y_positions = {}  # {(task, model): y_pos}
    task_centers = {}  # {task: center_y}

    current_y = 0
    for task in reversed(available_tasks):  # Reverse so first task is at top
        # Calculate positions for this task's bars
        task_y_positions = []
        for i, model in enumerate(LOSS_MODELS):
            # Reverse position so first model (Vanilla) is on top
            y_pos = current_y + (n_models - 1 - i) * bar_pitch
            y_positions[(task, model)] = y_pos
            task_y_positions.append(y_pos)

        # Store center position for task label
        task_centers[task] = np.mean(task_y_positions)

        # Move to next task group
        current_y += n_models * bar_pitch + task_gap

    # Create figure
    fig_height = max(6, n_tasks * 1.2)  # Compact vertical layout
    fig_width = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Collect data and plot bars
    for task in available_tasks:
        for model in LOSS_MODELS:
            vals = get_vals(task, model, metric_col)
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                y_pos = y_positions[(task, model)]
                color, edge_color, edge_width = get_bar_style(task, model)

                # Plot bar
                ax.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                       color=color, edgecolor=edge_color, linewidth=edge_width,
                       zorder=2,
                       error_kw={'elinewidth': 1.1, 'capsize': 3, 'capthick': 1.1,
                                 'ecolor': '#0b0b0b', 'zorder': 3})

    # Bonferroni-corrected significance testing
    print("\n" + "="*70)
    print(f"SIGNIFICANCE TESTING - {metric_name}")
    print("="*70)

    # Determine which comparisons to make
    # Three-way: Vanilla vs Hybrid, Hybrid vs Cox, Vanilla vs Cox
    comparisons = [
        ('SPARC_vanilla', 'SPARC_hybrid', 1.5, 0.5),
        ('SPARC_hybrid', 'SPARC_cox', 4.5, 0.5),
        ('SPARC_vanilla', 'SPARC_cox', 7.5, 0.5),
    ]

    for task in available_tasks:
        for model1, model2, x_offset, text_offset in comparisons:
            vals1 = get_vals(task, model1, metric_col)
            vals2 = get_vals(task, model2, metric_col)

            if len(vals1) > 1 and len(vals2) > 1:
                p_raw = paired_test(vals1, vals2)
                p_corrected = min(p_raw * n_tasks, 1.0)

                # Determine significance level
                if p_corrected < alpha:
                    star = '***' if p_corrected < 0.001 else ('**' if p_corrected < 0.01 else '*')
                else:
                    star = 'NS'

                print(f"  {TASK_DISPLAY[task]:25s} {LOSS_DISPLAY[model1]:15s} vs {LOSS_DISPLAY[model2]:15s}: "
                      f"p_raw={p_raw:.4f}, p_Bonf={p_corrected:.4f} → {star}")

                # Get bar positions
                y1 = y_positions[(task, model1)]
                y2 = y_positions[(task, model2)]

                # Calculate x position for bracket (to the right of the longer bar)
                mean1 = np.mean(vals1)
                std1 = np.std(vals1)
                mean2 = np.mean(vals2)
                std2 = np.std(vals2)

                # Use offset to avoid overlapping lines
                x_right = max(mean1 + std1, mean2 + std2) + x_offset

                # Draw vertical line connecting the two bars
                ax.plot([x_right, x_right], [y1, y2], 'k-', linewidth=1,
                       alpha=0.5 if star == 'NS' else 1.0)

                # Add significance text
                y_mid = (y1 + y2) / 2
                ax.text(x_right + text_offset, y_mid, star,
                       ha='left', va='center', fontsize=9,
                       alpha=0.6 if star == 'NS' else 1.0)

    print("="*70 + "\n")

    # Set y-axis labels (task names)
    y_ticks = [task_centers[task] for task in reversed(available_tasks)]
    y_labels = [TASK_DISPLAY[task] for task in reversed(available_tasks)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)

    # Set x-axis
    ax.set_xlabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
    x_max = 108  # Room for the three significance brackets
    ax.set_xlim(left=50, right=x_max)

    # Title
    dataset_display = dataset.replace('_', '-')
    title_text = f'Loss Function Comparison — {metric_name} — {dataset_display}'
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)

    # Grid and spines
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend: one swatch per loss function, plus the contour convention.
    # Placed below the axes - the bars run the full width, so any in-axes
    # position covers data or the significance brackets.
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=LOSS_COLOR[m], edgecolor=PLAIN_EDGE_COLOR,
                  linewidth=PLAIN_EDGE_WIDTH, label=LOSS_DISPLAY[m])
        for m in LOSS_MODELS
    ]
    legend_elements.append(
        Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=CHOSEN_EDGE_COLOR,
                  linewidth=CHOSEN_EDGE_WIDTH, label='Chosen for this task')
    )

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.11), ncol=len(legend_elements),
              fontsize=9.5, frameon=False, handlelength=1.5,
              handletextpad=0.6, columnspacing=1.8)

    plt.tight_layout()

    # Save figure
    test_suffix = '_' + STAT_TEST if STAT_TEST != 'wilcoxon' else ''
    out_pdf = f'{save_dir}/loss_comparison_{metric_suffix}_{modules}_{dataset}{test_suffix}_horizontal_succinct.pdf'
    out_png = f'{save_dir}/loss_comparison_{metric_suffix}_{modules}_{dataset}{test_suffix}_horizontal_succinct.png'

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFigure saved:")
    print(f"  {out_pdf}")
    print(f"  {out_png}\n")

# Create both figures (AUROC and C-index)

# Create AUROC figure
create_horizontal_barchart('test_auroc_nextinf_weighted', 'Weighted AUROC', 'auroc')

# Create C-index figure
create_horizontal_barchart('test_cindex_weighted', 'Weighted C-index', 'cindex')

print("Done.")
