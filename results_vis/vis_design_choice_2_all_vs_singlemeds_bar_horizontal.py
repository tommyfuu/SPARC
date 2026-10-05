"""All-meds vs single-med models per medication, from design_choice_comparison_2 CSVs.

One figure per side-effect task (weighted AUROC and weighted C-index panels), meds ordered by
positive ratio, bars mean +/- std over folds.
Usage: python vis_design_choice_2_all_vs_singlemeds_bar_horizontal.py [--dataset MSK_CAIA_timeshift]
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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, ttest_ind


# Both arms use the same folds, so the default test is paired (unpaired tests via -st).
# With n=10 folds the smallest Wilcoxon p is 0.00195, so the top comparisons all hit it.
def paired_test(vals1, vals2):
    """Two-sided p for vals1 vs vals2 using STAT_TEST; nan if not computable."""
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


# Arguments
parser = argparse.ArgumentParser(description='Horizontal bar chart: All Meds vs Single Med comparison')
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
                    help="AUROC panel: 'weighted' (default; per-infusion-group AUROC weighted by group "
                         "size) or 'overall' (one AUROC pooled over the valid groups). The C-index panel "
                         "is unchanged. 'overall' adds _overallauroc to the figure and summary names.")
args = parser.parse_args()
AUROC_COL = f'test_auroc_nextinf_{args.auroc_mode}'
AUROC_LABEL = 'Weighted AUROC' if args.auroc_mode == 'weighted' else 'Overall AUROC'
AM_SUFFIX = '_overallauroc' if args.auroc_mode == 'overall' else ''
STAT_TEST = args.stat_test

dataset = args.dataset
modules = args.modules

# Configuration
# --csv_dir overrides the results tree
_arg_csv_dir = getattr(args, 'csv_dir', None)
csv_dir = _arg_csv_dir if _arg_csv_dir else f'{sparc_paths.SPARC_RESULTS_VIS_ROOT}/design_choices_2_all_vs_singlemeds'
save_dir = args.save_dir if args.save_dir else csv_dir
os.makedirs(save_dir, exist_ok=True)

# Side-effect tasks
SIDE_EFFECT_TASKS = ['AKI', 'Liver_toxicity', 'febrile_neutropenia', 'anemia', 'neutropenia']

TASK_DISPLAY = {
    'AKI': 'Acute kidney injury',
    'Liver_toxicity': 'Grade 3 Liver injury',
    'febrile_neutropenia': 'Febrile neutropenia',
    'anemia': 'Severe anemia',
    'neutropenia': 'Neutropenia',
}

# Colours
MODEL_COLORS = {
    'All Meds': '#3B4992',
    'Single Med': '#AAAAAA',
}

MODEL_HATCHES = {
    'All Meds': None,
    'Single Med': None,
}

# Helper functions
def extract_medication(model_name):
    """'all_meds_on_X' / 'single_med_X' -> X."""
    if model_name == 'all_meds':
        return 'all_meds'
    elif model_name.startswith('all_meds_on_'):
        return model_name.replace('all_meds_on_', '')
    elif model_name.startswith('single_med_'):
        return model_name.replace('single_med_', '')
    else:
        return model_name

def extract_model_type(model_name):
    if model_name == 'all_meds' or model_name.startswith('all_meds_on_'):
        return 'All Meds'
    elif model_name.startswith('single_med_'):
        return 'Single Med'
    else:
        return 'Unknown'

# Per task
# cleaned per-task frames for the combined data summary
_summary_frames = []

for task in SIDE_EFFECT_TASKS:
    csv_path = f'{csv_dir}/SPARC_allmeds_vs_singlemeds_{modules}_{task}_{dataset}.csv'

    print(f"\n{'='*80}")
    print(f"Processing task: {task} ({TASK_DISPLAY[task]})")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"WARNING: File not found: {csv_path}")
        print(f"Skipping {task}...")
        continue

    df['medication'] = df['model'].apply(extract_medication)
    df['model_type'] = df['model'].apply(extract_model_type)

    df = df.dropna(subset=[AUROC_COL, 'test_cindex_weighted'])

    df['auroc_pct'] = df[AUROC_COL] * 100
    df['cindex_pct'] = df['test_cindex_weighted'] * 100

    _task_summary = df.copy()
    _task_summary.insert(0, 'task_display', TASK_DISPLAY.get(task, task))
    _task_summary.insert(0, 'task', task)
    _summary_frames.append(_task_summary)

    medications_set = set(df['medication'].unique())
    if len(medications_set) == 0:
        print(f"No valid medications found for {task}")
        continue

    # (n_pos, n_total, pos_ratio) per med
    med_info = {}
    for med in medications_set:
        med_df = df[df['medication'] == med]
        if len(med_df) > 0:
            num_pos = med_df['num_positive_samples'].iloc[0]
            num_total = med_df['num_test_samples'].iloc[0]
            pos_ratio = num_pos / num_total if num_total > 0 else 0
            med_info[med] = (num_pos, num_total, pos_ratio)

    # drop meds with no positives (keep all_meds)
    medications_filtered = []
    for med in medications_set:
        if med == 'all_meds':
            medications_filtered.append(med)
        elif med in med_info and med_info[med][0] > 0:
            medications_filtered.append(med)
        else:
            if med in med_info:
                print(f"  Skipping {med}: 0 positive samples")

    if len(medications_filtered) == 0:
        print(f"No medications with positive samples for {task}")
        continue

    # all_meds first, then by ascending positive ratio
    medications = []
    if 'all_meds' in medications_filtered:
        medications.append('all_meds')
        other_meds = [m for m in medications_filtered if m != 'all_meds']
    else:
        other_meds = medications_filtered

    other_meds_sorted = sorted(other_meds, key=lambda x: med_info[x][2] if x in med_info else 0)
    medications.extend(other_meds_sorted)

    print(f"Found {len(medications)} medications: {medications}")
    print(f"Positive ratios: {[(med, f'{med_info[med][2]*100:.1f}%') for med in medications if med in med_info]}")

    n_meds = len(medications)
    bar_height = 0.35
    med_gap = 0.15

    # AUROC left, C-index right
    fig_width = 16
    fig_height = max(6, n_meds * 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    ax_auroc, ax_cindex = axes

    # y positions
    y_positions = {}  # {(med, model_type): y_pos}
    med_centers = {}  # {med: center_y}

    current_y = 0
    for med in reversed(medications):  # first med at the top
        # all_meds has only the All Meds bar
        if med == 'all_meds':
            models_to_plot = ['All Meds']
        else:
            models_to_plot = ['All Meds', 'Single Med']

        med_y_positions = []
        for i, model_type in enumerate(models_to_plot):
            # All Meds on top
            y_pos = current_y + (len(models_to_plot) - 1 - i) * bar_height
            y_positions[(med, model_type)] = y_pos
            med_y_positions.append(y_pos)

        med_centers[med] = np.mean(med_y_positions)

        current_y += len(models_to_plot) * bar_height + med_gap

    # AUROC
    for med in medications:
        med_df = df[df['medication'] == med]

        if med == 'all_meds':
            models_to_plot = ['All Meds']
        else:
            models_to_plot = ['All Meds', 'Single Med']

        for model_type in models_to_plot:
            model_df = med_df[med_df['model_type'] == model_type]
            if len(model_df) > 0:
                auroc_vals = model_df['auroc_pct'].values
                mean_val = np.mean(auroc_vals)
                std_val = np.std(auroc_vals)
                y_pos = y_positions[(med, model_type)]
                color = MODEL_COLORS[model_type]
                hatch = MODEL_HATCHES[model_type]

                ax_auroc.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                            color=color, edgecolor='black', linewidth=1.2,
                            error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

    # AUROC tests
    print(f"\nAUROC Statistical Tests:")
    for med in medications:
        if med == 'all_meds':
            continue

        med_df = df[df['medication'] == med]
        all_meds_vals = med_df[med_df['model_type'] == 'All Meds']['auroc_pct'].values
        single_med_vals = med_df[med_df['model_type'] == 'Single Med']['auroc_pct'].values

        if len(all_meds_vals) > 1 and len(single_med_vals) > 1:
            p_val = paired_test(all_meds_vals, single_med_vals)
            sig_txt = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'NS'))

            print(f"  {med:30s}: All={np.mean(all_meds_vals):.2f}±{np.std(all_meds_vals):.2f}, "
                  f"Single={np.mean(single_med_vals):.2f}±{np.std(single_med_vals):.2f}, "
                  f"p={p_val:.4f} → {sig_txt}")

            y1 = y_positions[(med, 'All Meds')]
            y2 = y_positions[(med, 'Single Med')]

            mean1 = np.mean(all_meds_vals)
            std1 = np.std(all_meds_vals)
            mean2 = np.mean(single_med_vals)
            std2 = np.std(single_med_vals)

            x_right = max(mean1 + std1, mean2 + std2) + 1.5
            ax_auroc.plot([x_right, x_right], [y1, y2], 'k-', linewidth=1,
                         alpha=0.5 if sig_txt == 'NS' else 1.0)
            y_mid = (y1 + y2) / 2
            ax_auroc.text(x_right + 0.5, y_mid, sig_txt,
                         ha='left', va='center', fontsize=9,
                         alpha=0.6 if sig_txt == 'NS' else 1.0)

    # C-index
    for med in medications:
        med_df = df[df['medication'] == med]

        if med == 'all_meds':
            models_to_plot = ['All Meds']
        else:
            models_to_plot = ['All Meds', 'Single Med']

        for model_type in models_to_plot:
            model_df = med_df[med_df['model_type'] == model_type]
            if len(model_df) > 0:
                cindex_vals = model_df['cindex_pct'].values
                mean_val = np.mean(cindex_vals)
                std_val = np.std(cindex_vals)
                y_pos = y_positions[(med, model_type)]
                color = MODEL_COLORS[model_type]
                hatch = MODEL_HATCHES[model_type]

                ax_cindex.barh(y_pos, mean_val, height=bar_height, xerr=std_val,
                             color=color, edgecolor='black', linewidth=1.2,
                             error_kw={'elinewidth': 1.2, 'capsize': 3, 'capthick': 1.2})

    # C-index tests
    print(f"\nC-index Statistical Tests:")
    for med in medications:
        if med == 'all_meds':
            continue

        med_df = df[df['medication'] == med]
        all_meds_vals = med_df[med_df['model_type'] == 'All Meds']['cindex_pct'].values
        single_med_vals = med_df[med_df['model_type'] == 'Single Med']['cindex_pct'].values

        if len(all_meds_vals) > 1 and len(single_med_vals) > 1:
            p_val = paired_test(all_meds_vals, single_med_vals)
            sig_txt = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'NS'))

            print(f"  {med:30s}: All={np.mean(all_meds_vals):.2f}±{np.std(all_meds_vals):.2f}, "
                  f"Single={np.mean(single_med_vals):.2f}±{np.std(single_med_vals):.2f}, "
                  f"p={p_val:.4f} → {sig_txt}")

            y1 = y_positions[(med, 'All Meds')]
            y2 = y_positions[(med, 'Single Med')]

            mean1 = np.mean(all_meds_vals)
            std1 = np.std(all_meds_vals)
            mean2 = np.mean(single_med_vals)
            std2 = np.std(single_med_vals)

            x_right = max(mean1 + std1, mean2 + std2) + 1.5
            ax_cindex.plot([x_right, x_right], [y1, y2], 'k-', linewidth=1,
                          alpha=0.5 if sig_txt == 'NS' else 1.0)
            y_mid = (y1 + y2) / 2
            ax_cindex.text(x_right + 0.5, y_mid, sig_txt,
                          ha='left', va='center', fontsize=9,
                          alpha=0.6 if sig_txt == 'NS' else 1.0)

    # y labels: medication with sample counts
    y_ticks = [med_centers[med] for med in reversed(medications)]
    y_labels = []
    for med in reversed(medications):
        if med in med_info:
            num_pos, num_total, pos_ratio = med_info[med]
            label = f"{med}\n({int(num_pos)}/{int(num_total)}={pos_ratio*100:.1f}%)"
        else:
            label = med
        y_labels.append(label)

    ax_auroc.set_yticks(y_ticks)
    ax_auroc.set_yticklabels(y_labels, fontsize=9)

    ax_auroc.set_xlabel(f'{AUROC_LABEL} (%)', fontsize=12, fontweight='bold')
    ax_cindex.set_xlabel('Weighted C-index (%)', fontsize=12, fontweight='bold')

    ax_auroc.set_xlim(left=50, right=105)
    ax_cindex.set_xlim(left=50, right=105)

    ax_auroc.set_title(AUROC_LABEL, fontsize=12, fontweight='bold', pad=10)
    ax_cindex.set_title('Weighted C-index', fontsize=12, fontweight='bold', pad=10)

    for ax in [ax_auroc, ax_cindex]:
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    dataset_display = dataset.replace('_', '-')
    fig.suptitle(f'All Meds vs Single Med — {TASK_DISPLAY[task]} — {dataset_display}',
                fontsize=14, fontweight='bold', y=0.98)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS['All Meds'], edgecolor='black',
                 label='All Meds (trained on all medications)', linewidth=1.2),
        Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS['Single Med'], edgecolor='black',
                 label='Single Med (trained on specific medication)', linewidth=1.2),
    ]
    ax_cindex.legend(handles=legend_elements, loc='lower right', fontsize=9,
                    frameon=True, fancybox=True, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    test_suffix = '_' + STAT_TEST if STAT_TEST != 'wilcoxon' else ''
    output_pdf = f'{save_dir}/all_vs_singlemeds_{task}_{dataset}{test_suffix}{AM_SUFFIX}_bar_horizontal.pdf'
    output_png = f'{save_dir}/all_vs_singlemeds_{task}_{dataset}{test_suffix}{AM_SUFFIX}_bar_horizontal.png'

    fig.savefig(output_pdf, bbox_inches='tight')
    fig.savefig(output_png, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFigure saved:")
    print(f"  {output_pdf}")
    print(f"  {output_png}")

# Combined data summary
if _summary_frames:
    combined = pd.concat(_summary_frames, ignore_index=True)

    # long form: one row per (task, model, metric, fold), in %
    id_cols = [c for c in ['task', 'task_display', 'model', 'medication',
                           'model_type', 'val_fold'] if c in combined.columns]
    long_parts = []
    for metric_name, pct_col in [(f'{args.auroc_mode}_auroc', 'auroc_pct'),
                                 ('weighted_cindex', 'cindex_pct')]:
        if pct_col in combined.columns:
            part = combined[id_cols].copy()
            part['metric'] = metric_name
            part['metric_value'] = combined[pct_col].values
            long_parts.append(part)
    summary_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    # mean/std/n across folds per (task, model, metric)
    group_cols = [c for c in ['task', 'task_display', 'model', 'medication',
                              'model_type', 'metric'] if c in summary_long.columns]
    if not summary_long.empty:
        summary_wide = (summary_long
                        .groupby(group_cols, dropna=False)['metric_value']
                        .agg(mean='mean', std='std', n_folds='count')
                        .reset_index())
        summary_wide['std'] = summary_wide['std'].fillna(0.0)
    else:
        summary_wide = pd.DataFrame()

    long_path = f'{save_dir}/allmeds_vs_singlemeds_{modules}_{dataset}{AM_SUFFIX}_summary_long.csv'
    wide_path = f'{save_dir}/allmeds_vs_singlemeds_{modules}_{dataset}{AM_SUFFIX}_summary_wide.csv'
    summary_long.to_csv(long_path, index=False)
    summary_wide.to_csv(wide_path, index=False)
    print("\n" + "="*70)
    print("DATA SUMMARY EXPORT: design_choice_2_allmeds_vs_singlemeds")
    print("="*70)
    print(f"  tasks: {combined['task'].nunique()}, rows(long): {len(summary_long)}")
    print(f"  long: {long_path}")
    print(f"  wide: {wide_path}")
    print("="*70)

print("\n" + "="*80)
print("All plots completed!")
print("="*80)
