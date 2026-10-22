"""Per-task temporal-dependency metrics and figures from LIME attributions.

t=0 is the most recent visit before the prediction cutoff, t=19 the earliest (dates are reverse-sorted
in data_loaders_inf_risk_fixed.py). Importance uses mean-then-abs aggregation, as in lime_importance_risk_fixed.py.
Usage: python visualize_temporal_dependency_clarified.py [--gini table|time]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import entropy
import pandas as pd
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

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

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--gini', choices=['table', 'time'], default='table',
                 help="x-axis of the scatter. 'table' (default): 1 - Gini over every "
                      "timepoint x feature cell of the importance table (20 x 38), so features are "
                      "not averaged away first; saved as temporal_dependency_scatter_clarified_tablegini. "
                      "'time': the original 1 - Gini of the feature-averaged 20-timepoint vector "
                      "(temporal_dependency_scatter_clarified).")
GINI_MODE = _ap.parse_args().gini
base_path = sparc_paths.SPARC_LIME_BF_DIR + '/'
out_path = sparc_paths.SPARC_LIME_BF_DIR + '/'

# Task configuration
tasks = [
    'AKI_risk',
    'Liver_toxicity_risk',
    'anemia_risk',
    'neutropenia_risk',
    'febrile_neutropenia_risk',
    'binary',
    'binary_icu_pre'
]

task_display_names = {
    'AKI_risk': 'Acute kidney injury',
    'Liver_toxicity_risk': 'Grade 3 Liver injury',
    'anemia_risk': 'Severe anemia',
    'neutropenia_risk': 'Neutropenia',
    'febrile_neutropenia_risk': 'Febrile neutropenia',
    'binary': '180D Hosp. mort.',
    'binary_icu_pre': '30D ICU mort.'
}

# Input layout (parsimonious module)
T_MAX = 20
base_features = 37  # parsimonious module
F_TOTAL = 2 * (base_features + 1) + 1  # 77
n_actual = base_features + 1  # 38 (37 features + 1 timestamp)

print("=== Computing Temporal Dependency Metrics ===\n")

task_metrics = {}

for task in tasks:
    print(f"Processing {task}...")
    fold_lime_values = []

    for fold in range(10):
        fname = f'{base_path}/lime_df_val_fold_{fold}_{task}_MSK_CAIA.pickle'
        try:
            with open(fname, 'rb') as f:
                df = pickle.load(f)
                # Reshape: [instances, time, features]
                lime_xs_vals = df.iloc[:, :T_MAX * F_TOTAL].values.reshape(-1, T_MAX, F_TOTAL)
                # Keep only actual features (exclude missingness indicators and padding)
                lime_xs_actual = lime_xs_vals[:, :, :n_actual]
                fold_lime_values.append(lime_xs_actual)
        except FileNotFoundError:
            print(f"  Warning: Missing fold {fold}")
            continue

    if not fold_lime_values:
        print(f"  Skipping {task} - no data")
        continue

    # Stack: [n_folds, n_instances, T, n_actual]
    all_folds = np.array(fold_lime_values)

    # Mean-then-abs: signed mean over instances per fold, then abs, then mean over folds (and features).
    per_fold_signed_mean = all_folds.mean(axis=1)                       # [n_folds, T, n_actual], signed
    time_importance = np.abs(per_fold_signed_mean).mean(axis=(0, 2))    # [T]
    importance_table = np.abs(per_fold_signed_mean).mean(axis=0)        # [T, n_actual]

    # --gini time: 1 - Gini of the 20-timepoint vector (higher = more even).
    sorted_importance = np.sort(time_importance)
    n = len(sorted_importance)
    cumsum = np.cumsum(sorted_importance)
    gini = (2.0 * np.sum((np.arange(1, n+1) * sorted_importance))) / (n * cumsum[-1]) - (n + 1.0) / n
    temporal_uniformity_gini = 1.0 - gini

    # --gini table (default): 1 - Gini over all T x n_actual timepoint x feature cells,
    # so it mixes concentration across features with concentration across time.
    _cells = np.sort(importance_table.ravel())
    _n = _cells.size
    table_gini = (2.0 * np.sum(np.arange(1, _n + 1) * _cells)) / (_n * _cells.sum()) - (_n + 1.0) / _n
    table_uniformity_gini = 1.0 - table_gini

    # Per-timepoint history (t=1-19, mean/median) vs most recent (t=0); <1 means recency dominates.
    recent_tp_importance = time_importance[0]
    historical_tp_values = time_importance[1:]  # [19], t=1..19
    mean_historical_tp_importance = historical_tp_values.mean()
    median_historical_tp_importance = np.median(historical_tp_values)
    mean_historical_vs_recent_ratio = mean_historical_tp_importance / (recent_tp_importance + 1e-10)
    median_historical_vs_recent_ratio = median_historical_tp_importance / (recent_tp_importance + 1e-10)

    # Summed history vs t=0, and each side's share of the total.
    sum_historical_tp_importance = historical_tp_values.sum()
    sum_historical_vs_recent_ratio = sum_historical_tp_importance / (recent_tp_importance + 1e-10)
    total_importance_all20 = time_importance.sum()
    pct_importance_from_history = 100.0 * sum_historical_tp_importance / (total_importance_all20 + 1e-10)
    pct_importance_from_recent = 100.0 * recent_tp_importance / (total_importance_all20 + 1e-10)

    # Peak timepoint vs the median of the rest; Gini barely registers a single spike.
    peak_timepoint = int(np.argmax(time_importance))
    peak_value = time_importance[peak_timepoint]
    rest_values = np.delete(time_importance, peak_timepoint)  # the other 19
    median_of_rest = np.median(rest_values)
    peak_to_rest_ratio = peak_value / (median_of_rest + 1e-10)

    # Normalised entropy over timepoints (1 = uniform), a cross-check on Gini.
    time_prob = time_importance / (time_importance.sum() + 1e-10)
    temporal_entropy = entropy(time_prob)
    temporal_utilization = temporal_entropy / np.log(T_MAX)

    # oldest 5 / most recent 5; >1 means the oldest timepoints matter more.
    recent5_importance = time_importance[:5].mean()
    oldest5_importance = time_importance[-5:].mean()
    oldest5_vs_recent5_ratio = oldest5_importance / (recent5_importance + 1e-10)

    task_metrics[task] = {
        'temporal_uniformity_gini': temporal_uniformity_gini,
        'table_uniformity_gini': table_uniformity_gini,
        'temporal_utilization_entropy': temporal_utilization,
        'mean_historical_vs_recent_ratio': mean_historical_vs_recent_ratio,
        'median_historical_vs_recent_ratio': median_historical_vs_recent_ratio,
        'recent_tp_importance': recent_tp_importance,
        'mean_historical_tp_importance': mean_historical_tp_importance,
        'median_historical_tp_importance': median_historical_tp_importance,
        'peak_timepoint': peak_timepoint,
        'peak_value': peak_value,
        'median_of_rest': median_of_rest,
        'peak_to_rest_ratio': peak_to_rest_ratio,
        'oldest5_vs_recent5_ratio': oldest5_vs_recent5_ratio,
        'sum_historical_tp_importance': sum_historical_tp_importance,
        'sum_historical_vs_recent_ratio': sum_historical_vs_recent_ratio,
        'pct_importance_from_history': pct_importance_from_history,
        'pct_importance_from_recent': pct_importance_from_recent,
        'time_importance': time_importance,  # For detailed plotting
    }

    print(f"  Temporal Uniformity (1-Gini):          {temporal_uniformity_gini:.4f} (1=equal, 0=concentrated)")
    print(f"  Table Uniformity (1-Gini, T x F):      {table_uniformity_gini:.4f} (all timepoint x feature cells)")
    print(f"  Temporal Utilization (entropy):        {temporal_utilization:.4f} (1=equal, 0=concentrated)")
    print(f"  Mean Historical/Recent Ratio:          {mean_historical_vs_recent_ratio:.4f}  "
          f"(mean of t=1-19 [history] vs t=0 [most recent], not a sum)")
    print(f"  Median Historical/Recent Ratio:        {median_historical_vs_recent_ratio:.4f}  "
          f"(robust to a single outlier historical timepoint)")
    print(f"  Sum(historical)/Recent Ratio:          {sum_historical_vs_recent_ratio:.2f}x  "
          f"(total of t=1-19 vs t=0)")
    print(f"  Importance share: history={pct_importance_from_history:.1f}%  "
          f"most-recent={pct_importance_from_recent:.1f}%")
    print(f"  Peak timepoint:                        t={peak_timepoint} "
          f"({'most recent' if peak_timepoint == 0 else f'{peak_timepoint} steps back in time'}, "
          f"value={peak_value:.6f}, {peak_to_rest_ratio:.2f}x the median of the other 19)")
    print(f"  Most-recent (t=0) Importance:          {recent_tp_importance:.6f}")
    print(f"  Oldest5(t=15-19)/Recent5(t=0-4) Ratio: {oldest5_vs_recent5_ratio:.4f}\n")

# Figures

# Task colors: Okabe-Ito colorblind-safe palette
TASK_COLORS = {
    'binary': '#0072B2',                     # blue
    'AKI_risk': '#E69F00',                   # orange
    'Liver_toxicity_risk': '#009E73',        # bluish green
    'anemia_risk': '#56B4E9',                # sky blue
    'febrile_neutropenia_risk': '#D55E00',   # vermillion
    'neutropenia_risk': '#CC79A7',           # reddish purple
    'binary_icu_pre': '#F0E442',             # yellow
}

tasks_to_plot = [t for t in tasks if t in task_metrics]
colors = [TASK_COLORS[t] for t in tasks_to_plot]

# Bar: history vs t=0 share, sorted. The null is 95%, not 50%: with 19 historical
# timepoints vs 1 recent, equal per-timepoint weighting already gives 19/20.
NULL_HISTORY_SHARE = 100.0 * 19 / 20

order = sorted(tasks_to_plot, key=lambda t: task_metrics[t]['pct_importance_from_history'])
hist_pct = [task_metrics[t]['pct_importance_from_history'] for t in order]
recent_pct = [task_metrics[t]['pct_importance_from_recent'] for t in order]
labels = [task_display_names[t] for t in order]
bar_colors = [TASK_COLORS[t] for t in order]

fig_h, ax_h = plt.subplots(figsize=(9, 6))
y_pos = np.arange(len(order))
ax_h.barh(y_pos, hist_pct, color=bar_colors, edgecolor='black', linewidth=0.9,
          label='Share from history (t=1-19, summed)', zorder=2)
ax_h.barh(y_pos, recent_pct, left=hist_pct, color=bar_colors, alpha=0.35,
          edgecolor='black', linewidth=0.9, hatch='//',
          label='Share from most recent (t=0)', zorder=2)
for i, (h, r) in enumerate(zip(hist_pct, recent_pct)):
    ax_h.text(h / 2, y_pos[i], f'{h:.0f}%', ha='center', va='center',
              fontsize=10, fontweight='bold', color='white')
    ax_h.text(h + r / 2, y_pos[i], f'{r:.0f}%', ha='center', va='center',
              fontsize=9, fontweight='bold', color='black')
ax_h.axvline(NULL_HISTORY_SHARE, color='black', linestyle=':', linewidth=1.8, zorder=3,
             label='Null: flat/no recency effect (19/20 = 95%)')
ax_h.set_yticks(y_pos)
ax_h.set_yticklabels(labels, fontsize=11)
ax_h.set_xlabel('Share of total LIME importance across the 20-timepoint window (%)',
                fontsize=12, fontweight='bold')
ax_h.set_title('History is not ignored, but every task sits at or below the\n'
               '"no recency effect" null (dotted line): recency is somewhat\n'
               'OVER-weighted relative to naive equal-per-timepoint weighting',
               fontsize=13, fontweight='bold')
ax_h.set_xlim(0, 100)
ax_h.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
ax_h.spines['top'].set_visible(False)
ax_h.spines['right'].set_visible(False)
ax_h.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1,
            frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f'{out_path}/temporal_history_share_barh.pdf', dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}/temporal_history_share_barh.pdf")
plt.close()

# Scatter: 1 - Gini (per --gini) vs Sum(historical)/recent; dashed y=1 = equal totals.

fig, ax = plt.subplots(figsize=(11, 6.5))

_gini_key = 'table_uniformity_gini' if GINI_MODE == 'table' else 'temporal_uniformity_gini'
uniformity_values = [task_metrics[t][_gini_key] for t in tasks_to_plot]
sum_ratios = [task_metrics[t]['sum_historical_vs_recent_ratio'] for t in tasks_to_plot]

AXIS_GREY = '#4d4d4d'

for i, task in enumerate(tasks_to_plot):
    ax.scatter(uniformity_values[i], sum_ratios[i], s=220,
               c=colors[i], alpha=0.95, edgecolors='black', linewidths=1.3,
               label=task_display_names[task], zorder=3)

ax.axhline(1.0, color=AXIS_GREY, linestyle='--', linewidth=1.2, zorder=2,
           label='Sum(historical) = recent (ratio = 1)')

ax.set_title('Temporal dependency: history vs. most-recent measurement',
              fontsize=14, fontweight='bold', color='black')
ax.set_xlabel('Uniformity of importance across timepoints × features (1 − Gini)' if GINI_MODE == 'table'
              else 'Temporal uniformity (1 − Gini)', fontsize=12, color=AXIS_GREY)
ax.set_ylabel('Sum(historical) / most-recent importance ratio', fontsize=12, color=AXIS_GREY)
ax.margins(x=0.12, y=0.12)
ax.grid(True, linestyle='-', linewidth=0.8, color='#e0e0e0', zorder=0)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color(AXIS_GREY)
    ax.spines[spine].set_linewidth(1.0)
ax.tick_params(colors=AXIS_GREY, labelsize=10)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10,
          labelspacing=1.4, handletextpad=0.8, borderaxespad=0.5)

# Fixed margins: tight_layout squeezes the axes to fit the outside legend.
fig.subplots_adjust(left=0.09, right=0.68, top=0.9, bottom=0.12)
_scatter_stem = 'temporal_dependency_scatter_clarified' + ('_tablegini' if GINI_MODE == 'table' else '')
plt.savefig(f'{out_path}/{_scatter_stem}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{out_path}/{_scatter_stem}.png', dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}/{_scatter_stem}.pdf")
plt.close()

# Summary table

print()
print("SUMMARY: Temporal Dependency Metrics")
print()
print(f"{'Task':<24} {'Unif.(1-Gini)':<14} {'Table 1-Gini':<13} {'Mean H/R':<10} {'Median H/R':<12} "
      f"{'Sum H/R':<10} {'Hist.share%':<12} {'Peak t':<8} {'Peak/restMed':<13}")
print("-"*130)
for task in tasks_to_plot:
    m = task_metrics[task]
    print(f"{task_display_names[task]:<24} "
          f"{m['temporal_uniformity_gini']:<14.4f} "
          f"{m['table_uniformity_gini']:<13.4f} "
          f"{m['mean_historical_vs_recent_ratio']:<10.4f} "
          f"{m['median_historical_vs_recent_ratio']:<12.4f} "
          f"{m['sum_historical_vs_recent_ratio']:<10.2f} "
          f"{m['pct_importance_from_history']:<12.1f} "
          f"{m['peak_timepoint']:<8d} "
          f"{m['peak_to_rest_ratio']:<13.2f}")
print()

# CSV export
rows = []
for task in tasks_to_plot:
    m = task_metrics[task]
    row = {'task': task, 'task_display_name': task_display_names[task]}
    row.update({k: v for k, v in m.items() if k != 'time_importance'})
    rows.append(row)
pd.DataFrame(rows).to_csv(f'{out_path}/temporal_dependency_metrics_clarified.csv', index=False)
print(f"\nSaved: {out_path}/temporal_dependency_metrics_clarified.csv")

print("\nTime axis: t=0 is the most recent measurement before the")
print("prediction cutoff; t=19 is the earliest (dates are reverse-sorted in")
print("data_loaders_inf_risk_fixed.py).")
print("\nInterpretation:")
print("  - Temporal Uniformity (1-Gini): whole-distribution equality across all 20 timepoints.")
print("    * 1 = perfectly equal. Forgiving of a single outlier timepoint among 20;")
print("      check peak_to_rest_ratio too.")
print("  - Temporal Utilization (entropy): a second, independently-computed evenness measure.")
print("  - Mean/Median Historical/Recent Ratio: per-timepoint MEAN/MEDIAN comparison of the 19")
print("    historical timepoints (t=1-19) vs the single most recent one (t=0), not")
print("    historical-timepoints-summed vs recent.")
print("    * ~1.0 = a typical historical timepoint matters about as much as the most recent one.")
print("    * > 1.0 = history matters more than recency; < 1.0 = recency dominates.")
print("    * Median version is robust to the peak timepoint inflating the mean.")
print("  - Peak timepoint / peak_to_rest_ratio: which single timepoint is most important, and")
print("    how many times larger it is than the median of the other 19. peak_timepoint=0 means")
print("    the single most recent measurement dominates.")
print("  - Oldest5(t=15-19)/Recent5(t=0-4) Ratio: coarser 5-timepoint-block mean comparison;")
print("    >1 means the 5 oldest timepoints matter more than the 5 most recent ones.")
print("  - Sum(historical)/Recent Ratio & Hist.share%: benchmark against the 95% null")
print("    (19/20 historical timepoints), not against 0%. See temporal_history_share_barh.pdf.")
print("\nDone!")
