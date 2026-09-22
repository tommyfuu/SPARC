#!/usr/bin/env python3
"""
Forest plot: clinical-benchmark vs retrained vs SPARC-P across 4 MSK tasks
(AKI, Anemia, Febrile neutropenia, Neutropenia).

Reads point estimates + 95% CIs DIRECTLY from each benchmark script's output CSV
(the *_MSK_*.csv files), so the figure always reflects the latest runs. Falls back
to the hardcoded DATA block only if a CSV is missing -- and those fallback numbers
are PLACEHOLDERS marked TODO, NOT real results.

Which comparison row is plotted per task is controlled by COMPARISON_TAG (default
the patient-level ever-positive comparison, the primary head-to-head).
"""
import os
import glob
import argparse
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker

# --- Global style ------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333', 'axes.linewidth': 0.9,
    'xtick.color': '#333333', 'ytick.color': '#333333', 'text.color': '#333333',
})
# Palette taken from PROFESSIONAL_COLORS in
# vis_eval_1_across_tasks_t1e_controlled_bar_060826_horizontal_succinct_wilcoxon_all.py
# so the two figures read as one set. SPARC-P ranks first there, so it is
# assigned index 0 (#2C5F8D); keep that fixed here regardless of ordering.
COL_SPARC     = '#2C5F8D'   # deep blue   -- SPARC ensemble (matches grid figure)
COL_ORIGINAL  = '#E07A5F'   # terracotta  -- published benchmark
COL_RETRAINED = '#81B29A'   # sage green  -- benchmark refit on our data
COL_REF, COL_SIG = '#CFCFCF', '#555555'
# retained for any legacy reference
COL_BLUE, COL_SALMON, COL_RED, COL_GRAY = COL_RETRAINED, COL_ORIGINAL, COL_SPARC, COL_ORIGINAL

# Where to read results from, per task. Point each task at its benchmark CSV and
# name the three model rows (original comparator label, retrained label, SPARC).
# The labels must match the 'model' column values written by each eval script.
# Where the benchmark CSVs live. Defaults to the batch tree the eval scripts
# now write to; point --results_dir at results_vis_060726/eval_clinical_benchmarks
# to plot the older run instead.
_DEFAULT_MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_ROOT',
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models_060726')
_DEFAULT_RESULTS_DIR = os.path.join(
    os.environ.get('SPARC_RESULTS_ROOT',
                   f'{_DEFAULT_MODEL_ROOT}/results_vis_060726_batch_091426'),
    'eval_clinical_benchmarks')

_ap = argparse.ArgumentParser(
    description='Forest plot of SPARC-P vs published clinical risk scores.')
_ap.add_argument('--results_dir', default=_DEFAULT_RESULTS_DIR,
                 help='Directory holding the SPARC-P_vs_*.csv benchmark outputs '
                      '(default: %(default)s)')
_ap.add_argument('--save_dir', default=None,
                 help='Where to write the figure (default: --results_dir)')
_ap.add_argument('--allow-placeholder', action='store_true',
                 help='Permit the built-in PLACEHOLDER numbers when a CSV is '
                      'missing. OFF by default: the placeholders are copied from '
                      'the old CAIA figure and are NOT real MSK results, so a '
                      'silent fallback would publish fabricated values.')
_args = _ap.parse_args()

RESULTS_DIR = _args.results_dir
SAVE_DIR = _args.save_dir or RESULTS_DIR
ALLOW_PLACEHOLDER = _args.allow_placeholder
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Reading benchmark CSVs from: {RESULTS_DIR}")
print(f"Writing figure to          : {SAVE_DIR}")

# The comparison block to plot (the primary head-to-head). Each eval writes rows
# tagged by 'comparison'; the patient-level ever-positive one is the fair primary.
TASK_CONFIG = {
    'AKI': {
        'glob': 'SPARC-P_vs_MOTWANI_*.csv',
        'comparison': 'MOTWANI_firstcycle (cycle==0)',   # Motwani is first-cycle
        'orig_label': 'Motwani',
        'ret_label':  'Motwani_retrained',
        'ret_comparison': 'MOTWANI_retrained_firstcycle (cycle==0)',
        'sparc_label': 'SPARC-P_ensemble',
        'pretty': 'Motwani, et al.',
    },
    'Anemia': {
        'glob': 'SPARC-P_vs_DRANITSARIS_*.csv',
        'comparison': 'DRAN_patient_level_ever_positive',
        'orig_label': ['Dranitsaris', 'Dranitsaris_4var'],
        'ret_label':  ['Dranitsaris_retrained', 'Dranitsaris_retrained_4var'],
        'ret_comparison': 'DRAN_patient_level_ever_positive',
        'sparc_label': 'SPARC-P_ensemble_maxrisk',
        'pretty': 'Dranitsaris, et al.',
    },
    'Febrile neutropenia': {
        # Real MSK FENCE (first-cycle). This CSV has NO 'comparison' column -- rows
        # are keyed by model only, with the '_firstcycle' suffix.
        'glob': 'SPARC-P_vs_FENCE_*.csv',
        'comparison': None,   # no comparison column in this file
        'orig_label': 'FENCE_firstcycle',
        'ret_label':  'FENCE_retrained_firstcycle',
        'ret_comparison': None,
        'sparc_label': 'SPARC-P_ensemble_firstcycle',
        'pretty': 'FENCE',
    },
    'Neutropenia': {
        'glob': 'SPARC-P_vs_LYMAN_*.csv',
        'comparison': 'LYMAN_patient_level_ever_positive',
        'orig_label': 'Lyman_published',
        'ret_label':  'Lyman_retrained',
        'ret_comparison': 'LYMAN_patient_level_ever_positive',
        'sparc_label': 'SPARC-P_ensemble_maxrisk',
        'pretty': 'Lyman, et al.',
    },
}
tasks = ['AKI', 'Anemia', 'Febrile neutropenia', 'Neutropenia']

# Bars start here rather than at 0: AUROC below chance is not meaningful and a
# 0-anchored bar would compress all the signal into the right-hand third. Note
# that a lower CI bound below XMIN is clipped at the axis floor.
XMIN = 0.40
# Right edge of the axes. Significance markers sit in a fixed column just
# inside it rather than floating after each CI, matching the all-models grid.
XMAX = 1.02
SIG_X = XMAX - 0.02

# PLACEHOLDER fallback data (used only if a CSV is missing). TODO: these are NOT
# real MSK results -- they are copied from the old CAIA figure so the plot renders.
# Replace by running the eval scripts so the CSVs exist, or edit these in by hand.
PLACEHOLDER = {
    'AKI': [
        {'name': 'Motwani, et al.',           'auroc': 0.545, 'lo': 0.466, 'hi': 0.630, 'n': 521, 'ev': 43},
        {'name': 'Motwani, et al. retrained', 'auroc': 0.572, 'lo': 0.478, 'hi': 0.657, 'n': 521, 'ev': 43},
        {'name': 'SPARC ensemble',            'auroc': 0.530, 'lo': 0.447, 'hi': 0.609, 'n': 521, 'ev': 43},
    ],
    'Anemia': [
        {'name': 'Dranitsaris, et al.',           'auroc': 0.640, 'lo': 0.497, 'hi': 0.781, 'n': 254, 'ev': 19},
        {'name': 'Dranitsaris, et al. retrained', 'auroc': 0.651, 'lo': 0.522, 'hi': 0.765, 'n': 254, 'ev': 19},
        {'name': 'SPARC ensemble',                'auroc': 0.915, 'lo': 0.836, 'hi': 0.978, 'n': 254, 'ev': 19},
    ],
    'Febrile neutropenia': [
        {'name': 'FENCE',           'auroc': 0.497, 'lo': 0.286, 'hi': 0.723, 'n': 2039, 'ev': 10},
        {'name': 'FENCE retrained', 'auroc': 0.584, 'lo': 0.395, 'hi': 0.763, 'n': 2039, 'ev': 10},
        {'name': 'SPARC ensemble',  'auroc': 0.936, 'lo': 0.899, 'hi': 0.968, 'n': 2039, 'ev': 10},
    ],
    'Neutropenia': [
        {'name': 'Lyman, et al.',           'auroc': 0.510, 'lo': 0.42, 'hi': 0.60, 'n': 258, 'ev': 45},
        {'name': 'Lyman, et al. retrained', 'auroc': 0.510, 'lo': 0.41, 'hi': 0.60, 'n': 258, 'ev': 45},
        {'name': 'SPARC ensemble',          'auroc': 0.640, 'lo': 0.56, 'hi': 0.71, 'n': 258, 'ev': 45},
    ],
}


def _row_from_csv(df, comparison, model_label):
    """Extract auroc/lo/hi/n/ev for one (comparison, model) row."""
    sub = df[df['model'] == model_label]
    if comparison is not None:
        sub = sub[sub['comparison'] == comparison]
    sub = sub[pd.notna(sub['auroc'])]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {
        'auroc': float(r['auroc']),
        'lo': float(r['auroc_ci_lower']) if pd.notna(r.get('auroc_ci_lower')) else float(r['auroc']),
        'hi': float(r['auroc_ci_upper']) if pd.notna(r.get('auroc_ci_upper')) else float(r['auroc']),
        'n':  int(r['n']) if pd.notna(r.get('n')) else None,
        'ev': int(r['n_pos']) if pd.notna(r.get('n_pos')) else None,
    }


def load_task(task):
    """Load the three rows for a task from its CSV; fall back to PLACEHOLDER."""
    cfg = TASK_CONFIG[task]
    # Match on a stem: the eval scripts encode run options in the filename
    # (7var/8var, allcancer/allbreast, MSK/MSK_CAIA), which would otherwise
    # break an exact-name lookup every time an option changes.
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, cfg['glob'])))
    matches = [m for m in matches if 'patient_audit' not in os.path.basename(m)]
    if not matches:
        msg = (f"[{task}] no CSV matching {cfg['glob']} in {RESULTS_DIR}")
        if not ALLOW_PLACEHOLDER:
            raise SystemExit(
                msg + "\n  The built-in PLACEHOLDER values are copied from the old "
                "CAIA figure and are NOT real MSK results, so plotting them would "
                "silently fabricate data.\n  Run the corresponding eval script, "
                "point --results_dir at the right directory, or pass "
                "--allow-placeholder if you knowingly want a mock figure.")
        print(msg + " -> using PLACEHOLDER (NOT real).")
        return PLACEHOLDER[task], False, None
    if len(matches) > 1:
        print(f"[{task}] {len(matches)} files match {cfg['glob']}; using most recent: "
              f"{os.path.basename(matches[-1])}")
    path = matches[-1]
    print(f"[{task}] reading {os.path.basename(path)}")
    df = pd.read_csv(path)
    # Defensive: the CSV must have 'model' and 'auroc'. 'comparison' may be absent
    # in older-schema files -> treat as a single unnamed comparison.
    if 'model' not in df.columns or 'auroc' not in df.columns:
        print(f"[{task}] CSV missing 'model'/'auroc' columns "
              f"(has {list(df.columns)}) -> PLACEHOLDER.")
        return PLACEHOLDER[task], False, None
    has_comp = 'comparison' in df.columns

    def _resolve(c):
        if not has_comp:
            return None
        if c is not None:
            return c
        cands = [x for x in df['comparison'].dropna().unique() if 'patient_level' in str(x)]
        return cands[0] if cands else None

    orig_comp = _resolve(cfg.get('comparison'))
    ret_comp  = _resolve(cfg.get('ret_comparison', cfg.get('comparison')))
    sparc_label = cfg.get('sparc_label', 'SPARC-P_ensemble')

    def _row(label, c):
        # label may be a list of alternatives: the 7-variable run writes
        # 'Dranitsaris_4var' while the 8-variable run writes 'Dranitsaris'.
        labels = label if isinstance(label, (list, tuple)) else [label]
        sub = df[df['model'].isin(labels)]
        if has_comp and c is not None:
            sub = sub[sub['comparison'] == c]
        sub = sub[pd.notna(sub['auroc'])]
        if len(sub) == 0:
            return None
        r = sub.iloc[0]
        gl = lambda k: (float(r[k]) if k in r and pd.notna(r[k]) else None)
        return {'auroc': float(r['auroc']),
                'lo': gl('auroc_ci_lower') if gl('auroc_ci_lower') is not None else float(r['auroc']),
                'hi': gl('auroc_ci_upper') if gl('auroc_ci_upper') is not None else float(r['auroc']),
                'n':  int(r['n']) if 'n' in r and pd.notna(r['n']) else None,
                'ev': int(r['n_pos']) if 'n_pos' in r and pd.notna(r['n_pos']) else None}

    rows_out = []
    for label, c, pretty in [(cfg['orig_label'], orig_comp, cfg['pretty']),
                             (cfg['ret_label'],  ret_comp,  cfg['pretty'] + ' retrained'),
                             (sparc_label,       orig_comp, 'SPARC ensemble')]:
        rec = _row(label, c)
        if rec is None:
            print(f"[{task}] row '{label}' (comparison={c}) not found in CSV.")
            continue
        rec['name'] = pretty
        rows_out.append(rec)
    if len(rows_out) < 3:
        print(f"[{task}] fewer than 3 rows found -> using PLACEHOLDER. "
              f"(available models: {sorted(df['model'].dropna().unique().tolist())[:12]})")
        return PLACEHOLDER[task], False, None

    # ---- auto-parse DeLong significance. In the *_significance_test rows the
    # p-value is stored in the 'auroc' column; model col = 'SPARC-P_vs_<label>'. ----
    def _pval_for(label, comp):
        # try several candidate significance-row model labels (schemas differ).
        # label may be a list of alternatives (see _row).
        labels = label if isinstance(label, (list, tuple)) else [label]
        candidates = []
        for lab in labels:
            base = lab.replace('_firstcycle', '')
            candidates += [f'SPARC-P_vs_{lab}', f'SPARC-P_vs_{base}',
                           f'SPARC-P_vs_{base}_significance_test',
                           f'SPARC-P_vs_{lab}_significance_test']
        for cand in candidates:
            m = df[df['model'] == cand]
            if has_comp and comp is not None:
                sig_comp = f"{comp}_significance_test"
                m2 = m[m['comparison'] == sig_comp]
                m = m2 if len(m2) else m
            m = m[pd.notna(m['auroc'])]
            if len(m) > 0:
                return float(m.iloc[0]['auroc'])   # p-value in auroc column
        return None

    def _stars(p):
        if p is None:
            return ''
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'
    _sig = {'orig': _stars(_pval_for(cfg['orig_label'], orig_comp)),
            'ret':  _stars(_pval_for(cfg['ret_label'],  ret_comp))}
    return rows_out, True, _sig


# significance markers: manual fallback, OVERRIDDEN per task by auto-parsed
# DeLong p-values from the CSV *_significance_test rows when available.
significance = {
    'AKI':                 {'orig': 'ns',  'ret': 'ns'},
    'Anemia':              {'orig': 'ns',  'ret': 'ns'},
    'Febrile neutropenia': {'orig': 'ns',  'ret': 'ns'},
    'Neutropenia':         {'orig': 'ns',  'ret': 'ns'},
}

# --- assemble data -----------------------------------------------------------
data, is_real = {}, {}
for task in tasks:
    data[task], is_real[task], _autosig = load_task(task)
    if _autosig is not None:
        significance[task] = _autosig


def style_for(name):
    if 'SPARC' in name:
        return COL_RED, 'D', 6
    if 'retrained' in name:
        return COL_RETRAINED, 's', 6
    return COL_ORIGINAL, 'o', 6


# --- Layout: grouped horizontal bars (one group per task, three bars each) ---
# Matches the all-models grid figure: horizontal bars with error bars, models
# stacked within a task group on the y-axis.
fig, ax = plt.subplots(figsize=(9, 8.5))
BAR_H, ROW_GAP, TASK_GAP = 0.52, 0.62, 1.25
y_pos_all, tick_pos, tick_lab = [], [], []

y = 0.0
for ti, task in enumerate(tasks):
    if ti > 0:
        y -= TASK_GAP
    rows = data[task]
    task_ys = []
    for mi, row in enumerate(rows):
        yy = y - mi * ROW_GAP
        task_ys.append(yy); y_pos_all.append(yy)
        color, _marker, _ms = style_for(row['name'])

        # Bar from the axis floor to the point estimate, with the CI as the
        # error bar. xerr must be non-negative distances, not absolute bounds.
        lo, hi, est = row['lo'], row['hi'], row['auroc']
        xerr = [[max(0.0, est - lo)], [max(0.0, hi - est)]]
        ax.barh(yy, est - XMIN, left=XMIN, height=BAR_H, color=color,
                edgecolor='black', linewidth=0.9, zorder=2)
        ax.errorbar(est, yy, xerr=xerr, fmt='none', ecolor='black',
                    elinewidth=1.1, capsize=3, capthick=1.1, zorder=3)

        if 'retrained' in row['name']:
            star = significance[task]['ret']
        elif 'SPARC' in row['name']:
            star = None
        else:
            star = significance[task]['orig']
        if star:
            ax.text(SIG_X, yy, star, va='center', ha='right',
                    fontsize=9, fontweight='bold', color='black')

    center = sum(task_ys) / len(task_ys)
    tick_pos.append(center)
    _n, _ev = rows[0].get('n'), rows[0].get('ev')
    tag = '' if is_real[task] else '  [PLACEHOLDER]'
    tick_lab.append(f"{task}\nn={_n}, events={_ev}{tag}")
    y -= (len(rows) - 1) * ROW_GAP

ax.axvline(0.5, color=COL_REF, lw=1, zorder=1)
ax.set_xlabel('AUROC (%)', fontsize=12)
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(min(y_pos_all) - 0.5, max(y_pos_all) + 0.5)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# AUROC is stored as a fraction throughout; only the tick labels are percentages.
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=11)
ax.tick_params(axis='y', length=0)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=COL_ORIGINAL,  edgecolor='black', linewidth=0.9, label='Original benchmark'),
    Patch(facecolor=COL_RETRAINED, edgecolor='black', linewidth=0.9, label='Retrained benchmark'),
    Patch(facecolor=COL_SPARC,     edgecolor='black', linewidth=0.9, label='SPARC ensemble'),
]
# Below the axes rather than inside: the top-left corner is occupied by the
# AKI bars, and a horizontal strip under the x-label keeps the plot area clean.
ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.065),
          ncol=3, fontsize=9, frameon=False, handlelength=1.4, columnspacing=2.0)
ax.set_title('Clinical benchmark comparison (MSK)', fontsize=13, pad=12, loc='left')
ax.text(0.5, -0.115,
        'Significance shows SPARC vs original / retrained (DeLong): '
        '*** p<0.001, ** p<0.01, * p<0.05, ns = not significant',
        transform=ax.transAxes, ha='center', fontsize=8, color=COL_SIG)
plt.tight_layout()

out_png = os.path.join(SAVE_DIR, 'clinical_benchmark_forest_plot_MSK.png')
out_pdf = os.path.join(SAVE_DIR, 'clinical_benchmark_forest_plot_MSK.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print('saved', out_png, 'and', out_pdf)