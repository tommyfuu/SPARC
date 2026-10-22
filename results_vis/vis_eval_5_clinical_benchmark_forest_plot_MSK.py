#!/usr/bin/env python3
"""AUROC bar/forest plot: published clinical score vs retrained vs SPARC-P on 4 MSK
tasks (AKI, anemia, febrile neutropenia, neutropenia), read from the eval_5 CSVs.
--sparc_first puts SPARC on top of each task, then retrained, then published (files get _sparcfirst).
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

# style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333', 'axes.linewidth': 0.9,
    'xtick.color': '#333333', 'ytick.color': '#333333', 'text.color': '#333333',
})
# palette from PROFESSIONAL_COLORS in the vis_eval_1 across-tasks bar figure
COL_SPARC     = '#2C5F8D'   # SPARC ensemble
COL_ORIGINAL  = '#E07A5F'   # published benchmark
COL_RETRAINED = '#81B29A'   # benchmark refit on our data
COL_REF, COL_SIG = '#CFCFCF', '#555555'
COL_BLUE, COL_SALMON, COL_RED, COL_GRAY = COL_RETRAINED, COL_ORIGINAL, COL_SPARC, COL_ORIGINAL

_DEFAULT_MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_ROOT',
    '/path/to/SPARC_models')
_DEFAULT_RESULTS_DIR = os.environ.get(
    'SPARC_EVAL7_RESULTS_DIR',
    '/path/to/eval7_comparators_results')

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
_ap.add_argument('--sparc_first', action='store_true',
                 help='Within each task: SPARC, then retrained, then original benchmark; '
                      'adds _sparcfirst to the filenames')
_args = _ap.parse_args()
if _args.sparc_first:
    # solid black axis lines and ticks
    plt.rcParams.update({'axes.edgecolor': 'black', 'xtick.color': 'black',
                         'ytick.color': 'black', 'axes.linewidth': 1.0})

RESULTS_DIR = _args.results_dir
SAVE_DIR = _args.save_dir or RESULTS_DIR
ALLOW_PLACEHOLDER = _args.allow_placeholder
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Reading benchmark CSVs from: {RESULTS_DIR}")
print(f"Writing figure to          : {SAVE_DIR}")

# Per task: CSV glob, 'comparison' tag to plot, and 'model' labels as written by
# each eval script (a list means alternatives, e.g. 7- vs 8-variable runs).
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
        # FENCE CSV has no 'comparison' column; rows keyed by model only
        'glob': 'SPARC-P_vs_FENCE_*.csv',
        'comparison': None,
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

# axis floor; CI bounds below it are clipped
XMIN = 0.40
# significance markers sit in a fixed column at the right edge
XMAX = 1.0 if _args.sparc_first else 1.02   # --sparc_first: axis ends at 100%
SIG_X = XMAX - 0.02

# Placeholder numbers copied from the old CAIA figure -- not MSK results.
# Only used with --allow-placeholder when a CSV is missing.
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


def load_task(task):
    """Load the three rows for a task from its CSV; fall back to PLACEHOLDER."""
    cfg = TASK_CONFIG[task]
    # glob on a stem: run options are encoded in the filenames
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
    # 'comparison' may be absent in older files
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
        # label may be a list of alternatives
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
    if _args.sparc_first:
        rows_out = rows_out[::-1]
    if len(rows_out) < 3:
        print(f"[{task}] fewer than 3 rows found -> using PLACEHOLDER. "
              f"(available models: {sorted(df['model'].dropna().unique().tolist())[:12]})")
        return PLACEHOLDER[task], False, None

    # DeLong p-values: *_significance_test rows store p in 'auroc', model = 'SPARC-P_vs_<label>'
    def _pval_for(label, comp):
        # try each candidate label
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


# fallback markers; replaced by CSV DeLong p-values when present
significance = {
    'AKI':                 {'orig': 'ns',  'ret': 'ns'},
    'Anemia':              {'orig': 'ns',  'ret': 'ns'},
    'Febrile neutropenia': {'orig': 'ns',  'ret': 'ns'},
    'Neutropenia':         {'orig': 'ns',  'ret': 'ns'},
}

# assemble data
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


# grouped horizontal bars, one group per task
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

        # xerr takes distances from the estimate, not absolute bounds
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
# AUROC stays a fraction; only tick labels are percentages
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
if _args.sparc_first:
    legend_elems = legend_elems[::-1]
# legend below the axes
ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.065),
          ncol=3, fontsize=9, frameon=False, handlelength=1.4, columnspacing=2.0)
ax.set_title('Clinical benchmark comparison (MSK)', fontsize=13, pad=12, loc='left')
ax.text(0.5, -0.115,
        'Significance shows SPARC vs original / retrained (DeLong): '
        '*** p<0.001, ** p<0.01, * p<0.05, ns = not significant',
        transform=ax.transAxes, ha='center', fontsize=8, color=COL_SIG)
plt.tight_layout()

_sfx = '_sparcfirst' if _args.sparc_first else ''
out_png = os.path.join(SAVE_DIR, f'clinical_benchmark_forest_plot_MSK{_sfx}.png')
out_pdf = os.path.join(SAVE_DIR, f'clinical_benchmark_forest_plot_MSK{_sfx}.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print('saved', out_png, 'and', out_pdf)