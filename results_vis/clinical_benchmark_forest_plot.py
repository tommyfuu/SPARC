#!/usr/bin/env python3
"""
Forest plot comparing clinical benchmarks with SPARC-P across tasks.
Restyled to match the companion figure's palette and minimalist design:
muted blue / salmon pair, thin caps, plain-text significance markers,
clean spines, no bold, no boxed annotations.

The CSV-reading logic is retained at the bottom (commented) for use on the
cluster. The DATA block below carries the same values so the figure renders
anywhere without the source files.
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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Global style: quiet, sans-serif, minimal chrome --------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.9,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
})

# Palette borrowed from the companion figure.
COL_BLUE   = '#5DA5DA'   # muted blue  (matches "G-CSF" bars)
COL_SALMON = '#E8837B'   # salmon (matches "No G-CSF" bars)
COL_RED    = '#E0201B'   # pure red -> SPARC, the hero
COL_GRAY   = '#B0B0B0'   # neutral baseline for the original benchmark
COL_REF    = '#CFCFCF'   # faint reference line at 0.5
COL_SIG    = '#555555'   # significance-marker text

# --- Data (point estimate, 95% CI, n, events) --------------------------------
# name, auroc, ci_lower, ci_upper, n, events
data = {
    'AKI': [
        {'name': 'Motwani, et al.',           'auroc': 0.545, 'lo': 0.466, 'hi': 0.630, 'n': 521, 'ev': 43},
        {'name': 'Motwani, et al. retrained', 'auroc': 0.572, 'lo': 0.478, 'hi': 0.657, 'n': 521, 'ev': 43},
        {'name': 'SPARC ensemble',            'auroc': 0.530, 'lo': 0.447, 'hi': 0.609, 'n': 521, 'ev': 43},
    ],
    'Febrile neutropenia': [
        {'name': 'FENCE',           'auroc': 0.497, 'lo': 0.286, 'hi': 0.723, 'n': 2039, 'ev': 10},
        {'name': 'FENCE retrained', 'auroc': 0.584, 'lo': 0.395, 'hi': 0.763, 'n': 2039, 'ev': 10},
        {'name': 'SPARC ensemble',  'auroc': 0.936, 'lo': 0.899, 'hi': 0.968, 'n': 2039, 'ev': 10},
    ],
    'Anemia': [
        {'name': 'Dranitsaris, et al.',           'auroc': 0.640, 'lo': 0.497, 'hi': 0.781, 'n': 254, 'ev': 19},
        {'name': 'Dranitsaris, et al. retrained', 'auroc': 0.651, 'lo': 0.522, 'hi': 0.765, 'n': 254, 'ev': 19},
        {'name': 'SPARC ensemble',                'auroc': 0.915, 'lo': 0.836, 'hi': 0.978, 'n': 254, 'ev': 19},
    ],
}

# Each benchmark vs SPARC (DeLong). 'orig' -> original line, 'ret' -> retrained line.
significance = {
    'AKI':                 {'orig': 'ns',  'ret': 'ns'},
    'Febrile neutropenia': {'orig': '***', 'ret': '***'},
    'Anemia':              {'orig': '***', 'ret': '***'},
}

tasks = ['AKI', 'Febrile neutropenia', 'Anemia']


def style_for(name):
    """Return (color, marker, size) for a row based on its model type."""
    if 'SPARC' in name:
        return COL_RED, 'D', 6
    if 'retrained' in name:
        return COL_BLUE, 's', 6
    return COL_GRAY, 'o', 6


# --- Layout ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

row_gap  = 0.6   # between models within a task
task_gap = 1.4   # between tasks

y_pos_all = []
tick_pos, tick_lab = [], []

# which benchmark each row's significance key maps to
sig_key = {'original': 'orig', 'retrained': 'ret'}

y = 0.0
for ti, task in enumerate(tasks):
    if ti > 0:
        y -= task_gap
    rows = data[task]
    task_ys = []
    for mi, row in enumerate(rows):
        yy = y - mi * row_gap
        task_ys.append(yy)
        y_pos_all.append(yy)

        color, marker, msize = style_for(row['name'])

        # confidence interval line + thin caps
        ax.plot([row['lo'], row['hi']], [yy, yy], color=color, lw=1.3, zorder=2)
        for x in (row['lo'], row['hi']):
            ax.plot([x, x], [yy - 0.06, yy + 0.06], color=color, lw=1.3, zorder=2)
        # point estimate
        ax.plot(row['auroc'], yy, marker=marker, color=color, markersize=msize,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3)

        # significance star to the right of each benchmark line (vs SPARC)
        if 'retrained' in row['name']:
            star = significance[task]['ret']
        elif 'SPARC' in row['name']:
            star = None
        else:
            star = significance[task]['orig']
        if star:
            ax.text(row['hi'] + 0.015, yy, star, fontsize=9, color=COL_SIG,
                    va='center', ha='left')

    # task label + sample size, centered on the block
    center = sum(task_ys) / len(task_ys)
    tick_pos.append(center)
    tick_lab.append(f"{task}\nn={rows[0]['n']}, events={rows[0]['ev']}")
    y -= (len(rows) - 1) * row_gap

# faint reference line at chance
ax.axvline(0.5, color=COL_REF, lw=1, zorder=0)

# --- Axes --------------------------------------------------------------------
ax.set_xlabel('AUROC', fontsize=12)
ax.set_xlim(0.25, 1.02)
ax.set_ylim(min(y_pos_all) - 0.5, max(y_pos_all) + 0.5)
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks(tick_pos)
ax.set_yticklabels(tick_lab, fontsize=11)
ax.tick_params(axis='y', length=0)

# --- Legend & footnote -------------------------------------------------------
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_GRAY,
           markersize=7, markeredgecolor='white', label='Original benchmark'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COL_BLUE,
           markersize=7, markeredgecolor='white', label='Retrained benchmark'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_RED,
           markersize=7, markeredgecolor='white', label='SPARC ensemble'),
]
ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(0.005, 0.99),
          fontsize=9, frameon=False)

ax.set_title('Clinical benchmark comparison', fontsize=13, pad=12, loc='left')
fig.text(0.065, 0.02,
         'Significance shows SPARC vs original / retrained (DeLong): '
         '*** p<0.001, ** p<0.01, * p<0.05, ns = not significant',
         fontsize=8, color=COL_SIG)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_png = sparc_paths.SPARC_MODEL_ROOT + '/results_vis_060726/eval_clinical_benchmarks/clinical_benchmark_forest_plot.png'
out_pdf = sparc_paths.SPARC_MODEL_ROOT + '/results_vis_060726/eval_clinical_benchmarks/clinical_benchmark_forest_plot.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print('saved', out_png, 'and', out_pdf)