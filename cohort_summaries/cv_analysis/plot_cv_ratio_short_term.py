#!/usr/bin/env python3
"""Short-term (60-day window) intra/inter-patient CV-ratio figures from saved cv_analysis_fixed.py results.

cv_analysis_fixed.py plots only the long-term (whole-record) ratio, which depends on follow-up length; the short-term
ratio is only in its CSVs. Writes across-12-dataset mean +/- SD, MSK-CAIA panels, short vs long comparisons, and
cv_ratio_short_term_by_dataset.csv.
Usage: python plot_cv_ratio_short_term.py [--out_dir DIR]
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
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
CV = os.path.join(sparc_paths.COHORT_SUMMARIES_DIR, 'cv_analyses')   # cv_analysis_fixed.py writes here
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default=os.path.join(CV, 'short_term_figures'))
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# same 12 files as across_datasets/cv_ratio_summary_across_datasets_parsimonious_sti60_all.csv (reproduce its means)
FILES = {
    'MSK-CAIA': 'parsimonious_MSK_CAIA/cv_analysis_results_parsimonious_MSK_CAIA_sti60.csv',
    'DFCI (CAIA)': 'parsimonious_DFCI_CAIA/cv_analysis_results_DFCI_CAIA_all_parsimonious_sti60.csv',
    'FHCC (CAIA)': 'parsimonious_FHCC_CAIA/cv_analysis_results_FHCC_CAIA_all_parsimonious_sti60.csv',
    'JHU (CAIA)': 'parsimonious_JHU_CAIA/cv_analysis_results_JHU_CAIA_parsimonious_sti60.csv',
    **{f'INSIGHT {s}': f'parsimonious_INSIGHT_{s}/cv_analysis_results_parsimonious_insight_sti60.csv' for s in 'ABCDEF'},
    'MIMIC-IV': 'parsimonious_MIMICIV_hosp/cv_analysis_results_parsimonious_MIMICIV_hosp_sti60.csv',
    'EHRSHOT': 'parsimonious_stanfordEHRSHOT/cv_analysis_results_parsimonious_stanfordEHRSHOT_sti60.csv',
}
# feature order, display names and module colours as in the manuscript figure
FEATURES = [('RBC', 'RBC'), ('MCV', 'MCV'), ('MCH', 'MCH'), ('MCHC', 'MCHC'), ('RDW', 'RDW'), ('HGB', 'HGB'),
            ('HCT', 'HCT'), ('WBC', 'WBC'), ('Platelets', 'Platelets'), ('Neut', 'Neutrophil'), ('Mono', 'Monocytes'),
            ('Lymph', 'Lymphocytes'), ('Baso', 'Basophils'), ('Eos', 'Eosinophils'), ('Glucose', 'Glucose'),
            ('Calcium', 'Calcium'), ('Sodium', 'Sodium'), ('Potassium', 'Potassium'), ('CO2', 'CO2'),
            ('Creatinine', 'Creatinine'), ('Chloride', 'Chloride'), ('BUN', 'BUN'), ('Albumin', 'Albumin'),
            ('Protein, Total', 'Protein, Total'), ('ALP', 'ALP'), ('ALT', 'ALT'), ('AST', 'AST'),
            ('Bilirubin, Total', 'Bilirubin, Total'), ('BMI', 'BMI'), ('Blood Pressure BP Systolic', 'Systolic BP'),
            ('Blood Pressure Diastolic', 'Diastolic BP'), ('O2', 'O2'), ('Pulse Heart Rate', 'Pulse Hrt Rate'),
            ('Resp Rate', 'Resp Rate'), ('Temperature C', 'Temperature')]
COLOR = {**{f: '#d62728' for f in ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'HGB', 'HCT']},
         **{f: '#f79796' for f in ['WBC', 'Platelets', 'Neut', 'Mono', 'Lymph', 'Baso', 'Eos']},
         **{f: '#279631' for f in ['Glucose', 'Calcium', 'Sodium', 'Potassium', 'CO2', 'Creatinine', 'Chloride', 'BUN']},
         **{f: '#f77f13' for f in ['Albumin', 'Protein, Total', 'ALP', 'ALT', 'AST', 'Bilirubin, Total']},
         **{f: '#2177ad' for f in ['BMI', 'Blood Pressure BP Systolic', 'Blood Pressure Diastolic', 'O2',
                                   'Pulse Heart Rate', 'Resp Rate', 'Temperature C']}}
KEYS = [k for k, _ in FEATURES]
XLAB_SHORT = '(short-term intra-patient, 60-day windows / inter-patient)'

rows = []
for ds, rel in FILES.items():
    d = pd.read_csv(os.path.join(CV, rel)).set_index('feature')
    for k in KEYS:
        if k in d.index:
            rows.append({'dataset': ds, 'feature': k, 'short_term_ratio': d.at[k, 'short_term_cv_ratio'],
                         'long_term_ratio': d.at[k, 'long_term_cv_ratio'],
                         # 95% CI half-width = 1.96 * se / inter-patient CV (as cv_analysis_fixed.py plot 5b);
                         # SD across patients = se * sqrt(n_patients_short) / inter-patient CV
                         'short_term_ci95': 1.96 * d.at[k, 'short_term_se'] / d.at[k, 'inter_patient_cv']
                         if d.at[k, 'inter_patient_cv'] else np.nan,
                         'short_term_sd': d.at[k, 'short_term_se'] * np.sqrt(d.at[k, 'n_patients_short'])
                         / d.at[k, 'inter_patient_cv'] if d.at[k, 'inter_patient_cv'] else np.nan})
long = pd.DataFrame(rows)
long.to_csv(os.path.join(args.out_dir, 'cv_ratio_short_term_by_dataset.csv'), index=False)


def style_y(ax, fs=11):
    y = np.arange(len(FEATURES))
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in FEATURES], fontsize=fs)
    for t, (k, _) in zip(ax.get_yticklabels(), FEATURES):
        t.set_color(COLOR[k])
    ax.set_ylim(len(FEATURES) - 0.4, -0.6)
    ax.axvline(0.5, color='#e74c3c', ls='--', lw=1.8, alpha=0.8, label='CV ratio = 0.5')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', ls='--', lw=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    return y


def save(fig, name, fixed_size=True):
    """fixed_size: keep the 9x7 in canvas (no bbox trim) to match cv_analysis_fixed.py panel aspect."""
    if fixed_size:
        fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'{name}.{ext}'), dpi=200,
                    **({} if fixed_size else {'bbox_inches': 'tight'}))
    plt.close(fig)


# (a) across 12 datasets: mean +/- SD
g = long.groupby('feature').short_term_ratio
m, sd = g.mean().reindex(KEYS), g.std().reindex(KEYS)
fig, ax = plt.subplots(figsize=(9, 7))
y = style_y(ax)
ax.errorbar(m.values, y, xerr=sd.values, fmt='D', color='black', ms=3, elinewidth=1, capsize=2.5,
            label=f'Mean/std across {long.dataset.nunique()} datasets')
ax.set_xlabel(f'CV ratio across {long.dataset.nunique()} datasets\n{XLAB_SHORT}', fontsize=13)
ax.legend(frameon=True, fontsize=10, loc='upper right')
save(fig, 'cv_ratio_short_term_across12')

# (a') 60-day (filled) vs entire record (open), offset vertically
gl = long.groupby('feature').long_term_ratio
ml, sdl = gl.mean().reindex(KEYS), gl.std().reindex(KEYS)
fig, ax = plt.subplots(figsize=(9, 7))
y = style_y(ax)
ax.errorbar(ml.values, y + 0.18, xerr=sdl.values, fmt='D', mfc='white', mec='gray', color='gray', ms=3.5,
            elinewidth=1, capsize=2, label='entire record')
ax.errorbar(m.values, y - 0.18, xerr=sd.values, fmt='D', color='black', ms=3.5, elinewidth=1, capsize=2,
            label='60-day windows')
ax.set_xlabel(f'CV ratio across {long.dataset.nunique()} datasets\n(intra-patient / inter-patient; mean ± SD)',
              fontsize=10)
ax.legend(frameon=True, fontsize=10, loc='upper right')
save(fig, 'cv_ratio_short_vs_long_across12')

# (b) MSK-CAIA
ms = long[long.dataset == 'MSK-CAIA'].set_index('feature').reindex(KEYS)
fig, ax = plt.subplots(figsize=(9, 7))
y = style_y(ax)
ax.errorbar(ms.short_term_ratio.values, y, xerr=ms.short_term_ci95.fillna(0).values, fmt='o', color='#5B9AA9',
            ms=5, mec='black', mew=0.4, ecolor='#5B9AA9', elinewidth=1.2, capsize=3, zorder=3,
            label='60-day ratio (95% parametric CI)')
ax.set_xlabel(f'CV ratio in MSK dataset\n{XLAB_SHORT}', fontsize=13)
ax.legend(frameon=True, fontsize=10, loc='upper right')
save(fig, 'cv_ratio_short_term_MSK_CAIA')

# MSK-CAIA with +/- 1 SD across patients (clipped at 0)
fig, ax = plt.subplots(figsize=(9, 7))
y = style_y(ax)
lo = np.minimum(ms.short_term_sd.fillna(0).values, ms.short_term_ratio.values)
ax.errorbar(ms.short_term_ratio.values, y, xerr=[lo, ms.short_term_sd.fillna(0).values], fmt='o', color='#5B9AA9',
            ms=5, mec='black', mew=0.4, ecolor='#5B9AA9', elinewidth=1.2, capsize=3, zorder=3,
            label='60-day ratio (+/- 1 SD across patients)')
ax.set_xlabel(f'CV ratio in MSK dataset\n{XLAB_SHORT}', fontsize=13)
ax.legend(frameon=True, fontsize=10, loc='upper right')
save(fig, 'cv_ratio_short_term_MSK_CAIA_sd')

# MSK-CAIA: short-term (filled) vs long-term (open), sensitivity panel
fig, ax = plt.subplots(figsize=(9, 7))
y = style_y(ax)
ax.scatter(ms.long_term_ratio.values, y, s=28, facecolor='none', edgecolor='gray', lw=0.9, zorder=3,
           label='entire record')
ax.scatter(ms.short_term_ratio.values, y, s=28, color='#5B9AA9', edgecolor='black', lw=0.4, zorder=4,
           label='60-day windows')
for yy, a_, b_ in zip(y, ms.short_term_ratio.values, ms.long_term_ratio.values):
    ax.plot([a_, b_], [yy, yy], color='lightgray', lw=0.8, zorder=2)
ax.set_xlabel('CV ratio in MSK dataset\n(intra-patient / inter-patient)', fontsize=13)
ax.legend(frameon=True, fontsize=10, loc='upper right')
save(fig, 'cv_ratio_short_vs_long_MSK_CAIA')

# per dataset: short-term (filled) vs long-term (open)
dss = list(FILES)
fig, axes = plt.subplots(2, 6, figsize=(22, 13), sharey=True)
for ax, ds in zip(axes.ravel(), dss):
    d = long[long.dataset == ds].set_index('feature').reindex(KEYS)
    y = style_y(ax, fs=8)
    ax.scatter(d.long_term_ratio, y, s=22, facecolor='none', edgecolor='gray', lw=0.8, label='long-term (whole record)')
    ax.scatter(d.short_term_ratio, y, s=22, color='#5B9AA9', edgecolor='black', lw=0.4, label='short-term (60-day)')
    ax.set_title(ds, fontsize=11, loc='left')
    ax.set_xlabel('CV ratio', fontsize=9)
h, l = axes.ravel()[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10)
fig.suptitle('Intra- to inter-patient CV ratio per dataset: 60-day windows vs whole record', fontsize=13, x=0.01,
             ha='left')
plt.tight_layout(rect=(0, 0.03, 1, 0.97))
save(fig, 'cv_ratio_short_vs_long_by_dataset', fixed_size=False)

# short vs long agreement per dataset
from scipy.stats import spearmanr
sim = [(ds, spearmanr(g2.short_term_ratio, g2.long_term_ratio, nan_policy='omit')[0],
        g2.short_term_ratio.median(), g2.long_term_ratio.median()) for ds, g2 in long.groupby('dataset', sort=False)]
print(pd.DataFrame(sim, columns=['dataset', 'spearman_short_vs_long', 'median_short', 'median_long']).round(3)
      .to_string(index=False))
print(f'Saved to {args.out_dir}')
