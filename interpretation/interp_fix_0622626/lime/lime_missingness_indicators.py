"""Share of LIME importance carried by the missingness indicators vs. the values, per feature and task, from the
per-fold pickles of lime_importance_risk_fixed.py.

Static-feature indicators (gender, age) are excluded.

Usage: python lime_missingness_indicators.py [--lime_dir DIR] [--top_k 15]  (--top_k 0: all features)
"""
import os, re, glob, pickle, argparse, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42   # editable text in PDF
plt.rcParams['ps.fonttype'] = 42
from scipy.stats import spearmanr

_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(os.path.isdir(os.path.join(_R, d)) for d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS, CBC_DIFFERENTIAL_TESTS

parser = argparse.ArgumentParser()
parser.add_argument('--lime_dir', default=sparc_paths.SPARC_LIME_BF_DIR)
parser.add_argument('--dataset', default='MSK_CAIA')
parser.add_argument('--top_k', type=int, default=15, help='features per panel; 0 = all features')
args = parser.parse_args()

T = 20
STATIC = ['gender', 'age at seq']
NAMES = STATIC + list(BASIC_TESTS) + list(CBC_DIFFERENTIAL_TESTS) + list(BASIC_VITALS)   # parsimonious modules
# display names, as in the paper's feature-importance heatmaps
DISPLAY = {'gender': 'Gender', 'age at seq': 'Age', 'Neut': 'Neutrophil', 'Mono': 'Monocytes', 'Lymph': 'Lymphocytes',
           'Baso': 'Basophils', 'Eos': 'Eosinophils', 'Blood Pressure BP Systolic': 'Systolic BP',
           'Blood Pressure Diastolic': 'Diastolic BP', 'Pulse Heart Rate': 'Pulse Hrt Rate',
           'Temperature C': 'Temperature'}
LABEL_COLOR = {**{f: '#d62728' for f in ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'HGB', 'HCT']},
               **{f: '#f79796' for f in ['WBC', 'Platelets', 'Neut', 'Mono', 'Lymph', 'Baso', 'Eos']},
               **{f: '#279631' for f in ['Glucose', 'Calcium', 'Sodium', 'Potassium', 'CO2', 'Creatinine', 'Chloride',
                                         'BUN']},
               **{f: '#f77f13' for f in ['Albumin', 'Protein, Total', 'ALP', 'ALT', 'AST', 'Bilirubin, Total']},
               **{f: '#2177ad' for f in BASIC_VITALS}, **{f: 'gray' for f in STATIC}}
TASKS = {'neutropenia_risk': 'Neutropenia', 'febrile_neutropenia_risk': 'Febrile neutropenia',
         'anemia_risk': 'Severe anemia', 'AKI_risk': 'Acute kidney injury',
         'Liver_toxicity_risk': 'Grade 3 liver injury', 'binary': '180D hosp. mortality',
         'binary_icu_pre': '30D ICU mortality'}


def fold_agreement(M):
    r = [spearmanr(M[i], M[j])[0] for i in range(len(M)) for j in range(i + 1, len(M))]
    return float(np.median(r)) if r else np.nan


rows, summary = [], []
for task, title in TASKS.items():
    files = sorted(f for f in glob.glob(f'{args.lime_dir}/lime_df_val_fold_*_{task}_{args.dataset}.pickle')
                   if re.search(rf'fold_\d+_{task}_{args.dataset}\.pickle$', f))
    if not files:
        print(f'{task}: no LIME pickles, skipped')
        continue
    per_fold = []
    for f in files:
        with open(f, 'rb') as fh:
            df = pickle.load(fh)
        nx = sum(1 for c in df.columns if re.match(r'^t\d+_f\d+$', c))
        F = nx // T
        base = (F - 3) // 2          # layout: values, time stamp, indicators (values + time), padding
        assert base == len(NAMES), f'{f}: {base} base features, expected {len(NAMES)}'
        per_fold.append(df.iloc[:, :nx].values.reshape(-1, T, F).mean(axis=0))      # [T, F], signed
    S = np.abs(np.array(per_fold))                                                  # [folds, T, F]
    val_f = S[:, :, :base].sum(axis=1)                                              # [folds, base]
    ind_f = S[:, :, base + 1:2 * base + 1].sum(axis=1)
    time_f = S[:, :, base].sum(axis=1)
    lab = np.arange(len(STATIC), base)
    ind_f[:, :len(STATIC)] = 0.0                                                    # static: never missing
    val, ind = val_f.mean(0), ind_f.mean(0)
    total = val.sum() + time_f.mean() + ind.sum()
    for i, n in enumerate(NAMES):
        rows.append({'task': task, 'feature': n, 'value_importance': val[i], 'indicator_importance': ind[i],
                     'value_share_pct': 100 * val[i] / total, 'indicator_share_pct': 100 * ind[i] / total})
    summary.append({'task': task, 'n_folds': len(files),
                    'values_pct': 100 * val.sum() / total, 'time_pct': 100 * time_f.mean() / total,
                    'indicators_pct': 100 * ind.sum() / total,
                    'fold_agreement_values': fold_agreement(val_f[:, lab]),
                    'fold_agreement_indicators': fold_agreement(ind_f[:, lab])})

res, summ = pd.DataFrame(rows), pd.DataFrame(summary)
stem = os.path.join(args.lime_dir, f'lime_missingness_indicators_{args.dataset}')
res.to_csv(stem + '.csv', index=False)
summ.to_csv(stem + '_summary.csv', index=False)
print(summ.round(2).to_string(index=False))

# one panel per task: top-k lab/vital features by value + indicator share, stacked (static features have no
# indicator, so they are left out of the figure; they stay in the CSV and in the share denominator)
tasks = list(summ['task'])
ncol = 4
nrow = int(np.ceil(len(tasks) / ncol))
n_show = len(NAMES) - len(STATIC) if args.top_k <= 0 else args.top_k
fig_stem = stem + ('_allfeatures' if args.top_k <= 0 else '')
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 0.28 * n_show * nrow + 1.5 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, task in zip(axes, tasks):
    g = res[(res.task == task) & ~res.feature.isin(STATIC)].assign(tot=lambda x: x.value_share_pct + x.indicator_share_pct)
    g = g.nlargest(n_show, 'tot').iloc[::-1]
    y = np.arange(len(g))
    ax.barh(y, g.value_share_pct, color='#2C5F8D', label='Value')
    ax.barh(y, g.indicator_share_pct, left=g.value_share_pct, color='#E39B3B', label='Measured (indicator)')
    ax.set_ylim(-0.6, len(g) - 0.4)   # no empty band above/below the bars
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(f, f) for f in g.feature], fontsize=8)
    for t, f in zip(ax.get_yticklabels(), g.feature):
        t.set_color(LABEL_COLOR.get(f, 'black'))
    s = summ.set_index('task').loc[task]
    ax.set_title(f'{TASKS[task]}\nindicators: {s.indicators_pct:.0f}% of attribution\nfold agreement: '
                 f'values {s.fold_agreement_values:.2f}, indicators {s.fold_agreement_indicators:.2f}',
                 fontsize=9, loc='left')
    ax.set_xlabel('Share of total LIME attribution (%)', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for ax in axes[len(tasks):]:
    ax.axis('off')
axes[0].legend(fontsize=8, frameon=False, loc='lower right')
fig.suptitle('LIME importance of lab values and of whether they were measured (gender and age '
             'excluded)', fontsize=12, x=0.01, ha='left')
fig.text(0.01, -0.01, 'Indicators are perturbed by LIME as continuous values independently of the lab values, so '
         'indicator attributions are exploratory.', fontsize=8, style='italic', color='dimgray')
plt.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))   # keep the suptitle clear of the panels
plt.savefig(fig_stem + '.pdf', bbox_inches='tight', dpi=300)
plt.savefig(fig_stem + '.png', bbox_inches='tight', dpi=150)
print(f'Saved {fig_stem}.{{pdf,png}}, {stem}.csv and {stem}_summary.csv')
