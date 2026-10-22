#!/usr/bin/env python3
"""INSIGHT scaling: learning curves of pooled fine-tuning on subsampled training data, and pooled vs site-specific
fine-tuning at matched training size. Reads results from INSIGHT_scaling/ (or SPARC_INSIGHT_SCALING_DIR).
Usage: python scaling_analysis.py [--n_models 10]
"""
# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models'), os.path.join(_R, 'results_vis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
RESULTS = sparc_paths.INSIGHT_RESULTS_DIR
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams['pdf.fonttype'] = 42
import argparse
B = os.environ.get('SPARC_INSIGHT_SCALING_DIR', os.path.join(RESULTS, 'INSIGHT_scaling'))
_ap = argparse.ArgumentParser()
_ap.add_argument('--n_models', type=int, default=5, choices=[5, 10],
                 help='5: res/ vs res_baseline5/; 10: res10/ vs the original 10-model results (res_all)')
N_MODELS = _ap.parse_args().n_models
OUT = os.path.join(B, 'analysis' if N_MODELS == 5 else 'analysis10')
os.makedirs(OUT, exist_ok=True)
SITES = list('ABCDEF')
SITE_COLOR = dict(zip(SITES, plt.rcParams['axes.prop_cycle'].by_key()['color']))   # same in every figure
TASKS = ['AKI', 'febrile_neutropenia', 'Liver_toxicity', 'anemia', 'neutropenia', 'binary']
TASK_NAME = {'AKI': 'Acute kidney injury', 'febrile_neutropenia': 'Febrile neutropenia',
             'Liver_toxicity': 'Grade 3 liver injury', 'anemia': 'Severe anemia', 'neutropenia': 'Neutropenia',
             'binary': '180-day mortality'}


def ensemble_auroc(res_dir, site, task):
    f = os.path.join(res_dir, site, 'eval_across_models_insight',
                     f'model_performance_dict_parsimonious_{task}_SPARC-P_insight.csv')
    if not os.path.exists(f):
        return np.nan
    try:
        d = pd.read_csv(f)
    except pd.errors.EmptyDataError:   # still being written by a running evaluation
        return np.nan
    col = 'auroc' if task == 'binary' else 'test_auroc_nextinf_weighted'
    r = d[d['model'].astype(str).str.endswith('_ensemble')]
    if r.empty or col not in d or pd.isna(r[col].iloc[0]):
        return np.nan
    return float(r[col].iloc[0]) * 100


def fold_aurocs(res_dir, site, task):
    """{fold: AUROC (%)} of the individual fine-tuned models (rows 0-9 of the eval CSV), or {}"""
    f = os.path.join(res_dir, site, 'eval_across_models_insight',
                     f'model_performance_dict_parsimonious_{task}_SPARC-P_insight.csv')
    if not os.path.exists(f):
        return {}
    try:
        d = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return {}
    col = 'auroc' if task == 'binary' else 'test_auroc_nextinf_weighted'
    r = d[~d['model'].astype(str).str.endswith('_ensemble')]
    return {int(k): float(v) * 100 for k, v in zip(r['val_fold'], r[col]) if pd.notna(v)}


plan = pd.read_csv(os.path.join(B, 'plan.csv'))
# the 30 task x site cells of the pooled vs site-specific comparison (cells with too few positives, which give
# degenerate AUROCs of 0, 50 or 100, are not in it)
_cells = pd.read_csv(os.path.join(RESULTS, 'INSIGHT_finetune_lr3e4', 'vis', 'zeroshot_vs_finetune_scatter_data.csv'))
CELLS = set(map(tuple, _cells.loc[_cells.strategy == 'Pooled finetuning (from MSK finetune)', ['task', 'site']].values))
RES = os.path.join(B, 'res' if N_MODELS == 5 else 'res10')
_BASE = (os.path.join(B, 'res_baseline5') if N_MODELS == 5 else
         os.path.join(RESULTS, 'INSIGHT_finetune_lr3e4', 'res_all'))
FULL_POOLED = os.path.join(_BASE, 'insight-finetune-pooled-mskfinetuned')
SITE_SPEC = os.path.join(_BASE, 'insight-finetune-mskfinetuned')

# learning curves: fraction of pooled training patients -> per-site AUROC
frac_val = {'1/8': 1 / 8, '1/4': 1 / 4, '1/2': 1 / 2}
rows = []
for _, r in plan[plan.experiment == 'learning_curve'].iterrows():
    for s in SITES:
        rows.append({'task': r.task, 'site': s, 'fraction': frac_val[r.fraction],
                     'auroc': ensemble_auroc(os.path.join(RES, r.run_id), s, r.task)})
for t in TASKS:
    for s in SITES:
        rows.append({'task': t, 'site': s, 'fraction': 1.0, 'auroc': ensemble_auroc(FULL_POOLED, s, t)})
lc = pd.DataFrame(rows).dropna(subset=['auroc'])
lc = lc[[(t, si) in CELLS for t, si in zip(lc.task, lc.site)]]
lc.to_csv(os.path.join(OUT, 'learning_curves.csv'), index=False)

if len(lc):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, t in zip(axes.ravel(), TASKS):
        g = lc[lc.task == t]
        for s, gs in g.groupby('site'):
            gs = gs.sort_values('fraction')
            ax.plot(gs.fraction, gs.auroc, marker='o', lw=1, ms=3, alpha=0.55, label=f'site {s}', color=SITE_COLOR[s])
        # mean over the sites that have every fraction so far, so the curve is not shifted by sites dropping in or out
        full_sites = g.groupby('site').fraction.nunique()
        full_sites = full_sites[full_sites == g.fraction.nunique()].index
        m = g[g.site.isin(full_sites)].groupby('fraction').auroc.mean()
        if len(m):
            ax.plot(m.index, m.values, color='black', lw=2.2, marker='o', label='mean over sites')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1]); ax.set_xticklabels(['1/8', '1/4', '1/2', 'all'])
        ax.set_title(TASK_NAME[t], fontsize=11, loc='left')
        ax.set_ylabel('AUROC (%)' if t == 'binary' else 'Weighted AUROC (%)', fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Fraction of pooled INSIGHT training patients', fontsize=9)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=9)
    fig.suptitle(f'Pooled INSIGHT fine-tuning of SPARC-P on subsampled training data ({N_MODELS}-model ensembles; '
                 'per-site test splits)', fontsize=11, x=0.01, ha='left')
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT, f'learning_curves.{ext}'), bbox_inches='tight', dpi=200)
    plt.close()

# learning curves with uncertainty, as the titration line plots (vis_eval_4_titration_pertask_lineplot_with_
# significance.py): mean over the individual fine-tuned models (folds) with a 95% t-interval. Per task, each fold's
# value is the mean AUROC over the task's sites, so the band reflects variation between fine-tuned models.
from scipy import stats as _st
rows = []
_runs = [(r.task, frac_val[r.fraction], os.path.join(RES, r.run_id)) for _, r in
         plan[plan.experiment == 'learning_curve'].iterrows()] + [(t, 1.0, FULL_POOLED) for t in TASKS]
for t, fr, rd in _runs:
    per_site = {s_: fold_aurocs(rd, s_, t) for s_ in SITES if (t, s_) in CELLS}
    folds = sorted(set.intersection(*[set(v) for v in per_site.values() if v]) if any(per_site.values()) else [])
    for k in folds:
        rows.append({'task': t, 'fraction': fr, 'fold': k,
                     'auroc_mean_over_sites': np.mean([per_site[s_][k] for s_ in per_site if per_site[s_]])})
lcf = pd.DataFrame(rows)
lcf.to_csv(os.path.join(OUT, 'learning_curves_per_fold.csv'), index=False)
if len(lcf):
    summ_lc = []
    for (t, fr), g in lcf.groupby(['task', 'fraction']):
        v = g.auroc_mean_over_sites.values
        lo, hi = _st.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=_st.sem(v)) if len(v) > 1 else (v.mean(), v.mean())
        summ_lc.append({'task': t, 'fraction': fr, 'n_models': len(v), 'mean': v.mean(), 'ci_lower': lo, 'ci_upper': hi})
    summ_lc = pd.DataFrame(summ_lc)
    summ_lc.to_csv(os.path.join(OUT, 'learning_curves_mean_ci.csv'), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, t in zip(axes.ravel(), TASKS):
        g = summ_lc[summ_lc.task == t].sort_values('fraction')
        ax.fill_between(g.fraction, g.ci_lower, g.ci_upper, color='#2C5F8D', alpha=0.2, lw=0)
        ax.plot(g.fraction, g['mean'], color='#2C5F8D', marker='o', lw=2)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1]); ax.set_xticklabels(['1/8', '1/4', '1/2', 'all'])
        ax.set_title(TASK_NAME[t], fontsize=11, loc='left')
        ax.set_ylabel('AUROC (%)' if t == 'binary' else 'Weighted AUROC (%)', fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Fraction of pooled INSIGHT training patients', fontsize=9)
    fig.suptitle(f'Pooled INSIGHT fine-tuning on subsampled training data: mean over sites, mean and 95% CI over '
                 f'the {N_MODELS} fine-tuned models', fontsize=11, x=0.01, ha='left')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT, f'learning_curves_ci.{ext}'), bbox_inches='tight', dpi=200)
    plt.close()

# learning curves of the 10-model ENSEMBLE (as in the zero-shot vs fine-tuning scatterplot), with 95% CIs from the
# per-site bootstrap over test patients (eval_insight_multimodel_ci.py -nb 1000; subsampled runs in res10_ci/, full
# data in INSIGHT_weightedci/res_ft_pooled). Line = mean over the task's sites; the sites' test sets are independent,
# so CI = mean +/- 1.96 * sqrt(sum SE_i^2) / k with SE_i = (upper_i - lower_i) / (2 * 1.96).
def ensemble_ci(res_dir, site, task):
    f = os.path.join(res_dir, site, 'eval_across_models_insight',
                     f'model_performance_dict_parsimonious_{task}_SPARC-P_insight.csv')
    try:
        d = pd.read_csv(f)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    r = d[d['model'].astype(str).str.endswith('_ensemble')]
    cols = (('auroc', 'auroc_ci_lower', 'auroc_ci_upper') if task == 'binary' else
            ('test_auroc_nextinf_weighted', 'auroc_weighted_ci_lower', 'auroc_weighted_ci_upper'))
    if r.empty or not all(c in d for c in cols) or r[list(cols)].isna().any(axis=None):
        return None
    return tuple(float(r[c].iloc[0]) * 100 for c in cols)


if N_MODELS == 10:
    _CI_RES = os.path.join(B, 'res10_ci')
    _CI_FULL = os.path.join(RESULTS, 'INSIGHT_weightedci', 'res_ft_pooled')
    rows = []
    _runs = [(r.task, frac_val[r.fraction], os.path.join(_CI_RES, r.run_id)) for _, r in
             plan[plan.experiment == 'learning_curve'].iterrows()] + [(t, 1.0, _CI_FULL) for t in TASKS]
    for t, fr, rd in _runs:
        sites = [s_ for s_ in SITES if (t, s_) in CELLS]
        vals = {s_: ensemble_ci(rd, s_, t) for s_ in sites}
        if any(v is None for v in vals.values()):
            continue                                   # all sites needed, so every point averages the same sites
        m = np.mean([v[0] for v in vals.values()])
        se = np.sqrt(sum(((v[2] - v[1]) / (2 * 1.96)) ** 2 for v in vals.values())) / len(vals)
        rows.append({'task': t, 'fraction': fr, 'n_sites': len(vals), 'mean': m,
                     'ci_lower': m - 1.96 * se, 'ci_upper': m + 1.96 * se})
    ens = pd.DataFrame(rows)
    ens.to_csv(os.path.join(OUT, 'learning_curves_ensemble_ci.csv'), index=False)
    if len(ens):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
        for ax, t in zip(axes.ravel(), TASKS):
            # same layout as learning_curves.pdf: per-site ensemble lines, plus the mean with its 95% CI band
            for s_, gs in lc[lc.task == t].groupby('site'):
                gs = gs.sort_values('fraction')
                ax.plot(gs.fraction, gs.auroc, marker='o', lw=1, ms=3, alpha=0.55, label=f'site {s_}',
                        color=SITE_COLOR[s_])
            g = ens[ens.task == t].sort_values('fraction')
            ax.fill_between(g.fraction, g.ci_lower, g.ci_upper, color='black', alpha=0.12, lw=0,
                            label='95% CI of the mean')
            ax.plot(g.fraction, g['mean'], color='black', marker='o', lw=2.2, label='mean over sites')
            ax.set_xscale('log', base=2)
            ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1]); ax.set_xticklabels(['1/8', '1/4', '1/2', 'all'])
            ax.set_title(TASK_NAME[t], fontsize=11, loc='left')
            ax.set_ylabel('AUROC (%)' if t == 'binary' else 'Weighted AUROC (%)', fontsize=9)
            ax.grid(alpha=0.3, lw=0.5)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel('Fraction of pooled INSIGHT training patients', fontsize=9)
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=8, frameon=False, fontsize=9)
        fig.suptitle('Pooled INSIGHT fine-tuning of SPARC-P on subsampled training data (10-model ensembles; per-site '
                     'test splits; 95% CI of the mean from per-site bootstrap)', fontsize=11, x=0.01, ha='left')
        plt.tight_layout(rect=(0, 0.05, 1, 0.96))
        for ext in ('pdf', 'png'):
            plt.savefig(os.path.join(OUT, f'learning_curves_ensemble_ci.{ext}'), bbox_inches='tight', dpi=200)
        plt.close()

# size-matched: pooled subsample of the site's size vs site-specific (and full pooled) on that site
rows = []
for _, r in plan[plan.experiment == 'size_matched'].iterrows():
    s, t = r.matched_site, r.task
    rows.append({'task': t, 'site': s, 'n_train_patients_site': r.site_n_train_patients,
                 'pooled_fraction': float(r.fraction.split('=')[1]),
                 'pooled_size_matched': ensemble_auroc(os.path.join(RES, r.run_id), s, t),
                 'site_specific': ensemble_auroc(SITE_SPEC, s, t),
                 'pooled_full': ensemble_auroc(FULL_POOLED, s, t)})
sm = pd.DataFrame(rows)
sm = sm[[(t, si) in CELLS for t, si in zip(sm.task, sm.site)]]
sm['diff_matched_minus_site'] = sm.pooled_size_matched - sm.site_specific
sm['diff_full_minus_site'] = sm.pooled_full - sm.site_specific
sm.to_csv(os.path.join(OUT, 'size_matched.csv'), index=False)
ok = sm.dropna(subset=['pooled_size_matched', 'site_specific', 'pooled_full'])
summ = []
for name, col in [('pooled full data vs site-specific', 'diff_full_minus_site'),
                  ('pooled size-matched vs site-specific', 'diff_matched_minus_site')]:
    v = ok[col]
    summ.append({'comparison': name, 'n_cells': len(v), 'median_diff_auroc_pts': v.median() if len(v) else np.nan,
                 'mean_diff_auroc_pts': v.mean() if len(v) else np.nan, 'n_pooled_better': int((v > 0).sum()),
                 'wilcoxon_p': wilcoxon(v).pvalue if len(v) >= 6 else np.nan})
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT, 'size_matched_summary.csv'), index=False)
print(f'learning-curve points: {len(lc)}; size-matched cells complete: {len(ok)}/{len(sm)}')
print(summ.round(4).to_string(index=False))

if len(ok):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lo = np.nanmin(ok[['site_specific', 'pooled_size_matched', 'pooled_full']].values) - 2
    hi = np.nanmax(ok[['site_specific', 'pooled_size_matched', 'pooled_full']].values) + 2
    ax.plot([lo, hi], [lo, hi], color='gray', ls='--', lw=1)
    ax.scatter(ok.site_specific, ok.pooled_full, s=28, color='#9ecae1', edgecolor='k', lw=0.4,
               label='pooled, all data')
    ax.scatter(ok.site_specific, ok.pooled_size_matched, s=34, color='#2C5F8D', edgecolor='k', lw=0.4,
               label='pooled, subsampled to the site\'s size')
    for _, r in ok.iterrows():
        ax.plot([r.site_specific] * 2, [r.pooled_full, r.pooled_size_matched], color='gray', lw=0.5, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Site-specific fine-tuning, AUROC (%)'); ax.set_ylabel('Pooled fine-tuning, AUROC (%)')
    ax.set_title('Pooled vs site-specific fine-tuning at equal training size\n(task x site cells; above the '
                 'line = pooled better)', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT, f'size_matched.{ext}'), bbox_inches='tight', dpi=200)
    plt.close()
