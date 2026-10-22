"""Bootstrap CIs and pairwise significance for the eval_1 models comparison.

The eval script computes draws, CIs and p-values and writes a per-task cache (a draws pickle and a
plot-ready summary CSV); the vis script only loads the CSV. Resampling is stratified within
infusion groups, and all models share each draw (paired) when their test rows are aligned.
Metric variants: weighted_auroc, overall_auroc, weighted_auprc, overall_auprc.
"""

import os
import pickle
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
from sklearn.metrics import average_precision_score as _sk_average_precision_score


# Fast numpy AUROC/AP for the bootstrap loop; same tie handling as sklearn (average ranks for
# AUROC, distinct thresholds for AP). SPARC_BOOT_SKLEARN=1 uses sklearn instead.
def _fast_roc_auc_score(y_true, y_score):
    y = np.asarray(y_true).ravel().astype(bool)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y.sum()); n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError('Only one class present in y_true.')
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_average_precision_score(y_true, y_score):
    y = np.asarray(y_true).ravel().astype(np.float64)
    s = np.asarray(y_score, dtype=np.float64).ravel()
    order = np.argsort(-s, kind='mergesort')
    s = s[order]; y = y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]   # last index of each distinct score
    tps = np.cumsum(y)[last]
    fps = (last + 1) - tps
    if tps[-1] == 0:
        return 0.0
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


if os.environ.get('SPARC_BOOT_SKLEARN', '0') == '1':
    roc_auc_score, average_precision_score = _sk_roc_auc_score, _sk_average_precision_score
else:
    roc_auc_score, average_precision_score = _fast_roc_auc_score, _fast_average_precision_score


METRIC_VARIANTS = ['weighted_auroc', 'overall_auroc', 'weighted_auprc', 'overall_auprc']


# stratified bootstrap

def prepare_context(d, option):
    """Labels and usable infusion groups (with fixed weights) from the observed data.

    Groups that are single-class in the observed data are dropped here.
    """
    y_events = np.asarray(d['y_events'])
    time_to_events = np.asarray(d['time_to_events'])
    num_infusions = np.asarray(d['num_infusions'])

    # same substring check as calculate_weighted_c_index_by_infusion
    is_risk = 'risk' in option
    if is_risk:
        labels = ((y_events == 1) & (time_to_events == 1)).astype(np.int64)
    else:
        labels = (y_events == 1).astype(np.int64)

    unique_groups = np.unique(num_infusions)

    group_row_idx = []   # row indices per usable group
    group_sizes = []     # observed size per group = its weight
    for g in unique_groups:
        rows = np.flatnonzero(num_infusions == g)
        n_g = rows.size
        if n_g == 0:
            continue
        n_pos = int(labels[rows].sum())
        if n_pos == 0 or n_pos == n_g:
            continue  # single-class in observed data
        group_row_idx.append(rows)
        group_sizes.append(n_g)

    total_n = int(sum(group_sizes)) if group_sizes else 0
    return {
        'labels': labels,
        'group_row_idx': group_row_idx,
        'group_sizes': group_sizes,
        'total_n': total_n,
    }


def _one_iteration_all_models(preds_by_model, ctx, rng, model_names):
    """One stratified resample shared by all models and metric variants.

    Returns {model: {metric_variant: value}}.
    """
    group_row_idx = ctx['group_row_idx']
    group_sizes = ctx['group_sizes']
    total_n = ctx['total_n']
    labels = ctx['labels']

    nan_out = {m: {v: np.nan for v in METRIC_VARIANTS} for m in model_names}
    if total_n == 0:
        return nan_out

    # resample within each group to its observed size
    resampled_group_rows = [g_rows[rng.randint(0, n_g, size=n_g)]
                            for g_rows, n_g in zip(group_row_idx, group_sizes)]
    pooled_rows = np.concatenate(resampled_group_rows)
    y_pooled = labels[pooled_rows]
    n_pos_pooled = int(y_pooled.sum())
    pooled_valid = 0 < n_pos_pooled < y_pooled.size

    group_labels = [labels[gr] for gr in resampled_group_rows]
    # a resampled group can still be single-class; see WEIGHTED_MODE
    group_valid = [0 < int(yl.sum()) < yl.size for yl in group_labels]

    out = {}
    for m in model_names:
        preds_full = preds_by_model[m]
        res = {}
        w_auroc = _weighted(resampled_group_rows, group_labels, group_valid,
                            group_sizes, total_n, preds_full, roc_auc_score)
        o_auroc = float(roc_auc_score(y_pooled, preds_full[pooled_rows])) if pooled_valid else np.nan
        w_auprc = _weighted(resampled_group_rows, group_labels, group_valid,
                            group_sizes, total_n, preds_full, average_precision_score)
        o_auprc = float(average_precision_score(y_pooled, preds_full[pooled_rows])) if pooled_valid else np.nan

        res['weighted_auroc'] = w_auroc
        res['overall_auroc'] = o_auroc
        res['weighted_auprc'] = w_auprc
        res['overall_auprc'] = o_auprc
        out[m] = res
    return out


# Groups that are single-class in a resample: 'drop' (default) skips them and renormalises the
# weights, as eval_utils.nextinf_auroc_weighted_overall does; 'strict' makes the
# whole resample NaN, which leaves very few valid draws on sparse tasks.
WEIGHTED_MODE = os.environ.get('SPARC_BOOT_WEIGHTED_MODE', 'drop')


def _weighted(resampled_group_rows, group_labels, group_valid, group_sizes, total_n,
              preds_full, metric_fn):
    """Size-weighted mean of metric_fn over the groups of one resample."""
    num = 0.0
    den = 0
    for gr, yl, ok, n_g in zip(resampled_group_rows, group_labels, group_valid, group_sizes):
        if not ok:
            if WEIGHTED_MODE == 'strict':
                return np.nan
            continue
        num += metric_fn(yl, preds_full[gr]) * n_g
        den += n_g
    if den == 0:
        return np.nan
    return float(num / (total_n if WEIGHTED_MODE == 'strict' else den))


# full-task bootstrap


def _models_natively_aligned(preds_by_model, model_names, reference_model):
    """True if every model has the reference's y_events, time_to_events and num_infusions
    element-for-element (same rows, same order). No realignment is attempted (no sample ID)."""
    ref = preds_by_model[reference_model]
    ref_y = np.asarray(ref['y_events'])
    ref_t = np.asarray(ref['time_to_events'])
    ref_g = np.asarray(ref['num_infusions'])
    n_ref = ref_y.shape[0]
    for m in model_names:
        if m == reference_model:
            continue
        d = preds_by_model[m]
        y = np.asarray(d['y_events'])
        if (y.shape[0] != n_ref
                or not np.array_equal(y, ref_y)
                or not np.array_equal(np.asarray(d['time_to_events']), ref_t)
                or not np.array_equal(np.asarray(d['num_infusions']), ref_g)):
            return False
    return True


def bootstrap_task(preds_by_model, option, n_bootstrap, seed_entropy, reference_model=None):
    """Stratified bootstrap for one task, all metric variants.

    Paired (one shared draw per iteration) if all models' rows are aligned with the reference;
    otherwise unpaired, with each model resampled independently on its own rows.
    Returns (draws {metric_variant: {model: array}}, paired).
    """
    model_names = list(preds_by_model.keys())
    if reference_model is None:
        reference_model = model_names[0]

    natively_aligned = _models_natively_aligned(preds_by_model, model_names, reference_model)

    draws = {v: {m: np.empty(n_bootstrap, dtype=np.float64) for m in model_names}
             for v in METRIC_VARIANTS}

    if natively_aligned:
        # paired: shared ctx and rng, so draw i is the same rows for every model
        ctx = prepare_context(preds_by_model[reference_model], option)
        preds_arrays = {m: np.asarray(preds_by_model[m]['y_preds']) for m in model_names}
        rng = np.random.RandomState(np.random.SeedSequence(seed_entropy).generate_state(1)[0])
        for i in range(n_bootstrap):
            per_model = _one_iteration_all_models(preds_arrays, ctx, rng, model_names)
            for m in model_names:
                for v in METRIC_VARIANTS:
                    draws[v][m][i] = per_model[m][v]
        return draws, True

    # unpaired: own ctx and an independent sub-seed per model
    ss = np.random.SeedSequence(seed_entropy)
    child_seeds = ss.spawn(len(model_names))
    for mi, m in enumerate(model_names):
        ctx_m = prepare_context(preds_by_model[m], option)
        preds_m = {m: np.asarray(preds_by_model[m]['y_preds'])}
        rng_m = np.random.RandomState(child_seeds[mi].generate_state(1)[0])
        for i in range(n_bootstrap):
            per_model = _one_iteration_all_models(preds_m, ctx_m, rng_m, [m])
            for v in METRIC_VARIANTS:
                draws[v][m][i] = per_model[m][v]
    return draws, False


# point estimates, CIs, p-values

def point_estimate(preds_by_model_entry, option, metric_variant):
    """Point estimate on the observed data, using the same groups as the bootstrap."""
    ctx = prepare_context(preds_by_model_entry, option)
    preds = np.asarray(preds_by_model_entry['y_preds'])
    if ctx['total_n'] == 0:
        return np.nan
    metric_fn = roc_auc_score if 'auroc' in metric_variant else average_precision_score
    if metric_variant.startswith('weighted'):
        labels = ctx['labels']
        num = 0.0
        for gr, n_g in zip(ctx['group_row_idx'], ctx['group_sizes']):
            yl = labels[gr]
            if not (0 < int(yl.sum()) < yl.size):
                return np.nan
            num += metric_fn(yl, preds[gr]) * n_g
        return float(num / ctx['total_n'])
    else:
        labels = ctx['labels']
        pooled = np.concatenate(ctx['group_row_idx']) if ctx['group_row_idx'] else np.array([], dtype=np.int64)
        y_pooled = labels[pooled]
        if not (0 < int(y_pooled.sum()) < y_pooled.size):
            return np.nan
        return float(metric_fn(y_pooled, preds[pooled]))


def ci_from_draws(boot_vals, confidence_level=0.95):
    """Percentile CI, ignoring NaN draws."""
    vals = np.asarray(boot_vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    alpha = 1 - confidence_level
    lo = np.percentile(vals, (alpha / 2) * 100)
    hi = np.percentile(vals, (1 - alpha / 2) * 100)
    return (float(lo), float(hi))


def difference_pvalue(draws_sparc, draws_bench, paired):
    """Two-sided bootstrap p-value for sparc - bench.

    p = 2 * min(P(diff < 0), P(diff > 0)), with ties split evenly between the tails and
    non-finite draws dropped. Unpaired draws are differenced by index (independent samples).
    Returns (mean_diff, p_value, n_valid, prop_tie).
    """
    a = np.asarray(draws_sparc, dtype=np.float64)
    b = np.asarray(draws_bench, dtype=np.float64)

    if paired:
        diffs = a - b
        diffs = diffs[np.isfinite(diffs)]
    else:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            return (np.nan, np.nan, 0, np.nan)
        # independent samples, so differencing by index gives X* - Y*
        k = min(a.size, b.size)
        diffs = a[:k] - b[:k]

    n_valid = diffs.size
    if n_valid == 0:
        return (np.nan, np.nan, 0, np.nan)
    n_pos = np.mean(diffs > 0)
    n_neg = np.mean(diffs < 0)
    n_tie = np.mean(diffs == 0)
    prop_below = n_neg + n_tie / 2.0
    prop_above = n_pos + n_tie / 2.0
    p = min(1.0, 2 * min(prop_below, prop_above))
    return (float(np.mean(diffs)), float(p), int(n_valid), float(n_tie))


# cache paths and orchestration (eval)

def draws_cache_path(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed):
    return os.path.join(
        cache_dir,
        f'bootstrap_draws_{task}_{dataset}{finetuned_suffix}_{modules}_n{n_bootstrap}_seed{rseed}.pickle'
    )


def summary_cache_path(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed):
    return os.path.join(
        cache_dir,
        f'bootstrap_summary_{task}_{dataset}{finetuned_suffix}_{modules}_n{n_bootstrap}_seed{rseed}.csv'
    )


def run_and_cache_task(preds_by_model, *, task, option, dataset, finetuned_suffix, modules,
                       sparc_model, benchmark_models, per_fold_values_by_model, fold_mean_by_model,
                       cache_dir, n_bootstrap, rseed, task_index=0, confidence_level=0.95,
                       alpha=0.05, verbose=True):
    """Bootstrap draws, CIs and p-values for one task; prints them, writes both cache files and
    returns the summary rows and cache paths.

    preds_by_model: {model: {'y_preds', 'y_events', 'time_to_events', 'num_infusions'}};
    per_fold_values_by_model / fold_mean_by_model: {metric_variant: {model: values in 0-1}}, may be {}.
    """
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    model_names = list(preds_by_model.keys())

    # one seed per (rseed, task); all metric variants share the same draws
    seed_entropy = (int(rseed), int(task_index), 0)
    draws, paired = bootstrap_task(preds_by_model, option, n_bootstrap, seed_entropy,
                                   reference_model=sparc_model)
    test_type = 'paired' if paired else 'unpaired'
    if not paired:
        print()
        print("[pairwise_stats] models are not row-aligned (same test set, different row "
              "order,\n  no per-sample ID); using an unpaired two-sample bootstrap, "
              "which is\n  less powerful than the paired test.")
        print()

    # summary rows (plot-ready)
    barstat_rows = []   # per (model, metric): per-fold, fold-mean, point, CI
    pvalue_rows = []    # per (benchmark, metric): mean_diff, p
    for metric_variant in METRIC_VARIANTS:
        for m in model_names:
            point = point_estimate(preds_by_model[m], option, metric_variant)
            lo, hi = ci_from_draws(draws[metric_variant][m], confidence_level)
            folds = (per_fold_values_by_model.get(metric_variant, {}) or {}).get(m, [])
            fm = (fold_mean_by_model.get(metric_variant, {}) or {}).get(m, None)
            barstat_rows.append({
                'task': task, 'model': m, 'metric': metric_variant,
                'point_pct': point * 100 if point == point else None,
                'ci_lower_pct': lo * 100 if lo == lo else None,
                'ci_upper_pct': hi * 100 if hi == hi else None,
                'fold_mean_pct': fm * 100 if (fm is not None and fm == fm) else None,
                'per_fold_pct': ';'.join(f'{v*100:.6f}' for v in folds) if folds else '',
                'n_folds': len(folds),
                'n_bootstrap': n_bootstrap, 'rseed': rseed,
            })
        for benchmark in benchmark_models:
            if benchmark not in draws[metric_variant]:
                continue
            mean_diff, p, n_valid, prop_tie = difference_pvalue(
                draws[metric_variant][sparc_model], draws[metric_variant][benchmark], paired
            )
            pvalue_rows.append({
                'task': task, 'metric': metric_variant,
                'sparc_model': sparc_model, 'benchmark': benchmark,
                'mean_diff_pct': mean_diff * 100 if mean_diff == mean_diff else None,
                'p_value': p, 'n_valid_boot': n_valid, 'prop_tie': prop_tie,
                'test_type': test_type,
                'significant_at_alpha': (p < alpha) if p == p else None,
                'alpha': alpha, 'n_bootstrap': n_bootstrap, 'rseed': rseed,
            })

    # cache files
    draws_path = draws_cache_path(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed)
    with open(draws_path, 'wb') as f:
        pickle.dump({
            'task': task, 'option': option, 'dataset': dataset, 'modules': modules,
            'sparc_model': sparc_model, 'benchmark_models': list(benchmark_models),
            'model_names': model_names, 'metric_variants': METRIC_VARIANTS,
            'n_bootstrap': n_bootstrap, 'rseed': rseed, 'seed_entropy': seed_entropy,
            'test_type': test_type, 'paired': paired,
            'draws': draws,  # {metric_variant: {model: np.array(n_bootstrap)}}
        }, f)

    summary_path = summary_cache_path(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed)
    # barstat and pvalue rows in one CSV, tagged by row_type
    bar_df = pd.DataFrame(barstat_rows); bar_df.insert(0, 'row_type', 'barstat')
    pv_df = pd.DataFrame(pvalue_rows);  pv_df.insert(0, 'row_type', 'pvalue')
    summary_df = pd.concat([bar_df, pv_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print()
        print(f"BOOTSTRAP STATS for task={task}  (n_bootstrap={n_bootstrap}, seed={rseed})")
        print()
        for metric_variant in METRIC_VARIANTS:
            print(f"\n[{metric_variant}]")
            for m in model_names:
                r = next(x for x in barstat_rows if x['model'] == m and x['metric'] == metric_variant)
                pt = r['point_pct']; lo = r['ci_lower_pct']; hi = r['ci_upper_pct']; fm = r['fold_mean_pct']
                fold_str = r['per_fold_pct'].replace(';', ', ') if r['per_fold_pct'] else 'n/a'
                pt_s = f"{pt:.2f}%" if pt is not None else "n/a"
                ci_s = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[n/a]"
                fm_s = f"{fm:.2f}%" if fm is not None else "n/a"
                print(f"  {m:16s} point={pt_s:>8s}  CI={ci_s:>18s}  fold-mean={fm_s:>8s}")
                if fold_str != 'n/a':
                    print(f"                   per-fold: [{fold_str}]")
            for benchmark in benchmark_models:
                pr = [x for x in pvalue_rows if x['benchmark'] == benchmark and x['metric'] == metric_variant]
                if not pr:
                    continue
                pr = pr[0]
                p = pr['p_value']; md = pr['mean_diff_pct']
                if p is None:
                    star = 'n/a'
                elif p < alpha:
                    star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
                else:
                    star = 'NS'
                md_s = f"{md:+.2f}%" if md is not None else "n/a"
                p_s = f"{p:.4f}" if p is not None else "n/a"
                print(f"    {sparc_model} vs {benchmark:16s}: mean_diff={md_s:>8s}  p={p_s}  "
                      f"-> {star}  [{test_type}]")
        print("\nWrote draws cache   ->", draws_path)
        print("Wrote summary cache ->", summary_path)
        print()

    return {'barstat_rows': barstat_rows, 'pvalue_rows': pvalue_rows,
            'draws_path': draws_path, 'summary_path': summary_path}


# cache loading (vis)

def load_summary(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed):
    """Summary CSV as a DataFrame, or None if missing."""
    import pandas as pd
    path = summary_cache_path(cache_dir, task, dataset, finetuned_suffix, modules, n_bootstrap, rseed)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

