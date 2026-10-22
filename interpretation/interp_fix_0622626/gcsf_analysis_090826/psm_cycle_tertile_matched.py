#!/usr/bin/env python3
"""G-CSF effect by SPARC risk tertile, PSM exact on tertile and infusion cycle, from the saved per-infusion table.

Each G-CSF infusion gets its k nearest untreated infusions on logit PS (with replacement) in the same tertile and
cycle; controls beyond caliper_sd x pooled SD(logit PS) are dropped. CIs and p from a bootstrap over G-CSF patients, matches held fixed.
Called from ..._101426_cyclematch.py; standalone, it reads a saved per-infusion table with a propensity_score column.
Usage: python psm_cycle_tertile_matched.py --table <csv> [--k 1 10] [--caliper_sd 0.2] [--n_boot 2000]
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.neighbors import NearestNeighbors

plt.rcParams['pdf.fonttype'] = 42   # editable text in Illustrator

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cycle_tertile_psm(df, output_dir, dataset='MSK_CAIA', ks=(1, 10), caliper_sd=0.2, n_boot=2000, seed=1024):
    """df: one row per infusion with DMP_ID, cutoffdate, gcsf, outcome, propensity_score, sparc_risk_score
    (and infusion_cycle if available). Writes psm_cycle_tertile_1to{k}_{dataset}_3strata.{pdf,png,csv}; returns {k: table}."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    NUM_STRATA = 3
    d = df.sort_values(['DMP_ID', 'cutoffdate']).reset_index(drop=True)
    # cycle = test_dataset.num_infusions; saved tables lack it, so fall back to row order by cutoffdate
    d['cycle'] = d['infusion_cycle'] if 'infusion_cycle' in d else d.groupby('DMP_ID').cumcount()
    d['g'] = d['gcsf'].astype(str).str.lower().isin(['true', '1'])
    ps = d['propensity_score'].clip(1e-9, 1 - 1e-9)
    d['lps'] = np.log(ps / (1 - ps))
    sd = np.sqrt((d.loc[d.g, 'lps'].var() + d.loc[~d.g, 'lps'].var()) / 2)
    cal = caliper_sd * sd if caliper_sd > 0 else np.inf
    cuts = np.percentile(d['sparc_risk_score'], [100 * (i + 1) / NUM_STRATA for i in range(NUM_STRATA - 1)])
    d['s'] = np.digitize(d['sparc_risk_score'], cuts)
    labels = [f'Stratum 1 (<{cuts[0]*100:.1f}%)'] + \
             [f'Stratum {i+1} ({cuts[i-1]*100:.1f}-{cuts[i]*100:.1f}%)' for i in range(1, NUM_STRATA - 1)] + \
             [f'Stratum {NUM_STRATA} (≥{cuts[-1]*100:.1f}%)']
    print(f'{len(d)} infusions, {d.DMP_ID.nunique()} patients; caliper {caliper_sd:g} x {sd:.4f} = {cal:.4f} (logit PS)')


    def match(k):
        """per stratum: arrays of (G-CSF outcome, mean control outcome, patient id) and the distinct control rows"""
        out = {}
        for s in range(NUM_STRATA):
            yt, yc, pid, used, smd_t, smd_c, w_c = [], [], [], set(), [], [], []
            for _, a in d[d.s == s].groupby('cycle'):
                t, c = a[a.g], a[~a.g]
                if len(t) == 0 or len(c) == 0:
                    continue
                dist, ix = NearestNeighbors(n_neighbors=min(k, len(c))).fit(c[['lps']].values).kneighbors(t[['lps']].values)
                for i in range(len(t)):
                    ok = dist[i] <= cal
                    if not ok.any():
                        continue
                    cc = c.iloc[ix[i][ok]]
                    yt.append(t['outcome'].values[i]); yc.append(cc['outcome'].mean()); pid.append(t['DMP_ID'].values[i])
                    used.update(cc.index)
                    smd_t.append(t['lps'].values[i]); smd_c.extend(cc['lps']); w_c.extend([1 / len(cc)] * len(cc))
            out[s] = dict(yt=np.array(yt, float), yc=np.array(yc, float), pid=np.array(pid), used=used,
                          n_gcsf_all=int((d.g & (d.s == s)).sum()),
                          smd=(np.mean(smd_t) - np.average(smd_c, weights=w_c)) / sd)
        return out


    for k in ks:
        m = match(k)
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(NUM_STRATA):
            r = m[s]
            diff = r['yc'] - r['yt']
            arr = diff.mean()
            groups = list(pd.Series(np.arange(len(diff))).groupby(r['pid']).indices.values())
            bs = np.array([diff[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean()
                           for _ in range(n_boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
            rows.append({'stratum': labels[s], 'ratio': f'1:{k}', 'n_gcsf': len(diff), 'n_gcsf_before_matching': r['n_gcsf_all'],
                         'n_gcsf_patients': len(groups), 'n_distinct_controls': len(r['used']),
                         'gcsf_outcome_rate': r['yt'].mean(), 'no_gcsf_outcome_rate': r['yc'].mean(),
                         'absolute_risk_reduction': arr, 'arr_ci_lower': lo, 'arr_ci_upper': hi,
                         'nnt': 1 / arr if arr > 0 else np.inf,
                         'nnt_ci_lower': 1 / hi if hi > 0 else np.nan, 'nnt_ci_upper': 1 / lo if lo > 0 else np.inf,
                         'p_value': p, 'smd_logit_ps': r['smd'], 'caliper_logit': cal})
        res = pd.DataFrame(rows)
        stem = os.path.join(output_dir, f'psm_cycle_tertile_1to{k}_{dataset}_3strata')
        res.to_csv(stem + '.csv', index=False)
        print(f'\n1:{k}')
        print(res[['stratum', 'n_gcsf', 'n_distinct_controls', 'gcsf_outcome_rate', 'no_gcsf_outcome_rate',
                   'absolute_risk_reduction', 'arr_ci_lower', 'arr_ci_upper', 'nnt', 'p_value', 'smd_logit_ps']]
              .round(4).to_string(index=False))

        # same layout as the IPTW figure
        y_pos = np.arange(NUM_STRATA, dtype=float)
        BAR_H, ROW_GAP = 0.32, 0.40
        fig, ax = plt.subplots(figsize=(9, max(5.0, 1.15 * NUM_STRATA + 2.4)))
        ax.barh(y_pos - ROW_GAP / 2, res['gcsf_outcome_rate'], height=BAR_H, color='#3498db', edgecolor='black',
                linewidth=0.9, label='G-CSF', zorder=2)
        ax.barh(y_pos + ROW_GAP / 2, res['no_gcsf_outcome_rate'], height=BAR_H, color='#e74c3c', edgecolor='black',
                linewidth=0.9, label=f'No G-CSF (matched, 1:{k})', zorder=2)
        for i, r in res.iterrows():
            ax.text(r['gcsf_outcome_rate'] + 0.004, y_pos[i] - ROW_GAP / 2, f'n={r["n_gcsf"]:,}', ha='left', va='center',
                    fontsize=7.5, color='#333333')
            ax.text(r['no_gcsf_outcome_rate'] + 0.004, y_pos[i] + ROW_GAP / 2, f'n={r["n_distinct_controls"]:,} distinct',
                    ha='left', va='center', fontsize=7.5, color='#333333')
        rate_max = np.nanmax([res['gcsf_outcome_rate'].max(), res['no_gcsf_outcome_rate'].max()])
        x_hi = min(max(rate_max * 1.55, rate_max + 0.12), 1.0)
        ax.set_xlim(0, x_hi)
        _nnt = lambda x: f'{1 / x:.0f}' if x > 0 else '∞'
        for i, r in res.iterrows():
            pv = r['p_value']
            p_text = f'p<{2/n_boot:.3g}' if pv < 2 / n_boot else (f'p={pv:.3f}' if pv < 0.1 else f'p={pv:.2f}')
            nnt_text = (f'NNT {_nnt(r["absolute_risk_reduction"])} [{_nnt(r["arr_ci_upper"])}, {_nnt(r["arr_ci_lower"])}]'
                        if r['absolute_risk_reduction'] > 0 else 'NNT n/a (no benefit)')
            ax.text(x_hi * 0.99, y_pos[i], f'{nnt_text}, {p_text}', ha='right', va='center', fontsize=8, fontweight='bold')
        ax.set_xlabel('Neutropenia rate, matched (%)', fontsize=12)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f'Neutropenia rate by SPARC risk stratum (PSM 1:{k}, exact on tertile and infusion cycle, '
                     f'caliper {caliper_sd:g} SD)', fontsize=11, pad=12, loc='left')
        ax.set_yticks(y_pos); ax.set_yticklabels([s.replace(' (', '\n(') for s in res['stratum']], fontsize=11)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(stem + '.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(stem + '.png', bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved {stem}.{{pdf,png,csv}}')
        results[k] = res
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True, help='per-infusion csv (DMP_ID, cutoffdate, gcsf, outcome, '
                        'propensity_score, sparc_risk_score)')
    parser.add_argument('--dataset', default='MSK_CAIA')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 10], help='controls per G-CSF infusion')
    parser.add_argument('--caliper_sd', type=float, default=0.2)
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--output_dir', default=os.path.join(HERE, 'out_gcsf_bfTrue_psm_cycle_tertile'))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_cycle_tertile_psm(pd.read_csv(args.table, index_col=0), args.output_dir, args.dataset, args.k, args.caliper_sd,
                          args.n_boot, args.seed)
