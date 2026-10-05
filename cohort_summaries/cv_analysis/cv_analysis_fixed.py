#!/usr/bin/env python3
"""Inter-patient and long/short-term intra-patient coefficient of variation per feature,
for any SPARC dataset (practical or parsimonious module), loaded via dataset_loading.py.

Usage: python cv_analysis_fixed.py -ds MSK_CAIA -cp entire -m parsimonious [-sti 30] [-od DIR]
"""

import os
import sys
import argparse
import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from joblib import Parallel, delayed

# Path setup: resolve SPARC/data so dataset_loading.py is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # …/cohort_summaries
# walk up to the repo root (the directory holding data/ and models/)
_SPARC_ROOT = _SCRIPT_DIR
while _SPARC_ROOT != os.path.dirname(_SPARC_ROOT) and not all(
        os.path.isdir(os.path.join(_SPARC_ROOT, _d)) for _d in ('data', 'models')):
    _SPARC_ROOT = os.path.dirname(_SPARC_ROOT)
_DATA_DIR   = os.path.join(_SPARC_ROOT, 'data')                   # …/SPARC/data

for _p in [_DATA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
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

from dataset_loading import load_dataset
from MSK_featurename_list import (
    BASIC_VITALS, CBC_DIFFERENTIAL_TESTS, RBC_CBC_TESTS, BMP_TESTS, LFT_TESTS
)

# Argument parsing
parser = argparse.ArgumentParser(
    prog='cv_analysis_fixed',
    description='CV analysis (inter-patient + intra-patient) across SPARC datasets',
    epilog='Outputs CSV results and PDF plots to --output_dir.',
)
parser.add_argument(
    '-ds', '--dataset', type=str, default='MSK',
    help=(
        'Dataset name.  '
        'Options: MSK | MSK_CAIA | MSK_timeshift | MSK_CAIA_timeshift | '
        'stanfordEHRSHOT (or StanfordEHRSHOT) | MIMICIV_hosp'
    ),
)
parser.add_argument(
    '-sti', '--short_term_interval', type=int, default=60,
    help='Short-term sliding window size in days (default: 60).',
)
parser.add_argument(
    '-n', '--n_jobs', type=int, default=-2,
    help='Number of parallel workers.  -1=all cores, -2=all-1, positive=exact. (default: -2)',
)
parser.add_argument(
    '-vp', '--volume_path', type=str, default=sparc_paths.DATA_ROOT + '/',
    help='Base directory that contains all dataset sub-folders.',
)
parser.add_argument(
    '-od', '--output_dir', type=str, default='',
    help=(
        'Directory for results CSV and plots.  '
        'Defaults to $SPARC_COHORT_SUMMARIES_DIR (see sparc_paths.py)/cv_analyses/<dataset>.'
    ),
)
parser.add_argument(
    '-cp', '--caia_portion', type=str, default='entire',
    help='For CAIA datasets: prior | post | entire  (default: entire).',
)
parser.add_argument(
    '-is', '--impact_subset', action='store_true', default=False,
    help=(
        'When set, restrict to patients present in '
        '/data/morrisq/fuc/SPARC_data/MSK_CAIA/impact_deids.xlsx.  '
        'Only applies to MSK_CAIA and MSK_CAIA_timeshift.'
    ),
)
parser.add_argument(
    '-m', '--modules', type=str, default='practical',
    help='Module category: practical | parsimonious (default: practical).',
)
args = parser.parse_args()

# Resolve parameters
# Accept StanfordEHRSHOT as well as stanfordEHRSHOT
dataset = args.dataset
if dataset == 'StanfordEHRSHOT':
    dataset = 'stanfordEHRSHOT'

modules             = args.modules
short_term_interval = args.short_term_interval
data_source_path    = args.volume_path
caia_portion        = args.caia_portion
impact_subset       = args.impact_subset

# Define feature list based on modules
if modules == 'parsimonious':
    # clinical panel order (CBC -> differential -> BMP -> LFT -> vitals), as in
    # feature_pairwise_correlation_fixed.py
    FEATURE_LIST = (RBC_CBC_TESTS + CBC_DIFFERENTIAL_TESTS
                    + BMP_TESTS + LFT_TESTS + BASIC_VITALS)
else:
    FEATURE_LIST = RBC_CBC_TESTS + BMP_TESTS + LFT_TESTS + BASIC_VITALS

# Number of cores
if args.n_jobs == -1:
    n_cores = multiprocessing.cpu_count()
elif args.n_jobs == -2:
    n_cores = max(1, multiprocessing.cpu_count() - 1)
elif args.n_jobs < 0:
    n_cores = max(1, multiprocessing.cpu_count() + args.n_jobs + 1)
else:
    n_cores = min(args.n_jobs, multiprocessing.cpu_count())

# Output directory
if args.output_dir:
    out_dir = args.output_dir
else:
    out_dir = os.path.join(
        sparc_paths.RESULTS_ROOT + '/results_vis/cohort_summaries/cv_analyses',
        f'{modules}_{dataset}',
    )

# if impact_subset is True, append '_impact_subset' to the output directory name
if impact_subset and 'CAIA' in dataset:
    out_dir += '_impact_subset'
os.makedirs(out_dir, exist_ok=True)

print(f"Dataset            : {dataset}")
print(f"Modules            : {modules}")
print(f"Short-term window  : {short_term_interval} days")
print(f"CPU cores          : {n_cores}")
print(f"Output directory   : {out_dir}")

# Check for cached results
results_file = os.path.join(out_dir, f'cv_analysis_results_{modules}_{dataset}_sti{short_term_interval}.csv')

if os.path.exists(results_file):
    print(f"\nCached results found at {results_file}; loading and skipping to plots.")
    results_df = pd.read_csv(results_file)

else:
    # Load data via dataset_loading.py
    print(f"\nLoading dataset '{dataset}' (normalize=False, modules={modules}, option=binary)…")

    # Use option='binary'; it loads labs + vitals + demographics without meds.
    # caia_portion='entire' keeps all data for CAIA datasets (no temporal cutoff).
    data_dict_lab_dict, _, split_dict, _, _ = load_dataset(
        dataset=dataset,
        data_source_path=data_source_path,
        option='binary',
        modules=modules,
        valfold=0,
        mode='train',
        normalize=False,
        caia_portion=caia_portion if 'CAIA' in dataset else 'prior',  # 'prior' is ignored for non-CAIA
    )

    # Union of train + val + test = ALL patients in this dataset cohort.
    # For MSK_timeshift, the split dict already excludes MSK-only patients.
    all_patient_ids = set(
        list(split_dict.get('train', []))
        + list(split_dict.get('val', []))
        + list(split_dict.get('test', []))
    )
    print(f"Total patients after merging splits: {len(all_patient_ids):,}")

    # Optional: restrict to IMPACT-sequenced patients (CAIA datasets only)
    if impact_subset and 'CAIA' in dataset:
        _impact_path = sparc_paths.DATA_ROOT + '/MSK_CAIA/impact_deids.xlsx'
        _impact_ids  = set(pd.read_excel(_impact_path)['id'].astype(int).tolist())
        _before = len(all_patient_ids)
        all_patient_ids = all_patient_ids & _impact_ids
        print(f"Impact subset filter applied: {_before:,} → {len(all_patient_ids):,} patients")

    basic_lab_dict       = data_dict_lab_dict.get('basic_lab',       {})
    basic_vitals_dict    = data_dict_lab_dict.get('basic_vitals',    {})
    cbc_differential_dict = data_dict_lab_dict.get('cbc_differential', {})  # Only present for parsimonious module

    # Build long-format DataFrame
    print("Building long-format DataFrame…")

    _INVALID_STRS = frozenset({None, 'NA', 'nan', 'NAN', 'TNP', 'Negative', ''})

    def _try_float(v):
        """Return float(v) if valid, else None."""
        if v in _INVALID_STRS:
            return None
        try:
            fv = float(v)
            return None if (np.isnan(fv) or np.isinf(fv)) else fv
        except (ValueError, TypeError):
            return None

    data_rows = []

    # Labs
    for pid in tqdm(all_patient_ids, desc="  Labs"):
        if pid not in basic_lab_dict:
            continue
        for date, vals in basic_lab_dict[pid].items():
            for feat_idx, feat_name in enumerate(BASIC_TESTS):
                if feat_idx >= len(vals):
                    continue
                fv = _try_float(vals[feat_idx])
                if fv is not None:
                    data_rows.append({'pid': pid, 'date': date, 'feature': feat_name, 'value': fv})

    n_lab_rows = len(data_rows)
    print(f"  Lab rows collected: {n_lab_rows:,}")

    # Vitals: MIMICIV_hosp has empty vitals dicts; the loop just skips them naturally
    for pid in tqdm(all_patient_ids, desc="  Vitals"):
        if pid not in basic_vitals_dict:
            continue
        for date, vals in basic_vitals_dict[pid].items():
            for feat_idx, feat_name in enumerate(BASIC_VITALS):
                if feat_idx >= len(vals):
                    continue
                fv = _try_float(vals[feat_idx])
                if fv is not None:
                    data_rows.append({'pid': pid, 'date': date, 'feature': feat_name, 'value': fv})

    n_vital_rows = len(data_rows)
    print(f"  Vital rows collected: {n_vital_rows - n_lab_rows:,}")

    # CBC Differential: Only present for parsimonious module
    if cbc_differential_dict:
        for pid in tqdm(all_patient_ids, desc="  CBC Differential"):
            if pid not in cbc_differential_dict:
                continue
            for date, vals in cbc_differential_dict[pid].items():
                for feat_idx, feat_name in enumerate(CBC_DIFFERENTIAL_TESTS):
                    if feat_idx >= len(vals):
                        continue
                    fv = _try_float(vals[feat_idx])
                    if fv is not None:
                        data_rows.append({'pid': pid, 'date': date, 'feature': feat_name, 'value': fv})
        print(f"  CBC differential rows collected: {len(data_rows) - n_vital_rows:,}")

    df_long = pd.DataFrame(data_rows)
    del data_rows

    if df_long.empty:
        raise RuntimeError(
            "No data loaded — check that --dataset and --volume_path are correct."
        )

    n_obs      = len(df_long)
    n_patients = df_long['pid'].nunique()
    print(f"DataFrame: {n_obs:,} observations from {n_patients:,} patients")

    df_long = df_long.sort_values(['pid', 'feature', 'date']).reset_index(drop=True)

    # CV helper
    def calc_cv(values):
        """Coefficient of variation (%); returns NaN if <3 values or mean==0."""
        if len(values) < 3:
            return np.nan
        mean_val = np.mean(values)
        if mean_val == 0:
            return np.nan
        return (np.std(values, ddof=1) / abs(mean_val)) * 100.0

    # 1. Inter-patient CV
    print("Computing inter-patient CV…")
    inter_patient_cv = df_long.groupby('feature')['value'].apply(calc_cv).to_dict()

    # 2. Long-term intra-patient CV
    print("Computing long-term intra-patient CV…")
    lt_per_patient = df_long.groupby(['pid', 'feature'])['value'].apply(calc_cv)
    lt_cv_median   = lt_per_patient.groupby(level='feature').median().to_dict()
    lt_cv_std      = lt_per_patient.groupby(level='feature').std().to_dict()
    lt_cv_count    = lt_per_patient.groupby(level='feature').count().to_dict()

    # 2a. Bootstrap CI for long-term intra-patient CV
    def _bootstrap_ci(per_patient_cvs, n_bootstrap=10_000, ci=0.95,
                      chunk_size=100, rng=None):
        """Bootstrap CI (lo, hi) on the median of per_patient_cvs, computed in chunks."""
        rng = rng or np.random.default_rng(42)
        per_patient_cvs = np.array(per_patient_cvs)
        per_patient_cvs = per_patient_cvs[~np.isnan(per_patient_cvs)]

        if len(per_patient_cvs) == 0:
            return np.nan, np.nan

        n = len(per_patient_cvs)
        n_chunks = (n_bootstrap + chunk_size - 1) // chunk_size
        medians = []

        for _ in range(n_chunks):
            current_chunk = min(chunk_size, n_bootstrap - len(medians))
            # Vectorized: sample all at once for this chunk
            samples = rng.choice(per_patient_cvs, size=(current_chunk, n), replace=True)
            chunk_medians = np.median(samples, axis=1)
            medians.extend(chunk_medians)

        medians = np.array(medians[:n_bootstrap])
        lo = np.percentile(medians, (1 - ci) / 2 * 100)
        hi = np.percentile(medians, (1 + ci) / 2 * 100)
        return lo, hi

    print("Computing bootstrap CIs for long-term intra-patient CV…")
    lt_cv_bootstrap_lo = {}
    lt_cv_bootstrap_hi = {}
    for feat in tqdm(FEATURE_LIST, desc="  Bootstrap CI"):
        if feat in lt_per_patient.index.get_level_values('feature'):
            per_patient_vals = lt_per_patient.xs(feat, level='feature').values
            lo, hi = _bootstrap_ci(per_patient_vals)
            lt_cv_bootstrap_lo[feat] = lo
            lt_cv_bootstrap_hi[feat] = hi
        else:
            lt_cv_bootstrap_lo[feat] = np.nan
            lt_cv_bootstrap_hi[feat] = np.nan

    # 3. Short-term intra-patient CV, parallelised over features (one job per feature)
    print(
        f"Computing short-term intra-patient CV "
        f"(window={short_term_interval} days, {n_cores} cores, "
        f"parallelised over {len(FEATURE_LIST)} features)…"
    )

    # Pre-split df_long by feature once so workers receive compact arrays,
    # not a reference to the full DataFrame.
    feature_arrays = {}
    for feat, grp in df_long.groupby('feature'):
        # Sort by (pid, date); already sorted, but make sure
        feature_arrays[feat] = (
            grp['pid'].values,
            grp['date'].values,
            grp['value'].values,
        )

    def _windowed_cv_for_feature(feature, pid_arr, date_arr, val_arr, window_days):
        """Windowed CV for one feature. Returns (feature, {pid: median_cv}, {pid: n_valid_windows})."""
        step = max(1, window_days // 2)
        cv_dict  = {}
        win_dict = {}

        # Find contiguous blocks belonging to each patient
        # (array is already sorted by pid then date from df_long sort above)
        unique_pids, starts = np.unique(pid_arr, return_index=True)
        ends = np.append(starts[1:], len(pid_arr))

        for pid, lo_pid, hi_pid in zip(unique_pids, starts, ends):
            dates  = date_arr[lo_pid:hi_pid]
            values = val_arr[lo_pid:hi_pid]

            if len(dates) < 3:
                cv_dict[pid]  = np.nan
                win_dict[pid] = 0
                continue

            end = dates[-1]
            cur = dates[0]
            cvs = []

            while cur <= end:
                lo = np.searchsorted(dates, cur,              side='left')
                hi = np.searchsorted(dates, cur + window_days, side='left')
                w_vals = values[lo:hi]
                if len(w_vals) >= 3:
                    mean_v = np.mean(w_vals)
                    if mean_v != 0:
                        cv = (np.std(w_vals, ddof=1) / abs(mean_v)) * 100.0
                        if np.isfinite(cv):
                            cvs.append(cv)
                cur += step
                if cur > end:
                    break

            n_win = len(cvs)
            cv_dict[pid]  = float(np.median(cvs)) if n_win > 0 else np.nan
            win_dict[pid] = n_win

        return feature, cv_dict, win_dict

    feat_results = Parallel(n_jobs=n_cores, verbose=0)(
        delayed(_windowed_cv_for_feature)(
            feat,
            feature_arrays[feat][0],
            feature_arrays[feat][1],
            feature_arrays[feat][2],
            short_term_interval,
        )
        for feat in tqdm(FEATURE_LIST, desc="  Windowed CV")
        if feat in feature_arrays
    )
    del feature_arrays  # free memory

    # rebuild the (pid, feature)-indexed Series
    st_cv_records  = []
    st_win_records = []
    for feat, cv_dict, win_dict in feat_results:
        for pid, cv in cv_dict.items():
            st_cv_records.append( {'pid': pid, 'feature': feat, 'val': cv})
        for pid, nw in win_dict.items():
            st_win_records.append({'pid': pid, 'feature': feat, 'val': nw})

    st_cv_df  = pd.DataFrame(st_cv_records)
    st_win_df = pd.DataFrame(st_win_records)

    st_cv_median  = st_cv_df.groupby('feature')['val'].median().to_dict()
    st_cv_std     = st_cv_df.groupby('feature')['val'].std().to_dict()
    st_cv_count   = st_cv_df.groupby('feature')['val'].count().to_dict()
    st_win_mean   = st_win_df.groupby('feature')['val'].mean().to_dict()
    st_win_median = st_win_df.groupby('feature')['val'].median().to_dict()
    st_win_std    = st_win_df.groupby('feature')['val'].std().to_dict()

    # Compile results table
    print("Compiling results…")
    results = []
    for feat in FEATURE_LIST:
        inter_cv = inter_patient_cv.get(feat, np.nan)
        lt_cv    = lt_cv_median.get(feat, np.nan)
        st_cv    = st_cv_median.get(feat, np.nan)

        lt_ratio = lt_cv / inter_cv if (not np.isnan(inter_cv) and inter_cv != 0) else np.nan
        st_ratio = st_cv / inter_cv if (not np.isnan(inter_cv) and inter_cv != 0) else np.nan

        n_lt = lt_cv_count.get(feat, 0)
        n_st = st_cv_count.get(feat, 0)
        lt_se = (lt_cv_std.get(feat, 0) / np.sqrt(n_lt)) if n_lt > 0 else 0.0
        st_se = (st_cv_std.get(feat, 0) / np.sqrt(n_st)) if n_st > 0 else 0.0

        results.append({
            'feature':                    feat,
            'inter_patient_cv':           inter_cv,
            'long_term_intra_patient_cv': lt_cv,
            'short_term_intra_patient_cv': st_cv,
            'long_term_cv_ratio':         lt_ratio,
            'short_term_cv_ratio':        st_ratio,
            'long_term_se':               lt_se,
            'short_term_se':              st_se,
            'long_term_bootstrap_ci_lo':  lt_cv_bootstrap_lo.get(feat, np.nan),
            'long_term_bootstrap_ci_hi':  lt_cv_bootstrap_hi.get(feat, np.nan),
            'n_patients_long':            n_lt,
            'n_patients_short':           n_st,
            'avg_windows_per_patient':    st_win_mean.get(feat, np.nan),
            'median_windows_per_patient': st_win_median.get(feat, np.nan),
            'std_windows_per_patient':    st_win_std.get(feat, np.nan),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    print()
    print("CV Analysis Results:")
    print()
    print(results_df.to_string())
    print()


# canonical feature order for plots
# Reload FEATURE_LIST based on modules (in case loading from cached results)
if modules == 'parsimonious':
    # clinical panel order (CBC -> differential -> BMP -> LFT -> vitals), as in
    # feature_pairwise_correlation_fixed.py
    FEATURE_LIST = (RBC_CBC_TESTS + CBC_DIFFERENTIAL_TESTS
                    + BMP_TESTS + LFT_TESTS + BASIC_VITALS)
else:
    FEATURE_LIST = RBC_CBC_TESTS + BMP_TESTS + LFT_TESTS + BASIC_VITALS

results_df['feature'] = pd.Categorical(
    results_df['feature'], categories=FEATURE_LIST, ordered=True
)
results_df = results_df.sort_values('feature').reset_index(drop=True)

print("\nCreating visualizations…")

x = np.arange(len(results_df))
w = 0.25

# Plot 1 : Bar plot (linear y-axis)
fig, ax = plt.subplots(figsize=(14, 10))
ax.bar(x - w, results_df['inter_patient_cv'],            w,
       label='Inter-patient',                               color='#2E3B55', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.bar(x,     results_df['long_term_intra_patient_cv'],   w,
       label='Long-term intra-patient',                     color='#5B9AA9', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.bar(x + w, results_df['short_term_intra_patient_cv'],  w,
       label=f'Short-term intra-patient ({short_term_interval}d)', color='#8FB996', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Biomarker', fontsize=13, fontweight='bold')
ax.set_ylabel('CV (%)', fontsize=13, fontweight='bold')
ax.set_title(f'Coefficient of Variation Comparison ({modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df['feature'], rotation=45, ha='right', fontsize=11)
ax.legend(frameon=True, fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim(0, None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
p1 = os.path.join(out_dir, f'cv_comparison_barplot_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p1, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p1}")

# Plot 2 : Bar plot (log y-axis)
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w, results_df['inter_patient_cv'],            w,
       label='Inter-patient',                               color='#2E3B55', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.bar(x,     results_df['long_term_intra_patient_cv'],   w,
       label='Long-term intra-patient',                     color='#5B9AA9', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.bar(x + w, results_df['short_term_intra_patient_cv'],  w,
       label=f'Short-term intra-patient ({short_term_interval}d)', color='#8FB996', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}' if y >= 1 else f'{y:.1f}'))
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.set_xlabel('Biomarker', fontsize=13, fontweight='bold')
ax.set_ylabel('CV (%) — log scale', fontsize=13, fontweight='bold')
ax.set_title(f'Coefficient of Variation Comparison (Log Scale, {modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df['feature'], rotation=45, ha='right', fontsize=11)
ax.legend(frameon=True, fontsize=11, loc='upper left')
ax.grid(axis='y', which='major', alpha=0.5, linestyle='-',  linewidth=0.7)
ax.grid(axis='y', which='minor', alpha=0.2, linestyle='--', linewidth=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
p2 = os.path.join(out_dir, f'cv_comparison_barplot_logscale_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p2, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p2}")

# Plot 3 : CV ratio dot-plot (long-term intra / inter)
fig, ax = plt.subplots(figsize=(9, 7))
# Reverse so the first feature appears at the top
rev_df  = results_df.iloc[::-1].reset_index(drop=True)
y_pos   = np.arange(len(rev_df))
lt_err  = (rev_df['long_term_se'] / rev_df['inter_patient_cv'].replace(0, np.nan)).fillna(0)
ax.errorbar(
    rev_df['long_term_cv_ratio'], y_pos,
    xerr=lt_err, fmt='o', color='#5B9AA9',
    markersize=6, capsize=5, capthick=2, linewidth=2,
    markeredgecolor='black', markeredgewidth=0.5, elinewidth=1.5,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(rev_df['feature'], fontsize=11)
ax.set_xlabel('CV ratio\n(long-term intra-patient / inter-patient)',
              fontsize=13, fontweight='bold')
ax.set_title(f'Intra-patient to Inter-patient CV Ratio ({modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=0.5, color='#E53935', linestyle='--', linewidth=2, alpha=0.7,
           label='CV ratio = 0.5', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, fontsize=11, loc='best')
plt.tight_layout()
p3 = os.path.join(out_dir, f'cv_ratio_plot_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p3, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p3}")

# Plot 4 : Average valid windows per patient
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, results_df['avg_windows_per_patient'],
       color='#7B68A6', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.errorbar(
    x, results_df['avg_windows_per_patient'],
    yerr=results_df['std_windows_per_patient'],
    fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.5,
)
ax.set_xlabel('Biomarker', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Valid Windows per Patient', fontsize=13, fontweight='bold')
ax.set_title(
    f'Data Availability: Average Valid Windows per Patient ({modules} module)\n'
    f'(Window: {short_term_interval} days, ≥3 measurements required)',
    fontsize=14, fontweight='bold', pad=20,
)
ax.set_xticks(x)
ax.set_xticklabels(results_df['feature'], rotation=45, ha='right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim(0, None)
median_w = results_df['avg_windows_per_patient'].median()
ax.axhline(y=median_w, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
           label=f'Median: {median_w:.1f} windows')
ax.legend(frameon=True, fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
p4 = os.path.join(out_dir, f'avg_windows_per_patient_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p4, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p4}")

# Plot 5 : CV ratio dot-plot with SD (long-term intra / inter)
fig, ax = plt.subplots(figsize=(9, 7))
# Reverse so the first feature appears at the top
rev_df  = results_df.iloc[::-1].reset_index(drop=True)
y_pos   = np.arange(len(rev_df))
# Calculate SD from SEM: SD = SEM * sqrt(n)
lt_sd = rev_df['long_term_se'] * np.sqrt(rev_df['n_patients_long'])
# Convert SD to ratio scale
lt_err_sd = (lt_sd / rev_df['inter_patient_cv'].replace(0, np.nan)).fillna(0)
ax.errorbar(
    rev_df['long_term_cv_ratio'], y_pos,
    xerr=lt_err_sd, fmt='o', color='#5B9AA9',
    markersize=6, capsize=5, capthick=2, linewidth=2,
    markeredgecolor='black', markeredgewidth=0.5, elinewidth=1.5,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(rev_df['feature'], fontsize=11)
ax.set_xlabel('CV ratio\n(long-term intra-patient / inter-patient)',
              fontsize=13, fontweight='bold')
ax.set_title(f'Intra-patient to Inter-patient CV Ratio with SD ({modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=0.5, color='#E53935', linestyle='--', linewidth=2, alpha=0.7,
           label='CV ratio = 0.5', zorder=0)
ax.set_xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, fontsize=11, loc='best')
plt.tight_layout()
p5 = os.path.join(out_dir, f'cv_ratio_plot_wsd_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p5, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p5}")

# Plot 5a : CV ratio dot-plot with Bootstrap CI (long-term intra / inter)
fig, ax = plt.subplots(figsize=(9, 7))
# Reverse so the first feature appears at the top
rev_df  = results_df.iloc[::-1].reset_index(drop=True)
y_pos   = np.arange(len(rev_df))
# Calculate bootstrap CI error bars in ratio scale
lt_err_lo_bootstrap = (rev_df['long_term_cv_ratio'] -
                       (rev_df['long_term_bootstrap_ci_lo'] / rev_df['inter_patient_cv'].replace(0, np.nan))).fillna(0)
lt_err_hi_bootstrap = ((rev_df['long_term_bootstrap_ci_hi'] / rev_df['inter_patient_cv'].replace(0, np.nan)) -
                       rev_df['long_term_cv_ratio']).fillna(0)
ax.errorbar(
    rev_df['long_term_cv_ratio'], y_pos,
    xerr=[lt_err_lo_bootstrap, lt_err_hi_bootstrap],
    fmt='o', color='#5B9AA9',
    markersize=6, capsize=5, capthick=2, linewidth=2,
    markeredgecolor='black', markeredgewidth=0.5, elinewidth=1.5,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(rev_df['feature'], fontsize=11)
ax.set_xlabel('CV ratio\n(long-term intra-patient / inter-patient)',
              fontsize=13, fontweight='bold')
ax.set_title(f'Intra-patient to Inter-patient CV Ratio with Bootstrap CI ({modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=0.5, color='#E53935', linestyle='--', linewidth=2, alpha=0.7,
           label='CV ratio = 0.5', zorder=0)
ax.set_xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, fontsize=11, loc='best')
plt.tight_layout()
p5a = os.path.join(out_dir, f'cv_ratio_plot_wbootstrapci_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p5a, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p5a}")

# Plot 5b : CV ratio dot-plot with Parametric CI (long-term intra / inter)
fig, ax = plt.subplots(figsize=(9, 7))
# Reverse so the first feature appears at the top
rev_df  = results_df.iloc[::-1].reset_index(drop=True)
y_pos   = np.arange(len(rev_df))
# Calculate 95% parametric CI: 1.96 * SE
lt_ci95 = 1.96 * rev_df['long_term_se']
# Convert CI to ratio scale
lt_err_ci95 = (lt_ci95 / rev_df['inter_patient_cv'].replace(0, np.nan)).fillna(0)
ax.errorbar(
    rev_df['long_term_cv_ratio'], y_pos,
    xerr=lt_err_ci95, fmt='o', color='#5B9AA9',
    markersize=6, capsize=5, capthick=2, linewidth=2,
    markeredgecolor='black', markeredgewidth=0.5, elinewidth=1.5,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(rev_df['feature'], fontsize=11)
ax.set_xlabel('CV ratio\n(long-term intra-patient / inter-patient)',
              fontsize=13, fontweight='bold')
ax.set_title(f'Intra-patient to Inter-patient CV Ratio with 95% Parametric CI ({modules} module) — {dataset} Cohort',
             fontsize=14, fontweight='bold', pad=15)
ax.axvline(x=0.5, color='#E53935', linestyle='--', linewidth=2, alpha=0.7,
           label='CV ratio = 0.5', zorder=0)
ax.set_xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, fontsize=11, loc='best')
plt.tight_layout()
p5b = os.path.join(out_dir, f'cv_ratio_plot_wparaci_{modules}_{dataset}_sti{short_term_interval}.pdf')
plt.savefig(p5b, bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: {p5b}")

# Summary statistics
print()
print("SUMMARY STATISTICS")
print()
print(f"Dataset            : {dataset}")
print(f"Modules            : {modules}")
print(f"Short-term interval: {short_term_interval} days")
print(f"\nMedian inter-patient CV            : {results_df['inter_patient_cv'].median():.2f}%")
print(f"Median long-term intra-patient CV  : {results_df['long_term_intra_patient_cv'].median():.2f}%")
print(f"Median short-term intra-patient CV : {results_df['short_term_intra_patient_cv'].median():.2f}%")
print(f"\nMedian CV ratio (long-term  intra/inter): {results_df['long_term_cv_ratio'].median():.3f}")
print(f"Median CV ratio (short-term intra/inter): {results_df['short_term_cv_ratio'].median():.3f}")

print(f"\n{'Feature':<30} {'Inter-CV':>10} {'Long-CV':>10} {'Short-CV':>10} {'LT Ratio':>10} {'Avg Windows':>13}")
print("-" * 95)
for _, row in results_df.iterrows():
    inter = f"{row['inter_patient_cv']:.2f}%" if not np.isnan(row['inter_patient_cv']) else "  N/A  "
    lt    = f"{row['long_term_intra_patient_cv']:.2f}%" if not np.isnan(row['long_term_intra_patient_cv']) else "  N/A  "
    st    = f"{row['short_term_intra_patient_cv']:.2f}%" if not np.isnan(row['short_term_intra_patient_cv']) else "  N/A  "
    ltr   = f"{row['long_term_cv_ratio']:.3f}" if not np.isnan(row['long_term_cv_ratio']) else "  N/A "
    wnd   = f"{row['avg_windows_per_patient']:.1f}" if not np.isnan(row['avg_windows_per_patient']) else " N/A "
    print(f"{row['feature']:<30} {inter:>10} {lt:>10} {st:>10} {ltr:>10} {wnd:>13}")

print()
top5 = results_df.nlargest(5, 'long_term_cv_ratio')[
    ['feature', 'long_term_cv_ratio', 'inter_patient_cv', 'long_term_intra_patient_cv', 'n_patients_long']]
print("Top-5 features by long-term CV ratio (most variable relative to population):")
print(top5.to_string(index=False))

print()
bot5 = results_df.nsmallest(5, 'long_term_cv_ratio')[
    ['feature', 'long_term_cv_ratio', 'inter_patient_cv', 'long_term_intra_patient_cv', 'n_patients_long']]
print("Bottom-5 features by long-term CV ratio (most stable / setpoint-like):")
print(bot5.to_string(index=False))

print()
print("Analysis complete!")
print()
