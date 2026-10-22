#!/usr/bin/env python3
"""Spearman correlation heatmap of first-available lab/vital values, grouped and colored as in the LIME plots.

Datasets: MSK, MSK_CAIA, MSK_timeshift, MSK_CAIA_timeshift, StanfordEHRSHOT, MIMICIV_hosp.
Usage:
    python feature_pairwise_correlation_fixed.py -ds MSK_CAIA -m parsimonious
    python feature_pairwise_correlation_fixed.py -ds MSK_CAIA -cp entire -m parsimonious
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

# path setup: resolve SPARC/data so dataset_loading.py is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # …/cohort_summaries
# walk up to the repo root (holds data/ and models/)
_SPARC_ROOT = _SCRIPT_DIR
while _SPARC_ROOT != os.path.dirname(_SPARC_ROOT) and not all(
        os.path.isdir(os.path.join(_SPARC_ROOT, _d)) for _d in ('data', 'models')):
    _SPARC_ROOT = os.path.dirname(_SPARC_ROOT)
_DATA_DIR   = os.path.join(_SPARC_ROOT, 'data')                   # …/SPARC/data

for _p in [_DATA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
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

from dataset_loading import load_dataset
from MSK_featurename_list import (
    BASIC_TESTS, BASIC_VITALS, CBC_DIFFERENTIAL_TESTS,
    RBC_CBC_TESTS, BMP_TESTS, LFT_TESTS
)

# argument parsing
parser = argparse.ArgumentParser(
    prog='feature_pairwise_correlation_fixed',
    description='Pairwise feature correlation heatmap for parsimonious module',
    epilog='Outputs PDF heatmaps to --output_dir.',
)
parser.add_argument(
    '-ds', '--dataset', type=str, default='MSK_CAIA',
    help=(
        'Dataset name.  '
        'Options: MSK | MSK_CAIA | MSK_timeshift | MSK_CAIA_timeshift | '
        'StanfordEHRSHOT | MIMICIV_hosp'
    ),
)
parser.add_argument(
    '-vp', '--volume_path', type=str, default=sparc_paths.DATA_ROOT + '/',
    help='Base directory that contains all dataset sub-folders.',
)
parser.add_argument(
    '-od', '--output_dir', type=str, default='',
    help=(
        'Directory for output heatmaps.  '
        'Defaults to <SPARC_RESULTS_PATH>/results_vis/'
        'cohort_summaries/feature_correlation/<modules>_<dataset>.'
    ),
)
parser.add_argument(
    '-cp', '--caia_portion', type=str, default='prior',
    help='For CAIA datasets: prior | post | entire  (default: prior).',
)
parser.add_argument(
    '-is', '--impact_subset', action='store_true', default=False,
    help=(
        'When set, restrict to patients present in '
        '<SPARC_data>/MSK_CAIA/impact_deids.xlsx.  '
        'Only applies to MSK_CAIA and MSK_CAIA_timeshift.'
    ),
)
parser.add_argument(
    '-m', '--modules', type=str, default='parsimonious',
    help='Module category: practical | parsimonious (default: parsimonious).',
)
args = parser.parse_args()

# resolve parameters
dataset = args.dataset
modules = args.modules

# dataset_loading expects 'stanfordEHRSHOT' (lower-case 's')
dataset_load = dataset
if dataset_load == 'StanfordEHRSHOT':
    dataset_load = 'stanfordEHRSHOT'

data_source_path = args.volume_path
caia_portion     = args.caia_portion
impact_subset    = args.impact_subset

# MSK/MSK_timeshift store age-at-sequencing; all other datasets store age-at-first-measurement
msk_age_anchor = dataset_load in ('MSK', 'MSK_timeshift')

if args.output_dir:
    out_dir = args.output_dir
else:
    out_dir = os.path.join(
        sparc_paths.RESULTS_ROOT + '/results_vis/cohort_summaries/feature_correlation',
        f'{modules}_{dataset}',
    )
os.makedirs(out_dir, exist_ok=True)

print(f"Dataset            : {dataset}")
print(f"Modules            : {modules}")
print(f"Output directory   : {out_dir}")

# load data via dataset_loading.py
print(f"\nLoading dataset '{dataset_load}' (normalize=False, modules={modules}, option=binary)…")

data_dict_lab_dict, _, split_dict, _, _ = load_dataset(
    dataset=dataset_load,
    data_source_path=data_source_path,
    option='binary',
    modules=modules,
    valfold=0,
    mode='train',
    normalize=False,
    caia_portion=caia_portion if 'CAIA' in dataset_load else 'prior',
)

# Union of train + val + test = ALL patients in this dataset cohort.
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
age_dict             = data_dict_lab_dict.get('age',    {})   # {pid: [age_binned_5]}
gender_dict          = data_dict_lab_dict.get('gender', {})   # {pid: [female_oh]}  (MALE reduced)

# build per-patient first-value feature vectors
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

print("Building per-patient first-value feature vectors…")
records = {}

for pid in all_patient_ids:
    lab_dates   = set(basic_lab_dict.get(pid, {}).keys())
    vital_dates = set(basic_vitals_dict.get(pid, {}).keys())
    cbc_dates   = set(cbc_differential_dict.get(pid, {}).keys()) if cbc_differential_dict else set()
    all_dates   = lab_dates | vital_dates | cbc_dates
    if not all_dates:
        continue

    lab_vals   = {name: None for name in BASIC_TESTS}
    vital_vals = {name: None for name in BASIC_VITALS}
    cbc_vals   = {name: None for name in CBC_DIFFERENTIAL_TESTS} if cbc_differential_dict else {}

    # Walk dates ascending: for each feature take the first non-null value
    for date in sorted(all_dates):
        if date in basic_lab_dict.get(pid, {}):
            measurements = basic_lab_dict[pid][date]
            for i, feat_name in enumerate(BASIC_TESTS):
                if lab_vals[feat_name] is None and i < len(measurements):
                    fv = _try_float(measurements[i])
                    if fv is not None:
                        lab_vals[feat_name] = fv

        if date in basic_vitals_dict.get(pid, {}):
            measurements = basic_vitals_dict[pid][date]
            for i, feat_name in enumerate(BASIC_VITALS):
                if vital_vals[feat_name] is None and i < len(measurements):
                    fv = _try_float(measurements[i])
                    if fv is not None:
                        vital_vals[feat_name] = fv

        # Handle CBC differential features for parsimonious module
        if cbc_differential_dict and date in cbc_differential_dict.get(pid, {}):
            measurements = cbc_differential_dict[pid][date]
            for i, feat_name in enumerate(CBC_DIFFERENTIAL_TESTS):
                if cbc_vals[feat_name] is None and i < len(measurements):
                    fv = _try_float(measurements[i])
                    if fv is not None:
                        cbc_vals[feat_name] = fv

    # Demographics: age at first lab measurement (consistent with dataset_summaries_fixed.py)
    age_val = None
    if pid in age_dict and age_dict[pid]:
        try:
            age_bin = int(age_dict[pid][0])
            if msk_age_anchor:
                # Dates are days relative to sequencing; correct to age at first lab date
                first_date = min(all_dates) if all_dates else None
                if first_date is not None:
                    age_val = float(max(0, int((age_bin * 5 * 365 + first_date) / 365 / 5)))
                else:
                    age_val = float(age_bin)
            else:
                # CAIA / stanfordEHRSHOT / MIMICIV_hosp: already age at first measurement
                age_val = float(age_bin)
        except (ValueError, TypeError):
            pass

    female_val = _try_float(gender_dict[pid][0]) if (pid in gender_dict and gender_dict[pid]) else None

    record = {
        'gender': female_val,
        'age at seq': age_val,
        **lab_vals,
    }

    # Add CBC differential features if using parsimonious module
    if cbc_vals:
        record.update(cbc_vals)

    # Add vitals
    record.update(vital_vals)

    records[pid] = record

print(f"Patients with at least one recorded date: {len(records):,}")

# build DataFrame with clinical grouping order
feature_df = pd.DataFrame.from_dict(records, orient='index')

# order as in LIME plots: demographics, CBC, CBC differential, BMP, LFT, vitals

col_order = ['gender', 'age at seq']

# Add CBC tests (RBC panel)
for test in RBC_CBC_TESTS:
    if test in feature_df.columns:
        col_order.append(test)

# Add CBC Differential if present
if modules == 'parsimonious':
    for test in CBC_DIFFERENTIAL_TESTS:
        if test in feature_df.columns:
            col_order.append(test)

# Add BMP tests
for test in BMP_TESTS:
    if test in feature_df.columns:
        col_order.append(test)

# Add LFT tests
for test in LFT_TESTS:
    if test in feature_df.columns:
        col_order.append(test)

# Add vitals
for test in BASIC_VITALS:
    if test in feature_df.columns:
        col_order.append(test)

col_order = [c for c in col_order if c in feature_df.columns]
feature_df = feature_df[col_order]

n_valid = feature_df.notna().any(axis=1).sum()
print(f"Patients with at least one non-NaN feature: {n_valid:,}")

# display names (matching LIME plots)
FEATURE_DISPLAY_NAMES = {
    'gender': 'Gender',
    'age at seq': 'Age',
    # CBC
    'RBC': 'RBC',
    'MCV': 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'RDW': 'RDW',
    'HGB': 'HGB',
    'HCT': 'HCT',
    'WBC': 'WBC',
    'Platelets': 'Platelets',
    # CBC Differential
    'Neut': 'Neutrophil',
    'Mono': 'Monocytes',
    'Lymph': 'Lymphocytes',
    'Baso': 'Basophils',
    'Eos': 'Eosinophils',
    # BMP
    'Glucose': 'Glucose',
    'Calcium': 'Calcium',
    'Sodium': 'Sodium',
    'Potassium': 'Potassium',
    'CO2': 'CO2',
    'Creatinine': 'Creatinine',
    'Chloride': 'Chloride',
    'BUN': 'BUN',
    # LFT
    'Albumin': 'Albumin',
    'Protein, Total': 'Protein, Total',
    'ALP': 'ALP',
    'ALT': 'ALT',
    'AST': 'AST',
    'Bilirubin, Total': 'Bilirubin, Total',
    # Vitals
    'BMI': 'BMI',
    'Blood Pressure BP Systolic': 'Systolic BP',
    'Blood Pressure Diastolic': 'Diastolic BP',
    'O2': 'O2',
    'Pulse Heart Rate': 'Pulse Hrt Rate',
    'Resp Rate': 'Resp Rate',
    'Temperature C': 'Temperature',
}

display_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in col_order]

# color coding (matching LIME plots)
def get_feature_colors(feature_names):
    """Color features by clinical category: demographics gray, CBC red, CBC diff pink, BMP green, LFT orange, vitals blue."""
    colors = []

    for name in feature_names:
        if name in ['gender', 'age at seq']:
            colors.append('#808080')  # gray
        elif name in RBC_CBC_TESTS:
            colors.append('#d62728')  # red
        elif name in CBC_DIFFERENTIAL_TESTS:
            colors.append('#ff9896')  # pink
        elif name in BMP_TESTS:
            colors.append('#2ca02c')  # green
        elif name in LFT_TESTS:
            colors.append('#ff7f0e')  # orange
        elif name in BASIC_VITALS:
            colors.append('#1f77b4')  # blue
        else:
            colors.append('#7f7f7f')  # default gray

    return colors

feature_colors = get_feature_colors(col_order)

# compute correlation matrix
print("Computing Spearman correlation matrix…")
correlation_matrix = feature_df.corr(method='spearman')

# filter out features with < 100 patients
min_patients = 100
patient_counts = feature_df.notna().sum()
features_below_threshold = patient_counts[patient_counts < min_patients].index.tolist()

if features_below_threshold:
    print(f"\nFeatures with < {min_patients} patients (setting to NaN):")
    for feat in features_below_threshold:
        count = patient_counts[feat]
        print(f"  {feat}: {count} patients")

    # Set rows and columns to NaN for features below threshold
    for feat in features_below_threshold:
        correlation_matrix.loc[feat, :] = np.nan
        correlation_matrix.loc[:, feat] = np.nan
else:
    print(f"\nAll features have >= {min_patients} patients")

correlation_matrix.index = display_names
correlation_matrix.columns = display_names

# full correlation heatmap with color-coded labels
n_feats = len(correlation_matrix)
fig_size = max(14, n_feats * 0.6)

fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.95))
sns.heatmap(
    correlation_matrix,
    cmap='coolwarm', center=0,
    annot=False, cbar_kws={'shrink': 0.6},
    vmin=-1, vmax=1, ax=ax,
    square=True,
)

ax.set_title(
    f'Feature Pairwise Correlation Heatmap ({modules} module) — {dataset}\n'
    f'First Values per Patient (N={n_valid:,})',
    fontsize=14, fontweight='bold', pad=20, family='Arial',
)

# Color-code x-axis labels
for tick_label, color in zip(ax.get_xticklabels(), feature_colors):
    tick_label.set_color(color)
    tick_label.set_weight('bold')
    tick_label.set_fontsize(10)
    tick_label.set_family('Arial')

# Color-code y-axis labels
for tick_label, color in zip(ax.get_yticklabels(), feature_colors):
    tick_label.set_color(color)
    tick_label.set_weight('bold')
    tick_label.set_fontsize(10)
    tick_label.set_family('Arial')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

out_path = os.path.join(
    out_dir,
    f'feature_pairwise_correlation_heatmap_{modules}_{dataset}.pdf',
)
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {out_path}")

# Save correlation matrix as CSV for cross-dataset analysis
csv_path = os.path.join(
    out_dir,
    f'correlation_matrix_{modules}_{dataset}.csv',
)
correlation_matrix.to_csv(csv_path)
print(f"Saved: {csv_path}")

print("\nDone.")
