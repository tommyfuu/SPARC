#!/usr/bin/env python3
"""Cohort summary for one SPARC dataset: patient and measurement counts, visit dates per patient,
gender, and age bin at first measurement (for MSK, shifted from age at sequencing by the first date).

Usage: python dem_table_1_dataset_summaries.py -ds MSK_CAIA [-cp prior] [-od DIR]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# Path setup: resolve SPARC/data so dataset_loading.py is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # …/cohort_summaries/demographic_tables
# demographic_tables -> cohort_summaries -> SPARC
_SPARC_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))       # …/SPARC
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

# Argument parsing
parser = argparse.ArgumentParser(
    prog='dataset_summaries',
    description='Cohort summary statistics across SPARC datasets',
    epilog='Outputs a CSV summary and PDF plots to --output_dir.',
)
parser.add_argument(
    '-ds', '--dataset', type=str, default='MSK',
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
        'Directory for output CSV and plots.  '
        'Defaults to <SPARC_RESULTS_PATH>/results_vis/'
        'cohort_summaries/dataset_summaries/<dataset>.'
    ),
)
parser.add_argument(
    '-cp', '--caia_portion', type=str, default='prior',
    help='For CAIA datasets: prior | post | entire  (default: prior for MSK_CAIA dataset).',
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
    help='Module category: practical | parsimonious (default: practical).',
)
args = parser.parse_args()

# Resolve parameters
dataset = args.dataset
modules = args.modules

# dataset_loading expects 'stanfordEHRSHOT' (lower-case 's')
dataset_load = dataset
if dataset_load == 'StanfordEHRSHOT':
    dataset_load = 'stanfordEHRSHOT'

data_source_path = args.volume_path
caia_portion     = args.caia_portion
impact_subset    = args.impact_subset

# Whether this dataset stores age at sequencing (needs first-date correction)
msk_age_anchor = dataset_load in ('MSK', 'MSK_timeshift')

if args.output_dir:
    out_dir = args.output_dir
else:
    out_dir = os.path.join(
        sparc_paths.RESULTS_ROOT + '/results_vis/cohort_summaries/dataset_summaries',
        f'{modules}_{dataset}',
    )

if 'CAIA' in dataset and impact_subset:
    out_dir += '_impact_subset'

os.makedirs(out_dir, exist_ok=True)

print(f"Dataset          : {dataset}")
print(f"Modules          : {modules}")
print(f"CAIA portion     : {caia_portion}")
if 'CAIA' in dataset:
    print(f"Impact subset filter   : {impact_subset}")
print(f"Output directory : {out_dir}")

# Load data via dataset_loading.py
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

# Union of all splits = full cohort
all_patient_ids = set(
    list(split_dict.get('train', []))
    + list(split_dict.get('val',   []))
    + list(split_dict.get('test',  []))
)
print(f"Total patients after merging splits: {len(all_patient_ids):,}")

# Optional: restrict to IMPACT-sequenced patients (CAIA datasets only)
if impact_subset and 'CAIA' in dataset:
    _impact_path = sparc_paths.DATA_ROOT + '/MSK_CAIA/impact_deids.xlsx'
    _impact_ids  = set(pd.read_excel(_impact_path)['id'].astype(int).tolist())
    _before = len(all_patient_ids)
    all_patient_ids = all_patient_ids & _impact_ids
    print(f"Impact subset filter applied: {_before:,} → {len(all_patient_ids):,} patients")

n_patients = len(all_patient_ids)

basic_lab_dict       = data_dict_lab_dict.get('basic_lab',       {})
basic_vitals_dict    = data_dict_lab_dict.get('basic_vitals',    {})
cbc_differential_dict = data_dict_lab_dict.get('cbc_differential', {})  # Only present for parsimonious module
age_dict             = data_dict_lab_dict.get('age',    {})   # {pid: [age_bin]}
gender_dict          = data_dict_lab_dict.get('gender', {})   # {pid: [female_oh]}  (MALE reduced)

# Helper: count valid (non-NA) measurements
_INVALID_STRS = frozenset({'NA', 'nan', 'NAN', 'TNP', 'Negative', 'null', 'None', ''})

def _is_valid(v):
    if v is None:
        return False
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return False
    if str(v).strip() in _INVALID_STRS:
        return False
    try:
        fv = float(v)
        return not (np.isnan(fv) or np.isinf(fv))
    except (ValueError, TypeError):
        return False

def count_valid_measurements(data_dict, patient_ids):
    """Count non-NA measurement values across all patients in patient_ids."""
    total = 0
    for pid in patient_ids:
        if pid not in data_dict:
            continue
        for date_vals in data_dict[pid].values():
            for v in date_vals:
                if _is_valid(v):
                    total += 1
    return total

# 1. Measurement counts
print("Counting measurements…")
lab_counts    = count_valid_measurements(basic_lab_dict,    all_patient_ids)
vitals_counts = count_valid_measurements(basic_vitals_dict, all_patient_ids)
cbc_diff_counts = count_valid_measurements(cbc_differential_dict, all_patient_ids) if cbc_differential_dict else 0
total_counts  = lab_counts + vitals_counts + cbc_diff_counts

n_with_labs   = sum(1 for pid in all_patient_ids if pid in basic_lab_dict    and basic_lab_dict[pid])
n_with_vitals = sum(1 for pid in all_patient_ids if pid in basic_vitals_dict and basic_vitals_dict[pid])
n_with_cbc_diff = sum(1 for pid in all_patient_ids if pid in cbc_differential_dict and cbc_differential_dict[pid]) if cbc_differential_dict else 0

# 2. Unique visit-date distribution (union of lab + vitals + CBC diff dates per patient)
print("Computing visit-date distribution…")
dates_per_patient = []
for pid in all_patient_ids:
    lab_dates   = set(basic_lab_dict.get(pid, {}).keys())
    vital_dates = set(basic_vitals_dict.get(pid, {}).keys())
    cbc_dates   = set(cbc_differential_dict.get(pid, {}).keys()) if cbc_differential_dict else set()
    n_dates = len(lab_dates | vital_dates | cbc_dates)
    if n_dates > 0:
        dates_per_patient.append(n_dates)

dates_series = pd.Series(dates_per_patient)
dates_desc   = dates_series.describe(percentiles=[0.25, 0.5, 0.75])

# 3. Gender distribution
print("Computing gender distribution…")
n_female  = 0
n_male    = 0
n_unknown = 0

for pid in all_patient_ids:
    if pid not in gender_dict or not gender_dict[pid]:
        n_unknown += 1
        continue
    female_oh = gender_dict[pid][0]
    try:
        val = float(female_oh)
        if val == 1:
            n_female += 1
        else:
            n_male += 1
    except (ValueError, TypeError):
        n_unknown += 1

pct_female  = 100.0 * n_female  / n_patients if n_patients > 0 else 0.0
pct_male    = 100.0 * n_male    / n_patients if n_patients > 0 else 0.0
pct_unknown = 100.0 * n_unknown / n_patients if n_patients > 0 else 0.0

# 4. Age at first measurement distribution
print("Computing age-at-first-measurement distribution…")
age_bins_at_first_meas = []

for pid in all_patient_ids:
    if pid not in age_dict or not age_dict[pid]:
        continue

    try:
        age_bin = int(age_dict[pid][0])
    except (ValueError, TypeError):
        continue

    if msk_age_anchor:
        # Dates are in days relative to sequencing anchor; find earliest date
        lab_dates   = set(basic_lab_dict.get(pid, {}).keys())
        vital_dates = set(basic_vitals_dict.get(pid, {}).keys())
        all_dates   = lab_dates | vital_dates
        if not all_dates:
            # No measurement dates → fall back to sequencing age bin
            age_bins_at_first_meas.append(age_bin)
            continue
        first_date = min(all_dates)  # days since sequencing (can be negative or positive)
        # Estimated total days of life at first measurement
        total_days = age_bin * 5 * 365 + first_date
        # Convert back to 5-year bin
        estimated_bin = max(0, int(total_days / 365 / 5))
        age_bins_at_first_meas.append(estimated_bin)
    else:
        # CAIA / stanfordEHRSHOT / MIMICIV_hosp: age dict is already age at first measurement
        age_bins_at_first_meas.append(age_bin)

age_series = pd.Series(age_bins_at_first_meas)
# Convert bins to year-range labels for readability
age_labels = {b: f"{b*5}–{b*5+4}" for b in range(21)}

age_counts = age_series.value_counts().sort_index()
age_counts.index = [age_labels.get(b, str(b)) for b in age_counts.index]

age_desc = age_series.describe(percentiles=[0.25, 0.5, 0.75])
# Map summary statistics to year equivalents for printing
age_median_years_lo = int(age_desc['50%']) * 5
age_median_years_hi = int(age_desc['50%']) * 5 + 4

# 5. Print summary
SEPARATOR = "=" * 70

print(f"\n{SEPARATOR}")
print(f"DATASET SUMMARY: {dataset}  (caia_portion={caia_portion})")
print(SEPARATOR)
print(f"\n{'PATIENTS':}")
print(f"  Total patients                 : {n_patients:>12,}")
print(f"  Patients with lab data         : {n_with_labs:>12,}  ({100*n_with_labs/n_patients:.1f}%)")
print(f"  Patients with vitals data      : {n_with_vitals:>12,}  ({100*n_with_vitals/n_patients:.1f}%)")

print(f"\n{'MEASUREMENTS (non-NA values)':}")
print(f"  Lab measurements               : {lab_counts:>12,}")
print(f"  Vitals measurements            : {vitals_counts:>12,}")
print(f"  Total measurements             : {total_counts:>12,}")
print(f"  Avg lab measurements / patient : {lab_counts/max(n_with_labs,1):>12.1f}")

print(f"\n{'UNIQUE VISIT DATES PER PATIENT':}")
print(f"  N patients with ≥1 date        : {len(dates_per_patient):>12,}")
print(f"  Mean                           : {dates_desc['mean']:>12.1f}")
print(f"  Std                            : {dates_desc['std']:>12.1f}")
print(f"  Min                            : {dates_desc['min']:>12.0f}")
print(f"  25th percentile                : {dates_desc['25%']:>12.0f}")
print(f"  Median                         : {dates_desc['50%']:>12.0f}")
print(f"  75th percentile                : {dates_desc['75%']:>12.0f}")
print(f"  Max                            : {dates_desc['max']:>12.0f}")

print(f"\n{'GENDER DISTRIBUTION':}")
print(f"  Female                         : {n_female:>8,}  ({pct_female:5.1f}%)")
print(f"  Male                           : {n_male:>8,}  ({pct_male:5.1f}%)")
print(f"  Unknown / missing              : {n_unknown:>8,}  ({pct_unknown:5.1f}%)")

print(f"\n{'AGE AT FIRST MEASUREMENT (5-year bins)':}")
age_src = "age at sequencing corrected to first-measurement date" if msk_age_anchor else "age at first measurement"
print(f"  (Source: {age_src})")
print(f"  N patients with age data       : {len(age_bins_at_first_meas):>12,}")
print(f"  Median age bin                 : {int(age_desc['50%']):>12}  ({age_median_years_lo}–{age_median_years_hi} yrs)")
print(f"  Mean  age bin                  : {age_desc['mean']:>12.2f}  (~{age_desc['mean']*5:.0f} yrs midpoint)")
print(f"  Std   age bin                  : {age_desc['std']:>12.2f}")
print()
print("  Age group (yrs)  |  Count  |   %")
print("  " + "-" * 40)
for label, cnt in age_counts.items():
    pct = 100.0 * cnt / len(age_bins_at_first_meas) if age_bins_at_first_meas else 0.0
    print(f"  {label:<16} | {cnt:>7,} | {pct:5.1f}%")

print(f"\n{SEPARATOR}")

# 6. Save summary CSV
summary_rows = [
    ('n_patients',                    n_patients),
    ('n_with_labs',                   n_with_labs),
    ('n_with_vitals',                 n_with_vitals),
    ('lab_measurements',              lab_counts),
    ('vitals_measurements',           vitals_counts),
    ('total_measurements',            total_counts),
    ('avg_lab_measurements_per_pt',   round(lab_counts / max(n_with_labs, 1), 2)),
    ('n_dates_per_pt_mean',           round(float(dates_desc['mean']), 2)),
    ('n_dates_per_pt_std',            round(float(dates_desc['std']),  2)),
    ('n_dates_per_pt_min',            int(dates_desc['min'])),
    ('n_dates_per_pt_p25',            int(dates_desc['25%'])),
    ('n_dates_per_pt_median',         int(dates_desc['50%'])),
    ('n_dates_per_pt_p75',            int(dates_desc['75%'])),
    ('n_dates_per_pt_max',            int(dates_desc['max'])),
    ('n_female',                      n_female),
    ('n_male',                        n_male),
    ('n_gender_unknown',              n_unknown),
    ('pct_female',                    round(pct_female, 2)),
    ('pct_male',                      round(pct_male, 2)),
    ('age_median_bin',                int(age_desc['50%'])),
    ('age_median_years_lo',           age_median_years_lo),
    ('age_median_years_hi',           age_median_years_hi),
    ('age_mean_bin',                  round(float(age_desc['mean']), 2)),
    ('age_std_bin',                   round(float(age_desc['std']),  2)),
    ('n_patients_with_age',           len(age_bins_at_first_meas)),
]

summary_df = pd.DataFrame(summary_rows, columns=['metric', 'value'])
csv_path = os.path.join(out_dir, f'dataset_summary_{dataset}.csv')
summary_df.to_csv(csv_path, index=False)
print(f"Saved summary CSV : {csv_path}")

# Age distribution CSV
age_dist_df = age_counts.reset_index()
age_dist_df.columns = ['age_group_years', 'count']
age_dist_df['pct'] = (age_dist_df['count'] / age_dist_df['count'].sum() * 100).round(2)
age_dist_csv = os.path.join(out_dir, f'age_distribution_{dataset}.csv')
age_dist_df.to_csv(age_dist_csv, index=False)
print(f"Saved age dist CSV: {age_dist_csv}")

# 7. Plots

# Plot A: age distribution bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(age_dist_df))
ax.bar(x, age_dist_df['count'], color='#5B9AA9', edgecolor='black', linewidth=0.5, alpha=0.9)
ax.set_xticks(list(x))
ax.set_xticklabels(age_dist_df['age_group_years'], rotation=45, ha='right', fontsize=10)
ax.set_xlabel('Age group at first measurement (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of patients', fontsize=12, fontweight='bold')
age_src_short = 'corrected from seq. age' if msk_age_anchor else 'age at first measurement'
ax.set_title(
    f'Age Distribution at First Measurement — {dataset}\n'
    f'({age_src_short}; caia_portion={caia_portion})',
    fontsize=13, fontweight='bold', pad=12,
)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim(0, None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
age_plot = os.path.join(out_dir, f'age_distribution_{dataset}.pdf')
plt.savefig(age_plot, bbox_inches='tight')
plt.close()
print(f"Saved age plot    : {age_plot}")

# Plot B: gender bar chart
gender_labels  = ['Female', 'Male', 'Unknown']
gender_values  = [n_female, n_male, n_unknown]
gender_colors  = ['#E8A0BF', '#6BA3BE', '#C0C0C0']

fig2, ax2 = plt.subplots(figsize=(6, 5))
bars = ax2.bar(gender_labels, gender_values,
               color=gender_colors, edgecolor='black', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, gender_values):
    pct = 100.0 * val / n_patients if n_patients > 0 else 0.0
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(gender_values) * 0.01,
        f'{val:,}\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=10,
    )
ax2.set_ylabel('Number of patients', fontsize=12, fontweight='bold')
ax2.set_title(
    f'Gender Distribution — {dataset}\n(caia_portion={caia_portion})',
    fontsize=13, fontweight='bold', pad=12,
)
ax2.set_ylim(0, max(gender_values) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
gender_plot = os.path.join(out_dir, f'gender_distribution_{dataset}.pdf')
plt.savefig(gender_plot, bbox_inches='tight')
plt.close()
print(f"Saved gender plot : {gender_plot}")

# Plot C: visit-date count distribution (histogram)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.hist(dates_per_patient, bins=50, color='#7B68A6', edgecolor='black', linewidth=0.3, alpha=0.85)
ax3.axvline(x=float(dates_desc['50%']), color='red',    linestyle='--', linewidth=1.5,
            label=f"Median = {int(dates_desc['50%'])}")
ax3.axvline(x=float(dates_desc['mean']), color='orange', linestyle='--', linewidth=1.5,
            label=f"Mean   = {dates_desc['mean']:.1f}")
ax3.set_xlabel('Unique visit dates per patient', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of patients', fontsize=12, fontweight='bold')
ax3.set_title(
    f'Visit Date Count Distribution — {dataset}\n(caia_portion={caia_portion})',
    fontsize=13, fontweight='bold', pad=12,
)
ax3.legend(fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.tight_layout()
dates_plot = os.path.join(out_dir, f'visit_dates_distribution_{dataset}.pdf')
plt.savefig(dates_plot, bbox_inches='tight')
plt.close()
print(f"Saved dates plot  : {dates_plot}")

print("\nDone.")
