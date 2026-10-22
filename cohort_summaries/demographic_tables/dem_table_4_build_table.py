#!/usr/bin/env python3
"""Build the SPARC dataset demographics table: demographics_table.csv (percentages) and/or the
Table S1 counts CSV with race/ethnicity denominators and cancer-type counts. Suppressed cells
('n<=10') are kept verbatim; --check flags denominators larger than the cohort.
"""

import argparse
import csv
import glob
import os
import re
import sys

import pandas as pd

# shared ctypes_parse from cohort_summaries/cancer_types/
_CTYPES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cancer_types')
if _CTYPES not in sys.path:
    sys.path.insert(0, _CTYPES)

# cancer-type paths honour SPARC_CANCER_TYPES_DIR / SPARC_CANCER_TYPES_OUT
from ctypes_parse import (DATA_DIR as CTYPES_DIR, OUTPUT_DIR as CTYPES_OUT_DIR,
                          CANCER_TYPES_LIST_PATH)

DEFAULT_SOURCE_DIR = os.environ.get(
    'SPARC_COHORT_SUMMARIES_DIR', '/path/to/cohort_summaries')

# dataset registry, in publication row order
#   display     name in the counts table (label only, not used for file lookup)
#   pct_name    name in the percentages table
#   folder      subdirectory of dataset_summaries/
#   race        harmonized/<race>_{race,ethnicity}_harmonized.csv stem
#   ctypes      ctypes_<ctypes>_summary.txt stem
# None means no extract exists -> blank cells, not zeros.
DATASETS = [
    dict(display='MSK',               pct_name='MSK_CAIA',
         folder='parsimonious_MSK_CAIA',           race='MSK_CAIA',           ctypes='MSK'),
    dict(display='MSK (prospective)', pct_name='MSK_CAIA_timeshift',
         folder='parsimonious_MSK_CAIA_timeshift', race='MSK_CAIA_timeshift', ctypes=None),
    dict(display='CAIA-DFCI',         pct_name='DFCI_CAIA_all_060526',
         folder='parsimonious_DFCI_CAIA_all_060526', race='DFCI_CAIA',       ctypes='DFCI'),
    dict(display='CAIA-FHCC',         pct_name='FHCC_CAIA',
         folder='parsimonious_FHCC_CAIA',          race='FHCC_CAIA',          ctypes='FH'),
    dict(display='CAIA-JHU',          pct_name='JHU_CAIA',
         folder='parsimonious_JHU_CAIA',           race='JHU_CAIA',           ctypes=None),
    dict(display='INSIGHT-A',         pct_name='INSIGHT_A',
         folder='parsimonious_INSIGHT_A',          race='insight_A',          ctypes='INSIGHT-A'),
    dict(display='INSIGHT-B',         pct_name='INSIGHT_B',
         folder='parsimonious_INSIGHT_B',          race='insight_B',          ctypes='INSIGHT-B'),
    dict(display='INSIGHT-C',         pct_name='INSIGHT_C',
         folder='parsimonious_INSIGHT_C',          race='insight_C',          ctypes='INSIGHT-C'),
    dict(display='INSIGHT-D',         pct_name='INSIGHT_D',
         folder='parsimonious_INSIGHT_D',          race='insight_D',          ctypes='INSIGHT-D'),
    dict(display='INSIGHT-E',         pct_name='INSIGHT_E',
         folder='parsimonious_INSIGHT_E',          race='insight_E',          ctypes='INSIGHT-E'),
    dict(display='INSIGHT-F',         pct_name='INSIGHT_F',
         folder='parsimonious_INSIGHT_F',          race='insight_F',          ctypes='INSIGHT-F'),
    dict(display='MIMIC IV-hosp',      pct_name='MIMICIV_hosp',
         folder='parsimonious_MIMICIV_hosp',       race='MIMICIV_hosp',       ctypes='MIMICIV-hosp'),
    dict(display='EHRSHOT',           pct_name='StanfordEHRSHOT',
         folder='parsimonious_StanfordEHRSHOT',    race='EHRSHOT',            ctypes='EHRSHOT'),
]

RACE_CATEGORIES = [
    'White', 'Black or African American', 'Asian',
    'American Indian or Alaska Native',
    'Native Hawaiian or Other Pacific Islander',
    'Other/Unknown/Refused',
]
ETHNICITY_CATEGORIES = ['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown']

# Short column labels used by the percentages table.
RACE_PCT_LABEL = {
    'White': 'White (%)',
    'Black or African American': 'Black (%)',
    'Asian': 'Asian (%)',
    'American Indian or Alaska Native': 'AIAN (%)',
    'Native Hawaiian or Other Pacific Islander': 'NHPI (%)',
    'Other/Unknown/Refused': 'Other/Unknown (%)',
}
ETH_PCT_LABEL = {
    'Hispanic or Latino': 'Hispanic/Latino (%)',
    'Not Hispanic or Latino': 'Not Hispanic/Latino (%)',
    'Unknown': 'Ethnicity Unknown (%)',
}

FIRST_MAPPED_RE = (
    r'CANCER TYPE SUMMARY \(first-mapped type\).*?\n'
    r'.*?parent_cancer_type\s+#patients\s+% of cohort\n-+\n(.*)'
)
ROW_RE = re.compile(r'^(.*?)\s+([\d,]+)\s+([\d.]+)%$')
TOTAL_PATIENTS_RE = re.compile(
    r'Total patients with >=1 mapped cancer type:\s*([\d,]+)'
)

SUPPRESSED = 'n<=10'


# small helpers
def is_suppressed(cell):
    s = str(cell).strip().lower()
    return '<=' in s or '<' in s or 'suppress' in s


def suppress_count(count):
    """'<=10' for anything at or below 10, else the integer as a string."""
    if isinstance(count, str):
        if is_suppressed(count):
            return '<=10'
        try:
            count = float(count)
        except ValueError:
            return '<=10'
    return '<=10' if count <= 10 else str(int(count))


def _first_glob(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


# readers
def read_dataset_summary(folder):
    """n_patients / pct_male / age mean+std (converted from 5-year bins)."""
    path = (_first_glob(folder, 'dataset_summary_*.csv')
            or _first_glob(folder, 'dataset_summary_*_suppressed.csv'))
    if path is None:
        return None
    df = pd.read_csv(path)
    m = dict(zip(df['metric'], df['value']))
    mean_bin, std_bin = m.get('age_mean_bin'), m.get('age_std_bin')
    # Bin index -> years: lower bound is 5*bin, so the midpoint is 5*bin + 2.5.
    age_mean = mean_bin * 5 + 2.5 if mean_bin is not None else None
    age_std = std_bin * 5 if std_bin is not None else None
    return dict(n_patients=m.get('n_patients'), pct_male=m.get('pct_male'),
                age_mean=age_mean, age_std=age_std, source=os.path.basename(path))


def read_age_bins(folder):
    """({bin_label: suppressed_count}, file_found). <5 folds into 'LE4', 85+ into 'GE85';
    file_found=False means no age file (blank cells rather than zeros)."""
    path = _first_glob(folder, 'age_distribution_*.csv')
    if path is None:
        return {}, False
    df = pd.read_csv(path)
    bins, young, old = {}, 0, 0
    for _, row in df.iterrows():
        label, count = str(row['age_group_years']), row['count']
        if isinstance(count, str):
            # suppressed cells contribute their midpoint to folded bins
            count = 5 if is_suppressed(count) else _to_float(count, 5)
        if label.startswith('-') or label in ('0-4', '0–4'):
            young += count
            continue
        sep = '–' if '–' in label else ('-' if '-' in label else None)
        if sep:
            lo = label.split(sep)[0]
            if lo.isdigit() and int(lo) >= 85:
                old += count
                continue
        bins[label] = bins.get(label, 0) + count
    if young:
        bins['LE4'] = young
    if old:
        bins['GE85'] = old
    return {k: suppress_count(v) for k, v in bins.items()}, True


def _to_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_harmonized(race_dir, stem, kind):
    """{category: cell_as_written} or None. Cells stay strings so 'n<=10' survives."""
    if not stem:
        return None
    path = os.path.join(race_dir, f'{stem}_{kind}_harmonized.csv')
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def race_denominator(eth_cells):
    """Ethnicity total; used as the denominator because race tables are suppressed more often."""
    if not eth_cells:
        return None
    return sum(int(v) for v in eth_cells.values() if v.isdigit())


def read_ctypes(stem):
    """({type: count}, total_mapped) from ctypes_<stem>_summary.txt, or (None, None)."""
    if not stem:
        return None, None
    path = os.path.join(CTYPES_DIR, f'ctypes_{stem}_summary.txt')
    if not os.path.exists(path):
        return None, None
    text = open(path).read()
    compact = '\n'.join(l for l in text.split('\n') if l.strip())
    tm = TOTAL_PATIENTS_RE.search(compact)
    total = int(tm.group(1).replace(',', '')) if tm else None
    m = re.search(FIRST_MAPPED_RE, compact, re.S)
    if not m:
        return None, total
    counts = {}
    for line in m.group(1).split('\n'):
        line = line.strip()
        if not line or line.startswith('='):
            break
        mm = ROW_RE.match(line)
        if mm:
            label, count, _ = mm.groups()
            counts[label.strip()] = int(count.replace(',', ''))
    return counts, total


def load_named_types():
    """The 29 named parent types (excluding 'Unknown Primary')."""
    with open(CANCER_TYPES_LIST_PATH) as f:
        types = [l.strip() for l in f if l.strip()]
    return [t for t in types if t != 'Unknown Primary']


# collection
def collect(summaries_dir, race_dir):
    """One record per registry entry, with everything read but nothing formatted."""
    records = []
    for spec in DATASETS:
        folder = os.path.join(summaries_dir, spec['folder'])
        demo = read_dataset_summary(folder) if os.path.isdir(folder) else None
        if demo is None:
            print(f"  WARNING: no dataset summary for {spec['display']} "
                  f"({spec['folder']})")
        rec = dict(spec)
        rec['demo'] = demo
        rec['age_bins'], rec['has_age_file'] = (
            read_age_bins(folder) if os.path.isdir(folder) else ({}, False))
        rec['race'] = read_harmonized(race_dir, spec['race'], 'race')
        rec['eth'] = read_harmonized(race_dir, spec['race'], 'ethnicity')
        rec['denom'] = race_denominator(rec['eth'])
        rec['ctypes'], rec['ctype_total'] = read_ctypes(spec['ctypes'])
        records.append(rec)
    return records


def age_label(internal, style):
    """Internal bin label -> the column name each table uses."""
    if style == 'counts':
        if internal == 'LE4':
            return 'Age <=4'
        if internal == 'GE85':
            return 'Age>=85'          # no space, matching the published table
        return 'Age ' + internal.replace('–', '-')
    if internal == 'LE4':
        return 'Age ≤4'
    if internal == 'GE85':
        return 'Age ≥85'
    return 'Age ' + internal


def sorted_age_bins(records):
    labels = set()
    for r in records:
        labels.update(r['age_bins'])

    def key(label):
        if label == 'LE4':
            return -1
        if label == 'GE85':
            return 999
        sep = '–' if '–' in label else ('-' if '-' in label else None)
        lo = label.split(sep)[0] if sep else label
        return int(lo) if lo.isdigit() else 0

    return sorted(labels, key=key)


# output: percentages
def build_pct_table(records, age_labels):
    rows = []
    for r in records:
        if r['demo'] is None:
            continue
        d = r['demo']
        row = {
            'Dataset': r['pct_name'],
            'N Patients': int(d['n_patients']) if d['n_patients'] is not None else None,
            'Male (%)': f"{d['pct_male']:.2f}" if d['pct_male'] is not None else None,
            'Age at First Lab (mean ± std)': (
                f"{d['age_mean']:.1f} ± {d['age_std']:.1f}"
                if d['age_mean'] is not None and d['age_std'] is not None else None),
        }
        for cats, cells, labels in ((RACE_CATEGORIES, r['race'], RACE_PCT_LABEL),
                                    (ETHNICITY_CATEGORIES, r['eth'], ETH_PCT_LABEL)):
            total = sum(int(v) for v in (cells or {}).values() if v.isdigit())
            for cat in cats:
                col = labels[cat]
                cell = (cells or {}).get(cat)
                if cell is None or total == 0:
                    row[col] = ''
                elif cell.isdigit():
                    row[col] = f"{100 * int(cell) / total:.1f}"
                else:
                    row[col] = cell          # suppressed, kept verbatim
        for label in age_labels:
            row[age_label(label, 'pct')] = r['age_bins'].get(label, '')
        rows.append(row)
    # percentages table is sorted alphabetically by dataset label
    rows.sort(key=lambda r: r['Dataset'])
    return pd.DataFrame(rows)


# output: counts (Table S1 style)
def build_counts_table(records, age_labels, named_types):
    header = (['Dataset', '# Patients', 'Male (%)', 'Age at First Lab (mean ± std)',
               'Race/ethnicity denominator']
              + [f'{c} (race)' for c in RACE_CATEGORIES]
              + [f'{c} (ethnicity)' for c in ETHNICITY_CATEGORIES]
              + [age_label(l, 'counts') for l in age_labels]
              + ['Total patients with >=1 mapped cancer type']
              + named_types + ['Other/Unknown Primary'])
    rows = []
    for r in records:
        if r['demo'] is None:
            continue
        d = r['demo']
        row = [r['display'],
               int(d['n_patients']) if d['n_patients'] is not None else '',
               f"{d['pct_male']:.2f}" if d['pct_male'] is not None else '',
               (f"{d['age_mean']:.1f} ± {d['age_std']:.1f}"
                if d['age_mean'] is not None and d['age_std'] is not None else ''),
               r['denom'] if r['denom'] is not None else '']
        for cats, cells in ((RACE_CATEGORIES, r['race']),
                            (ETHNICITY_CATEGORIES, r['eth'])):
            for cat in cats:
                # no extract -> blank; category absent from an extract -> 0
                row.append('' if cells is None else cells.get(cat, 0))
        row += [r['age_bins'].get(l, 0 if r['has_age_file'] else '')
                for l in age_labels]
        if r['ctypes'] is None:
            row += [''] * (len(named_types) + 2)
        else:
            row.append(r['ctype_total'] if r['ctype_total'] is not None else '')
            row += [r['ctypes'].get(t, 0) for t in named_types]
            # residual bucket: 'Unknown Primary' plus any site 'others' catch-all
            residual = r['ctypes'].get('Unknown Primary', 0) + r['ctypes'].get('others', 0)
            row.append(residual)
        rows.append(row)
    return header, rows


# consistency report
def report(records):
    print()
    print('CONSISTENCY REPORT')
    print()
    print(f"{'dataset':<20}{'# patients':>11}{'race denom':>12}{'ctype total':>13}  flags")
    print('-' * 78)
    issues = []
    for r in records:
        if r['demo'] is None or r['demo']['n_patients'] is None:
            continue
        n = int(r['demo']['n_patients'])
        denom, ct = r['denom'], r['ctype_total']
        flags = []
        if denom is not None and denom > n:
            flags.append(f'race denom +{denom - n}')
            issues.append((r['display'], 'race', denom - n))
        if ct is not None and ct > n:
            flags.append(f'ctype total +{ct - n}')
            issues.append((r['display'], 'ctype', ct - n))
        print(f"{r['display']:<20}{n:>11}"
              f"{(denom if denom is not None else '--'):>12}"
              f"{(ct if ct is not None else '--'):>13}"
              f"  {'; '.join(flags)}")
    print('-' * 78)
    if not issues:
        print('No denominator exceeds its dataset patient count.')
    else:
        print(f'{len(issues)} discrepancies: a denominator aggregated over a wider')
        print('population than the modelling cohort (race extract or cancer-type')
        print('summary not restricted to the cohort).')
    return issues


def parse_args():
    p = argparse.ArgumentParser(
        description='Build the SPARC dataset demographics table(s).')
    p.add_argument('-s', '--source-dir', default=DEFAULT_SOURCE_DIR,
                   help='root cohort_summaries directory (default: %(default)s)')
    p.add_argument('-d', '--dataset-summaries-dir', default=None,
                   help='default: <source-dir>/dataset_summaries')
    p.add_argument('-r', '--race-ethnicity-dir', default=None,
                   help='default: <source-dir>/race_ethnicity/harmonized')
    p.add_argument('-o', '--out-dir', default=None,
                   help='where to write demographics_table.csv '
                        '(default: <source-dir>)')
    p.add_argument('--counts-out-dir', default=None,
                   help='where to write the Table S1 counts CSV (default: the '
                        'cancer_types dir from ctypes_parse, which is where the '
                        'published copy lives)')
    p.add_argument('-f', '--format', choices=['pct', 'counts', 'both'],
                   default='both', help='which table(s) to emit (default: both)')
    p.add_argument('--no-check', action='store_true',
                   help='skip the consistency report')
    return p.parse_args()


def main():
    args = parse_args()
    summaries = args.dataset_summaries_dir or os.path.join(args.source_dir,
                                                           'dataset_summaries')
    race = args.race_ethnicity_dir or os.path.join(args.source_dir,
                                                   'race_ethnicity', 'harmonized')
    out_dir = args.out_dir or args.source_dir
    # counts table defaults to the cancer-type output dir
    counts_dir = args.counts_out_dir or (args.out_dir if args.out_dir
                                         else CTYPES_OUT_DIR)

    if not os.path.isdir(summaries):
        raise SystemExit(f'dataset summaries directory not found: {summaries}')
    if not os.path.isdir(race):
        print(f'WARNING: race/ethnicity directory not found: {race}')
        print('         race and ethnicity columns will be blank')
    os.makedirs(out_dir, exist_ok=True)

    print(f'dataset summaries: {summaries}')
    print(f'race/ethnicity:    {race}')
    print(f'cancer types:      {CTYPES_DIR}')
    print(f'output (pct):      {out_dir}')
    print(f'output (counts):   {counts_dir}\n')

    records = collect(summaries, race)
    age_labels = sorted_age_bins(records)
    named_types = load_named_types()

    if args.format in ('pct', 'both'):
        df = build_pct_table(records, age_labels)
        path = os.path.join(out_dir, 'demographics_table.csv')
        df.to_csv(path, index=False)
        print(f'wrote {path}  ({df.shape[0]} datasets x {df.shape[1]} columns)')

    if args.format in ('counts', 'both'):
        header, rows = build_counts_table(records, age_labels, named_types)
        os.makedirs(counts_dir, exist_ok=True)
        path = os.path.join(
            counts_dir,
            'SPARC_Supplementary_Tables - Table S1 Dataset Demographics'
            '_with_ctypes_and_race.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f'wrote {path}  ({len(rows)} datasets x {len(header)} columns)')

    missing_race = [r['display'] for r in records
                    if r['race'] is None and r['display'] != 'DFCI-CAIA']
    missing_ct = [r['display'] for r in records if r['ctypes'] is None]
    if missing_race:
        print(f'\nno race/ethnicity extract: {", ".join(missing_race)}')
    if missing_ct:
        print(f'no cancer-type summary:    {", ".join(missing_ct)}')

    if not args.no_check:
        report(records)


if __name__ == '__main__':
    main()
