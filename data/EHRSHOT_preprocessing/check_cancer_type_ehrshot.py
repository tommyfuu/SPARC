#!/usr/bin/env python3
"""Map EHRSHOT cancer diagnoses to the 30 CAIA cancer types.

Writes DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_stanfordEHRSHOT.pickle,
{patient_id: {delta_day: [cancer_type, ...]}}, same shape as the MSK_CAIA and
INSIGHT versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EHRSHOT_CSV, COHORT_PICKLE, OUTPUT_PATH, CANCER_TYPES_CODE_DIR
from LOINC_to_test_dict import LOINIC_TO_BASIC_TEST_NAMES, LOINC_TO_VITALS_NAMES

sys.path.insert(0, CANCER_TYPES_CODE_DIR)
from check_cancer_types_insight import (
    CANCER_TYPES, OTHERS, SECONDARY, UNREACHABLE_TYPES,
    ICD10_RULES, normalise_code, resolve_code, self_test as _insight_self_test,
)


DEFAULT_OUT_DIR = OUTPUT_PATH

CHUNK_SIZE = 1_000_000

BASIC_LAB_VITALS_LOINC = {
    f'LOINC/{c}' for c in
    set(LOINIC_TO_BASIC_TEST_NAMES.keys()) | set(LOINC_TO_VITALS_NAMES.keys())
}


# stage 1: ICDO3 topography -> cancer type
def extract_topography(icdo3_code: str) -> str:
    """'8070/3-C34.9' -> 'C34.9'; 'NULL-C50.9' -> 'C50.9'; 'C64.9' -> 'C64.9'."""
    return icdo3_code.rsplit('-', 1)[-1]


def scan_observed_icdo3_codes(cohort_ids: set) -> Counter:
    """Count every distinct raw ICDO3/ code among condition_occurrence rows
    for the included cohort."""
    observed = Counter()
    if not os.path.exists(EHRSHOT_CSV):
        print(f'  ! missing: {EHRSHOT_CSV}')
        return observed
    print(f'  scanning {os.path.basename(EHRSHOT_CSV)} for ICDO3 condition codes ...', flush=True)
    n = 0
    for chunk in pd.read_csv(EHRSHOT_CSV, dtype=str, chunksize=CHUNK_SIZE,
                             usecols=['patient_id', 'code', 'omop_table']):
        n += len(chunk)
        chunk['patient_id'] = pd.to_numeric(chunk['patient_id'], errors='coerce').astype('Int64')
        cond = chunk[(chunk['omop_table'] == 'condition_occurrence') &
                     chunk['patient_id'].isin(cohort_ids)]
        icdo3 = cond[cond['code'].str.startswith('ICDO3/', na=False)]
        for code in icdo3['code']:
            observed[code] += 1
        if n % (CHUNK_SIZE * 10) == 0:
            print(f'    {n/1e6:.0f}M rows, {len(observed):,} distinct ICDO3 codes', flush=True)
    print(f'    done: {n:,} rows', flush=True)
    return observed


def build_code_map(out_dir: str, cohort_ids: set, rebuild: bool) -> dict:
    """Return {raw_icdo3_code: {...}} and persist it."""
    json_path = os.path.join(out_dir, 'icd_to_cancer_type_map_ehrshot.json')
    csv_path = os.path.join(out_dir, 'icd_to_cancer_type_map_ehrshot.csv')

    if os.path.exists(json_path) and not rebuild:
        with open(json_path) as f:
            cached = json.load(f)
        print(f'Loaded cached code map: {len(cached):,} codes  ({json_path})')
        return cached

    print('Building ICDO3 topography -> cancer type map from the codes actually present ...')
    observed = scan_observed_icdo3_codes(cohort_ids)

    code_map = {}
    for raw_code, n in observed.items():
        icdo3_body = raw_code[len('ICDO3/'):]
        topo = normalise_code(extract_topography(icdo3_body))
        ctype, tier, rule = resolve_code(topo, '10')
        code_map[raw_code] = {
            'topography': topo,
            'cancer_type': ctype,
            'tier': tier,
            'matched_rule': rule,
            'n_rows': n,
        }

    with open(json_path, 'w') as f:
        json.dump(code_map, f, indent=1, sort_keys=True)

    rows = sorted(code_map.values(), key=lambda r: (-r['n_rows'], r['topography']))
    for r, raw_code in zip(rows, sorted(code_map, key=lambda k: (-code_map[k]['n_rows'], code_map[k]['topography']))):
        r['raw_code'] = raw_code
    pd.DataFrame(rows)[['raw_code', 'topography', 'cancer_type', 'tier',
                        'matched_rule', 'n_rows']].to_csv(csv_path, index=False)

    unmapped = [r for r in rows if r['matched_rule'] is None]
    print(f'  {len(code_map):,} distinct ICDO3 codes')
    print(f'  {len(code_map) - len(unmapped):,} matched a rule, '
          f'{len(unmapped):,} unmatched')
    if unmapped:
        print('  unmatched (first 20):',
              ', '.join(f"{r['topography']}({r['n_rows']})" for r in unmapped[:20]))
    print(f'  -> {json_path}')
    print(f'  -> {csv_path}')
    return code_map


def active_lookup(code_map: dict, include_in_situ: bool,
                  include_secondary: bool, drop_others: bool) -> dict:
    """{raw_icdo3_code: cancer_type} for the tiers enabled this run."""
    keep_tiers = {'primary', 'unknown'}
    if include_in_situ:
        keep_tiers.add('in_situ')
    if include_secondary:
        keep_tiers.add('secondary')

    lookup = {}
    for raw_code, rec in code_map.items():
        ctype, tier = rec['cancer_type'], rec['tier']
        if ctype is None or tier not in keep_tiers:
            continue
        if drop_others and ctype == OTHERS:
            continue
        lookup[raw_code] = ctype
    return lookup


def load_cohort() -> set:
    with open(COHORT_PICKLE, 'rb') as f:
        cohort = set(pickle.load(f).keys())
    print(f'  cohort: {len(cohort):,} included patients  ({os.path.basename(COHORT_PICKLE)})')
    return cohort


def load_reference_dates(cache_path: str, rebuild: bool, cohort_ids: set) -> dict:
    """First basic-lab/vitals LOINC measurement date per patient, matching
    dictionarify.py's first_lab_test_date_dict convention."""
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, 'rb') as f:
            ref = pickle.load(f)
        print(f'Loaded {len(ref):,} reference dates from {cache_path}')
        return ref

    print('Building reference dates (first basic-lab/vitals date per patient) ...')
    first_date: dict = {}
    n = 0
    for chunk in pd.read_csv(EHRSHOT_CSV, dtype=str, chunksize=CHUNK_SIZE,
                             usecols=['patient_id', 'start', 'code', 'omop_table']):
        n += len(chunk)
        chunk['patient_id'] = pd.to_numeric(chunk['patient_id'], errors='coerce').astype('Int64')
        m = chunk[(chunk['omop_table'] == 'measurement') &
                  chunk['code'].isin(BASIC_LAB_VITALS_LOINC) &
                  chunk['patient_id'].isin(cohort_ids)]
        if len(m):
            dates = pd.to_datetime(m['start'], errors='coerce')
            for pid, d in zip(m['patient_id'], dates):
                if pd.isna(d):
                    continue
                cur = first_date.get(pid)
                if cur is None or d < cur:
                    first_date[pid] = d
        if n % (CHUNK_SIZE * 10) == 0:
            print(f'    {n/1e6:.0f}M rows, {len(first_date):,} patients', flush=True)
    print(f'  {n:,} rows scanned, {len(first_date):,} patients with a reference date')

    with open(cache_path, 'wb') as f:
        pickle.dump(first_date, f)
    print(f'  -> {cache_path}')
    return first_date


# stage 2: per-patient match
def scan_diagnoses(lookup: dict, cohort_ids: set) -> tuple:
    """-> {patient_id: {date_str: set(cancer_type)}}, plus row stats."""
    per_patient: dict = defaultdict(lambda: defaultdict(set))
    stats = Counter()

    print(f'  matching {os.path.basename(EHRSHOT_CSV)} ...', flush=True)
    n = 0
    for chunk in pd.read_csv(EHRSHOT_CSV, dtype=str, chunksize=CHUNK_SIZE,
                             usecols=['patient_id', 'start', 'code', 'omop_table']):
        n += len(chunk)
        chunk['patient_id'] = pd.to_numeric(chunk['patient_id'], errors='coerce').astype('Int64')
        cond = chunk[(chunk['omop_table'] == 'condition_occurrence') &
                     chunk['patient_id'].isin(cohort_ids) &
                     chunk['code'].str.startswith('ICDO3/', na=False)]
        if not len(cond):
            continue
        ctypes = cond['code'].map(lookup)
        hit = cond[ctypes.notna()]
        if not len(hit):
            continue
        htypes = ctypes[ctypes.notna()]
        dates = pd.to_datetime(hit['start'], errors='coerce')
        for pid, date, ctype in zip(hit['patient_id'], dates, htypes):
            if pd.isna(date):
                stats['condition_hit_no_date'] += 1
                continue
            per_patient[pid][date.strftime('%Y-%m-%d')].add(ctype)
            stats['condition_hit_with_date'] += 1
        if n % (CHUNK_SIZE * 10) == 0:
            print(f'    {n/1e6:.0f}M rows, {len(per_patient):,} patients hit', flush=True)
    stats['condition_rows'] = n
    print(f'    done: {n:,} rows', flush=True)

    return per_patient, stats


def summarise(calendar: dict, out_dir: str, make_plot: bool):
    """Patient counts by cancer type at the EARLIEST diagnosis date."""
    counts = Counter()
    for pid, date_to_types in calendar.items():
        if not date_to_types:
            continue
        earliest = min(date_to_types)
        for ct in date_to_types[earliest]:      # double-count ties, as in CAIA
            counts[ct] += 1

    if not counts:
        print('No patients with a mapped cancer diagnosis — nothing to summarise.')
        return None

    df = pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]),
                      columns=['cancer_type', 'n_patients'])
    csv_path = os.path.join(out_dir, 'ehrshot_cancer_type_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nEarliest-diagnosis cancer type  -> {csv_path}')
    print(df.to_string(index=False))

    if make_plot:
        plot_df = df.set_index('cancer_type').iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(plot_df))))
        plot_df.plot(kind='barh', ax=ax, width=0.8, legend=False,
                    color='#4C72B0', edgecolor='none')
        ax.set_xlabel('Patients (cancer type at earliest diagnosis date)')
        ax.set_ylabel('')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            p = os.path.join(out_dir, f'ehrshot_cancer_type_distribution.{ext}')
            fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Plot -> {os.path.join(out_dir, "ehrshot_cancer_type_distribution.pdf")}')
    return df


def parse_args():
    ap = argparse.ArgumentParser(
        prog='check_cancer_type_ehrshot',
        description='Map Stanford EHRSHOT ICD-O-3 topography codes to the 30 CAIA cancer '
                    'types and emit a per-patient DMP_ID -> DATE -> CANCER_TYPES pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-od', '--output-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--rebuild-code-map', action='store_true')
    ap.add_argument('--code-map-only', action='store_true',
                    help='Build/inspect the ICDO3 -> cancer type dictionary and stop.')
    ap.add_argument('--include-in-situ', action='store_true',
                    help='Also count carcinoma-in-situ topography codes (D00-D09) as a '
                         'cancer diagnosis.')
    ap.add_argument('--include-secondary', action='store_true',
                    help=f"Also emit '{SECONDARY}' for metastasis topography codes "
                         '(C77-C79/C7B).')
    ap.add_argument('--drop-others', action='store_true',
                    help=f"Drop the '{OTHERS}' bucket instead of keeping malignancies "
                         'that are none of the 30 types.')
    ap.add_argument('--rebuild-reference-dates', action='store_true')
    ap.add_argument('--build-reference-dates', action='store_true',
                    help='Build/refresh the reference-date cache and stop.')
    ap.add_argument('--self-test', action='store_true',
                    help='Check the (imported) ICD10_RULES rule table and exit.')
    ap.add_argument('--no-plot', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        sys.exit(1 if _insight_self_test() else 0)
    os.makedirs(args.output_dir, exist_ok=True)
    print('=' * 78)
    print('Stanford EHRSHOT cancer-type check')
    print(f'  raw data   : {EHRSHOT_CSV}')
    print(f'  output     : {args.output_dir}')
    print('=' * 78)

    print('\n[1/4] Cohort (patients kept by dictionarify_new.py)')
    cohort_ids = load_cohort()

    print('\n[2/4] ICD-O-3 topography -> cancer type dictionary')
    code_map = build_code_map(args.output_dir, cohort_ids, args.rebuild_code_map)
    lookup = active_lookup(code_map, args.include_in_situ,
                           args.include_secondary, args.drop_others)
    by_type = Counter(lookup.values())
    print(f'  active codes this run: {len(lookup):,} over {len(by_type)} cancer types')
    missing = [t for t in CANCER_TYPES if t not in by_type]
    if missing:
        print(f'  types with no code in this dataset: {", ".join(missing)}')
    print(f'  NOTE: {" and ".join(UNREACHABLE_TYPES)} are not derivable from ICD-O-3 '
          'topography codes\n        (histology, not site) — see UNREACHABLE_TYPES in '
          'cohort_summaries/cancer_types/check_cancer_types_insight.py.')
    if args.code_map_only:
        return

    print('\n[3/4] Reference dates')
    ref_cache = os.path.join(args.output_dir, 'patient_id_to_reference_date_ehrshot.pickle')
    ref_dates = load_reference_dates(ref_cache, args.rebuild_reference_dates, cohort_ids)
    if args.build_reference_dates:
        return

    print('\n[4/4] Per-person match')
    per_patient, stats = scan_diagnoses(lookup, cohort_ids)
    print('  row statistics:', dict(stats))
    print(f'  {len(per_patient):,} of {len(cohort_ids):,} cohort patients '
          f'({100 * len(per_patient) / max(len(cohort_ids), 1):.1f}%) '
          'have >=1 mapped cancer code')

    calendar = {pid: {d: sorted(ts) for d, ts in date_to_types.items()}
                for pid, date_to_types in per_patient.items()}

    cal_path = os.path.join(
        args.output_dir, 'patient_id_to_CALENDARDATE_to_CANCER_TYPES_dict_ehrshot.pickle')
    with open(cal_path, 'wb') as f:
        pickle.dump(calendar, f)
    print(f'  calendar-date pickle: {len(calendar):,} patients  -> {os.path.basename(cal_path)}')

    delta = {}
    no_ref = 0
    for pid, date_to_types in calendar.items():
        r = ref_dates.get(pid)
        if r is None:
            no_ref += 1
            continue
        d = {}
        for date_str, ts in date_to_types.items():
            off = int((pd.Timestamp(date_str) - r).days)
            d.setdefault(off, set()).update(ts)
        delta[pid] = {k: sorted(v) for k, v in d.items()}

    delta_path = os.path.join(
        args.output_dir, 'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_stanfordEHRSHOT.pickle')
    with open(delta_path, 'wb') as f:
        pickle.dump(delta, f)
    line = f'  delta pickle: {len(delta):,} patients'
    if no_ref:
        line += f' ({no_ref} dropped: no reference date)'
    line += f'  -> {os.path.basename(delta_path)}'
    print(line)

    summarise(calendar, args.output_dir, not args.no_plot)
    print('\nDone.')


if __name__ == '__main__':
    main()
