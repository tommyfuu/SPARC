#!/usr/bin/env python3
"""
Count MSK_CAIA / MSK_CAIA_timeshift patients per first-mapped parent cancer type,
restricted to the modeling cohort; writes ctypes_<label>_summary.txt for ctypes_parse.py.
Same-day ties are shown as "Type A + Type B" in the combined-labels table.

Usage:
    python cancer_type_summary_msk_caia.py [--datasets MSK_CAIA] [--output-dir DIR]
"""

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
import argparse
import contextlib
import json
import pickle
from collections import Counter

# dataset -> (label used in ctypes_<label>_summary.txt, cancer-types pickle, final JSON)
DATASETS = {
    'MSK_CAIA': (
        'MSK',
        'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_till2021_050726.pickle',
        'DMP_ID_to_dicts_final_till2021_031226.json'),
    'MSK_CAIA_timeshift': (
        'MSK-timeshift',
        'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_after2021_050726.pickle',
        'DMP_ID_to_dicts_final_after2021_031226.json'),
}


class _Tee:
    """Write to several streams at once (stdout + the summary file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def load_final_cohort_ids(json_path):
    """person_ids in DMP_ID_to_DATES_to_TESTSVALUES_DICT of the final JSON."""
    print(f'  Loading {json_path} (large, takes a while)...', flush=True)
    with open(json_path) as f:
        final_dicts = json.load(f)
    return {int(pid) for pid in final_dicts['DMP_ID_to_DATES_to_TESTSVALUES_DICT']}


def summarize_first_cancer_types(cancer_types_dict):
    """Per patient, the parent type(s) at the earliest day_delta.

    Returns (single_type_counts, combined_label_counts, n_patients_total,
    n_patients_multi_first); see the module docstring for how ties count.
    """
    single_type_counts = Counter()
    combined_label_counts = Counter()
    n_patients_multi_first = 0
    for day_map in cancer_types_dict.values():
        if not day_map:
            continue
        first_types = list(dict.fromkeys(day_map[min(day_map)]))  # de-dup, keep order
        single_type_counts.update(first_types)
        if len(first_types) > 1:
            n_patients_multi_first += 1
        combined_label_counts[' + '.join(sorted(first_types))] += 1
    n_patients_total = sum(1 for d in cancer_types_dict.values() if d)
    return single_type_counts, combined_label_counts, n_patients_total, n_patients_multi_first


def print_summary(dataset_label, single_type_counts, combined_label_counts,
                  n_patients_total, n_patients_multi_first):
    sep = '=' * 78
    print(f'\n{sep}')
    print(f'CANCER TYPE SUMMARY (combined labels, one row per patient) — {dataset_label}')
    print(sep)
    print(f'  Total patients with >=1 mapped cancer type: {n_patients_total:,}')
    print(f'  Patients with >1 type tied for first appearance: {n_patients_multi_first:,}')
    print(sep)
    print(f'{"label":<55} {"#patients":>10} {"% of cohort":>12}')
    print('-' * 78)
    for label, n in combined_label_counts.most_common():
        pct = 100.0 * n / n_patients_total if n_patients_total else 0.0
        print(f'  {label:<53} {n:>10,} {pct:>11.2f}%')
    print(sep)

    print(f'\n{sep}')
    print(f'CANCER TYPE SUMMARY (first-mapped type) — {dataset_label}')
    print(sep)
    print('(Secondary/diagnostic view — a patient with a same-day multi-type '
          'tie is counted once under EACH tied type below, so these rows do '
          'NOT sum to the total patient count above. See the combined-label '
          'table above for the canonical one-row-per-patient summary.)')
    print(sep)
    print(f'\n{"parent_cancer_type":<45} {"#patients":>10} {"% of cohort":>12}')
    print('-' * 78)
    for cancer_type, n in single_type_counts.most_common():
        pct = 100.0 * n / n_patients_total if n_patients_total else 0.0
        print(f'  {cancer_type:<43} {n:>10,} {pct:>11.2f}%')
    print(sep)


def run(dataset, out_dir):
    label, pickle_name, json_name = DATASETS[dataset]
    data_dir = os.path.join(sparc_paths.DATA_ROOT, dataset)
    pickle_path = os.path.join(data_dir, pickle_name)
    json_path = os.path.join(data_dir, json_name)

    with open(pickle_path, 'rb') as f:
        cancer_types_dict = pickle.load(f)
    print(f'Loaded: {pickle_path}')
    print(f'  {len(cancer_types_dict):,} patients in cancer-types dict '
          f'(before cohort restriction)')

    print("\nRestricting to the pipeline's actual final modeling cohort...")
    cohort = load_final_cohort_ids(json_path)
    restricted = {pid: d for pid, d in cancer_types_dict.items() if int(pid) in cohort}
    print(f'  Loaded: {json_path}')
    print(f'  {len(cohort):,} patients in final cohort')
    print(f'  Dropped {len(cancer_types_dict) - len(restricted):,} patient(s) from the '
          'cancer-types dict not present in the final cohort')
    print(f'  {len(restricted):,} patients remaining after cohort restriction')
    print(f'  {len(cohort - {int(p) for p in restricted}):,} cohort patients '
          'with no mapped cancer type')

    print_summary(os.path.basename(pickle_path), *summarize_first_cancer_types(restricted))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument('--output-dir', default=sparc_paths.CANCER_TYPES_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        out_path = os.path.join(args.output_dir, f'ctypes_{DATASETS[dataset][0]}_summary.txt')
        with open(out_path, 'w') as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            run(dataset, args.output_dir)
        print(f'\n-> {out_path}\n')


if __name__ == '__main__':
    main()
