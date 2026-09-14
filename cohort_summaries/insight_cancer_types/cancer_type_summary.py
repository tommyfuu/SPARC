"""
Standalone summary tool: counts patients per first-mapped parent cancer type.

Reads every DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_<dataset>.pickle it can
find, shape:
    {person_id: {day_delta: [parent_cancer_type, ...]}}

and reports, for each patient, the parent cancer type(s) recorded at their
EARLIEST day_delta (their first mapped cancer type). If more than one parent
type first appears on that same earliest day, the patient is counted once
under each of those types (co-first-appearance), and separately under a
combined "type_a + type_b" label so ties aren't silently double counted
without a way to see them.

Output
------
Everything is printed to stdout (so it shows up in a terminal / job log) AND,
unless WRITE_TXT is turned off, written to one plain-text report per input
pickle:

    DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_insight_A.pickle
        -> ctypes_INSIGHT-A_summary.txt

The file lands next to its pickle unless OUTPUT_DIR is set.

Usage — just run it, no arguments/CLI flags at all:
    python cancer_type_summary.py

This scans a set of candidate root directories (cwd, /output, /input,
/external_data, or $SPARC_OUTPUT_DIR if set) for files matching
FILENAME_PATTERN and processes ALL of them, so a single run covers every
institution A-F. Set PICKLE_PATHS to target specific files instead.

Deliberately takes no CLI arguments (no argparse) — this is meant to be
dropped as-is into execution harnesses (e.g. a Rhino Code Object) that may
invoke it with their own extra argv; argparse would reject unrecognized
flags and abort before printing anything.

main() runs unconditionally at import time (no `if __name__ == '__main__'`
guard) so it also executes when a harness imports/execs this file as a
module rather than running it as a script.
"""
import fnmatch
import os
import pickle
import re
from collections import Counter

# Filename pattern the auto-scan looks for.
FILENAME_PATTERN = 'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_*.pickle'

# Everything before the dataset tag in that filename.
FILENAME_PREFIX = 'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_'

# Root directories the auto-scan walks, in priority order. Only existing
# directories are actually scanned. Covers both a typical local run
# (an output dir under cwd) and a locked-down/mounted environment (Rhino-style
# /input, /output, /external_data mounts).
SCAN_ROOTS = [
    os.environ.get('SPARC_OUTPUT_DIR'),
    os.getcwd(),
    '/output',
    '/input',
    '/external_data',
    '/external_data/prod-workgroup-data/code-artifacts',
]

# Set to an explicit list of paths to skip auto-scanning entirely, e.g.
#   PICKLE_PATHS = ['/path/to/..._insight_A.pickle']
PICKLE_PATHS = None

# Write a ctypes_<TAG>_summary.txt report per pickle in addition to printing.
WRITE_TXT = True

# Where the .txt reports go. None -> alongside each source pickle.
OUTPUT_DIR = None

# Name of each report. {tag} is derived from the pickle filename
# ('insight_A' -> 'INSIGHT-A').
TXT_NAME_TEMPLATE = 'ctypes_{tag}_summary.txt'

# Set to True to also list every patient id grouped by cancer type.
LIST_PATIENTS = False

# Set to False to skip the "what is actually mounted" dump. Off by default
# here because os.walk over a large repo checkout is slow and drowns the
# report; turn it on when debugging a harness that mounts unknown inputs.
PRINT_MOUNTED_INPUTS = False


def find_cancer_types_pickles(scan_roots=None, pattern=FILENAME_PATTERN):
    """
    Walk scan_roots looking for files matching `pattern`. Returns ALL matches,
    de-duplicated by real path (the roots overlap in practice) and sorted by
    filename so institutions come out in A-F order.
    """
    found = {}
    for root in (scan_roots or SCAN_ROOTS):
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    path = os.path.join(dirpath, fname)
                    found.setdefault(os.path.realpath(path), path)
    return [found[k] for k in sorted(found, key=lambda p: os.path.basename(p))]


def dataset_tag(pickle_path):
    """
    'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_insight_A.pickle' -> 'INSIGHT-A'

    Anything that isn't an insight_<site> file keeps its own stem, so e.g.
    the MSK_CAIA pickles produce ctypes_MSK_CAIA_till2021_050726_summary.txt.
    """
    stem = os.path.basename(pickle_path)
    if stem.startswith(FILENAME_PREFIX):
        stem = stem[len(FILENAME_PREFIX):]
    stem = re.sub(r'\.pickle$', '', stem)
    m = re.match(r'^insight[_-](.+)$', stem, flags=re.IGNORECASE)
    if m:
        return f'INSIGHT-{m.group(1).upper()}'
    return stem


def print_mounted_inputs(emit, roots=None):
    """
    Print every file found under each candidate root (cwd, /output, /input,
    /external_data, $SPARC_OUTPUT_DIR) so a job log always shows exactly
    what's actually mounted/available, even if the pickle search below finds
    nothing or the job errors out afterward.
    """
    sep = '=' * 78
    emit(f'\n{sep}')
    emit('MOUNTED INPUTS (all files under each candidate root)')
    emit(sep)
    for root in (roots or SCAN_ROOTS):
        if not root:
            continue
        if not os.path.isdir(root):
            emit(f'\n{root}  (not present)')
            continue
        n_files = 0
        emit(f'\n{root}/')
        for dirpath, _dirs, files in os.walk(root):
            for fname in sorted(files):
                emit(f'  {os.path.join(dirpath, fname)}')
                n_files += 1
        if n_files == 0:
            emit('  (empty)')
    emit(f'\n{sep}')


def load_cancer_types_dict(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f'Cannot find: {pickle_path}\n'
            'This file is produced by cancer_type_check_insight.py.')
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def summarize_first_cancer_types(cancer_types_dict):
    """
    For each patient, find the parent cancer type(s) at their earliest
    day_delta and tally patient counts per type.

    Returns:
        single_type_counts: Counter {parent_cancer_type: n_patients} —
            counts a patient once under EACH type present on their earliest
            day (so a patient with a same-day tie is counted under both).
        combined_label_counts: Counter {"Type A" or "Type A + Type B": n} —
            counts each patient exactly once, using a combined label when
            multiple types tied for first appearance.
        n_patients_total: total patients in the dict.
        n_patients_multi_first: patients whose first day had >1 parent type.
        patients_by_type: dict {parent_cancer_type: [pid, ...]} — same
            multi-counting rule as single_type_counts.
        patients_by_combined_label: dict {combined_label: [pid, ...]} —
            each patient appears in exactly one list.
    """
    single_type_counts = Counter()
    combined_label_counts = Counter()
    patients_by_type = {}            # parent_cancer_type -> [pid, ...]
    patients_by_combined_label = {}  # combined label -> [pid, ...]
    n_patients_multi_first = 0

    for pid, day_map in cancer_types_dict.items():
        if not day_map:
            continue
        earliest_day = min(day_map.keys())
        first_types = day_map[earliest_day]

        # de-dup while preserving order, in case of any upstream duplicates
        seen = set()
        ordered_types = []
        for t in first_types:
            if t not in seen:
                ordered_types.append(t)
                seen.add(t)
        if not ordered_types:
            continue

        for t in ordered_types:
            single_type_counts[t] += 1
            patients_by_type.setdefault(t, []).append(pid)

        if len(ordered_types) > 1:
            n_patients_multi_first += 1
            label = ' + '.join(sorted(ordered_types))
        else:
            label = ordered_types[0]
        combined_label_counts[label] += 1
        patients_by_combined_label.setdefault(label, []).append(pid)

    n_patients_total = sum(1 for _, d in cancer_types_dict.items() if d)
    return (single_type_counts, combined_label_counts, n_patients_total,
            n_patients_multi_first, patients_by_type, patients_by_combined_label)


def print_patients_by_type(emit, patients_by_combined_label):
    """Print every patient id grouped under its first-mapped cancer type
    (combined label — each patient listed exactly once)."""
    sep = '=' * 78
    emit(f'\n{sep}')
    emit('PATIENTS BY CANCER TYPE (first-mapped, one row per patient)')
    emit(sep)
    for label in sorted(patients_by_combined_label,
                        key=lambda k: -len(patients_by_combined_label[k])):
        pids = sorted(patients_by_combined_label[label])
        emit(f'\n{label}  ({len(pids):,} patients)')
        emit('-' * 78)
        emit(', '.join(str(p) for p in pids))
    emit(f'\n{sep}')


def print_summary(emit, dataset_label, source_path, n_in_dict,
                  single_type_counts, combined_label_counts,
                  n_patients_total, n_patients_multi_first):
    sep = '=' * 78
    emit(sep)
    emit(f'CANCER TYPE SUMMARY (first-mapped type) — {dataset_label}')
    emit(sep)
    emit(f'  Source: {source_path}')
    emit(f'  Patients in dict: {n_in_dict:,}')
    emit(f'  Total patients with >=1 mapped cancer type: {n_patients_total:,}')
    emit(f'  Patients with >1 type tied for first appearance: '
         f'{n_patients_multi_first:,}')
    emit(sep)

    emit(f'\n{"parent_cancer_type":<45} {"#patients":>10} {"% of cohort":>12}')
    emit('-' * 78)
    for cancer_type, n in single_type_counts.most_common():
        pct = 100.0 * n / n_patients_total if n_patients_total else 0.0
        emit(f'  {cancer_type:<43} {n:>10,} {pct:>11.2f}%')
    emit('-' * 78)
    emit(f'  {"TOTAL (sums >100% when ties exist)":<43} '
         f'{sum(single_type_counts.values()):>10,}')
    emit(sep)
    emit('(Note: rows sum to more than total patients if any patient had '
         'multiple cancer types tie for first appearance — see combined '
         'labels below for a partition that counts each patient once.)')

    emit(f'\n{sep}')
    emit('CANCER TYPE SUMMARY (combined labels, one row per patient)')
    emit(sep)
    emit(f'{"label":<55} {"#patients":>10} {"% of cohort":>12}')
    emit('-' * 78)
    for label, n in combined_label_counts.most_common():
        pct = 100.0 * n / n_patients_total if n_patients_total else 0.0
        emit(f'  {label:<53} {n:>10,} {pct:>11.2f}%')
    emit('-' * 78)
    emit(f'  {"TOTAL":<53} {sum(combined_label_counts.values()):>10,}')
    emit(sep)


def process_one(pickle_path):
    """Summarise a single pickle: print to stdout and, if enabled, write the
    matching ctypes_<TAG>_summary.txt. Returns the report path, or None."""
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    cancer_types_dict = load_cancer_types_dict(pickle_path)
    tag = dataset_tag(pickle_path)

    (single_type_counts, combined_label_counts, n_patients_total,
     n_patients_multi_first, _patients_by_type,
     patients_by_combined_label) = summarize_first_cancer_types(cancer_types_dict)

    print_summary(emit, tag, pickle_path, len(cancer_types_dict),
                  single_type_counts, combined_label_counts,
                  n_patients_total, n_patients_multi_first)

    if LIST_PATIENTS:
        print_patients_by_type(emit, patients_by_combined_label)

    if not WRITE_TXT:
        return None

    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(pickle_path))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, TXT_NAME_TEMPLATE.format(tag=tag))
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  -> wrote {out_path}')
    return out_path


def main():
    if PRINT_MOUNTED_INPUTS:
        print_mounted_inputs(print)

    paths = PICKLE_PATHS
    if not paths:
        print(f'No PICKLE_PATHS set — scanning for "{FILENAME_PATTERN}" '
              f'under: {[r for r in SCAN_ROOTS if r]}')
        paths = find_cancer_types_pickles()
        if not paths:
            raise FileNotFoundError(
                f'Could not find any file matching "{FILENAME_PATTERN}" '
                f'under {[r for r in SCAN_ROOTS if r]}.\n'
                'These files are produced by cancer_type_check_insight.py. '
                'Set PICKLE_PATHS at the top of this script to point at them '
                'directly.')
    print(f'Found {len(paths)} pickle(s):')
    for p in paths:
        print(f'  {p}')

    written = []
    for path in paths:
        out = process_one(path)
        if out:
            written.append(out)

    if written:
        print(f'\n{"=" * 78}')
        print(f'Wrote {len(written)} report(s):')
        for w in written:
            print(f'  {w}')
        print('=' * 78)


main()
