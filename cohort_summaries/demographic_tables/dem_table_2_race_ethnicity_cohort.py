"""
Generate cohort-restricted race/ethnicity summary CSVs for a SPARC dataset.

Why this script exists
----------------------
The site-level race/ethnicity export covers every patient in the raw MSK
extract. The SPARC modelling cohort is smaller: `caia_portion='prior'`
truncates each patient's labs to dates <= their cutoff day and then drops
patients whose lab record is empty afterwards, and `_load_caia()` finally
restricts every dict to `shared['labs'].keys()`. For MSK-CAIA that is:

    Removed 4708 patients with no labs data after prior filtering
    (228755 -> 224047)

So aggregating the raw export produces 228,755 people while the cohort has
224,047, and the demographics table then mixes two populations in one row.

This script aggregates from a *patient-level* race/ethnicity map restricted to
the cohort that `load_dataset()` actually returns, so the totals match
`n_patients` in the dataset summary by construction — for any caia_portion, and
for any future change to the cohort definition.

Output matches the schema the harmonizer consumes
(`11_create_harmonized_race_ethnicity.py`):
    MSK_CAIA_race.csv        race_concept_name,count
    MSK_CAIA_ethnicity.csv   ethnicity_concept_name,count

Small cells are suppressed as 'n<=10', consistent with the existing files.

Usage
-----
    # MSK-CAIA (defaults)
    python generate_race_ethnicity_cohort.py

    # MSK-CAIA timeshift
    python generate_race_ethnicity_cohort.py -ds MSK_CAIA_timeshift --stem MSK_CAIA_timeshift

    # MIMIC-IV (patient-level pickle written by
    # data/MIMICIV_preprocessing/generate_race_ethnicity_summaries.py)
    python generate_race_ethnicity_cohort.py -ds MIMICIV_hosp --stem MIMICIV_hosp \
        -p .../SPARC_data/MIMICIV_hosp/subject_id_to_race_ethnicity_MIMICIV_hosp.pickle
"""

import argparse
import os
import pickle
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Path setup — resolve SPARC/data so dataset_loading.py is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # …/cohort_summaries/demographic_tables
# two levels up now that this lives in a subfolder: demographic_tables -> cohort_summaries -> SPARC
_SPARC_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))       # …/SPARC
_DATA_DIR = os.path.join(_SPARC_ROOT, 'data')                     # …/SPARC/data

if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from dataset_loading import load_dataset  # noqa: E402

DEFAULT_VOLUME_PATH = '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data/'
DEFAULT_RACE_PICKLE = (
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data/MSK_CAIA/'
    'DMP_ID_to_race_ethnicity_MSK_CAIA_test_all_michele.pickle'
)
DEFAULT_OUT_DIR = (
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models/results_vis'
    '/cohort_summaries/race_ethnicity/MSK_CAIA'
)

SUPPRESS_THRESHOLD = 10
SUPPRESS_LABEL = f'n<={SUPPRESS_THRESHOLD}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate cohort-restricted race/ethnicity CSVs for a SPARC dataset.'
    )
    parser.add_argument('-ds', '--dataset', default='MSK_CAIA',
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('-cp', '--caia_portion', default='prior',
                        choices=['prior', 'post', 'entire'],
                        help='CAIA temporal portion (default: %(default)s). Ignored for '
                             'non-CAIA datasets. For MSK_CAIA this is what '
                             'removes the 4,708 patients.')
    parser.add_argument('-m', '--modules', default='parsimonious',
                        help='Module category (default: %(default)s)')
    parser.add_argument('-vp', '--volume_path', default=DEFAULT_VOLUME_PATH,
                        help='Base path to datasets (default: %(default)s)')
    parser.add_argument('-p', '--race_pickle', default=DEFAULT_RACE_PICKLE,
                        help='Patient-level {id: (race, ethnicity)} pickle '
                             '(default: %(default)s)')
    parser.add_argument('-o', '--output_dir', default=DEFAULT_OUT_DIR,
                        help='Where to write the CSVs (default: %(default)s)')
    parser.add_argument('--stem', default=None,
                        help='Output filename stem (default: the dataset name)')
    parser.add_argument('--no-suppress', action='store_true',
                        help='Write raw counts instead of suppressing cells '
                             f'with <= {SUPPRESS_THRESHOLD} patients')
    return parser.parse_args()


def get_cohort_ids(args):
    """Return the patient IDs load_dataset() actually yields for this cohort.

    Mirrors dataset_summaries_fixed.py exactly so that the totals here and
    n_patients there cannot drift apart.
    """
    print(f"Loading dataset '{args.dataset}' "
          f"(modules={args.modules}, caia_portion={args.caia_portion})…")
    _, _, split_dict, _, _ = load_dataset(
        dataset=args.dataset,
        data_source_path=args.volume_path,
        option='binary',
        modules=args.modules,
        valfold=0,
        mode='train',
        normalize=False,
        caia_portion=args.caia_portion,
    )
    cohort = set(
        list(split_dict.get('train', []))
        + list(split_dict.get('val', []))
        + list(split_dict.get('test', []))
    )
    print(f"Cohort size after merging splits: {len(cohort):,}")
    return cohort


def load_patient_level_map(path):
    """Load {patient_id: (race, ethnicity)}."""
    with open(path, 'rb') as f:
        mapping = pickle.load(f)
    print(f"Patient-level race/ethnicity map: {len(mapping):,} patients")
    return mapping


def tally(mapping, cohort, index):
    """Count one field (0=race, 1=ethnicity) over the cohort.

    Returns (counter, n_matched, n_missing).
    """
    counts = Counter()
    missing = 0
    for pid in cohort:
        value = mapping.get(pid)
        if value is None:
            missing += 1
            continue
        counts[value[index]] += 1
    return counts, sum(counts.values()), missing


def write_csv(counts, path, column, suppress=True):
    """Write a {category,count} CSV sorted by count descending."""
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    with open(path, 'w') as f:
        f.write(f'{column},count\n')
        for category, n in rows:
            cell = SUPPRESS_LABEL if (suppress and n <= SUPPRESS_THRESHOLD) else str(n)
            f.write(f'{category},{cell}\n')
    return rows


def main():
    args = parse_args()
    stem = args.stem or args.dataset
    os.makedirs(args.output_dir, exist_ok=True)

    cohort = get_cohort_ids(args)
    mapping = load_patient_level_map(args.race_pickle)

    # Patient IDs may be stored as int in one source and str in the other.
    sample_cohort = next(iter(cohort))
    sample_map = next(iter(mapping))
    if type(sample_cohort) is not type(sample_map):
        print(f"Note: reconciling id types "
              f"({type(sample_cohort).__name__} vs {type(sample_map).__name__})")
        mapping = {type(sample_cohort)(k): v for k, v in mapping.items()}

    race_counts, n_race, missing = tally(mapping, cohort, 0)
    eth_counts, n_eth, _ = tally(mapping, cohort, 1)

    race_path = os.path.join(args.output_dir, f'{stem}_race.csv')
    eth_path = os.path.join(args.output_dir, f'{stem}_ethnicity.csv')
    write_csv(race_counts, race_path, 'race_concept_name', not args.no_suppress)
    write_csv(eth_counts, eth_path, 'ethnicity_concept_name', not args.no_suppress)

    print(f"\nCohort patients            : {len(cohort):,}")
    print(f"With race/ethnicity record : {n_race:,}")
    if missing:
        print(f"WARNING: {missing:,} cohort patients absent from the "
              f"patient-level map — they are omitted from the counts, so the "
              f"totals below fall short of the cohort size by that amount.")
    print(f"Race total                 : {n_race:,}")
    print(f"Ethnicity total            : {n_eth:,}")
    print(f"\nWrote {race_path}")
    print(f"Wrote {eth_path}")
    print("\nNext: re-run 11_create_harmonized_race_ethnicity.py, then "
          "demographics_table.py.")


if __name__ == '__main__':
    main()
