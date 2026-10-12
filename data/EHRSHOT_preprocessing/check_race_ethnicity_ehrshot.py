#!/usr/bin/env python3
"""Race and ethnicity counts for the 546-patient EHRSHOT cohort.

Writes EHRSHOT_race.csv and EHRSHOT_ethnicity.csv to
$SPARC_RACE_ETHNICITY_DIR/EHRSHOT/ in the MSK_CAIA format (counts <= 10 shown as
'n<=10'), for dem_table_3_harmonize_race_ethnicity.py.

Race/1..5 are OMOP 'Race' vocabulary concept codes (see RACE_LABELS). Patients
with no Race/ or Ethnicity/ row count as 'No matching concept'.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import pickle

from config import EHRSHOT_CSV, COHORT_PICKLE, RACE_ETHNICITY_OUTPUT_PATH

DEFAULT_OUT_DIR = RACE_ETHNICITY_OUTPUT_PATH

CHUNK_SIZE = 1_000_000
SUPPRESS_THRESHOLD = 10

ETHNICITY_LABELS = {
    'Ethnicity/Not Hispanic': 'Not Hispanic or Latino',
    'Ethnicity/Hispanic': 'Hispanic or Latino',
}

# OMOP 'Race' vocabulary (concept.csv, vocabulary_id == 'Race')
RACE_LABELS = {
    'Race/1': 'American Indian or Alaska Native',
    'Race/2': 'Asian',
    'Race/3': 'Black or African American',
    'Race/4': 'Native Hawaiian or Other Pacific Islander',
    'Race/5': 'White',
}


def load_cohort() -> set:
    with open(COHORT_PICKLE, 'rb') as f:
        cohort = set(pickle.load(f).keys())
    print(f'  cohort: {len(cohort):,} included patients  ({os.path.basename(COHORT_PICKLE)})')
    return cohort


def scan_person_rows(cohort_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Race/ and Ethnicity/ person rows for cohort patients."""
    race_chunks, eth_chunks = [], []
    n = 0
    for chunk in pd.read_csv(EHRSHOT_CSV, dtype=str, chunksize=CHUNK_SIZE,
                             usecols=['patient_id', 'code', 'omop_table']):
        n += len(chunk)
        chunk['patient_id'] = pd.to_numeric(chunk['patient_id'], errors='coerce').astype('Int64')
        p = chunk[(chunk['omop_table'] == 'person') & chunk['patient_id'].isin(cohort_ids)]
        race_chunks.append(p[p['code'].str.startswith('Race/', na=False)][['patient_id', 'code']])
        eth_chunks.append(p[p['code'].str.startswith('Ethnicity/', na=False)][['patient_id', 'code']])
        if n % (CHUNK_SIZE * 10) == 0:
            print(f'    {n/1e6:.0f}M rows scanned', flush=True)
    print(f'    done: {n:,} rows')
    return pd.concat(race_chunks), pd.concat(eth_chunks)


def counts_with_missing(codes: pd.Series, cohort_size: int, rename: dict | None = None) -> pd.Series:
    counts = codes.value_counts()
    if rename:
        counts = counts.rename(rename)
    missing = cohort_size - counts.sum()
    if missing > 0:
        counts['No matching concept'] = missing
    return counts.sort_values(ascending=False)


def suppress(counts: pd.Series, threshold: int) -> list[tuple[str, object]]:
    return [(label, n if n > threshold else f'n<={threshold}') for label, n in counts.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-od', '--output-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--suppress-threshold', type=int, default=SUPPRESS_THRESHOLD,
                    help=f'Counts <= this are shown as "n<=N" (default {SUPPRESS_THRESHOLD}).')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('[1/2] Cohort')
    cohort_ids = load_cohort()

    print('\n[2/2] Race / ethnicity rows')
    race_df, eth_df = scan_person_rows(cohort_ids)
    print(f'  race rows: {len(race_df):,} ({race_df["patient_id"].nunique():,} patients)')
    print(f'  ethnicity rows: {len(eth_df):,} ({eth_df["patient_id"].nunique():,} patients)')

    race_counts = counts_with_missing(race_df['code'], len(cohort_ids), rename=RACE_LABELS)
    eth_counts = counts_with_missing(eth_df['code'], len(cohort_ids), rename=ETHNICITY_LABELS)

    race_out = pd.DataFrame(suppress(race_counts, args.suppress_threshold),
                            columns=['race_concept_name', 'count'])
    eth_out = pd.DataFrame(suppress(eth_counts, args.suppress_threshold),
                           columns=['ethnicity_concept_name', 'count'])

    race_path = os.path.join(args.output_dir, 'EHRSHOT_race.csv')
    eth_path = os.path.join(args.output_dir, 'EHRSHOT_ethnicity.csv')
    race_out.to_csv(race_path, index=False)
    eth_out.to_csv(eth_path, index=False)

    print(f'\n-> {race_path}')
    print(race_out.to_string(index=False))
    print(f'\n-> {eth_path}')
    print(eth_out.to_string(index=False))


if __name__ == '__main__':
    main()
