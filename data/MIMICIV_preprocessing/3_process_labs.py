"""
Process lab values for ALL_INCLUDED_TESTS (46 tests).
Creates date-indexed lab value dictionaries.
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from config import (
    MIMIC_RAW_PATH, OUTPUT_PATH,
    ALL_INCLUDED_TESTS, MIMIC_TO_SPARC_LAB_MAPPING
)

def main():
    print("="*80)
    print("Step 3: Processing Lab Values (ALL_INCLUDED_TESTS)")
    print("="*80)

    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    print("\n2. Loading anchor dates...")
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)

    print("\n3. Loading d_labitems.csv.gz...")
    d_labitems = pd.read_csv(f'{MIMIC_RAW_PATH}/d_labitems.csv.gz', compression='gzip')
    itemid_to_label = dict(zip(d_labitems['itemid'].astype(str), d_labitems['label']))

    # Create reverse mapping: itemid -> SPARC test name
    print("\n4. Creating itemid to SPARC test name mapping...")
    itemid_to_sparc_test = {}
    for sparc_test, itemids in MIMIC_TO_SPARC_LAB_MAPPING.items():
        for itemid in itemids:
            itemid_to_sparc_test[itemid] = sparc_test

    print(f"   Mapped {len(itemid_to_sparc_test)} MIMIC itemids to SPARC tests")

    # labevents is large, so read in chunks
    print("\n5. Loading and processing labevents.csv.gz...")
    print("   WARNING: This file is very large (~2.6GB compressed). Processing in chunks...")

    # Initialize output structure: {subject_id: {date_delta: [test1, test2, ..., test46]}}
    subject_to_date_to_labs = defaultdict(lambda: defaultdict(lambda: ['NA'] * len(ALL_INCLUDED_TESTS)))

    chunk_size = 1_000_000
    total_rows = 0
    matched_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        f'{MIMIC_RAW_PATH}/labevents.csv.gz',
        compression='gzip',
        chunksize=chunk_size,
        usecols=['subject_id', 'itemid', 'charttime', 'valuenum']
    )):
        total_rows += len(chunk)
        print(f"   Processing chunk {chunk_idx + 1}: {total_rows:,} rows so far...")

        chunk = chunk[chunk['subject_id'].isin(cancer_subjects)]

        chunk['itemid_str'] = chunk['itemid'].astype(str)

        # Filter to relevant lab tests
        chunk = chunk[chunk['itemid_str'].isin(itemid_to_sparc_test.keys())]
        matched_rows += len(chunk)

        if len(chunk) == 0:
            continue

        for _, row in chunk.iterrows():
            subject_id = row['subject_id']
            itemid_str = row['itemid_str']
            charttime = pd.to_datetime(row['charttime'])
            valuenum = row['valuenum']

            if subject_id not in anchor_dates:
                continue

            # Calculate date delta (days from anchor)
            anchor = anchor_dates[subject_id]
            date_delta = (charttime - anchor).days

            sparc_test = itemid_to_sparc_test[itemid_str]

            test_idx = ALL_INCLUDED_TESTS.index(sparc_test)

            # keep the first value per day
            if subject_to_date_to_labs[subject_id][date_delta][test_idx] == 'NA':
                subject_to_date_to_labs[subject_id][date_delta][test_idx] = valuenum

    print(f"\n   Total lab rows processed: {total_rows:,}")
    print(f"   Matched cancer patient lab rows: {matched_rows:,}")
    print(f"   Unique patients with labs: {len(subject_to_date_to_labs):,}")

    print("\n6. Converting to regular dict format...")
    output_dict = {}
    for subject_id, date_dict in tqdm(subject_to_date_to_labs.items(), desc="   Converting"):
        output_dict[subject_id] = {date: labs for date, labs in date_dict.items()}

    print("\n7. Lab data statistics:")
    total_encounters = sum(len(dates) for dates in output_dict.values())
    print(f"   Total patient-date encounters: {total_encounters:,}")

    # Count how many tests have data
    test_data_counts = [0] * len(ALL_INCLUDED_TESTS)
    for subject_id, date_dict in output_dict.items():
        for date, labs in date_dict.items():
            for i, val in enumerate(labs):
                if val != 'NA' and not pd.isna(val):
                    test_data_counts[i] += 1

    print(f"\n   Data availability per test:")
    for i, test_name in enumerate(ALL_INCLUDED_TESTS):
        if test_data_counts[i] > 0:
            print(f"     {test_name}: {test_data_counts[i]:,} measurements")

    print("\n8. Saving output...")
    output_path = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_all_included_lab_values_list_dict.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"   Saved: MIMICIV_subjectid_to_date_to_all_included_lab_values_list_dict.pkl")
    print(f"   File contains {len(output_dict):,} patients")

    print("\n" + "="*80)
    print("Step 3 Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
