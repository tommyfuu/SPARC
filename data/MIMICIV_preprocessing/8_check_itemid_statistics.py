"""
Check mean and std for each itemid separately to identify problematic codes.
This helps identify which specific lab codes are causing outliers.
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
    print("Step 8: Checking Individual ItemID Statistics")
    print("="*80)

    print("\n1. Loading d_labitems.csv.gz...")
    d_labitems = pd.read_csv(f'{MIMIC_RAW_PATH}/d_labitems.csv')
    itemid_to_label = dict(zip(d_labitems['itemid'].astype(str), d_labitems['label']))
    print(f"   Loaded {len(itemid_to_label):,} lab item definitions")

    print("\n2. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    print("\n3. Loading anchor dates...")
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)

    print("\n4. Creating itemid to SPARC test mapping...")
    itemid_to_sparc = {}
    sparc_to_itemids = {}

    for sparc_test, itemids in MIMIC_TO_SPARC_LAB_MAPPING.items():
        sparc_to_itemids[sparc_test] = itemids
        for itemid in itemids:
            itemid_to_sparc[int(itemid)] = sparc_test

    print("\n5. Loading labevents.csv.gz...")
    print("   This will load the full file into memory...")

    relevant_itemids = set(itemid_to_sparc.keys())

    # Load in chunks and filter
    chunks = []
    chunk_size = 5_000_000
    total_rows = 0

    # # generate a simple read of
    # lab_df = pd.read_csv(f'{MIMIC_RAW_PATH}/labevents.csv.gz', compression='gzip')
    # labitems_df = pd.read_csv(f'{MIMIC_RAW_PATH}/d_labitems.csv.gz', compression='gzip')
    for chunk_idx, chunk in enumerate(pd.read_csv(
        f'{MIMIC_RAW_PATH}/labevents.csv.gz',
        compression='gzip',
        chunksize=chunk_size,
        # usecols=['subject_id', 'itemid', 'valuenum']
    )):
        total_rows += len(chunk)
        if chunk_idx % 2 == 0:
            print(f"   Chunk {chunk_idx + 1}: {total_rows:,} rows...")

        # Filter to cancer patients and relevant itemids
        mask = (chunk['subject_id'].isin(cancer_subjects)) & (chunk['itemid'].isin(relevant_itemids))
        chunk = chunk[mask].copy()

        if len(chunk) > 0:
            chunks.append(chunk)

    print(f"\n   Total rows processed: {total_rows:,}")

    if not chunks:
        print("   No matching lab data found!")
        return

    print("\n6. Combining chunks...")
    labevents_df = pd.concat(chunks, ignore_index=True)
    print(f"   Total cancer patient lab values: {len(labevents_df):,}")

    labevents_df['sparc_test'] = labevents_df['itemid'].map(itemid_to_sparc)

    print("\n7. Calculating statistics per itemid...")
    print("\n" + "="*100)
    print("PER-ITEMID STATISTICS")
    print("="*100)

    # Group by SPARC test name
    for sparc_test in sorted(sparc_to_itemids.keys()):
        itemids = sparc_to_itemids[sparc_test]

        if not itemids:
            continue

        test_data = labevents_df[labevents_df['sparc_test'] == sparc_test].copy()

        if len(test_data) == 0:
            continue

        print(f"\n{sparc_test}")
        print("-" * 100)
        print(f"{'ItemID':<10} {'Label':<40} {'Count':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 100)

        # Calculate stats for each itemid
        for itemid in itemids:
            itemid_int = int(itemid)
            itemid_data = test_data[test_data['itemid'] == itemid_int]['valuenum'].dropna()

            if len(itemid_data) == 0:
                label = itemid_to_label.get(itemid, 'Unknown')
                print(f"{itemid:<10} {label:<40} {'No data':>10}")
                continue

            values = itemid_data.values
            count = len(values)
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            label = itemid_to_label.get(itemid, 'Unknown')

            # Highlight problematic ones (very high max or std > 100% of mean)
            warning = ""
            if max_val > 10 * mean:
                warning = " WARNING: EXTREME MAX"
            elif std > mean:
                warning = " WARNING: HIGH STD"

            print(f"{itemid:<10} {label:<40} {count:>10,} {mean:>12.2f} {std:>12.2f} {min_val:>12.2f} {max_val:>12.2f}{warning}")

        # Also show combined stats
        all_values = test_data['valuenum'].dropna().values
        if len(all_values) > 0:
            print("-" * 100)
            print(f"{'COMBINED':<10} {'All itemids for ' + sparc_test:<40} "
                  f"{len(all_values):>10,} {np.mean(all_values):>12.2f} {np.std(all_values):>12.2f} "
                  f"{np.min(all_values):>12.2f} {np.max(all_values):>12.2f}")

    # Summary of problematic itemids
    print("\n" + "="*100)
    print("SUMMARY: Problematic ItemIDs (Extreme Outliers)")
    print("="*100)

    problematic = []

    for sparc_test in sparc_to_itemids.keys():
        itemids = sparc_to_itemids[sparc_test]

        if not itemids:
            continue

        test_data = labevents_df[labevents_df['sparc_test'] == sparc_test].copy()

        for itemid in itemids:
            itemid_int = int(itemid)
            itemid_data = test_data[test_data['itemid'] == itemid_int]['valuenum'].dropna()

            if len(itemid_data) == 0:
                continue

            values = itemid_data.values
            mean = np.mean(values)
            std = np.std(values)
            max_val = np.max(values)

            # Flag if max > 10x mean OR std > mean
            if max_val > 10 * mean or std > mean:
                label = itemid_to_label.get(itemid, 'Unknown')
                problematic.append({
                    'sparc_test': sparc_test,
                    'itemid': itemid,
                    'label': label,
                    'count': len(values),
                    'mean': mean,
                    'std': std,
                    'max': max_val,
                    'max_to_mean_ratio': max_val / mean if mean > 0 else 0,
                    'std_to_mean_ratio': std / mean if mean > 0 else 0
                })

    if problematic:
        problematic_df = pd.DataFrame(problematic)
        problematic_df = problematic_df.sort_values('max_to_mean_ratio', ascending=False)

        print("\nTop 20 most problematic itemids (by max/mean ratio):")
        print("-" * 100)
        print(f"{'SPARC Test':<25} {'ItemID':<10} {'Label':<30} {'Max/Mean':>10} {'Std/Mean':>10}")
        print("-" * 100)

        for idx, row in problematic_df.head(20).iterrows():
            print(f"{row['sparc_test']:<25} {row['itemid']:<10} {row['label']:<30} "
                  f"{row['max_to_mean_ratio']:>10.1f} {row['std_to_mean_ratio']:>10.1f}")

        output_file = f'{OUTPUT_PATH}/MIMICIV_problematic_itemids.csv'
        problematic_df.to_csv(output_file, index=False)
        print(f"\nSaved full problematic itemid list to: {output_file}")
    else:
        print("\nNo problematic itemids found!")

    print("\n" + "="*100)
    print("Step 8 Complete!")
    print("="*100)

if __name__ == '__main__':
    main()
