"""
Process medication data from emar.
Extracts ALL medications with dosages using vectorized operations.
"""

import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from config import MIMIC_RAW_PATH, OUTPUT_PATH, normalize_drug_name
import re

def extract_dosage(dose_str):
    """Extract first numeric value from dosage string."""
    if pd.isna(dose_str) or not isinstance(dose_str, str):
        return None
    match = re.search(r'(\d+\.?\d*)', dose_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def main():
    print("="*80)
    print("Step 4: Processing Medications")
    print("="*80)

    # Check if output already exists
    output_file = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_all_medications_wdosage_dict.pkl'

    if os.path.exists(output_file):
        print(f"\n⚠️  Output file already exists:")
        print(f"   - {os.path.basename(output_file)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 4.")
            return
        print("Re-running step 4...")

    # Load cancer patients
    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    # Load anchor dates
    print("\n2. Loading anchor dates...")
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)

    # Step 1: Load emar with emar_id
    print("\n3. Loading emar.csv.gz (medication administrations)...")
    emar_chunks = []
    chunk_size = 2_000_000
    total_rows = 0
    matched_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        f'{MIMIC_RAW_PATH}/emar.csv.gz',
        compression='gzip',
        chunksize=chunk_size,
        usecols=['subject_id', 'emar_id', 'charttime', 'medication', 'event_txt']
    )):
        total_rows += len(chunk)
        if chunk_idx % 2 == 0:
            print(f"   Chunk {chunk_idx + 1}: {total_rows:,} rows...")

        # Filter: cancer patients + administered meds
        mask = (chunk['subject_id'].isin(cancer_subjects)) & \
               (chunk['event_txt'].str.lower() == 'administered')
        chunk = chunk[mask].copy()
        matched_rows += len(chunk)

        if len(chunk) > 0:
            emar_chunks.append(chunk)

    print(f"   Total rows: {total_rows:,}, Matched: {matched_rows:,}")

    if not emar_chunks:
        print("\n⚠️  No medication data found!")
        with open(output_file, 'wb') as f:
            pickle.dump({}, f)
        return

    # Combine and process dates
    print("\n4. Processing dates (vectorized)...")
    emar_df = pd.concat(emar_chunks, ignore_index=True)
    print(f"   Total administrations: {len(emar_df):,}")

    emar_df['charttime'] = pd.to_datetime(emar_df['charttime'])
    emar_df['anchor'] = emar_df['subject_id'].map(anchor_dates)
    emar_df = emar_df.dropna(subset=['anchor'])
    emar_df['date_delta'] = (emar_df['charttime'] - emar_df['anchor']).dt.days
    print(f"   After date processing: {len(emar_df):,} records")

    # Step 2: Load dosages from emar_detail
    print("\n5. Loading emar_detail.csv.gz for dosages...")
    needed_emar_ids = set(emar_df['emar_id'].unique())
    print(f"   Need dosages for {len(needed_emar_ids):,} unique emar_ids")

    dosage_chunks = []
    total_detail_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        f'{MIMIC_RAW_PATH}/emar_detail.csv.gz',
        compression='gzip',
        chunksize=2_000_000,
        usecols=['emar_id', 'dose_due', 'dose_given']
    )):
        total_detail_rows += len(chunk)
        if chunk_idx % 2 == 0:
            print(f"   Detail chunk {chunk_idx + 1}: {total_detail_rows:,} rows...")

        # Filter to needed emar_ids
        chunk = chunk[chunk['emar_id'].isin(needed_emar_ids)].copy()

        if len(chunk) > 0:
            dosage_chunks.append(chunk)

    print(f"   Total detail rows: {total_detail_rows:,}")

    # Extract dosages
    if dosage_chunks:
        print("\n6. Extracting dosages (vectorized)...")
        dosage_df = pd.concat(dosage_chunks, ignore_index=True)

        # Extract numeric dosages
        dosage_df['dosage'] = dosage_df['dose_given'].apply(extract_dosage)
        dosage_df['dosage'] = dosage_df['dosage'].fillna(
            dosage_df['dose_due'].apply(extract_dosage)
        )

        # Create dosage lookup by emar_id
        dosage_df = dosage_df.dropna(subset=['dosage'])
        dosage_lookup = dosage_df.groupby('emar_id')['dosage'].first().to_dict()
        print(f"   Dosages extracted: {len(dosage_lookup):,}")

        # Merge dosages
        emar_df['dosage'] = emar_df['emar_id'].map(dosage_lookup)
    else:
        print("\n6. No dosage data found")
        emar_df['dosage'] = None

    # Build output dictionary
    print("\n7. Building output dictionary...")

    # Normalize medication names (high-risk drugs -> UPPERCASE)
    print("   Normalizing medication names...")
    emar_df['medication_normalized'] = emar_df['medication'].apply(normalize_drug_name)

    # Remove duplicates (using normalized names)
    emar_df = emar_df.drop_duplicates(subset=['subject_id', 'date_delta', 'medication_normalized', 'dosage'])
    print(f"   Unique medication records: {len(emar_df):,}")

    # Group and convert to nested dict
    subject_to_date_to_meds = {}

    grouped = emar_df.groupby(['subject_id', 'date_delta'])
    for (subject_id, date_delta), group in tqdm(grouped, desc="   Processing"):
        if subject_id not in subject_to_date_to_meds:
            subject_to_date_to_meds[subject_id] = {}

        # Use normalized medication names
        med_dosage_list = list(zip(group['medication_normalized'], group['dosage']))
        subject_to_date_to_meds[subject_id][date_delta] = med_dosage_list

    # Add empty dicts for patients without meds
    for subject_id in cancer_subjects:
        if subject_id not in subject_to_date_to_meds:
            subject_to_date_to_meds[subject_id] = {}

    # Statistics
    print("\n8. Medication data statistics:")
    patients_with_meds = len([s for s in subject_to_date_to_meds if subject_to_date_to_meds[s]])
    print(f"   Patients with medications: {patients_with_meds:,} / {len(cancer_subjects):,}")

    total_encounters = sum(len(dates) for dates in subject_to_date_to_meds.values())
    print(f"   Patient-date encounters: {total_encounters:,}")

    total_meds = sum(sum(len(meds) for meds in dates.values()) for dates in subject_to_date_to_meds.values())
    print(f"   Total medications: {total_meds:,}")

    # Count unique meds and dosage coverage
    all_meds = set()
    meds_with_dosage = 0
    for dates in subject_to_date_to_meds.values():
        for meds in dates.values():
            for med_name, dosage in meds:
                all_meds.add(med_name)
                if dosage is not None and not pd.isna(dosage):
                    meds_with_dosage += 1

    print(f"   Unique medications: {len(all_meds):,}")
    if total_meds > 0:
        print(f"   Medications with dosage: {meds_with_dosage:,} ({100*meds_with_dosage/total_meds:.1f}%)")

    # Count high-risk drugs
    from config import INCLUDED_DRUGS
    high_risk_counts = {}
    for dates in subject_to_date_to_meds.values():
        for meds in dates.values():
            for med_name, _ in meds:
                if med_name in INCLUDED_DRUGS:
                    high_risk_counts[med_name] = high_risk_counts.get(med_name, 0) + 1

    if high_risk_counts:
        print(f"\n   High-risk drugs found:")
        for drug in sorted(high_risk_counts.keys()):
            print(f"     {drug}: {high_risk_counts[drug]:,} administrations")
    else:
        print(f"\n   ⚠️  No high-risk drugs found in dataset!")

    # Save output
    print("\n9. Saving output...")
    with open(output_file, 'wb') as f:
        pickle.dump(subject_to_date_to_meds, f)
    print(f"   Saved: {os.path.basename(output_file)}")

    print("\n" + "="*80)
    print("Step 4 Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
