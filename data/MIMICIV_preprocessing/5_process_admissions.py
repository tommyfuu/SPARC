"""
Process hospital admission and discharge dates.
Creates admission-discharge tuple lists for each patient.
"""

import os
import pandas as pd
import pickle
from tqdm import tqdm
from config import MIMIC_RAW_PATH, OUTPUT_PATH

def main():
    print("="*80)
    print("Step 5: Processing Hospital Admissions")
    print("="*80)

    # Check if output already exists
    output_file = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_admission_discharge_dict_112325.pkl'

    if os.path.exists(output_file):
        print(f"\nWARNING: Output file already exists:")
        print(f"   - {os.path.basename(output_file)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 5.")
            return
        print("Re-running step 5...")

    # Load cancer patients
    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    # Load anchor dates
    print("\n2. Loading anchor dates...")
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)

    # Load admissions
    print("\n3. Loading admissions.csv.gz...")
    admissions_df = pd.read_csv(f'{MIMIC_RAW_PATH}/admissions.csv.gz', compression='gzip')

    # Filter to cancer patients
    admissions_df = admissions_df[admissions_df['subject_id'].isin(cancer_subjects)]
    print(f"   Cancer patient admissions: {len(admissions_df):,}")

    # Convert to datetime
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

    # Create admission-discharge tuples
    print("\n4. Creating admission-discharge tuples...")
    subject_to_admissions = {}

    for subject_id in tqdm(cancer_subjects, desc="   Processing"):
        if subject_id not in anchor_dates:
            continue

        # Get all admissions for this patient
        patient_admissions = admissions_df[admissions_df['subject_id'] == subject_id]

        if len(patient_admissions) == 0:
            subject_to_admissions[subject_id] = []
            continue

        # Convert to delta days from anchor
        anchor = anchor_dates[subject_id]
        admission_tuples = []

        for _, row in patient_admissions.iterrows():
            admit_delta = (row['admittime'] - anchor).days
            discharge_delta = (row['dischtime'] - anchor).days
            admission_tuples.append((admit_delta, discharge_delta))

        # Sort by admission date
        admission_tuples.sort(key=lambda x: x[0])

        subject_to_admissions[subject_id] = admission_tuples

    # Print statistics
    print("\n5. Admission statistics:")
    patients_with_admissions = sum(1 for v in subject_to_admissions.values() if v)
    print(f"   Patients with admissions: {patients_with_admissions:,} / {len(subject_to_admissions):,}")

    total_admissions = sum(len(v) for v in subject_to_admissions.values())
    print(f"   Total admissions: {total_admissions:,}")

    avg_admissions = total_admissions / patients_with_admissions if patients_with_admissions > 0 else 0
    print(f"   Average admissions per patient: {avg_admissions:.2f}")

    # Save output
    print("\n6. Saving output...")
    output_path = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_admission_discharge_dict_112325.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(subject_to_admissions, f)
    print(f"   Saved: MIMICIV_subjectid_to_admission_discharge_dict_112325.pkl")

    print("\n" + "="*80)
    print("Step 5 Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
