"""
Process demographic data: age, gender, death dates, last contact dates.
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
from config import MIMIC_RAW_PATH, OUTPUT_PATH

def main():
    print()
    print("Step 2: Processing Demographics")
    print()

    # Check if outputs already exist
    output_files = [
        f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_gender_dict_080625.pkl',
        f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_age_dict_binned5_080625.pkl',
        f'{OUTPUT_PATH}/MIMIDIV_subjectid_to_deathdate_dict_112325.pkl',
        f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_lastcontactdate_dict_112325.pkl',
        f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl'
    ]

    if all(os.path.exists(f) for f in output_files):
        print("\nWARNING: Output files already exist:")
        for f in output_files:
            print(f"   - {os.path.basename(f)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 2.")
            return
        print("Re-running step 2...")

    # Load cancer patients
    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    # Load patients table
    print("\n2. Loading patients.csv.gz...")
    patients_df = pd.read_csv(f'{MIMIC_RAW_PATH}/patients.csv.gz', compression='gzip')

    # Filter to cancer patients
    cancer_patients_df = patients_df[patients_df['subject_id'].isin(cancer_subjects)].copy()
    print(f"   Cancer patients in patients table: {len(cancer_patients_df):,}")

    # Load admissions to get anchor dates
    print("\n3. Loading admissions.csv.gz...")
    admissions_df = pd.read_csv(f'{MIMIC_RAW_PATH}/admissions.csv.gz', compression='gzip')
    cancer_admissions = admissions_df[admissions_df['subject_id'].isin(cancer_subjects)].copy()

    # Convert admission times to datetime
    cancer_admissions['admittime'] = pd.to_datetime(cancer_admissions['admittime'])
    cancer_admissions['dischtime'] = pd.to_datetime(cancer_admissions['dischtime'])

    # Get first admission date for each patient (as anchor)
    first_admission = cancer_admissions.groupby('subject_id')['admittime'].min().to_dict()

    # Process demographics
    print("\n4. Processing gender...")
    subject_to_gender = dict(zip(cancer_patients_df['subject_id'], cancer_patients_df['gender']))
    # Convert M/F to MALE/FEMALE
    subject_to_gender = {k: 'MALE' if v == 'M' else 'FEMALE' for k, v in subject_to_gender.items()}
    print(f"   Gender distribution:")
    from collections import Counter
    gender_counts = Counter(subject_to_gender.values())
    for gender, count in gender_counts.items():
        print(f"     {gender}: {count:,}")

    print("\n5. Processing age (binned by 5 years)...")
    subject_to_age_binned = {}
    for _, row in tqdm(cancer_patients_df.iterrows(), total=len(cancer_patients_df), desc="   Processing"):
        subject_id = row['subject_id']
        anchor_age = row['anchor_age']
        # Bin age by 5 years: age // 5
        age_binned = anchor_age // 5
        subject_to_age_binned[subject_id] = age_binned

    age_dist = Counter(subject_to_age_binned.values())
    print(f"   Age bin distribution (bins of 5 years):")
    for age_bin in sorted(age_dist.keys()):
        print(f"     Bin {age_bin} ({age_bin*5}-{age_bin*5+4} years): {age_dist[age_bin]:,}")

    print("\n6. Processing death dates...")
    subject_to_deathdate = {}
    for _, row in tqdm(cancer_patients_df.iterrows(), total=len(cancer_patients_df), desc="   Processing"):
        subject_id = row['subject_id']
        dod = row['dod']  # date of death

        if pd.notna(dod) and subject_id in first_admission:
            dod_datetime = pd.to_datetime(dod)
            anchor_datetime = first_admission[subject_id]
            # Calculate days from anchor (de-identified delta)
            days_delta = (dod_datetime - anchor_datetime).days
            subject_to_deathdate[subject_id] = days_delta
        else:
            subject_to_deathdate[subject_id] = np.nan

    n_deaths = sum(1 for v in subject_to_deathdate.values() if not pd.isna(v))
    print(f"   Patients with death date: {n_deaths:,} / {len(subject_to_deathdate):,}")

    print("\n7. Processing last contact dates...")
    # Last contact = max of (discharge time of last admission, death date, or last lab date)
    # For now, use last admission discharge time
    last_admission_discharge = cancer_admissions.groupby('subject_id')['dischtime'].max().to_dict()

    subject_to_lastcontact = {}
    for subject_id in tqdm(cancer_subjects, desc="   Processing"):
        if subject_id in last_admission_discharge and subject_id in first_admission:
            last_contact_datetime = last_admission_discharge[subject_id]
            anchor_datetime = first_admission[subject_id]
            days_delta = (last_contact_datetime - anchor_datetime).days
            subject_to_lastcontact[subject_id] = days_delta
        else:
            subject_to_lastcontact[subject_id] = 0  # Default to 0 if no admissions

    print(f"   Last contact dates computed for {len(subject_to_lastcontact):,} patients")

    # Save outputs
    print("\n8. Saving outputs...")

    with open(f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_gender_dict_080625.pkl', 'wb') as f:
        pickle.dump(subject_to_gender, f)
    print(f"   Saved: MIMICIV_cancer_subjectid_to_gender_dict_080625.pkl")

    with open(f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_age_dict_binned5_080625.pkl', 'wb') as f:
        pickle.dump(subject_to_age_binned, f)
    print(f"   Saved: MIMICIV_cancer_subjectid_to_age_dict_binned5_080625.pkl")

    with open(f'{OUTPUT_PATH}/MIMIDIV_subjectid_to_deathdate_dict_112325.pkl', 'wb') as f:
        pickle.dump(subject_to_deathdate, f)
    print(f"   Saved: MIMIDIV_subjectid_to_deathdate_dict_112325.pkl")

    with open(f'{OUTPUT_PATH}/MIMICIV_cancer_subjectid_to_lastcontactdate_dict_112325.pkl', 'wb') as f:
        pickle.dump(subject_to_lastcontact, f)
    print(f"   Saved: MIMICIV_cancer_subjectid_to_lastcontactdate_dict_112325.pkl")

    # Save first admission dates (anchor dates) for use in other scripts
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'wb') as f:
        pickle.dump(first_admission, f)
    print(f"   Saved: subject_to_anchor_date_dict.pkl")

    print()
    print("Step 2 Complete!")
    print()

if __name__ == '__main__':
    main()
