"""
Extract cancer patients from MIMIC-IV based on ICD diagnosis codes.
Creates cancer patient list and cancer type mappings.
"""

import os
import pandas as pd
import pickle
from tqdm import tqdm
from config import (
    MIMIC_RAW_PATH, OUTPUT_PATH,
    is_cancer_code_icd10, is_cancer_code_icd9,
    map_icd10_to_cancer_type
)

def main():
    print()
    print("Step 1: Extracting Cancer Patients")
    print()

    # Check if outputs already exist
    output_files = [
        f'{OUTPUT_PATH}/cancer_subject_ids.pkl',
        f'{OUTPUT_PATH}/MIMICIV_subjectid_to_mapped_cancer_type_dict_080625.pkl'
    ]

    if all(os.path.exists(f) for f in output_files):
        print("\nWARNING: Output files already exist:")
        for f in output_files:
            print(f"   - {os.path.basename(f)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 1.")
            return
        print("Re-running step 1...")

    # Load diagnoses
    print("\n1. Loading diagnoses_icd.csv.gz...")
    diag_df = pd.read_csv(f'{MIMIC_RAW_PATH}/diagnoses_icd.csv.gz', compression='gzip')
    print(f"   Total diagnosis records: {len(diag_df):,}")

    # Load diagnosis code descriptions
    print("\n2. Loading d_icd_diagnoses.csv.gz...")
    d_icd_df = pd.read_csv(f'{MIMIC_RAW_PATH}/d_icd_diagnoses.csv.gz', compression='gzip')

    # Identify cancer patients
    print("\n3. Identifying cancer patients...")
    cancer_mask_icd10 = diag_df['icd_version'] == 10
    cancer_mask_icd9 = diag_df['icd_version'] == 9

    # Filter cancer codes
    icd10_cancer = diag_df[cancer_mask_icd10].copy()
    icd10_cancer['is_cancer'] = icd10_cancer['icd_code'].apply(is_cancer_code_icd10)

    icd9_cancer = diag_df[cancer_mask_icd9].copy()
    icd9_cancer['is_cancer'] = icd9_cancer['icd_code'].apply(is_cancer_code_icd9)

    # Get cancer patients
    cancer_subjects_icd10 = set(icd10_cancer[icd10_cancer['is_cancer']]['subject_id'].unique())
    cancer_subjects_icd9 = set(icd9_cancer[icd9_cancer['is_cancer']]['subject_id'].unique())
    cancer_subjects = cancer_subjects_icd10 | cancer_subjects_icd9

    print(f"   Cancer patients (ICD-10): {len(cancer_subjects_icd10):,}")
    print(f"   Cancer patients (ICD-9): {len(cancer_subjects_icd9):,}")
    print(f"   Total unique cancer patients: {len(cancer_subjects):,}")

    # Map patients to primary cancer types
    print("\n4. Mapping patients to cancer types...")
    subject_to_cancer_type = {}

    for subject_id in tqdm(cancer_subjects, desc="   Processing"):
        # Get all cancer diagnoses for this patient
        patient_diags_icd10 = icd10_cancer[
            (icd10_cancer['subject_id'] == subject_id) &
            (icd10_cancer['is_cancer'])
        ]['icd_code'].tolist()

        if patient_diags_icd10:
            # Use first cancer diagnosis (could also use most frequent)
            cancer_type = map_icd10_to_cancer_type(patient_diags_icd10[0])
            subject_to_cancer_type[subject_id] = cancer_type
        else:
            # ICD-9 only patient - map to 'others' for now
            subject_to_cancer_type[subject_id] = 'others'

    # Print distribution
    print("\n5. Cancer type distribution:")
    from collections import Counter
    cancer_type_counts = Counter(subject_to_cancer_type.values())
    for ctype, count in cancer_type_counts.most_common():
        print(f"   {ctype}: {count:,}")

    # Save outputs
    print("\n6. Saving outputs...")

    # Save cancer subject IDs
    cancer_subject_list = sorted(list(cancer_subjects))
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'wb') as f:
        pickle.dump(cancer_subject_list, f)
    print(f"   Saved: cancer_subject_ids.pkl ({len(cancer_subject_list):,} patients)")

    # Save cancer type mapping (matching naming convention)
    with open(f'{OUTPUT_PATH}/MIMICIV_subjectid_to_mapped_cancer_type_dict_080625.pkl', 'wb') as f:
        pickle.dump(subject_to_cancer_type, f)
    print(f"   Saved: MIMICIV_subjectid_to_mapped_cancer_type_dict_080625.pkl")

    print()
    print("Step 1 Complete!")
    print()

if __name__ == '__main__':
    main()
