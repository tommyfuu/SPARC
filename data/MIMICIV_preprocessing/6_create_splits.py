"""
Create train/val/test splits for MIMIC-IV data.
Uses 60/20/20 split (similar to stanfordEHRSHOT).
"""

import os
import pickle
import random
from config import OUTPUT_PATH

def main():
    print("="*80)
    print("Step 6: Creating Train/Val/Test Splits")
    print("="*80)

    # Check if output already exists
    output_file = f'{OUTPUT_PATH}/split_dict_1v1v1_112425.pickle'

    if os.path.exists(output_file):
        print(f"\n⚠️  Output file already exists:")
        print(f"   - {os.path.basename(output_file)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 6.")
            return
        print("Re-running step 6...")

    # Load cancer patients
    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        all_subjects = pickle.load(f)
    print(f"   Total cancer patients: {len(all_subjects):,}")

    # Filter to patients with lab data
    print("\n2. Filtering to patients with lab data...")
    try:
        with open(f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_all_included_lab_values_list_dict.pkl', 'rb') as f:
            lab_dict = pickle.load(f)
        subjects_with_labs = [s for s in all_subjects if s in lab_dict and len(lab_dict[s]) > 0]
    except FileNotFoundError:
        print("   WARNING: Lab data file not found. Using all cancer patients.")
        subjects_with_labs = all_subjects

    print(f"   Patients with lab data: {len(subjects_with_labs):,}")

    # Create splits
    print("\n3. Creating splits (60% train / 20% val / 20% test)...")
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(subjects_with_labs)

    n = len(subjects_with_labs)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_subjects = subjects_with_labs[:n_train]
    val_subjects = subjects_with_labs[n_train:n_train + n_val]
    test_subjects = subjects_with_labs[n_train + n_val:]

    print(f"   Train: {len(train_subjects):,} patients")
    print(f"   Val:   {len(val_subjects):,} patients")
    print(f"   Test:  {len(test_subjects):,} patients")

    # Create split dict
    split_dict = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }

    # Save output
    print("\n4. Saving output...")
    output_path = f'{OUTPUT_PATH}/split_dict_1v1v1_112425.pickle'
    with open(output_path, 'wb') as f:
        pickle.dump(split_dict, f)
    print(f"   Saved: split_dict_1v1v1_112425.pickle")

    print("\n" + "="*80)
    print("Step 6 Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
