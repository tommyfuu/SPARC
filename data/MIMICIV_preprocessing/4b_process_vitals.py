"""
Process vital signs data from omr.
"""

import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from config import MIMIC_RAW_PATH, OUTPUT_PATH, ALL_VITALS
import re

def parse_blood_pressure_vectorized(bp_series):
    """
    Vectorized blood pressure parsing.
    Returns tuple of (systolic_series, diastolic_series).
    """
    def parse_bp(s):
        if pd.isna(s) or not isinstance(s, str):
            return None, None
        match = re.search(r'(\d+)/(\d+)', s)
        if match:
            try:
                return float(match.group(1)), float(match.group(2))
            except:
                return None, None
        return None, None

    parsed = bp_series.apply(parse_bp)
    systolic = parsed.apply(lambda x: x[0] if x else None)
    diastolic = parsed.apply(lambda x: x[1] if x else None)
    return systolic, diastolic

def main():
    print("="*80)
    print("Step 4b: Processing Vital Signs")
    print("="*80)
    print("NOTE: MIMIC-IV hosp module has limited vitals.")
    print("      Full vitals are in ICU chartevents (requires separate download).")
    print()

    output_file = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_vitals_values_list_dict.pkl'

    if os.path.exists(output_file):
        print(f"\nWARNING: Output file already exists:")
        print(f"   - {os.path.basename(output_file)}")
        response = input("\nSkip this step? (y/n): ")
        if response.lower() == 'y':
            print("Skipping step 4b.")
            return
        print("Re-running step 4b...")

    print("1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    print("\n2. Loading anchor dates...")
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)

    print("\n3. Loading and processing omr.csv.gz...")
    omr_df = pd.read_csv(f'{MIMIC_RAW_PATH}/omr.csv.gz', compression='gzip')
    print(f"   Total omr records: {len(omr_df):,}")

    omr_df = omr_df[omr_df['subject_id'].isin(cancer_subjects)].copy()
    print(f"   Cancer patient omr records: {len(omr_df):,}")

    omr_df['chartdate'] = pd.to_datetime(omr_df['chartdate'])

    omr_df['anchor'] = omr_df['subject_id'].map(anchor_dates)
    omr_df = omr_df.dropna(subset=['anchor'])

    omr_df['date_delta'] = (omr_df['chartdate'] - omr_df['anchor']).dt.days

    print("\n4. Processing vital signs (vectorized)...")

    vital_indices = {
        'BMI (kg/m2)': 0,
        'Blood Pressure': (1, 2),  # Systolic and Diastolic
        'O2 saturation pulseoxymetry': 5,
        'Heart Rate': 6,
        'Pulse': 6,
        'Respiratory Rate': 7,
        'Temperature Degrees F': 8,
        'Temperature F': 8,
        'Temperature C': 8,
    }

    subject_to_date_to_vitals = {}

    for result_name, vital_idx in vital_indices.items():
        if result_name not in omr_df['result_name'].values:
            continue

        print(f"   Processing {result_name}...")
        vital_data = omr_df[omr_df['result_name'] == result_name].copy()

        if len(vital_data) == 0:
            continue

        if result_name == 'BMI (kg/m2)':
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][0] = row['value']

        elif result_name == 'Blood Pressure':
            systolic, diastolic = parse_blood_pressure_vectorized(vital_data['result_value'])
            vital_data['systolic'] = systolic
            vital_data['diastolic'] = diastolic
            vital_data = vital_data.dropna(subset=['systolic', 'diastolic'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][1] = row['systolic']
                subject_to_date_to_vitals[sid][date_delta][2] = row['diastolic']

        elif result_name == 'O2 saturation pulseoxymetry':
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][5] = row['value']

        elif result_name in ['Heart Rate', 'Pulse']:
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][6] = row['value']

        elif result_name == 'Respiratory Rate':
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][7] = row['value']

        elif result_name in ['Temperature Degrees F', 'Temperature F']:
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])
            # Convert F to C
            vital_data['value'] = (vital_data['value'] - 32) * 5.0 / 9.0

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][8] = row['value']

        elif result_name == 'Temperature C':
            vital_data['value'] = pd.to_numeric(vital_data['result_value'], errors='coerce')
            vital_data = vital_data.dropna(subset=['value'])

            for _, row in vital_data.iterrows():
                sid = row['subject_id']
                date_delta = row['date_delta']
                if sid not in subject_to_date_to_vitals:
                    subject_to_date_to_vitals[sid] = {}
                if date_delta not in subject_to_date_to_vitals[sid]:
                    subject_to_date_to_vitals[sid][date_delta] = ['NA'] * len(ALL_VITALS)
                subject_to_date_to_vitals[sid][date_delta][8] = row['value']

    # Add empty dicts for patients without vitals
    for subject_id in cancer_subjects:
        if subject_id not in subject_to_date_to_vitals:
            subject_to_date_to_vitals[subject_id] = {}

    print("\n5. Vital signs data statistics:")
    print(f"   Unique patients with vitals: {len([s for s in subject_to_date_to_vitals if subject_to_date_to_vitals[s]]):,}")
    total_encounters = sum(len(dates) for dates in subject_to_date_to_vitals.values())
    print(f"   Total patient-date encounters: {total_encounters:,}")

    vital_data_counts = [0] * len(ALL_VITALS)
    for dates in subject_to_date_to_vitals.values():
        for vitals in dates.values():
            for i, val in enumerate(vitals):
                if val != 'NA' and not pd.isna(val):
                    vital_data_counts[i] += 1

    print(f"\n   Data availability per vital:")
    for i, vital_name in enumerate(ALL_VITALS):
        if vital_data_counts[i] > 0:
            print(f"     {vital_name}: {vital_data_counts[i]:,} measurements")
        else:
            print(f"     {vital_name}: No data (ICU chartevents needed)")

    print("\n6. Saving output...")
    with open(output_file, 'wb') as f:
        pickle.dump(subject_to_date_to_vitals, f)
    print(f"   Saved: {os.path.basename(output_file)}")
    print(f"   File contains {len(subject_to_date_to_vitals):,} patients")

    print("\n" + "="*80)
    print("Step 4b Complete!")
    print("="*80)
    print("\nNOTE: For complete vital signs, process ICU chartevents module.")

if __name__ == '__main__':
    main()
