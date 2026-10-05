"""
Calculate mean and standard deviation for all lab tests and vital signs.
Useful for understanding data distribution and normalization.
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import OUTPUT_PATH, ALL_INCLUDED_TESTS, ALL_VITALS

def main():
    print("="*80)
    print("Step 7: Calculating Lab and Vital Statistics")
    print("="*80)

    print("\n1. Loading lab data...")
    lab_file = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_all_included_lab_values_list_dict.pkl'

    try:
        with open(lab_file, 'rb') as f:
            lab_dict = pickle.load(f)
        print(f"   Loaded lab data: {len(lab_dict):,} patients")
    except FileNotFoundError:
        print(f"   WARNING: Lab file not found: {lab_file}")
        print("   Run step 3 first!")
        lab_dict = None

    if lab_dict:
        print("\n2. Calculating lab test statistics...")

        lab_values = {test: [] for test in ALL_INCLUDED_TESTS}

        for subject_id, date_dict in tqdm(lab_dict.items(), desc="   Processing"):
            for date_delta, test_values in date_dict.items():
                for i, test_name in enumerate(ALL_INCLUDED_TESTS):
                    value = test_values[i]
                    if value != 'NA' and not pd.isna(value):
                        try:
                            lab_values[test_name].append(float(value))
                        except (ValueError, TypeError):
                            pass

        lab_stats = {}
        print("\n   Lab Test Statistics:")
        print("   " + "-"*76)
        print(f"   {'Test Name':<30} {'Count':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("   " + "-"*76)

        for test_name in ALL_INCLUDED_TESTS:
            values = lab_values[test_name]
            if len(values) > 0:
                values_array = np.array(values)
                stats = {
                    'count': len(values),
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'median': np.median(values_array),
                    'q25': np.percentile(values_array, 25),
                    'q75': np.percentile(values_array, 75),
                }
                lab_stats[test_name] = stats

                print(f"   {test_name:<30} {stats['count']:>10,} "
                      f"{stats['mean']:>12.2f} {stats['std']:>12.2f} "
                      f"{stats['min']:>12.2f} {stats['max']:>12.2f}")
            else:
                lab_stats[test_name] = None
                print(f"   {test_name:<30} {'No data':>10}")

        print("   " + "-"*76)

    print("\n3. Loading vital signs data...")
    vital_file = f'{OUTPUT_PATH}/MIMICIV_subjectid_to_date_to_vitals_values_list_dict.pkl'

    try:
        with open(vital_file, 'rb') as f:
            vital_dict = pickle.load(f)
        print(f"   Loaded vital data: {len(vital_dict):,} patients")
    except FileNotFoundError:
        print(f"   WARNING: Vital file not found: {vital_file}")
        print("   Run step 4b first!")
        vital_dict = None

    if vital_dict:
        print("\n4. Calculating vital sign statistics...")

        vital_values = {vital: [] for vital in ALL_VITALS}

        for subject_id, date_dict in tqdm(vital_dict.items(), desc="   Processing"):
            for date_delta, vital_vals in date_dict.items():
                for i, vital_name in enumerate(ALL_VITALS):
                    value = vital_vals[i]
                    if value != 'NA' and not pd.isna(value):
                        try:
                            vital_values[vital_name].append(float(value))
                        except (ValueError, TypeError):
                            pass

        vital_stats = {}
        print("\n   Vital Sign Statistics:")
        print("   " + "-"*76)
        print(f"   {'Vital Name':<30} {'Count':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("   " + "-"*76)

        for vital_name in ALL_VITALS:
            values = vital_values[vital_name]
            if len(values) > 0:
                values_array = np.array(values)
                stats = {
                    'count': len(values),
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'median': np.median(values_array),
                    'q25': np.percentile(values_array, 25),
                    'q75': np.percentile(values_array, 75),
                }
                vital_stats[vital_name] = stats

                print(f"   {vital_name:<30} {stats['count']:>10,} "
                      f"{stats['mean']:>12.2f} {stats['std']:>12.2f} "
                      f"{stats['min']:>12.2f} {stats['max']:>12.2f}")
            else:
                vital_stats[vital_name] = None
                print(f"   {vital_name:<30} {'No data':>10}")

        print("   " + "-"*76)

    print("\n5. Saving statistics...")

    if lab_dict:
        output_file = f'{OUTPUT_PATH}/MIMICIV_lab_statistics_mean_std.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(lab_stats, f)
        print(f"   Saved: {output_file}")

        # Also save as CSV for easy viewing
        csv_file = f'{OUTPUT_PATH}/MIMICIV_lab_statistics_mean_std.csv'
        lab_stats_df = pd.DataFrame.from_dict(lab_stats, orient='index')
        lab_stats_df.index.name = 'test_name'
        lab_stats_df.to_csv(csv_file)
        print(f"   Saved: {csv_file}")

    if vital_dict:
        output_file = f'{OUTPUT_PATH}/MIMICIV_vital_statistics_mean_std.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(vital_stats, f)
        print(f"   Saved: {output_file}")

        # Also save as CSV for easy viewing
        csv_file = f'{OUTPUT_PATH}/MIMICIV_vital_statistics_mean_std.csv'
        vital_stats_df = pd.DataFrame.from_dict(vital_stats, orient='index')
        vital_stats_df.index.name = 'vital_name'
        vital_stats_df.to_csv(csv_file)
        print(f"   Saved: {csv_file}")

    print("\n6. Summary:")

    if lab_dict:
        total_tests = len(ALL_INCLUDED_TESTS)
        tests_with_data = sum(1 for stats in lab_stats.values() if stats is not None)
        print(f"   Lab tests with data: {tests_with_data} / {total_tests}")

        # Find tests with insufficient data
        sparse_tests = [test for test, stats in lab_stats.items()
                       if stats is not None and stats['count'] < 100]
        if sparse_tests:
            print(f"\n   WARNING: Sparse lab tests (<100 measurements):")
            for test in sparse_tests:
                count = lab_stats[test]['count'] if lab_stats[test] else 0
                print(f"     - {test}: {count} measurements")

    if vital_dict:
        total_vitals = len(ALL_VITALS)
        vitals_with_data = sum(1 for stats in vital_stats.values() if stats is not None)
        print(f"\n   Vital signs with data: {vitals_with_data} / {total_vitals}")

        # Find vitals with insufficient data
        sparse_vitals = [vital for vital, stats in vital_stats.items()
                        if stats is not None and stats['count'] < 100]
        if sparse_vitals:
            print(f"\n   WARNING: Sparse vital signs (<100 measurements):")
            for vital in sparse_vitals:
                count = vital_stats[vital]['count'] if vital_stats[vital] else 0
                print(f"     - {vital}: {count} measurements")

    print("\n" + "="*80)
    print("Step 7 Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
