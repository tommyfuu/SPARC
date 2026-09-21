"""
Generate a demographics table from dataset summaries.
Creates a CSV with rows=datasets, columns=gender (M%), age at first lab (mean±std),
race/ethnicity percentages, and age distribution bins.

Author: Claude Code
Date: 2026-09-05
Updated: 2026-09-10 (added race/ethnicity columns)
"""

import argparse
import os
import pandas as pd
import glob
from pathlib import Path

# Default root holding the cohort summary outputs. Override with --source-dir.
DEFAULT_SOURCE_DIR = "/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models/results_vis/cohort_summaries"

# Populated from command-line arguments in main().
BASE_DIR = None
RACE_ETHNICITY_DIR = None
OUTPUT_PATH = None


def parse_args():
    """Parse command-line arguments for input/output directories."""
    parser = argparse.ArgumentParser(
        description="Generate a demographics table from dataset summaries."
    )
    parser.add_argument(
        "-s", "--source-dir", default=DEFAULT_SOURCE_DIR,
        help="Root cohort_summaries directory holding the summary outputs "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--dataset-summaries-dir", default=None,
        help="Directory of per-dataset summary folders "
             "(default: <source-dir>/dataset_summaries)"
    )
    parser.add_argument(
        "-r", "--race-ethnicity-dir", default=None,
        help="Directory of harmonized race/ethnicity CSVs "
             "(default: <source-dir>/race_ethnicity/harmonized)"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(DEFAULT_SOURCE_DIR, "demographics_table.csv"),
        help="Path to write the demographics table CSV (default: %(default)s)"
    )
    args = parser.parse_args()

    if args.dataset_summaries_dir is None:
        args.dataset_summaries_dir = os.path.join(args.source_dir, "dataset_summaries")
    if args.race_ethnicity_dir is None:
        args.race_ethnicity_dir = os.path.join(
            args.source_dir, "race_ethnicity", "harmonized"
        )
    return args

def extract_dataset_name_from_folder(folder_name):
    """Extract clean dataset name from folder name."""
    # Remove 'parsimonious_' prefix and 'practical_' prefix
    name = folder_name.replace('parsimonious_', '').replace('practical_', '')
    return name

def find_dataset_summary_file(folder_path):
    """Find the dataset_summary CSV file in the folder."""
    # Try different naming patterns
    patterns = [
        "dataset_summary_*.csv",
        "dataset_summary_*_suppressed.csv"
    ]

    for pattern in patterns:
        files = glob.glob(os.path.join(folder_path, pattern))
        if files:
            return files[0]

    return None

def find_age_distribution_file(folder_path):
    """Find the age_distribution CSV file in the folder."""
    pattern = "age_distribution_*.csv"
    files = glob.glob(os.path.join(folder_path, pattern))
    if files:
        return files[0]
    return None

def suppress_count(count):
    """Suppress counts <= 10 by showing '<=10'."""
    # Handle string inputs (already suppressed values)
    if isinstance(count, str):
        if 'suppress' in count.lower() or '<' in count:
            return "<=10"
        try:
            count = float(count)
        except ValueError:
            return "<=10"

    # Handle numeric inputs
    if count <= 10:
        return "<=10"
    return str(int(count))

def extract_age_bins(age_distribution_path):
    """Extract age bin counts from age distribution CSV."""
    if not os.path.exists(age_distribution_path):
        return {}

    # Read the age distribution CSV
    df = pd.read_csv(age_distribution_path)

    # Consolidate age bins
    consolidated_bins = {}
    young_count = 0  # For age ≤4
    old_count = 0    # For age ≥85

    for _, row in df.iterrows():
        age_group = row['age_group_years']
        count = row['count']

        # Convert count to numeric (handle suppressed values)
        if isinstance(count, str):
            if 'suppress' in count.lower() or '<' in count:
                count = 5  # Use midpoint of suppressed range for aggregation
            else:
                try:
                    count = float(count)
                except ValueError:
                    count = 5  # Default for unparseable values

        # Handle negative ages or 0-4: consolidate into ≤4
        if age_group.startswith('-') or age_group == '0–4' or age_group == '0-4':
            young_count += count
        # Handle 85-89, 90-94, and higher: consolidate into ≥85
        elif age_group in ['85–89', '85-89', '90–94', '90-94']:
            old_count += count
        # Check if it's a numeric age group >= 85
        elif '–' in age_group or '-' in age_group:
            separator = '–' if '–' in age_group else '-'
            first_part = age_group.split(separator)[0]
            if first_part.isdigit() and int(first_part) >= 85:
                old_count += count
            else:
                if age_group not in consolidated_bins:
                    consolidated_bins[age_group] = 0
                consolidated_bins[age_group] += count
        # Keep other age groups as-is (shouldn't happen with standard age bins)
        else:
            if age_group not in consolidated_bins:
                consolidated_bins[age_group] = 0
            consolidated_bins[age_group] += count

    # Add the consolidated groups
    if young_count > 0:
        consolidated_bins['≤4'] = young_count
    if old_count > 0:
        consolidated_bins['≥85'] = old_count

    # Apply suppression
    for age_group in consolidated_bins:
        consolidated_bins[age_group] = suppress_count(consolidated_bins[age_group])

    return consolidated_bins

def map_dataset_name_to_race_file(dataset_name):
    """Map dataset display name to race/ethnicity file prefix."""
    # Handle INSIGHT datasets (can be INSIGHT_A or INSIGHT-A)
    if 'INSIGHT' in dataset_name:
        # Extract letter (A, B, C, D, E, or F)
        if '_' in dataset_name:
            letter = dataset_name.split('_')[1]
        elif '-' in dataset_name:
            letter = dataset_name.split('-')[1]
        else:
            letter = dataset_name.replace('INSIGHT', '')
        return f"insight_{letter}"
    elif dataset_name == 'MIMICIV_hosp':
        return 'MIMICIV_hosp'
    elif dataset_name == 'MSK_CAIA_timeshift':
        return 'MSK_CAIA_timeshift'
    elif dataset_name == 'MSK_CAIA':
        return 'MSK_CAIA'
    elif dataset_name == 'JHU_CAIA':
        return 'JHU_CAIA'
    elif dataset_name == 'FHCC_CAIA':
        return 'FHCC_CAIA'
    elif dataset_name == 'EHRSHOT' or dataset_name == 'StanfordEHRSHOT':
        return 'EHRSHOT'
    elif dataset_name == 'DFCI_CAIA_all_060526':
        return None  # No race/ethnicity data for DFCI
    else:
        return dataset_name

def extract_race_ethnicity(dataset_name):
    """Extract race and ethnicity percentages from harmonized CSVs."""
    file_prefix = map_dataset_name_to_race_file(dataset_name)

    # If mapping returns None, no race/ethnicity data available
    if file_prefix is None:
        return {}

    race_file = os.path.join(RACE_ETHNICITY_DIR, f"{file_prefix}_race_harmonized.csv")
    ethnicity_file = os.path.join(RACE_ETHNICITY_DIR, f"{file_prefix}_ethnicity_harmonized.csv")

    result = {}

    # Extract race data
    if os.path.exists(race_file):
        race_df = pd.read_csv(race_file)
        total = race_df['count'].sum()

        for _, row in race_df.iterrows():
            category = row['race_concept_name']
            count = row['count']
            pct = 100 * count / total if total > 0 else 0

            # Create shortened column names
            if category == 'White':
                result['White (%)'] = pct
            elif category == 'Black or African American':
                result['Black (%)'] = pct
            elif category == 'Asian':
                result['Asian (%)'] = pct
            elif category == 'American Indian or Alaska Native':
                result['AIAN (%)'] = pct
            elif category == 'Native Hawaiian or Other Pacific Islander':
                result['NHPI (%)'] = pct
            elif category == 'Other/Unknown/Refused':
                result['Other/Unknown (%)'] = pct

    # Extract ethnicity data
    if os.path.exists(ethnicity_file):
        eth_df = pd.read_csv(ethnicity_file)
        total = eth_df['count'].sum()

        for _, row in eth_df.iterrows():
            category = row['ethnicity_concept_name']
            count = row['count']
            pct = 100 * count / total if total > 0 else 0

            if category == 'Hispanic or Latino':
                result['Hispanic/Latino (%)'] = pct
            elif category == 'Not Hispanic or Latino':
                result['Not Hispanic/Latino (%)'] = pct
            elif category == 'Unknown':
                result['Ethnicity Unknown (%)'] = pct

    return result

def extract_demographics(dataset_summary_path):
    """Extract gender and age demographics from dataset summary CSV."""
    if not os.path.exists(dataset_summary_path):
        print(f"Warning: {dataset_summary_path} not found")
        return None

    # Read the CSV
    df = pd.read_csv(dataset_summary_path)

    # Extract relevant metrics
    metrics = {}
    for _, row in df.iterrows():
        metrics[row['metric']] = row['value']

    # Extract gender percentage (M%)
    pct_male = metrics.get('pct_male', None)

    # Extract age information (in bins)
    age_mean_bin = metrics.get('age_mean_bin', None)
    age_std_bin = metrics.get('age_std_bin', None)

    # Convert age bins to years
    # Each bin represents a 5-year interval (0-4, 5-9, 10-14, etc.)
    # Bin value is the lower bound divided by 5
    # To get the mean age: age_mean_bin * 5 + 2.5 (midpoint of the bin)
    # To get the std: age_std_bin * 5
    if age_mean_bin is not None and age_std_bin is not None:
        age_mean_years = age_mean_bin * 5 + 2.5
        age_std_years = age_std_bin * 5
    else:
        age_mean_years = None
        age_std_years = None

    # Extract patient count
    n_patients = metrics.get('n_patients', None)

    return {
        'pct_male': pct_male,
        'age_mean_years': age_mean_years,
        'age_std_years': age_std_years,
        'n_patients': n_patients
    }

def main():
    """Main function to generate demographics table."""

    global BASE_DIR, RACE_ETHNICITY_DIR, OUTPUT_PATH
    args = parse_args()
    BASE_DIR = args.dataset_summaries_dir
    RACE_ETHNICITY_DIR = args.race_ethnicity_dir
    OUTPUT_PATH = args.output

    if not os.path.isdir(BASE_DIR):
        raise SystemExit(f"Dataset summaries directory not found: {BASE_DIR}")
    if not os.path.isdir(RACE_ETHNICITY_DIR):
        print(f"Warning: race/ethnicity directory not found: {RACE_ETHNICITY_DIR}")
        print("Race and ethnicity columns will be left blank.")

    out_parent = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    os.makedirs(out_parent, exist_ok=True)

    print(f"Dataset summaries: {BASE_DIR}")
    print(f"Race/ethnicity:    {RACE_ETHNICITY_DIR}")
    print(f"Output:            {OUTPUT_PATH}\n")

    # Get all dataset folders (exclude 'insight' parent folder)
    dataset_folders = []
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path) and item != 'insight':
            dataset_folders.append(item)

    # Sort for consistent ordering
    dataset_folders.sort()

    # First pass: collect all unique age bins across datasets
    all_age_bins = set()
    dataset_info = {}
    dataset_order = []  # Keep track of order

    for folder in dataset_folders:
        folder_path = os.path.join(BASE_DIR, folder)
        dataset_name = extract_dataset_name_from_folder(folder)

        # Special handling for INSIGHT datasets - they're labeled A-F
        if 'INSIGHT' in folder:
            display_name = folder.replace('parsimonious_', '')
        else:
            display_name = dataset_name

        # Find files
        summary_file = find_dataset_summary_file(folder_path)
        age_dist_file = find_age_distribution_file(folder_path)

        if summary_file is None:
            print(f"Warning: No dataset summary found for {folder}")
            continue

        # Extract demographics
        demo = extract_demographics(summary_file)

        if demo is None:
            print(f"Warning: Could not extract demographics for {folder}")
            continue

        # Extract age bins
        age_bins = {}
        if age_dist_file:
            age_bins = extract_age_bins(age_dist_file)
            all_age_bins.update(age_bins.keys())

        # Extract race/ethnicity data
        race_eth_data = extract_race_ethnicity(display_name)

        # Store info for second pass
        dataset_info[display_name] = {
            'demo': demo,
            'age_bins': age_bins,
            'race_eth': race_eth_data
        }
        dataset_order.append(display_name)

        print(f"Processed: {display_name}")

    # Sort age bins for consistent column ordering
    def sort_key(age_bin):
        # Handle special cases
        if age_bin == '≤4':
            return -1  # Put at the beginning
        elif age_bin == '≥85':
            return 999  # Put at the end

        # Handle different formats like "5–9", "10-14"
        if '–' in age_bin:
            first_part = age_bin.split('–')[0]
        elif '-' in age_bin:
            first_part = age_bin.split('-')[0]
        else:
            first_part = age_bin

        try:
            return int(first_part) if first_part else 0
        except ValueError:
            return 0

    sorted_age_bins = sorted(all_age_bins, key=sort_key)

    # Second pass: build the table with all columns
    demographics_data = []

    for display_name in dataset_order:
        info = dataset_info[display_name]
        demo = info['demo']
        age_bins = info['age_bins']
        race_eth = info['race_eth']

        # Format the base data
        row_data = {
            'Dataset': display_name,
            'N Patients': int(demo['n_patients']) if demo['n_patients'] is not None else None,
            'Male (%)': f"{demo['pct_male']:.2f}" if demo['pct_male'] is not None else None,
            'Age at First Lab (mean ± std)': f"{demo['age_mean_years']:.1f} ± {demo['age_std_years']:.1f}"
                if demo['age_mean_years'] is not None and demo['age_std_years'] is not None else None
        }

        # Add race columns
        race_cols = ['White (%)', 'Black (%)', 'Asian (%)', 'AIAN (%)', 'NHPI (%)', 'Other/Unknown (%)']
        for col in race_cols:
            if col in race_eth:
                row_data[col] = f"{race_eth[col]:.1f}"
            else:
                row_data[col] = ""

        # Add ethnicity columns
        eth_cols = ['Hispanic/Latino (%)', 'Not Hispanic/Latino (%)', 'Ethnicity Unknown (%)']
        for col in eth_cols:
            if col in race_eth:
                row_data[col] = f"{race_eth[col]:.1f}"
            else:
                row_data[col] = ""

        # Add age bin columns
        for age_bin in sorted_age_bins:
            col_name = f"Age {age_bin}"
            row_data[col_name] = age_bins.get(age_bin, "")

        demographics_data.append(row_data)

    # Create DataFrame
    df_demographics = pd.DataFrame(demographics_data)

    # Save to CSV
    df_demographics.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDemographics table saved to: {OUTPUT_PATH}")
    print(f"\nTable shape: {df_demographics.shape[0]} datasets × {df_demographics.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df_demographics.columns):
        print(f"  {i+1:2d}. {col}")
    print(f"\nTable preview (first 10 columns):")
    print(df_demographics.iloc[:, :10].to_string(index=False))

if __name__ == "__main__":
    main()
