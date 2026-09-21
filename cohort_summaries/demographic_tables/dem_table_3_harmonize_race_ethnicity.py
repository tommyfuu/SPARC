"""
Create harmonized race/ethnicity mappings across all datasets.
Maps all race categories to 6 standard categories:
1. White
2. Black or African American
3. Asian
4. American Indian or Alaska Native
5. Native Hawaiian or Other Pacific Islander
6. Other/Unknown/Refused

Generates harmonized CSVs for all datasets.
"""

import argparse
import os
import pandas as pd
from collections import Counter

# Default location of the per-dataset raw race/ethnicity CSVs.
DEFAULT_RACE_ETHNICITY_DIR = (
    "/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models"
    "/results_vis/cohort_summaries/race_ethnicity"
)


def parse_args():
    """Parse command-line arguments for input/output directories."""
    parser = argparse.ArgumentParser(
        description="Create harmonized race/ethnicity mappings across all datasets."
    )
    parser.add_argument(
        "-r", "--race-ethnicity-dir", default=DEFAULT_RACE_ETHNICITY_DIR,
        help="Directory holding per-dataset raw race/ethnicity CSVs "
             "(default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Where to write harmonized CSVs (default: <race-ethnicity-dir>/harmonized)"
    )
    return parser.parse_args()

# Define the harmonization mapping
RACE_HARMONIZATION_MAP = {
    # Standard categories (keep as-is)
    'White': 'White',
    'Black or African American': 'Black or African American',
    'Asian': 'Asian',
    'American Indian or Alaska Native': 'American Indian or Alaska Native',
    'Native Hawaiian or Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',

    # American Indian / Alaska Native reported separately (FHCC_CAIA)
    'American Indian': 'American Indian or Alaska Native',
    'Alaska Native': 'American Indian or Alaska Native',

    # Asian subcategories
    'Asian Indian': 'Asian',
    'Chinese': 'Asian',
    'Filipino': 'Asian',
    'Japanese': 'Asian',
    'Korean': 'Asian',
    'Vietnamese': 'Asian',
    'Hmong': 'Asian',  # Southeast Asian ethnic group
    'Thai': 'Asian',   # Southeast Asian
    'Pakistani': 'Asian',  # South Asian

    # Pacific Islander subcategories
    'Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
    'Samoan': 'Native Hawaiian or Other Pacific Islander',
    'Melanesian': 'Native Hawaiian or Other Pacific Islander',  # Pacific Islander group

    # Other/Unknown/Refused categories
    'No matching concept': 'Other/Unknown/Refused',
    'Other': 'Other/Unknown/Refused',
    'Multiple race': 'Other/Unknown/Refused',
    'Refuse to answer': 'Other/Unknown/Refused',
    'Hispanic': 'Other/Unknown/Refused',  # This is ethnicity, not race

    # EHRSHOT anonymized codes
    'Race/1': 'Other/Unknown/Refused',
    'Race/2': 'Other/Unknown/Refused',
    'Race/3': 'Other/Unknown/Refused',
    'Race/4': 'Other/Unknown/Refused',
    'Race/5': 'Other/Unknown/Refused',
}

# Ethnicity is already standardized, but create mapping for consistency
ETHNICITY_HARMONIZATION_MAP = {
    'Hispanic or Latino': 'Hispanic or Latino',
    'Not Hispanic or Latino': 'Not Hispanic or Latino',
    'No matching concept': 'Unknown',
}

def harmonize_race_csv(input_file, output_file):
    """Harmonize a race CSV file using the mapping."""
    df = pd.read_csv(input_file)

    # Ensure count is numeric
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)

    # Apply mapping
    df['harmonized_race'] = df['race_concept_name'].map(RACE_HARMONIZATION_MAP)

    # Check for unmapped categories. These must never be dropped silently:
    # an unmapped label would become NaN and disappear in the groupby below,
    # removing those patients from the denominator entirely.
    unmapped = df[df['harmonized_race'].isna()]
    if not unmapped.empty:
        print(f"  WARNING: Unmapped race categories in {input_file}:")
        for cat in unmapped['race_concept_name'].unique():
            n = int(df.loc[df['race_concept_name'] == cat, 'count'].sum())
            print(f"    - {cat} (n={n:,}) -> Other/Unknown/Refused")
        print("    Add these to RACE_HARMONIZATION_MAP if they belong elsewhere.")
        df['harmonized_race'] = df['harmonized_race'].fillna('Other/Unknown/Refused')

    # Aggregate by harmonized category
    harmonized_counts = df.groupby('harmonized_race')['count'].sum().reset_index()
    harmonized_counts.columns = ['race_concept_name', 'count']

    # Sort by count descending
    harmonized_counts = harmonized_counts.sort_values('count', ascending=False)

    # Save
    harmonized_counts.to_csv(output_file, index=False)

    return harmonized_counts

def harmonize_ethnicity_csv(input_file, output_file):
    """Harmonize an ethnicity CSV file using the mapping."""
    df = pd.read_csv(input_file)

    # Ensure count is numeric
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)

    # Apply mapping
    df['harmonized_ethnicity'] = df['ethnicity_concept_name'].map(ETHNICITY_HARMONIZATION_MAP)

    # Check for unmapped categories (see note in harmonize_race_csv).
    unmapped = df[df['harmonized_ethnicity'].isna()]
    if not unmapped.empty:
        print(f"  WARNING: Unmapped ethnicity categories in {input_file}:")
        for cat in unmapped['ethnicity_concept_name'].unique():
            n = int(df.loc[df['ethnicity_concept_name'] == cat, 'count'].sum())
            print(f"    - {cat} (n={n:,}) -> Unknown")
        print("    Add these to ETHNICITY_HARMONIZATION_MAP if they belong elsewhere.")
        df['harmonized_ethnicity'] = df['harmonized_ethnicity'].fillna('Unknown')

    # Aggregate by harmonized category
    harmonized_counts = df.groupby('harmonized_ethnicity')['count'].sum().reset_index()
    harmonized_counts.columns = ['ethnicity_concept_name', 'count']

    # Sort by count descending
    harmonized_counts = harmonized_counts.sort_values('count', ascending=False)

    # Save
    harmonized_counts.to_csv(output_file, index=False)

    return harmonized_counts

def main():
    print("="*80)
    print("Creating Harmonized Race/Ethnicity Mappings")
    print("="*80)

    args = parse_args()
    base_dir = args.race_ethnicity_dir
    output_dir = args.output_dir or f'{base_dir}/harmonized'
    os.makedirs(output_dir, exist_ok=True)

    # display name -> (input file stem, output file stem)
    # The input stem is how the raw CSVs are named inside {base_dir}/{display}/;
    # the output stem is the name downstream scripts expect in harmonized/.
    # DFCI_CAIA and FHCC_CAIA are the datasets whose raw files carry an
    # '_all' infix; the rest use the bare stem.
    datasets = {
        'DFCI_CAIA': ('DFCI_CAIA_all', 'DFCI_CAIA'),
        'EHRSHOT': ('EHRSHOT', 'EHRSHOT'),
        'FHCC_CAIA': ('FHCC_CAIA_all', 'FHCC_CAIA'),
        'INSIGHT-A': ('insight_A', 'insight_A'),
        'INSIGHT-B': ('insight_B', 'insight_B'),
        'INSIGHT-C': ('insight_C', 'insight_C'),
        'INSIGHT-D': ('insight_D', 'insight_D'),
        'INSIGHT-E': ('insight_E', 'insight_E'),
        'INSIGHT-F': ('insight_F', 'insight_F'),
        'JHU_CAIA': ('JHU_CAIA', 'JHU_CAIA'),
        'MIMICIV_hosp': ('MIMICIV_hosp', 'MIMICIV_hosp'),
        'MSK_CAIA': ('MSK_CAIA', 'MSK_CAIA'),
        'MSK_CAIA_timeshift': ('MSK_CAIA_timeshift', 'MSK_CAIA_timeshift'),
    }

    print("\n" + "="*80)
    print("HARMONIZATION MAPPING")
    print("="*80)
    print("\nRace categories will be mapped to:")
    print("  1. White")
    print("  2. Black or African American")
    print("  3. Asian")
    print("  4. American Indian or Alaska Native")
    print("  5. Native Hawaiian or Other Pacific Islander")
    print("  6. Other/Unknown/Refused")
    print("\nEthnicity categories will be mapped to:")
    print("  1. Hispanic or Latino")
    print("  2. Not Hispanic or Latino")
    print("  3. Unknown")

    print("\n" + "="*80)
    print("Processing Datasets")
    print("="*80)

    all_harmonized_race = Counter()
    all_harmonized_ethnicity = Counter()

    for dataset_display, (input_stem, output_stem) in sorted(datasets.items()):
        print(f"\n{dataset_display}:")

        # Process race
        race_file = f"{base_dir}/{dataset_display}/{input_stem}_race.csv"
        if os.path.exists(race_file):
            output_race_file = f"{output_dir}/{output_stem}_race_harmonized.csv"
            harmonized_race = harmonize_race_csv(race_file, output_race_file)
            print(f"  ✓ Race harmonized → {output_race_file}")

            # Add to overall counts
            for _, row in harmonized_race.iterrows():
                all_harmonized_race[row['race_concept_name']] += row['count']
        else:
            print(f"  ✗ Race file not found: {race_file}")

        # Process ethnicity
        ethnicity_file = f"{base_dir}/{dataset_display}/{input_stem}_ethnicity.csv"
        if os.path.exists(ethnicity_file):
            output_ethnicity_file = f"{output_dir}/{output_stem}_ethnicity_harmonized.csv"
            harmonized_ethnicity = harmonize_ethnicity_csv(ethnicity_file, output_ethnicity_file)
            print(f"  ✓ Ethnicity harmonized → {output_ethnicity_file}")

            # Add to overall counts
            for _, row in harmonized_ethnicity.iterrows():
                all_harmonized_ethnicity[row['ethnicity_concept_name']] += row['count']
        else:
            print(f"  ✗ Ethnicity file not found: {ethnicity_file}")

    # Create combined summary across all datasets
    print("\n" + "="*80)
    print("Creating Combined Summary")
    print("="*80)

    # Combined race
    combined_race_df = pd.DataFrame([
        {'race_concept_name': race, 'count': count}
        for race, count in sorted(all_harmonized_race.items(), key=lambda x: -x[1])
    ])
    combined_race_file = f"{output_dir}/ALL_DATASETS_race_harmonized.csv"
    combined_race_df.to_csv(combined_race_file, index=False)
    print(f"\n✓ Combined race summary saved: {combined_race_file}")
    print("\nCombined race distribution:")
    total_race = combined_race_df['count'].sum()
    for _, row in combined_race_df.iterrows():
        pct = 100 * row['count'] / total_race
        print(f"  {row['race_concept_name']}: {row['count']:,} ({pct:.1f}%)")

    # Combined ethnicity
    combined_ethnicity_df = pd.DataFrame([
        {'ethnicity_concept_name': ethnicity, 'count': count}
        for ethnicity, count in sorted(all_harmonized_ethnicity.items(), key=lambda x: -x[1])
    ])
    combined_ethnicity_file = f"{output_dir}/ALL_DATASETS_ethnicity_harmonized.csv"
    combined_ethnicity_df.to_csv(combined_ethnicity_file, index=False)
    print(f"\n✓ Combined ethnicity summary saved: {combined_ethnicity_file}")
    print("\nCombined ethnicity distribution:")
    total_ethnicity = combined_ethnicity_df['count'].sum()
    for _, row in combined_ethnicity_df.iterrows():
        pct = 100 * row['count'] / total_ethnicity
        print(f"  {row['ethnicity_concept_name']}: {row['count']:,} ({pct:.1f}%)")

    print("\n" + "="*80)
    print("Harmonization Complete!")
    print("="*80)
    print(f"\nAll harmonized files saved to: {output_dir}")

if __name__ == '__main__':
    main()
