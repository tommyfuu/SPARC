"""
Generate race and ethnicity summary CSVs for MIMIC-IV cancer patients.
Creates two CSV files in the race_ethnicity output directory:
- MIMICIV_hosp_race.csv
- MIMICIV_hosp_ethnicity.csv
"""

# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
import os
import pandas as pd
import pickle
from collections import Counter
from config import MIMIC_RAW_PATH, OUTPUT_PATH

def map_race_to_standard(race_value):
    """Map MIMIC-IV race values to standard OMOP race concepts."""
    if pd.isna(race_value):
        return 'No matching concept'

    race_upper = str(race_value).upper()

    # Standard mappings
    if 'WHITE' in race_upper:
        return 'White'
    elif 'BLACK' in race_upper or 'AFRICAN' in race_upper or 'CAPE VERDEAN' in race_upper or 'CARIBBEAN' in race_upper:
        return 'Black or African American'
    elif 'ASIAN' in race_upper:
        return 'Asian'
    elif 'AMERICAN INDIAN' in race_upper or 'ALASKA NATIVE' in race_upper:
        return 'American Indian or Alaska Native'
    elif 'HAWAIIAN' in race_upper or 'PACIFIC ISLANDER' in race_upper:
        return 'Native Hawaiian or Other Pacific Islander'
    elif 'MULTIPLE' in race_upper:
        return 'Multiple race'
    elif 'UNABLE TO OBTAIN' in race_upper or 'DECLINED' in race_upper or 'UNKNOWN' in race_upper:
        return 'Refuse to answer'
    elif 'HISPANIC' in race_upper or 'LATINO' in race_upper or 'PORTUGUESE' in race_upper or 'SOUTH AMERICAN' in race_upper:
        # These are ethnicity values, but in MIMIC-IV they're stored in the race field
        # Map to "Other" for race categorization
        return 'Other'
    else:
        return 'No matching concept'

def is_hispanic_latino(race_value):
    """Determine if race value indicates Hispanic/Latino ethnicity."""
    if pd.isna(race_value):
        return False

    race_upper = str(race_value).upper()
    hispanic_keywords = ['HISPANIC', 'LATINO', 'LATINA', 'PORTUGUESE', 'BRAZILIAN',
                         'SOUTH AMERICAN', 'GUATEMALAN', 'SALVADORAN', 'MEXICAN',
                         'COLUMBIAN', 'HONDURAN', 'CUBAN', 'DOMINICAN', 'PUERTO RICAN',
                         'CENTRAL AMERICAN']

    return any(keyword in race_upper for keyword in hispanic_keywords)

def main():
    print("="*80)
    print("Generating Race/Ethnicity Summary CSVs for MIMIC-IV")
    print("="*80)

    # Define output directory. Override with SPARC_RACE_ETHNICITY_DIR (the
    # per-dataset folder) so this runs on hosts with a different mount.
    output_dir = os.environ.get(
        'SPARC_RACE_ETHNICITY_DIR',
        sparc_paths.RESULTS_ROOT + '/results_vis/cohort_summaries/race_ethnicity/MIMICIV_hosp',
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output directory: {output_dir}")

    # Load cancer patients
    print("\n1. Loading cancer patient list...")
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f"   Cancer patients: {len(cancer_subjects):,}")

    # Load admissions
    print("\n2. Loading admissions.csv.gz...")
    admissions_df = pd.read_csv(f'{MIMIC_RAW_PATH}/admissions.csv.gz', compression='gzip')

    # Filter to cancer patients
    cancer_admissions = admissions_df[admissions_df['subject_id'].isin(cancer_subjects)].copy()
    print(f"   Cancer admissions: {len(cancer_admissions):,}")

    # Get unique subjects with their race information
    # Use the first admission for each patient to get their race
    patient_race = cancer_admissions.groupby('subject_id')['race'].first()
    print(f"   Unique cancer patients with race data: {len(patient_race):,}")

    # Process Race
    print("\n3. Processing race data...")
    race_standard = patient_race.apply(map_race_to_standard)
    race_counts = Counter(race_standard)

    # Create race DataFrame
    race_df = pd.DataFrame([
        {'race_concept_name': race, 'count': count}
        for race, count in sorted(race_counts.items(), key=lambda x: -x[1])
    ])

    # Save race CSV
    race_output = f'{output_dir}/MIMICIV_hosp_race.csv'
    race_df.to_csv(race_output, index=False)
    print(f"   Saved: {race_output}")
    print(f"   Race distribution:")
    for _, row in race_df.iterrows():
        print(f"     {row['race_concept_name']}: {row['count']:,}")

    # Process Ethnicity
    print("\n4. Processing ethnicity data...")
    ethnicity_status = patient_race.apply(lambda x: 'Hispanic or Latino' if is_hispanic_latino(x) else 'Not Hispanic or Latino')
    ethnicity_counts = Counter(ethnicity_status)

    # Create ethnicity DataFrame
    ethnicity_df = pd.DataFrame([
        {'ethnicity_concept_name': ethnicity, 'count': count}
        for ethnicity, count in sorted(ethnicity_counts.items(), key=lambda x: -x[1])
    ])

    # Save ethnicity CSV
    ethnicity_output = f'{output_dir}/MIMICIV_hosp_ethnicity.csv'
    ethnicity_df.to_csv(ethnicity_output, index=False)
    print(f"   Saved: {ethnicity_output}")
    print(f"   Ethnicity distribution:")
    for _, row in ethnicity_df.iterrows():
        print(f"     {row['ethnicity_concept_name']}: {row['count']:,}")

    # Patient-level export, so counts can be re-aggregated against the actual
    # modelling cohort (cancer_subjects is slightly wider than load_dataset()).
    print("\n5. Writing patient-level race/ethnicity map...")
    patient_level = {
        int(subject_id): (race_standard[subject_id], ethnicity_status[subject_id])
        for subject_id in patient_race.index
    }
    pickle_output = f'{output_dir}/subject_id_to_race_ethnicity_MIMICIV_hosp.pickle'
    with open(pickle_output, 'wb') as f:
        pickle.dump(patient_level, f)
    print(f"   Saved: {pickle_output}  ({len(patient_level):,} patients)")

    print("\n" + "="*80)
    print("Race/Ethnicity Summary Generation Complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {race_output}")
    print(f"  - {ethnicity_output}")

if __name__ == '__main__':
    main()
