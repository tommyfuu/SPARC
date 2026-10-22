"""Paths for Stanford EHRSHOT preprocessing."""

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

REPO_ROOT = _R

# override with environment variables if the data is mounted elsewhere:
#     export SPARC_EHRSHOT_ASSETS=/path/to/EHRSHOT_ASSETS
#     export SPARC_EHRSHOT_OUTPUT_PATH=/path/to/SPARC_data/stanfordEHRSHOT
EHRSHOT_ASSETS_PATH = sparc_paths.EHRSHOT_ASSETS_DIR
OUTPUT_PATH = os.environ.get(
    'SPARC_EHRSHOT_OUTPUT_PATH',
    os.path.join(sparc_paths.DATA_ROOT, 'stanfordEHRSHOT'),
)

# raw EHRSHOT release (EHRSHOT_ASSETS layout)
EHRSHOT_CSV = os.path.join(EHRSHOT_ASSETS_PATH, 'data', 'ehrshot.csv')
CONCEPT_CSV = os.path.join(EHRSHOT_ASSETS_PATH, 'femr', 'logs', 'omop_dir', 'concept.csv')
# LOINC table from https://loinc.org/downloads/, unpacked into data/LoincTable/
LOINC_CSV = os.path.join(EHRSHOT_ASSETS_PATH, 'data', 'LoincTable', 'Loinc.csv')

# intermediates written by check_cancer.py
ICD_CODES_CSV = os.path.join(OUTPUT_PATH, 'icd_codes.csv')
CANCER_PATIENTS_CSV = os.path.join(OUTPUT_PATH, 'cancer_patients_ehrshot.csv')

# cohort: patients dictionarify_new.py kept (>=1 lab)
COHORT_PICKLE = os.path.join(
    OUTPUT_PATH, 'stanford_ehrshots_age_at_first_lab_test_dict_binned5.pickle')

# read by dem_table_3_harmonize_race_ethnicity.py
RACE_ETHNICITY_OUTPUT_PATH = os.path.join(sparc_paths.RACE_ETHNICITY_DIR, 'EHRSHOT')

# check_cancer_types_insight.py (shared ICD-10 rules)
CANCER_TYPES_CODE_DIR = os.path.join(REPO_ROOT, 'cohort_summaries', 'cancer_types')

# create the output dir, but stay importable if it isn't writable
try:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
except OSError as _e:
    print(f"Warning: could not create OUTPUT_PATH {OUTPUT_PATH!r}: {_e}")
