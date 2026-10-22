# EHRSHOT preprocessing

Builds the Stanford EHRSHOT cancer cohort (`stanfordEHRSHOT`) used for SPARC
external validation.

## Prerequisites

EHRSHOT requires signing the Stanford AIMI data use agreement:
https://redivis.com/datasets/53gc-8rhx41kgt

Download `EHRSHOT_ASSETS`. The steps below read:

- `data/ehrshot.csv`: the flattened OMOP event table
- `femr/logs/omop_dir/concept.csv`: the OMOP concept table, used for drug codes

They also need the LOINC table (https://loinc.org/downloads/), unpacked so that
`Loinc.csv` sits at `EHRSHOT_ASSETS/data/LoincTable/Loinc.csv`.

## Configuration

`config.py` takes both paths from the environment, falling back to `sparc_paths`:

```bash
export SPARC_EHRSHOT_ASSETS=/path/to/EHRSHOT_ASSETS
export SPARC_EHRSHOT_OUTPUT_PATH=/path/to/SPARC_data/stanfordEHRSHOT
```

Without `SPARC_EHRSHOT_OUTPUT_PATH`, outputs go to `$SPARC_DATA_PATH/stanfordEHRSHOT`,
which is where `data/dataset_loading.py` looks for them.

## Running

Run in order; each step reads the previous one's output.

```bash
python check_cancer.py                  # cancer patients from ICD-O-3 condition codes -> cancer_patients_ehrshot.csv
python dictionarify_new.py              # labs, vitals, demographics, death/contact, admissions, drugs -> stanford_ehrshots_*.pickle
python check_cancer_type_ehrshot.py     # cancer type per patient -> DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_stanfordEHRSHOT.pickle
```

Things to know:

- **check_cancer.py** reads the whole 3.2 GB `ehrshot.csv` into memory.
- **dictionarify_new.py** runs top to bottom like a notebook. It defines the
  546-patient cohort: patients with at least one lab in `ALL_INCLUDED_TESTS`.
  The two scripts after it read that cohort from
  `stanford_ehrshots_age_at_first_lab_test_dict_binned5.pickle`.
- **check_cancer_type_ehrshot.py** reuses the ICD-10 rule table in
  `cohort_summaries/cancer_types/check_cancer_types_insight.py`. Run
  `--self-test` to check it, or `--help` for the in-situ, secondary and
  `others` options.
- EHRSHOT has no ICU table. `dictionarify_new.py` stores ER-to-inpatient
  (`Visit/ERIP`) stays in its `icu_*` pickles as a proxy, and `dataset_loading.py`
  refuses the ICU tasks for this dataset.

## Race and ethnicity summaries

```bash
python check_race_ethnicity_ehrshot.py
```

Writes `EHRSHOT_race.csv` and `EHRSHOT_ethnicity.csv` under
`$SPARC_RACE_ETHNICITY_DIR/EHRSHOT/`, where
`cohort_summaries/demographic_tables/dem_table_3_harmonize_race_ethnicity.py`
reads them. Counts of 10 or fewer are suppressed.
