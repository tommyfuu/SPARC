# MIMIC-IV preprocessing

Builds the MIMIC-IV hospital cohort used for SPARC external validation.

## Prerequisites

MIMIC-IV v3.1 requires credentialed PhysioNet access (CITI training + a signed
data use agreement): https://physionet.org/content/mimiciv/3.1/

Download the `hosp` module. The steps below read `patients`, `admissions`,
`diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems`, `emar`,
`emar_detail` and `omr`.

## Configuration

`config.py` takes both paths from the environment, falling back to `sparc_paths`:

```bash
export SPARC_MIMIC_RAW_PATH=/path/to/mimic_iv_3.1_mult/hosp
export SPARC_MIMIC_OUTPUT_PATH=/path/to/SPARC_data/MIMICIV_hosp
```

## Running

Run in order; each step reads the previous one's output.

```bash
python 1_extract_cancer_patients.py   # cancer cohort from ICD codes
python 2_process_demographics.py      # age, gender, death dates
python 3_process_labs.py              # the 46 tests in ALL_INCLUDED_TESTS
python 4_process_medications.py       # medications with dosages
python 4b_process_vitals.py           # vitals from the omr table
python 5_process_admissions.py        # admissions and discharges
python 6_create_splits.py             # 60/20/20 train/val/test split
```

Two things to know:

- **4b** only recovers BMI and blood pressure. O2 saturation, heart rate,
  respiratory rate and temperature live in the ICU `chartevents` module, which
  the `hosp` download does not include, so those four stay `NA` for everyone.

- **6** is deterministic but does not reproduce the split the released models
  were trained on — roughly 1% of patients land in a different partition. Do not
  re-run it unless you intend to retrain.

## Race and ethnicity summaries

```bash
python generate_race_ethnicity_summaries.py
```

Writes `MIMICIV_hosp_race.csv` and `MIMICIV_hosp_ethnicity.csv` under
`cohort_summaries/race_ethnicity/MIMICIV_hosp/`, not to `SPARC_MIMIC_OUTPUT_PATH`.

