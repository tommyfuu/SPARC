# Input data

SPARC reads one **bundle** per dataset: every patient's labs, vitals, demographics, hospital
stays, medications and follow-up. Put it in a folder named `<SITE>_CAIA` under a data root
and pass `-ds <SITE>_CAIA -vp <data root>` to `sparc train`. The `_CAIA` suffix tells the
loader to read a bundle.

## Formats

The folder holds the bundle in one of these formats:

| Format | Files | Notes |
|---|---|---|
| JSON | `DMP_ID_to_dicts_final.json` | The original SPARC format. |
| Pickle | `DMP_ID_to_dicts_final.pkl` | The same dict, pickled. Loads faster than JSON. |
| CSV | `patients.csv`, `labs.csv`, and optionally `vitals.csv`, `medications.csv`, `admissions.csv` | Tidy tables, described below. |
| Parquet | The same table names, ending in `.parquet` | Needs `pip install -e ".[parquet]"`. |

From Python, [`tables_to_bundle`](api.md) builds a bundle from pandas DataFrames and
`write_bundle` saves it. Keep one format per folder; if there are several, JSON is read first,
then pickle, then tables. `sparc convert SRC DST --format json|pickle|csv|parquet` converts
between them, and `sparc validate SRC` checks a bundle and prints a summary.

!!! warning "Normalization cache"
    The first run on a dataset z-scores the labs and vitals and caches the result in the
    dataset folder (`normalized_data_dict_*.pickle` and `normalization_factor_dict_*.pickle`).
    The cache is not refreshed when the data changes, so delete these files after editing the
    data. `sparc convert` and `sparc synthetic` delete them for you.

## Tables

Patient ids must be integers. Every `day` is a whole number of days from an anchor you
choose per patient, such as the first record; all tables for a patient must use the same
anchor. Instead of any `*_day` column you can give a `*_date` column (for example `date`,
`admit_date`) in `YYYY-MM-DD` form, optionally with a time; it becomes calendar days since
`index_date` in `patients`. Rows of patients who are not in `patients` are ignored, with a
warning.

### patients (required)

One row per patient.

| Column | Required | Meaning |
|---|---|---|
| `patient_id` | yes | Integer id. |
| `sex` | yes | `F`/`Female` or `M`/`Male` (or 1 and 0). Other values are encoded like `M`, as in the original bundle. |
| `age` | yes | Age in years at the first record. It is stored as a 5-year bin. |
| `fold` | no | 0–9 are cross-validation folds and 10 is the held-out test set. If the column is missing, each patient's fold is derived from its id, so it does not change when patients are added. |
| `last_contact_day` | for supervised tasks | Last day of follow-up. Supervised runs stop if a patient has none. |
| `death_day` | no | Day of death; leave empty if the patient is alive. |
| `index_date` | with dates | The date that is day 0. |

### labs (required) and vitals

One row per patient and day: `patient_id`, `day`, then one column per test, named as in
[Features and units](#features-and-units). Any subset of tests is fine. You can also use long
format, with columns `patient_id`, `day`, `test` and `value`.

Missing and non-numeric values (such as `<0.5`) are treated as missing. If there are several
rows for the same patient and day, the first non-missing value of each test is kept.

### medications

One row per administration: `patient_id`, `day`, `medication`, and optionally `end_day`
(which defaults to `day`). Names are generic drug names; they are stored in upper case and
matched against `-me`, for example `-me CISPLATIN`.

### admissions

One row per hospital stay, with `patient_id`, `admit_day` and `discharge_day`. Each ICU
episode is an extra row with `icu` set to 1 (the column defaults to 0), so a hospital stay that
included ICU time is two rows. Binary tasks use each patient's last stay in row order.

### What each task needs

| Task (`-o`) | Tables and columns used |
|---|---|
| `pretrain` | labs, vitals; admissions if present |
| `binary` | labs, vitals, admissions, `last_contact_day`, `death_day`. Predicts death within `-d` days of discharge from the last stay, from data during that stay. |
| `binary_icu_pre` | The same, with the last ICU episode. Predicts death within `-d` days of ICU admission, from data before it. |
| `neutropenia`, `febrile_neutropenia`, `anemia`, `AKI`, `Liver_toxicity`, each optionally with `_risk` | labs, vitals, medications, `last_contact_day`. Predicts the event after each infusion of the `-me` drugs. `_risk` trains on time to event. |

## Features and units

The parsimonious models (`-m parsimonious`), which are the ones in the paper, use these 28 labs and
7 vitals, plus sex and age. Use these units; the CBC differential must be absolute counts, not
percentages. For example, neutropenia is `Neut` < 1.0 (ANC < 1000/µL). `sparc validate` warns
when a test's median is unusual for its unit, such as temperature in °F.

| Group | Tests (unit) |
|---|---|
| CBC | `RBC` (10⁶/µL), `HGB` (g/dL), `HCT` (%), `MCV` (fL), `MCH` (pg), `MCHC` (g/dL), `RDW` (%), `WBC` (10³/µL), `Platelets` (10³/µL) |
| CBC differential | `Neut`, `Mono`, `Lymph`, `Baso`, `Eos` (all 10³/µL) |
| Metabolic | `Glucose` (mg/dL), `Calcium` (mg/dL), `Sodium` (mEq/L), `Potassium` (mEq/L), `CO2` (mEq/L), `Creatinine` (mg/dL), `Chloride` (mEq/L), `BUN` (mg/dL) |
| Liver | `Albumin` (g/dL), `Protein, Total` (g/dL), `ALP` (U/L), `ALT` (U/L), `AST` (U/L), `Bilirubin, Total` (mg/dL) |
| Vitals | `BMI` (kg/m²), `Blood Pressure BP Systolic` (mmHg), `Blood Pressure Diastolic` (mmHg), `O2` (%), `Pulse Heart Rate` (bpm), `Resp Rate` (breaths/min), `Temperature C` (°C) |

`-m practical` uses the same features without the CBC differential. The other names in
`ALL_INCLUDED_TESTS` and `VITALS` (`data/MSK_featurename_list.py`), such as tumour markers and
invasive blood pressure, are accepted but not used by either model.

## Bundle keys

The JSON and pickle files hold one dict. Its keys map patient id to:

| Key | Value |
|---|---|
| `DMP_ID_to_DATES_to_TESTSVALUES_DICT` | `{day: [46 lab values in ALL_INCLUDED_TESTS order, null if missing]}` |
| `DMP_ID_to_DATES_to_VITALSVALUES_DICT` | `{day: [9 vital values in VITALS order]}` |
| `DMP_ID_to_GENDER_dict_oh` | `[1]` female, `[0]` otherwise |
| `DMP_ID_to_firstlabdateagebinned_dict` | age // 5 |
| `DMP_ID_to_fold_num_dict` | fold, 0–10 |
| `DMP_ID_to_last_contact_date_dict_delta` | day |
| `DMP_ID_to_death_days_dict` | day (deceased patients only) |
| `DMP_ID_to_inpatient_dates_dict_delta`, `DMP_ID_to_icu_dates_dict_delta` | `[[admit_day, discharge_day], ...]` |
| `DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta` | `{day: [[DRUG, end_day], ...]}` |

All ten keys must be present, even if some are empty. Only patients with labs are loaded. The
MSK bundles also hold `DMP_ID_to_cutoff_day_dict` (the 2021 split for `-cp`, kept as a
`cutoff_day` column in tables) and `DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_delta_unfiltered`, which
only the regimen analysis in `results_vis/` uses and the tables do not carry.

## Other data sources

The in-house cohorts (`-ds MSK`, `MSK_timeshift`, `MIMICIV_hosp`, `stanfordEHRSHOT`) are read by
their own loaders in `data/dataset_loading.py`, from the lab's preprocessed files under
`sparc_paths.DATA_ROOT`. Output of other pipelines can be written as the tables above, or as the
JSON bundle, and loaded as a `*_CAIA` bundle.
