# INSIGHT preprocessing (PCORnet CDM to SPARC)

Converts INSIGHT PCORnet CDM CSV exports into the
SPARC pickles loaded by `external_validation_insight/insight_data.py`
(`load_dataset(dataset='insight' | 'insight_pooled')`). Six pipelines (labs, vitals, medications, admissions,
stage, demographics), each split into `load.py`, `clean.py` and `build_dict.py`.

## Paths

Default paths in `sparc_paths.py` are placeholders; set:

```bash
export SPARC_INSIGHT_RAW_DIR=/path/to/insight-data-raw      # PCORnet CDM CSVs
export SPARC_INSIGHT_DATA_DIR=/path/to/insight-processed    # preprocessing output
export SPARC_INSIGHT_RESULTS_DIR=/path/to/insight-results   # evaluation outputs
```

## Running

```bash
cd external_validation_insight/preprocessing
python preflight_check.py                                            # checks input files and columns
nohup bash run_preprocessing_parallel.sh > preprocessing.log 2>&1 &  # or run_preprocessing.sh (sequential)
```

The run scripts call `run_preprocessing_split.py --run-preprocessing` (all pipelines, then split by site),
`make_views.py` and `create_splits.py`. Output:

- `<out>/{A..F}/`: per-site pickles; `<out>/overall/`: all sites merged
- `<out>/insight_stratified/`, `<out>/insight_pooled_stratified/`: 1:1:1 split JSONs
- `<out>-fullsite/`: no split, every patient scored (zero-shot evaluation)

Single pipelines: `python run_preprocessing.py --pipelines labs vitals meds admissions stage demographics`
(`demographics` alone runs in seconds). Needs ~120-200 GB RAM; `lab_result_cm.csv` alone is ~50 GB.

## Configuration (`config.py`)

| Environment variable | Default |
|---|---|
| `INSIGHT_DATA_DIR` (input CSVs) | `sparc_paths.INSIGHT_RAW_DIR` |
| `INSIGHT_OUTPUT_DIR` | `sparc_paths.INSIGHT_DATA_DIR` |
| `INSIGHT_DEMOGRAPHICS_FILE` | `demographics.csv` or `demographic.csv`, whichever exists |
| `RXNORM_CACHE_FILE` | bundled `code_mapping/rxnorm_cache.json` (no internet needed for cached CUIs) |

LOINC maps, accepted units, vital value ranges and PCORnet code maps are also in `config.py`.

## Inputs

`enrollment.csv`, `demographics.csv`, `lab_result_cm.csv`, `vital.csv`, `obs_clin.csv`, `prescribing.csv`,
`med_admin.csv` (optional), `encounter.csv`, `diagnosis.csv`. Large files are read in 500K-row chunks with early
filters (LOINC, encounter type, ICD-10 prefix).

## Outputs

Date keys are day offsets from each patient's first lab date (enrollment start date if no labs).

| File | Structure |
|---|---|
| `ssid_to_date_to_labs_dict.pickle` | `{ssid: {offset: {test_name: value}}}` |
| `ssid_to_date_to_vitals_dict.pickle` | `{ssid: {offset: [BMI, SBP, DBP, inv_SBP, inv_DBP, O2, HR, RR, Temp_C]}}` |
| `ssid_to_date_to_oral_meds_dict.pickle` | `{ssid: {start_offset: [[name, '', '', end_offset], ...]}}` |
| `ssid_to_date_to_nonoral_meds_dict.pickle` | `{ssid: {start_offset: [[name, '', '', end_offset, dose, unit, nan, qty, freq, duration], ...]}}` |
| `ssid_to_hospital_adm_dsch_dict.pickle` | `{ssid: [(admit_offset, discharge_offset), ...]}` |
| `ssid_to_date_to_STAGE_dict.pickle` | `{ssid: {offset: 4.0}}` |
| `ssid_to_demographics_dict.pickle` | `{ssid: {age, age_binned_5, sex, race, ethnicity}}` (static) |

## Pipeline notes

- **Labs:** LOINC codes mapped to SPARC test names; per-test unit checks; the 9999999 sentinel value dropped; one value per patient, test and day.
- **Vitals:** BMI and blood pressure from `vital.csv`; O2 saturation, heart rate, respiratory rate and temperature
  (F converted to C) from `obs_clin.csv`, with implausible values dropped. Invasive BP is not recorded in INSIGHT and is always NaN.
- **Medications:** RxNorm CUIs resolved to drug names; prescriptions split into oral and non-oral by route;
  `med_admin.csv` administrations are added to the non-oral dict. Only the drug name is used by SPARC.
- **Admissions:** inpatient encounters (`enc_type` IP or EI); missing discharge date treated as same day. No ICU
  encounters in INSIGHT.
- **Stage:** Stage 4 only, from metastasis ICD-10 codes (C77-C79, C800); stages 1-3 are not inferable from ICD-10.
- **Demographics:** PCORnet codes for sex, race and `hispanic` decoded via `config.py`; unmapped codes become
  "Unknown". Age is taken at the earliest enrollment date; `age_binned_5 = int(age // 5)`, matching MSK.
