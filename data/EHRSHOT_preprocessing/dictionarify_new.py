import os
import pandas as pd
from config import OUTPUT_PATH, CANCER_PATIENTS_CSV, LOINC_CSV, CONCEPT_CSV

# read data
data = pd.read_csv(CANCER_PATIENTS_CSV)

data['omop_table'].value_counts()
data_measurements = data[data['omop_table'] == 'measurement']
data_observation = data[data['omop_table'] == 'observation']

## split code into code_cat / code_num
data_measurements['code_cat'] = data_measurements['code'].apply(lambda x: x.split('/')[0])
data_measurements['code_num'] = data_measurements['code'].apply(lambda x: x.split('/')[1])


## data_measurement_labs
data_measurements_labs = data_measurements[data_measurements['code_cat'] == 'LOINC']
data_measurements_labs['code_num'].value_counts()

all_included_labs = data_measurements_labs['code_num'].unique()

# fetch from the Loinc files for mapping
loinc = pd.read_csv(LOINC_CSV)
loinc_codes_to_description_dict = dict(zip(loinc['LOINC_NUM'], loinc['LONG_COMMON_NAME']))
data_measurements_labs['description'] = data_measurements_labs['code_num'].apply(lambda x: loinc_codes_to_description_dict[x] if x in loinc_codes_to_description_dict else None)

# check for loinc codes for WBC
loinc_wbc = loinc[loinc["COMPONENT"].str.contains("Leukocytes")]
from LOINC_to_test_dict import *
all_labs_vitals = list(LOINC_TO_ALL_INCLUDED_TESTS.keys()) + list(LOINC_TO_VITALS_NAMES.keys())

# check those features included
data_measurement_labs_included = data_measurements_labs[data_measurements_labs['code_num'].isin(all_labs_vitals)]

# check the number of patients with each of the basic lab and vitals
data_measurement_labs_included['code_num'].value_counts()

# get the lab test names as well as value counts
all_mappings = {**LOINC_TO_ALL_INCLUDED_TESTS, **LOINC_TO_VITALS_NAMES}
code_counts = data_measurement_labs_included['code_num'].value_counts()
for code, count in code_counts.items():
    test_name = all_mappings.get(code, 'NOT MAPPED')
    print(f"{code:15s} -> {test_name:30s}: {count:6,} records")


# patient_id -> date -> {test name: value}, for tests in ALL_INCLUDED_TESTS
patient_id_to_date_to_all_included_test_values_dict = {}
from MSK_featurename_list import *
for _, row in data_measurement_labs_included.iterrows():
    patient_id = row['patient_id']
    date = row['start'].split(' ')[0]
    code_num = row['code_num']

    test_name = LOINC_TO_ALL_INCLUDED_TESTS.get(code_num)
    if test_name in ALL_INCLUDED_TESTS:
        if patient_id not in patient_id_to_date_to_all_included_test_values_dict:
            patient_id_to_date_to_all_included_test_values_dict[patient_id] = {}
        if date not in patient_id_to_date_to_all_included_test_values_dict[patient_id]:
            patient_id_to_date_to_all_included_test_values_dict[patient_id][date] = {}
        patient_id_to_date_to_all_included_test_values_dict[patient_id][date][test_name] = row['value']

patient_id_to_date_to_all_included_vitals_values_dict = {}
for _, row in data_measurement_labs_included.iterrows():
    patient_id = row['patient_id']
    date = row['start'].split(' ')[0]
    code_num = row['code_num']

    test_name = LOINC_TO_VITALS_NAMES.get(code_num)
    if test_name in VITALS:
        if patient_id not in patient_id_to_date_to_all_included_vitals_values_dict:
            patient_id_to_date_to_all_included_vitals_values_dict[patient_id] = {}
        if date not in patient_id_to_date_to_all_included_vitals_values_dict[patient_id]:
            patient_id_to_date_to_all_included_vitals_values_dict[patient_id][date] = {}
        patient_id_to_date_to_all_included_vitals_values_dict[patient_id][date][test_name] = row['value']

# save the dictionary
import pickle
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_cancer_patient_id_to_date_to_all_included_test_values_dict.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_date_to_all_included_test_values_dict, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_cancer_patient_id_to_date_to_all_included_vitals_values_dict.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_date_to_all_included_vitals_values_dict, f)

## align vitals to the lab dates of each patient
patient_id_to_date_to_all_included_vitals_values_dict_matched = {}

for patient_id, date_vals in patient_id_to_date_to_all_included_test_values_dict.items():
    patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id] = {}
    if patient_id not in patient_id_to_date_to_all_included_vitals_values_dict:
        patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id] = {date: {} for date in date_vals.keys()}
    else:
        for date in date_vals.keys():
            if date in patient_id_to_date_to_all_included_vitals_values_dict[patient_id]:
                patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id][date] = patient_id_to_date_to_all_included_vitals_values_dict[patient_id][date]
            else:
                patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id][date] = {}

# patients and records per vital
missingness_vitals_to_patient_ids = {}
for vital in VITALS:
    num_patients = 0
    num_instances = 0
    for patient_id in patient_id_to_date_to_all_included_vitals_values_dict_matched.keys():
        current_exists = False
        for date in patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id].keys():
            if vital in patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id][date]:
                num_instances += 1
                current_exists = True
        if current_exists:
            num_patients += 1
        else:
            if vital not in missingness_vitals_to_patient_ids:
                missingness_vitals_to_patient_ids[vital] = [patient_id]
            else:
                missingness_vitals_to_patient_ids[vital].append(patient_id)
    print(vital, num_patients, num_instances)

# find patients that miss all vitals
missing_all_vitals = []
for patient_id in patient_id_to_date_to_all_included_vitals_values_dict_matched.keys():
    current_missing = True
    for date in patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id].keys():
        if patient_id_to_date_to_all_included_vitals_values_dict_matched[patient_id][date]:
            current_missing = False
            break
    if current_missing:
        missing_all_vitals.append(patient_id)

import numpy as np
num_dates_per_patient = [len(dates) for dates in patient_id_to_date_to_all_included_vitals_values_dict_matched.values()]
np.percentile(num_dates_per_patient, [5, 25, 50, 75, 95])

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_cancer_patient_id_to_date_to_all_included_vitals_values_dict_matched.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_date_to_all_included_vitals_values_dict_matched, f)


## load back the all lab dict and the matched vital dict
import pickle
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_cancer_patient_id_to_date_to_all_included_test_values_dict.pickle'), 'rb') as f:
    patient_id_to_date_to_all_included_test_values_dict = pickle.load(f)
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_cancer_patient_id_to_date_to_all_included_vitals_values_dict_matched.pickle'), 'rb') as f:
    patient_id_to_date_to_all_included_vitals_values_dict_matched = pickle.load(f)

# check for demographics to get birthdate, gender, and race (as a proxy for genetic ancestry)
data_demographics = data[data['omop_table'] == 'person']
birthdate_df = data_demographics[data_demographics['code']=='SNOMED/3950001']
# gender: code contains 'Gender'
gender_df = data_demographics[data_demographics['code'].str.contains('Gender')]
# race: code contains 'Race'
race_df = data_demographics[data_demographics['code'].str.contains('Race')]
def one_hot_encode_dict(static_dict, feature_reduced = '', unique_vals = []):
    '''one-hot encoding with the ability to reduce a feature out'''
    if unique_vals == []:
        unique_vals = list(set(static_dict.values()))
    print(unique_vals)
    if feature_reduced != '' and feature_reduced in unique_vals:
        unique_vals.remove(feature_reduced)
        print(unique_vals)
    indices = [unique_vals.index(val) if val in unique_vals else -1 for val in static_dict.values()]
    static_dict_out = {k: [0]*len(unique_vals) for k in static_dict}
    for i, k in enumerate(static_dict.keys()):
        if indices[i]!=-1:
            static_dict_out[k][indices[i]] = 1
    return static_dict_out, unique_vals

# turn gender into one-hot encoding, with Gender/M as the reduced feature
gender_dict = dict(gender_df[['patient_id', 'code']].values)
gender_unique_values = ['Gender/M', 'Gender/F']
patient_id_to_gender_dict_oh = one_hot_encode_dict(gender_dict, feature_reduced='Gender/M', unique_vals=gender_unique_values)

# genetic ancestry left out: Race/1..5 don't map cleanly onto EUR/ADM/EAS/AFR/SAS/NAM

# age at first lab test (MSK uses age at sequencing), in 5-year bins
birthdate_df['start'] = pd.to_datetime(birthdate_df['start'])
birthdate_df['start'] = birthdate_df['start'].dt.strftime('%Y-%m-%d')
birthdate_dict = dict(birthdate_df[['patient_id', 'start']].values)

# find the first lab test date for each patient
first_lab_test_date_dict = {}
for patient_id, date_to_values in patient_id_to_date_to_all_included_test_values_dict.items():
    first_lab_test_date_dict[patient_id] = sorted(date_to_values.keys())[0]

# find the age at first lab test for each patient
from datetime import datetime
from datetime import date
def calculate_days_elapsed(first_date, second_date):
    '''calculate age at test given the date of last contact and the birthdate'''
    first_date = datetime.strptime(first_date, "%Y-%m-%d").date()
    second_date = datetime.strptime(second_date, "%Y-%m-%d").date()
    delta = second_date-first_date
    return delta.days
def calculate_age(birth_date, test_date):
    '''calculate age at test given the date of a test and the birthdate'''
    birth_date = datetime.strptime(birth_date, "%Y-%m-%d").date()
    test_date = datetime.strptime(test_date, "%Y-%m-%d").date()
    age = test_date.year - birth_date.year - ((test_date.month, test_date.day) < (birth_date.month, birth_date.day))
    return min(90, age) # Cap at 90

age_at_first_lab_test_dict = {patient_id: calculate_age(birthdate_dict[patient_id], first_lab_test_date_dict[patient_id]) for patient_id in first_lab_test_date_dict.keys()}
age_at_first_lab_test_dict_binned5 = {patient_id: age//5 for patient_id, age in age_at_first_lab_test_dict.items()}


# patient_id -> days since first lab -> values in ALL_INCLUDED_TESTS order ('NAN' if missing)
mapped_patient_id_to_date_to_all_included_lab_values_list_dict = {}
for patient_id, date_to_values in patient_id_to_date_to_all_included_test_values_dict.items():
    mapped_patient_id_to_date_to_all_included_lab_values_list_dict[patient_id] = {}
    for date, values in date_to_values.items():
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        date_since_first_lab_test = calculate_days_elapsed(first_lab_test_date, date)
        mapped_patient_id_to_date_to_all_included_lab_values_list_dict[patient_id][date_since_first_lab_test] = [values[feature] if feature in values else 'NAN' for feature in ALL_INCLUDED_TESTS]

# Check how many patients have vitals but no lab data
patients_with_vitals = set(patient_id_to_date_to_all_included_vitals_values_dict_matched.keys())
patients_with_labs = set(first_lab_test_date_dict.keys())
patients_vitals_only = patients_with_vitals - patients_with_labs

print(f"Patients with vitals only (no labs): {len(patients_vitals_only)}")
print(f"Total patients with vitals: {len(patients_with_vitals)}")
print(f"Patients with labs: {len(patients_with_labs)}")

# same for vitals, in VITALS order
mapped_patient_id_to_date_to_vitals_values_list_dict = {}
for patient_id, date_to_values in patient_id_to_date_to_all_included_vitals_values_dict_matched.items():
    # no first lab date to anchor to
    if patient_id not in first_lab_test_date_dict:
        continue

    mapped_patient_id_to_date_to_vitals_values_list_dict[patient_id] = {}
    for date, values in date_to_values.items():
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        date_since_first_lab_test = calculate_days_elapsed(first_lab_test_date, date)
        mapped_patient_id_to_date_to_vitals_values_list_dict[patient_id][date_since_first_lab_test] = [values[feature] if feature in values else 'NAN' for feature in VITALS]

# save the dictionaries
import pickle
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_mapped_cancer_patient_id_to_date_to_all_included_lab_values_list_dict.pickle'), 'wb') as f:
    pickle.dump(mapped_patient_id_to_date_to_all_included_lab_values_list_dict, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_mapped_cancer_patient_id_to_date_to_vitals_values_list_dict.pickle'), 'wb') as f:
    pickle.dump(mapped_patient_id_to_date_to_vitals_values_list_dict, f)

# save patient_id_to_gender_dict_oh
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_gender_dict_oh.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_gender_dict_oh, f)

# save age_at_first_lab_test_dict_binned5
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_age_at_first_lab_test_dict_binned5.pickle'), 'wb') as f:
    pickle.dump(age_at_first_lab_test_dict_binned5, f)


## death, last contact, admission/discharge dates relative to first lab test

import pickle
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_mapped_cancer_patient_id_to_date_to_all_included_lab_values_list_dict.pickle'), 'rb') as f:
    mapped_patient_id_to_date_to_all_included_lab_values_list_dict = pickle.load(f)

## Filter data to only include patients with lab data
data = data[data['patient_id'].isin(mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys())]

data_death = data[data['omop_table'] == 'death']
data_death['start'] = pd.to_datetime(data_death['start'])
data_death['start'] = data_death['start'].dt.strftime('%Y-%m-%d')
patient_id_to_death_date_dict = dict(data_death[['patient_id', 'start']].values)

patient_id_to_last_contact_date_dict = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    patient_data = data[data['patient_id'] == patient_id]
    patient_data['start'] = pd.to_datetime(patient_data['start'])
    last_contact_date = patient_data['start'].max().strftime('%Y-%m-%d')
    patient_id_to_last_contact_date_dict[patient_id] = last_contact_date


# Check visit_occurrence table
data_visits = data[data['omop_table'] == 'visit_occurrence']

print(f"Total visit records: {len(data_visits)}")
print("\nColumns available:")
print(data_visits.columns.tolist())

print("\nSample visit records:")
print(data_visits.head())

print("\nVisit type codes:")
print(data_visits['code'].value_counts().head(20))


patient_id_to_ip_admission_discharge_dates_dict = {}
# inpatient stays
data_visits = data[data['omop_table'] == 'visit_occurrence']
data_inpatient = data_visits[data_visits['code'].isin(['Visit/IP', 'Visit/ERIP'])]

patient_id_to_ip_admission_discharge_dates_dict = {}

for _, row in data_inpatient.iterrows():
    patient_id = row['patient_id']
    admission_date = row['start'].split(' ')[0] if pd.notna(row['start']) else None
    discharge_date = row['end'].split(' ')[0] if pd.notna(row['end']) else None
    visit_id = row['visit_id']

    if patient_id not in patient_id_to_ip_admission_discharge_dates_dict:
        patient_id_to_ip_admission_discharge_dates_dict[patient_id] = []

    if admission_date:
        patient_id_to_ip_admission_discharge_dates_dict[patient_id].append({
            'admission': admission_date,
            'discharge': discharge_date,
            'visit_id': visit_id
        })

for patient_id in patient_id_to_ip_admission_discharge_dates_dict:
    patient_id_to_ip_admission_discharge_dates_dict[patient_id].sort(
        key=lambda x: x['admission']
    )

print(f"Total inpatient visits: {len(data_inpatient)}")
print(f"Unique patients with hospitalizations: {len(patient_id_to_ip_admission_discharge_dates_dict)}")
print(f"Patients in cancer cohort: {len(patient_id_to_date_to_all_included_test_values_dict)}")


# Build patient_id_to_icu_admission_discharge_dates_dict
# EHRSHOT has no ICU table; ER-to-inpatient (ERIP) stays stand in
patient_id_to_icu_admission_discharge_dates_dict = {}

data_visits = data[data['omop_table'] == 'visit_occurrence']
data_erip = data_visits[data_visits['code'] == 'Visit/ERIP']

for _, row in data_erip.iterrows():
    patient_id = row['patient_id']
    admission_date = row['start'].split(' ')[0] if pd.notna(row['start']) else None
    discharge_date = row['end'].split(' ')[0] if pd.notna(row['end']) else None
    visit_id = row['visit_id']

    if patient_id not in patient_id_to_icu_admission_discharge_dates_dict:
        patient_id_to_icu_admission_discharge_dates_dict[patient_id] = []

    if admission_date:
        patient_id_to_icu_admission_discharge_dates_dict[patient_id].append({
            'admission': admission_date,
            'discharge': discharge_date,
            'visit_id': visit_id
        })

for patient_id in patient_id_to_icu_admission_discharge_dates_dict:
    patient_id_to_icu_admission_discharge_dates_dict[patient_id].sort(
        key=lambda x: x['admission']
    )

print(f"Unique patients with ERIP visits (proxy for ICU/severe illness): {len(patient_id_to_icu_admission_discharge_dates_dict)}")
print(f"Total ERIP visits: {sum(len(v) for v in patient_id_to_icu_admission_discharge_dates_dict.values())}")

# days from first lab test to each event
patient_id_to_death_date_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_death_date_dict:
        death_date = patient_id_to_death_date_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_death_date_dict_delta[patient_id] = calculate_days_elapsed(first_lab_test_date, death_date)
    else:
        patient_id_to_death_date_dict_delta[patient_id] = None

patient_id_to_last_contact_date_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_last_contact_date_dict:
        last_contact_date = patient_id_to_last_contact_date_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_last_contact_date_dict_delta[patient_id] = calculate_days_elapsed(first_lab_test_date, last_contact_date)
    else:
        patient_id_to_last_contact_date_dict_delta[patient_id] = None

patient_id_to_ip_admission_dates_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_ip_admission_discharge_dates_dict:
        ip_admissions = patient_id_to_ip_admission_discharge_dates_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_ip_admission_dates_dict_delta[patient_id] = [calculate_days_elapsed(first_lab_test_date, admission['admission']) for admission in ip_admissions]
    else:
        patient_id_to_ip_admission_dates_dict_delta[patient_id] = None

patient_id_to_ip_discharge_dates_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_ip_admission_discharge_dates_dict:
        ip_discharges = patient_id_to_ip_admission_discharge_dates_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_ip_discharge_dates_dict_delta[patient_id] = [
            calculate_days_elapsed(first_lab_test_date, discharge['discharge'])
            if discharge['discharge'] is not None else None
            for discharge in ip_discharges
        ]
    else:
        patient_id_to_ip_discharge_dates_dict_delta[patient_id] = None

patient_id_to_icu_admission_dates_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_icu_admission_discharge_dates_dict:
        icu_admissions = patient_id_to_icu_admission_discharge_dates_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_icu_admission_dates_dict_delta[patient_id] = [calculate_days_elapsed(first_lab_test_date, admission['admission']) for admission in icu_admissions]
    else:
        patient_id_to_icu_admission_dates_dict_delta[patient_id] = None
patient_id_to_icu_discharge_dates_dict_delta = {}
for patient_id in mapped_patient_id_to_date_to_all_included_lab_values_list_dict.keys():
    if patient_id in patient_id_to_icu_admission_discharge_dates_dict:
        icu_discharges = patient_id_to_icu_admission_discharge_dates_dict[patient_id]
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_icu_discharge_dates_dict_delta[patient_id] = [
            calculate_days_elapsed(first_lab_test_date, discharge['discharge'])
            if discharge['discharge'] is not None else None
            for discharge in icu_discharges
        ]
    else:
        patient_id_to_icu_discharge_dates_dict_delta[patient_id] = None

# Save all the delta dictionaries
print("Saving delta dictionaries...")
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_death_date_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_death_date_dict_delta, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_last_contact_date_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_last_contact_date_dict_delta, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_ip_admission_dates_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_ip_admission_dates_dict_delta, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_ip_discharge_dates_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_ip_discharge_dates_dict_delta, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_icu_admission_dates_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_icu_admission_dates_dict_delta, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_icu_discharge_dates_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_icu_discharge_dates_dict_delta, f)

print("All dictionaries saved successfully!")


# get med data (drug_exposure rows)
med_data = data[data['omop_table'] == 'drug_exposure']

med_data = med_data[['patient_id', 'start', 'end', 'code', 'value', 'unit',
       'visit_id']]

myelo_drugs = ['DOCETAXEL','SACITUZUMAB GOVITECAN','CISPLATIN','TOPOTECAN','IRINOTECAN','IRINOTECAN LIPOSOMAL','CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE','BENDAMUSTINE','CARBOPLATIN','ETOPOSIDE','CABAZITAXEL']
lt_drugs = ['IPILIMUMAB','ATEZOLIZUMAB','NIVOLUMAB','PEMBROLIZUMAB','AVELUMAB','CEMIPLIMAB','DOSTARLIMAB']
aki_drugs = ['CISPLATIN', 'IFOSFAMIDE','PEMETREXED', 'METHOTREXATE', 'PAMIDRONATE', 'ZOLEDRONIC ACID']
included_drugs = myelo_drugs + lt_drugs + aki_drugs


included_drugs = list(set(included_drugs))

concept_df = pd.read_csv(CONCEPT_CSV)

drug_concepts = concept_df[concept_df['vocabulary_id'].isin(['RxNorm', 'NDC', 'ATC', 'RxNorm Extension'])]

drug_code_mapping = {}
for drug in included_drugs:
    matches = drug_concepts[drug_concepts['concept_name'].str.upper().str.contains(drug.upper(), na=False)]

    # remove ones mentioning oral
    matches = matches[~matches['concept_name'].str.upper().str.contains('ORAL', na=False)]
    print(f"\n{'='*80}")
    print(f"Drug: {drug}")
    print(f"{'='*80}")

    if len(matches) > 0:
        print(f"Found {len(matches)} matches:")
        for idx, row in matches.head(10).iterrows():
            print(f"  Code: {str(row['concept_code']):20s} | Vocab: {row['vocabulary_id']:15s} | Name: {row['concept_name']}")

        drug_code_mapping[drug] = matches['concept_code'].tolist()
    else:
        print(f"  ⚠️   NO MATCHES FOUND")
        drug_code_mapping[drug] = []

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total drugs searched: {len(included_drugs)}")
print(f"Drugs found: {sum(1 for codes in drug_code_mapping.values() if len(codes) > 0)}")
print(f"Drugs NOT found: {sum(1 for codes in drug_code_mapping.values() if len(codes) == 0)}")

not_found = [drug for drug, codes in drug_code_mapping.items() if len(codes) == 0]
if not_found:
    print(f"\n⚠️   Drugs not found:")
    for drug in not_found:
        print(f"  - {drug}")

# save the drug_code_mapping
with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_drug_code_mapping.pickle'), 'wb') as f:
    pickle.dump(drug_code_mapping, f)

# drug exposure dates per patient, and days since first lab test
med_data['code_cat'] = med_data['code'].apply(lambda x: x.split('/')[0] if pd.notna(x) and '/' in str(x) else
None)
med_data['code_num'] = med_data['code'].apply(lambda x: x.split('/')[1] if pd.notna(x) and '/' in str(x) else
None)

print("Sample med_data codes:")
print(med_data['code'].head(20))
print("\nMed data code format:")
print(med_data['code_cat'].value_counts())


print("Sample med_data codes:")
print(med_data[['code', 'code_cat', 'code_num']].head(20))

print("\nMed data code format:")
print(med_data['code_cat'].value_counts())

all_drug_concept_codes = []
drug_name_to_codes = {}

for drug_name, codes in drug_code_mapping.items():
    if codes:
        all_drug_concept_codes.extend(codes)
        drug_name_to_codes[drug_name] = codes

print(f"\nTotal concept codes to search: {len(all_drug_concept_codes)}")

sample_codes = med_data['code_num'].dropna().unique()[:10]
print(f"\nSample code_num values: {sample_codes}")

concept_code_to_drug_name = {}
for drug_name, codes in drug_name_to_codes.items():
    for code in codes:
        concept_code_to_drug_name[str(code)] = drug_name

# exposures to the included drugs
med_data_filtered = med_data[med_data['code_num'].astype(str).isin([str(c) for c in all_drug_concept_codes])]

print(f"\nTotal drug exposure records for included drugs: {len(med_data_filtered)}")
print(f"Unique patients with these drugs: {med_data_filtered['patient_id'].nunique()}")

med_data_filtered['drug_name'] = med_data_filtered['code_num'].astype(str).map(concept_code_to_drug_name)

print("\nDrug exposure summary:")
print(med_data_filtered.groupby('drug_name').agg({
    'patient_id': 'nunique',
    'start': 'count'
}).rename(columns={'patient_id': 'n_patients', 'start': 'n_exposures'}).sort_values('n_patients',
ascending=False))

patient_id_to_drug_exposure_dates_dict = {}

for _, row in med_data_filtered.iterrows():
    patient_id = row['patient_id']
    drug_name = row['drug_name']
    exposure_date = row['start'].split(' ')[0] if pd.notna(row['start']) else None

    if patient_id not in patient_id_to_drug_exposure_dates_dict:
        patient_id_to_drug_exposure_dates_dict[patient_id] = {}

    if drug_name not in patient_id_to_drug_exposure_dates_dict[patient_id]:
        patient_id_to_drug_exposure_dates_dict[patient_id][drug_name] = []

    if exposure_date:
        patient_id_to_drug_exposure_dates_dict[patient_id][drug_name].append(exposure_date)

for patient_id in patient_id_to_drug_exposure_dates_dict:
    for drug_name in patient_id_to_drug_exposure_dates_dict[patient_id]:
        patient_id_to_drug_exposure_dates_dict[patient_id][drug_name].sort()

print(f"\nPatients with drug exposure data: {len(patient_id_to_drug_exposure_dates_dict)}")

# days from first lab test
patient_id_to_drug_exposure_dates_dict_delta = {}

for patient_id in patient_id_to_drug_exposure_dates_dict:
    if patient_id in first_lab_test_date_dict:
        first_lab_test_date = first_lab_test_date_dict[patient_id]
        patient_id_to_drug_exposure_dates_dict_delta[patient_id] = {}

        for drug_name, exposure_dates in patient_id_to_drug_exposure_dates_dict[patient_id].items():
            patient_id_to_drug_exposure_dates_dict_delta[patient_id][drug_name] = [
                calculate_days_elapsed(first_lab_test_date, exp_date)
                for exp_date in exposure_dates
            ]

print(f"Patients with both lab data and drug exposure: {len(patient_id_to_drug_exposure_dates_dict_delta)}")

import pickle

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_drug_exposure_dates_dict.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_drug_exposure_dates_dict, f)

with open(os.path.join(OUTPUT_PATH, 'stanford_ehrshots_patient_id_to_drug_exposure_dates_dict_delta.pickle'), 'wb') as f:
    pickle.dump(patient_id_to_drug_exposure_dates_dict_delta, f)

print("\nDrug exposure dictionaries saved!")

print("\nExample patient drug exposures (delta days from first lab test):")
for patient_id in list(patient_id_to_drug_exposure_dates_dict_delta.keys())[:3]:
    print(f"\nPatient {patient_id}:")
    for drug_name, delta_days in patient_id_to_drug_exposure_dates_dict_delta[patient_id].items():
        print(f"  {drug_name}: {delta_days[:5]}...")