from config import EHRSHOT_CSV, ICD_CODES_CSV, CANCER_PATIENTS_CSV

# read ehrshot.csv line by line, keep rows with ICD codes
with open(EHRSHOT_CSV, 'r') as f:
    data = f.readlines()

# check the total # of unique patients
patients = set()
for row in data[1:]:
    patient_id = row.split(',')[1]
    patients.add(patient_id)

# get icd rows
icd_code_rows = []
for row in data:
    if 'ICD' in row:
        icd_code_rows.append(row)


# save the rows with ICD codes to a new file
with open(ICD_CODES_CSV, 'w') as f:
    for row in icd_code_rows:
        f.write(row)


# load the icd_codes.csv file and check for cancer
import pandas as pd

icd_df = pd.read_csv(ICD_CODES_CSV)

icd_df['omop_table'].value_counts()

icd_condition_df = icd_df[icd_df['omop_table'].str.contains('condition_occurrence')]

# check for cancer
icd_condition_df['ICD_cleaned'] = icd_condition_df['code'].str.split('/').str[-1]

icd_condition_df['ICD_cleaned'].value_counts()
def reclean(row):
    if '-' in row['ICD_cleaned']:
        return row['ICD_cleaned'].split('-')[1]
    else:
        return row['ICD_cleaned']
icd_condition_df['ICD_cleaned'] = icd_condition_df.apply(reclean, axis=1)

# cancer = topography C00-C99
icd_condition_df['Cancer'] = icd_condition_df['ICD_cleaned'].str.contains('C[0-9][0-9]')
icd_condition_df['Cancer'].value_counts()

cancer_df = icd_condition_df[icd_condition_df['Cancer'] == True]

# get the number of cancer patients (patient_id)
cancer_df['patient_id'].nunique()

# get cancer patient ids as a list
cancer_patient_ids = cancer_df['patient_id'].unique().tolist()
cancer_patient_ids = [str(x) for x in cancer_patient_ids]
cancer_patient_ids = set(cancer_patient_ids)

# all rows for cancer patients, with the header
cancer_patient_rows = []
cancer_patient_rows.append(data[0])
for row in data[1:]:
    if row.split(',')[1] in cancer_patient_ids:
        cancer_patient_rows.append(row)

# save the cancer patient subfile
with open(CANCER_PATIENTS_CSV, 'w') as f:
    for row in cancer_patient_rows:
        f.write(row)


# read the above cancer patients data with pandas
import pandas as pd
cancer_patients_df = pd.read_csv(CANCER_PATIENTS_CSV)