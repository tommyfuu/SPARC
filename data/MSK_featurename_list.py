ALL_INCLUDED_TESTS = ['ALP', 'ALT', 'AST', 'Albumin', 'Alphafetoprotein', 'BUN', 'Baso', 'Beta-2 Microglobulin', 'Bilirubin, Total', 'Blast', 'CEA', 'CO2', 'Calcium', 
                        'Cancer Antigen 125', 'Cancer Antigen 15-3', 'Cancer Antigen 19-9', 'Chloride', 'Conjugated Bili (mg)', 'Creatinine', 'Eos', 'Glucose', 'HCT', 'HGB', 
                        'Immature Granulocyte', 'Luc', 'Lymph', 'MCH', 'MCHC', 'MCV', 'Megakaryocyte Fragment', 'Meta', 'Mono', 'Myelo', 'Neut', 'Nucleated RBC', 'PSA', 
                        'Platelets', 'Potassium', 'Promyelo', 'Protein, Total', 'RBC', 'RDW', 'Sodium', 'TSH', 'Variant Lymph', 'WBC']


# for basic features, let's use tests in the CBC and CMP panels
## 7 Red Blood Cell (RBC) tests: RBC count, Mean Corpuscular Volume (MCV), Mean Corpuscular Hemoglobin (MCH), MCH Concentration (MCHC), Red Cell Distribution Width (RDW), Hemoglobin (HGB), Hematocrit (HCT).
## 2 other tests included in the Complete Blood Count tests: White Blood Cells (WBC), Platelate count (PLT).
RBC_CBC_TESTS = ['RBC', 'MCV', 'MCH', 'MCHC', 'RDW', 'HGB', 'HCT', 'WBC', 'Platelets']
## 8 Basic Metabolic Panel tests (BMP): Glucose, Calcium, Sodium, Potassium, Carbon Dioxide, creatinine, Chloride, Blood Urea Nitrogen (BUN)
## 6 liver function tests: Albumin, Total protein, Alkaline Phosphatase (ALP, although in the original data it's called ALK), Alanine Transaminase (ALT), Aspartate Aminotransferase (AST), Bilirubin.
CMP_LIVER_TESTS = ['Glucose', 'Calcium', 'Sodium', 'Potassium', 'CO2', 'Creatinine', 'Chloride', 'BUN', 'Albumin', 'Protein, Total', 'ALP', 'ALT', 'AST', 'Bilirubin, Total']

BMP_TESTS = ['Glucose', 'Calcium', 'Sodium', 'Potassium', 'CO2', 'Creatinine', 'Chloride', 'BUN']
LFT_TESTS = ['Albumin', 'Protein, Total', 'ALP', 'ALT', 'AST', 'Bilirubin, Total']
CBC_DIFFERENTIAL_TESTS = ['Neut', 'Mono', 'Lymph', 'Baso', 'Eos']
BASIC_TESTS = RBC_CBC_TESTS + CMP_LIVER_TESTS
BASIC_TESTS_INDICES = [ALL_INCLUDED_TESTS.index(test) for test in BASIC_TESTS]
EXTRA_TESTS = [test for test in ALL_INCLUDED_TESTS if test not in BASIC_TESTS]
EXTRA_TESTS_INDICES = [ALL_INCLUDED_TESTS.index(test) for test in EXTRA_TESTS]
PARSIMONIOUS_TESTS = RBC_CBC_TESTS + CBC_DIFFERENTIAL_TESTS + CMP_LIVER_TESTS 

ANEMIA_INDEX = ALL_INCLUDED_TESTS.index('HGB')
NEUTROPHIL_PERCENT_INDEX = ALL_INCLUDED_TESTS.index('Neut')
WBC_INDEX = ALL_INCLUDED_TESTS.index('WBC')
CREATININE_INDEX = ALL_INCLUDED_TESTS.index('Creatinine')
AST_INDEX = ALL_INCLUDED_TESTS.index('AST')
ALT_INDEX = ALL_INCLUDED_TESTS.index('ALT')
ALP_INDEX = ALL_INCLUDED_TESTS.index('ALP')
BILIRUBIN_INDEX = ALL_INCLUDED_TESTS.index('Bilirubin, Total')


VITALS = ['BMI', 'Blood Pressure BP Systolic', 'Blood Pressure Diastolic', 'Invasive Blood Pressure BP Systolic', 'Invasive Blood Pressure Diastolic', 'O2', 'Pulse Heart Rate', 'Resp Rate', 'Temperature C']

Temperature_C_INDEX = VITALS.index('Temperature C')
BASIC_VITALS = ['BMI', 'Blood Pressure BP Systolic', 'Blood Pressure Diastolic', 'O2', 'Pulse Heart Rate', 'Resp Rate', 'Temperature C']
EXTRA_VITALS = ['Invasive Blood Pressure BP Systolic', 'Invasive Blood Pressure Diastolic']
MET_STATUS_LIST = ['adrenal_glands', 'Bone', 'CNS_Brain', 'Intra-Abdominal','Liver', 'Lung', 'lymph_nodes', 'Other', 'Pleura', 'Reproductive_Organs']
BASIC_VITALS_INDICES = [VITALS.index(test) for test in BASIC_VITALS]
EXTRA_VITALS_INDICES = [VITALS.index(test) for test in EXTRA_VITALS]

GENDERS = ['MALE', 'FEMALE']
PRIMARY_CTYPES = ['Non-Small Cell Lung Cancer', 'Colorectal Cancer', 'Breast Cancer', 'Prostate Cancer', 'Pancreatic Cancer', 'Mature B-Cell Neoplasms', 'others', 'Endometrial Cancer', 'Ovarian Cancer', 'Soft Tissue Sarcoma', 'Bladder Cancer', 'Esophagogastric Cancer', 'Glioma', 'Melanoma', 'Cancer of Unknown Primary', 'Hepatobiliary Cancer', 'Renal Cell Carcinoma', 'Leukemia', 'Thyroid Cancer', 'Myelodysplastic Workup', 'Head and Neck Cancer', 'Gastrointestinal Stromal Tumor', 'Bone Cancer', 'Germ Cell Tumor', 'Small Cell Lung Cancer', 'Mature T and NK Neoplasms', 'Uterine Sarcoma', 'Myeloproliferative Neoplasms', 'Mesothelioma', 'Skin Cancer, Non-Melanoma', 'Appendiceal Cancer', 'Salivary Gland Cancer', 'Cervical Cancer', 'Myelodysplastic Syndromes']
PRIMARY_CTYPES_wo_discrepancy = PRIMARY_CTYPES.copy()
PRIMARY_CTYPES_wo_discrepancy.remove('others')
PRIMARY_CTYPES_wo_discrepancy.remove('Cancer of Unknown Primary')

PRIMARY_CTYPES_TO_BE_PREDICTED = ['Non-Small Cell Lung Cancer', 'Colorectal Cancer', 'Breast Cancer', 'Prostate Cancer', 'Pancreatic Cancer', 'Endometrial Cancer', 'Ovarian Cancer', 'Soft Tissue Sarcoma', 'Bladder Cancer', 'Esophagogastric Cancer', 'Glioma', 'Melanoma', 'Cancer of Unknown Primary', 'Hepatobiliary Cancer', 'Renal Cell Carcinoma', 'Thyroid Cancer', 'Head and Neck Cancer', 'Gastrointestinal Stromal Tumor', 'Germ Cell Tumor', 'Bone Cancer', 'Small Cell Lung Cancer', 'Uterine Sarcoma', 'Skin Cancer, Non-Melanoma', 'Mesothelioma', 'Appendiceal Cancer', 'Salivary Gland Cancer', 'Cervical Cancer', 'Peripheral Nervous System']
PRIMARY_CTYPES_TO_BE_PREDICTED.remove('Cancer of Unknown Primary')

GENETIC_ANCESTRIES = ['EUR', 'ADM', 'EAS', 'AFR', 'SAS', 'NAM']


SMOKE_list = ['Former/Current smoker', 'Never', 'Unknown']
# import pandas as pd
# path = '/data/morrisq/fuc/frozen_step6_dePHI-ed_data_101623/'
# df_cleaned_341 = pd.read_csv(f'/data/morrisq/fuc/impact_genomic_ft/outputs_050725/feature_table_full_first_341_harmonized.csv')
# mutation_features = list(set(df_cleaned_341.columns)-set(['PATIENT_ID', 'YOST_INDEX_IMPUTED_MEDIAN', 'CURRENT_AGE_DEID', 'SEX']))

clinical_features = ['YOST_INDEX_IMPUTED_MEDIAN', 'ECOG_KPS'] + SMOKE_list + PRIMARY_CTYPES
# clinical_features.remove('Unknown')
clinical_features.remove('others')


MSK_numfeature_dict = {'gender': 1,
                    'age': 1,
                    'primary_ctype': len(PRIMARY_CTYPES)-1,
                    'genetic_ancestry': len(GENETIC_ANCESTRIES),
                    'basic_lab': len(BASIC_TESTS),
                    'extra_lab': len(EXTRA_TESTS),
                    'basic_vitals': len(BASIC_VITALS),
                    'extra_vitals': len(EXTRA_VITALS),
                    'met_status': len(MET_STATUS_LIST),
                    # 'mutation': len(mutation_features),
                    # 'mutation': 5127,
                     'mutation': 5044,
                    'clinical': len(clinical_features)-1,
                    'cbc_differential': len(CBC_DIFFERENTIAL_TESTS)
                    }

MSK_firstprinciple_numfeature_dict = {
    ## neutropenia
    'neutropenia_lab': 1, # WBC
    'neutropenia_vitals': 0, # WBC
    ## FN
    'febrile_neutropenia_lab': 1, # WBC + temperature
    'febrile_neutropenia_vitals': 1, # WBC + temperature
    ## Anemia
    'anemia_lab': 1, # HGB
    'anemia_vitals': 0,
    ## AKI
    'AKI_lab': 1, # creatinine
    'AKI_vitals': 0,
    ## Liver toxicity
    'Liver_toxicity_lab': 4, # AST + ALT + ALP + bilirubin
    'Liver_toxicity_vitals': 0,
}


total_static_len = sum([v for k,v in MSK_numfeature_dict.items() if k in ['gender', 'age', 'primary_ctype', 'genetic_ancestry', 'met_status']])
total_dynamic_len = sum([v for k,v in MSK_numfeature_dict.items() if k in ['basic_lab', 'extra_lab', 'basic_vitals', 'extra_vitals']])

MSK_ts_feat_start_dict = {'basic_lab': 0, 'extra_lab': MSK_numfeature_dict['basic_lab'], 'vitals': MSK_numfeature_dict['basic_lab']+MSK_numfeature_dict['extra_lab']}

def one_hot_encode_dict(static_dict, feature_reduced = '', unique_vals = []):
    '''one-hot encoding with the ability to reduce a feature out'''
    if unique_vals == []:
        unique_vals = list(set(static_dict.values())) # a sorted list of unique values
    print(unique_vals)
    if feature_reduced != '' and feature_reduced in unique_vals:
        unique_vals.remove(feature_reduced)
        print(unique_vals)
    indices = [unique_vals.index(val) if val in unique_vals else -1 for val in static_dict.values()]
    static_dict_out = {k: [0]*len(unique_vals) for k in static_dict}
    ## assigns one to appropriate indices
    for i, k in enumerate(static_dict.keys()):
        if indices[i]!=-1:
            static_dict_out[k][indices[i]] = 1
    return static_dict_out, unique_vals


# Mapping rules
timeshift_ctype_mapping_dict = {
    'Appendiceal Cancer': ['Mucinous Adenocarcinoma of the Appendix', 'Appendiceal Adenocarcinoma', 'Goblet Cell Adenocarcinoma of the Appendix',  'Low-grade Appendiceal Mucinous Neoplasm', 'Well-Differentiated Neuroendocrine Tumor of the Appendix'],
    'Non-Small Cell Lung Cancer': ['Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma', 'Non-Small Cell Lung Cancer', 'Adenocarcinoma, NOS', 'Poorly Differentiated Non-Small Cell Lung Cancer', 'Pleomorphic Carcinoma of the Lung', 'Lung Carcinoid', 'Large Cell Neuroendocrine Carcinoma',
                                    'Lung Adenosquamous Carcinoma', 'Atypical Lung Carcinoid', 'Lung Neuroendocrine Tumor', 'Adenoid Cystic Carcinoma of the Lung'],
    'Breast Cancer': ['Breast Invasive Ductal Carcinoma', 'Invasive Breast Carcinoma', 'Breast Invasive Lobular Carcinoma', 'Metaplastic Breast Cancer',
            'Breast Invasive Carcinoma, NOS', 'Breast Cancer', 'Breast Angiosarcoma', 'Breast Mixed Ductal and Lobular Carcinoma', 'Breast Invasive Cancer, NOS'],
    'Bone Cancer': ['Bone Cancer', 'Giant Cell Tumor of Bone', 'Osteosarcoma', 'Chondrosarcoma', 'Ewing Sarcoma', 'Osteoblastic Osteosarcoma',
            'Dedifferentiated Chondrosarcoma', 'Chordoma', 'Conventional Type Chordoma', 'Parosteal Osteosarcoma'],
    'Colorectal Cancer': ['Colon Adenocarcinoma', 'Rectal Adenocarcinoma', 'Colorectal Adenocarcinoma', 'Mucinous Adenocarcinoma of the Colon and Rectum', 'Signet Ring Cell Adenocarcinoma of the Colon and Rectum'],
    'Prostate Cancer': ['Prostate Cancer', 'Prostate Adenocarcinoma', 'Prostate Neuroendocrine Carcinoma', 'Small Intestinal Carcinoma', 'Prostate Small Cell Carcinoma'],
    'Pancreatic Cancer': ['Pancreatic Adenocarcinoma', 'Pancreatic Neuroendocrine Tumor', 'Adenosquamous Carcinoma of the Pancreas', 'Pancreatobiliary Ampullary Carcinoma', 'Mixed Ampullary Carcinoma', 
        'Pancreatic Neuroendocrine Carcinoma', 'Acinar Cell Carcinoma of the Pancreas', 'Undifferentiated Carcinoma of the Pancreas', 'Intraductal Papillary Mucinous Neoplasm', 'Solid Pseudopapillary Neoplasm of the Pancreas', 'Intestinal Ampullary Carcinoma'],
    'Mature B-Cell Neoplasms': ['Mature B-Cell Neoplasms'],
    'Endometrial Cancer': ['Uterine Endometrioid Carcinoma', 'Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma', 'Endometrial Carcinoma','Uterine Mixed Endometrial Carcinoma', 'Endometrial Stromal Sarcoma', 'Low-Grade Endometrial Stromal Sarcoma', 'High-Grade Endometrial Stromal Sarcoma', 'Uterine Clear Cell Carcinoma', 'Peritoneal Serous Carcinoma',
        'Uterine Dedifferentiated Carcinoma', 'Poorly Differentiated Carcinoma of the Uterus', 'Uterine Mesonephric Carcinoma'],
    'Ovarian Cancer': [ 'Low-Grade Serous Ovarian Cancer', 'High-Grade Serous Ovarian Cancer', 'Clear Cell Ovarian Cancer', 'High-Grade Serous Fallopian Tube Cancer', 'Endometrioid Ovarian Cancer', 'Mucinous Ovarian Cancer','Serous Ovarian Cancer', 'Serous Borderline Ovarian Tumor', 'Ovarian Cancer, Other', 'Mucinous Borderline Ovarian Tumor', 'Mixed Ovarian Carcinoma', 'Ovarian Carcinosarcoma / Malignant Mixed Mesodermal Tumor', 'Ovarian Epithelial Tumor', 'Ovarian Carcinosarcoma/Malignant Mixed Mesodermal Tumor', 'Ovarian Epithelial Tumor', "Serous Borderline Ovarian Tumor, Micropapillary"],
    'Bladder Cancer': ['Bladder Urothelial Carcinoma',  'Bladder Cancer',  'Small Cell Bladder Cancer', 'Bladder Squamous Cell Carcinoma', 'Bladder Adenocarcinoma', 'Plasmacytoid/Signet Ring Cell Bladder Carcinoma', 'Upper Tract Urothelial Carcinoma', 'Urethral Urothelial Carcinoma', 'Urethral Cancer'],
    'Esophagogastric Cancer': ['Esophageal Adenocarcinoma', 'Esophagogastric Adenocarcinoma', 'Esophageal Poorly Differentiated Carcinoma', 'High-Grade Neuroendocrine Carcinoma of the Esophagus', 'Adenocarcinoma of the Gastroesophageal Junction', 'Stomach Adenocarcinoma', 'Esophageal Squamous Cell Carcinoma', 'Gastrointestinal Neuroendocrine Tumors of the Esophagus/Stomach', 'Duodenal Adenocarcinoma', 'Small Bowel Well-Differentiated Neuroendocrine Tumor', 'Small Bowel Cancer',
        'Diffuse Type Stomach Adenocarcinoma', 'Poorly Differentiated Carcinoma of the Stomach', 'High-Grade Neuroendocrine Carcinoma of the Colon and Rectum', 'Ampullary Carcinoma', 'Intestinal Type Stomach Adenocarcinoma',
        'Medullary Carcinoma of the Colon', 'Signet Ring Cell Carcinoma of the Stomach', 'Tubular Adenoma of the Colon', 'Perihilar Cholangiocarcinoma', 'Gastrointestinal Neuroendocrine Tumors', 'High-Grade Neuroendocrine Carcinoma of the Stomach', 'Adenosquamous Carcinoma of the Stomach'],
    'Soft Tissue Sarcoma': ['Soft Tissue Sarcoma', 'Ewing Sarcoma of Soft Tissue', 'Sarcoma, NOS', 'Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma', 'Dedifferentiated Liposarcoma', 'Leiomyosarcoma', 'Myxofibrosarcoma', 'Desmoplastic Small-Round-Cell Tumor', 'Desmoid/Aggressive Fibromatosis', 'Well-Differentiated Liposarcoma', 'Myxoid / Round-Cell Liposarcoma', 'Synovial Sarcoma', 'Angiosarcoma', 'Solitary Fibrous Tumor/Hemangiopericytoma',
    'Myxoid/Round-Cell Liposarcoma', 'Malignant Peripheral Nerve Sheath Tumor', 'Embryonal Rhabdomyosarcoma', 'Alveolar Rhabdomyosarcoma', 'Rhabdomyosarcoma',
        'Round Cell Sarcoma, NOS', 'Epithelioid Sarcoma', 'Atypical Lipomatous Tumor', 'Low-Grade Fibromyxoid Sarcoma', 'Tenosynovial Giant Cell Tumor Diffuse Type', 'Pleomorphic Liposarcoma', 'Epithelioid Hemangioendothelioma', 'Myxoid/Round-Cell Liposarcoma',
        'Malignant Peripheral Nerve Sheath Tumor', 'Embryonal Rhabdomyosarcoma', 'Alveolar Rhabdomyosarcoma', 'Rhabdomyosarcoma', 'Round Cell Sarcoma, NOS',
        'Epithelioid Sarcoma', 'Atypical Lipomatous Tumor', 'Low-Grade Fibromyxoid Sarcoma', 'Tenosynovial Giant Cell Tumor Diffuse Type', 'Pleomorphic Liposarcoma', 'Clear Cell Sarcoma', 'Epithelioid Hemangioendothelioma', 'Perivascular Epithelioid Cell Tumor', 'Radiation-Associated Sarcoma'],
    'Cancer of Unknown Primary': ['Cancer of Unknown Primary', 'Poorly Differentiated Carcinoma, NOS', 'Squamous Cell Carcinoma, NOS', 'Neuroendocrine Carcinoma, NOS', 'Neuroendocrine Tumor, NOS', 'Undifferentiated Malignant Neoplasm', 'Small Cell Carcinoma of Unknown Primary',
        'Cancer of Unknown Primary, NOS', 'Unknown'],
    'Germ Cell Tumor': ['Mixed Germ Cell Tumor', 'Extra Gonadal Germ Cell Tumor',  'Non-Seminomatous Germ Cell Tumor', 'Germ Cell Tumor with Somatic-Type Malignancy', 'Germ Cell Tumor', 'Dysgerminoma', 'Seminoma',
        'Granulosa Cell Tumor', 'Sex Cord Stromal Tumors', 'Teratoma', 'Yolk Sac Tumor', 'Choriocarcinoma', 'Germinoma'],
    'Glioma': ['Glioma', 'Glioma, NOS', 'High-Grade Glioma, NOS','Low-Grade Glioma, NOS', 'Ganglioneuroblastoma', 'Diffuse Intrinsic Pontine Glioma','Glioblastoma', 'Glioblastoma Multiforme', 'Oligodendroglioma', 'Paraganglioma','Ganglioglioma', 'Anaplastic Oligodendroglioma', 'Diffuse Glioma', 'Diffuse Astrocytoma', 'Pilocytic Astrocytoma', 'Anaplastic Astrocytoma', 'Astrocytoma', 'Medulloblastoma',
        'Meningioma', 'Atypical Meningioma'],
    'Gastrointestinal Stromal Tumor': ['Gastrointestinal Stromal Tumor'],
    'Head and Neck Cancer': ['Head and Neck Squamous Cell Carcinoma', 'Head and Neck Carcinoma, Other', 'Head and Neck Neuroendocrine Carcinoma',  'Head and Neck Squamous Cell Carcinoma of Unknown Primary',  'NUT Midline Carcinoma of the Head and Neck', 'Oropharynx Squamous Cell Carcinoma', 'Oral Cavity Squamous Cell Carcinoma', 'Adenoid Cystic Carcinoma', 'Larynx Squamous Cell Carcinoma', 'Sinonasal Squamous Cell Carcinoma', 'Nasopharyngeal Carcinoma'],
    # 'Leukemia': [ 'Acute Leukemias of Ambiguous Lineage'],
    'Leukemia': [],
    'Melanoma': ['Acral Melanoma', 'Cutaneous Melanoma', 'Melanoma', 'Head and Neck Mucosal Melanoma', 'Melanoma of Unknown Primary',  'Uveal Melanoma', 'Anorectal Mucosal Melanoma', 'Mucosal Melanoma of the Vulva/Vagina', 'Desmoplastic Melanoma'],
    'Mature T and NK Neoplasms': ['Mature T and NK Neoplasms'],
    'Renal Cell Carcinoma': ['Renal Cell Carcinoma', 'Renal Clear Cell Carcinoma', 'Chromophobe Renal Cell Carcinoma', 'Papillary Renal Cell Carcinoma', 'FH-Deficient Renal Cell Carcinoma', 'Renal Non-Clear Cell Carcinoma', 'Translocation-Associated Renal Cell Carcinoma', 'Unclassified Renal Cell Carcinoma', 'Kidney Renal Clear Cell Papillary Carcinoma', 'Renal Clear Cell Carcinoma with Sarcomatoid Features', 'Renal Mucinous Tubular Spindle Cell Carcinoma', 'Renal Oncocytoma'],
    'Hepatobiliary Cancer': ['Intrahepatic Cholangiocarcinoma', 'Cholangiocarcinoma', 'Hepatocellular Carcinoma', 'Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma', 'Hepatoblastoma', "Gallbladder Adenocarcinoma, NOS", 'Extrahepatic Cholangiocarcinoma', 'Gallbladder Cancer'],
    'Small Cell Lung Cancer': ['Small Cell Lung Cancer', 'Combined Small Cell Lung Carcinoma'],
    'Thyroid Cancer': ['Thyroid Cancer','Papillary Thyroid Cancer', 'Poorly Differentiated Thyroid Cancer',  'Anaplastic Thyroid Cancer', 'Medullary Thyroid Cancer', 'Follicular Thyroid Cancer', 'Hurthle Cell Thyroid Cancer', 'Well-Differentiated Thyroid Cancer', ],
    'Mesothelioma': ['Mesothelioma', 'Pleural Mesothelioma, Epithelioid Type',  'Peritoneal Mesothelioma', 'Pleural Mesothelioma, Biphasic Type', 'Pleural Mesothelioma', 'Testicular Mesothelioma', 'Pleural Mesothelioma, Sarcomatoid Type'],
    'Cervical Cancer': ['Cervical Squamous Cell Carcinoma', 'Endocervical Adenocarcinoma', 'Cervical Adenocarcinoma', 'Cervical Clear Cell Carcinoma', 'Cervical Adenosquamous Carcinoma','Mixed Cervical Carcinoma', 'Cervical Rhabdomyosarcoma', 'Cervical Neuroendocrine Tumor', 'Squamous Cell Carcinoma of the Vulva/Vagina', 'Small Cell Carcinoma of the Cervix'],
    'Uterine Sarcoma': [ 'Uterine Carcinosarcoma/Uterine Malignant Mixed Mullerian Tumor',  'Uterine Leiomyosarcoma', 'Uterine Sarcoma/Mesenchymal', 'Uterine Adenosarcoma', 'Uterine Sarcoma, Other', 'Undifferentiated Uterine Sarcoma', 'Other Uterine Tumor', 'Uterine Undifferentiated Carcinoma', 'Uterine Smooth Muscle Tumor'],
    'Mature B-Cell Neoplasms': ['Mature B-Cell Neoplasms'],
    'Skin Cancer, Non-Melanoma': ['Skin Adnexal Carcinoma', 'Merkel Cell Carcinoma', 'Cutaneous Squamous Cell Carcinoma', 'Extramammary Paget Disease', 'Basal Cell Carcinoma'],
    'Salivary Gland Cancer': ['Salivary Duct Carcinoma', 'Salivary Carcinoma', 'Salivary Carcinoma, Other','Salivary Adenocarcinoma', 'Acinic Cell Carcinoma', 'Mucoepidermoid Carcinoma'], 
    'others': []  # Fallback for unclassified
}


myelo_drugs = ['DOCETAXEL','SACITUZUMAB GOVITECAN','CISPLATIN','TOPOTECAN','IRINOTECAN','IRINOTECAN LIPOSOMAL','CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE','BENDAMUSTINE','CARBOPLATIN','ETOPOSIDE','CABAZITAXEL']
lt_drugs = ['IPILIMUMAB','ATEZOLIZUMAB','NIVOLUMAB','PEMBROLIZUMAB','AVELUMAB','CEMIPLIMAB','DOSTARLIMAB']
aki_drugs = ['CISPLATIN', 'IFOSFAMIDE','PEMETREXED', 'METHOTREXATE', 'PAMIDRONATE', 'ZOLEDRONIC ACID']
included_drugs = myelo_drugs + lt_drugs + aki_drugs
# feature-module config: lab tests grouped by toxicity for per-module tokenization.
#   erythroid        -> anemia
#   myeloid_plt      -> neutropenia / febrile neutropenia / thrombocytopenia
#   renal_metabolic  -> AKI
#   hepatic          -> liver toxicity
# Vitals form their own module; static features and time are handled separately.
# Groups are by test name and resolved against the loader's column order at runtime.

# lab groupings
LAB_MODULE_ERYTHROID      = ['HGB', 'HCT', 'RBC', 'MCV', 'MCH', 'MCHC', 'RDW']
LAB_MODULE_MYELOID_PLT    = ['WBC', 'Neut', 'Mono', 'Lymph', 'Baso', 'Eos', 'Platelets']
LAB_MODULE_RENAL_METAB    = ['Creatinine', 'BUN', 'Sodium', 'Potassium', 'Chloride',
                             'CO2', 'Calcium', 'Glucose']
LAB_MODULE_HEPATIC        = ['ALT', 'AST', 'ALP', 'Bilirubin, Total',
                             'Albumin', 'Protein, Total']


# tumor markers / specialized assays, only in the 'all' / extra set
LAB_MODULE_TUMOR_MARKERS  = ['Alphafetoprotein', 'CEA', 'PSA',
                             'Cancer Antigen 125', 'Cancer Antigen 15-3',
                             'Cancer Antigen 19-9', 'Beta-2 Microglobulin', 'TSH']
LAB_MODULE_IMMATURE_MYELOID = ['Blast', 'Promyelo', 'Myelo', 'Meta',
                               'Immature Granulocyte', 'Luc', 'Variant Lymph',
                               'Megakaryocyte Fragment', 'Nucleated RBC',
                               'Conjugated Bili (mg)']

# (module_name, tests) in token order; a test goes to the first module that claims it,
# unclaimed tests are pooled into a trailing 'other_labs' module
LAB_MODULE_DEFINITION = [
    ('erythroid',         LAB_MODULE_ERYTHROID),
    ('myeloid_plt',       LAB_MODULE_MYELOID_PLT),
    ('renal_metabolic',   LAB_MODULE_RENAL_METAB),
    ('hepatic',           LAB_MODULE_HEPATIC),
    ('tumor_markers',     LAB_MODULE_TUMOR_MARKERS),
    ('immature_myeloid',  LAB_MODULE_IMMATURE_MYELOID),
]

# vitals: a single module
VITALS_MODULE_DEFINITION = [
    ('vitals', VITALS),  # uses whatever vital subset is present
]


def build_module_index_config(modular_featuredim_dict,
                              advanced_model='default'):
    """Map each clinical module to its column indices in the loader's [static | dynamic] feature vector.

    Returns a dict with 'modules' [(name, cols)], 'static_start', 'n_static', 'dynamic_start', 'n_dynamic', 'n_base'.
    """
    if advanced_model != 'default':
        # per-module tokenizer only supports advanced_model='default'
        pass

    static_keys_order = ['gender', 'age', 'primary_ctype', 'genetic_ancestry',
                         'met_status', 'clinical', 'mutation']
    dynamic_keys_order = ['basic_lab', 'extra_lab', 'cbc_differential',
                          'basic_vitals', 'extra_vitals']

    # Static block size (prepended by the loader)
    n_static = sum(modular_featuredim_dict.get(k, 0) for k in static_keys_order)

    # Resolve dynamic block: build the ordered list of (test_name, col_index)
    col = n_static
    name_to_idx = {}    # for labs
    vital_name_to_idx = {}

    if modular_featuredim_dict.get('basic_lab', 0) > 0:
        for nm in BASIC_TESTS:
            name_to_idx[nm] = col
            col += 1
    if modular_featuredim_dict.get('extra_lab', 0) > 0:
        for nm in EXTRA_TESTS:
            name_to_idx[nm] = col
            col += 1
    if modular_featuredim_dict.get('cbc_differential', 0) > 0:
        for nm in CBC_DIFFERENTIAL_TESTS:
            # names overlap with EXTRA_TESTS; last writer (cbc_differential) wins
            name_to_idx[nm] = col
            col += 1
    if modular_featuredim_dict.get('basic_vitals', 0) > 0:
        for nm in BASIC_VITALS:
            vital_name_to_idx[nm] = col
            col += 1
    if modular_featuredim_dict.get('extra_vitals', 0) > 0:
        for nm in [v for v in VITALS if v not in BASIC_VITALS]:
            vital_name_to_idx[nm] = col
            col += 1

    n_dynamic = col - n_static

    # Resolve clinical groupings against the columns actually present
    modules = []
    claimed = set()
    for mod_name, test_names in LAB_MODULE_DEFINITION:
        idxs = [name_to_idx[t] for t in test_names if t in name_to_idx]
        if idxs:
            modules.append((mod_name, idxs))
            claimed.update(idxs)

    # pool unclaimed labs (only expected for 'all')
    leftover = [i for i in name_to_idx.values() if i not in claimed]
    if leftover:
        modules.append(('other_labs', sorted(leftover)))

    if vital_name_to_idx:
        modules.append(('vitals', sorted(vital_name_to_idx.values())))

    return {
        'modules': modules,
        'static_start': 0 if n_static > 0 else None,
        'n_static': n_static,
        'dynamic_start': n_static,
        'n_dynamic': n_dynamic,
        'n_base': n_static + n_dynamic,
    }
