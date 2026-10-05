"""Paths, lab/vital/drug mappings and ICD helpers for MIMIC-IV preprocessing (all labs and vitals, not a basic subset)."""

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
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths

# override with environment variables if the data is mounted elsewhere:
#     export SPARC_MIMIC_RAW_PATH=/path/to/mimic_iv_3.1_mult/hosp
#     export SPARC_MIMIC_OUTPUT_PATH=/path/to/SPARC_data/MIMICIV_hosp
MIMIC_RAW_PATH = os.environ.get(
    'SPARC_MIMIC_RAW_PATH',
    sparc_paths.MIMIC_RAW_DIR,
)
OUTPUT_PATH = os.environ.get(
    'SPARC_MIMIC_OUTPUT_PATH',
    sparc_paths.DATA_ROOT + '/MIMICIV_hosp',
)

# create the output dir, but stay importable if it isn't writable
try:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
except OSError as _e:
    print(f"Warning: could not create OUTPUT_PATH {OUTPUT_PATH!r}: {_e}")

# Lab test mappings (MIMIC-IV to SPARC)

# ALL lab tests expected by SPARC (46 total - same as ALL_INCLUDED_TESTS)
ALL_INCLUDED_TESTS = [
    'ALP', 'ALT', 'AST', 'Albumin', 'Alphafetoprotein', 'BUN', 'Baso',
    'Beta-2 Microglobulin', 'Bilirubin, Total', 'Blast', 'CEA', 'CO2', 'Calcium',
    'Cancer Antigen 125', 'Cancer Antigen 15-3', 'Cancer Antigen 19-9',
    'Chloride', 'Conjugated Bili (mg)', 'Creatinine', 'Eos', 'Glucose',
    'HCT', 'HGB', 'Immature Granulocyte', 'Luc', 'Lymph', 'MCH', 'MCHC',
    'MCV', 'Megakaryocyte Fragment', 'Meta', 'Mono', 'Myelo', 'Neut',
    'Nucleated RBC', 'PSA', 'Platelets', 'Potassium', 'Promyelo',
    'Protein, Total', 'RBC', 'RDW', 'Sodium', 'TSH', 'Variant Lymph', 'WBC'
]

# MIMIC-IV LOINC/itemid mappings to SPARC test names
# Source: https://mimic.mit.edu/docs/iv/modules/hosp/d_labitems/
MIMIC_TO_SPARC_LAB_MAPPING = {
    # CBC panel
    'RBC': ['51279'],  # Red Blood Cells
    'MCV': ['51250'],  # Mean Corpuscular Volume
    'MCH': ['51248'],  # Mean Corpuscular Hemoglobin
    'MCHC': ['51249'],  # Mean Corpuscular Hemoglobin Concentration
    'RDW': ['51277'],  # Red Cell Distribution Width
    'HGB': ['51222', '50811'],  # Hemoglobin (Hematology + Blood Gas)
    'HCT': ['51221', '50810'],  # Hematocrit (Hematology + Blood Gas)
    'WBC': ['51301'],  # White Blood Cells
    'Platelets': ['51265'],  # Platelet Count

    # CBC Differential
    'Neut': ['51256'],  # Neutrophils
    'Lymph': ['51244'],  # Lymphocytes
    'Mono': ['51253'],  # Monocyte Count
    'Eos': ['51200'],  # Eosinophils
    'Baso': ['51146'],  # Basophils
    'Immature Granulocyte': [],  # Not in MIMIC-IV
    'Blast': ['51148'],  # Blasts
    'Meta': ['51251'],  # Metamyelocytes
    'Myelo': ['51255'],  # Myelocytes
    'Promyelo': ['51269'],  # Promyelocytes
    'Nucleated RBC': ['51257'],  # Nucleated Red Cells
    'Luc': [],  # Large Unstained Cells (not in MIMIC-IV)
    'Variant Lymph': [],  # Not a separate itemid in MIMIC-IV
    'Megakaryocyte Fragment': [],  # Not in MIMIC-IV

    # BMP panel
    'Glucose': ['50809', '50931'],  # Glucose
    'Calcium': ['50893'],  # Calcium, Total
    'Sodium': ['50983', '50824'],  # Sodium
    'Potassium': ['50971', '50822'],  # Potassium
    'CO2': ['50804'],  # CO2 (Bicarbonate)
    'Creatinine': ['50912'],  # Creatinine
    'Chloride': ['50902', '50806'],  # Chloride
    'BUN': ['51006'],  # Blood Urea Nitrogen

    # LFT panel
    'Albumin': ['50862'],  # Albumin
    'Protein, Total': ['50976'],  # Total Protein
    'ALP': ['50863'],  # Alkaline Phosphatase
    'ALT': ['50861'],  # Alanine Aminotransferase
    'AST': ['50878'],  # Aspartate Aminotransferase
    'Bilirubin, Total': ['50885'],  # Bilirubin, Total
    'Conjugated Bili (mg)': ['50883'],  # Conjugated/Direct Bilirubin

    # Tumor markers (itemids checked against d_labitems)
    'CEA': ['50900'],  # Carcinoembyronic Antigen (CEA)
    'PSA': [],  # PSA not available in MIMIC-IV hosp module
    'Alphafetoprotein': ['50864'],  # Alpha-Fetoprotein (AFP)
    'Cancer Antigen 125': ['50892'],  # CA-125
    'Cancer Antigen 15-3': [],  # CA 15-3 not in MIMIC-IV
    'Cancer Antigen 19-9': ['51579'],  # CA 19-9
    'Beta-2 Microglobulin': ['50881'],  # Beta-2 Microglobulin

    # Other
    'TSH': ['50993'],  # Thyroid Stimulating Hormone
}

# Vital sign mappings (MIMIC-IV to SPARC)

# MIMIC-IV hosp has very limited vitals; only the omr table has:
# Only available in omr table:
#   - Blood Pressure (various positions)
#   - BMI
#   - Height
#   - Weight
# Not in hosp (would need ICU chartevents):
#   - Heart Rate / Pulse
#   - O2 Saturation
#   - Respiratory Rate
#   - Temperature
#   - Invasive Blood Pressure

# ALL vitals expected by SPARC (9 total - same as VITALS)
ALL_VITALS = [
    'BMI',
    'Blood Pressure BP Systolic',
    'Blood Pressure Diastolic',
    'Invasive Blood Pressure BP Systolic',
    'Invasive Blood Pressure Diastolic',
    'O2',
    'Pulse Heart Rate',
    'Resp Rate',
    'Temperature C'
]

# What's actually available from omr table in hosp module
AVAILABLE_VITALS_IN_OMR = [
    'Blood Pressure',  # Can be parsed into Systolic/Diastolic
    'Blood Pressure Sitting',
    'Blood Pressure Standing',
    'Blood Pressure Standing (1 min)',
    'Blood Pressure Standing (3 mins)',
    'Blood Pressure Lying',
    'BMI (kg/m2)',
    'BMI',
    'Height (Inches)',
    'Height',
    'Weight (Lbs)',
    'Weight',
    'eGFR'
]

# High-risk medications

# Myelosuppression drugs (neutropenia, anemia, thrombocytopenia risk)
MYELO_DRUGS = [
    'DOCETAXEL', 'SACITUZUMAB GOVITECAN', 'CISPLATIN', 'TOPOTECAN',
    'IRINOTECAN', 'IRINOTECAN LIPOSOMAL', 'CYCLOPHOSPHAMIDE', 'DOXORUBICIN',
    'VINCRISTINE', 'BENDAMUSTINE', 'CARBOPLATIN', 'ETOPOSIDE', 'CABAZITAXEL'
]

# Liver toxicity (LT) drugs - Immune checkpoint inhibitors
LT_DRUGS = [
    'IPILIMUMAB', 'ATEZOLIZUMAB', 'NIVOLUMAB', 'PEMBROLIZUMAB',
    'AVELUMAB', 'CEMIPLIMAB', 'DOSTARLIMAB'
]

# Acute kidney injury (AKI) drugs
AKI_DRUGS = [
    'CISPLATIN', 'IFOSFAMIDE', 'PEMETREXED', 'METHOTREXATE',
    'PAMIDRONATE', 'ZOLEDRONIC ACID'
]

# Combined list of all high-risk drugs for side effect prediction
INCLUDED_DRUGS = list(set(MYELO_DRUGS + LT_DRUGS + AKI_DRUGS))

def normalize_drug_name(drug_name):
    """Map a drug name to its UPPERCASE standard name if it matches an included drug, else return it unchanged."""
    if not isinstance(drug_name, str):
        return drug_name

    drug_upper = drug_name.upper()

    # Check if any included drug is in the medication name
    for target_drug in INCLUDED_DRUGS:
        if target_drug in drug_upper:
            return target_drug  # Return standardized UPPERCASE name

    return drug_name  # Return original if not in included list

# ICD codes for cancer diagnosis

# ICD-10 cancer codes (C00-D49)
ICD10_CANCER_PREFIXES = [
    'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',  # Malignant neoplasms
    'D0', 'D1', 'D2', 'D3', 'D4'  # In situ, benign, uncertain behavior
]

# ICD-9 cancer codes (140-239)
ICD9_CANCER_RANGES = [
    ('140', '239')  # Neoplasms
]

# Cancer type mapping (ICD to readable names)

ICD10_TO_CANCER_TYPE = {
    'C00': 'Head and Neck Cancer', 'C01': 'Head and Neck Cancer',
    'C02': 'Head and Neck Cancer', 'C03': 'Head and Neck Cancer',
    'C04': 'Head and Neck Cancer', 'C05': 'Head and Neck Cancer',
    'C06': 'Head and Neck Cancer', 'C07': 'Salivary Gland Cancer',
    'C08': 'Salivary Gland Cancer', 'C09': 'Head and Neck Cancer',
    'C10': 'Head and Neck Cancer', 'C11': 'Head and Neck Cancer',
    'C12': 'Head and Neck Cancer', 'C13': 'Head and Neck Cancer',
    'C14': 'Head and Neck Cancer', 'C15': 'Esophagogastric Cancer',
    'C16': 'Esophagogastric Cancer', 'C17': 'Esophagogastric Cancer',
    'C18': 'Colorectal Cancer', 'C19': 'Colorectal Cancer',
    'C20': 'Colorectal Cancer', 'C21': 'Colorectal Cancer',
    'C22': 'Hepatobiliary Cancer', 'C23': 'Hepatobiliary Cancer',
    'C24': 'Hepatobiliary Cancer', 'C25': 'Pancreatic Cancer',
    'C26': 'Esophagogastric Cancer', 'C30': 'Head and Neck Cancer',
    'C31': 'Head and Neck Cancer', 'C32': 'Head and Neck Cancer',
    'C33': 'Non-Small Cell Lung Cancer', 'C34': 'Non-Small Cell Lung Cancer',
    'C37': 'Non-Small Cell Lung Cancer', 'C38': 'Non-Small Cell Lung Cancer',
    'C39': 'Non-Small Cell Lung Cancer', 'C40': 'Bone Cancer',
    'C41': 'Bone Cancer', 'C43': 'Melanoma',
    'C44': 'Skin Cancer, Non-Melanoma', 'C45': 'Mesothelioma',
    'C46': 'Skin Cancer, Non-Melanoma', 'C47': 'Soft Tissue Sarcoma',
    'C48': 'Soft Tissue Sarcoma', 'C49': 'Soft Tissue Sarcoma',
    'C50': 'Breast Cancer', 'C51': 'Cervical Cancer',
    'C52': 'Cervical Cancer', 'C53': 'Cervical Cancer',
    'C54': 'Endometrial Cancer', 'C55': 'Endometrial Cancer',
    'C56': 'Ovarian Cancer', 'C57': 'Ovarian Cancer',
    'C58': 'Ovarian Cancer', 'C60': 'Prostate Cancer',
    'C61': 'Prostate Cancer', 'C62': 'Germ Cell Tumor',
    'C63': 'Prostate Cancer', 'C64': 'Renal Cell Carcinoma',
    'C65': 'Renal Cell Carcinoma', 'C66': 'Renal Cell Carcinoma',
    'C67': 'Bladder Cancer', 'C68': 'Bladder Cancer',
    'C69': 'others', 'C70': 'Glioma', 'C71': 'Glioma',
    'C72': 'Glioma', 'C73': 'Thyroid Cancer',
    'C74': 'others', 'C75': 'others',
    'C76': 'Cancer of Unknown Primary', 'C77': 'Cancer of Unknown Primary',
    'C78': 'Cancer of Unknown Primary', 'C79': 'Cancer of Unknown Primary',
    'C80': 'Cancer of Unknown Primary', 'C81': 'Mature B-Cell Neoplasms',
    'C82': 'Mature B-Cell Neoplasms', 'C83': 'Mature B-Cell Neoplasms',
    'C84': 'Mature T and NK Neoplasms', 'C85': 'Mature B-Cell Neoplasms',
    'C86': 'Mature T and NK Neoplasms', 'C88': 'Mature B-Cell Neoplasms',
    'C90': 'Mature B-Cell Neoplasms', 'C91': 'Leukemia',
    'C92': 'Leukemia', 'C93': 'Leukemia',
    'C94': 'Leukemia', 'C95': 'Leukemia',
    'C96': 'Leukemia', 'D0': 'others',
    'D1': 'others', 'D2': 'others',
    'D3': 'others', 'D4': 'Myelodysplastic Syndromes',
}

# Utility functions

def is_cancer_code_icd10(code):
    """Check if ICD-10 code is cancer-related."""
    if not isinstance(code, str):
        return False
    code = code.strip().upper()
    for prefix in ICD10_CANCER_PREFIXES:
        if code.startswith(prefix):
            return True
    return False

def is_cancer_code_icd9(code):
    """Check if ICD-9 code is cancer-related."""
    if not isinstance(code, str):
        return False
    code = code.strip()
    try:
        code_num = float(code)
        for start, end in ICD9_CANCER_RANGES:
            if float(start) <= code_num <= float(end):
                return True
    except:
        pass
    return False

def map_icd10_to_cancer_type(code):
    """Map ICD-10 code to cancer type."""
    if not isinstance(code, str):
        return 'others'
    code = code.strip().upper()
    # Try 3-character prefix match
    prefix = code[:3]
    return ICD10_TO_CANCER_TYPE.get(prefix, 'others')

if __name__ == '__main__':
    print(f"Configuration loaded. Output path: {OUTPUT_PATH}")
    print(f"Expected {len(ALL_INCLUDED_TESTS)} lab tests (ALL_INCLUDED_TESTS)")
    print(f"Expected {len(ALL_VITALS)} vitals (VITALS)")
