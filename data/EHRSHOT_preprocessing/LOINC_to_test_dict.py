"""LOINC code -> SPARC test name, checked against the codes present in EHRSHOT (OMOP concept.csv)."""

# CBC, CMP and liver panel
LOINIC_TO_BASIC_TEST_NAMES = {
    '789-8': 'RBC',               # 10*6/uL
    '787-2': 'MCV',               # fL
    '785-6': 'MCH',               # pg
    '786-4': 'MCHC',              # g/dL
    '788-0': 'RDW',               # %
    '718-7': 'HGB',               # g/dL
    '4544-3': 'HCT',              # %
    '6690-2': 'WBC',              # 10*3/uL
    '777-3': 'Platelets',         # 10*3/uL
    '2345-7': 'Glucose',          # mg/dL
    '17861-6': 'Calcium',         # mg/dL
    '2951-2': 'Sodium',           # mEq/L
    '2823-3': 'Potassium',        # mEq/L
    '2028-9': 'CO2',              # mEq/L
    '2160-0': 'Creatinine',       # mg/dL
    '2075-0': 'Chloride',         # mEq/L
    '3094-0': 'BUN',              # mg/dL
    '1751-7': 'Albumin',          # g/dL
    '2885-2': 'Protein, Total',   # g/dL
    '6768-6': 'ALP',              # U/L
    '1742-6': 'ALT',              # U/L
    '1920-8': 'AST',              # U/L
    '1975-2': 'Bilirubin, Total', # mg/dL
}

LOINC_TO_VITALS_NAMES = {
    '39156-5': 'BMI',
    '8480-6': 'Blood Pressure BP Systolic',
    '8462-4': 'Blood Pressure Diastolic',
    # invasive BP has no records in EHRSHOT, kept as placeholders
    'INVASIVE-SYS': 'Invasive Blood Pressure BP Systolic',
    'INVASIVE-DIA': 'Invasive Blood Pressure Diastolic',
    # nearly all EHRSHOT O2 is SNOMED 252465000 (pulse oximetry), not LOINC
    '2713-6': 'O2',
    '8867-4': 'Pulse Heart Rate',
    '9279-1': 'Resp Rate',
    '8310-5': 'Temperature C',
}

# absolute counts (10*3/uL); the % codes are 770-8, 5905-5, 736-9, 706-2, 713-8
LOINC_TO_CBC_DIFFERENTIALS_NAMES = {
    '751-8': 'Neut',
    '742-7': 'Mono',
    '731-0': 'Lymph',
    '704-7': 'Baso',
    '711-2': 'Eos',
}

# in ALL_INCLUDED_TESTS order (MSK_featurename_list.py)
LOINC_TO_ALL_INCLUDED_TESTS = {
    '6768-6': 'ALP',
    '1742-6': 'ALT',
    '1920-8': 'AST',
    '1751-7': 'Albumin',
    '1834-1': 'Alphafetoprotein',
    '3094-0': 'BUN',
    '704-7': 'Baso',
    '1952-1': 'Beta-2 Microglobulin',
    '1975-2': 'Bilirubin, Total',
    '26446-5': 'Blast',
    '2039-6': 'CEA',
    '2028-9': 'CO2',
    '17861-6': 'Calcium',
    '10334-1': 'Cancer Antigen 125',
    '6875-9': 'Cancer Antigen 15-3',
    '2010-7': 'Cancer Antigen 19-9',
    '2075-0': 'Chloride',
    '1968-7': 'Conjugated Bili (mg)',
    '2160-0': 'Creatinine',
    '711-2': 'Eos',
    '2345-7': 'Glucose',
    '4544-3': 'HCT',
    '718-7': 'HGB',
    '53115-2': 'Immature Granulocyte',
    '26463-0': 'Luc',
    '731-0': 'Lymph',
    '785-6': 'MCH',
    '786-4': 'MCHC',
    '787-2': 'MCV',
    '19252-6': 'Megakaryocyte Fragment',
    '28541-1': 'Meta',
    '742-7': 'Mono',
    '26498-6': 'Myelo',
    '751-8': 'Neut',
    '30392-5': 'Nucleated RBC',
    '2857-1': 'PSA',
    '777-3': 'Platelets',
    '2823-3': 'Potassium',
    '13598-8': 'Promyelo',
    '2885-2': 'Protein, Total',
    '789-8': 'RBC',
    '788-0': 'RDW',
    '2951-2': 'Sodium',
    '3016-3': 'TSH',
    '35050-8': 'Variant Lymph',
    '6690-2': 'WBC',
}

ALL_LOINC_MAPPINGS = {
    **LOINIC_TO_BASIC_TEST_NAMES,
    **LOINC_TO_VITALS_NAMES,
    **LOINC_TO_CBC_DIFFERENTIALS_NAMES
}
