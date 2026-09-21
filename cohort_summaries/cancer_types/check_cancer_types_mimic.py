#!/usr/bin/env python3
"""
check_cancer_types_mimic.py

Build a SPARC-style cancer-type dictionary for MIMIC-IV:

    DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MIMICIV_hosp.pickle
        {subject_id: {delta_day: [cancer_type, ...]}}

This follows the same approach as check_cancer_types_insight.py:
1. Uses the exact same ICD-9-CM / ICD-10-CM rule table to map billing codes
   to 30 canonical CAIA cancer types
2. Loads diagnoses_icd.csv.gz and links to admissions.csv.gz for dates
3. Converts dates to delta days (days from anchor_date, which is each patient's first admission)
4. Builds the nested dictionary format expected by eval_6_across_cancer_types_comparison.py

Usage:
    python check_cancer_types_mimic.py
    python check_cancer_types_mimic.py --rebuild-code-map  # force rebuild of ICD cache
    python check_cancer_types_mimic.py --include-in-situ   # include D00-D09 in situ codes
"""

import os
import sys
import argparse
import json
import pickle
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
MIMIC_RAW_PATH = '/data/morrisq/fuc/mimic_iv_3.1_mult/hosp'
OUTPUT_PATH = '/data/morrisq/fuc/SPARC_data/MIMICIV_hosp'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INSIGHT_CODE_MAP_PATH = f'{SCRIPT_DIR}/insight_cancer_types/icd_to_cancer_type_map_insight.json'

# ═══════════════════════════════════════════════════════════════════════════
# The 30 canonical CAIA cancer types  (from all_CAIA_cancer_types.txt)
# ═══════════════════════════════════════════════════════════════════════════
CANCER_TYPES = [
    'Appendix Cancer', 'Bladder Cancer', 'Bone Cancer', 'Breast Cancer',
    'Cervical Cancer', 'Colorectal Cancer', 'Endometrial Cancer',
    'Esophagogastric Cancer', 'GIST', 'Germ Cell Tumor', 'Glioma',
    'Head and Neck Cancer', 'Hepatobiliary Cancer', 'Leukemia', 'Lymphoma',
    'Mature T and NK Neoplasms', 'Mesothelioma', 'Myelodysplastic Syndromes',
    'Neuroendocrine Tumor', 'Non-Small Cell Lung Cancer', 'Ovarian Cancer',
    'Pancreatic Cancer', 'Prostate Cancer', 'Renal Cell Carcinoma', 'Sarcoma',
    'Skin Cancer', 'Small Cell Lung Cancer', 'Thyroid Cancer', 'Unknown Primary',
    'Uterine Sarcoma',
]
OTHERS = 'others'
SECONDARY = 'Metastasis (secondary site)'

# ═══════════════════════════════════════════════════════════════════════════
# ICD -> cancer type rule table (IDENTICAL to check_cancer_types_insight.py)
# ═══════════════════════════════════════════════════════════════════════════
ICD10_RULES = {
    # Head and neck
    'C00': ('Head and Neck Cancer', 'primary'), 'C01': ('Head and Neck Cancer', 'primary'),
    'C02': ('Head and Neck Cancer', 'primary'), 'C03': ('Head and Neck Cancer', 'primary'),
    'C04': ('Head and Neck Cancer', 'primary'), 'C05': ('Head and Neck Cancer', 'primary'),
    'C06': ('Head and Neck Cancer', 'primary'), 'C07': ('Head and Neck Cancer', 'primary'),
    'C08': ('Head and Neck Cancer', 'primary'), 'C09': ('Head and Neck Cancer', 'primary'),
    'C10': ('Head and Neck Cancer', 'primary'), 'C11': ('Head and Neck Cancer', 'primary'),
    'C12': ('Head and Neck Cancer', 'primary'), 'C13': ('Head and Neck Cancer', 'primary'),
    'C14': ('Head and Neck Cancer', 'primary'), 'C30': ('Head and Neck Cancer', 'primary'),
    'C31': ('Head and Neck Cancer', 'primary'), 'C32': ('Head and Neck Cancer', 'primary'),

    # Upper GI
    'C15': ('Esophagogastric Cancer', 'primary'), 'C16': ('Esophagogastric Cancer', 'primary'),
    'C17': (OTHERS, 'primary'),

    # Lower GI
    'C18': ('Colorectal Cancer', 'primary'), 'C181': ('Appendix Cancer', 'primary'),
    'C19': ('Colorectal Cancer', 'primary'), 'C20': ('Colorectal Cancer', 'primary'),
    'C21': ('Colorectal Cancer', 'primary'),

    # Hepatobiliary / pancreas
    'C22': ('Hepatobiliary Cancer', 'primary'), 'C23': ('Hepatobiliary Cancer', 'primary'),
    'C24': ('Hepatobiliary Cancer', 'primary'), 'C25': ('Pancreatic Cancer', 'primary'),
    'C26': (OTHERS, 'primary'),

    # Thorax
    'C33': ('Non-Small Cell Lung Cancer', 'primary'), 'C34': ('Non-Small Cell Lung Cancer', 'primary'),
    'C37': (OTHERS, 'primary'), 'C38': (OTHERS, 'primary'), 'C384': ('Mesothelioma', 'primary'),
    'C39': (OTHERS, 'primary'), 'C45': ('Mesothelioma', 'primary'),

    # Bone / soft tissue / skin
    'C40': ('Bone Cancer', 'primary'), 'C41': ('Bone Cancer', 'primary'),
    'C43': ('Skin Cancer', 'primary'), 'C44': ('Skin Cancer', 'primary'),
    'C4A': ('Skin Cancer', 'primary'), 'C46': ('Sarcoma', 'primary'),
    'C47': ('Sarcoma', 'primary'), 'C49': ('Sarcoma', 'primary'),
    'C49A': ('GIST', 'primary'),

    # Gynaecological
    'C48': ('Ovarian Cancer', 'primary'), 'C480': ('Sarcoma', 'primary'),
    'C50': ('Breast Cancer', 'primary'), 'C51': (OTHERS, 'primary'),
    'C52': (OTHERS, 'primary'), 'C53': ('Cervical Cancer', 'primary'),
    'C54': ('Endometrial Cancer', 'primary'), 'C55': ('Endometrial Cancer', 'primary'),
    'C56': ('Ovarian Cancer', 'primary'), 'C57': ('Ovarian Cancer', 'primary'),
    'C58': (OTHERS, 'primary'),

    # Male genital / urinary
    'C60': (OTHERS, 'primary'), 'C61': ('Prostate Cancer', 'primary'),
    'C62': ('Germ Cell Tumor', 'primary'), 'C63': (OTHERS, 'primary'),
    'C64': ('Renal Cell Carcinoma', 'primary'), 'C65': ('Bladder Cancer', 'primary'),
    'C66': ('Bladder Cancer', 'primary'), 'C67': ('Bladder Cancer', 'primary'),
    'C68': ('Bladder Cancer', 'primary'),

    # CNS / eye / endocrine
    'C69': (OTHERS, 'primary'), 'C70': (OTHERS, 'primary'),
    'C71': ('Glioma', 'primary'), 'C72': ('Glioma', 'primary'),
    'C73': ('Thyroid Cancer', 'primary'), 'C74': (OTHERS, 'primary'),
    'C75': (OTHERS, 'primary'), 'C7A': ('Neuroendocrine Tumor', 'primary'),

    # Unknown / secondary
    'C76': ('Unknown Primary', 'unknown'), 'C80': ('Unknown Primary', 'unknown'),
    'C77': (SECONDARY, 'secondary'), 'C78': (SECONDARY, 'secondary'),
    'C79': (SECONDARY, 'secondary'), 'C7B': (SECONDARY, 'secondary'),

    # Haematological
    'C81': ('Lymphoma', 'primary'), 'C82': ('Lymphoma', 'primary'),
    'C83': ('Lymphoma', 'primary'), 'C84': ('Mature T and NK Neoplasms', 'primary'),
    'C85': ('Lymphoma', 'primary'), 'C86': ('Mature T and NK Neoplasms', 'primary'),
    'C88': ('Lymphoma', 'primary'), 'C90': ('Lymphoma', 'primary'),
    'C91': ('Leukemia', 'primary'), 'C915': ('Mature T and NK Neoplasms', 'primary'),
    'C92': ('Leukemia', 'primary'), 'C93': ('Leukemia', 'primary'),
    'C94': ('Leukemia', 'primary'), 'C95': ('Leukemia', 'primary'),
    'C96': ('Lymphoma', 'primary'),
    'D46': ('Myelodysplastic Syndromes', 'primary'),
    'D45': (OTHERS, 'primary'), 'D47': (OTHERS, 'primary'),
    'D472': (None, None),  # MGUS - premalignant
    'D47Z2': (None, None), # Castleman - not malignant
    'D47Z1': ('Lymphoma', 'primary'),

    # Carcinoma in situ (kept only with --include-in-situ)
    'D00': ('Esophagogastric Cancer', 'in_situ'), 'D000': ('Head and Neck Cancer', 'in_situ'),
    'D01': ('Colorectal Cancer', 'in_situ'), 'D015': ('Hepatobiliary Cancer', 'in_situ'),
    'D017': (OTHERS, 'in_situ'), 'D019': (OTHERS, 'in_situ'),
    'D02': ('Non-Small Cell Lung Cancer', 'in_situ'), 'D020': ('Head and Neck Cancer', 'in_situ'),
    'D03': ('Skin Cancer', 'in_situ'), 'D04': ('Skin Cancer', 'in_situ'),
    'D05': ('Breast Cancer', 'in_situ'), 'D06': ('Cervical Cancer', 'in_situ'),
    'D07': (OTHERS, 'in_situ'), 'D070': ('Endometrial Cancer', 'in_situ'),
    'D075': ('Prostate Cancer', 'in_situ'), 'D09': (OTHERS, 'in_situ'),
    'D090': ('Bladder Cancer', 'in_situ'), 'D091': ('Bladder Cancer', 'in_situ'),
    'D093': ('Thyroid Cancer', 'in_situ'), 'D098': ('Unknown Primary', 'in_situ'),
    'D099': ('Unknown Primary', 'in_situ'),

    # Excluded
    'D3A': (None, None),
    **{f'D{i}': (None, None) for i in range(10, 50) if i not in [46, 45, 47]},
}

ICD09_RULES = {
    # Head and neck 140-149, 160-161
    '140': ('Head and Neck Cancer', 'primary'), '141': ('Head and Neck Cancer', 'primary'),
    '142': ('Head and Neck Cancer', 'primary'), '143': ('Head and Neck Cancer', 'primary'),
    '144': ('Head and Neck Cancer', 'primary'), '145': ('Head and Neck Cancer', 'primary'),
    '146': ('Head and Neck Cancer', 'primary'), '147': ('Head and Neck Cancer', 'primary'),
    '148': ('Head and Neck Cancer', 'primary'), '149': ('Head and Neck Cancer', 'primary'),
    '160': ('Head and Neck Cancer', 'primary'), '161': ('Head and Neck Cancer', 'primary'),

    # GI 150-159
    '150': ('Esophagogastric Cancer', 'primary'), '151': ('Esophagogastric Cancer', 'primary'),
    '152': (OTHERS, 'primary'), '153': ('Colorectal Cancer', 'primary'),
    '1535': ('Appendix Cancer', 'primary'), '154': ('Colorectal Cancer', 'primary'),
    '155': ('Hepatobiliary Cancer', 'primary'), '156': ('Hepatobiliary Cancer', 'primary'),
    '157': ('Pancreatic Cancer', 'primary'), '158': ('Ovarian Cancer', 'primary'),
    '1580': ('Sarcoma', 'primary'), '159': (OTHERS, 'primary'),

    # Thorax 162-165
    '162': ('Non-Small Cell Lung Cancer', 'primary'), '163': ('Mesothelioma', 'primary'),
    '164': (OTHERS, 'primary'), '165': (OTHERS, 'primary'),

    # Bone / soft tissue / skin / breast 170-176
    '170': ('Bone Cancer', 'primary'), '171': ('Sarcoma', 'primary'),
    '172': ('Skin Cancer', 'primary'), '173': ('Skin Cancer', 'primary'),
    '174': ('Breast Cancer', 'primary'), '175': ('Breast Cancer', 'primary'),
    '176': ('Sarcoma', 'primary'),

    # Gynaecological 179-184
    '179': ('Endometrial Cancer', 'primary'), '180': ('Cervical Cancer', 'primary'),
    '181': (OTHERS, 'primary'), '182': ('Endometrial Cancer', 'primary'),
    '183': ('Ovarian Cancer', 'primary'), '184': (OTHERS, 'primary'),

    # Male genital / urinary 185-189
    '185': ('Prostate Cancer', 'primary'), '186': ('Germ Cell Tumor', 'primary'),
    '187': (OTHERS, 'primary'), '188': ('Bladder Cancer', 'primary'),
    '189': ('Renal Cell Carcinoma', 'primary'),
    '1891': ('Bladder Cancer', 'primary'), '1892': ('Bladder Cancer', 'primary'),
    '1893': ('Bladder Cancer', 'primary'), '1894': ('Bladder Cancer', 'primary'),
    '1898': ('Bladder Cancer', 'primary'), '1899': ('Bladder Cancer', 'primary'),

    # CNS / eye / endocrine 190-194
    '190': (OTHERS, 'primary'), '191': ('Glioma', 'primary'),
    '192': ('Glioma', 'primary'), '193': ('Thyroid Cancer', 'primary'),
    '194': (OTHERS, 'primary'),

    # Unknown / secondary 195-199
    '195': ('Unknown Primary', 'unknown'), '196': (SECONDARY, 'secondary'),
    '197': (SECONDARY, 'secondary'), '198': (SECONDARY, 'secondary'),
    '199': ('Unknown Primary', 'unknown'),

    # Haematological 200-208
    '200': ('Lymphoma', 'primary'), '201': ('Lymphoma', 'primary'),
    '202': ('Lymphoma', 'primary'), '2021': ('Mature T and NK Neoplasms', 'primary'),
    '2022': ('Mature T and NK Neoplasms', 'primary'), '203': ('Lymphoma', 'primary'),
    '204': ('Leukemia', 'primary'), '205': ('Leukemia', 'primary'),
    '206': ('Leukemia', 'primary'), '207': ('Leukemia', 'primary'),
    '208': ('Leukemia', 'primary'),

    # Neuroendocrine 209
    '209': (None, None), '2090': ('Neuroendocrine Tumor', 'primary'),
    '2091': ('Neuroendocrine Tumor', 'primary'), '2092': ('Neuroendocrine Tumor', 'primary'),
    '2093': ('Neuroendocrine Tumor', 'primary'), '2097': (SECONDARY, 'secondary'),

    # Carcinoma in situ 230-234 (kept only with --include-in-situ)
    '230': (OTHERS, 'in_situ'), '2300': ('Head and Neck Cancer', 'in_situ'),
    '2301': ('Esophagogastric Cancer', 'in_situ'), '2302': ('Esophagogastric Cancer', 'in_situ'),
    '2303': ('Colorectal Cancer', 'in_situ'), '2304': ('Colorectal Cancer', 'in_situ'),
    '2305': ('Colorectal Cancer', 'in_situ'), '2306': ('Colorectal Cancer', 'in_situ'),
    '2308': ('Hepatobiliary Cancer', 'in_situ'),
    '231': ('Non-Small Cell Lung Cancer', 'in_situ'), '2310': ('Head and Neck Cancer', 'in_situ'),
    '232': ('Skin Cancer', 'in_situ'), '233': (OTHERS, 'in_situ'),
    '2330': ('Breast Cancer', 'in_situ'), '2331': ('Cervical Cancer', 'in_situ'),
    '2332': ('Endometrial Cancer', 'in_situ'), '2334': ('Prostate Cancer', 'in_situ'),
    '2337': ('Bladder Cancer', 'in_situ'), '2339': ('Bladder Cancer', 'in_situ'),
    '234': (OTHERS, 'in_situ'),

    # MDS from uncertain behaviour
    '23872': ('Myelodysplastic Syndromes', 'primary'),
    '23873': ('Myelodysplastic Syndromes', 'primary'),
    '23874': ('Myelodysplastic Syndromes', 'primary'),
    '23875': ('Myelodysplastic Syndromes', 'primary'),

    # Benign / uncertain: excluded
    **{str(c): (None, None) for c in range(210, 230)},
    **{str(c): (None, None) for c in range(235, 240)},
}


def normalise_code(code):
    """'C50.919' -> 'C50919';  ' 153.5 ' -> '1535'."""
    return str(code).strip().upper().replace('.', '').replace(' ', '')


def is_candidate_code(code_nodot, icd_version):
    """Cheap pre-filter: could this raw code possibly be a neoplasm code?"""
    if icd_version == 10:
        return code_nodot[:1] == 'C' or code_nodot[:2] in ('D0', 'D1', 'D2', 'D3', 'D4')
    if icd_version == 9:
        head = code_nodot[:3]
        return head.isdigit() and 140 <= int(head) <= 239
    return False


def resolve_code(code_nodot, icd_version):
    """Longest-prefix lookup. Returns (cancer_type, tier, matched_rule)."""
    rules = ICD10_RULES if icd_version == 10 else ICD09_RULES
    for n in range(len(code_nodot), 2, -1):
        hit = rules.get(code_nodot[:n])
        if hit is not None:
            return hit[0], hit[1], code_nodot[:n]
    return None, None, None


def load_insight_code_map(include_in_situ, include_secondary, drop_others):
    """Load the precomputed ICD code map from INSIGHT."""
    if not os.path.exists(INSIGHT_CODE_MAP_PATH):
        print(f'  ! Warning: INSIGHT code map not found at {INSIGHT_CODE_MAP_PATH}')
        print(f'    Falling back to rule-based lookup')
        return active_lookup_from_rules(include_in_situ, include_secondary, drop_others)

    with open(INSIGHT_CODE_MAP_PATH, 'r') as f:
        code_map = json.load(f)

    keep_tiers = {'primary', 'unknown'}
    if include_in_situ:
        keep_tiers.add('in_situ')
    if include_secondary:
        keep_tiers.add('secondary')

    # Convert to {(code_nodot, icd_version): cancer_type}
    lookup = {}
    for key, rec in code_map.items():
        # key format: "10|C50919" or "09|1535"
        code_type, code = key.split('|')
        icd_version = 10 if code_type == '10' else 9

        ctype = rec.get('cancer_type')
        tier = rec.get('tier')

        if ctype is None or tier not in keep_tiers:
            continue
        if drop_others and ctype == OTHERS:
            continue

        # Store with the matched rule (prefix) as key for longest-prefix matching
        rule = rec.get('matched_rule')
        if rule:
            lookup[(rule, icd_version)] = ctype

    print(f'  Loaded {len(lookup):,} codes from INSIGHT map')
    return lookup


def active_lookup_from_rules(include_in_situ, include_secondary, drop_others):
    """Build a lookup dict from the rule table (fallback if INSIGHT map unavailable)."""
    keep_tiers = {'primary', 'unknown'}
    if include_in_situ:
        keep_tiers.add('in_situ')
    if include_secondary:
        keep_tiers.add('secondary')

    lookup = {}
    for rules, version in [(ICD10_RULES, 10), (ICD09_RULES, 9)]:
        for code, (ctype, tier) in rules.items():
            if ctype is None or tier not in keep_tiers:
                continue
            if drop_others and ctype == OTHERS:
                continue
            lookup[(code, version)] = ctype
    return lookup


def main():
    parser = argparse.ArgumentParser(description='Build MIMIC-IV cancer type dictionary')
    parser.add_argument('--include-in-situ', action='store_true',
                        help='Include carcinoma in situ codes (D00-D09, 230-234)')
    parser.add_argument('--include-secondary', action='store_true',
                        help='Include metastasis/secondary site codes')
    parser.add_argument('--drop-others', action='store_true',
                        help='Drop the "others" bucket')
    parser.add_argument('--rebuild-code-map', action='store_true',
                        help='Force rebuild of code cache')
    args = parser.parse_args()

    print('=' * 78)
    print('MIMIC-IV cancer-type check')
    print(f'  raw data   : {MIMIC_RAW_PATH}')
    print(f'  output     : {OUTPUT_PATH}')
    print('=' * 78)

    # Build lookup
    print('\n[1/4] Building ICD code lookup table...')
    lookup = load_insight_code_map(args.include_in_situ, args.include_secondary, args.drop_others)
    print(f'  Active codes: {len(lookup):,}')

    # Load cancer patient cohort
    print('\n[2/4] Loading cancer patient cohort...')
    with open(f'{OUTPUT_PATH}/cancer_subject_ids.pkl', 'rb') as f:
        cancer_subjects = set(pickle.load(f))
    print(f'  Cancer patients: {len(cancer_subjects):,}')

    # Load anchor dates (date de-identification reference)
    print('\n[3/4] Loading anchor dates...')
    with open(f'{OUTPUT_PATH}/subject_to_anchor_date_dict.pkl', 'rb') as f:
        anchor_dates = pickle.load(f)
    print(f'  Anchor dates: {len(anchor_dates):,}')

    # Load diagnoses and link to admissions
    print('\n[4/4] Processing diagnoses...')
    print('  Loading diagnoses_icd.csv.gz...')
    diag_df = pd.read_csv(f'{MIMIC_RAW_PATH}/diagnoses_icd.csv.gz', compression='gzip')
    diag_df = diag_df[diag_df['subject_id'].isin(cancer_subjects)]
    print(f'    Diagnosis records for cancer patients: {len(diag_df):,}')

    print('  Loading admissions.csv.gz...')
    adm_df = pd.read_csv(f'{MIMIC_RAW_PATH}/admissions.csv.gz', compression='gzip',
                         usecols=['subject_id', 'hadm_id', 'admittime'])
    adm_df['admittime'] = pd.to_datetime(adm_df['admittime'])
    hadm_to_date = dict(zip(adm_df['hadm_id'], adm_df['admittime']))
    hadm_to_subject = dict(zip(adm_df['hadm_id'], adm_df['subject_id']))
    print(f'    Admissions loaded: {len(adm_df):,}')

    # Match diagnoses to cancer types
    print('  Matching diagnoses to cancer types...')
    subject_to_date_to_types = defaultdict(lambda: defaultdict(set))
    matched = 0
    no_date = 0

    for _, row in diag_df.iterrows():
        subject_id = row['subject_id']
        hadm_id = row['hadm_id']
        icd_code = normalise_code(row['icd_code'])
        icd_version = int(row['icd_version'])

        # Check if code matches a cancer type
        if not is_candidate_code(icd_code, icd_version):
            continue

        ctype, tier, rule = resolve_code(icd_code, icd_version)
        if ctype is None:
            continue

        key = (rule, icd_version)
        if key not in lookup:
            continue

        ctype = lookup[key]

        # Get admission date
        admit_date = hadm_to_date.get(hadm_id)
        if admit_date is None:
            no_date += 1
            continue

        # Convert to delta days
        anchor_date = anchor_dates.get(subject_id)
        if anchor_date is None:
            no_date += 1
            continue

        delta = (admit_date - anchor_date).days
        subject_to_date_to_types[subject_id][delta].add(ctype)
        matched += 1

    print(f'    Matched diagnoses: {matched:,}')
    print(f'    No date/anchor: {no_date:,}')
    print(f'    Patients with cancer type diagnoses: {len(subject_to_date_to_types):,}')

    # Convert sets to sorted lists
    result = {}
    for subject_id, date_dict in subject_to_date_to_types.items():
        result[subject_id] = {delta: sorted(list(types))
                             for delta, types in date_dict.items()}

    # Also create a calendar date version for inspection
    calendar_result = {}
    for subject_id, delta_dict in result.items():
        anchor_date = anchor_dates[subject_id]
        calendar_result[subject_id] = {
            (anchor_date + pd.Timedelta(days=delta)).strftime('%Y-%m-%d'): types
            for delta, types in delta_dict.items()
        }

    # Save outputs
    delta_path = f'{OUTPUT_PATH}/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MIMICIV_hosp.pickle'
    with open(delta_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'\n✓ Saved delta dictionary: {delta_path}')
    print(f'  {len(result):,} patients')

    calendar_path = f'{OUTPUT_PATH}/subjectid_to_CALENDARDATE_to_CANCER_TYPES_dict_MIMICIV_hosp.pickle'
    with open(calendar_path, 'wb') as f:
        pickle.dump(calendar_result, f)
    print(f'✓ Saved calendar dictionary: {calendar_path}')

    # Summary statistics
    print('\n' + '=' * 78)
    print('SUMMARY: Cancer types at earliest diagnosis')
    print('=' * 78)

    earliest_types = Counter()
    for subject_id, delta_dict in result.items():
        if not delta_dict:
            continue
        earliest_delta = min(delta_dict.keys())
        for ctype in delta_dict[earliest_delta]:
            earliest_types[ctype] += 1

    for ctype, count in earliest_types.most_common():
        print(f'  {ctype:<35} {count:>6,}')

    print('=' * 78)
    print('Done!')


if __name__ == '__main__':
    main()
