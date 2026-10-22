#!/usr/bin/env python3
"""
cancer_type_check_insight.py

Build, for every INSIGHT institution (A-F), a SPARC-style cancer-type dictionary

    DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_insight_{SITE}.pickle
        {ssid: {delta_day: [cancer_type, ...]}}

so that the earliest cancer diagnosis of a patient is simply

    earliest_delta = min(d.keys());  types = d[earliest_delta]

which is exactly how the MSK_CAIA equivalent
(DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_*.pickle) is consumed by
results_vis/test_fixed*/eval_6_across_cancer_types_comparison.py and
clinical_benchmark_helpers.py.


Two stages, exactly as requested
--------------------------------
(1) CODE DICTIONARY.  Every ICD-9-CM / ICD-10-CM billing code that occurs in the
    INSIGHT neoplasm range is resolved to one of the 30 canonical CAIA cancer
    types (cohort_summaries/fixed/all_CAIA_cancer_types.txt) plus an `others`
    bucket, using the explicit, auditable rule table below (ICD10_RULES /
    ICD09_RULES).  The expansion over the codes actually present in the data is
    cached to

        icd_to_cancer_type_map_insight.json   (machine readable, the cache)
        icd_to_cancer_type_map_insight.csv    (human readable: code, type,
                                               tier, matched rule, n rows)

    Re-runs load the JSON cache instead of re-scanning; --rebuild-code-map
    forces a rescan.  The rule table is the single source of truth, so the
    mapping is fully reproducible: delete the cache, re-run, diff the CSV.

(2) PER-PERSON MATCH.  condition.csv and diagnosis.csv are streamed, rows are
    kept when the patient is in the processed cohort and the code is in the
    map, and {ssid: {date: {types}}} is accumulated and written per site.


Which patients are "included"
-----------------------------
The Insight preprocessing pipeline (SPARC_data_preprocessing_insight)
defines the cohort as follows:

  * run_preprocessing_split.py assigns a patient to a site with the
    ssid -> site_id map read from demographics.csv (207,041 unique ssids), and
    writes one pickle set per site.  NOTE: the upstream per-pipeline builders
    key on ssid ALONE; they never filter the raw CSV rows by site_id.  We do the
    same here -- rows are matched on ssid, and the site of a patient is the
    demographics.csv site.  (condition.csv/diagnosis.csv carry their own
    reporting-site column which does NOT always agree with demographics.csv;
    following the upstream convention avoids silently dropping records.)

  * create_splits.py then defines the usable cohort per site as
        all_pids = sorted(labs.keys())            # ssid_to_date_to_labs_dict
    i.e. a patient counts as included iff they have at least one lab record.
    The train/val/test JSONs under insight_stratified/split/ are built from
    exactly that set.

  --cohort-source labs   (default) uses the keys of each site's
                         ssid_to_date_to_labs_dict.pickle
  --cohort-source splits uses the union of insight_stratified/split/{train,
                         val,test}/{SITE}.json  (cheap; 9 fewer patients in A)


Date convention (the "delta" in the filename)
---------------------------------------------
All Insight pickles use
    delta = (event_date - reference_date).days
    reference_date = first result_date in lab_result_cm.csv for that patient,
                     falling back to the earliest enr_start_date in
                     enrollment.csv for patients with no labs
(see SPARC_data_preprocessing_insight/shared/utils.build_reference_dates).

Recovering those reference dates requires one pass over the 53 GB
lab_result_cm.csv, so the result is cached to

    ssid_to_reference_date_insight.pickle

Build it once with --build-reference-dates (or let the script do it on first
run).  A second dictionary keyed by real ISO calendar dates is always written
alongside the delta one, so the mapping can be checked without the reference
dates:

    ssid_to_CALENDARDATE_to_CANCER_TYPES_dict_insight_{SITE}.pickle
        {ssid: {'YYYY-MM-DD': [cancer_type, ...]}}


Usage
-----
  # everything (builds the reference-date cache on first run);
  # --raw-dir and --processed-dir are required for every run except --self-test
  python cancer_type_check_insight.py --raw-dir /path/to/insight-data-raw \\
      --processed-dir /path/to/insight-data-processed

  # only rebuild the ICD -> cancer type dictionary and inspect it
  python cancer_type_check_insight.py --rebuild-code-map --code-map-only

  # calendar-date pickles only, skip the 53 GB lab scan
  python cancer_type_check_insight.py --no-delta

  # a single site, and include in-situ / metastatic codes
  python cancer_type_check_insight.py --sites A --include-in-situ --include-secondary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
# The INSIGHT data location is never defaulted: --raw-dir and --processed-dir are
# required, and configure_paths() fills these in from them.
RAW_DIR = PROCESSED_DIR = SPLIT_DIR = None
CONDITION_CSV = DIAGNOSIS_CSV = DEMOGRAPHICS_CSV = ENROLLMENT_CSV = LAB_CSV = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_SCRIPT_DIR, 'insight_cancer_types')


def configure_paths(raw_dir, processed_dir):
    global RAW_DIR, PROCESSED_DIR, SPLIT_DIR
    global CONDITION_CSV, DIAGNOSIS_CSV, DEMOGRAPHICS_CSV, ENROLLMENT_CSV, LAB_CSV
    RAW_DIR, PROCESSED_DIR = raw_dir, processed_dir
    SPLIT_DIR = os.path.join(PROCESSED_DIR, 'insight_stratified', 'split')
    CONDITION_CSV    = os.path.join(RAW_DIR, 'condition.csv')
    DIAGNOSIS_CSV    = os.path.join(RAW_DIR, 'diagnosis.csv')
    DEMOGRAPHICS_CSV = os.path.join(RAW_DIR, 'demographics.csv')
    ENROLLMENT_CSV   = os.path.join(RAW_DIR, 'enrollment.csv')
    LAB_CSV          = os.path.join(RAW_DIR, 'lab_result_cm.csv')


SITES = ['A', 'B', 'C', 'D', 'E', 'F']
CHUNK_SIZE = 500_000


# ═══════════════════════════════════════════════════════════════════════════
# The 30 canonical CAIA cancer types  (cohort_summaries/fixed/all_CAIA_cancer_types.txt)
# ═══════════════════════════════════════════════════════════════════════════
CANCER_TYPES = [
    'Appendix Cancer',
    'Bladder Cancer',
    'Bone Cancer',
    'Breast Cancer',
    'Cervical Cancer',
    'Colorectal Cancer',
    'Endometrial Cancer',
    'Esophagogastric Cancer',
    'GIST',
    'Germ Cell Tumor',
    'Glioma',
    'Head and Neck Cancer',
    'Hepatobiliary Cancer',
    'Leukemia',
    'Lymphoma',
    'Mature T and NK Neoplasms',
    'Mesothelioma',
    'Myelodysplastic Syndromes',
    'Neuroendocrine Tumor',
    'Non-Small Cell Lung Cancer',
    'Ovarian Cancer',
    'Pancreatic Cancer',
    'Prostate Cancer',
    'Renal Cell Carcinoma',
    'Sarcoma',
    'Skin Cancer',
    'Small Cell Lung Cancer',
    'Thyroid Cancer',
    'Unknown Primary',
    'Uterine Sarcoma',
]
# 31st bucket, mirroring the 'others' entry of OMOP_maps.PRIMARY_CTYPES: a
# malignant neoplasm that is genuinely none of the 30 (thymus, vulva, penis,
# eye, adrenal, myeloproliferative neoplasms, ...).  Kept rather than dropped so
# that "earliest cancer diagnosis" is not silently moved later in time.
OTHERS = 'others'

# Codes that prove cancer but name no primary site.  Emitted as a separate
# label only with --include-secondary, because C77-C79 is a metastasis, not a
# primary type, and would otherwise win the earliest-date tie-break.
SECONDARY = 'Metastasis (secondary site)'

# ── Two of the 30 types cannot be produced from billing codes ───────────────
# 'Small Cell Lung Cancer' : ICD-10-CM C34.x / ICD-9-CM 162.x encode the SITE
#     only; small-cell vs non-small-cell is a histology (ICD-O-3 morphology)
#     distinction that PCORnet CDM does not carry.  Every lung primary is
#     therefore mapped to 'Non-Small Cell Lung Cancer', the majority histology.
# 'Uterine Sarcoma'        : likewise indistinguishable from endometrial
#     carcinoma at C54.x / 182.x, so uterine-corpus primaries all map to
#     'Endometrial Cancer'.
# Both are flagged in the run report so the limitation is never invisible.
UNREACHABLE_TYPES = ['Small Cell Lung Cancer', 'Uterine Sarcoma']


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — the ICD -> cancer type rule table
# ═══════════════════════════════════════════════════════════════════════════
# Codes are normalised to UPPERCASE with the decimal point removed before
# matching ('C50.919' -> 'C50919', '153.5' -> '1535').  The LONGEST matching
# prefix wins, so a specific subcategory always overrides its parent.
#
# tier semantics
#   'primary'   a named primary site -> one of the 30 types (or `others`)
#   'unknown'   cancer proven, primary site not stated  -> 'Unknown Primary'
#   'secondary' metastasis / secondary site             -> kept only with
#                                                          --include-secondary
#   'in_situ'   carcinoma in situ                       -> kept only with
#                                                          --include-in-situ
#   None        deliberately excluded (benign, uncertain behaviour, ...)

ICD10_RULES: dict[str, tuple[str | None, str]] = {
    # ── Head and neck: lip, oral cavity, pharynx, salivary, sinuses, larynx ──
    'C00': ('Head and Neck Cancer', 'primary'),   # lip
    'C01': ('Head and Neck Cancer', 'primary'),   # base of tongue
    'C02': ('Head and Neck Cancer', 'primary'),   # other tongue
    'C03': ('Head and Neck Cancer', 'primary'),   # gum
    'C04': ('Head and Neck Cancer', 'primary'),   # floor of mouth
    'C05': ('Head and Neck Cancer', 'primary'),   # palate
    'C06': ('Head and Neck Cancer', 'primary'),   # other mouth
    'C07': ('Head and Neck Cancer', 'primary'),   # parotid gland
    'C08': ('Head and Neck Cancer', 'primary'),   # other salivary glands
    'C09': ('Head and Neck Cancer', 'primary'),   # tonsil
    'C10': ('Head and Neck Cancer', 'primary'),   # oropharynx
    'C11': ('Head and Neck Cancer', 'primary'),   # nasopharynx
    'C12': ('Head and Neck Cancer', 'primary'),   # pyriform sinus
    'C13': ('Head and Neck Cancer', 'primary'),   # hypopharynx
    'C14': ('Head and Neck Cancer', 'primary'),   # other lip/oral/pharynx
    'C30': ('Head and Neck Cancer', 'primary'),   # nasal cavity, middle ear
    'C31': ('Head and Neck Cancer', 'primary'),   # accessory sinuses
    'C32': ('Head and Neck Cancer', 'primary'),   # larynx

    # ── Upper GI ─────────────────────────────────────────────────────────────
    'C15': ('Esophagogastric Cancer', 'primary'), # oesophagus
    'C16': ('Esophagogastric Cancer', 'primary'), # stomach
    'C17': (OTHERS, 'primary'),                   # small intestine

    # ── Lower GI ─────────────────────────────────────────────────────────────
    'C18':  ('Colorectal Cancer', 'primary'),     # colon
    'C181': ('Appendix Cancer',   'primary'),     # appendix  (overrides C18)
    'C19':  ('Colorectal Cancer', 'primary'),     # rectosigmoid junction
    'C20':  ('Colorectal Cancer', 'primary'),     # rectum
    'C21':  ('Colorectal Cancer', 'primary'),     # anus / anal canal

    # ── Hepatobiliary / pancreas ─────────────────────────────────────────────
    'C22': ('Hepatobiliary Cancer', 'primary'),   # liver, intrahepatic ducts
    'C23': ('Hepatobiliary Cancer', 'primary'),   # gallbladder
    'C24': ('Hepatobiliary Cancer', 'primary'),   # other biliary tract
    'C25': ('Pancreatic Cancer',    'primary'),
    'C26': (OTHERS, 'primary'),                   # ill-defined digestive

    # ── Thorax ───────────────────────────────────────────────────────────────
    'C33':  ('Non-Small Cell Lung Cancer', 'primary'),  # trachea
    'C34':  ('Non-Small Cell Lung Cancer', 'primary'),  # bronchus and lung
    'C37':  (OTHERS, 'primary'),                        # thymus
    'C38':  (OTHERS, 'primary'),                        # heart, mediastinum
    'C384': ('Mesothelioma', 'primary'),                # pleura
    'C39':  (OTHERS, 'primary'),                        # ill-defined respiratory
    'C45':  ('Mesothelioma', 'primary'),

    # ── Bone / soft tissue / skin ────────────────────────────────────────────
    'C40':  ('Bone Cancer', 'primary'),
    'C41':  ('Bone Cancer', 'primary'),
    'C43':  ('Skin Cancer', 'primary'),           # malignant melanoma
    'C44':  ('Skin Cancer', 'primary'),           # other/unspecified skin
    'C4A':  ('Skin Cancer', 'primary'),           # Merkel cell carcinoma
    'C46':  ('Sarcoma', 'primary'),               # Kaposi sarcoma
    'C47':  ('Sarcoma', 'primary'),               # peripheral nerves / ANS
    'C49':  ('Sarcoma', 'primary'),               # connective / soft tissue
    'C49A': ('GIST', 'primary'),                  # gastrointestinal stromal tumour

    # ── Gynaecological ───────────────────────────────────────────────────────
    'C48':  ('Ovarian Cancer', 'primary'),        # peritoneum (C48.1/C48.2):
                                                  # primary peritoneal serous carcinoma,
                                                  # grouped with ovarian in OncoTree
    'C480': ('Sarcoma', 'primary'),               # retroperitoneum.  C49 carries an
                                                  # Excludes1 for retroperitoneum, so
                                                  # retroperitoneal soft-tissue sarcoma
                                                  # (liposarcoma, leiomyosarcoma) MUST be
                                                  # coded here and cannot appear as C49.x
    'C50': ('Breast Cancer',      'primary'),
    'C51': (OTHERS, 'primary'),                   # vulva
    'C52': (OTHERS, 'primary'),                   # vagina
    'C53': ('Cervical Cancer',    'primary'),
    'C54': ('Endometrial Cancer', 'primary'),     # uterine corpus (see UNREACHABLE_TYPES)
    'C55': ('Endometrial Cancer', 'primary'),     # uterus, part unspecified
    'C56': ('Ovarian Cancer',     'primary'),
    'C57': ('Ovarian Cancer',     'primary'),     # fallopian tube & other adnexa
    'C58': (OTHERS, 'primary'),                   # placenta

    # ── Male genital / urinary ───────────────────────────────────────────────
    'C60': (OTHERS, 'primary'),                   # penis
    'C61': ('Prostate Cancer',   'primary'),
    'C62': ('Germ Cell Tumor',   'primary'),      # testis
    'C63': (OTHERS, 'primary'),                   # other male genital
    'C64': ('Renal Cell Carcinoma', 'primary'),   # kidney, except renal pelvis
    'C65': ('Bladder Cancer', 'primary'),         # renal pelvis      \
    'C66': ('Bladder Cancer', 'primary'),         # ureter             > urothelial
    'C67': ('Bladder Cancer', 'primary'),         # bladder            > tract
    'C68': ('Bladder Cancer', 'primary'),         # other urinary     /

    # ── CNS / eye / endocrine ────────────────────────────────────────────────
    'C69': (OTHERS, 'primary'),                   # eye and adnexa
    'C70': (OTHERS, 'primary'),                   # meninges
    'C71': ('Glioma', 'primary'),                 # brain
    'C72': ('Glioma', 'primary'),                 # spinal cord, cranial nerves
    'C73': ('Thyroid Cancer', 'primary'),
    'C74': (OTHERS, 'primary'),                   # adrenal gland
    'C75': (OTHERS, 'primary'),                   # other endocrine
    'C7A': ('Neuroendocrine Tumor', 'primary'),   # malignant neuroendocrine tumours

    # ── No stated primary site ───────────────────────────────────────────────
    'C76': ('Unknown Primary', 'unknown'),        # other and ill-defined sites
    'C80': ('Unknown Primary', 'unknown'),        # site unspecified / disseminated

    # ── Secondary (metastatic) sites ─────────────────────────────────────────
    'C77': (SECONDARY, 'secondary'),              # lymph nodes
    'C78': (SECONDARY, 'secondary'),              # respiratory / digestive
    'C79': (SECONDARY, 'secondary'),              # other sites
    'C7B': (SECONDARY, 'secondary'),              # secondary neuroendocrine

    # ── Haematological ───────────────────────────────────────────────────────
    'C81':  ('Lymphoma', 'primary'),              # Hodgkin lymphoma
    'C82':  ('Lymphoma', 'primary'),              # follicular lymphoma
    'C83':  ('Lymphoma', 'primary'),              # non-follicular lymphoma
    'C84':  ('Mature T and NK Neoplasms', 'primary'),
    'C85':  ('Lymphoma', 'primary'),              # other/unspecified NHL
    'C86':  ('Mature T and NK Neoplasms', 'primary'),
    'C88':  ('Lymphoma', 'primary'),              # malignant immunoproliferative
    'C90':  ('Lymphoma', 'primary'),              # multiple myeloma / plasma cell.
                                                  # JUDGEMENT CALL: myeloma is not a
                                                  # lymphoma, but the 30-type list has no
                                                  # plasma-cell category and OncoTree files
                                                  # both under Mature B-Cell Neoplasms, so
                                                  # 'Lymphoma' is the nearest of the 30.
                                                  # ~400k rows — retarget here if you would
                                                  # rather they land in `others`.
    'C91':  ('Leukemia', 'primary'),              # lymphoid leukaemia
    'C915': ('Mature T and NK Neoplasms', 'primary'),  # adult T-cell lymphoma/leukaemia
    'C92':  ('Leukemia', 'primary'),              # myeloid leukaemia
    'C93':  ('Leukemia', 'primary'),              # monocytic leukaemia
    'C94':  ('Leukemia', 'primary'),              # other specified leukaemias
    'C95':  ('Leukemia', 'primary'),              # leukaemia of unspecified cell type
    'C96':  ('Lymphoma', 'primary'),              # other lymphoid/haematopoietic

    'D46':  ('Myelodysplastic Syndromes', 'primary'),
    'D45':  (OTHERS, 'primary'),                  # polycythaemia vera (a true MPN)
    'D47':  (OTHERS, 'primary'),                  # other chronic myeloproliferative:
                                                  # D47.1 CMPD, D47.3 essential
                                                  # thrombocythaemia, D47.4 myelofibrosis,
                                                  # D47.0 mast cell neoplasms
    'D472': (None, None),                         # MGUS — a premalignant clonal state,
                                                  # NOT a malignancy.  The single highest
                                                  # volume code in the D47 family; letting
                                                  # it through would hand tens of thousands
                                                  # of patients a false cancer date.
    'D47Z2':(None, None),                         # Castleman disease — lymphoproliferative,
                                                  # not malignant
    'D47Z1':('Lymphoma', 'primary'),              # post-transplant lymphoproliferative
                                                  # disorder — managed as a lymphoma

    # ── Carcinoma in situ (kept only with --include-in-situ) ─────────────────
    'D00':  ('Esophagogastric Cancer', 'in_situ'),  # D00.1 oesophagus, D00.2 stomach
    'D000': ('Head and Neck Cancer', 'in_situ'),    # D00.0x lip / oral cavity / pharynx —
                                                    # the same split as D02.0 larynx below,
                                                    # and the majority of the D00 family
    'D01':  ('Colorectal Cancer', 'in_situ'),       # other digestive
    'D015': ('Hepatobiliary Cancer', 'in_situ'),
    'D017': (OTHERS, 'in_situ'),
    'D019': (OTHERS, 'in_situ'),
    'D02':  ('Non-Small Cell Lung Cancer', 'in_situ'),  # respiratory / middle ear
    'D020': ('Head and Neck Cancer', 'in_situ'),        # larynx
    'D03':  ('Skin Cancer', 'in_situ'),             # melanoma in situ
    'D04':  ('Skin Cancer', 'in_situ'),             # other skin in situ
    'D05':  ('Breast Cancer', 'in_situ'),
    'D06':  ('Cervical Cancer', 'in_situ'),
    'D07':  (OTHERS, 'in_situ'),                    # other genital organs
    'D070': ('Endometrial Cancer', 'in_situ'),
    'D075': ('Prostate Cancer', 'in_situ'),
    'D09':  (OTHERS, 'in_situ'),                    # other/unspecified in situ
    'D090': ('Bladder Cancer', 'in_situ'),
    'D091': ('Bladder Cancer', 'in_situ'),          # other urinary
    'D093': ('Thyroid Cancer', 'in_situ'),          # thyroid & other endocrine
    'D098': ('Unknown Primary', 'in_situ'),
    'D099': ('Unknown Primary', 'in_situ'),

    # ── Explicitly excluded ──────────────────────────────────────────────────
    'D3A': (None, None),   # benign neuroendocrine tumours
    'D10': (None, None), 'D11': (None, None), 'D12': (None, None),
    'D13': (None, None), 'D14': (None, None), 'D15': (None, None),
    'D16': (None, None), 'D17': (None, None), 'D18': (None, None),
    'D19': (None, None), 'D20': (None, None), 'D21': (None, None),
    'D22': (None, None), 'D23': (None, None), 'D24': (None, None),
    'D25': (None, None), 'D26': (None, None), 'D27': (None, None),
    'D28': (None, None), 'D29': (None, None), 'D30': (None, None),
    'D31': (None, None), 'D32': (None, None), 'D33': (None, None),
    'D34': (None, None), 'D35': (None, None), 'D36': (None, None),
    'D37': (None, None), 'D38': (None, None), 'D39': (None, None),
    'D40': (None, None), 'D41': (None, None), 'D42': (None, None),
    'D43': (None, None), 'D44': (None, None), 'D48': (None, None),
    'D49': (None, None),   # neoplasm of unspecified behaviour
}

ICD09_RULES: dict[str, tuple[str | None, str]] = {
    # ── Head and neck 140-149, 160-161 ───────────────────────────────────────
    '140': ('Head and Neck Cancer', 'primary'), '141': ('Head and Neck Cancer', 'primary'),
    '142': ('Head and Neck Cancer', 'primary'), '143': ('Head and Neck Cancer', 'primary'),
    '144': ('Head and Neck Cancer', 'primary'), '145': ('Head and Neck Cancer', 'primary'),
    '146': ('Head and Neck Cancer', 'primary'), '147': ('Head and Neck Cancer', 'primary'),
    '148': ('Head and Neck Cancer', 'primary'), '149': ('Head and Neck Cancer', 'primary'),
    '160': ('Head and Neck Cancer', 'primary'),   # nasal cavity / sinuses
    '161': ('Head and Neck Cancer', 'primary'),   # larynx

    # ── GI 150-159 ───────────────────────────────────────────────────────────
    '150':  ('Esophagogastric Cancer', 'primary'),  # oesophagus
    '151':  ('Esophagogastric Cancer', 'primary'),  # stomach
    '152':  (OTHERS, 'primary'),                    # small intestine
    '153':  ('Colorectal Cancer', 'primary'),       # colon
    '1535': ('Appendix Cancer',   'primary'),       # appendix (overrides 153)
    '154':  ('Colorectal Cancer', 'primary'),       # rectum / rectosigmoid / anus
    '155':  ('Hepatobiliary Cancer', 'primary'),    # liver / intrahepatic ducts
    '156':  ('Hepatobiliary Cancer', 'primary'),    # gallbladder / biliary
    '157':  ('Pancreatic Cancer', 'primary'),
    '158':  ('Ovarian Cancer', 'primary'),          # peritoneum (158.8/158.9)
    '1580': ('Sarcoma', 'primary'),                 # retroperitoneum — same reasoning as
                                                    # ICD-10 C48.0 (171 excludes it too)
    '159':  (OTHERS, 'primary'),                    # ill-defined digestive

    # ── Thorax 162-165 ───────────────────────────────────────────────────────
    '162': ('Non-Small Cell Lung Cancer', 'primary'),
    '163': ('Mesothelioma', 'primary'),             # pleura
    '164': (OTHERS, 'primary'),                     # thymus, heart, mediastinum
    '165': (OTHERS, 'primary'),                     # ill-defined respiratory

    # ── Bone / soft tissue / skin / breast 170-176 ───────────────────────────
    '170': ('Bone Cancer', 'primary'),
    '171': ('Sarcoma', 'primary'),                  # connective / soft tissue
    '172': ('Skin Cancer', 'primary'),              # malignant melanoma
    '173': ('Skin Cancer', 'primary'),              # other skin
    '174': ('Breast Cancer', 'primary'),            # female breast
    '175': ('Breast Cancer', 'primary'),            # male breast
    '176': ('Sarcoma', 'primary'),                  # Kaposi sarcoma

    # ── Gynaecological 179-184 ───────────────────────────────────────────────
    '179': ('Endometrial Cancer', 'primary'),       # uterus, part unspecified
    '180': ('Cervical Cancer', 'primary'),
    '181': (OTHERS, 'primary'),                     # placenta
    '182': ('Endometrial Cancer', 'primary'),       # uterine body
    '183': ('Ovarian Cancer', 'primary'),           # ovary & uterine adnexa
    '184': (OTHERS, 'primary'),                     # other female genital

    # ── Male genital / urinary 185-189 ───────────────────────────────────────
    '185':  ('Prostate Cancer', 'primary'),
    '186':  ('Germ Cell Tumor', 'primary'),         # testis
    '187':  (OTHERS, 'primary'),                    # other male genital
    '188':  ('Bladder Cancer', 'primary'),
    # 189 = "kidney and other and unspecified urinary organs" mixes the renal
    # parenchyma with the urothelial tract, so every subcode is spelled out and
    # the bare 3-digit category (a non-billable header that still occurs in the
    # data) follows the plurality of the family, which is kidney.
    '189':  ('Renal Cell Carcinoma', 'primary'),    # 189.0 kidney except pelvis (-> C64)
                                                    # and the bare category both land here
    '1891': ('Bladder Cancer', 'primary'),          # renal pelvis           (-> C65)
    '1892': ('Bladder Cancer', 'primary'),          # ureter                 (-> C66)
    '1893': ('Bladder Cancer', 'primary'),          # urethra                (-> C68.0)
    '1894': ('Bladder Cancer', 'primary'),          # paraurethral glands
    '1898': ('Bladder Cancer', 'primary'),          # other urinary organs
    '1899': ('Bladder Cancer', 'primary'),          # urinary organ, site unspecified

    # ── CNS / eye / endocrine 190-194 ────────────────────────────────────────
    '190': (OTHERS, 'primary'),                     # eye
    '191': ('Glioma', 'primary'),                   # brain
    '192': ('Glioma', 'primary'),                   # other nervous system
    '193': ('Thyroid Cancer', 'primary'),
    '194': (OTHERS, 'primary'),                     # other endocrine

    # ── Unknown / secondary 195-199 ──────────────────────────────────────────
    '195': ('Unknown Primary', 'unknown'),          # other ill-defined sites
    '196': (SECONDARY, 'secondary'),                # lymph nodes
    '197': (SECONDARY, 'secondary'),                # respiratory / digestive
    '198': (SECONDARY, 'secondary'),                # other specified sites
    '199': ('Unknown Primary', 'unknown'),          # site unspecified

    # ── Haematological 200-208 ───────────────────────────────────────────────
    '200':  ('Lymphoma', 'primary'),                # reticulosarcoma / NHL
    '201':  ('Lymphoma', 'primary'),                # Hodgkin
    '202':  ('Lymphoma', 'primary'),                # other lymphoid/histiocytic
    '2021': ('Mature T and NK Neoplasms', 'primary'),  # mycosis fungoides
    '2022': ('Mature T and NK Neoplasms', 'primary'),  # Sezary disease
    '203':  ('Lymphoma', 'primary'),                # multiple myeloma / plasma cell
    '204':  ('Leukemia', 'primary'), '205': ('Leukemia', 'primary'),
    '206':  ('Leukemia', 'primary'), '207': ('Leukemia', 'primary'),
    '208':  ('Leukemia', 'primary'),

    # ── Neuroendocrine 209 ───────────────────────────────────────────────────
    '209':  (None, None),                           # benign carcinoid (209.4-.6)
    '2090': ('Neuroendocrine Tumor', 'primary'),    # malignant carcinoid, small bowel
    '2091': ('Neuroendocrine Tumor', 'primary'),    # malignant carcinoid, large bowel
    '2092': ('Neuroendocrine Tumor', 'primary'),    # malignant carcinoid, other
    '2093': ('Neuroendocrine Tumor', 'primary'),    # malignant carcinoid, unspecified
    '2097': (SECONDARY, 'secondary'),               # secondary neuroendocrine

    # ── Carcinoma in situ 230-234 (kept only with --include-in-situ) ─────────
    '230':  (OTHERS, 'in_situ'),
    '2300': ('Head and Neck Cancer',      'in_situ'),
    '2301': ('Esophagogastric Cancer',    'in_situ'),
    '2302': ('Esophagogastric Cancer',    'in_situ'),
    '2303': ('Colorectal Cancer',         'in_situ'),
    '2304': ('Colorectal Cancer',         'in_situ'),
    '2305': ('Colorectal Cancer',         'in_situ'),
    '2306': ('Colorectal Cancer',         'in_situ'),
    '2308': ('Hepatobiliary Cancer',      'in_situ'),
    '231':  ('Non-Small Cell Lung Cancer','in_situ'),
    '2310': ('Head and Neck Cancer',      'in_situ'),   # larynx
    '232':  ('Skin Cancer',               'in_situ'),
    '233':  (OTHERS, 'in_situ'),
    '2330': ('Breast Cancer',      'in_situ'),
    '2331': ('Cervical Cancer',    'in_situ'),
    '2332': ('Endometrial Cancer', 'in_situ'),
    '2334': ('Prostate Cancer',    'in_situ'),
    '2337': ('Bladder Cancer',     'in_situ'),
    '2339': ('Bladder Cancer',     'in_situ'),
    '234':  (OTHERS, 'in_situ'),

    # ── Uncertain behaviour 235-239: excluded, except MDS ────────────────────
    '235': (None, None), '236': (None, None), '237': (None, None),
    '238': (None, None), '239': (None, None),
    '23872': ('Myelodysplastic Syndromes', 'primary'),  # refractory anaemia w/o sideroblasts
    '23873': ('Myelodysplastic Syndromes', 'primary'),  # refractory anaemia w/ sideroblasts
    '23874': ('Myelodysplastic Syndromes', 'primary'),  # MDS with del(5q)
    '23875': ('Myelodysplastic Syndromes', 'primary'),  # MDS, unspecified

    # ── Benign 210-229: excluded ─────────────────────────────────────────────
    **{str(c): (None, None) for c in range(210, 230)},
}

# ICD-9-CM code-family ranges that are neoplasms at all (used to decide which
# raw rows are even worth looking at).
ICD09_NEOPLASM_LO, ICD09_NEOPLASM_HI = 140, 239


def normalise_code(code: str) -> str:
    """'C50.919' -> 'C50919';  ' 153.5 ' -> '1535'."""
    return code.strip().upper().replace('.', '').replace(' ', '')


def is_candidate_code(code_nodot: str, code_type: str) -> bool:
    """Cheap pre-filter: could this raw code possibly be a neoplasm code?"""
    if code_type == '10':
        return code_nodot[:1] == 'C' or code_nodot[:2] in ('D0', 'D1', 'D2', 'D3', 'D4')
    if code_type == '09':
        head = code_nodot[:3]
        return head.isdigit() and ICD09_NEOPLASM_LO <= int(head) <= ICD09_NEOPLASM_HI
    return False   # 'SM' (SNOMED) and anything else: not a billing ICD code


def resolve_code(code_nodot: str, code_type: str):
    """Longest-prefix lookup.  Returns (cancer_type, tier, matched_rule)."""
    rules = ICD10_RULES if code_type == '10' else ICD09_RULES
    for n in range(len(code_nodot), 2, -1):
        hit = rules.get(code_nodot[:n])
        if hit is not None:
            return hit[0], hit[1], code_nodot[:n]
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — expand the rule table over the codes present in the data, and cache
# ═══════════════════════════════════════════════════════════════════════════
def scan_observed_codes(paths_and_cols) -> Counter:
    """Count every (code_type, raw_code) pair in the neoplasm range."""
    observed = Counter()
    for path, code_col, type_col in paths_and_cols:
        if not os.path.exists(path):
            print(f'  ! missing, skipped: {path}')
            continue
        print(f'  scanning {os.path.basename(path)} ...', flush=True)
        n = 0
        for chunk in pd.read_csv(path, dtype=str, chunksize=CHUNK_SIZE,
                                 usecols=[code_col, type_col]):
            n += len(chunk)
            codes = chunk[code_col].fillna('').str.upper().str.replace('.', '', regex=False).str.strip()
            types = chunk[type_col].fillna('')
            for c, t in zip(codes, types):
                if c and is_candidate_code(c, t):
                    observed[(t, c)] += 1
            if n % (CHUNK_SIZE * 40) == 0:
                print(f'    {n/1e6:.0f}M rows, {len(observed):,} distinct codes', flush=True)
        print(f'    done: {n:,} rows', flush=True)
    return observed


def build_code_map(out_dir: str, rebuild: bool) -> dict:
    """Return {f'{code_type}|{code_nodot}': {...}} and persist it."""
    json_path = os.path.join(out_dir, 'icd_to_cancer_type_map_insight.json')
    csv_path  = os.path.join(out_dir, 'icd_to_cancer_type_map_insight.csv')

    if os.path.exists(json_path) and not rebuild:
        with open(json_path) as f:
            cached = json.load(f)
        print(f'Loaded cached code map: {len(cached):,} codes  ({json_path})')
        return cached

    print('Building ICD -> cancer type map from the raw code universe ...')
    observed = scan_observed_codes([
        (CONDITION_CSV, 'condition', 'condition_type'),
        (DIAGNOSIS_CSV, 'dx',        'dx_type'),
    ])

    code_map = {}
    for (code_type, code), n in observed.items():
        ctype, tier, rule = resolve_code(code, code_type)
        code_map[f'{code_type}|{code}'] = {
            'code_type':   'ICD-10-CM' if code_type == '10' else 'ICD-9-CM',
            'code':        code,
            'cancer_type': ctype,
            'tier':        tier,
            'matched_rule': rule,
            'n_rows':      n,
        }

    with open(json_path, 'w') as f:
        json.dump(code_map, f, indent=1, sort_keys=True)

    rows = sorted(code_map.values(),
                  key=lambda r: (r['code_type'], -r['n_rows'], r['code']))
    pd.DataFrame(rows)[['code_type', 'code', 'cancer_type', 'tier',
                        'matched_rule', 'n_rows']].to_csv(csv_path, index=False)

    unmapped = [r for r in rows if r['matched_rule'] is None]
    print(f'  {len(code_map):,} distinct neoplasm codes')
    print(f'  {len(code_map) - len(unmapped):,} matched a rule, '
          f'{len(unmapped):,} unmatched')
    if unmapped:
        print('  unmatched (first 20):',
              ', '.join(f"{r['code']}({r['n_rows']})" for r in unmapped[:20]))
    print(f'  -> {json_path}')
    print(f'  -> {csv_path}')
    return code_map


def active_lookup(code_map: dict, include_in_situ: bool,
                  include_secondary: bool, drop_others: bool) -> dict:
    """{f'{code_type}|{code}': cancer_type} for the tiers enabled this run."""
    keep_tiers = {'primary', 'unknown'}
    if include_in_situ:
        keep_tiers.add('in_situ')
    if include_secondary:
        keep_tiers.add('secondary')

    lookup = {}
    for key, rec in code_map.items():
        ctype, tier = rec['cancer_type'], rec['tier']
        if ctype is None or tier not in keep_tiers:
            continue
        if drop_others and ctype == OTHERS:
            continue
        lookup[key] = ctype
    return lookup


# ═══════════════════════════════════════════════════════════════════════════
# Cohort
# ═══════════════════════════════════════════════════════════════════════════
def load_cohorts(source: str, sites) -> dict[str, set]:
    cohorts = {}
    if source == 'labs':
        for site in sites:
            p = os.path.join(PROCESSED_DIR, site, 'ssid_to_date_to_labs_dict.pickle')
            if not os.path.exists(p):
                print(f'  ! {p} missing — falling back to split JSONs for site {site}')
                cohorts[site] = _cohort_from_splits(site)
                continue
            print(f'  loading cohort {site} from {os.path.basename(p)} ...', flush=True)
            with open(p, 'rb') as f:
                cohorts[site] = set(pickle.load(f).keys())
    else:
        for site in sites:
            cohorts[site] = _cohort_from_splits(site)
    for site in sites:
        print(f'  site {site}: {len(cohorts[site]):,} included patients')
    return cohorts


def _cohort_from_splits(site: str) -> set:
    ids = set()
    for sp in ('train', 'val', 'test'):
        p = os.path.join(SPLIT_DIR, sp, f'{site}.json')
        if os.path.exists(p):
            with open(p) as f:
                ids |= set(json.load(f))
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# Reference dates  (delta = event_date - reference_date, in days)
# ═══════════════════════════════════════════════════════════════════════════
def load_reference_dates(cache_path: str, rebuild: bool,
                         needed: set) -> dict[str, pd.Timestamp]:
    """Replicates SPARC_data_preprocessing_insight/shared/utils.build_reference_dates:
    first lab result_date per patient, enrollment enr_start_date as fallback."""
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, 'rb') as f:
            ref = pickle.load(f)
        print(f'Loaded {len(ref):,} reference dates from {cache_path}')
        return ref

    print('Building reference dates (this reads the full lab_result_cm.csv) ...')
    enr = {}
    if os.path.exists(ENROLLMENT_CSV):
        df = pd.read_csv(ENROLLMENT_CSV, dtype={'ssid': str},
                         usecols=['ssid', 'enr_start_date'])
        df['enr_start_date'] = pd.to_datetime(df['enr_start_date'], errors='coerce')
        df = df.dropna(subset=['enr_start_date'])
        enr = df.groupby('ssid')['enr_start_date'].min().to_dict()
        print(f'  {len(enr):,} enrollment dates')

    first_lab: dict[str, pd.Timestamp] = {}
    n = 0
    for chunk in pd.read_csv(LAB_CSV, dtype=str, chunksize=CHUNK_SIZE,
                             usecols=['ssid', 'result_date']):
        n += len(chunk)
        chunk['result_date'] = pd.to_datetime(chunk['result_date'], errors='coerce')
        chunk = chunk.dropna(subset=['result_date'])
        if needed:
            chunk = chunk[chunk['ssid'].isin(needed)]
        if len(chunk):
            mins = chunk.groupby('ssid')['result_date'].min()
            for ssid, d in mins.items():
                cur = first_lab.get(ssid)
                if cur is None or d < cur:
                    first_lab[ssid] = d
        if n % (CHUNK_SIZE * 100) == 0:
            print(f'    {n/1e6:.0f}M lab rows, {len(first_lab):,} patients', flush=True)
    print(f'  {n:,} lab rows scanned, {len(first_lab):,} patients with labs')

    ref = dict(enr)
    ref.update(first_lab)          # first-lab date takes precedence
    with open(cache_path, 'wb') as f:
        pickle.dump(ref, f)
    print(f'  -> {cache_path}')
    return ref


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — per-person match
# ═══════════════════════════════════════════════════════════════════════════
def scan_diagnoses(lookup: dict, cohort_ids: set,
                   sources: list) -> tuple[dict, Counter]:
    """-> {ssid: {date_str: set(cancer_type)}}, plus per-source row stats."""
    per_patient: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    stats = Counter()

    files = []
    if 'condition' in sources:
        files.append((CONDITION_CSV, 'condition', 'condition_type', 'report_date', 'condition'))
    if 'diagnosis' in sources:
        files.append((DIAGNOSIS_CSV, 'dx', 'dx_type', 'admit_date', 'diagnosis'))

    for path, code_col, type_col, date_col, tag in files:
        if not os.path.exists(path):
            print(f'  ! missing, skipped: {path}')
            continue
        print(f'  matching {os.path.basename(path)} ...', flush=True)
        n = 0
        for chunk in pd.read_csv(path, dtype=str, chunksize=CHUNK_SIZE,
                                 usecols=['ssid', code_col, type_col, date_col]):
            n += len(chunk)
            chunk = chunk[chunk['ssid'].isin(cohort_ids)]
            if not len(chunk):
                continue
            keys = (chunk[type_col].fillna('') + '|'
                    + chunk[code_col].fillna('').str.upper()
                                     .str.replace('.', '', regex=False).str.strip())
            types = keys.map(lookup)
            hit = chunk[types.notna()]
            if not len(hit):
                continue
            htypes = types[types.notna()]
            dates = pd.to_datetime(hit[date_col], errors='coerce')
            for ssid, date, ctype in zip(hit['ssid'], dates, htypes):
                if pd.isna(date):
                    stats[f'{tag}_hit_no_date'] += 1
                    continue
                per_patient[ssid][date.strftime('%Y-%m-%d')].add(ctype)
                stats[f'{tag}_hit_with_date'] += 1
            if n % (CHUNK_SIZE * 40) == 0:
                print(f'    {n/1e6:.0f}M rows, {len(per_patient):,} patients hit', flush=True)
        stats[f'{tag}_rows'] = n
        print(f'    done: {n:,} rows', flush=True)

    return per_patient, stats


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════
def summarise(per_site_calendar: dict, out_dir: str, make_plot: bool):
    """Per-site patient counts by cancer type at the EARLIEST diagnosis date."""
    rows = []
    for site, pdict in per_site_calendar.items():
        counts = Counter()
        for ssid, date_to_types in pdict.items():
            if not date_to_types:
                continue
            earliest = min(date_to_types)
            for ct in date_to_types[earliest]:      # double-count ties, as in CAIA
                counts[ct] += 1
        for ct, n in counts.items():
            rows.append({'site': site, 'cancer_type': ct, 'n_patients': n})

    if not rows:
        print('No patients with a mapped cancer diagnosis — nothing to summarise.')
        return None

    df = pd.DataFrame(rows)
    wide = (df.pivot_table(index='cancer_type', columns='site',
                           values='n_patients', fill_value=0, aggfunc='sum')
              .astype(int))
    wide['TOTAL'] = wide.sum(axis=1)
    wide = wide.sort_values('TOTAL', ascending=False)
    csv_path = os.path.join(out_dir, 'insight_cancer_type_summary.csv')
    wide.to_csv(csv_path)
    print(f'\nEarliest-diagnosis cancer type by site  -> {csv_path}')
    print(wide.to_string())

    if make_plot:
        plot_df = wide.drop(columns='TOTAL').head(30).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(plot_df))))
        plot_df.plot(kind='barh', stacked=True, ax=ax, width=0.8,
                     colormap='tab20', edgecolor='none')
        ax.set_xlabel('Patients (cancer type at earliest diagnosis date)')
        ax.set_ylabel('')
        ax.legend(title='Institution', frameon=False,
                  bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            p = os.path.join(out_dir, f'insight_cancer_type_distribution.{ext}')
            fig.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Plot -> {os.path.join(out_dir, "insight_cancer_type_distribution.pdf")}')
    return wide


# ═══════════════════════════════════════════════════════════════════════════
# Self-test — guards the two assumptions the rule table silently depends on:
#   (a) the matcher treats letters as ordinary prefix characters, so the
#       alphabetic ICD-10 subcategories (C49.A GIST, C4A Merkel, C7A NET,
#       D46.Z, C96.A, D47.Z1) are not truncated at the first non-digit;
#   (b) every child override actually outranks its parent, and every bare
#       category header lands somewhere deliberate.
# Run:  python cancer_type_check_insight.py --self-test
# ═══════════════════════════════════════════════════════════════════════════
SELF_TEST_CASES = [
    # (raw code, code_type, expected cancer type, expected tier)

    # ── (a) alphabetic subcategories must survive normalisation ─────────────
    ('C49.A0', '10', 'GIST', 'primary'),          # GIST vs C49 Sarcoma parent
    ('C49.A2', '10', 'GIST', 'primary'),
    ('C49.A9', '10', 'GIST', 'primary'),
    ('C49.9',  '10', 'Sarcoma', 'primary'),       # ...and the parent still wins here
    ('C4A.9',  '10', 'Skin Cancer', 'primary'),   # Merkel: top-level, not a C4 child
    ('C4A.111','10', 'Skin Cancer', 'primary'),
    ('C7A.010','10', 'Neuroendocrine Tumor', 'primary'),
    ('C7A.020','10', 'Neuroendocrine Tumor', 'primary'),   # appendiceal carcinoid
    ('C7B.02', '10', SECONDARY, 'secondary'),     # C7B must NOT collapse into C7A
    ('D46.Z',  '10', 'Myelodysplastic Syndromes', 'primary'),
    ('C96.A',  '10', 'Lymphoma', 'primary'),
    ('D47.Z1', '10', 'Lymphoma', 'primary'),      # PTLD

    # ── (b) child overrides must outrank their parent ───────────────────────
    ('C18.1',  '10', 'Appendix Cancer', 'primary'),
    ('C18.9',  '10', 'Colorectal Cancer', 'primary'),
    ('C48.0',  '10', 'Sarcoma', 'primary'),       # retroperitoneum
    ('C48.1',  '10', 'Ovarian Cancer', 'primary'),# peritoneum proper
    ('C48.2',  '10', 'Ovarian Cancer', 'primary'),
    ('C91.5',  '10', 'Mature T and NK Neoplasms', 'primary'),
    ('C91.00', '10', 'Leukemia', 'primary'),
    ('C38.4',  '10', 'Mesothelioma', 'primary'),
    ('C38.0',  '10', OTHERS, 'primary'),
    ('153.5',  '09', 'Appendix Cancer', 'primary'),
    ('153.9',  '09', 'Colorectal Cancer', 'primary'),
    ('158.0',  '09', 'Sarcoma', 'primary'),
    ('158.9',  '09', 'Ovarian Cancer', 'primary'),
    ('189.0',  '09', 'Renal Cell Carcinoma', 'primary'),
    ('189.1',  '09', 'Bladder Cancer', 'primary'),
    ('189.9',  '09', 'Bladder Cancer', 'primary'),
    ('202.1',  '09', 'Mature T and NK Neoplasms', 'primary'),
    ('202.8',  '09', 'Lymphoma', 'primary'),
    ('238.72', '09', 'Myelodysplastic Syndromes', 'primary'),
    ('209.00', '09', 'Neuroendocrine Tumor', 'primary'),
    ('209.30', '09', 'Neuroendocrine Tumor', 'primary'),
    ('209.72', '09', SECONDARY, 'secondary'),

    # ── bare category headers land somewhere deliberate ─────────────────────
    ('189',    '09', 'Renal Cell Carcinoma', 'primary'),   # plurality of the family
    ('158',    '09', 'Ovarian Cancer', 'primary'),
    ('C48',    '10', 'Ovarian Cancer', 'primary'),
    ('C49',    '10', 'Sarcoma', 'primary'),
    ('230',    '09', OTHERS, 'in_situ'),

    # ── premalignant / benign must be excluded, not bucketed into `others` ──
    ('D47.2',  '10', None, None),                 # MGUS
    ('D47.Z2', '10', None, None),                 # Castleman disease
    ('D3A.00', '10', None, None),                 # benign neuroendocrine
    ('D49.9',  '10', None, None),                 # unspecified behaviour
    ('D32.0',  '10', None, None),                 # benign meningioma
    ('209.60', '09', None, None),                 # benign carcinoid
    ('238.4',  '09', None, None),                 # polycythaemia vera (uncertain)
    ('215.0',  '09', None, None),                 # benign soft tissue

    # ── tiers ───────────────────────────────────────────────────────────────
    ('C80.1',  '10', 'Unknown Primary', 'unknown'),
    ('C76.0',  '10', 'Unknown Primary', 'unknown'),
    ('C78.7',  '10', SECONDARY, 'secondary'),
    ('C77.0',  '10', SECONDARY, 'secondary'),
    ('D05.9',  '10', 'Breast Cancer', 'in_situ'),
    ('D00.00', '10', 'Head and Neck Cancer', 'in_situ'),   # oral, not oesophagogastric
    ('D00.1',  '10', 'Esophagogastric Cancer', 'in_situ'),
    ('D02.0',  '10', 'Head and Neck Cancer', 'in_situ'),   # larynx, not lung
    ('D02.20', '10', 'Non-Small Cell Lung Cancer', 'in_situ'),
    ('D01.5',  '10', 'Hepatobiliary Cancer', 'in_situ'),
    ('D09.0',  '10', 'Bladder Cancer', 'in_situ'),
    ('D09.3',  '10', 'Thyroid Cancer', 'in_situ'),
    ('D07.0',  '10', 'Endometrial Cancer', 'in_situ'),
    ('D07.5',  '10', 'Prostate Cancer', 'in_situ'),
    ('D07.1',  '10', OTHERS, 'in_situ'),
    ('233.0',  '09', 'Breast Cancer', 'in_situ'),
    ('199.1',  '09', 'Unknown Primary', 'unknown'),
    ('197.7',  '09', SECONDARY, 'secondary'),
]


def self_test() -> int:
    """Assert the rule table behaves as documented.  Returns the failure count."""
    failures = []

    for raw, code_type, want_type, want_tier in SELF_TEST_CASES:
        code = normalise_code(raw)
        if not is_candidate_code(code, code_type):
            failures.append(f'{raw} ({code_type}): rejected by is_candidate_code')
            continue
        got_type, got_tier, rule = resolve_code(code, code_type)
        if got_type != want_type or got_tier != want_tier:
            failures.append(
                f'{raw} ({code_type}) -> {got_type!r}/{got_tier!r} via rule {rule!r}; '
                f'expected {want_type!r}/{want_tier!r}')

    # Reachability: every one of the 30 types must be produced by some rule,
    # except the two that are histology rather than site.
    produced = {t for t, _ in ICD10_RULES.values()} | {t for t, _ in ICD09_RULES.values()}
    for ctype in CANCER_TYPES:
        reachable = ctype in produced
        expected = ctype not in UNREACHABLE_TYPES
        if reachable != expected:
            failures.append(
                f'reachability: {ctype!r} is '
                f'{"reachable" if reachable else "unreachable"}, expected the opposite')

    # No rule may emit a label outside the sanctioned vocabulary.
    allowed = set(CANCER_TYPES) | {OTHERS, SECONDARY, None}
    for name, table in (('ICD10_RULES', ICD10_RULES), ('ICD09_RULES', ICD09_RULES)):
        for key, (ctype, tier) in table.items():
            if ctype not in allowed:
                failures.append(f'{name}[{key!r}] emits unknown label {ctype!r}')
            if (ctype is None) != (tier is None):
                failures.append(f'{name}[{key!r}] has mismatched type/tier {ctype!r}/{tier!r}')

    # A key that resolves identically to its own parent is dead weight.
    for name, table in (('ICD10_RULES', ICD10_RULES), ('ICD09_RULES', ICD09_RULES)):
        for key, val in table.items():
            for n in range(len(key) - 1, 2, -1):
                parent = table.get(key[:n])
                if parent is not None:
                    if parent == val:
                        failures.append(
                            f'{name}[{key!r}] is redundant: identical to parent {key[:n]!r}')
                    break

    print(f'self-test: {len(SELF_TEST_CASES)} code cases + reachability + '
          f'vocabulary + redundancy')
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print(f'  - {f}')
    else:
        print('all checks passed')
    return len(failures)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        prog='cancer_type_check_insight',
        description='Map INSIGHT ICD-9/ICD-10 codes to the 30 CAIA cancer types '
                    'and emit per-institution DMP_ID -> DATE -> CANCER_TYPES pickles.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir', default=None,
                    help='folder with the raw INSIGHT CSVs (condition.csv, diagnosis.csv, '
                         'demographics.csv, enrollment.csv, lab_result_cm.csv); required '
                         'unless --self-test')
    ap.add_argument('--processed-dir', default=None,
                    help='folder with the processed per-site INSIGHT pickles and '
                         'insight_stratified/split/; required unless --self-test')
    ap.add_argument('-od', '--output-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--sites', nargs='+', default=SITES, choices=SITES)
    ap.add_argument('--sources', nargs='+', default=['condition', 'diagnosis'],
                    choices=['condition', 'diagnosis'],
                    help='condition.csv holds the problem list (report_date is '
                         'blank for ~69%% of rows); diagnosis.csv holds the '
                         'encounter billing codes and always has admit_date. '
                         'Default: both.')
    ap.add_argument('--cohort-source', default='labs', choices=['labs', 'splits'])
    ap.add_argument('--rebuild-code-map', action='store_true')
    ap.add_argument('--code-map-only', action='store_true',
                    help='Build/inspect the ICD -> cancer type dictionary and stop.')
    ap.add_argument('--include-in-situ', action='store_true',
                    help='Also count carcinoma-in-situ codes (ICD-10 D00-D09, '
                         'ICD-9 230-234) as a cancer diagnosis.')
    ap.add_argument('--include-secondary', action='store_true',
                    help=f"Also emit '{SECONDARY}' for metastasis codes "
                         '(ICD-10 C77-C79/C7B, ICD-9 196-198).')
    ap.add_argument('--drop-others', action='store_true',
                    help=f"Drop the '{OTHERS}' bucket instead of keeping malignancies "
                         'that are none of the 30 types.')
    ap.add_argument('--no-delta', action='store_true',
                    help='Skip the delta pickles (and the 53 GB lab scan they need); '
                         'write only the calendar-date pickles.')
    ap.add_argument('--rebuild-reference-dates', action='store_true')
    ap.add_argument('--build-reference-dates', action='store_true',
                    help='Build/refresh the reference-date cache and stop.')
    ap.add_argument('--self-test', action='store_true',
                    help='Check the ICD rule table against SELF_TEST_CASES '
                         '(normalisation, override precedence, tiers, '
                         'reachability) and exit.')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    if not args.self_test and not (args.raw_dir and args.processed_dir):
        ap.error('--raw-dir and --processed-dir are required (unless --self-test)')
    return args


def main():
    args = parse_args()
    if args.self_test:
        sys.exit(1 if self_test() else 0)
    configure_paths(args.raw_dir, args.processed_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print('=' * 78)
    print('INSIGHT cancer-type check')
    print(f'  raw data   : {RAW_DIR}')
    print(f'  processed  : {PROCESSED_DIR}')
    print(f'  output     : {args.output_dir}')
    print(f'  sites      : {" ".join(args.sites)}')
    print(f'  sources    : {" ".join(args.sources)}')
    print('=' * 78)

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    print('\n[1/3] ICD billing code -> cancer type dictionary')
    code_map = build_code_map(args.output_dir, args.rebuild_code_map)
    lookup = active_lookup(code_map, args.include_in_situ,
                           args.include_secondary, args.drop_others)
    by_type = Counter(lookup.values())
    print(f'  active codes this run: {len(lookup):,} '
          f'over {len(by_type)} cancer types')
    missing = [t for t in CANCER_TYPES if t not in by_type]
    if missing:
        print(f'  types with no code in this dataset: {", ".join(missing)}')
    print(f'  NOTE: {" and ".join(UNREACHABLE_TYPES)} are not derivable from ICD '
          'site codes\n        (histology, not site) — see UNREACHABLE_TYPES in this file.')
    if args.code_map_only:
        return

    # ── Cohort ──────────────────────────────────────────────────────────────
    print('\n[2/3] Cohort (patients present in the processed data)')
    cohorts = load_cohorts(args.cohort_source, args.sites)
    all_ids = set().union(*cohorts.values()) if cohorts else set()
    print(f'  total: {len(all_ids):,} patients')

    ref_dates = None
    if not args.no_delta or args.build_reference_dates:
        cache = os.path.join(args.output_dir, 'ssid_to_reference_date_insight.pickle')
        ref_dates = load_reference_dates(cache, args.rebuild_reference_dates, all_ids)
        if args.build_reference_dates:
            return

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    print('\n[3/3] Per-person match')
    per_patient, stats = scan_diagnoses(lookup, all_ids, args.sources)
    print('  row statistics:', dict(stats))
    print(f'  {len(per_patient):,} of {len(all_ids):,} cohort patients '
          f'({100 * len(per_patient) / max(len(all_ids), 1):.1f}%) '
          'have >=1 mapped cancer code')

    per_site_calendar = {}
    for site in args.sites:
        ids = cohorts[site]
        cal = {ssid: {d: sorted(ts) for d, ts in per_patient[ssid].items()}
               for ssid in ids if ssid in per_patient}
        per_site_calendar[site] = cal

        p = os.path.join(
            args.output_dir,
            f'ssid_to_CALENDARDATE_to_CANCER_TYPES_dict_insight_{site}.pickle')
        with open(p, 'wb') as f:
            pickle.dump(cal, f)

        line = (f'  site {site}: {len(cal):,}/{len(ids):,} patients '
                f'({100 * len(cal) / max(len(ids), 1):.1f}%)  -> {os.path.basename(p)}')

        if ref_dates is not None:
            delta = {}
            no_ref = 0
            for ssid, date_to_types in cal.items():
                r = ref_dates.get(ssid)
                if r is None:
                    no_ref += 1
                    continue
                d = {}
                for date_str, ts in date_to_types.items():
                    off = int((pd.Timestamp(date_str) - r).days)
                    d.setdefault(off, set()).update(ts)
                delta[ssid] = {k: sorted(v) for k, v in d.items()}
            dp = os.path.join(
                args.output_dir,
                f'DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_insight_{site}.pickle')
            with open(dp, 'wb') as f:
                pickle.dump(delta, f)
            line += f'\n            delta: {len(delta):,} patients'
            if no_ref:
                line += f' ({no_ref} dropped: no reference date)'
            line += f'  -> {os.path.basename(dp)}'
        print(line)

    summarise(per_site_calendar, args.output_dir, not args.no_plot)
    print('\nDone.')


if __name__ == '__main__':
    main()
