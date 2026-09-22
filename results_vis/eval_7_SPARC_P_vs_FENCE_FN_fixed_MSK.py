"""SPARC-P vs FENCE, febrile neutropenia, first cycle only.

Task is locked to febrile_neutropenia and evaluation restricted to each
patient's first chemotherapy cycle. Patients on weekly cisplatin are excluded,
matching FENCE's own exclusion of the weekly-cisplatin + concomitant-RT group,
where FN is not cycle-dependent.

Two comparators: the FENCE nomogram, mapping our data onto its 8 predictors to
produce both the integer score and the cycle-1 FN probability (the probability
is used as the continuous score); and FENCE_retrained, logistic regression on
the same features fit on train+val.

SPARC-P first-cycle n / n_pos / metrics are taken from the existing CSV to
avoid re-running inference; the first-cycle labels and FENCE scores are still
computed here. Both models are scored on the same first-cycle patients with the
same next-infusion FN labels, compared by AUROC and AUPRC.

Output CSV: $SPARC_RESULTS_ROOT/eval_clinical_benchmarks
"""

import os
import json
import pickle
import sys, inspect
import argparse

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset

from SPARC_inf_risk_modular import SPARC

from MSK_featurename_list import (
    build_module_index_config,
    ALL_INCLUDED_TESTS,
    CREATININE_INDEX, BILIRUBIN_INDEX, PRIMARY_CTYPES,
)
# Named indices into the raw lab_def_dict value vector (ordered by ALL_INCLUDED_TESTS)
ALBUMIN_INDEX = ALL_INCLUDED_TESTS.index('Albumin')

# Age bins are 5-year buckets (age_at_first_seq_binned_5); age ~ AGE_BIN_YEARS * bin
AGE_BIN_YEARS = 5.0

from sklearn.metrics import roc_auc_score, average_precision_score
from delong_test import delong_roc_test


# ============================  FENCE NOMOGRAM  ==============================
# Integer point coefficients from Aagaard et al., JNCI Cancer Spectrum 2018
# (Table 2, "Coefficient to use in FENCE score calculation"). Exact regression
# coefficients (used for the probability formula) are kept separately below.

FENCE_INTERCEPT = -7.561          # beta_0 (exact), for probability calculation
FENCE_CYCLE_LENGTH_DAYS = 21      # t in the probability formula

# --- Integer points (for the FENCE score / risk-group stratification) -------
FENCE_POINTS = {
    'sex':   {'male': 0, 'female': 8},
    'age':   {'<65': 0, '65-74': 9, '>=75': 4},
    'stage': {'adjuvant': 0, 'neoadjuvant': -11, 'advanced': 7},
    'albumin': {'low': 10, 'normal': 0, 'high': -7, 'missing': -3},
    'bilirubin': {'<5': 1, '5-25': 0, '>25': 17, 'missing': -29},
    'egfr': {'<=60': 13, '60-90': 4, '>=90': 0, 'missing': 19},
    'infection': {'no': 0, 'yes': 10},
    'n_drugs': {1: 0, 2: 10, 3: 47, 4: 56},
    # chemotherapy class points (each present class adds its points)
    'taxane': 31, 'platinum': -27, 'nonplatinum_alkylating': -2,
    'topoisomerase': -5, 'antimetabolite': -16, 'vinca': -4, 'other_chemo': -50,
    # cancer type points (FENCE's 17 categories)
    'cancer': {
        'gastric': 0, 'brain': -22, 'head_neck': 52, 'oesophageal': 46,
        'breast': 32, 'mesothelioma': 28, 'nsclc': 47, 'sclc': 75,
        'colorectal': -23, 'ovarian': 28, 'cervical_endometrial': -33,
        'bladder': 29, 'prostate': 6, 'testicular': 85, 'neuroendocrine': 55,
        'lymphoma_dlbcl': 7, 'other': 10,
    },
}

# --- Exact regression coefficients (for the cycle-1 probability formula) -----
# Only the levels that carry nonzero exact coefficients are listed; reference
# levels are 0. Used to compute Prob(FN in cycle one).
FENCE_EXACT = {
    'sex':   {'male': 0.0, 'female': 0.327},
    'age':   {'<65': 0.0, '65-74': 0.351, '>=75': 0.154},
    'stage': {'adjuvant': 0.0, 'neoadjuvant': -0.426, 'advanced': 0.267},
    'albumin': {'low': 0.413, 'normal': 0.0, 'high': -0.284, 'missing': -0.104},
    'bilirubin': {'<5': 0.035, '5-25': 0.0, '>25': 0.687, 'missing': -1.174},
    'egfr': {'<=60': 0.502, '60-90': 0.170, '>=90': 0.0, 'missing': 0.747},
    'infection': {'no': 0.0, 'yes': 0.409},
    'n_drugs': {1: 0.0, 2: 0.376, 3: 1.886, 4: 2.217},
    'taxane': 1.241, 'platinum': -1.073, 'nonplatinum_alkylating': -0.065,
    'topoisomerase': -0.189, 'antimetabolite': -0.627, 'vinca': -0.163,
    'other_chemo': -1.982,
    'cancer': {
        'gastric': 0.0, 'brain': -0.891, 'head_neck': 2.073, 'oesophageal': 1.845,
        'breast': 1.276, 'mesothelioma': 1.096, 'nsclc': 1.870, 'sclc': 3.014,
        'colorectal': -0.897, 'ovarian': 1.116, 'cervical_endometrial': -1.299,
        'bladder': 1.156, 'prostate': 0.244, 'testicular': 3.407,
        'neuroendocrine': 2.189, 'lymphoma_dlbcl': 0.281, 'other': 0.398,
    },
}

# FENCE risk groups by integer score
def fence_risk_group(score):
    if score <= 16:
        return 'low'
    elif score <= 35:
        return 'intermediate'
    elif score <= 52:
        return 'high'
    else:
        return 'very_high'


# Cancer-type mapping: MSK PRIMARY_CTYPES vocabulary -> FENCE category.
CTYPE_TO_FENCE = {
    'Non-Small Cell Lung Cancer': 'nsclc',
    'Small Cell Lung Cancer': 'sclc',
    'Colorectal Cancer': 'colorectal',
    'Appendiceal Cancer': 'colorectal',
    'Breast Cancer': 'breast',
    'Prostate Cancer': 'prostate',
    'Ovarian Cancer': 'ovarian',
    'Endometrial Cancer': 'cervical_endometrial',
    'Cervical Cancer': 'cervical_endometrial',
    'Uterine Sarcoma': 'cervical_endometrial',
    'Bladder Cancer': 'bladder',
    'Esophagogastric Cancer': 'gastric',
    'Glioma': 'brain',
    'Head and Neck Cancer': 'head_neck',
    'Salivary Gland Cancer': 'head_neck',
    'Mesothelioma': 'mesothelioma',
    'Germ Cell Tumor': 'testicular',
    'Mature B-Cell Neoplasms': 'lymphoma_dlbcl',
    # explicitly 'other'
    'Pancreatic Cancer': 'other',
    'Soft Tissue Sarcoma': 'other',
    'Bone Cancer': 'other',
    'Melanoma': 'other',
    'Skin Cancer, Non-Melanoma': 'other',
    'Hepatobiliary Cancer': 'other',
    'Renal Cell Carcinoma': 'other',
    'Thyroid Cancer': 'other',
    'Gastrointestinal Stromal Tumor': 'other',
    'Leukemia': 'other',
    'Mature T and NK Neoplasms': 'other',
    'Myeloproliferative Neoplasms': 'other',
    'Myelodysplastic Syndromes': 'other',
    'Myelodysplastic Workup': 'other',
    'Cancer of Unknown Primary': 'other',
    'others': 'other',
}

_UNMAPPED_CTYPES = {}


def map_ctype_to_fence(ctype_str):
    if ctype_str is None:
        return 'other'
    s = str(ctype_str).strip()
    if s in CTYPE_TO_FENCE:
        return CTYPE_TO_FENCE[s]
    _UNMAPPED_CTYPES[s] = _UNMAPPED_CTYPES.get(s, 0) + 1
    return 'other'


# Weekly-cisplatin detection
def is_weekly_cisplatin(drug_schedule, max_gap_days=10, min_runs=3):
    if not isinstance(drug_schedule, dict):
        return False
    cis_days = sorted(
        day for day, entries in drug_schedule.items()
        if any(str(e.get('name', '')).strip().upper() == 'CISPLATIN' for e in entries)
    )
    if len(cis_days) < min_runs:
        return False
    gaps = np.diff(cis_days)
    if len(gaps) == 0:
        return False
    close = gaps[gaps <= max_gap_days]
    if len(close) >= (min_runs - 1) and float(np.median(gaps)) <= max_gap_days:
        return True
    return False


# CKD-EPI eGFR (2009)
def compute_egfr_ckd_epi(creat_mgdl, age, is_female):
    if creat_mgdl is None or age is None or creat_mgdl <= 0:
        return None
    if is_female:
        k, alpha, sexf = 0.7, -0.329, 1.018
    else:
        k, alpha, sexf = 0.9, -0.411, 1.0
    ratio = creat_mgdl / k
    egfr = 141 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.209) * (0.993 ** age) * sexf
    return egfr


# Band helpers
def age_band(age):
    if age is None:
        return '<65'
    if age < 65:
        return '<65'
    if age < 75:
        return '65-74'
    return '>=75'


def egfr_band(egfr):
    if egfr is None:
        return 'missing'
    if egfr <= 60:
        return '<=60'
    if egfr < 90:
        return '60-90'
    return '>=90'


def albumin_band(albumin_gL, is_female, age):
    if albumin_gL is None:
        return 'missing'
    low, high = 35.0, 50.0
    if albumin_gL < low:
        return 'low'
    if albumin_gL > high:
        return 'high'
    return 'normal'


def bilirubin_band(bili_umolL):
    if bili_umolL is None:
        return '5-25'
    if bili_umolL < 5:
        return '<5'
    if bili_umolL <= 25:
        return '5-25'
    return '>25'


# Compute FENCE integer score and cycle-1 probability
def fence_score_and_prob(feats):
    P = FENCE_POINTS
    E = FENCE_EXACT

    n_drugs = min(max(int(feats.get('n_drugs', 1)), 1), 4)

    score = 0
    lin = FENCE_INTERCEPT

    score += P['sex'][feats['sex']];                   lin += E['sex'][feats['sex']]
    score += P['age'][feats['age_band']];              lin += E['age'][feats['age_band']]
    score += P['cancer'][feats['cancer']];             lin += E['cancer'][feats['cancer']]
    score += P['stage'][feats['stage']];               lin += E['stage'][feats['stage']]
    score += P['albumin'][feats['albumin_band']];      lin += E['albumin'][feats['albumin_band']]
    score += P['bilirubin'][feats['bilirubin_band']];  lin += E['bilirubin'][feats['bilirubin_band']]
    score += P['egfr'][feats['egfr_band']];            lin += E['egfr'][feats['egfr_band']]
    score += P['infection'][feats['infection']];       lin += E['infection'][feats['infection']]
    score += P['n_drugs'][n_drugs];                    lin += E['n_drugs'][n_drugs]

    for cls in feats.get('classes', set()):
        if cls in P:
            score += P[cls]
            lin += E[cls]

    # Prob(FN in cycle one) = 1 - exp( t * (-exp(lin)) )
    prob = 1.0 - np.exp(FENCE_CYCLE_LENGTH_DAYS * (-np.exp(lin)))
    prob = float(np.clip(prob, 0.0, 1.0))
    return score, prob


# Build feature matrix for logistic regression
def build_fence_feature_matrix(features_list):
    """Build one-hot encoded feature matrix for logistic regression."""
    X = []
    for feats in features_list:
        row = []
        # Sex: female vs male (ref)
        row.append(1.0 if feats['sex'] == 'female' else 0.0)
        # Age: <65 (ref), 65-74, >=75
        row.append(1.0 if feats['age_band'] == '65-74' else 0.0)
        row.append(1.0 if feats['age_band'] == '>=75' else 0.0)
        # Stage: adjuvant (ref), neoadjuvant, advanced
        row.append(1.0 if feats['stage'] == 'neoadjuvant' else 0.0)
        row.append(1.0 if feats['stage'] == 'advanced' else 0.0)
        # Albumin: normal (ref), low, high, missing
        row.append(1.0 if feats['albumin_band'] == 'low' else 0.0)
        row.append(1.0 if feats['albumin_band'] == 'high' else 0.0)
        row.append(1.0 if feats['albumin_band'] == 'missing' else 0.0)
        # Bilirubin: 5-25 (ref), <5, >25, missing
        row.append(1.0 if feats['bilirubin_band'] == '<5' else 0.0)
        row.append(1.0 if feats['bilirubin_band'] == '>25' else 0.0)
        row.append(1.0 if feats['bilirubin_band'] == 'missing' else 0.0)
        # eGFR: >=90 (ref), <=60, 60-90, missing
        row.append(1.0 if feats['egfr_band'] == '<=60' else 0.0)
        row.append(1.0 if feats['egfr_band'] == '60-90' else 0.0)
        row.append(1.0 if feats['egfr_band'] == 'missing' else 0.0)
        # Infection: no (ref), yes
        row.append(1.0 if feats['infection'] == 'yes' else 0.0)
        # n_drugs: 1 (ref), 2, 3, 4
        n_drugs = min(max(int(feats.get('n_drugs', 1)), 1), 4)
        row.append(1.0 if n_drugs == 2 else 0.0)
        row.append(1.0 if n_drugs == 3 else 0.0)
        row.append(1.0 if n_drugs == 4 else 0.0)
        # Chemo classes
        classes = feats.get('classes', set())
        for cls in ['taxane', 'platinum', 'nonplatinum_alkylating',
                   'topoisomerase', 'antimetabolite', 'vinca', 'other_chemo']:
            row.append(1.0 if cls in classes else 0.0)
        # Cancer type: gastric (ref), then 16 others
        cancer = feats['cancer']
        for ctype in ['brain', 'head_neck', 'oesophageal', 'breast', 'mesothelioma',
                     'nsclc', 'sclc', 'colorectal', 'ovarian', 'cervical_endometrial',
                     'bladder', 'prostate', 'testicular', 'neuroendocrine',
                     'lymphoma_dlbcl', 'other']:
            row.append(1.0 if cancer == ctype else 0.0)
        X.append(row)

    feature_names = [
        'sex_female', 'age_65_74', 'age_gte75',
        'stage_neoadjuvant', 'stage_advanced',
        'albumin_low', 'albumin_high', 'albumin_missing',
        'bilirubin_lt5', 'bilirubin_gt25', 'bilirubin_missing',
        'egfr_lte60', 'egfr_60_90', 'egfr_missing',
        'infection_yes', 'n_drugs_2', 'n_drugs_3', 'n_drugs_4',
        'taxane', 'platinum', 'nonplatinum_alkylating',
        'topoisomerase', 'antimetabolite', 'vinca', 'other_chemo',
        'cancer_brain', 'cancer_head_neck', 'cancer_oesophageal', 'cancer_breast',
        'cancer_mesothelioma', 'cancer_nsclc', 'cancer_sclc', 'cancer_colorectal',
        'cancer_ovarian', 'cancer_cervical_endometrial', 'cancer_bladder',
        'cancer_prostate', 'cancer_testicular', 'cancer_neuroendocrine',
        'cancer_lymphoma_dlbcl', 'cancer_other'
    ]
    return np.array(X), feature_names


# Chemo-class lookup
def load_med_classes(path):
    with open(path, 'r') as f:
        meds = json.load(f)
    out = {}
    for m in meds:
        out[str(m['name']).strip().upper()] = m
    return out


def classes_for_drugs(drug_names, med_class_map):
    classes = set()
    chemo_names = set()
    for name in drug_names:
        m = med_class_map.get(str(name).strip().upper())
        if m is None:
            continue
        if not m.get('is_chemo', False):
            continue
        chemo_names.add(str(name).strip().upper())
        if m.get('is_platinum'):
            classes.add('platinum')
        if m.get('is_taxane'):
            classes.add('taxane')
        if m.get('is_topoisomerase_inhibitor'):
            classes.add('topoisomerase')
        if m.get('is_alkalating_agent'):
            classes.add('nonplatinum_alkylating')
        if m.get('is_vinorelbine'):
            classes.add('vinca')
    return len(chemo_names), classes


# ===============================  ARG PARSE  ================================
# Roots for the argparse defaults below; each is overridable by env var or by
# the corresponding CLI flag. Results go to results_vis_060726_batch_091426 so
# runs here do not overwrite results_vis_060726; reference CSVs are still read
# from the original tree.
SPARC_DATA_ROOT = os.environ.get(
    'SPARC_DATA_ROOT', '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data')
SPARC_MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_ROOT', '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models_060726')
SPARC_RESULTS_ROOT = os.environ.get(
    'SPARC_RESULTS_ROOT', f'{SPARC_MODEL_ROOT}/results_vis_060726_batch_091426')
SPARC_REF_ROOT = os.environ.get(
    'SPARC_REF_ROOT', f'{SPARC_MODEL_ROOT}/results_vis_060726')

parser = argparse.ArgumentParser(prog='eval_SPARC_P_vs_FENCE_first_cycle_FN_fixed')
parser.add_argument("-r", "--rseed", type=int, default=1024)
parser.add_argument("-d", "--days", type=int, default=30)
parser.add_argument("-m", "--modules", type=str, default='parsimonious')
parser.add_argument("-ds", "--dataset", type=str, default='MSK')
parser.add_argument("-me", "--med_of_choice", type=str,
                    default='DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                            'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                            'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL')
parser.add_argument("-cp", "--caia_portion", type=str, default="prior")
parser.add_argument("-z", "--zeroshot", type=str, default='True')
parser.add_argument("-mm", "--missing_mask", type=str, default="True")
parser.add_argument("-en", "--ensemble", type=str, default='True')
parser.add_argument("-dsp", "--data_source_path", type=str, default=f'{SPARC_DATA_ROOT}/')
parser.add_argument("-mp", "--model_path", type=str, default=f'{SPARC_MODEL_ROOT}')
parser.add_argument("-rp", "--results_path", type=str,
                    default=f'{SPARC_RESULTS_ROOT}/eval_clinical_benchmarks')
parser.add_argument("--existing_csv", type=str,
                    default=f'{SPARC_REF_ROOT}/'
                            'eval_across_models_valthresh_disabled/model_performance_dict_across_models_parsimonious_febrile_neutropenia_MSK_CAIA_ensemble.csv')
parser.add_argument("--weekly_cisplatin_max_gap", type=int, default=10)
parser.add_argument("--exclude_weekly_cisplatin", type=str, default='True')
args = parser.parse_args()

rseed = args.rseed
days = args.days
modules = args.modules
dataset = args.dataset
caia_portion = args.caia_portion.lower()
zeroshot = args.zeroshot.lower() == 'true'
missing_mask = args.missing_mask.lower() == 'true'
ensemble = args.ensemble.lower() == 'true'
data_source_path = args.data_source_path
model_path = args.model_path
results_path = args.results_path
existing_csv_path = args.existing_csv
exclude_weekly_cisplatin = args.exclude_weekly_cisplatin.lower() == 'true'

# LOCKED settings
option = 'febrile_neutropenia'
add_num_infusion = True

med_of_choice = [m.strip() for m in args.med_of_choice.split(',')]
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        result_meds.append([m.strip() for m in med_str.split('+')])
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print("med_of_choice:", med_of_choice)

# Architecture params
batch_size = 512
model_include_gcsf = False
use_module_tokenizer = True
use_continuous_time = True
post_mlp_dropout = 0.2
use_sigmoid = True
schedule_option = 'default'
learning_rate = 1e-4
include_gcsf = False
component = 'parsimonious'


# Load SPARC-P models and generate predictions for DeLong test
print("\n" + "="*80)
print("GENERATING SPARC-P PREDICTIONS FOR DELONG TEST")
print("="*80)

# Model configuration
model_type_suffix = 'hybrid'  # Febrile_neutropenia finetuned checkpoints use the 'hybrid' suffix
sparc_p_folder = (f'{model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/'
                  f'{modules}_finetuned_models_{model_type_suffix}_{option}_20tp_topchemos_att_modular')
print("SPARC-P folder:", sparc_p_folder)
all_folds = list(range(10))

# Load data first to get modular_featuredim_dict (use base option name)
print(f"Loading dataset from: {data_source_path}")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option=option, modules=modules,
    valfold=0, mode='test_all' if 'timeshift' in dataset else 'test', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)
print("Data loaded.")

# Build module index config
module_index_cfg = build_module_index_config(modular_featuredim_dict, advanced_model='default')
n_static_features = module_index_cfg['n_static']

# Set option suffix and model settings AFTER data loading, BEFORE data module init
model_set_option = 'side_effects_risk'  # Model expects this format, not task name
option = f'{option}_risk'  # febrile_neutropenia_risk
anc_mode = True if 'CAIA' in dataset else False

# Initialize data module (with _risk suffix in task)
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)
print("Data module initialized (test_shuffle=False for deterministic ordering)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sparc_fold_store = {}

def first_cycle_mask(num_infusions_tensor):
    """Boolean mask selecting the first cycle. Cycle indexing may be 0- or 1-based;
    use the minimum present infusion index as 'first cycle'."""
    uniq = torch.unique(num_infusions_tensor)
    first_idx = uniq.min()
    return (num_infusions_tensor == first_idx), first_idx.item()

# Generate predictions from each fold
for val_fold in all_folds:
    print(f"\n=== SPARC-P fold {val_fold} ===")
    pl.seed_everything(rseed)
    model_folder = f'{sparc_p_folder}/fold{val_fold}'
    ckpt = [el for el in os.listdir(model_folder) if 'epoch' in el][0]
    ckpt = os.path.join(model_folder, ckpt)

    model = SPARC.load_from_checkpoint(
        ckpt, modular_featuredim_dict=modular_featuredim_dict, option=model_set_option,
        advanced_model="default", max_time_len=20, med_as_feature=False, meds_l=med_of_choice,
        loss_option='default', schedule=schedule_option, side_effects_num_infusion=add_num_infusion,
        include_gcsf=model_include_gcsf, use_gcsf=model_include_gcsf, use_sigmoid=use_sigmoid,
        temporal_aggregation='attention', missing_mask=missing_mask,
        module_index_cfg=module_index_cfg,
        use_module_tokenizer=use_module_tokenizer, use_continuous_time=use_continuous_time,
        n_static_features=n_static_features, post_mlp_dropout=post_mlp_dropout, lr=learning_rate,
    ).to(device)
    model.eval()

    all_preds, all_events, all_ttes, all_infus, all_dmp = [], [], [], [], []
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            xs_combined, y, mask, _, num_infusion, dmp_ids = model.data_prep(
                batch, time_calibration=model.time_calibration)
            if num_infusion is None:
                num_infusion = torch.zeros(xs_combined.shape[0], dtype=torch.float32)
            xs_combined = xs_combined.to(device)
            mask = mask.to(device)
            num_infusion = num_infusion.to(device)
            y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                  num_infusion=num_infusion, gcsf_groundtruth=None)

            fail = torch.as_tensor(y[0], device=device)
            tte = torch.as_tensor(y[1], device=device)
            if fail.dim() == 0:
                fail = fail.unsqueeze(0); tte = tte.unsqueeze(0)

            all_preds.append(y_pred_logits.detach().cpu())
            all_events.append(fail.detach().cpu())
            all_ttes.append(tte.detach().cpu())
            all_infus.append(num_infusion.detach().cpu())
            if isinstance(dmp_ids, (list, tuple)):
                all_dmp.extend(list(dmp_ids))
            elif torch.is_tensor(dmp_ids):
                all_dmp.extend(dmp_ids.cpu().tolist())
            else:
                all_dmp.extend(list(dmp_ids))

    all_preds = torch.cat(all_preds).numpy().ravel()
    all_events = torch.cat(all_events).numpy().ravel()
    all_ttes = torch.cat(all_ttes).numpy().ravel()
    all_infus = torch.cat(all_infus).numpy().ravel()
    all_dmp = np.array(all_dmp)

    # ---- FIRST CYCLE filter ----
    infus_t = torch.tensor(all_infus)
    fc_mask, first_idx = first_cycle_mask(infus_t)
    fc_mask = fc_mask.numpy()
    print(f"First-cycle index = {first_idx}; first-cycle rows: {fc_mask.sum()} / {len(fc_mask)}")

    fc_preds = all_preds[fc_mask]
    fc_events = all_events[fc_mask]
    fc_ttes = all_ttes[fc_mask]
    fc_dmp = all_dmp[fc_mask]

    # Next-infusion FN label on first cycle
    fc_label = ((fc_events == 1) & (fc_ttes == 1)).astype(int)

    sparc_fold_store[val_fold] = {
        'preds': fc_preds, 'label': fc_label, 'dmp': fc_dmp,
    }
    print(f"Fold {val_fold}: {len(fc_preds)} first-cycle rows, {int(fc_label.sum())} next-infusion FN positives")

# Build ensemble - align by DMP_ID across folds
print("\nBuilding SPARC-P first-cycle ensemble (aligned by patient) ...")
# Reference patient set/labels from fold 0
ref = sparc_fold_store[0]
ref_label_by_dmp = {d: l for d, l in zip(ref['dmp'], ref['label'])}
patients = list(ref_label_by_dmp.keys())

pred_accum = {d: [] for d in patients}
for f in all_folds:
    s = sparc_fold_store[f]
    pmap = {d: p for d, p in zip(s['dmp'], s['preds'])}
    for d in patients:
        if d in pmap:
            pred_accum[d].append(pmap[d])

sparc_first_cycle_dmp = [d for d in patients if len(pred_accum[d]) > 0]
sparc_first_cycle_preds = np.array([np.mean(pred_accum[d]) for d in sparc_first_cycle_dmp])
sparc_first_cycle_labels = np.array([ref_label_by_dmp[d] for d in sparc_first_cycle_dmp])
print(f"Ensemble first-cycle patients: {len(sparc_first_cycle_dmp)}, positives: {int(sparc_first_cycle_labels.sum())}")

# Compute SPARC-P metrics for CSV
auroc_existing = roc_auc_score(sparc_first_cycle_labels, sparc_first_cycle_preds)
auprc_existing = average_precision_score(sparc_first_cycle_labels, sparc_first_cycle_preds)
def _bootstrap_ci_temp(y, p, metric='auroc', n_boot=1000, seed=1024):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, pb) if metric == 'auroc' else average_precision_score(yb, pb))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

auroc_ci = _bootstrap_ci_temp(sparc_first_cycle_labels, sparc_first_cycle_preds, 'auroc')
auprc_ci = _bootstrap_ci_temp(sparc_first_cycle_labels, sparc_first_cycle_preds, 'auprc')
auroc_ci_lower_existing, auroc_ci_upper_existing = auroc_ci
auprc_ci_lower_existing, auprc_ci_upper_existing = auprc_ci
print(f"SPARC-P first-cycle AUROC: {auroc_existing:.4f} [{auroc_ci_lower_existing:.4f}, {auroc_ci_upper_existing:.4f}]")
print(f"SPARC-P first-cycle AUPRC: {auprc_existing:.4f} [{auprc_ci_lower_existing:.4f}, {auprc_ci_upper_existing:.4f}]")


# Load auxiliary data for FENCE
print("\nLoading FENCE auxiliary data ...")

# Drugs dict (for weekly cisplatin exclusion)
drugs_dict_path = f'{data_source_path}/MSK_CAIA/DMP_ID_to_DATE_to_DRUG_MED_DICT_delta_till2021_031226.pickle'
if os.path.exists(drugs_dict_path):
    with open(drugs_dict_path, 'rb') as f:
        drugs_dict = pickle.load(f)
    print(f"  Drugs dict loaded: {len(drugs_dict)} patients")
else:
    drugs_dict = {}
    print("  [WARN] Drugs dict not found -> weekly-cisplatin exclusion disabled")

# Med classes (for FENCE chemo class scoring)
med_classes_path = f'{data_source_path}/MSK_CAIA/classified_medications.json'
if os.path.exists(med_classes_path):
    med_class_map = load_med_classes(med_classes_path)
    print(f"  Med class map loaded: {len(med_class_map)} medications")
else:
    med_class_map = {}
    print("  [WARN] Med class map not found -> FENCE class features unavailable")

# Cancer type map
CTYPE_DICT = data_dict_lab_dict.get('DMP_ID_to_PRIMARY_CTYPE_dict', {})
if not CTYPE_DICT:
    print("  [WARN] PRIMARY_CTYPE_dict empty -> all ctypes default to 'other'")


# Build FENCE feature extractor
LAB_DEF = data_dict_lab_dict.get('lab_def_dict', {})


def _safe_int(x):
    try:
        return int(x)
    except:
        return None


def _baseline_lab_value(dmp_id, test_index, cutoff_date=None, lookback=90):
    patient = LAB_DEF.get(dmp_id)
    if not patient:
        return None
    dates = sorted(patient.keys())
    if cutoff_date is not None:
        window = [d for d in dates if (cutoff_date - lookback) <= d <= cutoff_date]
        candidate_dates = window if window else [d for d in dates if d <= cutoff_date]
        if not candidate_dates:
            candidate_dates = dates
        candidate_dates = sorted(candidate_dates, reverse=True)
    else:
        candidate_dates = dates
    for d in candidate_dates:
        vals = patient[d]
        if test_index < len(vals):
            v = vals[test_index]
            if v is not None:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        return fv
                except:
                    continue
    return None


def get_age_years(dmp_id):
    a = data_dict_lab_dict.get('age', {}).get(dmp_id)
    if a is None:
        return None
    try:
        b = float(a[0]) if isinstance(a, (list, tuple, np.ndarray)) else float(a)
    except:
        return None
    return b * AGE_BIN_YEARS


def get_sex(dmp_id):
    g = data_dict_lab_dict.get('gender', {}).get(dmp_id)
    if g is None:
        return 'male'
    try:
        female_flag = float(g[0]) if isinstance(g, (list, tuple, np.ndarray)) else float(g)
    except:
        return 'male'
    return 'female' if int(round(female_flag)) == 1 else 'male'


def build_fence_features(dmp_id, cutoff_date=None):
    age = get_age_years(dmp_id)
    sex = get_sex(dmp_id)
    is_female = (sex == 'female')

    creat = _baseline_lab_value(dmp_id, CREATININE_INDEX, cutoff_date)
    egfr = compute_egfr_ckd_epi(creat, age, is_female)
    alb = _baseline_lab_value(dmp_id, ALBUMIN_INDEX, cutoff_date)
    bili = _baseline_lab_value(dmp_id, BILIRUBIN_INDEX, cutoff_date)

    cancer = map_ctype_to_fence(CTYPE_DICT.get(dmp_id))

    # Stage: default to 'advanced' (most conservative)
    stage = 'advanced'

    # Infection: default to 'no'
    infection = 'no'

    # n_drugs and classes: from first-cycle drugs
    drug_sched = drugs_dict.get(dmp_id, {})
    first_cycle_drugs = set()
    if isinstance(drug_sched, dict):
        for day, entries in drug_sched.items():
            if cutoff_date is None or _safe_int(day) <= 0:
                for e in entries:
                    if 'name' in e:
                        first_cycle_drugs.add(str(e['name']).strip().upper())
    n_drugs, classes = classes_for_drugs(first_cycle_drugs, med_class_map)

    feats = {
        'sex': sex,
        'age_band': age_band(age),
        'cancer': cancer,
        'stage': stage,
        'albumin_band': albumin_band(alb, is_female, age),
        'bilirubin_band': bilirubin_band(bili),
        'egfr_band': egfr_band(egfr),
        'infection': infection,
        'n_drugs': n_drugs if n_drugs > 0 else 1,
        'classes': classes,
    }
    return feats


# Build split_dict / test set
if 'CAIA' in dataset:
    print(f"Using split_dict from load_dataset. Test size: {len(split_dict['test'])}")
elif dataset == 'stanfordEHRSHOT':
    with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_all_included_lab_values_list_dict.pickle', 'rb') as h:
        D = pickle.load(h)
    tids = [d for d in D.keys() if d in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': tids}
elif dataset == 'MIMICIV_hosp':
    with open(f'{data_source_path}/{dataset}/split_dict_1v1v1_112425.pickle', 'rb') as h:
        fr = pickle.load(h)
    tids = [d for d in fr['test'] if d in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': tids}

# Weekly-cisplatin exclusion
if exclude_weekly_cisplatin and drugs_dict:
    before = len(split_dict['test'])
    excluded = [d for d in split_dict['test']
                if is_weekly_cisplatin(drugs_dict.get(d), max_gap_days=args.weekly_cisplatin_max_gap)]
    split_dict['test'] = [d for d in split_dict['test'] if d not in set(excluded)]
    print(f"\n[COHORT] Weekly-cisplatin exclusion: removed {before - len(split_dict['test'])} "
          f"of {before} patients -> {len(split_dict['test'])} remain")
else:
    print("\n[COHORT] Weekly-cisplatin exclusion skipped")


# Precompute FENCE features + score for every test patient
print("\nComputing FENCE scores for test cohort ...")
fence_by_patient = {}
for dmp_id in split_dict['test']:
    feats = build_fence_features(dmp_id)
    score, prob = fence_score_and_prob(feats)
    fence_by_patient[dmp_id] = {'score': score, 'prob': prob,
                                'group': fence_risk_group(score), 'feats': feats}

if _UNMAPPED_CTYPES:
    print(f"\n[FENCE][WARN] {len(_UNMAPPED_CTYPES)} distinct cancer-type strings did not map "
          f"to a FENCE category and defaulted to 'other'. Top offenders:")
    for s, c in sorted(_UNMAPPED_CTYPES.items(), key=lambda x: -x[1])[:15]:
        print(f"    {c:6d}  {s}")

print(f"FENCE scored {len(fence_by_patient)} patients. "
      f"Group counts: " +
      ", ".join(f"{g}={sum(1 for v in fence_by_patient.values() if v['group']==g)}"
                for g in ['low', 'intermediate', 'high', 'very_high']))


# Extract first-cycle labels (lightweight, no model inference)
print("\n=== Extracting first-cycle labels from test set (no SPARC-P inference) ===")
pl.seed_everything(rseed)
anc_mode = True if 'CAIA' in dataset else False

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)
print("Data module initiated")

# Extract labels by iterating test dataloader (lightweight)
# First pass: find global first cycle index
print("  First pass: finding global first cycle index...")
all_num_inf = []
for batch in data_module.test_dataloader():
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except:
        continue
    num_inf_np = num_inf.cpu().numpy() if isinstance(num_inf, torch.Tensor) else np.array(num_inf)
    all_num_inf.extend(num_inf_np.tolist())

first_cycle_idx = int(np.min(all_num_inf)) if all_num_inf else 1
print(f"  Global first cycle index: {first_cycle_idx}")

# Second pass: extract first-cycle labels
print("  Second pass: extracting first-cycle labels...")
first_cycle_labels_by_dmp = {}
for batch in data_module.test_dataloader():
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except:
        continue

    fail = y[0].cpu().numpy() if isinstance(y[0], torch.Tensor) else np.array(y[0])
    tte = y[1].cpu().numpy() if isinstance(y[1], torch.Tensor) else np.array(y[1])
    num_inf = num_inf.cpu().numpy() if isinstance(num_inf, torch.Tensor) else np.array(num_inf)
    dmp_ids = dmp_ids.cpu().tolist() if isinstance(dmp_ids, torch.Tensor) else list(dmp_ids)

    for i in range(len(dmp_ids)):
        dmp_id = dmp_ids[i]
        if num_inf[i] == first_cycle_idx and dmp_id not in first_cycle_labels_by_dmp:
            label = 1 if (fail[i] == 1 and tte[i] == 1) else 0
            first_cycle_labels_by_dmp[dmp_id] = label

print(f"Extracted first-cycle labels for {len(first_cycle_labels_by_dmp)} patients")
print(f"  Positives: {sum(first_cycle_labels_by_dmp.values())}")


# Train FENCE_retrained model (logistic regression on train+val)
print("\n=== Training FENCE_retrained logistic regression model (train+val combined) ===")

# Load train+val data
_, _, split_dict_trainval, _, _ = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option='febrile_neutropenia',
    modules=modules, valfold=0, mode='train', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)

# Get both train and val patient IDs
train_val_ids = list(split_dict_trainval['train']) + list(split_dict_trainval['val'])
print(f"Combined train+val: {len(split_dict_trainval['train'])} train + {len(split_dict_trainval['val'])} val = {len(train_val_ids)} total patients")

# Exclude weekly cisplatin if needed
if exclude_weekly_cisplatin and drugs_dict:
    before = len(train_val_ids)
    excluded_train = [d for d in train_val_ids
                     if is_weekly_cisplatin(drugs_dict.get(d), max_gap_days=args.weekly_cisplatin_max_gap)]
    train_val_ids = [d for d in train_val_ids if d not in set(excluded_train)]
    print(f"  Weekly-cisplatin exclusion: removed {before - len(train_val_ids)}")

# Create combined split_dict
split_dict_combined = {
    'train': train_val_ids,
    'val': [],
    'test': split_dict_trainval['test']
}

# Create data module for combined train+val
data_module_combined = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict_combined, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)

# Extract first-cycle features and labels from train+val
train_features_list = []
train_labels = []
train_seen_patients = set()

print("Extracting first-cycle features from train+val patients...")
# First pass: find global first cycle index for train data
all_train_num_inf = []
for batch in data_module_combined.train_dataloader():
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except:
        continue
    num_inf_np = num_inf.cpu().numpy() if isinstance(num_inf, torch.Tensor) else np.array(num_inf)
    all_train_num_inf.extend(num_inf_np.tolist())

train_first_cycle_idx = int(np.min(all_train_num_inf)) if all_train_num_inf else 1
print(f"  Train first cycle index: {train_first_cycle_idx}")

# Second pass: extract features
for batch in data_module_combined.train_dataloader():
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except:
        continue

    fail = y[0].cpu().numpy() if isinstance(y[0], torch.Tensor) else np.array(y[0])
    tte = y[1].cpu().numpy() if isinstance(y[1], torch.Tensor) else np.array(y[1])
    num_inf = num_inf.cpu().numpy() if isinstance(num_inf, torch.Tensor) else np.array(num_inf)
    dmp_ids = dmp_ids.cpu().tolist() if isinstance(dmp_ids, torch.Tensor) else list(dmp_ids)

    for i in range(len(dmp_ids)):
        dmp_id = dmp_ids[i]
        if num_inf[i] == train_first_cycle_idx and dmp_id not in train_seen_patients:
            train_seen_patients.add(dmp_id)
            feats = build_fence_features(dmp_id)
            train_features_list.append(feats)
            label = 1 if (fail[i] == 1 and tte[i] == 1) else 0
            train_labels.append(label)

# Build feature matrix
X_train, feature_names = build_fence_feature_matrix(train_features_list)
y_train = np.array(train_labels)

print(f"Training samples: {len(y_train)} unique patients, {int(y_train.sum())} positives ({y_train.sum()/len(y_train):.2%})")

# Fit logistic regression
print("Fitting logistic regression model...")
lr_model = LogisticRegression(random_state=rseed, max_iter=1000, class_weight='balanced', solver='lbfgs')
lr_model.fit(X_train, y_train)

print(f"\nTrained model intercept: {lr_model.intercept_[0]:.4f}")
print("Top 10 coefficients:")
coef_sorted = sorted(zip(feature_names, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for fname, coef in coef_sorted[:10]:
    print(f"  {fname:30s}: {coef:7.4f}")

# Compute predictions for test set
print("\nComputing FENCE_retrained predictions on test set...")
test_patients_with_fence = [d for d in first_cycle_labels_by_dmp.keys() if d in fence_by_patient]
test_features_list = [fence_by_patient[d]['feats'] for d in test_patients_with_fence]
X_test, _ = build_fence_feature_matrix(test_features_list)
fence_retrained_probs_full = lr_model.predict_proba(X_test)[:, 1]

# Map back to dictionary
fence_retrained_probs_dict = dict(zip(test_patients_with_fence, fence_retrained_probs_full))
print(f"FENCE_retrained predictions computed for {len(fence_retrained_probs_dict)} test patients")


# Comparison: SPARC-P (from CSV), FENCE, FENCE_retrained
print("\n=== Building comparison data ===")

# Build patient-level dictionary for SPARC-P first-cycle predictions
sparc_pred_by_dmp = {}
for i, dmp in enumerate(sparc_first_cycle_dmp):
    if dmp not in sparc_pred_by_dmp:  # Take first occurrence only
        sparc_pred_by_dmp[dmp] = sparc_first_cycle_preds[i]

# Align all models to patients with FENCE scores AND first-cycle labels AND SPARC-P predictions
ens_dmp = [d for d in first_cycle_labels_by_dmp.keys()
           if d in fence_by_patient and d in sparc_pred_by_dmp]
ens_label = np.array([first_cycle_labels_by_dmp[d] for d in ens_dmp])
fence_prob = np.array([fence_by_patient[d]['prob'] for d in ens_dmp])
fence_retrained_probs = np.array([fence_retrained_probs_dict.get(d, np.nan) for d in ens_dmp])
sparc_prob = np.array([sparc_pred_by_dmp[d] for d in ens_dmp])

# Drop patients with no FENCE_retrained score
valid = ~np.isnan(fence_retrained_probs)
ens_dmp_v = [ens_dmp[i] for i in range(len(ens_dmp)) if valid[i]]
ens_label_v = ens_label[valid]
fence_prob_v = fence_prob[valid]
fence_retrained_probs_v = fence_retrained_probs[valid]
sparc_prob_v = sparc_prob[valid]

print(f"Final comparison set: {len(ens_dmp_v)} patients, {int(ens_label_v.sum())} positives ({ens_label_v.mean():.2%})")
print(f"  (aligned SPARC-P, FENCE, FENCE_retrained, and labels)")


def _bootstrap_ci(y, p, metric='auroc', n_boot=1000, seed=1024):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, pb) if metric == 'auroc' else average_precision_score(yb, pb))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


rows = []

# FENCE original
if len(np.unique(ens_label_v)) == 2:
    # Compute metrics on aligned cohort
    sp_auroc = roc_auc_score(ens_label_v, sparc_prob_v)
    sp_auprc = average_precision_score(ens_label_v, sparc_prob_v)
    sp_auroc_ci = _bootstrap_ci(ens_label_v, sparc_prob_v, 'auroc')
    sp_auprc_ci = _bootstrap_ci(ens_label_v, sparc_prob_v, 'auprc')

    fe_auroc = roc_auc_score(ens_label_v, fence_prob_v)
    fe_auprc = average_precision_score(ens_label_v, fence_prob_v)
    fe_auroc_ci = _bootstrap_ci(ens_label_v, fence_prob_v, 'auroc')
    fe_auprc_ci = _bootstrap_ci(ens_label_v, fence_prob_v, 'auprc')

    # DeLong test for SPARC-P vs FENCE original
    try:
        delong_orig = delong_roc_test(ens_label_v, sparc_prob_v, fence_prob_v)
        print(f"DeLong test (SPARC-P vs FENCE): z={delong_orig['z_score']:.4f}, p={delong_orig['pvalue']:.4e}")
    except Exception as e:
        print(f"[WARN] DeLong test failed: {e}")
        delong_orig = {'pvalue': np.nan, 'z_score': np.nan}

    print(f"\n================= FIRST-CYCLE FN COMPARISON =================")
    print(f"n = {len(ens_label_v)}, positives = {int(ens_label_v.sum())} ({ens_label_v.mean():.2%})")
    print(f"SPARC-P (ensemble): AUROC={sp_auroc:.4f} [{sp_auroc_ci[0]:.4f},{sp_auroc_ci[1]:.4f}]  "
          f"AUPRC={sp_auprc:.4f} [{sp_auprc_ci[0]:.4f},{sp_auprc_ci[1]:.4f}]")
    print(f"FENCE (original)  : AUROC={fe_auroc:.4f} [{fe_auroc_ci[0]:.4f},{fe_auroc_ci[1]:.4f}]  "
          f"AUPRC={fe_auprc:.4f} [{fe_auprc_ci[0]:.4f},{fe_auprc_ci[1]:.4f}]")

    # SPARC-P row
    rows.append({
        'model': 'SPARC-P_ensemble_firstcycle',
        'n': len(ens_label_v),
        'n_pos': int(ens_label_v.sum()),
        'auroc': sp_auroc,
        'auroc_ci_lower': sp_auroc_ci[0],
        'auroc_ci_upper': sp_auroc_ci[1],
        'auprc': sp_auprc,
        'auprc_ci_lower': sp_auprc_ci[0],
        'auprc_ci_upper': sp_auprc_ci[1]
    })

    rows.append({
        'model': 'FENCE_firstcycle',
        'n': len(ens_label_v),
        'n_pos': int(ens_label_v.sum()),
        'auroc': fe_auroc,
        'auroc_ci_lower': fe_auroc_ci[0],
        'auroc_ci_upper': fe_auroc_ci[1],
        'auprc': fe_auprc,
        'auprc_ci_lower': fe_auprc_ci[0],
        'auprc_ci_upper': fe_auprc_ci[1]
    })

    # FENCE retrained
    fer_auroc = roc_auc_score(ens_label_v, fence_retrained_probs_v)
    fer_auprc = average_precision_score(ens_label_v, fence_retrained_probs_v)
    fer_auroc_ci = _bootstrap_ci(ens_label_v, fence_retrained_probs_v, 'auroc')
    fer_auprc_ci = _bootstrap_ci(ens_label_v, fence_retrained_probs_v, 'auprc')

    # DeLong test for SPARC-P vs FENCE retrained
    try:
        delong_ret = delong_roc_test(ens_label_v, sparc_prob_v, fence_retrained_probs_v)
        print(f"DeLong test (SPARC-P vs FENCE_retrained): z={delong_ret['z_score']:.4f}, p={delong_ret['pvalue']:.4e}")
    except Exception as e:
        print(f"[WARN] DeLong test failed: {e}")
        delong_ret = {'pvalue': np.nan, 'z_score': np.nan}

    print(f"FENCE (retrained) : AUROC={fer_auroc:.4f} [{fer_auroc_ci[0]:.4f},{fer_auroc_ci[1]:.4f}]  "
          f"AUPRC={fer_auprc:.4f} [{fer_auprc_ci[0]:.4f},{fer_auprc_ci[1]:.4f}]")

    rows.append({
        'model': 'FENCE_retrained_firstcycle',
        'n': len(ens_label_v),
        'n_pos': int(ens_label_v.sum()),
        'auroc': fer_auroc,
        'auroc_ci_lower': fer_auroc_ci[0],
        'auroc_ci_upper': fer_auroc_ci[1],
        'auprc': fer_auprc,
        'auprc_ci_lower': fer_auprc_ci[0],
        'auprc_ci_upper': fer_auprc_ci[1]
    })

    # Add significance test result rows
    rows.append({
        'model': 'SPARC-P_vs_FENCE_significance_test',
        'n': len(ens_label_v),
        'n_pos': int(ens_label_v.sum()),
        'auroc': delong_orig['pvalue'],
        'auroc_ci_lower': delong_orig['z_score'],
        'auroc_ci_upper': sp_auroc - fe_auroc,
        'auprc': None,
        'auprc_ci_lower': None,
        'auprc_ci_upper': None
    })
    rows.append({
        'model': 'SPARC-P_vs_FENCE_retrained_significance_test',
        'n': len(ens_label_v),
        'n_pos': int(ens_label_v.sum()),
        'auroc': delong_ret['pvalue'],
        'auroc_ci_lower': delong_ret['z_score'],
        'auroc_ci_upper': sp_auroc - fer_auroc,
        'auprc': None,
        'auprc_ci_lower': None,
        'auprc_ci_upper': None
    })
else:
    print("\n[WARN] First-cycle labels are single-class; cannot compute AUROC/AUPRC.")


# Save
_wc = '_exclWeeklyCis' if exclude_weekly_cisplatin else ''
save_name = (f'{results_path}/SPARC-P_vs_FENCE_firstcycle_FN_FIXED_'
             f'{modules}_{dataset}_{caia_portion}{_wc}.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(save_name, index=False)
print(f"\nSaved comparison to: {save_name}")
print(df.to_string(index=False))

# Audit file
audit_rows = []
for d in ens_dmp_v:
    info = fence_by_patient.get(d, {})
    feats = info.get('feats', {})
    audit_rows.append({
        'dmp_id': d,
        'fence_score': info.get('score'),
        'fence_prob': info.get('prob'),
        'fence_retrained_prob': fence_retrained_probs_dict.get(d),
        'fence_group': info.get('group'),
        **{f'feat_{k}': (','.join(v) if isinstance(v, set) else v)
           for k, v in feats.items()}
    })
audit_name = save_name.replace('.csv', '_FENCE_patient_audit.csv')
pd.DataFrame(audit_rows).to_csv(audit_name, index=False)
print(f"Saved FENCE per-patient audit to: {audit_name}")
