"""SPARC-P vs the Motwani cisplatin-AKI nomogram on the first cisplatin course.

Motwani et al., J Clin Oncol 2018;36(7):682-688. Points: age 61-70 = 1, 71-90 = 2.5;
first-course dose 101-150 mg = 1, >150 mg = 3; hypertension = 2; albumin <=3.5 g/dL = 2.
prob: odds = 0.04/0.96 * 1.49**score (--prob_anchor=asco: logit fit to 0/3.5/8.5 -> 0.03/0.12/0.51).
Outcome is our next-infusion AKI label (event==1 & tte==1), not Motwani's 14-day creatinine rule.
"""

import os
import pickle
import sys, inspect
import re
import argparse

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# put repo root, data/ and models/ on sys.path
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
    CREATININE_INDEX,
)
# index into lab_def_dict value vectors
ALBUMIN_INDEX = ALL_INCLUDED_TESTS.index('Albumin')

# age is stored in 5-year bins
AGE_BIN_YEARS = 5.0

from delong_test import delong_roc_test

# torchmetrics-backed AUROC/AUPRC to match eval_1 exactly (sklearn differs ~1e-3 on ties)
from torchmetrics import AUROC as _TMAUROC, AveragePrecision as _TMAP
_tm_auroc = _TMAUROC(task='binary')
_tm_auprc = _TMAP(task='binary')


def roc_auc_score(y_true, y_score):
    yt = torch.as_tensor(np.asarray(y_true)).int()
    ys = torch.as_tensor(np.asarray(y_score)).float()
    _tm_auroc.reset()
    return _tm_auroc(ys, yt).item()


def average_precision_score(y_true, y_score):
    yt = torch.as_tensor(np.asarray(y_true)).int()
    ys = torch.as_tensor(np.asarray(y_score)).float()
    _tm_auprc.reset()
    return _tm_auprc(ys, yt).item()


# Motwani CP-AKI score (range 0-9.5)
MOTWANI_POINTS = {
    'age':     {'<=60': 0.0, '61-70': 1.0, '71-90': 2.5},
    'dose':    {'<=100': 0.0, '101-150': 1.0, '>150': 3.0},   # first-course cisplatin mg
    'htn':     {'no': 0.0, 'yes': 2.0},
    'albumin': {'>3.5': 0.0, '2.0-3.5': 2.0, '<2.0': 2.0},    # any hypoalbuminaemia band = 2
}

# score -> prob (JCO / Saito 2025): incidence 0.04 at score 0, OR 1.49 per point
MOTWANI_BASELINE_INCIDENCE = 0.04
MOTWANI_OR_PER_POINT = 1.49

# ASCO anchors 0/3.5/8.5 -> 0.03/0.12/0.51; logit-linear fit through the end points
_ASCO_S = np.array([0.0, 3.5, 8.5])
_ASCO_P = np.array([0.03, 0.12, 0.51])
_ASCO_LOGIT = np.log(_ASCO_P / (1 - _ASCO_P))
_ASCO_SLOPE = (_ASCO_LOGIT[-1] - _ASCO_LOGIT[0]) / (_ASCO_S[-1] - _ASCO_S[0])
_ASCO_INTERCEPT = _ASCO_LOGIT[0]


def motwani_prob_from_score(score, anchor='jco'):
    if anchor == 'asco':
        logit = _ASCO_INTERCEPT + _ASCO_SLOPE * score
        return float(1.0 / (1.0 + np.exp(-logit)))
    odds0 = MOTWANI_BASELINE_INCIDENCE / (1.0 - MOTWANI_BASELINE_INCIDENCE)
    odds = odds0 * (MOTWANI_OR_PER_POINT ** score)
    return float(odds / (1.0 + odds))


# risk groups by total score
def motwani_risk_group(score):
    if score <= 1.0:
        return 'low'
    elif score <= 3.5:
        return 'intermediate'
    elif score <= 6.0:
        return 'high'
    else:
        return 'very_high'


def age_band(age):
    if age is None:
        return '<=60'          # reference (0 pts) when unavailable
    if age <= 60:
        return '<=60'
    if age <= 70:
        return '61-70'
    return '71-90'


def dose_band(dose_mg):
    if dose_mg is None:
        return '<=100'         # reference (0 pts) when unavailable
    if dose_mg <= 100:
        return '<=100'
    if dose_mg <= 150:
        return '101-150'
    return '>150'


def albumin_band(albumin_gdl):
    """Albumin in g/dL; >3.5 is the reference band."""
    if albumin_gdl is None:
        return '>3.5'          # reference (0 pts) when unavailable
    if albumin_gdl > 3.5:
        return '>3.5'
    if albumin_gdl >= 2.0:
        return '2.0-3.5'
    return '<2.0'


def motwani_score_and_prob(feats, anchor='jco', include_dose=True):
    """include_dose=False drops the dose term (3-variable model) when dose is unavailable."""
    P = MOTWANI_POINTS
    score = (P['age'][feats['age_band']]
             + P['htn'][feats['htn']]
             + P['albumin'][feats['albumin_band']])
    if include_dose:
        score += P['dose'][feats['dose_band']]
    prob = motwani_prob_from_score(score, anchor=anchor)
    return score, prob


def build_motwani_feature_matrix(features_list, include_dose=True):
    """One-hot Motwani bands (reference levels dropped) for the retrained LR."""
    X = []
    for feats in features_list:
        row = []

        row.append(1.0 if feats['age_band'] == '61-70' else 0.0)
        row.append(1.0 if feats['age_band'] == '71-90' else 0.0)

        if include_dose:
            row.append(1.0 if feats['dose_band'] == '101-150' else 0.0)
            row.append(1.0 if feats['dose_band'] == '>150' else 0.0)

        row.append(1.0 if feats['htn'] == 'yes' else 0.0)

        row.append(1.0 if feats['albumin_band'] == '2.0-3.5' else 0.0)
        row.append(1.0 if feats['albumin_band'] == '<2.0' else 0.0)

        X.append(row)

    if include_dose:
        feature_names = [
            'age_61_70', 'age_71_90',
            'dose_101_150', 'dose_gt150',
            'htn_yes',
            'albumin_2_3p5', 'albumin_lt2'
        ]
    else:
        feature_names = [
            'age_61_70', 'age_71_90',
            'htn_yes',
            'albumin_2_3p5', 'albumin_lt2'
        ]

    return np.array(X), feature_names


# weekly-cisplatin detection (from the FENCE script; off by default here)
def is_weekly_cisplatin(drug_schedule, max_gap_days=10, min_runs=3):
    if not isinstance(drug_schedule, dict):
        return False
    cis_days = sorted(
        day for day, entries in drug_schedule.items()
        if any(_entry_name(e) == 'CISPLATIN' for e in entries)
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


# args
# roots can be overridden by env var or CLI flag
SPARC_DATA_ROOT = os.environ.get(
    'SPARC_DATA_ROOT', '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data')
SPARC_MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_ROOT', '/data1/morrisq/fuc/SPARC_CAIA_092426')
SPARC_RESULTS_ROOT = os.environ.get(
    'SPARC_RESULTS_ROOT', '/data1/morrisq/fuc/SPARC_CAIA_092426/results_vis_eval1_bfTrue')
SPARC_REF_ROOT = os.environ.get(
    'SPARC_REF_ROOT', '/nonexistent/no_bf_false_reference_row')

parser = argparse.ArgumentParser(prog='eval_SPARC_P_vs_MOTWANI_first_cycle_AKI')
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
parser.add_argument("-mp", "--model_path", type=str,
                    default=os.environ.get('SPARC_EVAL7_MODEL_ROOT', '/data1/morrisq/fuc/SPARC_MSK_sparcp_bfTrue_092926'))
parser.add_argument("-rp", "--results_path", type=str,
                    default=f'{SPARC_RESULTS_ROOT}/eval7_comparators_MSK_bfTrue')
parser.add_argument("--weekly_cisplatin_max_gap", type=int, default=10)
parser.add_argument("--exclude_weekly_cisplatin", type=str, default='False',
                    help="Motwani did NOT exclude weekly cisplatin; default False.")
parser.add_argument("--require_cisplatin_cycle1", type=str, default='True',
                    help="True: keep only patients whose first chemo cycle contains cisplatin "
                         "(matches the Motwani 'first course of cisplatin' cohort).")
parser.add_argument("--prob_anchor", type=str, default='jco', choices=['jco', 'asco'],
                    help="score->probability map. 'jco': incidence 0.04 @ score0, OR 1.49/pt "
                         "(default). 'asco': logit-linear fit to 0/3.5/8.5 -> 0.03/0.12/0.51.")
parser.add_argument("--hypertension_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/df_icd_og_060525.csv',
                    help="Hypertension source. .csv: the de-identified ICD billing "
                         "table (DMP_ID, DTE_SINCE_SEQ, CC_CLSF_CD), filtered to "
                         "ICD-9 401-405 / ICD-10 I10-I16. .pkl/.pickle: a pre-built "
                         "{DMP_ID: {delta_day: [codes]}} dict; key -1 = pre-existing.")
parser.add_argument("--diagnostics", type=str, default='True',
                    help="True: also print the all-cycles SPARC-P diagnostic block. "
                         "False: baseline only -- just the first-infusion (cycle 0) "
                         "SPARC-P vs Motwani comparison, matching Motwani's design.")
parser.add_argument("--existing_csv", type=str,
                    default=f'{SPARC_REF_ROOT}/'
                            'eval_across_models_valthresh_disabled/model_performance_dict_across_models_parsimonious_AKI_MSK_CAIA_ensemble.csv',
                    help="Path to existing CSV with all-cycle SPARC-P metrics (for reference)")
parser.add_argument("-bf", "--batch_first", type=str, default='True',
                    help="True | False -- build SPARC with batch_first=True so the encoder "
                         "attends over time. MUST match how the checkpoints were trained; "
                         "a mismatch loads silently.")
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
exclude_weekly_cisplatin = args.exclude_weekly_cisplatin.lower() == 'true'
require_cisplatin_cycle1 = args.require_cisplatin_cycle1.lower() == 'true'
prob_anchor = args.prob_anchor
hypertension_path = args.hypertension_path
run_diagnostics = args.diagnostics.lower() == 'true'
existing_csv_path = args.existing_csv

option = 'AKI'
add_num_infusion = True                  # needed for cycle indexing

med_of_choice = [m.strip() for m in args.med_of_choice.split(',')]
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        result_meds.append([m.strip() for m in med_str.split('+')])
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print("med_of_choice:", med_of_choice)

# architecture params (match 060726 training)
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

print("Loading dataset ...")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option=option, modules=modules,
    valfold=0, mode='test_all' if 'timeshift' in dataset else 'test', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)
print("Data loaded.")

module_index_cfg = build_module_index_config(modular_featuredim_dict, advanced_model='default')
n_static_features = module_index_cfg['n_static']

print("\nLoading Motwani auxiliary data ...")

def _try_load_pickle(path, label):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  [{label}] loaded ({len(obj)} entries) from {path}")
        return obj
    print(f"  [{label}] NOT FOUND at {path} -> will default")
    return {}

def _load_hypertension(path):
    """{DMP_ID: {delta_day: [codes]}} of hypertension diagnoses, from a pickle or the ICD csv.

    Matches ICD-9 401-405 and ICD-10 I10-I16 (the MSK table is mostly ICD-9).
    """
    if not path or not os.path.exists(path):
        print(f"  [hypertension] NOT FOUND at {path} -> hypertension term disabled")
        return {}

    if str(path).lower().endswith(('.pkl', '.pickle')):
        return _try_load_pickle(path, 'hypertension')

    import pandas as _pd
    from collections import defaultdict as _dd

    ICD9 = ('401', '402', '403', '404', '405')
    ICD10 = ('I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16')

    def _is_htn(code):
        c = str(code).strip().upper().replace(' ', '')
        return c.startswith(ICD9) or c.startswith(ICD10)

    out = _dd(lambda: _dd(list))
    n_rows = 0
    # large file, read in chunks
    for chunk in _pd.read_csv(path, usecols=['DMP_ID', 'DTE_SINCE_SEQ', 'CC_CLSF_CD'],
                              chunksize=500_000, low_memory=False):
        hit = chunk[chunk['CC_CLSF_CD'].map(_is_htn)]
        for _pid, _day, _code in zip(hit['DMP_ID'], hit['DTE_SINCE_SEQ'], hit['CC_CLSF_CD']):
            if _pd.isna(_day):
                continue
            out[_pid][int(_day)].append(str(_code).strip())
            n_rows += 1
    htn = {pid: dict(days) for pid, days in out.items()}
    print(f"  [hypertension] {len(htn)} patients / {n_rows} coded rows from {path}")
    return htn


# MSK non-oral meds: {DMP_ID: {date: [entry, ...]}}, entry is a list:
# [0] name, [1] class, [2] regimen, [3] category, [4] dose, [5] unit,
# [6] duration_days, [7] day_offset, [8] frequency
DRUG_NAME_IDX = 0
DRUG_DOSE_IDX = 4
DRUG_UNIT_IDX = 5
drugs_dict = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})
print(f"  [drugs] MSK non-oral meds (w/ dosage): {len(drugs_dict)} patients "
      f"from data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT']")

htn_dict = _load_hypertension(hypertension_path)

# print one cisplatin entry to check dose/unit positions
def _first_cisplatin_entry():
    for pid, sched in drugs_dict.items():
        if not isinstance(sched, dict):
            continue
        for day, entries in sched.items():
            for e in entries:
                try:
                    if str(e[DRUG_NAME_IDX]).strip().upper() == 'CISPLATIN':
                        return e
                except (IndexError, TypeError):
                    continue
    return None


_samp = _first_cisplatin_entry()
MOTWANI_INCLUDE_DOSE = _samp is not None
if _samp is not None:
    print("  [dose] sample CISPLATIN entry (verify dose@[4] / unit@[5]):")
    print(f"    {_samp}")
    try:
        print(f"    -> parsed dose={_samp[DRUG_DOSE_IDX]} unit={_samp[DRUG_UNIT_IDX]}")
    except (IndexError, TypeError):
        print("    [dose][WARN] entry shorter than expected; dose index may differ!")
        MOTWANI_INCLUDE_DOSE = False
else:
    print("  [dose][WARN] no CISPLATIN entries found -> dose unavailable, "
          "Motwani reverts to 3-variable.")

# no usable dose -> 3-variable Motwani (age + HTN + albumin)
MOTWANI_N_VARS = 4 if MOTWANI_INCLUDE_DOSE else 3
if not MOTWANI_INCLUDE_DOSE:
    print("  [dose] Cisplatin dose unavailable in this data source -> running Motwani "
          "as a 3-VARIABLE model (age + hypertension + albumin); dose term dropped.")


def _coerce_mg(val):
    """Dose as float; accepts numbers or strings like '100 mg'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r'[-+]?\d*\.?\d+', str(val))
    return float(m.group()) if m else None


# per-patient accessors
LAB_DEF = data_dict_lab_dict.get('lab_def_dict', {})
if not LAB_DEF:
    print("[WARN] data_dict_lab_dict['lab_def_dict'] is empty; albumin will default to "
          "reference band for all patients.")


def get_age_years(dmp_id):
    a = data_dict_lab_dict.get('age', {}).get(dmp_id)
    if a is None:
        return None
    try:
        b = float(a[0]) if isinstance(a, (list, tuple, np.ndarray)) else float(a)
    except Exception:
        return None
    return b * AGE_BIN_YEARS


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
                except Exception:
                    continue
    return None


def get_albumin_gdl(dmp_id, cutoff_date=None):
    # MSK albumin is already g/dL
    return _baseline_lab_value(dmp_id, ALBUMIN_INDEX, cutoff_date)


def _htn_lookup(dmp_id):
    """Lookup tolerant of int/str DMP_ID keys."""
    if dmp_id in htn_dict:
        return htn_dict[dmp_id]
    for alt in (str(dmp_id), _safe_int(dmp_id)):
        if alt is not None and alt in htn_dict:
            return htn_dict[alt]
    return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def get_hypertension(dmp_id, cutoff_date=None):
    """'yes' if any HTN diagnosis day <= cutoff_date; key -1 (pre-existing) always counts."""
    rec = _htn_lookup(dmp_id)
    if not rec:
        return 'no'
    if isinstance(rec, dict):
        htn_days = list(rec.keys())
    else:
        htn_days = list(rec)
    if cutoff_date is None:
        return 'yes' if len(htn_days) > 0 else 'no'
    for d in htn_days:
        try:
            di = int(d)
        except Exception:
            continue
        if di == -1 or di <= cutoff_date:   # -1 sentinel = present before sequence start
            return 'yes'
    return 'no'


def _entry_name(e):
    try:
        return str(e[DRUG_NAME_IDX]).strip().upper()
    except (IndexError, TypeError):
        return ''


def get_cycle1_day_and_drugs(dmp_id):
    """First cisplatin day and that day's drug entries (cycle 1 = first cisplatin course)."""
    sched = drugs_dict.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return None, []
    cis_days = [day for day, entries in sched.items()
                if any(_entry_name(e) == 'CISPLATIN' for e in entries)]
    if not cis_days:
        return None, []
    c1 = min(cis_days)
    return c1, sched[c1]


def cycle1_has_cisplatin(cycle1_entries):
    return any(_entry_name(e) == 'CISPLATIN' for e in cycle1_entries)


def get_cycle1_cisplatin_dose_mg(cycle1_entries):
    """Total cisplatin mg on the cycle-1 day, or None. Warns on non-mg units."""
    if not MOTWANI_INCLUDE_DOSE:
        return None
    total = 0.0
    found = False
    for e in cycle1_entries:
        if _entry_name(e) == 'CISPLATIN':
            try:
                mg = _coerce_mg(e[DRUG_DOSE_IDX])
                unit = str(e[DRUG_UNIT_IDX]).strip().upper() if len(e) > DRUG_UNIT_IDX else 'MG'
            except (IndexError, TypeError):
                continue
            if mg is not None:
                if unit not in ('MG', 'MILLIGRAM', 'MILLIGRAMS', ''):
                    print(f"    [dose][WARN] cisplatin unit is {unit!r}, not mg -- "
                          f"bands assume mg; verify conversion.")
                total += mg
                found = True
    return total if found else None


def build_motwani_features(dmp_id):
    age = get_age_years(dmp_id)
    c1_day, c1_entries = get_cycle1_day_and_drugs(dmp_id)
    has_cis = cycle1_has_cisplatin(c1_entries) if c1_entries else False
    dose_mg = get_cycle1_cisplatin_dose_mg(c1_entries) if c1_entries else None
    albumin = get_albumin_gdl(dmp_id, cutoff_date=c1_day)
    htn = get_hypertension(dmp_id, cutoff_date=c1_day)

    feats = {
        'age_band': age_band(age),
        'dose_band': dose_band(dose_mg),
        'htn': htn,
        'albumin_band': albumin_band(albumin),
        # raw values for the audit csv
        '_age': age, '_dose_mg': dose_mg, '_albumin_gdl': albumin,
        '_cycle1_day': c1_day, '_has_cisplatin_cycle1': has_cis,
    }
    return feats


# SPARC-P model folder
all_model_path = model_path
base_option = 'AKI'
model_type_suffix = 'risk'   # AKI checkpoints use 'risk', not 'hybrid'

if dataset == 'MSK':
    sparc_p_folder = (f'{all_model_path}/{modules}_MSK_060726/SPARC_finetuned_models/'
                      f'{modules}_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
elif dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift', 'stanfordEHRSHOT'] or (dataset == 'MIMICIV_hosp' and zeroshot):
    sparc_p_folder = (f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/'
                      f'{modules}_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
else:
    if args.batch_first == 'True':
        raise SystemExit("No batch_first=True MIMICIV_hosp-finetuned SPARC exists; the only MIMICIV-finetuned models are the published bf=False ones. Use -z True (zero-shot from the bf=True CAIA models) or run the published repo.")
    mimiciv_sparc_path = f'{SPARC_MODEL_ROOT}/parsimonious_MIMICIV_hosp_060726'
    sparc_p_folder = (f'{mimiciv_sparc_path}/SPARC_finetuned_models/'
                      f'parsimonious_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
print("SPARC-P folder:", sparc_p_folder)
if dataset == 'MSK' and not os.path.isdir(sparc_p_folder):
    print(f"[SPARC-P][WARN] MSK model folder not found at the guessed path above. "
          f"Set the correct MSK AKI checkpoint folder before running.")

# test set
if 'CAIA' in dataset:
    print(f"Using split_dict from load_dataset. Test size: {len(split_dict['test'])}")
elif dataset == 'MSK':
    # MSK test set = fold 10 of fold_dict_11
    with open(f'{data_source_path}/MSK/fold_dict_11.pickle', 'rb') as h:
        fold_dict = pickle.load(h)
    tids = [d for d in fold_dict[10]
            if d in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': tids}
    print(f"MSK test size (fold 10): {len(tids)}")
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

# optional weekly-cisplatin exclusion
if exclude_weekly_cisplatin and drugs_dict:
    before = len(split_dict['test'])
    excluded = set(d for d in split_dict['test']
                   if is_weekly_cisplatin(drugs_dict.get(d), max_gap_days=args.weekly_cisplatin_max_gap))
    split_dict['test'] = [d for d in split_dict['test'] if d not in excluded]
    print(f"\n[COHORT] Weekly-cisplatin exclusion: removed {before - len(split_dict['test'])} "
          f"of {before} -> {len(split_dict['test'])} remain")
else:
    print("\n[COHORT] Weekly-cisplatin exclusion OFF (Motwani did not exclude weekly cisplatin)")

model_set_option = 'side_effects_risk'
option = f'{option}_risk'   # AKI_risk
all_folds = list(range(10))

# Motwani score for every test patient
print("\nComputing Motwani scores for test cohort ...")
motwani_by_patient = {}
n_no_cis_c1 = 0
for dmp_id in split_dict['test']:
    feats = build_motwani_features(dmp_id)
    score, prob = motwani_score_and_prob(feats, anchor=prob_anchor,
                                         include_dose=MOTWANI_INCLUDE_DOSE)
    if not feats['_has_cisplatin_cycle1']:
        n_no_cis_c1 += 1
    motwani_by_patient[dmp_id] = {'score': score, 'prob': prob,
                                  'group': motwani_risk_group(score), 'feats': feats}

# coverage
_n = len(motwani_by_patient)
_dose_avail = sum(1 for v in motwani_by_patient.values() if v['feats']['_dose_mg'] is not None)
_alb_avail = sum(1 for v in motwani_by_patient.values() if v['feats']['_albumin_gdl'] is not None)
_htn_yes = sum(1 for v in motwani_by_patient.values() if v['feats']['htn'] == 'yes')
print(f"Motwani scored {_n} patients. "
      f"cisplatin@cycle1: {_n - n_no_cis_c1}/{_n}; "
      f"dose available: {_dose_avail}/{_n}; albumin available: {_alb_avail}/{_n}; "
      f"HTN=yes: {_htn_yes}/{_n}")
print("  Group counts: " +
      ", ".join(f"{g}={sum(1 for v in motwani_by_patient.values() if v['group']==g)}"
                for g in ['low', 'intermediate', 'high', 'very_high']))
if _dose_avail == 0:
    print("  [dose][WARN] No cisplatin doses parsed -> every dose defaults to the <=100 band "
          "(0 pts). Fix the dose field detection before trusting the comparison.")

# SPARC-P inference on all cycles (masked to cycle 1 later)
pl.seed_everything(rseed)
anc_mode = True if 'CAIA' in dataset else False

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)   # fixed order, needed for the positional ensemble
print("data module initiated (G-CSF-free, test_shuffle=False)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sparc_fold_store = {}

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
    batch_first=(args.batch_first == 'True'),
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

    # next-infusion AKI label (event==1 & tte==1)
    row_label = ((all_events == 1) & (all_ttes == 1)).astype(int)
    sparc_fold_store[val_fold] = {
        'preds': all_preds, 'label': row_label, 'dmp': all_dmp, 'cycle': all_infus,
    }
    print(f"fold {val_fold}: {len(all_preds)} rows, {int(row_label.sum())} next-inf AKI positives")


# Ensemble = row-wise mean across folds, as in eval_1. Folds see the same test rows
# in the same order, so fold 0's labels/dmp/cycle are used. Don't key by (dmp, cycle):
# a patient can have several windows at the same cycle.
print("\nBuilding SPARC-P ensemble (positional row-wise mean across folds) ...")

# positional stacking needs equal row counts and matching labels across folds
fold_lens = {f: len(sparc_fold_store[f]['preds']) for f in all_folds}
if len(set(fold_lens.values())) != 1:
    raise RuntimeError(f"Fold row counts differ {fold_lens}; positional ensemble "
                       f"stacking is invalid. Check that all folds share the same "
                       f"test set / ordering.")

ref = sparc_fold_store[0]
for f in all_folds[1:]:
    if not np.array_equal(sparc_fold_store[f]['label'], ref['label']):
        n_mismatch = int((sparc_fold_store[f]['label'] != ref['label']).sum())
        print(f"[ENSEMBLE][WARN] fold {f} labels differ from fold 0 in {n_mismatch} rows; "
              f"test-set ordering may not be aligned across folds.")

fold_preds = np.stack([sparc_fold_store[f]['preds'] for f in all_folds], axis=0)  # (10, N)
ens_pred_all = fold_preds.mean(axis=0)                                             # (N,)
ens_label_all = ref['label'].astype(int)
ens_dmp_all = ref['dmp']
ens_cycle_all = ref['cycle'].astype(int)
print(f"Ensemble rows: {len(ens_pred_all)}, positives: {int(ens_label_all.sum())}")


# first-cycle mask + Motwani prob per row
first_idx = int(ens_cycle_all.min())
fc_mask = (ens_cycle_all == first_idx)

motwani_prob_rows = np.array([
    motwani_by_patient.get(d, {}).get('prob', np.nan) for d in ens_dmp_all
])

# Motwani cohort: first cycle contains cisplatin
if require_cisplatin_cycle1:
    has_cis_row = np.array([
        bool(motwani_by_patient.get(d, {}).get('feats', {}).get('_has_cisplatin_cycle1', False))
        for d in ens_dmp_all
    ])
    motwani_prob_rows = np.where(has_cis_row, motwani_prob_rows, np.nan)


# Motwani_retrained: LR on the same features, fit on train+val
print("\n=== Training MOTWANI_retrained logistic regression model (train+val combined) ===")

_, _, split_dict_trainval, _, _ = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option=option,
    modules=modules, valfold=0, mode='train', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=False,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)

train_val_ids = list(split_dict_trainval['train']) + list(split_dict_trainval['val'])
print(f"Combined train+val: {len(train_val_ids)} patients")

if exclude_weekly_cisplatin:
    before = len(train_val_ids)
    excluded_train = [d for d in train_val_ids
                     if is_weekly_cisplatin(drugs_dict.get(d), max_gap_days=args.weekly_cisplatin_max_gap)]
    train_val_ids = [d for d in train_val_ids if d not in set(excluded_train)]
    print(f"  Weekly-cisplatin exclusion: removed {before - len(train_val_ids)}")

split_dict_combined = {
    'train': train_val_ids,
    'val': [],
    'test': split_dict_trainval['test']
}

data_module_combined = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict_combined, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)

# first-cycle features/labels from train+val
train_features_list = []
train_labels = []
train_seen_patients = set()

print("Extracting first-cycle features from train+val patients...")

# pass 1: global first cycle index
all_num_inf_train = []
for batch in data_module_combined.train_dataloader():
    try:
        (xs, times, num_infusion, dmp_ids), y = batch
    except:
        continue

    num_inf_np = num_infusion.cpu().numpy() if isinstance(num_infusion, torch.Tensor) else np.array(num_infusion)
    all_num_inf_train.extend(num_inf_np.tolist())

train_first_idx = int(np.min(all_num_inf_train)) if all_num_inf_train else 0
print(f"  Train first cycle index: {train_first_idx}")

# pass 2: one first-cycle row per patient
for batch in data_module_combined.train_dataloader():
    try:
        (xs, times, num_infusion, dmp_ids), y = batch
    except:
        continue

    if isinstance(y, (list, tuple)) and len(y) >= 2:
        fail = y[0]
        tte = y[1]
    else:
        continue

    if isinstance(fail, torch.Tensor):
        fail = fail.cpu().numpy()
    else:
        fail = np.array(fail)

    if isinstance(tte, torch.Tensor):
        tte = tte.cpu().numpy()
    else:
        tte = np.array(tte)

    if isinstance(num_infusion, torch.Tensor):
        num_infusion = num_infusion.cpu().numpy()
    else:
        num_infusion = np.array(num_infusion)

    if isinstance(dmp_ids, torch.Tensor):
        dmp_ids = dmp_ids.cpu().tolist()
    elif not isinstance(dmp_ids, (list, tuple)):
        dmp_ids = list(dmp_ids)

    for i in range(len(dmp_ids)):
        dmp_id = dmp_ids[i]
        if num_infusion[i] == train_first_idx and dmp_id not in train_seen_patients:
            feats = build_motwani_features(dmp_id)

            if require_cisplatin_cycle1:
                if not feats.get('_has_cisplatin_cycle1', False):
                    continue

            train_seen_patients.add(dmp_id)
            train_features_list.append(feats)

            label = 1 if (fail[i] == 1 and tte[i] == 1) else 0
            train_labels.append(label)

X_train, feature_names = build_motwani_feature_matrix(train_features_list, include_dose=MOTWANI_INCLUDE_DOSE)
y_train = np.array(train_labels)

print(f"Training samples: {len(y_train)} unique patients, {int(y_train.sum())} positives ({y_train.sum()/len(y_train) if len(y_train) > 0 else 0:.2%})")

if len(y_train) > 0 and len(np.unique(y_train)) == 2:
    print("Fitting logistic regression model...")
    lr_model = LogisticRegression(random_state=rseed, max_iter=1000, class_weight='balanced', solver='lbfgs')
    lr_model.fit(X_train, y_train)

    print(f"\nTrained model intercept: {lr_model.intercept_[0]:.4f}")
    print(f"Top coefficients:")
    for fname, coef in zip(feature_names, lr_model.coef_[0]):
        print(f"  {fname:20s}: {coef:7.4f}")

    # predict on unique first-cycle test patients (same cisplatin filter as Motwani)
    fc_dmps_at_test = ens_dmp_all[fc_mask]
    fc_test_dmp_unique = []
    seen = set()
    for d in fc_dmps_at_test:
        if d not in seen and d in motwani_by_patient:
            if require_cisplatin_cycle1:
                if not motwani_by_patient[d]['feats'].get('_has_cisplatin_cycle1', False):
                    continue
            fc_test_dmp_unique.append(d)
            seen.add(d)

    test_features_list = [motwani_by_patient[d]['feats'] for d in fc_test_dmp_unique]
    X_test, _ = build_motwani_feature_matrix(test_features_list, include_dose=MOTWANI_INCLUDE_DOSE)
    motwani_retrained_probs_test = lr_model.predict_proba(X_test)[:, 1]

    patient_to_pred = {fc_test_dmp_unique[i]: motwani_retrained_probs_test[i]
                       for i in range(len(fc_test_dmp_unique))}

    # broadcast back to every first-cycle row of that patient
    motwani_retrained_prob_rows = np.full(len(ens_dmp_all), np.nan)
    fc_indices = np.where(fc_mask)[0]
    fc_dmps = ens_dmp_all[fc_mask]

    for i, d in enumerate(fc_dmps):
        if d in patient_to_pred:
            motwani_retrained_prob_rows[fc_indices[i]] = patient_to_pred[d]

    print(f"\nMOTWANI_retrained predictions computed for {len(fc_test_dmp_unique)} unique test patients "
          f"(cisplatin-filter={require_cisplatin_cycle1}), "
          f"mapped to {int((~np.isnan(motwani_retrained_prob_rows[fc_mask])).sum())} first-cycle rows")
else:
    print(f"[WARN] Cannot train MOTWANI_retrained: insufficient training data or single-class labels")
    motwani_retrained_prob_rows = np.full(len(ens_dmp_all), np.nan)


# SPARC-P vs comparator on the same cycle-1 rows
def _bootstrap_ci(y, p, metric='auroc', n_boot=1000, seed=1024):
    # sklearn here: torchmetrics is far slower in a 1000x loop, and CIs don't need the exact match
    from sklearn.metrics import roc_auc_score as _sk_auroc, average_precision_score as _sk_auprc
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(_sk_auroc(yb, pb) if metric == 'auroc'
                    else _sk_auprc(yb, pb))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def valid_infusion_mask(labels, cycles):
    """Rows whose cycle has both classes (the eval_1 'overall' row selection)."""
    keep = np.zeros(len(labels), dtype=bool)
    for c in np.unique(cycles):
        m = cycles == c
        s = labels[m].sum()
        if s > 0 and s < m.sum():
            keep |= m
    return keep


def _overall_auroc_auprc(preds, labels, cycles):
    """AUROC/AUPRC over valid-infusion rows (test_auroc_nextinf_overall)."""
    keep = valid_infusion_mask(labels, cycles)
    y, p = labels[keep], preds[keep]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, 0, 0
    return roc_auc_score(y, p), average_precision_score(y, p), int(keep.sum()), int(y.sum())


def _weighted_auroc_auprc(preds, labels, cycles):
    """Per-cycle AUROC/AUPRC, mean weighted by cycle size (test_auroc_nextinf_weighted)."""
    per = []
    for c in np.unique(cycles):
        m = cycles == c
        y, p = labels[m], preds[m]
        s = y.sum()
        if 0 < s < m.sum():
            per.append({
                'auroc': roc_auc_score(y, p),
                'auprc': average_precision_score(y, p),
                'n': int(m.sum()),
            })
    if not per:
        return np.nan, np.nan, 0, 0
    tot = sum(r['n'] for r in per)
    w_auroc = sum(r['auroc'] * r['n'] / tot for r in per)
    w_auprc = sum(r['auprc'] * r['n'] / tot for r in per)
    return w_auroc, w_auprc, len(per), tot


rows = []


def _run_comparison(tag, mask, comparator_prob, comparator_name):
    y = ens_label_all[mask]
    sp = ens_pred_all[mask]
    cp = comparator_prob[mask]
    valid = ~np.isnan(cp)
    y, sp, cp = y[valid], sp[valid], cp[valid]
    n, npos = len(y), int(y.sum())
    print(f"\n================= {tag} =================")
    print(f"n = {n}, positives = {npos} ({(npos/n if n else 0):.2%})")
    if n == 0 or len(np.unique(y)) < 2:
        print("[WARN] single-class or empty subset; cannot compute AUROC/AUPRC.")
        rows.append({'comparison': tag, 'model': 'SPARC-P_ensemble', 'n': n, 'n_pos': npos,
                     'auroc': None, 'auprc': None})
        rows.append({'comparison': tag, 'model': comparator_name, 'n': n, 'n_pos': npos,
                     'auroc': None, 'auprc': None})
        return
    sp_auroc = roc_auc_score(y, sp); sp_auprc = average_precision_score(y, sp)
    sp_a_ci = _bootstrap_ci(y, sp, 'auroc'); sp_p_ci = _bootstrap_ci(y, sp, 'auprc')
    ce_auroc = roc_auc_score(y, cp); ce_auprc = average_precision_score(y, cp)
    ce_a_ci = _bootstrap_ci(y, cp, 'auroc'); ce_p_ci = _bootstrap_ci(y, cp, 'auprc')

    # DeLong test on paired AUROCs
    try:
        delong_result = delong_roc_test(y, sp, cp)
        delong_pvalue = delong_result['pvalue']
        delong_z = delong_result['z_score']
        print(f"DeLong test: z={delong_z:.4f}, p={delong_pvalue:.4e}")
    except Exception as e:
        print(f"[WARN] DeLong test failed: {e}")
        delong_pvalue = np.nan
        delong_z = np.nan

    print(f"SPARC-P (ensemble): AUROC={sp_auroc:.4f} [{sp_a_ci[0]:.4f},{sp_a_ci[1]:.4f}]  "
          f"AUPRC={sp_auprc:.4f} [{sp_p_ci[0]:.4f},{sp_p_ci[1]:.4f}]")
    print(f"{comparator_name:<18}: AUROC={ce_auroc:.4f} [{ce_a_ci[0]:.4f},{ce_a_ci[1]:.4f}]  "
          f"AUPRC={ce_auprc:.4f} [{ce_p_ci[0]:.4f},{ce_p_ci[1]:.4f}]")
    rows.append({'comparison': tag, 'model': 'SPARC-P_ensemble', 'n': n, 'n_pos': npos,
                 'auroc': sp_auroc, 'auroc_ci_lower': sp_a_ci[0], 'auroc_ci_upper': sp_a_ci[1],
                 'auprc': sp_auprc, 'auprc_ci_lower': sp_p_ci[0], 'auprc_ci_upper': sp_p_ci[1]})
    rows.append({'comparison': tag, 'model': comparator_name, 'n': n, 'n_pos': npos,
                 'auroc': ce_auroc, 'auroc_ci_lower': ce_a_ci[0], 'auroc_ci_upper': ce_a_ci[1],
                 'auprc': ce_auprc, 'auprc_ci_lower': ce_p_ci[0], 'auprc_ci_upper': ce_p_ci[1]})
    # significance row: p-value in auroc, z in auroc_ci_lower, AUROC diff in auroc_ci_upper
    rows.append({'comparison': f'{tag}_significance_test',
                 'model': f'SPARC-P_vs_{comparator_name}',
                 'n': n, 'n_pos': npos,
                 'auroc': delong_pvalue,
                 'auroc_ci_lower': delong_z,
                 'auroc_ci_upper': sp_auroc - ce_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})


# Diagnostics: SPARC-P only, across all cycles (Motwani is first-course only)
if run_diagnostics:
    print("\n############### SPARC-P ALL-CYCLES DIAGNOSTIC ###############")
    print(f"[DIAG] per-infusion label / count breakdown (ensemble rows):")
    print(f"[DIAG] {'cycle':>6} {'n':>7} {'n_pos':>7} {'pos_rate':>9}")
    for iv in np.unique(ens_cycle_all)[:25]:
        m = ens_cycle_all == iv
        npos = int(ens_label_all[m].sum())
        n = int(m.sum())
        print(f"[DIAG] {iv:>6.0f} {n:>7} {npos:>7} {(npos/n if n else 0):>9.2%}")

    # overall metric as reported by eval_1 (valid-infusion rows only)
    ov_auroc, ov_auprc, ov_n, ov_pos = _overall_auroc_auprc(ens_pred_all, ens_label_all, ens_cycle_all)
    if not np.isnan(ov_auroc):
        keep = valid_infusion_mask(ens_label_all, ens_cycle_all)
        ov_a_ci = _bootstrap_ci(ens_label_all[keep], ens_pred_all[keep], 'auroc')
        ov_p_ci = _bootstrap_ci(ens_label_all[keep], ens_pred_all[keep], 'auprc')
        n_valid_cyc = int(sum(
            1 for c in np.unique(ens_cycle_all)
            if 0 < ens_label_all[ens_cycle_all == c].sum() < (ens_cycle_all == c).sum()
        ))
        print(f"\n[DIAG] SPARC-P overall (valid-infusion pooling, MATCHES reference): "
              f"n={ov_n}, positives={ov_pos} ({(ov_pos/ov_n if ov_n else 0):.2%}), "
              f"valid cycles={n_valid_cyc}")
        print(f"[DIAG] SPARC-P overall AUROC={ov_auroc:.4f} [{ov_a_ci[0]:.4f},{ov_a_ci[1]:.4f}]  "
              f"AUPRC={ov_auprc:.4f} [{ov_p_ci[0]:.4f},{ov_p_ci[1]:.4f}]")
        rows.append({'comparison': 'SPARC-P_overall_validinfusion_diagnostic', 'model': 'SPARC-P_ensemble',
                     'n': ov_n, 'n_pos': ov_pos,
                     'auroc': ov_auroc, 'auroc_ci_lower': ov_a_ci[0], 'auroc_ci_upper': ov_a_ci[1],
                     'auprc': ov_auprc, 'auprc_ci_lower': ov_p_ci[0], 'auprc_ci_upper': ov_p_ci[1]})
    else:
        print("[DIAG] no valid-infusion rows; cannot compute reference-matching overall AUROC.")

    # weighted metric (test_auroc_nextinf_weighted)
    w_auroc, w_auprc, w_ncyc, w_n = _weighted_auroc_auprc(ens_pred_all, ens_label_all, ens_cycle_all)
    if not np.isnan(w_auroc):
        print(f"[DIAG] SPARC-P WEIGHTED (per-cycle, sample-weighted, MATCHES reference "
              f"test_auroc_nextinf_weighted): AUROC={w_auroc:.4f}  AUPRC={w_auprc:.4f}  "
              f"(over {w_ncyc} valid cycles, {w_n} rows)")
        rows.append({'comparison': 'SPARC-P_weighted_nextinf_diagnostic', 'model': 'SPARC-P_ensemble',
                     'n': w_n, 'n_pos': ov_pos if not np.isnan(ov_auroc) else None,
                     'auroc': w_auroc, 'auprc': w_auprc})
    else:
        print("[DIAG] no valid-infusion cycles; cannot compute weighted AUROC.")

    # naive all-rows AUROC, for contrast only (not what eval_1 reports)
    _y_all = ens_label_all
    _sp_all = ens_pred_all
    if len(np.unique(_y_all)) >= 2:
        sp_all_auroc = roc_auc_score(_y_all, _sp_all)
        sp_all_auprc = average_precision_score(_y_all, _sp_all)
        print(f"[DIAG] SPARC-P ALL rows (naive, incl. all-neg cycles): n={len(_y_all)}, "
              f"positives={int(_y_all.sum())} ({_y_all.mean():.2%}); "
              f"AUROC={sp_all_auroc:.4f}  AUPRC={sp_all_auprc:.4f}")
        rows.append({'comparison': 'SPARC-P_all_rows_naive_diagnostic', 'model': 'SPARC-P_ensemble',
                     'n': len(_y_all), 'n_pos': int(_y_all.sum()),
                     'auroc': sp_all_auroc, 'auprc': sp_all_auprc})
    else:
        print("[DIAG] all-rows subset is single-class; cannot compute AUROC.")

    # later cycles only
    _later = ens_cycle_all > first_idx
    if _later.sum() > 0 and len(np.unique(ens_label_all[_later])) >= 2:
        y_l, sp_l = ens_label_all[_later], ens_pred_all[_later]
        sp_l_auroc = roc_auc_score(y_l, sp_l)
        sp_l_auprc = average_precision_score(y_l, sp_l)
        sp_l_a_ci = _bootstrap_ci(y_l, sp_l, 'auroc')
        sp_l_p_ci = _bootstrap_ci(y_l, sp_l, 'auprc')
        print(f"[DIAG] SPARC-P cycles>{first_idx}: n={int(_later.sum())}, positives={int(y_l.sum())} "
              f"({y_l.mean():.2%})")
        print(f"[DIAG] SPARC-P cycles>{first_idx}: AUROC={sp_l_auroc:.4f} "
              f"[{sp_l_a_ci[0]:.4f},{sp_l_a_ci[1]:.4f}]  "
              f"AUPRC={sp_l_auprc:.4f} [{sp_l_p_ci[0]:.4f},{sp_l_p_ci[1]:.4f}]")
        rows.append({'comparison': f'SPARC-P_cycles_gt_{first_idx}_diagnostic', 'model': 'SPARC-P_ensemble',
                     'n': int(_later.sum()), 'n_pos': int(y_l.sum()),
                     'auroc': sp_l_auroc, 'auroc_ci_lower': sp_l_a_ci[0], 'auroc_ci_upper': sp_l_a_ci[1],
                     'auprc': sp_l_auprc, 'auprc_ci_lower': sp_l_p_ci[0], 'auprc_ci_upper': sp_l_p_ci[1]})
    print("############################################################\n")

_motwani_label = 'Motwani' if MOTWANI_INCLUDE_DOSE else 'Motwani_3var'
_run_comparison(f'MOTWANI_firstcycle (cycle=={first_idx})', fc_mask,
                motwani_prob_rows, _motwani_label)

_motwani_retrained_label = 'Motwani_retrained' if MOTWANI_INCLUDE_DOSE else 'Motwani_retrained_3var'
_run_comparison(f'MOTWANI_retrained_firstcycle (cycle=={first_idx})', fc_mask,
                motwani_retrained_prob_rows, _motwani_retrained_label)

# per-fold first-cycle counts
for f in all_folds:
    s = sparc_fold_store[f]
    fc = (s['cycle'] == first_idx)
    rows.append({'comparison': 'per_fold', 'model': f'SPARC-P_fold{f}_firstcycle',
                 'n': int(fc.sum()), 'n_pos': int(s['label'][fc].sum()),
                 'auroc': None, 'auprc': None})

# settings row
rows.append({'comparison': 'NOTE', 'model':
             f'outcome=OUR_next_infusion_AKI (not Motwani 14d rule); '
             f'prob_anchor={prob_anchor}; require_cisplatin_cycle1={require_cisplatin_cycle1}; '
             f'exclude_weekly_cisplatin={exclude_weekly_cisplatin}; '
             f'motwani_vars={MOTWANI_N_VARS}'
             + ('' if MOTWANI_INCLUDE_DOSE else
                ' (3-VARIABLE: cisplatin dose unavailable in data source, dose term dropped)'),
             'n': None, 'n_pos': None, 'auroc': None, 'auprc': None})

_wc = '_exclWeeklyCis' if exclude_weekly_cisplatin else ''
save_name = (f'{results_path}/SPARC-P_vs_MOTWANI_firstcycle_FIXED_'
             f'{modules}_{dataset}_{caia_portion}_{prob_anchor}{_wc}.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(save_name, index=False)
print(f"\nSaved comparison to: {save_name}")
print(df.to_string(index=False))

# per-patient Motwani audit
audit_rows = []
seen = set()
for d in ens_dmp_all:
    if d in seen:
        continue
    seen.add(d)
    info = motwani_by_patient.get(d, {})
    feats = info.get('feats', {})
    audit_rows.append({
        'dmp_id': d, 'motwani_score': info.get('score'),
        'motwani_prob': info.get('prob'), 'motwani_group': info.get('group'),
        'age': feats.get('_age'), 'age_band': feats.get('age_band'),
        'cisplatin_dose_mg': feats.get('_dose_mg'), 'dose_band': feats.get('dose_band'),
        'hypertension': feats.get('htn'),
        'albumin_gdl': feats.get('_albumin_gdl'), 'albumin_band': feats.get('albumin_band'),
        'has_cisplatin_cycle1': feats.get('_has_cisplatin_cycle1'),
        'cycle1_day': feats.get('_cycle1_day'),
    })
audit_name = save_name.replace('.csv', '_MOTWANI_patient_audit.csv')
pd.DataFrame(audit_rows).to_csv(audit_name, index=False)
print(f"Saved Motwani per-patient audit to: {audit_name}")