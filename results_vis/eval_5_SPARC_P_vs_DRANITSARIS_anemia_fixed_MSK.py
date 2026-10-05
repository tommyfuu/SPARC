"""SPARC-P vs the Dranitsaris severe-anemia (Hb<8 g/dL) model on the MSK anemia cohort.

Comparators: published model as a linear predictor (beta = ln OR, Dranitsaris et al.,
Table 2 training-set ORs; intercept unreported, so prob is uncalibrated but AUROC-invariant)
and the same features refit with logistic regression on train+val.
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
)

HGB_INDEX  = ALL_INCLUDED_TESTS.index('HGB')
MCV_INDEX  = ALL_INCLUDED_TESTS.index('MCV')
CREAT_INDEX = ALL_INCLUDED_TESTS.index('Creatinine')

AGE_BIN_YEARS = 5.0

from delong_test import delong_roc_test

# torchmetrics AUROC/AUPRC to match the reference eval
from torchmetrics import AUROC as _TMAUROC, AveragePrecision as _TMAP
_tm_auroc = _TMAUROC(task='binary')
_tm_auprc = _TMAP(task='binary')


def roc_auc_score(y_true, y_score):
    yt = torch.as_tensor(np.asarray(y_true)).int()
    ys = torch.as_tensor(np.asarray(y_score)).float()
    return _tm_auroc(ys, yt).item()


def average_precision_score(y_true, y_score):
    yt = torch.as_tensor(np.asarray(y_true)).int()
    ys = torch.as_tensor(np.asarray(y_score)).float()
    return _tm_auprc(ys, yt).item()


# Dranitsaris severe-CIA model
# beta = ln(OR), Table 2 training-set ORs.
DRAN_BETA = {
    'age_gt65':   np.log(1.47),   # +0.38526
    'hb_low':     np.log(3.14),   # +1.14422 (M<14 / F<12 g/dL)
    'steroid':    np.log(1.83),   # +0.60432
    'mcv_lt81':   np.log(1.06),   # +0.05827
    'mcv_gt99':   np.log(3.20),   # +1.16315
    'gfr_lt60':   np.log(2.01),   # +0.69813
    # TNM stage vs I/II
    'stage_iii':  np.log(1.57),   # +0.45108
    'stage_iv':   np.log(1.71),   # +0.53649
    # Chemo regimen vs AC->TXT/TAC (approximate mapping, see get_regimen_category)
    'regimen_ac_tax':      np.log(1.43),
    'regimen_carbo_tax':   np.log(1.50),
    'regimen_carbo_txt':   np.log(2.10),
    'regimen_cmf':         np.log(0.52),
    'regimen_other':       np.log(1.08),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_egfr_ckd_epi(creat_mgdl, age, is_female):
    """CKD-EPI 2009 eGFR (mL/min/1.73m^2), same as the FENCE script."""
    if creat_mgdl is None or age is None or creat_mgdl <= 0:
        return None
    if is_female:
        k, alpha, sexf = 0.7, -0.329, 1.018
    else:
        k, alpha, sexf = 0.9, -0.411, 1.0
    ratio = creat_mgdl / k
    return 141 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.209) * (0.993 ** age) * sexf


def dranitsaris_features_to_lp(feats, include_steroid=True, include_stage=True,
                               include_regimen=True):
    """Dranitsaris linear predictor (sum of beta*x)."""
    B = DRAN_BETA
    lp = 0.0
    if feats['age'] is not None and feats['age'] > 65:
        lp += B['age_gt65']
    if feats['hb'] is not None and feats['sex'] is not None:
        thresh = 12.0 if feats['sex'] == 'female' else 14.0
        if feats['hb'] < thresh:
            lp += B['hb_low']
    if include_steroid and feats.get('steroid'):
        lp += B['steroid']
    if feats['mcv'] is not None:
        if feats['mcv'] < 81:
            lp += B['mcv_lt81']
        elif feats['mcv'] > 99:
            lp += B['mcv_gt99']
    if feats['egfr'] is not None and feats['egfr'] < 60:
        lp += B['gfr_lt60']
    if include_stage:
        st = feats.get('stage_num')
        if st == 3:
            lp += B['stage_iii']
        elif st is not None and st >= 4:
            lp += B['stage_iv']
    if include_regimen:
        reg = feats.get('regimen', 'reference')
        key = {'ac_tax': 'regimen_ac_tax', 'carbo_tax': 'regimen_carbo_tax',
               'carbo_txt': 'regimen_carbo_txt', 'cmf': 'regimen_cmf',
               'other': 'regimen_other'}.get(reg)
        if key:
            lp += B[key]
    return lp


def dranitsaris_score_and_prob(feats, include_steroid=True):
    lp = dranitsaris_features_to_lp(feats, include_steroid=include_steroid)
    return lp, _sigmoid(lp)


def build_feature_matrix(features_list, include_steroid=True):
    """Indicator matrix for the retrained LR (same terms as the published model)."""
    X = []
    _REG = ['ac_tax', 'carbo_tax', 'carbo_txt', 'cmf', 'other']  # reference dropped
    for feats in features_list:
        row = []
        row.append(1.0 if (feats['age'] is not None and feats['age'] > 65) else 0.0)
        if feats['hb'] is not None and feats['sex'] is not None:
            thresh = 12.0 if feats['sex'] == 'female' else 14.0
            row.append(1.0 if feats['hb'] < thresh else 0.0)
        else:
            row.append(0.0)
        if include_steroid:
            row.append(1.0 if feats.get('steroid') else 0.0)
        if feats['mcv'] is not None:
            row.append(1.0 if feats['mcv'] < 81 else 0.0)
            row.append(1.0 if feats['mcv'] > 99 else 0.0)
        else:
            row.append(0.0); row.append(0.0)
        row.append(1.0 if (feats['egfr'] is not None and feats['egfr'] < 60) else 0.0)
        st = feats.get('stage_num')
        row.append(1.0 if st == 3 else 0.0)
        row.append(1.0 if (st is not None and st >= 4) else 0.0)
        reg = feats.get('regimen', 'reference')
        for rc in _REG:
            row.append(1.0 if reg == rc else 0.0)
        X.append(row)

    feature_names = ['age_gt65', 'hb_low']
    if include_steroid:
        feature_names.append('steroid')
    feature_names.extend(['mcv_lt81', 'mcv_gt99', 'gfr_lt60', 'stage_iii', 'stage_iv'])
    feature_names.extend([f'regimen_{r}' for r in _REG])

    return np.array(X), feature_names


# Arguments
# Default roots; override via env var or CLI flag.
SPARC_DATA_ROOT = os.environ.get(
    'SPARC_DATA_ROOT', '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data')
SPARC_MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_ROOT', '/data1/morrisq/fuc/SPARC_CAIA_092426')
SPARC_RESULTS_ROOT = os.environ.get(
    'SPARC_RESULTS_ROOT', '/data1/morrisq/fuc/SPARC_CAIA_092426/results_vis_eval1_bfTrue')
SPARC_REF_ROOT = os.environ.get(
    'SPARC_REF_ROOT', '/nonexistent/no_bf_false_reference_row')

parser = argparse.ArgumentParser(prog='eval_SPARC_P_vs_DRANITSARIS_anemia_fixed')
parser.add_argument("-r", "--rseed", type=int, default=1024)
parser.add_argument("-d", "--days", type=int, default=30)
parser.add_argument("-m", "--modules", type=str, default='parsimonious')
parser.add_argument("-ds", "--dataset", type=str, default='MSK')
parser.add_argument("-me", "--med_of_choice", type=str,
                    default='DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                            'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                            'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL')
parser.add_argument("-cp", "--caia_portion", type=str, default="post")
parser.add_argument("-z", "--zeroshot", type=str, default='True')
parser.add_argument("-mm", "--missing_mask", type=str, default="True")
parser.add_argument("-en", "--ensemble", type=str, default='True')
parser.add_argument("-dsp", "--data_source_path", type=str, default=f'{SPARC_DATA_ROOT}/')
parser.add_argument("-mp", "--model_path", type=str,
                    default=os.environ.get('SPARC_EVAL7_MODEL_ROOT', '/data1/morrisq/fuc/SPARC_MSK_sparcp_bfTrue_092926'))
parser.add_argument("-rp", "--results_path", type=str,
                    default=f'{SPARC_RESULTS_ROOT}/eval7_comparators_MSK_bfTrue')
parser.add_argument("--existing_csv", type=str,
                    default=f'{SPARC_REF_ROOT}/'
                            'eval_across_models_valthresh_disabled/model_performance_dict_across_models_parsimonious_anemia_MSK_ensemble.csv')
parser.add_argument("--diagnostics", type=str, default='True')
parser.add_argument("--steroid_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/DMP_ID_to_DATE_to_STEROID_dict.pkl',
                    help="Steroid exposure source. .pkl/.pickle: {DMP_ID: {day: [drug, ...]}}. .csv: the legacy dexamethasone export.")
parser.add_argument("--steroid_marker", type=str, default='LT_steroids',
                    help="Comma-separated drug names counted as steroid exposure when --steroid_path is a pickle. Default: dexamethasone, methylprednisolone, prednisone (methotrexate is excluded -- not a corticosteroid). Unused for the legacy CSV source.")
parser.add_argument("--stage_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/DMP_ID_to_DATE_to_STAGE_DICT_MSK_021026.pickle')
parser.add_argument("--demographic_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/dePHIed_demographic_df_MSK.csv')
parser.add_argument("--med_classification_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/med_classification_llm.pickle')
parser.add_argument("--cohort_version", type=str, default='all',
                    choices=['all', 'all_breast', 'capecitabine', 'both'],
                    help="Which cohort to evaluate. "
                         "'all' (default): NO cancer-type restriction -- the full "
                         "MSK anemia cohort (~1773 patients), well-powered and "
                         "consistent with the other clinical benchmarks. "
                         "'all_breast': breast only, i.e. Dranitsaris's derivation "
                         "cancer type (~63 patients -- faithful external validation "
                         "but very small). "
                         "'both': DEPRECATED -- it is not a union; it falls through "
                         "to the same breast-only set as 'all_breast'. "
                         "'capecitabine': breast AND capecitabine -- ALWAYS EMPTY, "
                         "because capecitabine is oral and the meds source is "
                         "non-oral only; the script exits if selected.")
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
existing_csv_path = args.existing_csv
run_diagnostics = args.diagnostics.lower() == 'true'
steroid_path = args.steroid_path
steroid_marker = args.steroid_marker
stage_path = args.stage_path
demographic_path = args.demographic_path
med_classification_path = args.med_classification_path
cohort_version = args.cohort_version

# Fixed for this analysis
option = 'anemia'
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

# Architecture params (anemia uses 'hybrid' finetuned checkpoints)
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

# Optional all-cycles reference row from an existing eval CSV
print(f"\nLoading existing CSV (optional) from: {existing_csv_path}")
HAVE_EXISTING_CSV = os.path.exists(existing_csv_path)
if HAVE_EXISTING_CSV:
    existing_df = pd.read_csv(existing_csv_path)
    _rows = existing_df[existing_df['model'] == 'SPARC-P_ensemble']
    HAVE_EXISTING_CSV = len(_rows) > 0
if HAVE_EXISTING_CSV:
    sparc_p_ensemble_row = _rows.iloc[0]
    n_test_existing = int(sparc_p_ensemble_row['n_test'])
    n_pos_overall_existing = int(sparc_p_ensemble_row['n_pos_overall'])
    n_pos_next_infusion_existing = int(sparc_p_ensemble_row['n_pos_next_infusion'])
    auroc_existing = float(sparc_p_ensemble_row['test_auroc_nextinf_overall'])
    auprc_existing = float(sparc_p_ensemble_row['test_auprc_nextinf_overall'])
    auroc_ci_lower_existing = float(sparc_p_ensemble_row['auroc_ci_lower'])
    auroc_ci_upper_existing = float(sparc_p_ensemble_row['auroc_ci_upper'])
    auprc_ci_lower_existing = float(sparc_p_ensemble_row['auprc_ci_lower'])
    auprc_ci_upper_existing = float(sparc_p_ensemble_row['auprc_ci_upper'])
    print(f"Existing SPARC-P ensemble metrics (all-cycles reference):")
    print(f"  n_test={n_test_existing}, n_pos_overall={n_pos_overall_existing}, "
          f"n_pos_next_infusion={n_pos_next_infusion_existing}")
    print(f"  AUROC={auroc_existing:.4f} [{auroc_ci_lower_existing:.4f}, {auroc_ci_upper_existing:.4f}]")
    print(f"  AUPRC={auprc_existing:.4f} [{auprc_ci_lower_existing:.4f}, {auprc_ci_upper_existing:.4f}]")
else:
    print("  [existing CSV not found] -> skipping the all-cycles reference diagnostic row. "
          "SPARC-P first-cycle + patient-level metrics are computed directly below from "
          "this script's own fold inference (no external CSV needed).")
    n_test_existing = n_pos_overall_existing = n_pos_next_infusion_existing = None
    auroc_existing = auprc_existing = None
    auroc_ci_lower_existing = auroc_ci_upper_existing = None
    auprc_ci_lower_existing = auprc_ci_upper_existing = None

print("\nLoading dataset ...")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option=option, modules=modules,
    valfold=0, mode='test_all' if 'timeshift' in dataset else 'test', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)
print("Data loaded.")

module_index_cfg = build_module_index_config(modular_featuredim_dict, advanced_model='default')
n_static_features = module_index_cfg['n_static']

print("\nLoading Dranitsaris auxiliary data (MSK) ...")

def _try_load_pickle(path, label):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  [{label}] loaded ({len(obj)} entries) from {path}")
        return obj
    print(f"  [{label}] NOT FOUND at {path} -> will default")
    return {}

import pandas as _pd
from collections import defaultdict as _dd

# Steroid exposure -> {DMP_ID: [days]}.
# .pkl: {DMP_ID: {day: [drug, ...]}}; counts dexamethasone, methylprednisolone,
#   prednisone (the file also has methotrexate, which is not a corticosteroid).
# .csv: legacy dexamethasone-only export.
steroid_days = {}
_is_pickle = str(steroid_path).lower().endswith(('.pkl', '.pickle'))

if _is_pickle and os.path.exists(steroid_path):
    _marker = (args.steroid_marker or '').strip()
    if _marker and _marker.lower() != 'lt_steroids':
        _STEROID_DRUGS = {m.strip().lower() for m in _marker.split(',') if m.strip()}
    else:
        _STEROID_DRUGS = {'dexamethasone', 'methylprednisolone', 'prednisone'}
    with open(steroid_path, 'rb') as _fh:
        _sdict = pickle.load(_fh)
    tmp = _dd(list)
    for _pid, _day_map in _sdict.items():
        if not isinstance(_day_map, dict):
            continue
        for _day, _drugs in _day_map.items():
            if not isinstance(_drugs, (list, tuple)):
                _drugs = [_drugs]
            if any(str(g).strip().lower() in _STEROID_DRUGS for g in _drugs):
                tmp[_pid].append(int(_day))
    steroid_days = dict(tmp)
    print(f"  [steroid] {len(steroid_days)} patients from {steroid_path} "
          f"(pickle; drugs={sorted(_STEROID_DRUGS)})")
elif os.path.exists(steroid_path):
    _sdf = _pd.read_csv(steroid_path)
    _id_col = next((c for c in _sdf.columns if c.upper() in ('DMP_ID', 'MRN', 'PATIENT_ID')), None)
    _day_col = next((c for c in _sdf.columns
                     if 'anchor' in c.lower() or 'dte_since' in c.lower() or 'day' in c.lower()), None)
    if _id_col and _day_col:
        tmp = _dd(list)
        for _pid, _day in zip(_sdf[_id_col], _sdf[_day_col]):
            if _pd.notna(_day):
                tmp[_pid].append(int(_day))
        steroid_days = dict(tmp)
        print(f"  [steroid] {len(steroid_days)} patients from {steroid_path} "
              f"(id={_id_col}, day={_day_col})")
    else:
        print(f"  [steroid][WARN] could not find id/day columns in {steroid_path}; "
              f"columns={list(_sdf.columns)} -> steroid disabled")
else:
    print(f"  [steroid] NOT FOUND at {steroid_path} -> steroid term disabled")
DRAN_INCLUDE_STEROID = len(steroid_days) > 0
DRAN_N_VARS = 8 if DRAN_INCLUDE_STEROID else 7

# TNM stage {DMP_ID: {day: 1-4}}
stage_dict = _try_load_pickle(stage_path, 'stage')

# Cancer type, for the breast filter
CTYPE_DICT = {}
if os.path.exists(demographic_path):
    _dfd = _pd.read_csv(demographic_path)
    _ct_col = 'primary_cancer_type_th' if 'primary_cancer_type_th' in _dfd.columns else 'primary_cancer_type'
    CTYPE_DICT = dict(zip(_dfd['DMP_ID'], _dfd[_ct_col]))
    print(f"  [cancer_type] {len(CTYPE_DICT)} patients (col {_ct_col})")
else:
    print(f"  [cancer_type][WARN] demographic CSV not found at {demographic_path}")

# LLM drug classification, for regimen mapping
med_cls = _try_load_pickle(med_classification_path, 'med_classification')
med_cls = {str(k).strip().upper(): v for k, v in med_cls.items()}

MED_DICT = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})
print(f"  [meds] {len(MED_DICT)} patients in meds dict")

LAB_DEF = data_dict_lab_dict.get('lab_def_dict', {})
if not LAB_DEF:
    print("[WARN] lab_def_dict empty; labs default to missing.")


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def get_age_years(dmp_id):
    a = data_dict_lab_dict.get('age', {}).get(dmp_id)
    if a is None:
        return None
    try:
        b = float(a[0]) if isinstance(a, (list, tuple, np.ndarray)) else float(a)
    except Exception:
        return None
    return b * AGE_BIN_YEARS


def get_sex(dmp_id):
    g = data_dict_lab_dict.get('gender', {}).get(dmp_id)
    if g is None:
        return 'male'
    try:
        female_flag = float(g[0]) if isinstance(g, (list, tuple, np.ndarray)) else float(g)
    except Exception:
        return 'male'
    return 'female' if int(round(female_flag)) == 1 else 'male'


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


def _lookup_flexible(d, dmp_id):
    if dmp_id in d:
        return d[dmp_id]
    for alt in (str(dmp_id), _safe_int(dmp_id)):
        if alt is not None and alt in d:
            return d[alt]
    return None


def _patient_index_day(dmp_id):
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return None
    days = [d for d in sched.keys() if _safe_int(d) is not None]
    return min(int(d) for d in days) if days else None


def _entry_name(e):
    try:
        return str(e[0]).strip().upper()
    except (IndexError, TypeError):
        return str(e).strip().upper()


def get_steroid(dmp_id, cutoff_date=None):
    """True if any steroid day is on/before cutoff (index day if cutoff is None)."""
    days = steroid_days.get(dmp_id)
    if days is None:
        days = steroid_days.get(str(dmp_id)) or steroid_days.get(_safe_int(dmp_id))
    if not days:
        return False
    cut = cutoff_date if cutoff_date is not None else _patient_index_day(dmp_id)
    if cut is None:
        return len(days) > 0
    return any(_safe_int(d) is not None and int(d) <= cut for d in days)


def get_stage_num(dmp_id, cutoff_date=None):
    """Max TNM stage (1-4) on/before cutoff, or None."""
    rec = _lookup_flexible(stage_dict, dmp_id)
    if not rec:
        return None
    if isinstance(rec, dict):
        if cutoff_date is not None:
            days = [d for d in rec.keys() if _safe_int(d) is not None and int(d) <= cutoff_date]
        else:
            days = [d for d in rec.keys() if _safe_int(d) is not None]
        if not days:
            days = [d for d in rec.keys() if _safe_int(d) is not None]
        vals = [rec[d] for d in days]
    else:
        vals = list(rec)
    numeric = [_safe_int(v) for v in vals if _safe_int(v) is not None]
    return max(numeric) if numeric else None


def get_regimen_category(dmp_id, cutoff_date=None):
    """Approximate Dranitsaris regimen category from drug classes on the index day.

    Returns 'reference' (AC->TXT/TAC), 'ac_tax', 'carbo_tax', 'carbo_txt', 'cmf' or 'other'.
    """
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return 'other'
    idx = _patient_index_day(dmp_id)
    if idx is None:
        return 'other'
    entries = sched.get(idx, sched.get(str(idx), []))
    classes = set()
    names = set()
    for e in entries:
        nm = _entry_name(e)
        names.add(nm)
        m = med_cls.get(nm)
        if m and m.get('is_chemotherapy'):
            classes.add(str(m.get('chemo_class', '')).lower())
    has_plat = 'platinum' in classes
    has_tax = 'taxane' in classes
    has_alk = 'nonplatinum_alkylating' in classes
    has_anti = 'antimetabolite' in classes
    if has_plat and has_tax:
        return 'carbo_tax'
    if has_plat:
        return 'carbo_txt'
    if has_anti and has_alk:
        return 'cmf'          # CMF = cyclophosphamide/methotrexate/5-FU
    if has_alk and has_tax:
        return 'reference'    # AC->TXT / TAC (anthracycline+alkylating -> taxane)
    if has_tax:
        return 'ac_tax'
    return 'other'


# Cohort filters
def is_breast(dmp_id):
    ct = CTYPE_DICT.get(dmp_id)
    return ct is not None and 'breast' in str(ct).strip().lower()


def takes_capecitabine(dmp_id):
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict):
        return False
    for entries in sched.values():
        for e in entries:
            if _entry_name(e) == 'CAPECITABINE':
                return True
    return False


def build_dranitsaris_features(dmp_id, cutoff_date=None):
    age = get_age_years(dmp_id)
    sex = get_sex(dmp_id)
    creat = _baseline_lab_value(dmp_id, CREAT_INDEX, cutoff_date)
    egfr = compute_egfr_ckd_epi(creat, age, sex == 'female')
    return {
        'age': age,
        'sex': sex,
        'hb':  _baseline_lab_value(dmp_id, HGB_INDEX, cutoff_date),   # g/dL, raw
        'mcv': _baseline_lab_value(dmp_id, MCV_INDEX, cutoff_date),   # fL, raw
        'egfr': egfr,
        'steroid': get_steroid(dmp_id, cutoff_date),
        'stage_num': get_stage_num(dmp_id, cutoff_date),   # 1-4 or None
        'regimen': get_regimen_category(dmp_id, cutoff_date),  # dranitsaris category
        '_creat': creat, '_cutoff': cutoff_date,
    }


# SPARC-P model folder
all_model_path = model_path
base_option = 'anemia'
model_type_suffix = 'hybrid'

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

# Test split
if 'CAIA' in dataset:
    print(f"Using split_dict from load_dataset. Test size: {len(split_dict['test'])}")
elif dataset == 'MSK':
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

# Cohort restriction (see --cohort_version). Adjuvant setting can't be enforced (no surgery data).
_test_all = list(split_dict['test'])
_breast_test = [d for d in _test_all if is_breast(d)]
_cape_test = [d for d in _breast_test if takes_capecitabine(d)]
print(f"\n[COHORT] test: all={len(_test_all)}, breast={len(_breast_test)}, "
      f"breast+capecitabine={len(_cape_test)}")
if cohort_version == 'all':
    split_dict['test'] = _test_all
elif cohort_version == 'capecitabine':
    split_dict['test'] = _cape_test
else:
    split_dict['test'] = _breast_test   # 'all_breast'; 'both' falls through here
print(f"[COHORT] using '{cohort_version}' -> test size {len(split_dict['test'])}")
if len(split_dict['test']) == 0:
    print("\n[COHORT][FATAL] selected cohort is EMPTY -> nothing to evaluate.")
    if cohort_version == 'capecitabine':
        print("  Capecitabine is an ORAL drug, but the meds source is "
              "DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT (non-oral only), so capecitabine "
              "patients cannot be identified here. Use --cohort_version all or all_breast.")
    import sys as _sys
    _sys.exit(1)

model_set_option = 'side_effects_risk'
option = f'{option}_risk'   # anemia_risk
all_folds = list(range(10))

# SPARC-P inference, all test rows
pl.seed_everything(rseed)
anc_mode = True if 'CAIA' in dataset else False

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)
print("data module initiated (G-CSF-free, test_shuffle=False)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sparc_fold_store = {}

print("\n=== Running SPARC-P inference for FIRST CYCLE only ===")
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

    row_label = ((all_events == 1) & (all_ttes == 1)).astype(int)
    sparc_fold_store[val_fold] = {
        'preds': all_preds, 'label': row_label, 'dmp': all_dmp, 'cycle': all_infus,
    }
    print(f"fold {val_fold}: {len(all_preds)} rows, {int(row_label.sum())} next-inf anemia positives")


# Ensemble: row-wise mean across folds (same test order in every fold)
print("\nBuilding SPARC-P ensemble (positional row-wise mean across folds) ...")
fold_lens = {f: len(sparc_fold_store[f]['preds']) for f in all_folds}
if len(set(fold_lens.values())) != 1:
    raise RuntimeError(f"Fold row counts differ {fold_lens}; positional stacking invalid.")

ref = sparc_fold_store[0]
for f in all_folds[1:]:
    if not np.array_equal(sparc_fold_store[f]['label'], ref['label']):
        n_mismatch = int((sparc_fold_store[f]['label'] != ref['label']).sum())
        print(f"[ENSEMBLE][WARN] fold {f} labels differ from fold 0 in {n_mismatch} rows.")

fold_preds = np.stack([sparc_fold_store[f]['preds'] for f in all_folds], axis=0)
ens_pred_all = fold_preds.mean(axis=0)
ens_label_all = ref['label'].astype(int)
ens_dmp_all = ref['dmp']
ens_cycle_all = ref['cycle'].astype(int)
print(f"Ensemble rows: {len(ens_pred_all)}, positives: {int(ens_label_all.sum())}")


# Dranitsaris score per patient, baseline features
print("\nComputing Dranitsaris severe-anemia scores (patient baseline features) ...")
dran_by_patient = {}
for dmp_id in np.unique(ens_dmp_all):
    feats = build_dranitsaris_features(dmp_id, cutoff_date=None)
    lp, prob = dranitsaris_score_and_prob(feats, include_steroid=DRAN_INCLUDE_STEROID)
    dran_by_patient[dmp_id] = {'lp': lp, 'prob': prob, 'feats': feats}

_n = len(dran_by_patient)
def _avail(key):
    return sum(1 for v in dran_by_patient.values() if v['feats'].get(key) is not None)
print(f"Dranitsaris scored {_n} patients. coverage: "
      f"age={_avail('age')}, hb={_avail('hb')}, mcv={_avail('mcv')}, egfr={_avail('egfr')}; "
      f"steroid=yes:{sum(1 for v in dran_by_patient.values() if v['feats']['steroid'])}, "
      f"female:{sum(1 for v in dran_by_patient.values() if v['feats']['sex']=='female')}")

dran_prob_rows = np.array([dran_by_patient.get(d, {}).get('prob', np.nan) for d in ens_dmp_all])

first_idx = int(ens_cycle_all.min())
print(f"\nFirst cycle index: {first_idx}")

# Retrained Dranitsaris: one LR on train+val
print("\n=== Training DRAN_retrained logistic regression model (train+val combined) ===")

_, _, split_dict_trainval, _, _ = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option='anemia', modules=modules,
    valfold=0, mode='train', rseed=rseed,
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)

train_val_ids = list(split_dict_trainval['train']) + list(split_dict_trainval['val'])
# Same cohort restriction as the test set
_before = len(train_val_ids)
if cohort_version != 'all':
    train_val_ids = [d for d in train_val_ids if is_breast(d)]
    if cohort_version == 'capecitabine':
        train_val_ids = [d for d in train_val_ids if takes_capecitabine(d)]
print(f"Combined train+val: {_before} -> {len(train_val_ids)} after "
      f"'{cohort_version}' cohort restriction")

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

# One first-cycle row per train+val patient
train_features_list = []
train_labels = []
train_seen_patients = set()

print("Extracting first-cycle features from train+val patients...")
for batch in data_module_combined.train_dataloader():
    # batch = ((xs, times, num_infusion, DMP_ID), (fail, tte))
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except (ValueError, TypeError):
        print(f"  [WARN] Unexpected batch structure, skipping batch")
        continue

    if isinstance(y, (list, tuple)) and len(y) >= 2:
        fail = y[0]
        tte = y[1]
    else:
        print(f"  [WARN] Unexpected y structure, skipping batch")
        continue

    if isinstance(fail, torch.Tensor):
        fail = fail.cpu().numpy()
    else:
        fail = np.array(fail)

    if isinstance(tte, torch.Tensor):
        tte = tte.cpu().numpy()
    else:
        tte = np.array(tte)

    if isinstance(num_inf, torch.Tensor):
        num_inf = num_inf.cpu().numpy()
    else:
        num_inf = np.array(num_inf)

    if isinstance(dmp_ids, torch.Tensor):
        dmp_ids = dmp_ids.cpu().tolist()
    elif not isinstance(dmp_ids, (list, tuple)):
        dmp_ids = list(dmp_ids)

    for i in range(len(dmp_ids)):
        dmp_id = dmp_ids[i]
        if int(round(float(num_inf[i]))) == first_idx and dmp_id not in train_seen_patients:
            train_seen_patients.add(dmp_id)
            feats = build_dranitsaris_features(dmp_id, cutoff_date=None)
            train_features_list.append(feats)
            # next-infusion anemia label
            label = 1 if (fail[i] == 1 and tte[i] == 1) else 0
            train_labels.append(label)

X_train, feature_names = build_feature_matrix(train_features_list, include_steroid=DRAN_INCLUDE_STEROID)
y_train = np.array(train_labels)

print(f"Training samples: {len(y_train)} unique patients, {int(y_train.sum())} positives ({y_train.sum()/len(y_train):.2%})")

print("Fitting logistic regression model...")
lr_model = LogisticRegression(random_state=rseed, max_iter=1000, class_weight='balanced', solver='lbfgs')
lr_model.fit(X_train, y_train)

print(f"\nTrained model coefficients:")
for fname, coef in zip(feature_names, lr_model.coef_[0]):
    print(f"  {fname:15s}: {coef:7.4f}")
print(f"  {'intercept':15s}: {lr_model.intercept_[0]:7.4f}")

print("\nComparing to original Dranitsaris coefficients:")
print(f"  {'age_gt65':15s}: {np.log(1.47):7.4f} (original)")
print(f"  {'hb_low':15s}: {np.log(3.14):7.4f} (original)")
if DRAN_INCLUDE_STEROID:
    print(f"  {'steroid':15s}: {np.log(1.83):7.4f} (original)")
print(f"  {'mcv_lt81':15s}: {np.log(1.06):7.4f} (original)")
print(f"  {'mcv_gt99':15s}: {np.log(3.20):7.4f} (original)")
print(f"  {'gfr_lt60':15s}: {np.log(2.01):7.4f} (original)")

print("\nComputing DRAN_retrained predictions on test set...")
features_list = [dran_by_patient[d]['feats'] for d in np.unique(ens_dmp_all)]
X_test, _ = build_feature_matrix(features_list, include_steroid=DRAN_INCLUDE_STEROID)

test_probs = lr_model.predict_proba(X_test)[:, 1]

dmp_to_prob = dict(zip(np.unique(ens_dmp_all), test_probs))
dran_retrained_probs = np.array([dmp_to_prob[d] for d in ens_dmp_all])
print(f"DRAN_retrained predictions computed for {len(np.unique(ens_dmp_all))} test patients")


# Metrics helpers
def _bootstrap_ci(y, p, metric='auroc', n_boot=1000, seed=1024):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, pb) if metric == 'auroc'
                    else average_precision_score(yb, pb))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def valid_infusion_mask(labels, cycles):
    keep = np.zeros(len(labels), dtype=bool)
    for c in np.unique(cycles):
        m = cycles == c
        s = labels[m].sum()
        if s > 0 and s < m.sum():
            keep |= m
    return keep


def _overall_auroc_auprc(preds, labels, cycles):
    keep = valid_infusion_mask(labels, cycles)
    y, p = labels[keep], preds[keep]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, 0, 0
    return roc_auc_score(y, p), average_precision_score(y, p), int(keep.sum()), int(y.sum())


def _weighted_auroc_auprc(preds, labels, cycles):
    per = []
    for c in np.unique(cycles):
        m = cycles == c
        y, p = labels[m], preds[m]
        s = y.sum()
        if 0 < s < m.sum():
            per.append({'auroc': roc_auc_score(y, p),
                        'auprc': average_precision_score(y, p), 'n': int(m.sum())})
    if not per:
        return np.nan, np.nan, 0, 0
    tot = sum(r['n'] for r in per)
    return (sum(r['auroc'] * r['n'] / tot for r in per),
            sum(r['auprc'] * r['n'] / tot for r in per), len(per), tot)


rows = []

# Diagnostic: all-cycles SPARC-P row copied from the existing CSV
if run_diagnostics and HAVE_EXISTING_CSV:
    print("\n############### SPARC-P ALL-CYCLES FROM CSV ###############")
    print(f"Using existing CSV values:")
    print(f"  n_test={n_test_existing}, n_pos_next_infusion={n_pos_next_infusion_existing}")
    print(f"  AUROC={auroc_existing:.4f} [{auroc_ci_lower_existing:.4f}, {auroc_ci_upper_existing:.4f}]")
    print(f"  AUPRC={auprc_existing:.4f} [{auprc_ci_lower_existing:.4f}, {auprc_ci_upper_existing:.4f}]")

    rows.append({'comparison': 'SPARC-P_overall_validinfusion_diagnostic', 'model': 'SPARC-P_ensemble',
                 'n': n_test_existing, 'n_pos': n_pos_next_infusion_existing,
                 'auroc': auroc_existing, 'auroc_ci_lower': auroc_ci_lower_existing, 'auroc_ci_upper': auroc_ci_upper_existing,
                 'auprc': auprc_existing, 'auprc_ci_lower': auprc_ci_lower_existing, 'auprc_ci_upper': auprc_ci_upper_existing})
    print("############################################################\n")
elif run_diagnostics:
    print("\n[DIAG] all-cycles reference row skipped (no existing CSV). First-cycle + "
          "patient-level SPARC-P below are computed from this script's own inference.\n")


# Comparison harness
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
    print(f"{comparator_name:<16}: AUROC={ce_auroc:.4f} [{ce_a_ci[0]:.4f},{ce_a_ci[1]:.4f}]  "
          f"AUPRC={ce_auprc:.4f} [{ce_p_ci[0]:.4f},{ce_p_ci[1]:.4f}]")
    rows.append({'comparison': tag, 'model': 'SPARC-P_ensemble', 'n': n, 'n_pos': npos,
                 'auroc': sp_auroc, 'auroc_ci_lower': sp_a_ci[0], 'auroc_ci_upper': sp_a_ci[1],
                 'auprc': sp_auprc, 'auprc_ci_lower': sp_p_ci[0], 'auprc_ci_upper': sp_p_ci[1]})
    rows.append({'comparison': tag, 'model': comparator_name, 'n': n, 'n_pos': npos,
                 'auroc': ce_auroc, 'auroc_ci_lower': ce_a_ci[0], 'auroc_ci_upper': ce_a_ci[1],
                 'auprc': ce_auprc, 'auprc_ci_lower': ce_p_ci[0], 'auprc_ci_upper': ce_p_ci[1]})
    # DeLong row: p-value in auroc, z in auroc_ci_lower, AUROC diff in auroc_ci_upper
    rows.append({'comparison': f'{tag}_significance_test',
                 'model': f'SPARC-P_vs_{comparator_name}',
                 'n': n, 'n_pos': npos,
                 'auroc': delong_pvalue,
                 'auroc_ci_lower': delong_z,
                 'auroc_ci_upper': sp_auroc - ce_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})


_dran_label = 'Dranitsaris' if DRAN_INCLUDE_STEROID else 'Dranitsaris_4var'
_dran_retrained_label = 'Dranitsaris_retrained' if DRAN_INCLUDE_STEROID else 'Dranitsaris_retrained_4var'

# (a) first cycle only
first_idx = int(ens_cycle_all.min())
fc_mask = (ens_cycle_all == first_idx)
print(f"\n\n=== FIRST CYCLE COMPARISON (cycle=={first_idx}) ===")
_run_comparison(f'DRAN_firstcycle (cycle=={first_idx})', fc_mask, dran_prob_rows, _dran_label)
_run_comparison(f'DRAN_retrained_firstcycle (cycle=={first_idx})', fc_mask, dran_retrained_probs, _dran_retrained_label)

# (b) Patient-level ever-positive (primary); SPARC-P aggregated by max risk.
patient_ids = np.unique(ens_dmp_all)
pat_label, pat_sparc_max, pat_sparc_mean, pat_dran, pat_dran_retrained = [], [], [], [], []
for d in patient_ids:
    m = ens_dmp_all == d
    pat_label.append(int(ens_label_all[m].max()))
    pat_sparc_max.append(float(ens_pred_all[m].max()))
    pat_sparc_mean.append(float(ens_pred_all[m].mean()))
    pat_dran.append(dran_by_patient.get(d, {}).get('prob', np.nan))
    pat_dran_retrained.append(dran_retrained_probs[np.where(ens_dmp_all == d)[0][0]])

pat_label = np.array(pat_label); pat_sparc_max = np.array(pat_sparc_max)
pat_sparc_mean = np.array(pat_sparc_mean); pat_dran = np.array(pat_dran)
pat_dran_retrained = np.array(pat_dran_retrained)

_valid = ~np.isnan(pat_dran)
_y = pat_label[_valid]; _sp = pat_sparc_max[_valid]; _dr = pat_dran[_valid]; _dr_ret = pat_dran_retrained[_valid]
n_pat, n_pos_pat = len(_y), int(_y.sum())
print(f"\n================= DRAN_patient_level_ever_positive =================")
print(f"patients = {n_pat}, ever-positive = {n_pos_pat} ({(n_pos_pat/n_pat if n_pat else 0):.2%})")
print(f"(SPARC-P aggregated by MAX risk across each patient's cycles)")
if n_pat > 0 and len(np.unique(_y)) >= 2:
    sp_auroc = roc_auc_score(_y, _sp); sp_auprc = average_precision_score(_y, _sp)
    sp_a_ci = _bootstrap_ci(_y, _sp, 'auroc'); sp_p_ci = _bootstrap_ci(_y, _sp, 'auprc')
    dr_auroc = roc_auc_score(_y, _dr); dr_auprc = average_precision_score(_y, _dr)
    dr_a_ci = _bootstrap_ci(_y, _dr, 'auroc'); dr_p_ci = _bootstrap_ci(_y, _dr, 'auprc')
    dr_ret_auroc = roc_auc_score(_y, _dr_ret); dr_ret_auprc = average_precision_score(_y, _dr_ret)
    dr_ret_a_ci = _bootstrap_ci(_y, _dr_ret, 'auroc'); dr_ret_p_ci = _bootstrap_ci(_y, _dr_ret, 'auprc')
    spm_auroc = roc_auc_score(_y, pat_sparc_mean[_valid])

    try:
        delong_orig = delong_roc_test(_y, _sp, _dr)
        delong_ret = delong_roc_test(_y, _sp, _dr_ret)
        print(f"DeLong test (SPARC vs {_dran_label}): z={delong_orig['z_score']:.4f}, p={delong_orig['pvalue']:.4e}")
        print(f"DeLong test (SPARC vs {_dran_retrained_label}): z={delong_ret['z_score']:.4f}, p={delong_ret['pvalue']:.4e}")
    except Exception as e:
        print(f"[WARN] DeLong test failed: {e}")
        delong_orig = {'pvalue': np.nan, 'z_score': np.nan}
        delong_ret = {'pvalue': np.nan, 'z_score': np.nan}

    print(f"SPARC-P (max) : AUROC={sp_auroc:.4f} [{sp_a_ci[0]:.4f},{sp_a_ci[1]:.4f}]  "
          f"AUPRC={sp_auprc:.4f} [{sp_p_ci[0]:.4f},{sp_p_ci[1]:.4f}]")
    print(f"SPARC-P (mean): AUROC={spm_auroc:.4f}   (transparency only)")
    print(f"{_dran_label:<20}: AUROC={dr_auroc:.4f} [{dr_a_ci[0]:.4f},{dr_a_ci[1]:.4f}]  "
          f"AUPRC={dr_auprc:.4f} [{dr_p_ci[0]:.4f},{dr_p_ci[1]:.4f}]")
    print(f"{_dran_retrained_label:<20}: AUROC={dr_ret_auroc:.4f} [{dr_ret_a_ci[0]:.4f},{dr_ret_a_ci[1]:.4f}]  "
          f"AUPRC={dr_ret_auprc:.4f} [{dr_ret_p_ci[0]:.4f},{dr_ret_p_ci[1]:.4f}]")
    rows.append({'comparison': 'DRAN_patient_level_ever_positive', 'model': 'SPARC-P_ensemble_maxrisk',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': sp_auroc, 'auroc_ci_lower': sp_a_ci[0], 'auroc_ci_upper': sp_a_ci[1],
                 'auprc': sp_auprc, 'auprc_ci_lower': sp_p_ci[0], 'auprc_ci_upper': sp_p_ci[1]})
    rows.append({'comparison': 'DRAN_patient_level_ever_positive', 'model': 'SPARC-P_ensemble_meanrisk',
                 'n': n_pat, 'n_pos': n_pos_pat, 'auroc': spm_auroc, 'auprc': None})
    rows.append({'comparison': 'DRAN_patient_level_ever_positive', 'model': _dran_label,
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': dr_auroc, 'auroc_ci_lower': dr_a_ci[0], 'auroc_ci_upper': dr_a_ci[1],
                 'auprc': dr_auprc, 'auprc_ci_lower': dr_p_ci[0], 'auprc_ci_upper': dr_p_ci[1]})
    rows.append({'comparison': 'DRAN_patient_level_ever_positive', 'model': _dran_retrained_label,
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': dr_ret_auroc, 'auroc_ci_lower': dr_ret_a_ci[0], 'auroc_ci_upper': dr_ret_a_ci[1],
                 'auprc': dr_ret_auprc, 'auprc_ci_lower': dr_ret_p_ci[0], 'auprc_ci_upper': dr_ret_p_ci[1]})
    rows.append({'comparison': 'DRAN_patient_level_ever_positive_significance_test',
                 'model': f'SPARC-P_vs_{_dran_label}',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': delong_orig['pvalue'],
                 'auroc_ci_lower': delong_orig['z_score'],
                 'auroc_ci_upper': sp_auroc - dr_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})
    rows.append({'comparison': 'DRAN_patient_level_ever_positive_significance_test',
                 'model': f'SPARC-P_vs_{_dran_retrained_label}',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': delong_ret['pvalue'],
                 'auroc_ci_lower': delong_ret['z_score'],
                 'auroc_ci_upper': sp_auroc - dr_ret_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})
else:
    print("[WARN] single-class or empty at patient level; cannot compute AUROC.")

# Per-fold first-cycle counts
for f in all_folds:
    s = sparc_fold_store[f]
    fc = (s['cycle'] == first_idx)
    rows.append({'comparison': 'per_fold_firstcycle', 'model': f'SPARC-P_fold{f}_firstcycle',
                 'n': int(fc.sum()), 'n_pos': int(s['label'][fc].sum()),
                 'auroc': None, 'auprc': None})

rows.append({'comparison': 'NOTE', 'model':
             f'outcome=OUR_anemia_grade3_Hb<8 (MATCHES Dranitsaris endpoint); '
             f'dran_vars={DRAN_N_VARS}' + ('' if DRAN_INCLUDE_STEROID else ' (no steroid src)') +
             f'; DROPPED TNM stage + chemo regimen (breast-specific); '
             f'intercept omitted in original (AUROC-invariant); '
             f'PATIENT-LEVEL ever-positive = primary comparison (SPARC-P max-risk agg); '
             f'external validation (breast model -> pan-cancer cohort); '
             f'DRAN_retrained = logistic regression fit on train+val set (single model)',
             'n': None, 'n_pos': None, 'auroc': None, 'auprc': None})

# Save
_vartag = f'{DRAN_N_VARS}var'
# Cohort in filename so versions don't overwrite; 'both' writes the allbreast file.
_cohort_tag = {'all': 'allcancer', 'capecitabine': 'capecitabine'}.get(cohort_version, 'allbreast')
save_name = (f'{results_path}/SPARC-P_vs_DRANITSARIS_anemia_FIXED_'
             f'{modules}_{dataset}_{caia_portion}_{_vartag}_{_cohort_tag}.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(save_name, index=False)
print(f"\nSaved comparison to: {save_name}")
print(df.to_string(index=False))

# Per-patient audit
audit_rows = []
for d in np.unique(ens_dmp_all):
    info = dran_by_patient.get(d, {})
    f = info.get('feats', {})
    idx = np.where(ens_dmp_all == d)[0][0]
    retrained_prob = dran_retrained_probs[idx]
    audit_rows.append({
        'dmp_id': d, 'dran_lp': info.get('lp'), 'dran_prob': info.get('prob'),
        'dran_retrained_prob': retrained_prob,
        'age': f.get('age'), 'sex': f.get('sex'), 'hb_gdl': f.get('hb'),
        'mcv_fl': f.get('mcv'), 'creat': f.get('_creat'), 'egfr': f.get('egfr'),
        'steroid': f.get('steroid'),
    })
audit_name = save_name.replace('.csv', '_DRAN_patient_audit.csv')
pd.DataFrame(audit_rows).to_csv(audit_name, index=False)
print(f"Saved Dranitsaris per-patient audit to: {audit_name}")