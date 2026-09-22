"""SPARC-P vs the Lyman neutropenic-complications risk model.

SPARC-P n / n_pos / metrics come from the existing across-models CSV
(--existing_csv) to avoid re-running inference; first-cycle and patient-level
results are regenerated here.

Two comparators:
  * Lyman_retrained (PRIMARY, report this one): logistic regression on Lyman's
    available feature set, refit on our train+val. Needs no published
    coefficients, so it is a fair test of whether those features carry signal
    in our cohort.
  * Lyman_published (SECONDARY): the published-style additive score, but
    LYMAN_BETA currently holds PLACEHOLDER coefficients -- Cancer 2011 Table 3
    is not yet encoded, so its ranking is not faithful. Do not report it until
    the real weights are inserted.

Reference: Lyman GH, Kuderer NM, Crawford J, et al. Predicting individual risk
of neutropenic complications in patients receiving cancer chemotherapy. Cancer.
2011;117(9):1917-1927. doi:10.1002/cncr.25691

Features: age, WBC, AST, ALP, bilirubin and CKD-EPI GFR are active. Prior
chemotherapy, immunosuppressive drug use, chemotherapy agent type and cancer
type are placeholder-inert -- their accessors return 0 until wired. Planned RDI
is dropped, since it needs actual and standard dose and is not reconstructable.

Lyman's endpoint is a cycle-1 composite of severe neutropenia (ANC<500) or
febrile neutropenia (ANC<1000 with fever). We score against our neutropenia
next-infusion label (event==1 & tte==1), so this is an endpoint-definition
deviation -- confirm the grade alignment. Being a first-cycle model, the
primary comparison is patient-level ever-positive plus first-cycle. Lyman's
pan-cancer prospective cohort (breast/lung/CRC/ovary/lymphoma) makes this
external validation with reasonably aligned cohort type.

prob = sigmoid(linear predictor); the intercept is omitted in the published
variant and fit in the retrained one. Either way AUROC is invariant to it.
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

AST_INDEX   = ALL_INCLUDED_TESTS.index('AST')
ALP_INDEX   = ALL_INCLUDED_TESTS.index('ALP')
TBIL_INDEX  = ALL_INCLUDED_TESTS.index('Bilirubin, Total')
WBC_INDEX   = ALL_INCLUDED_TESTS.index('WBC')
CREAT_INDEX = ALL_INCLUDED_TESTS.index('Creatinine')

AGE_BIN_YEARS = 5.0

# Import DeLong test for significance testing
from delong_test import delong_roc_test

# torchmetrics for AUROC/AUPRC (parity with reference eval)
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


# ============================  LYMAN NEUTROPENIA  ==========================
# The PUBLISHED Lyman model's exact point weights (Cancer 2011 Table 3) are not
# encoded -- LYMAN_BETA below is PLACEHOLDER (used only for the 'Lyman_published'
# comparator, whose ranking is therefore NOT faithful). The PRIMARY Lyman
# comparator here is 'Lyman_retrained': a logistic regression using Lyman's
# FEATURE SET but refit on our own train+val data, which needs no published
# coefficients from the actual model. This is now the FAITHFUL published Lyman
# comparator (Cancer 2011 Table 3), with the reported intercept -> calibrated.
# beta = ln(OR).
LYMAN_INTERCEPT = -3.423
LYMAN_BETA = {
    # binary indicators
    'age_ge65':    np.log(1.297),   # +0.260  (age >= 65)
    'prior_chemo': np.log(1.925),   # +0.655
    'ast_gt35':    np.log(1.422),   # +0.352  (AST > 35 U/L)
    'alp_gt120':   np.log(1.469),   # +0.385  (ALP > 120 U/L)
    'tbil_gt1':    np.log(2.152),   # +0.766  (bilirubin > 1 mg/dL)
    'immunosupp':  np.log(1.554),   # +0.441
    'planned_rdi': np.log(2.018),   # +0.702  (planned RDI >=85%) -- UNAVAILABLE (dropped)
    # continuous (per unit)
    'gfr':         np.log(0.993),   # -0.00702 per mL/min
    'wbc':         np.log(0.930),   # -0.07257 per 10^3/mm^3
    # cancer type (vs colorectum reference)
    'cancer_sclc':     np.log(1.556),   # +0.442
    'cancer_nsclc':    np.log(0.594),   # -0.521
    'cancer_ovary':    np.log(0.515),   # -0.664
    'cancer_breast':   np.log(0.842),   # -0.172
    'cancer_lymphoma': np.log(0.510),   # -0.673
    # chemotherapy classes (each a separate indicator; union over cycle-1 drugs)
    'chemo_anthracyclines':   np.log(7.353),    # +1.995
    'chemo_platinum':         np.log(1.830),    # +0.604
    'chemo_taxanes':          np.log(2.850),    # +1.047
    'chemo_alkylating':       np.log(5.853),    # +1.767
    'chemo_topoisomerase_II': np.log(8.815),    # +2.177
    'chemo_gemcitabine':      np.log(3.092),    # +1.129
    'chemo_topoisomerase_I':  np.log(18.579),   # +2.922
    'chemo_vinorelbine':      np.log(4.218),    # +1.439
    # CSF prophylaxis (strong protective)
    'csf_prophylaxis':        np.log(0.120),    # -2.120
}

# Which optional terms are actually available in this dataset (set at load time):
LYMAN_HAS_RDI = False   # planned RDI unavailable (no dose data) -> term dropped
LYMAN_HAS_CSF = False   # set True if a G-CSF-prophylaxis source loads


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_egfr_ckd_epi(creat_mgdl, age, is_female):
    """CKD-EPI 2009 eGFR (mL/min/1.73m^2); identical to the FENCE accessor."""
    if creat_mgdl is None or age is None or creat_mgdl <= 0:
        return None
    if is_female:
        k, alpha, sexf = 0.7, -0.329, 1.018
    else:
        k, alpha, sexf = 0.9, -0.411, 1.0
    ratio = creat_mgdl / k
    return 141 * (min(ratio, 1) ** alpha) * (max(ratio, 1) ** -1.209) * (0.993 ** age) * sexf


def lyman_features_to_lp(feats):
    """Faithful published-Lyman linear predictor (Table 3 betas + intercept).
    Continuous GFR/WBC enter as raw value * beta. Binary thresholds AST>35,
    ALP>120, bili>1. Cancer type vs colorectum. Chemo classes each an indicator.
    Missing continuous -> that term contributes 0. planned RDI dropped (no data).
    CSF prophylaxis included only if available."""
    B = LYMAN_BETA
    lp = LYMAN_INTERCEPT
    if feats['age'] is not None and feats['age'] >= 65:
        lp += B['age_ge65']
    if feats.get('prior_chemo'):
        lp += B['prior_chemo']
    if feats['ast'] is not None and feats['ast'] > 35:
        lp += B['ast_gt35']
    if feats['alp'] is not None and feats['alp'] > 120:
        lp += B['alp_gt120']
    if feats['tbil'] is not None and feats['tbil'] > 1.0:
        lp += B['tbil_gt1']
    if feats.get('immunosupp'):
        lp += B['immunosupp']
    if LYMAN_HAS_RDI and feats.get('planned_rdi'):
        lp += B['planned_rdi']
    if feats['egfr'] is not None:
        lp += B['gfr'] * feats['egfr']
    if feats['wbc'] is not None:
        lp += B['wbc'] * feats['wbc']
    # cancer type (colorectum = reference, contributes 0)
    ct = feats.get('cancer_lyman')
    if ct in ('sclc', 'nsclc', 'ovary', 'breast', 'lymphoma'):
        lp += B[f'cancer_{ct}']
    # chemo classes (union over cycle-1 drugs)
    for cls in (feats.get('chemo_classes') or set()):
        key = f'chemo_{cls}'
        if key in B:
            lp += B[key]
    # CSF prophylaxis
    if LYMAN_HAS_CSF and feats.get('csf_prophylaxis'):
        lp += B['csf_prophylaxis']
    return lp


def lyman_score_and_prob(feats):
    lp = lyman_features_to_lp(feats)
    return lp, _sigmoid(lp)


# Wired accessors. prior_chemo and immunosuppressive use are populated from real
# sources (globals MED_DICT / CHEMO_NAMES / steroid_dict, loaded below before any
# call). index_day = the patient's first treatment day (min med day); "prior" and
# the steroid window are measured relative to it. chemo_type / cancer_type remain
# inert placeholders.
def _patient_index_day(dmp_id):
    """First (earliest) medication-administration day for the patient = index day
    of the treatment course. None if no meds."""
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return None
    days = [d for d in sched.keys() if _safe_int(d) is not None]
    return min(int(d) for d in days) if days else None


def get_prior_chemo(dmp_id, cutoff_date=None):
    """True if the patient received any chemotherapy agent (per LLM classification)
    on a day STRICTLY BEFORE the index day. Approximates Lyman's 'prior
    chemotherapy' within the available med record (undercounts chemo predating the
    record; documented)."""
    if not LYMAN_WIRE_PRIOR_CHEMO:
        return False
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return False
    idx = _patient_index_day(dmp_id)
    if idx is None:
        return False
    for day, entries in sched.items():
        di = _safe_int(day)
        if di is None or di >= idx:      # strictly before index
            continue
        for e in entries:
            try:
                nm = str(e[0]).strip().upper()   # meds dict entry: [name, ...]
            except (IndexError, TypeError):
                nm = str(e).strip().upper()
            if nm in CHEMO_NAMES:
                return True
    return False


def get_immunosuppressive_use(dmp_id, cutoff_date=None):
    """True if a steroid was administered within Lyman's window before the index
    day: day in [index - steroid_window_hi, index - steroid_window_lo]. This
    excludes single-dose peri-chemo antiemetic premedication (which would make the
    flag near-constant). Set --steroid_window_hi huge to count ANY prior steroid."""
    if not LYMAN_WIRE_IMMUNOSUPP:
        return False
    rec = steroid_dict.get(dmp_id)
    if rec is None:
        rec = steroid_dict.get(str(dmp_id)) or steroid_dict.get(_safe_int(dmp_id))
    if not isinstance(rec, dict) or not rec:
        return False
    idx = _patient_index_day(dmp_id)
    if idx is None:
        # no med record -> fall back to any steroid before day 0
        return any(_safe_int(d) is not None and int(d) < 0 for d in rec.keys())
    lo = idx - steroid_window_hi     # earlier bound
    hi = idx - steroid_window_lo     # later bound (closer to index)
    for day in rec.keys():
        di = _safe_int(day)
        if di is not None and lo <= di <= hi:
            return True
    return False


def get_chemo_type_highrisk(dmp_id, cutoff_date=None):
    return False   # still inert placeholder


def get_cancer_type_highrisk(dmp_id, cutoff_date=None):
    return False   # still inert placeholder


def build_feature_matrix(features_list, include_placeholders=True):
    """Feature matrix for the retrained logistic regression, using Lyman's full
    available feature set: continuous age/wbc/ast/alp/tbil/egfr, Lyman-threshold
    abnormality flags (AST>35, ALP>120, bili>1), prior_chemo, immunosupp, cancer
    buckets, chemo-class indicators, and CSF prophylaxis. RDI dropped (no data).
    Returns X, feature_names."""
    _CANCER = ['sclc', 'nsclc', 'ovary', 'breast', 'lymphoma']  # colorectum=ref
    _CHEMO = _LYMAN_CLS_KEYS
    X = []
    for feats in features_list:
        row = []
        row.append(feats['age'] if feats['age'] is not None else 0.0)
        row.append(feats['wbc'] if feats['wbc'] is not None else 0.0)
        row.append(feats['ast'] if feats['ast'] is not None else 0.0)
        row.append(feats['alp'] if feats['alp'] is not None else 0.0)
        row.append(feats['tbil'] if feats['tbil'] is not None else 0.0)
        row.append(feats['egfr'] if feats['egfr'] is not None else 0.0)
        # Lyman-threshold abnormality flags
        row.append(1.0 if (feats['age'] is not None and feats['age'] >= 65) else 0.0)
        row.append(1.0 if (feats['ast'] is not None and feats['ast'] > 35) else 0.0)
        row.append(1.0 if (feats['alp'] is not None and feats['alp'] > 120) else 0.0)
        row.append(1.0 if (feats['tbil'] is not None and feats['tbil'] > 1.0) else 0.0)
        if include_placeholders:
            row.append(1.0 if feats.get('prior_chemo') else 0.0)
            row.append(1.0 if feats.get('immunosupp') else 0.0)
            # cancer buckets (colorectum ref dropped)
            ct = feats.get('cancer_lyman')
            for c in _CANCER:
                row.append(1.0 if ct == c else 0.0)
            # chemo class indicators
            cc = feats.get('chemo_classes') or set()
            for c in _CHEMO:
                row.append(1.0 if c in cc else 0.0)
            # CSF prophylaxis
            row.append(1.0 if feats.get('csf_prophylaxis') else 0.0)
        X.append(row)
    feature_names = ['age', 'wbc', 'ast', 'alp', 'tbil', 'egfr',
                     'age_ge65', 'ast_gt35', 'alp_gt120', 'tbil_gt1']
    if include_placeholders:
        feature_names += ['prior_chemo', 'immunosupp']
        feature_names += [f'cancer_{c}' for c in _CANCER]
        feature_names += [f'chemo_{c}' for c in _CHEMO]
        feature_names += ['csf_prophylaxis']
    return np.array(X), feature_names


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

parser = argparse.ArgumentParser(prog='eval_SPARC_P_vs_LYMAN_neutropenia_retrained')
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
parser.add_argument("-mp", "--model_path", type=str, default=f'{SPARC_MODEL_ROOT}')
parser.add_argument("-rp", "--results_path", type=str,
                    default=f'{SPARC_RESULTS_ROOT}/eval_clinical_benchmarks')
parser.add_argument("--existing_csv", type=str,
                    default=f'{SPARC_REF_ROOT}/'
                            'eval_across_models_valthresh_disabled/model_performance_dict_across_models_parsimonious_neutropenia_MSK_ensemble.csv')
parser.add_argument("--diagnostics", type=str, default='True')
parser.add_argument("--lyman_chemo_class_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/chemo_lyman_classification_llm.pickle',
                    help="Lyman per-drug chemo-class multilabel dict.")
parser.add_argument("--demographic_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/dePHIed_demographic_df_MSK.csv',
                    help="Demographic CSV with primary_cancer_type_th for Lyman cancer term.")
parser.add_argument("--med_classification_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/med_classification_llm.pickle',
                    help="LLM med classification {MED_NAME(upper): {is_chemotherapy, ...}}.")
parser.add_argument("--steroid_path", type=str,
                    default=f'{SPARC_DATA_ROOT}/MSK/DMP_ID_to_DATE_to_STEROID_dict.pkl',
                    help="{DMP_ID: {day: ['dexamethasone', ...]}}; Lyman immunosuppressive-use source.")
parser.add_argument("--steroid_window_lo", type=int, default=14,
                    help="Lyman immunosuppressive window lower bound (days before index).")
parser.add_argument("--steroid_window_hi", type=int, default=90,
                    help="Lyman immunosuppressive window upper bound (days before index). "
                         "Set --steroid_window_hi very large (e.g. 100000) to count ANY prior steroid.")
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
med_classification_path = args.med_classification_path
lyman_chemo_class_path = args.lyman_chemo_class_path
demographic_path = args.demographic_path
steroid_path = args.steroid_path
steroid_window_lo = args.steroid_window_lo
steroid_window_hi = args.steroid_window_hi

# LOCKED settings
option = 'neutropenia'
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

# Architecture params (neutropenia uses 'hybrid' finetuned checkpoints)
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

# Load existing CSV to get n and n_pos
print(f"\nLoading existing CSV (optional) from: {existing_csv_path}")
HAVE_EXISTING_CSV = os.path.exists(existing_csv_path)
if HAVE_EXISTING_CSV:
    existing_df = pd.read_csv(existing_csv_path)
    HAVE_EXISTING_CSV = len(existing_df[existing_df['model'] == 'SPARC-P_ensemble']) > 0
if not HAVE_EXISTING_CSV:
    print("  [existing CSV not found] -> skipping all-cycles reference row; first-cycle + "
          "patient-level metrics computed from this script's own inference.")
_existing_df_guard = HAVE_EXISTING_CSV
if HAVE_EXISTING_CSV:
    sparc_p_ensemble_row = existing_df[existing_df['model'] == 'SPARC-P_ensemble'].iloc[0]
    n_test_existing = int(sparc_p_ensemble_row['n_test'])
    n_pos_overall_existing = int(sparc_p_ensemble_row['n_pos_overall'])
    n_pos_next_infusion_existing = int(sparc_p_ensemble_row['n_pos_next_infusion'])
    auroc_existing = float(sparc_p_ensemble_row['test_auroc_nextinf_overall'])
    auprc_existing = float(sparc_p_ensemble_row['test_auprc_nextinf_overall'])
    auroc_ci_lower_existing = float(sparc_p_ensemble_row['auroc_ci_lower'])
    auroc_ci_upper_existing = float(sparc_p_ensemble_row['auroc_ci_upper'])
    auprc_ci_lower_existing = float(sparc_p_ensemble_row['auprc_ci_lower'])
    auprc_ci_upper_existing = float(sparc_p_ensemble_row['auprc_ci_upper'])
    print(f"Existing SPARC-P ensemble metrics:")
    print(f"  n_test={n_test_existing}, n_pos_overall={n_pos_overall_existing}, n_pos_next_infusion={n_pos_next_infusion_existing}")
    print(f"  AUROC={auroc_existing:.4f} [{auroc_ci_lower_existing:.4f}, {auroc_ci_upper_existing:.4f}]")
    print(f"  AUPRC={auprc_existing:.4f} [{auprc_ci_lower_existing:.4f}, {auprc_ci_upper_existing:.4f}]")
else:
    n_test_existing = n_pos_overall_existing = n_pos_next_infusion_existing = None
    auroc_existing = auprc_existing = None
    auroc_ci_lower_existing = auroc_ci_upper_existing = None
    auprc_ci_lower_existing = auprc_ci_upper_existing = None

# Load dataset
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

# Lyman term summary
print("\n[LYMAN] active terms: age, WBC, AST, ALP, bilirubin, GFR")
print("[LYMAN] WIRED terms: prior_chemo (med dict + LLM classification), "
      "immunosupp (steroid dict, windowed), chemo classes (LLM), cancer type, CSF prophylaxis. DROPPED: planned RDI (no dose data)")
print("[LYMAN] DROPPED: planned RDI (not reconstructable)")
print("[LYMAN] PRIMARY comparator = Lyman_retrained (features refit on our train+val).")
print("[LYMAN][WARN] 'Lyman_published' uses PLACEHOLDER coefficients -> not faithful "
      "until Cancer 2011 Table 3 weights are inserted.")

def _try_load_pickle(path, label):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  [{label}] loaded ({len(obj)} entries) from {path}")
        return obj
    print(f"  [{label}] NOT FOUND at {path} -> will default")
    return {}

# Sources for the two now-wired categorical predictors:
#   * prior chemo: med dict (loaded by load_dataset) + LLM med classification
#   * immunosuppressive use: dated steroid dict (Lyman's window applied)
print("\n[LYMAN] loading prior-chemo + immunosuppressive + class/cancer/CSF sources ...")
MED_DICT = data_dict_lab_dict.get('DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT', {})
print(f"  [meds] {len(MED_DICT)} patients in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT")

# Lyman per-drug chemo-class multilabel dict: {MED(upper): {anthracyclines:bool,...}}
lyman_chemo_cls = _try_load_pickle(lyman_chemo_class_path, 'lyman_chemo_class')
lyman_chemo_cls = {str(k).strip().upper(): v for k, v in lyman_chemo_cls.items()}
# any Lyman class True (or the immunosuppressive flag) => it's a chemo/relevant drug
_LYMAN_CLS_KEYS = ['anthracyclines', 'platinum', 'taxanes', 'alkylating',
                   'topoisomerase_II', 'gemcitabine', 'topoisomerase_I', 'vinorelbine']
CHEMO_NAMES = {name for name, c in lyman_chemo_cls.items()
               if isinstance(c, dict) and any(c.get(k) for k in _LYMAN_CLS_KEYS)}
print(f"  [lyman_chemo_class] {len(lyman_chemo_cls)} meds, {len(CHEMO_NAMES)} with >=1 chemo class")

steroid_dict = _try_load_pickle(steroid_path, 'steroid')

# Cancer type (demographic csv) -> Lyman buckets
import pandas as _pd
CTYPE_DICT = {}
if os.path.exists(demographic_path):
    _dfd = _pd.read_csv(demographic_path)
    _ct_col = 'primary_cancer_type_th' if 'primary_cancer_type_th' in _dfd.columns else 'primary_cancer_type'
    CTYPE_DICT = dict(zip(_dfd['DMP_ID'], _dfd[_ct_col]))
    print(f"  [cancer_type] {len(CTYPE_DICT)} patients (col {_ct_col})")
else:
    print(f"  [cancer_type] demographic CSV not found at {demographic_path} -> cancer term = reference")

# CSF prophylaxis from gcsf_dict (strong protective term, OR 0.12)
GCSF_DICT = data_dict_lab_dict.get('gcsf_dict', {})
LYMAN_HAS_CSF = len(GCSF_DICT) > 0
print(f"  [csf] gcsf_dict: {len(GCSF_DICT)} patients -> CSF prophylaxis term "
      f"{'ENABLED' if LYMAN_HAS_CSF else 'disabled'}")

LYMAN_WIRE_PRIOR_CHEMO = len(MED_DICT) > 0 and len(CHEMO_NAMES) > 0
LYMAN_WIRE_IMMUNOSUPP  = len(steroid_dict) > 0
print(f"  [LYMAN] prior_chemo wired: {LYMAN_WIRE_PRIOR_CHEMO}; "
      f"immunosupp wired: {LYMAN_WIRE_IMMUNOSUPP} "
      f"(window {steroid_window_lo}-{steroid_window_hi} days pre-index); "
      f"RDI dropped (no dose data)")


# Accessors
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


def _entry_name(e):
    try:
        return str(e[0]).strip().upper()
    except (IndexError, TypeError):
        return str(e).strip().upper()


def get_lyman_chemo_classes(dmp_id):
    """Union of Lyman chemo classes over the cycle-1 (index-day) drugs, with
    dedup rules that respect Lyman's category intent:
      * platinum takes precedence over alkylating (cisplatin/carbo -> platinum only)
      * anthracyclines takes precedence over topoisomerase_II (doxorubicin ->
        anthracyclines only)
    Returns a set of Lyman class names present."""
    sched = MED_DICT.get(dmp_id)
    if not isinstance(sched, dict) or not sched:
        return set()
    idx = _patient_index_day(dmp_id)
    if idx is None:
        return set()
    entries = sched.get(idx, sched.get(str(idx), []))
    classes = set()
    for e in entries:
        nm = _entry_name(e)
        c = lyman_chemo_cls.get(nm)
        if not isinstance(c, dict):
            continue
        present = {k for k in _LYMAN_CLS_KEYS if c.get(k)}
        # dedup rules
        if 'platinum' in present and 'alkylating' in present:
            present.discard('alkylating')
        if 'anthracyclines' in present and 'topoisomerase_II' in present:
            present.discard('topoisomerase_II')
        classes |= present
    return classes


# Lyman cancer-type buckets (reference = colorectum). Map primary_cancer_type_th.
_LYMAN_CANCER_MAP = {
    'Small Cell Lung Cancer': 'sclc',
    'Non-Small Cell Lung Cancer': 'nsclc',
    'Ovarian Cancer': 'ovary',
    'Breast Cancer': 'breast',
    'Mature B-Cell Neoplasms': 'lymphoma',
    'Mature T and NK Neoplasms': 'lymphoma',
    'Colorectal Cancer': 'colorectum',   # reference (contributes 0)
    'Appendiceal Cancer': 'colorectum',
}


def get_lyman_cancer(dmp_id):
    """Map to Lyman's cancer buckets; unknown/other -> 'colorectum' reference (0)."""
    ct = CTYPE_DICT.get(dmp_id)
    if ct is None:
        return 'colorectum'
    return _LYMAN_CANCER_MAP.get(str(ct).strip(), 'colorectum')


def get_csf_prophylaxis(dmp_id, index_day=None):
    """True if prophylactic G-CSF given around cycle 1 (before/at the index day),
    from gcsf_dict. Structure tolerant: {DMP_ID: {day: ...}} or {DMP_ID: [days]}."""
    if not LYMAN_HAS_CSF:
        return False
    rec = _lookup_flexible(GCSF_DICT, dmp_id)
    if not rec:
        return False
    idx = index_day if index_day is not None else _patient_index_day(dmp_id)
    days = list(rec.keys()) if isinstance(rec, dict) else list(rec)
    for d in days:
        di = _safe_int(d)
        if di is None:
            continue
        # prophylaxis ~ given on/around the index cycle (within +/- a few days of idx)
        if idx is None or (idx - 3) <= di <= (idx + 3):
            return True
    return False


def build_lyman_features(dmp_id, cutoff_date=None):
    age = get_age_years(dmp_id)
    sex = get_sex(dmp_id)
    creat = _baseline_lab_value(dmp_id, CREAT_INDEX, cutoff_date)
    egfr = compute_egfr_ckd_epi(creat, age, sex == 'female')
    idx = _patient_index_day(dmp_id)
    return {
        'age':  age,
        'sex':  sex,
        'wbc':  _baseline_lab_value(dmp_id, WBC_INDEX, cutoff_date),
        'ast':  _baseline_lab_value(dmp_id, AST_INDEX, cutoff_date),
        'alp':  _baseline_lab_value(dmp_id, ALP_INDEX, cutoff_date),
        'tbil': _baseline_lab_value(dmp_id, TBIL_INDEX, cutoff_date),
        'egfr': egfr,
        'prior_chemo':   get_prior_chemo(dmp_id, cutoff_date),
        'immunosupp':    get_immunosuppressive_use(dmp_id, cutoff_date),
        'chemo_classes': get_lyman_chemo_classes(dmp_id),     # set of Lyman classes
        'cancer_lyman':  get_lyman_cancer(dmp_id),            # Lyman cancer bucket
        'csf_prophylaxis': get_csf_prophylaxis(dmp_id, idx),
        'planned_rdi':   False,   # unavailable (no dose data)
        '_creat': creat, '_cutoff': cutoff_date,
    }


# Model folder (SPARC-P, neutropenia -> hybrid)
all_model_path = model_path
base_option = 'neutropenia'
model_type_suffix = 'hybrid'   # anemia/neutropenia/FN use 'hybrid' finetuned checkpoints

if dataset == 'MSK':
    sparc_p_folder = (f'{all_model_path}/{modules}_MSK_060726/SPARC_finetuned_models/'
                      f'{modules}_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
elif dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift', 'stanfordEHRSHOT'] or (dataset == 'MIMICIV_hosp' and zeroshot):
    sparc_p_folder = (f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/'
                      f'{modules}_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
else:
    mimiciv_sparc_path = f'{SPARC_MODEL_ROOT}/parsimonious_MIMICIV_hosp_060726'
    sparc_p_folder = (f'{mimiciv_sparc_path}/SPARC_finetuned_models/'
                      f'parsimonious_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular')
print("SPARC-P folder:", sparc_p_folder)

# Build split_dict / test set
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

model_set_option = 'side_effects_risk'
option = f'{option}_risk'   # neutropenia_risk
all_folds = list(range(10))

# SPARC-P inference (FIRST CYCLE ONLY - for regeneration)
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
    print(f"fold {val_fold}: {len(all_preds)} rows, {int(row_label.sum())} next-inf neutropenia positives")


# Ensemble (positional row-wise mean; matches reference)
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


# Lyman score per patient (baseline features)
print("\nComputing Lyman neutropenia scores (patient baseline features) ...")
lyman_by_patient = {}
for dmp_id in np.unique(ens_dmp_all):
    feats = build_lyman_features(dmp_id, cutoff_date=None)
    lp, prob = lyman_score_and_prob(feats)
    lyman_by_patient[dmp_id] = {'lp': lp, 'prob': prob, 'feats': feats}

_n = len(lyman_by_patient)
def _avail(key):
    return sum(1 for v in lyman_by_patient.values() if v['feats'].get(key) is not None)
print(f"Lyman scored {_n} patients. coverage: "
      f"age={_avail('age')}, wbc={_avail('wbc')}, ast={_avail('ast')}, alp={_avail('alp')}, "
      f"tbil={_avail('tbil')}, egfr={_avail('egfr')}")
_n_prior = sum(1 for v in lyman_by_patient.values() if v['feats'].get('prior_chemo'))
_n_immun = sum(1 for v in lyman_by_patient.values() if v['feats'].get('immunosupp'))
print(f"  [WIRED] prior_chemo=yes: {_n_prior}/{_n} ({_n_prior/_n if _n else 0:.1%}); "
      f"immunosupp=yes: {_n_immun}/{_n} ({_n_immun/_n if _n else 0:.1%})")
print("  [WIRED] chemo_classes, cancer_lyman, csf_prophylaxis from real sources; RDI dropped")

lyman_prob_rows = np.array([lyman_by_patient.get(d, {}).get('prob', np.nan) for d in ens_dmp_all])

# Get first cycle index for later use
first_idx = int(ens_cycle_all.min())
print(f"\nFirst cycle index: {first_idx}")

# Train LYMAN_retrained model (single model on combined train+val)
print("\n=== Training LYMAN_retrained logistic regression model (train+val combined) ===")

# Load train+val data (use mode='train' which includes train+val for fold 0)
# Since we're doing external validation on test set, we can use all non-test data
_, _, split_dict_trainval, _, _ = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option='neutropenia', modules=modules,
    valfold=0, mode='train', rseed=rseed,  # fold doesn't matter, we'll combine train+val
    competing_risk_death_check=False, include_gcsf=include_gcsf,
    data_fraction_option='default', quick_test=False, caia_portion=caia_portion,
)

# Get both train and val patient IDs
train_val_ids = list(split_dict_trainval['train']) + list(split_dict_trainval['val'])
print(f"Combined train+val: {len(split_dict_trainval['train'])} train + {len(split_dict_trainval['val'])} val = {len(train_val_ids)} total patients")

# Create combined split_dict
split_dict_combined = {
    'train': train_val_ids,
    'val': [],
    'test': split_dict_trainval['test']
}

# Create data module with combined train+val
data_module_combined = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict_combined, dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode, test_shuffle=False)

# Get first-cycle data for training from combined train+val set
train_features_list = []
train_labels = []
train_seen_patients = set()

print("Extracting first-cycle features from train+val patients...")
# Iterate through train dataloader
for batch in data_module_combined.train_dataloader():
    # Batch structure: ((xs, times, num_infusion, DMP_ID), y)
    # where y = (fail, tte) for risk tasks
    try:
        (xs, times, num_inf, dmp_ids), y = batch
    except (ValueError, TypeError):
        print(f"  [WARN] Unexpected batch structure, skipping batch")
        continue

    # Extract labels
    if isinstance(y, (list, tuple)) and len(y) >= 2:
        fail = y[0]
        tte = y[1]
    else:
        print(f"  [WARN] Unexpected y structure, skipping batch")
        continue

    # Convert to numpy arrays
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

    # For each sample in batch
    for i in range(len(dmp_ids)):
        dmp_id = dmp_ids[i]
        # Only use the FIRST cycle and only once per patient. Use first_idx (the
        # minimum cycle index actually present) rather than a hardcoded value, so
        # this matches the 0-indexed cycle numbering used everywhere else.
        if int(round(float(num_inf[i]))) == first_idx and dmp_id not in train_seen_patients:
            train_seen_patients.add(dmp_id)
            feats = build_lyman_features(dmp_id, cutoff_date=None)
            train_features_list.append(feats)
            # Label: next-infusion neutropenia (event==1 & tte==1)
            label = 1 if (fail[i] == 1 and tte[i] == 1) else 0
            train_labels.append(label)

# Build feature matrix
X_train, feature_names = build_feature_matrix(train_features_list, include_placeholders=True)
y_train = np.array(train_labels)

# DIAGNOSTIC: training set health. A near-zero positive count or a degenerate
# feature matrix (all-zero columns) explains a chance-level retrained AUROC.
print(f"[TRAIN DIAG] first_idx used for training filter = {first_idx}")
print(f"[TRAIN DIAG] training patients = {len(y_train)}, positives = {int(y_train.sum())} "
      f"({(y_train.sum()/len(y_train) if len(y_train) else 0):.2%})")
if len(y_train) > 0:
    col_means = X_train.mean(axis=0)
    col_nonzero = (X_train != 0).mean(axis=0)
    print(f"[TRAIN DIAG] feature | mean | frac_nonzero:")
    for nm, mn, nz in zip(feature_names, col_means, col_nonzero):
        flag = '  <-- DEGENERATE (mostly 0)' if nz < 0.02 else ''
        print(f"[TRAIN DIAG]   {nm:12s} {mn:10.3f} {nz:6.2%}{flag}")
if len(y_train) == 0 or y_train.sum() == 0 or y_train.sum() == len(y_train):
    print("[TRAIN DIAG][WARN] training set has no usable positive/negative split -> "
          "retrained model will be degenerate. Check cycle indexing / cohort.")

print(f"Training samples: {len(y_train)} unique patients, {int(y_train.sum())} positives ({y_train.sum()/len(y_train):.2%})")

# Fit logistic regression
print("Fitting logistic regression model...")
lr_model = LogisticRegression(random_state=rseed, max_iter=1000, class_weight='balanced', solver='lbfgs')
lr_model.fit(X_train, y_train)

print(f"\nTrained model coefficients:")
for fname, coef in zip(feature_names, lr_model.coef_[0]):
    print(f"  {fname:15s}: {coef:7.4f}")
print(f"  {'intercept':15s}: {lr_model.intercept_[0]:7.4f}")

print("\n(Published Lyman coefficients not available -> no original-vs-refit comparison; "
      "the retrained model is the primary Lyman comparator.)")

# Compute predictions for test set
print("\nComputing LYMAN_retrained predictions on test set...")
features_list = [lyman_by_patient[d]['feats'] for d in np.unique(ens_dmp_all)]
X_test, _ = build_feature_matrix(features_list, include_placeholders=True)

# Predict probabilities
test_probs = lr_model.predict_proba(X_test)[:, 1]  # probability of class 1

# Map back to all rows
dmp_to_prob = dict(zip(np.unique(ens_dmp_all), test_probs))
lyman_retrained_probs = np.array([dmp_to_prob[d] for d in ens_dmp_all])
print(f"LYMAN_retrained predictions computed for {len(np.unique(ens_dmp_all))} test patients")


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

# DIAGNOSTIC: SPARC-P across all cycles (overall + weighted) - FROM CSV
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

    # DeLong test for comparing AUROCs
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
    # Add significance test result row
    rows.append({'comparison': f'{tag}_significance_test',
                 'model': f'SPARC-P_vs_{comparator_name}',
                 'n': n, 'n_pos': npos,
                 'auroc': delong_pvalue,  # Store p-value in auroc column for convenience
                 'auroc_ci_lower': delong_z,  # Store z-score in ci_lower column
                 'auroc_ci_upper': sp_auroc - ce_auroc,  # Store AUROC difference
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})


_lyman_label = 'Lyman_published'
_lyman_retrained_label = 'Lyman_retrained'

# (a) first-cycle only (regenerated)
first_idx = int(ens_cycle_all.min())
fc_mask = (ens_cycle_all == first_idx)
print(f"\n\n=== FIRST CYCLE COMPARISON (cycle=={first_idx}) ===")
_run_comparison(f'LYMAN_firstcycle (cycle=={first_idx})', fc_mask, lyman_prob_rows, _lyman_label)
_run_comparison(f'LYMAN_retrained_firstcycle (cycle=={first_idx})', fc_mask, lyman_retrained_probs, _lyman_retrained_label)

# (b) PATIENT-LEVEL EVER-POSITIVE (primary): both models per patient.
#     Lyman is a first-cycle/baseline model; SPARC-P aggregated by MAX risk.
patient_ids = np.unique(ens_dmp_all)
pat_label, pat_sparc_max, pat_sparc_mean, pat_lyman, pat_lyman_retrained = [], [], [], [], []
for d in patient_ids:
    m = ens_dmp_all == d
    pat_label.append(int(ens_label_all[m].max()))
    pat_sparc_max.append(float(ens_pred_all[m].max()))
    pat_sparc_mean.append(float(ens_pred_all[m].mean()))
    pat_lyman.append(lyman_by_patient.get(d, {}).get('prob', np.nan))
    # For retrained, get the prob for this patient from any row
    pat_lyman_retrained.append(lyman_retrained_probs[np.where(ens_dmp_all == d)[0][0]])

pat_label = np.array(pat_label); pat_sparc_max = np.array(pat_sparc_max)
pat_sparc_mean = np.array(pat_sparc_mean); pat_lyman = np.array(pat_lyman)
pat_lyman_retrained = np.array(pat_lyman_retrained)

_valid = ~np.isnan(pat_lyman)
_y = pat_label[_valid]; _sp = pat_sparc_max[_valid]; _dr = pat_lyman[_valid]; _dr_ret = pat_lyman_retrained[_valid]
n_pat, n_pos_pat = len(_y), int(_y.sum())
print(f"\n================= LYMAN_patient_level_ever_positive =================")
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

    # DeLong tests for patient-level comparisons
    try:
        delong_orig = delong_roc_test(_y, _sp, _dr)
        delong_ret = delong_roc_test(_y, _sp, _dr_ret)
        print(f"DeLong test (SPARC vs {_lyman_label}): z={delong_orig['z_score']:.4f}, p={delong_orig['pvalue']:.4e}")
        print(f"DeLong test (SPARC vs {_lyman_retrained_label}): z={delong_ret['z_score']:.4f}, p={delong_ret['pvalue']:.4e}")
    except Exception as e:
        print(f"[WARN] DeLong test failed: {e}")
        delong_orig = {'pvalue': np.nan, 'z_score': np.nan}
        delong_ret = {'pvalue': np.nan, 'z_score': np.nan}

    print(f"SPARC-P (max) : AUROC={sp_auroc:.4f} [{sp_a_ci[0]:.4f},{sp_a_ci[1]:.4f}]  "
          f"AUPRC={sp_auprc:.4f} [{sp_p_ci[0]:.4f},{sp_p_ci[1]:.4f}]")
    print(f"SPARC-P (mean): AUROC={spm_auroc:.4f}   (transparency only)")
    print(f"{_lyman_label:<20}: AUROC={dr_auroc:.4f} [{dr_a_ci[0]:.4f},{dr_a_ci[1]:.4f}]  "
          f"AUPRC={dr_auprc:.4f} [{dr_p_ci[0]:.4f},{dr_p_ci[1]:.4f}]")
    print(f"{_lyman_retrained_label:<20}: AUROC={dr_ret_auroc:.4f} [{dr_ret_a_ci[0]:.4f},{dr_ret_a_ci[1]:.4f}]  "
          f"AUPRC={dr_ret_auprc:.4f} [{dr_ret_p_ci[0]:.4f},{dr_ret_p_ci[1]:.4f}]")
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive', 'model': 'SPARC-P_ensemble_maxrisk',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': sp_auroc, 'auroc_ci_lower': sp_a_ci[0], 'auroc_ci_upper': sp_a_ci[1],
                 'auprc': sp_auprc, 'auprc_ci_lower': sp_p_ci[0], 'auprc_ci_upper': sp_p_ci[1]})
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive', 'model': 'SPARC-P_ensemble_meanrisk',
                 'n': n_pat, 'n_pos': n_pos_pat, 'auroc': spm_auroc, 'auprc': None})
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive', 'model': _lyman_label,
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': dr_auroc, 'auroc_ci_lower': dr_a_ci[0], 'auroc_ci_upper': dr_a_ci[1],
                 'auprc': dr_auprc, 'auprc_ci_lower': dr_p_ci[0], 'auprc_ci_upper': dr_p_ci[1]})
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive', 'model': _lyman_retrained_label,
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': dr_ret_auroc, 'auroc_ci_lower': dr_ret_a_ci[0], 'auroc_ci_upper': dr_ret_a_ci[1],
                 'auprc': dr_ret_auprc, 'auprc_ci_lower': dr_ret_p_ci[0], 'auprc_ci_upper': dr_ret_p_ci[1]})
    # Add significance test results for patient-level
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive_significance_test',
                 'model': f'SPARC-P_vs_{_lyman_label}',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': delong_orig['pvalue'],
                 'auroc_ci_lower': delong_orig['z_score'],
                 'auroc_ci_upper': sp_auroc - dr_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})
    rows.append({'comparison': 'LYMAN_patient_level_ever_positive_significance_test',
                 'model': f'SPARC-P_vs_{_lyman_retrained_label}',
                 'n': n_pat, 'n_pos': n_pos_pat,
                 'auroc': delong_ret['pvalue'],
                 'auroc_ci_lower': delong_ret['z_score'],
                 'auroc_ci_upper': sp_auroc - dr_ret_auroc,
                 'auprc': None, 'auprc_ci_lower': None, 'auprc_ci_upper': None})
else:
    print("[WARN] single-class or empty at patient level; cannot compute AUROC.")

# per-fold transparency
for f in all_folds:
    s = sparc_fold_store[f]
    fc = (s['cycle'] == first_idx)
    rows.append({'comparison': 'per_fold_firstcycle', 'model': f'SPARC-P_fold{f}_firstcycle',
                 'n': int(fc.sum()), 'n_pos': int(s['label'][fc].sum()),
                 'auroc': None, 'auprc': None})

# NOTE row
rows.append({'comparison': 'NOTE', 'model':
             'outcome=OUR_neutropenia_label (Lyman endpoint is SN-or-FN composite, cycle1); '
             'active terms=age,WBC,AST,ALP,bilirubin,GFR; '
             'placeholder-inert=prior_chemo,immunosupp,chemo_type,cancer_type; DROPPED=planned RDI; '
             'Lyman_retrained = logistic regression on Lyman features refit on train+val (PRIMARY); '
             'Lyman_published = PLACEHOLDER coefficients (NOT faithful until Cancer 2011 Table 3 inserted); '
             'PATIENT-LEVEL ever-positive = primary comparison (SPARC-P max-risk agg)',
             'n': None, 'n_pos': None, 'auroc': None, 'auprc': None})

# Save
save_name = (f'{results_path}/SPARC-P_vs_LYMAN_neutropenia_'
             f'{modules}_{dataset}_{caia_portion}_retrained.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(save_name, index=False)
print(f"\nSaved comparison to: {save_name}")
print(df.to_string(index=False))

# Per-patient audit
audit_rows = []
for d in np.unique(ens_dmp_all):
    info = lyman_by_patient.get(d, {})
    f = info.get('feats', {})
    idx = np.where(ens_dmp_all == d)[0][0]
    retrained_prob = lyman_retrained_probs[idx]
    audit_rows.append({
        'dmp_id': d, 'lyman_lp': info.get('lp'), 'lyman_published_prob': info.get('prob'),
        'lyman_retrained_prob': retrained_prob,
        'age': f.get('age'), 'sex': f.get('sex'), 'wbc': f.get('wbc'),
        'ast': f.get('ast'), 'alp': f.get('alp'), 'tbil': f.get('tbil'),
        'creat': f.get('_creat'), 'egfr': f.get('egfr'),
        'prior_chemo': f.get('prior_chemo'), 'immunosupp': f.get('immunosupp'),
        'chemo_type': f.get('chemo_type'), 'cancer_type': f.get('cancer_type'),
    })
audit_name = save_name.replace('.csv', '_LYMAN_patient_audit.csv')
pd.DataFrame(audit_rows).to_csv(audit_name, index=False)
print(f"Saved Lyman per-patient audit to: {audit_name}")