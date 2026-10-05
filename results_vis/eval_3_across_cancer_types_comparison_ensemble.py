"""SPARC-P performance by cancer type for one task (optionally the 10-fold mean ensemble, -en).
Patients with several cancer types on their earliest date count in each; types need
>= -nc patients and >= -np positives. The 'all' row includes patients without a cancer
type, matching the eval_1/eval_2 baseline. Run -ca True to see the distribution first.

-ci bootstrap adds 95% bootstrap CIs of the ensemble AUROC (*_ensembleCI.csv) and saves the
ensemble predictions (*_ensembleCI_preds.pickle).

Usage: python eval_3_across_cancer_types_comparison_ensemble.py -o neutropenia -ca False -nc 50 -np 10
"""

import os
import pickle
import sys, inspect
import pandas as pd
import numpy as np
import re
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset
from MSK_featurename_list import build_module_index_config
from SPARC_inf_risk_modular import *
for model_option in ['lasttp_MLP', 'raindrop', 'rnn']:
    sparc_paths.add_benchmark_to_path(model_option)
from lasttp_MLP import *
from raindrop import *
from rnn import *
from torchmetrics import AUROC, F1Score, AveragePrecision
data_source_path=sparc_paths.DATA_ROOT + '/'


from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, F1Score, AveragePrecision, Accuracy, Precision, Recall
c_index = ConcordanceIndex()
loss_fn = nn.BCELoss(reduction='none')
auroc = AUROC(task='binary')
accuracy = Accuracy(task='binary')
balanced_accuracy = Accuracy(task='binary', average='macro')
precision = Precision(task='binary')
recall = Recall(task='binary')
f1 = F1Score(task='binary')
auprc = AveragePrecision(task='binary')

from eval_utils import calculate_weighted_c_index_by_infusion

# cancer type helpers

def get_patient_cancer_type(dmp_id, ctype_dict):
    """All cancer types on the patient's earliest date (list), or None."""
    if dmp_id not in ctype_dict:
        return None

    date_to_ctypes = ctype_dict[dmp_id]
    if not date_to_ctypes:
        return None

    # earliest = most negative date
    earliest_date = min(date_to_ctypes.keys())
    cancer_types = date_to_ctypes[earliest_date]

    if not cancer_types or len(cancer_types) == 0:
        return None

    return cancer_types


def collapse_cancer_types(cancer_type):
    """Collapse skin cancers and NHL subtypes; drop the GI-tract parent category (None)."""
    if cancer_type is None:
        return None

    ctype_lower = cancer_type.lower()

    if any(term in ctype_lower for term in ['skin cancer', 'malignant neoplasm of skin', 'malignant melanoma']):
        return "Skin cancer"

    if 'malignant neoplasm of gastrointestinal tract' in ctype_lower:
        return None

    if any(term in ctype_lower for term in ['mature t-cell and/or nk cell neoplasm', "non-hodgkin's lymphoma"]):
        return "Non-Hodgkin lymphoma"

    return cancer_type


def analyze_cancer_type_distribution(ctype_dict, test_patient_ids=None):
    """Unique patients per collapsed cancer type (multi-type patients counted in each)."""
    cancer_type_stats = {}

    patient_ids = test_patient_ids if test_patient_ids else ctype_dict.keys()

    for dmp_id in patient_ids:
        if dmp_id not in ctype_dict:
            continue

        cancer_types = get_patient_cancer_type(dmp_id, ctype_dict)
        if cancer_types is None:
            continue

        for cancer_type in cancer_types:
            collapsed_type = collapse_cancer_types(cancer_type)
            if collapsed_type is None:
                continue

            if collapsed_type not in cancer_type_stats:
                cancer_type_stats[collapsed_type] = {
                    'unique_patients': set(),
                    'original_types': set()
                }

            cancer_type_stats[collapsed_type]['unique_patients'].add(dmp_id)
            cancer_type_stats[collapsed_type]['original_types'].add(cancer_type)

    final_stats = {}
    for ctype, stats in cancer_type_stats.items():
        final_stats[ctype] = {
            'num_unique_patients': len(stats['unique_patients']),
            'original_types': list(stats['original_types'])
        }

    return final_stats


def count_positive_cases_by_cancer_type(ctype_dict, data_module, option):
    """Positive test samples per collapsed cancer type."""
    ctype_to_positive_count = {}

    dmp_ids = data_module.test_dataset.dataset_DMP_IDs
    labels = data_module.test_dataset.ys

    for i, dmp_id in enumerate(dmp_ids):
        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
            label = labels[i][0]  # side-effect y = (event, time_to_event)
        else:
            label = labels[i]

        if label != 1:
            continue

        cancer_types = get_patient_cancer_type(dmp_id, ctype_dict)
        if cancer_types is None:
            continue

        for cancer_type in cancer_types:
            collapsed_type = collapse_cancer_types(cancer_type)
            if collapsed_type is None:
                continue

            if collapsed_type not in ctype_to_positive_count:
                ctype_to_positive_count[collapsed_type] = 0
            ctype_to_positive_count[collapsed_type] += 1

    return ctype_to_positive_count


def print_cancer_type_analysis(ctype_stats, ctype_stats_all=None):
    """Print the test-set cancer type distribution, with whole-dataset counts if given."""
    print("=" * 80)
    print("CANCER TYPE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    sorted_ctypes = sorted(ctype_stats.items(),
                          key=lambda x: x[1]['num_unique_patients'],
                          reverse=True)

    total_ctypes = len(sorted_ctypes)
    total_patients = sum([stats['num_unique_patients'] for _, stats in sorted_ctypes])

    print(f"Total number of unique cancer types (Test Set): {total_ctypes}")
    print(f"Total patients with cancer type data (Test Set): {total_patients}")

    if ctype_stats_all:
        total_ctypes_all = len(ctype_stats_all)
        total_patients_all = sum([stats['num_unique_patients'] for _, stats in ctype_stats_all.items()])
        print(f"Total number of unique cancer types (Entire Dataset): {total_ctypes_all}")
        print(f"Total patients with cancer type data (Entire Dataset): {total_patients_all}")

    print("\n" + "=" * 80)

    for i, (ctype, stats) in enumerate(sorted_ctypes[:30], 1):  # Show top 30
        print(f"\nCancer Type #{i}: {ctype}")

        if ctype_stats_all and ctype in ctype_stats_all:
            stats_all = ctype_stats_all[ctype]
            print(f"  Test Set: {stats['num_unique_patients']} patients")
            print(f"  Entire Dataset: {stats_all['num_unique_patients']} patients")

            pct_patients = (stats['num_unique_patients'] / stats_all['num_unique_patients'] * 100) if stats_all['num_unique_patients'] > 0 else 0
            print(f"  Test coverage: {pct_patients:.1f}% patients")
        else:
            print(f"  Unique patients: {stats['num_unique_patients']}")

        # list originals if several were collapsed
        if len(stats['original_types']) > 1:
            print(f"  Collapsed from: {', '.join(stats['original_types'])}")

    if total_ctypes > 30:
        print(f"\n... and {total_ctypes - 30} more cancer types")

    print("\n" + "=" * 80)


# args

import argparse
parser = argparse.ArgumentParser(
            prog='eval_across_cancer_types_comparison',
            description='calculate the model performance comparison across cancer types',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-m", "--modules", type=str, default = 'practical', help='module category, either "all", "allnonICD" or connect categories by dashes such as')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK_CAIA', help='dataset used in training, options: MSK, MSK_CAIA, MSK_CAIA_timeshift')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-rt", "--retrain", type=str, default = 'False', help='whether to retrain the model or not, options: True, False; only legit for AKI_2wks for the MIMICIV_hosp dataset')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-ca", "--ctype_analysis_only", type=str, default = "True", help="whether to only run cancer type analysis without model evaluation")
parser.add_argument("-nc", "--N_ctype", type=int, default = 50, help="minimum number of patients required for a cancer type to be included")
parser.add_argument("-np", "--N_pos", type=int, default = 10, help="minimum number of positive cases required for a cancer type to be included")
parser.add_argument("-en", "--ensemble", type=str, default='True', help='whether to compute ensemble predictions by averaging over 10 folds for each model-cancer-type combination, options: True, False')
parser.add_argument("-bf", "--batch_first", type=str, default='True',
                    help="True | False -- build SPARC with batch_first=True so the encoder "
                         "attends over time. MUST match how the checkpoints were trained; "
                         "a mismatch loads silently.")
parser.add_argument("-rp", "--results_path", type=str, default=None,
                    help='root directory for outputs; defaults to sparc_paths.SPARC_RESULTS_VIS_ROOT')
parser.add_argument("-ci", "--ci_method", type=str, default='none', choices=['none', 'bootstrap'],
                    help='bootstrap: 95%% CIs of the ensemble AUROC, written to *_ensembleCI.csv '
                         'together with the ensemble predictions')
parser.add_argument("-nb", "--n_bootstrap", type=int, default=1000)

args = parser.parse_args()
_RESULTS_ROOT = args.results_path or sparc_paths.SPARC_RESULTS_VIS_ROOT
ci_method = args.ci_method
n_bootstrap = args.n_bootstrap


def ensemble_bootstrap_ci(preds, events, times, infusions, n_boot, seed):
    """95% bootstrap CIs of the ensemble's next-infusion AUROC, as in eval_1.

    Resamples rows of the infusion groups that have both classes (the groups the
    point estimate uses); weighted and overall AUROC come from the same draws.
    """
    from sklearn.metrics import roc_auc_score
    from eval_utils import nextinf_auroc_weighted_overall
    y = ((np.asarray(events) == 1) & (np.asarray(times) == 1)).astype(int)
    p = np.asarray(preds, dtype=float)
    inf = np.asarray(infusions)
    keep = np.zeros(len(y), dtype=bool)
    for g in np.unique(inf):
        m = inf == g
        if 0 < y[m].sum() < m.sum():
            keep |= m
    if not keep.any():
        return {}
    y, p, inf = y[keep], p[keep], inf[keep]
    rng = np.random.default_rng(seed)
    overall, weighted = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        overall.append(roc_auc_score(y[idx], p[idx]))
        w, _ = nextinf_auroc_weighted_overall(p[idx], y[idx], inf[idx])
        if not np.isnan(w):
            weighted.append(w)
    out = {}
    if overall:
        out['auroc_ci_lower'], out['auroc_ci_upper'] = np.percentile(overall, [2.5, 97.5])
    if weighted:
        out['auroc_weighted_ci_lower'], out['auroc_weighted_ci_upper'] = np.percentile(weighted, [2.5, 97.5])
    return out


def binary_bootstrap_ci(preds, events, n_boot, seed):
    """95% bootstrap CI of the ensemble AUROC for the binary (mortality) tasks."""
    from sklearn.metrics import roc_auc_score
    y = (np.asarray(events) == 1).astype(int)
    p = np.asarray(preds, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return {}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {'auroc_ci_lower': lo, 'auroc_ci_upper': hi}

rseed = args.rseed
option = args.option
med_of_choice = args.med_of_choice
days = args.days
modules = args.modules
dataset = args.dataset
retrain = args.retrain.lower() == 'true'
add_num_infusion = args.add_num_infusion.lower() == 'true'
mode = args.mode.lower()
ctype_analysis_only = args.ctype_analysis_only.lower() == 'true'
N_ctype = args.N_ctype
N_pos = args.N_pos
ensemble = args.ensemble.lower() == 'true'

# hardcoded per-task presets; the last block wins, but the CLI args are re-applied below
rseed = 1024
option = 'neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False


rseed = 1024
option = 'febrile_neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False

rseed = 1024
option = 'anemia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False

rseed = 1024
option = 'Liver_toxicity'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB"
add_num_infusion = True
ctype_analysis_only = False

rseed = 1024
option = 'AKI'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = True
ctype_analysis_only = False
N_ctype = 50
N_pos = 10


rseed = 1024
option = 'binary_icu_pre'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = False
ctype_analysis_only = False
N_ctype = 50
N_pos = 10

rseed = 1024
option = 'binary'
days = 180
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = False
ctype_analysis_only = False
N_ctype = 50
N_pos = 10

# re-apply CLI args over the presets above
option = args.option
med_of_choice = args.med_of_choice
days = args.days
modules = args.modules
dataset = args.dataset
add_num_infusion = args.add_num_infusion.lower() == 'true'
ctype_analysis_only = args.ctype_analysis_only.lower() == 'true'
N_ctype = args.N_ctype
N_pos = args.N_pos






if dataset not in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
    raise ValueError(f"Cancer type analysis only supported for MSK_CAIA datasets, got {dataset}")

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# 'A+B+C' combos become nested lists
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        split_meds = med_str.split('+')
        split_meds = [med.strip() for med in split_meds]
        result_meds.append(split_meds)
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print("med_of_choice is a list of medications:", med_of_choice)

batch_size = 512
competing_risk_death_check = False
include_gcsf = False

if modules == 'all':
    data_modules_to_include = ['gender', 'age', 'basic_lab', 'basic_vitals', 'extra_lab', 'extra_vitals']
    component = 'all'
elif modules == 'mutation_clinical':
    data_modules_to_include = ['gender', 'age', 'mutation', 'clinical']
    component = 'mutation_clinical_only'
elif modules == 'practical':
    data_modules_to_include = ['gender', 'age', 'basic_lab', 'basic_vitals']
    component = 'ts_only'
elif modules == 'parsimonious':
    data_modules_to_include = ['gender', 'age', 'cbc_differential', 'basic_lab', 'basic_vitals']
    component = 'parsimonious'

print("Checkpoint 1: Loading dataset using dataset_loading.py")

data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=0,  # data loaded once; folds only change the model
    mode='test_all' if 'timeshift' in dataset else 'test',
    rseed=rseed,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=include_gcsf,
    data_fraction_option='default',
    quick_test=False,
    med_regimen_eval=False,
)

print("modular_featuredim_dict", modular_featuredim_dict)
print("Data loading complete")

# cancer type data

print("\nCheckpoint 2: Loading cancer type data")

ctype_dict_path = f'{data_source_path}/MSK_CAIA/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_till2021_050726.pickle'
with open(ctype_dict_path, 'rb') as f:
    ctype_dict = pickle.load(f)

print(f"Loaded cancer type data for {len(ctype_dict)} patients")

# step 1: distribution

print("\n" + "="*80)
print("STEP 1: CANCER TYPE DISTRIBUTION ANALYSIS")
print("="*80)
print("Analyzing cancer type distribution for manual selection...")
print("")

test_patient_ids = set(split_dict['test'])

ctype_stats_test = analyze_cancer_type_distribution(ctype_dict, test_patient_ids)
ctype_stats_all = analyze_cancer_type_distribution(ctype_dict)

print_cancer_type_analysis(ctype_stats_test, ctype_stats_all)

print("\n" + "="*80)
print("TOP 30 MOST COMMON CANCER TYPES (for manual selection)")
print("="*80)
print("\nCopy the cancer types you want to analyze and add them to cancer_type_list below:")
print("")

sorted_ctypes = sorted(ctype_stats_test.items(),
                      key=lambda x: x[1]['num_unique_patients'],
                      reverse=True)

top_ctype_dict = {}
for i, (ctype, stats) in enumerate(sorted_ctypes[:30], 1):
    top_ctype_dict[ctype] = {
        'num_unique_patients': stats['num_unique_patients']
    }
    print(f"{i:2d}. '{ctype}'")

    if ctype in ctype_stats_all:
        stats_all = ctype_stats_all[ctype]
        print(f"    Test: {stats['num_unique_patients']} patients | "
              f"Entire: {stats_all['num_unique_patients']} patients")

        pct_patients = (stats['num_unique_patients'] / stats_all['num_unique_patients'] * 100) if stats_all['num_unique_patients'] > 0 else 0
        print(f"    Coverage: {pct_patients:.1f}% patients")
    else:
        print(f"    Test: {stats['num_unique_patients']} patients")

print("\n" + "="*80)
print("INSTRUCTIONS FOR NEXT STEP:")
print("="*80)
print("1. Review the cancer types above")
print("2. Adjust N_ctype and N_pos thresholds if needed (use -nc and -np arguments)")
print("3. Set ctype_analysis_only = False (or use -ca False)")
print("4. Re-run the script to perform model evaluation on automatically selected cancer types")
print("")
print("NOTE: Cancer types are automatically filtered based on:")
print(f"  - N_ctype (minimum patients): {N_ctype}")
print(f"  - N_pos (minimum positive cases): {N_pos}")
print("="*80)

if ctype_analysis_only:
    print("\nCancer type analysis complete. Exiting without model evaluation.")
    print("Edit the script to add your selected cancer types and set ctype_analysis_only=False")
    sys.exit(0)

# task type (before building data modules)

if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
    model_set_option = 'side_effects_risk'
    option = f'{option}_risk'
elif option in ['binary', 'binary_icu', 'binary_icu_pre']:
    model_set_option = 'binary'
    # keep option as-is (no _risk suffix)
else:
    model_set_option = option

# auto-filter cancer types by -nc / -np

print("\n" + "="*80)
print("STEP 1.5: COUNTING POSITIVE CASES BY CANCER TYPE")
print("="*80)
print(f"Creating data module to count positive cases...")

# data module over all test patients, only to count positives
advanced_model = 'default'
all_test_split_dict = {
    'train': [],
    'val': [],
    'test': list(test_patient_ids)
}

temp_data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, all_test_split_dict, dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model=advanced_model, control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=include_infusion_dates, premed_only=False, competing_risk_death_check=False,
    anc=('CAIA' in dataset)
)

print(f"Counting positive cases across cancer types...")
ctype_to_positive_count = count_positive_cases_by_cancer_type(ctype_dict, temp_data_module, option)

print("\n" + "="*80)
print("APPLYING AUTOMATIC THRESHOLDS")
print("="*80)
print(f"Threshold requirements:")
print(f"  - Minimum patients (N_ctype): {N_ctype}")
print(f"  - Minimum positive cases (N_pos): {N_pos}")
print("")

cancer_type_list = ['all']
filtered_ctypes = []

for ctype, stats in sorted_ctypes:
    num_patients = stats['num_unique_patients']
    num_positives = ctype_to_positive_count.get(ctype, 0)

    if num_patients >= N_ctype and num_positives >= N_pos:
        filtered_ctypes.append((ctype, num_patients, num_positives))
        cancer_type_list.append(ctype)
        print(f"ok INCLUDED: '{ctype}'")
        print(f"    Patients: {num_patients} (≥{N_ctype}), Positives: {num_positives} (≥{N_pos})")
    else:
        print(f"FAIL EXCLUDED: '{ctype}'")
        print(f"    Patients: {num_patients} ({'PASS' if num_patients >= N_ctype else 'FAIL'}), "
              f"Positives: {num_positives} ({'PASS' if num_positives >= N_pos else 'FAIL'})")

print("\n" + "="*80)
print(f"FILTERING SUMMARY")
print("="*80)
print(f"Total cancer types after collapsing: {len(sorted_ctypes)}")
print(f"Cancer types meeting thresholds: {len(filtered_ctypes)}")
print(f"Cancer types to evaluate: {len(cancer_type_list)} (including 'all')")
print("="*80)

print("\n" + "="*80)
print("STEP 2: MODEL EVALUATION ON SELECTED CANCER TYPES")
print("="*80)

# model paths

all_model_path = sparc_paths.SPARC_MODEL_ROOT

# folder prefix: 'hybrid' for FN/N/A, 'risk' for AKI/LT, none for binary
# option may already carry the _risk suffix here
base_option = option.replace('_risk', '')
if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
    folder_prefix = 'hybrid'
elif base_option in ['AKI', 'Liver_toxicity']:
    folder_prefix = 'risk'
elif option in ['binary', 'binary_icu', 'binary_icu_pre']:
    folder_prefix = ''
else:
    folder_prefix = ''

if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
    if model_set_option == 'side_effects_risk':
        model_folder_dict = {
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{folder_prefix}_{base_option}_20tp_topchemos_att_modular'
        }
    else:
        model_folder_dict = {
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{option}_20tp_topchemos_att_modular'
        }
else:
    raise ValueError(f"Dataset {dataset} not yet supported for 060726 models")

_ensemble_suffix = '_ensemble' if ensemble else ''
_ci_suffix = 'CI' if ci_method == 'bootstrap' else ''
save_name = f'{_RESULTS_ROOT}/eval_across_ctypes/model_performance_dict_across_ctypes_{modules}_{option}_{dataset}_060726{_ensemble_suffix}{_ci_suffix}.csv'
ensemble_preds_save = {}

print(f"Using split_dict from load_dataset for {dataset}")
print(f"  Test set size: {len(split_dict['test'])}")

model_performance_dict = {}

# resume from the saved csv if present
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)

    for index, row in saved_df.iterrows():
        model_name = row['model']
        val_fold = row['val_fold']
        ctype_str = row['cancer_type']
        unique_num_patients = row['unique_num_patients']
        unique_positives = row['unique_positives']

        if ctype_str not in model_performance_dict:
            model_performance_dict[ctype_str] = {
                'SPARC-P-all': {},
                model_name: {}
            }

        metrics = {
            'c_index_overall': row.get('c_index_overall'),
            'c_index_weighted': row.get('c_index_weighted'),
            'auroc_nextinf_overall': row.get('auroc_nextinf_overall'),
            'auroc_nextinf_weighted': row.get('auroc_nextinf_weighted'),
            'auroc': row.get('auroc'),
            'auprc': row.get('auprc'),
            'unique_num_patients': unique_num_patients,
            'unique_positives': unique_positives,
        }

        if model_name not in model_performance_dict[ctype_str]:
            model_performance_dict[ctype_str][model_name] = {}

        model_performance_dict[ctype_str][model_name][val_fold] = metrics
        model_performance_dict[ctype_str]['SPARC-P-all'][val_fold] = metrics

    print("Loaded existing model performance data from csv")

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# columns cover both side-effect and binary metrics
all_columns = ['cancer_type', 'model', 'val_fold',
               'c_index_overall', 'c_index_weighted', 'auroc_nextinf_overall', 'auroc_nextinf_weighted',
               'auroc', 'auprc',
               'unique_num_patients', 'unique_positives'] + (['auroc_ci_lower', 'auroc_ci_upper', 'auroc_weighted_ci_lower', 'auroc_weighted_ci_upper'] if ci_method == 'bootstrap' else [])

# evaluation loop

for ctype in cancer_type_list:
    print(f"\n{'='*80}")
    print(f"Processing cancer type: {ctype}")
    print(f"{'='*80}")

    ctype_str = ctype

    if ctype_str in model_performance_dict and 'SPARC-P-all' in model_performance_dict[ctype_str]:
        all_folds_exist = all(fold in model_performance_dict[ctype_str]['SPARC-P-all'] for fold in all_folds)
        if all_folds_exist:
            print(f"Data module for cancer type {ctype_str} already exists for all folds, skipping")
            continue

    pl.seed_everything(rseed)

    advanced_model = 'default'

    if ctype == 'all':
        # all test patients, including those without a cancer type (matches eval_1/eval_2)
        filtered_patient_ids = list(test_patient_ids)
    else:
        # a patient matches if any of their cancer types collapses to ctype
        filtered_patient_ids = []
        for pid in test_patient_ids:
            patient_ctypes = get_patient_cancer_type(pid, ctype_dict)
            if patient_ctypes is None:
                continue

            for patient_ctype in patient_ctypes:
                collapsed_ctype = collapse_cancer_types(patient_ctype)
                if collapsed_ctype == ctype:
                    filtered_patient_ids.append(pid)
                    break

    if ctype == 'all':
        print(f"Found {len(filtered_patient_ids)} patients in 'all' category (includes patients without cancer type data)")
    else:
        print(f"Found {len(filtered_patient_ids)} patients with cancer type '{ctype}'")

    if len(filtered_patient_ids) == 0:
        print(f"No eligible patients for cancer type {ctype_str}, recording null entries and skipping")
        if ctype_str not in model_performance_dict:
            model_performance_dict[ctype_str] = {}
        null_entry = {
            'c_index_overall': None, 'c_index_weighted': None,
            'auroc_nextinf_overall': None, 'auroc_nextinf_weighted': None,
            'auroc': None, 'auprc': None,
            'unique_num_patients': 0, 'unique_positives': 0,
        }
        for _model_key in ['SPARC-P', 'SPARC-P-all']:
            if _model_key not in model_performance_dict[ctype_str]:
                model_performance_dict[ctype_str][_model_key] = {}
            for val_fold in all_folds:
                model_performance_dict[ctype_str][_model_key][val_fold] = null_entry.copy()
        continue

    filtered_split_dict = {
        'train': [],
        'val': [],
        'test': filtered_patient_ids
    }

    data_module = FlexibleFeatureDataModule(data_dict_lab_dict, filtered_split_dict, dataset = dataset, component = component,
            batch_size = 512, max_time_len = 20, num_workers=1,
            task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
            advanced_model = advanced_model, control_lack_confounds = False, side_effect_censoring = False,
            include_infusion_dates = include_infusion_dates, premed_only = False, competing_risk_death_check = False,
            anc=('CAIA' in dataset))

    print("data module initiated")

    # test_dataloader() crashes on an empty dataset
    if len(data_module.test_dataset) == 0:
        print(f"No eligible data points for cancer type {ctype_str}, recording null entries and skipping")
        if ctype_str not in model_performance_dict:
            model_performance_dict[ctype_str] = {}
        null_entry = {
            'c_index_overall': None, 'c_index_weighted': None,
            'auroc_nextinf_overall': None, 'auroc_nextinf_weighted': None,
            'auroc': None, 'auprc': None,
            'unique_num_patients': 0, 'unique_positives': 0,
        }
        for _model_key in ['SPARC-P', 'SPARC-P-all']:
            if _model_key not in model_performance_dict[ctype_str]:
                model_performance_dict[ctype_str][_model_key] = {}
            for val_fold in all_folds:
                model_performance_dict[ctype_str][_model_key][val_fold] = null_entry.copy()
        continue

    unique_num_patients = len(set(data_module.test_dataloader().dataset.dataset_DMP_IDs))
    # unique_positives = unique patients who ever had the event
    pos_patient_ids = set()
    for i in range(len(data_module.test_dataset.dataset_DMP_IDs)):
        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
            label = data_module.test_dataset.ys[i][0]  # side-effect y = (event, time_to_event)
        else:
            label = data_module.test_dataset.ys[i]
        if label == 1:
            pos_patient_ids.add(data_module.test_dataset.dataset_DMP_IDs[i])
    unique_positives = len(pos_patient_ids)
    print("unique_num_patients", unique_num_patients, "unique_positives", unique_positives)

    if ctype_str not in model_performance_dict:
        model_performance_dict[ctype_str] = {}

    # only SPARC-P is evaluated per cancer type
    model_option = 'SPARC-P'

    if model_option not in model_performance_dict[ctype_str]:
        model_performance_dict[ctype_str][model_option] = {}

    for val_fold in all_folds:
        pl.seed_everything(rseed)

        if val_fold in model_performance_dict[ctype_str][model_option] and \
           len(model_performance_dict[ctype_str][model_option][val_fold]) > 0:
            print(f"Performance for cancer type {ctype_str} and model {model_option} fold {val_fold} already exists, skipping")
            continue

        model_performance_dict[ctype_str][model_option][val_fold] = {}

        print(f"Processing {model_option} for fold {val_fold}")

        model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        print(model_folder)
        if not os.path.exists(model_folder):
            print(f"Model folder {model_folder} does not exist, skipping")
            continue
        existing_files = os.listdir(model_folder)

        # first 'epoch' checkpoint (not last.ckpt)
        if not any(['epoch' in el for el in existing_files]):
            print(f"No epoch checkpoint found in {model_folder}, skipping")
            continue
        optimal_model_path = [el for el in existing_files if 'epoch' in el][0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(model_option, "optimal_model_path", optimal_model_path)

        module_index_cfg = build_module_index_config(
            modular_featuredim_dict, advanced_model='default'
        )
        n_static_features = module_index_cfg['n_static']

        if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
            loss_option = 'hybrid'
        elif base_option in ['AKI', 'Liver_toxicity']:
            loss_option = 'default'
        else:
            loss_option = 'default'

        model = SPARC.load_from_checkpoint(
            optimal_model_path,
            modular_featuredim_dict=modular_featuredim_dict,
            option=model_set_option,
            advanced_model="default",
            max_time_len=20,
            med_as_feature=False,
            meds_l=med_of_choice,
            side_effects_num_infusion=include_infusion_dates,
            temporal_aggregation='attention',
            module_index_cfg=module_index_cfg,
            use_module_tokenizer=True,
            use_continuous_time=True,
            n_static_features=n_static_features,
            post_mlp_dropout=0.2,
            loss_option=loss_option,
        batch_first=(args.batch_first == 'True'),
    )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model {model_option} on device: {device}")

        # collect all batches, then compute metrics once
        pl.seed_everything(rseed)

        all_y_preds = []
        all_y_events = []
        all_time_to_events = []
        all_num_infusions = []

        batch_count = 0
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                if include_gcsf:
                    xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                else:
                    xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                    gcsf_groundtruth = None

                xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                if add_num_infusion and num_infusion is not None:
                    num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                    # side-effect y = (event, time_to_event)
                    fail_indicators = torch.as_tensor(y[0], device=device)
                    time_to_event = torch.as_tensor(y[1], device=device)
                else:
                    # binary: time_to_event set to 1
                    fail_indicators = torch.as_tensor(y, device=device)
                    time_to_event = torch.ones_like(fail_indicators, device=device).float()

                if fail_indicators.dim() == 0:
                    fail_indicators = fail_indicators.unsqueeze(0)
                if time_to_event.dim() == 0:
                    time_to_event = time_to_event.unsqueeze(0)

                all_y_preds.append(y_pred_logits.detach().cpu())
                all_y_events.append(fail_indicators.detach().cpu())
                all_time_to_events.append(time_to_event.detach().cpu())
                if num_infusion is not None:
                    all_num_infusions.append(num_infusion.detach().cpu())
                else:
                    all_num_infusions.append(torch.zeros(fail_indicators.shape[0], dtype=torch.long))

                batch_count += 1

        print(f"Total batches: {batch_count}")
        print(f"Total samples: {sum(t.numel() for t in all_y_events)}")
        print(f"Positive events: {sum((t == 1).sum().item() for t in all_y_events)}")

        all_y_preds = torch.cat(all_y_preds, dim=0)
        all_y_events = torch.cat(all_y_events, dim=0)
        all_time_to_events = torch.cat(all_time_to_events, dim=0)
        all_num_infusions = torch.cat(all_num_infusions, dim=0)

        print(f"After concatenation - Total samples: {len(all_y_events)}, Positive: {(all_y_events == 1).sum().item()}")

        if len(all_y_events) > 0 and (all_y_events == 1).sum().item() > 0:
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                # side-effect: C-index and next-infusion AUROC
                c_idx_dict = calculate_weighted_c_index_by_infusion(
                    all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                    mode='val', option=option
                )
                model_performance_dict[ctype_str][model_option][val_fold]['c_index_overall'] = c_idx_dict.get('c_index_overall', 0.5)
                model_performance_dict[ctype_str][model_option][val_fold]['c_index_weighted'] = c_idx_dict.get('c_index_weighted', 0.5)
                model_performance_dict[ctype_str][model_option][val_fold]['auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
                model_performance_dict[ctype_str][model_option][val_fold]['auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
                print(f"C-index overall: {c_idx_dict.get('c_index_overall', 0.5)}, C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5)}")
                print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
            else:
                # binary: AUROC/AUPRC
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
                auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()

                model_performance_dict[ctype_str][model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[ctype_str][model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")
        else:
            # no positives: fill defaults
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                model_performance_dict[ctype_str][model_option][val_fold]['c_index_overall'] = 0.5
                model_performance_dict[ctype_str][model_option][val_fold]['c_index_weighted'] = 0.5
                model_performance_dict[ctype_str][model_option][val_fold]['auroc_nextinf_overall'] = None
                model_performance_dict[ctype_str][model_option][val_fold]['auroc_nextinf_weighted'] = None
            else:
                model_performance_dict[ctype_str][model_option][val_fold]['auroc'] = 0.5
                model_performance_dict[ctype_str][model_option][val_fold]['auprc'] = 0.5
            print("No positive cases found in this group")

        model_performance_dict[ctype_str][model_option][val_fold]['unique_num_patients'] = unique_num_patients
        model_performance_dict[ctype_str][model_option][val_fold]['unique_positives'] = unique_positives

        # keep raw predictions for the ensemble
        if ensemble:
            model_performance_dict[ctype_str][model_option][val_fold]['y_preds'] = all_y_preds.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['y_events'] = all_y_events.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['time_to_events'] = all_time_to_events.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['num_infusions'] = all_num_infusions.cpu().numpy()

    # mean ensemble over folds (only when all 10 have predictions)
    if ensemble:
        for ctype_str in model_performance_dict.keys():
            for model_option in [k for k in model_performance_dict[ctype_str].keys() if not k.endswith('_ensemble')]:
                all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                all_folds_complete = all(fold in model_performance_dict[ctype_str][model_option] and
                                        'y_preds' in model_performance_dict[ctype_str][model_option][fold]
                                        for fold in all_folds)

                if all_folds_complete:
                    print(f"\n{'='*80}")
                    print(f"Computing ensemble for {model_option} on {ctype_str}")
                    print(f"{'='*80}")

                    fold_preds = [model_performance_dict[ctype_str][model_option][fold]['y_preds'] for fold in all_folds]
                    fold_events = [model_performance_dict[ctype_str][model_option][fold]['y_events'] for fold in all_folds]
                    fold_times = [model_performance_dict[ctype_str][model_option][fold]['time_to_events'] for fold in all_folds]
                    fold_infusions = [model_performance_dict[ctype_str][model_option][fold]['num_infusions'] for fold in all_folds]

                    assert len(set(len(p) for p in fold_preds)) == 1, "All folds should have same number of test samples"

                    ensemble_preds = np.mean(fold_preds, axis=0)
                    ensemble_events = fold_events[0]  # labels identical across folds
                    ensemble_times = fold_times[0]
                    ensemble_infusions = fold_infusions[0]

                    import torch
                    ensemble_preds_tensor = torch.tensor(ensemble_preds, dtype=torch.float32)
                    ensemble_events_tensor = torch.tensor(ensemble_events, dtype=torch.float32)
                    ensemble_times_tensor = torch.tensor(ensemble_times, dtype=torch.float32)
                    ensemble_infusions_tensor = torch.tensor(ensemble_infusions, dtype=torch.float32)

                    if len(ensemble_events_tensor) > 0 and (ensemble_events_tensor == 1).sum().item() > 0:
                        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                            c_idx_dict = calculate_weighted_c_index_by_infusion(
                                ensemble_preds_tensor, ensemble_events_tensor, ensemble_times_tensor, ensemble_infusions_tensor,
                                mode='val', option=option
                            )

                            unique_patients = len(set([fold_infusions[0][i] for i in range(len(fold_infusions[0]))]))
                            unique_positives = (ensemble_events_tensor == 1).sum().item()

                            ensemble_name = f'{model_option}_ensemble'
                            if ensemble_name not in model_performance_dict[ctype_str]:
                                model_performance_dict[ctype_str][ensemble_name] = {}

                            model_performance_dict[ctype_str][ensemble_name]['ensemble'] = {
                                'c_index_overall': c_idx_dict.get('c_index_overall', 0.5),
                                'c_index_weighted': c_idx_dict.get('c_index_weighted', 0.5),
                                'auroc_nextinf_overall': c_idx_dict.get('auroc_next_infusion_overall', None),
                                'auroc_nextinf_weighted': c_idx_dict.get('auroc_next_infusion_weighted', None),
                                'unique_num_patients': unique_patients,
                                'unique_positives': unique_positives,
                            }
                            if ci_method == 'bootstrap':
                                if (ctype_str, model_option) not in ensemble_preds_save:
                                    ensemble_preds_save[(ctype_str, model_option)] = {
                                        'y_preds': np.asarray(ensemble_preds), 'y_events': np.asarray(ensemble_events),
                                        'time_to_events': np.asarray(ensemble_times), 'num_infusions': np.asarray(ensemble_infusions),
                                        'ci': ensemble_bootstrap_ci(ensemble_preds, ensemble_events, ensemble_times,
                                                                    ensemble_infusions, n_bootstrap, rseed)}
                                model_performance_dict[ctype_str][ensemble_name]['ensemble'].update(
                                    ensemble_preds_save[(ctype_str, model_option)]['ci'])

                            print(f"Ensemble C-index overall: {c_idx_dict.get('c_index_overall', 0.5):.4f}, "
                                  f"C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5):.4f}")
                            print(f"Ensemble AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
                        else:
                            from torchmetrics import AUROC, AveragePrecision
                            auroc_fn = AUROC(task='binary')
                            auprc_fn = AveragePrecision(task='binary')

                            auroc_score = auroc_fn(ensemble_preds_tensor, ensemble_events_tensor.int()).item()
                            auprc_score = auprc_fn(ensemble_preds_tensor, ensemble_events_tensor.int()).item()

                            ensemble_name = f'{model_option}_ensemble'
                            if ensemble_name not in model_performance_dict[ctype_str]:
                                model_performance_dict[ctype_str][ensemble_name] = {}

                            model_performance_dict[ctype_str][ensemble_name]['ensemble'] = {
                                'auroc': auroc_score,
                                'auprc': auprc_score,
                                'unique_num_patients': len(set([fold_infusions[0][i] for i in range(len(fold_infusions[0]))])),
                                'unique_positives': (ensemble_events_tensor == 1).sum().item(),
                            }

                            if ci_method == 'bootstrap':
                                if (ctype_str, model_option) not in ensemble_preds_save:
                                    ensemble_preds_save[(ctype_str, model_option)] = {
                                        'y_preds': np.asarray(ensemble_preds), 'y_events': np.asarray(ensemble_events),
                                        'time_to_events': np.asarray(ensemble_times), 'num_infusions': np.asarray(ensemble_infusions),
                                        'ci': binary_bootstrap_ci(ensemble_preds, ensemble_events, n_bootstrap, rseed)}
                                model_performance_dict[ctype_str][ensemble_name]['ensemble'].update(
                                    ensemble_preds_save[(ctype_str, model_option)]['ci'])

                            print(f"Ensemble AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")
                    print(f"{'='*80}\n")

    # checkpoint to csv after each cancer type
    model_performance_df = pd.DataFrame(columns=all_columns)
    for ctype_str_key in model_performance_dict.keys():
        for model_option_key in model_performance_dict[ctype_str_key].keys():
            for val_fold_key in model_performance_dict[ctype_str_key][model_option_key].keys():
                row = {
                    'cancer_type': ctype_str_key,
                    'model': model_option_key,
                    'val_fold': val_fold_key
                }
                # metrics only, not prediction arrays
                fold_data = {k: v for k, v in model_performance_dict[ctype_str_key][model_option_key][val_fold_key].items()
                            if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
                row.update(fold_data)
                row_values = [row.get(col, None) for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)
    model_performance_df = model_performance_df.drop_duplicates(subset=['cancer_type', 'model', 'val_fold'])
    model_performance_df.to_csv(save_name, index=False)

model_performance_df = pd.DataFrame(columns=all_columns)
for ctype_str in model_performance_dict.keys():
    for model_option in model_performance_dict[ctype_str].keys():
        for val_fold in model_performance_dict[ctype_str][model_option].keys():
            row = {
                'cancer_type': ctype_str,
                'model': model_option,
                'val_fold': val_fold
            }
            # metrics only, not prediction arrays
            fold_data = {k: v for k, v in model_performance_dict[ctype_str][model_option][val_fold].items()
                        if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
            row.update(fold_data)
            row_values = [row.get(col, None) for col in all_columns]
            model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)

model_performance_df = model_performance_df.drop_duplicates(subset=['cancer_type', 'model', 'val_fold'])
model_performance_df.to_csv(save_name, index=False)

print("\nModel performance dict saved to csv")
print(f"Results saved to: {save_name}")

if ci_method == 'bootstrap':
    with open(save_name.replace('.csv', '_preds.pickle'), 'wb') as f:
        pickle.dump(ensemble_preds_save, f)
    print(f"Ensemble predictions saved to: {save_name.replace('.csv', '_preds.pickle')}")
