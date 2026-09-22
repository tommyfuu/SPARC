"""Model performance broken down by cancer type, for one task.

With --ensemble (-en), the 10 folds' predictions are averaged and written as
'{model}_ensemble' with val_fold='ensemble'.

Three things to know:
  * A patient with several cancer types on their earliest date is counted in
    each of them.
  * Cancer types must clear both -nc (minimum patients, default 50) and -np
    (minimum positives, default 10) to be evaluated.
  * The 'all' row includes every test patient, including those with no cancer
    type annotation, so it matches the baseline in eval_1 and eval_2.

Run with -ca True first to see the distribution, then -ca False to evaluate.

Usage:
    python eval_3_across_cancer_types_comparison_ensemble.py -o neutropenia -ca True -nc 50 -np 10
    python eval_3_across_cancer_types_comparison_ensemble.py -o neutropenia -ca False -nc 50 -np 10
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
# Use the 060726 versions of data loaders and model
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

# Import shared evaluation utilities
from eval_utils import calculate_weighted_c_index_by_infusion

# CANCER TYPE MAPPING AND COLLAPSING FUNCTIONS

def get_patient_cancer_type(dmp_id, ctype_dict):
    """
    Get ALL cancer types for a patient from the earliest date.

    IMPORTANT: This function now returns a LIST to enable double-counting
    patients with multiple cancer types on the same earliest date.

    Args:
        dmp_id: Patient DMP ID
        ctype_dict: Dictionary mapping DMP_ID to dates to cancer types

    Returns:
        list or None: List of all cancer types from the earliest date, or None if not found
    """
    if dmp_id not in ctype_dict:
        return None

    date_to_ctypes = ctype_dict[dmp_id]
    if not date_to_ctypes:
        return None

    # Get the earliest date (most negative number)
    earliest_date = min(date_to_ctypes.keys())
    cancer_types = date_to_ctypes[earliest_date]

    if not cancer_types or len(cancer_types) == 0:
        return None

    # Return ALL cancer types from the earliest date (enables double-counting)
    return cancer_types


def collapse_cancer_types(cancer_type):
    """
    Apply cancer type collapsing rules:
    1. Merge "Skin cancer", "Malignant neoplasm of skin", and "Malignant melanoma" -> "Skin cancer"
    2. Remove "Malignant neoplasm of gastrointestinal tract" (return None)
    3. Merge "Mature T-cell AND/OR NK cell neoplasm" and "Non-Hodgkin's lymphoma (clinical)" -> "Non-Hodgkin lymphoma"

    Args:
        cancer_type: Original cancer type string

    Returns:
        str or None: Collapsed cancer type, or None if should be excluded
    """
    if cancer_type is None:
        return None

    # Normalize to lowercase for comparison
    ctype_lower = cancer_type.lower()

    # Rule 1: Collapse skin cancers
    if any(term in ctype_lower for term in ['skin cancer', 'malignant neoplasm of skin', 'malignant melanoma']):
        return "Skin cancer"

    # Rule 2: Exclude GI tract parent category
    if 'malignant neoplasm of gastrointestinal tract' in ctype_lower:
        return None

    # Rule 3: Merge lymphomas
    if any(term in ctype_lower for term in ['mature t-cell and/or nk cell neoplasm', "non-hodgkin's lymphoma"]):
        return "Non-Hodgkin lymphoma"

    # Return original if no rules apply
    return cancer_type


def analyze_cancer_type_distribution(ctype_dict, test_patient_ids=None):
    """
    Analyze the distribution of cancer types in the dataset.

    IMPORTANT: Patients with multiple cancer types on the earliest date are
    counted once for EACH cancer type (double-counting enabled).

    Args:
        ctype_dict: Dictionary mapping DMP_ID to dates to cancer types
        test_patient_ids: Optional set of test patient IDs to filter by

    Returns:
        dict: Dictionary with cancer type statistics
    """
    cancer_type_stats = {}

    # Determine which patients to analyze
    patient_ids = test_patient_ids if test_patient_ids else ctype_dict.keys()

    for dmp_id in patient_ids:
        if dmp_id not in ctype_dict:
            continue

        # Get ALL cancer types for this patient (returns a list)
        cancer_types = get_patient_cancer_type(dmp_id, ctype_dict)
        if cancer_types is None:
            continue

        # Iterate through ALL cancer types for this patient (enables double-counting)
        for cancer_type in cancer_types:
            # Apply collapsing rules
            collapsed_type = collapse_cancer_types(cancer_type)
            if collapsed_type is None:
                continue

            # Update statistics
            if collapsed_type not in cancer_type_stats:
                cancer_type_stats[collapsed_type] = {
                    'unique_patients': set(),
                    'original_types': set()
                }

            cancer_type_stats[collapsed_type]['unique_patients'].add(dmp_id)
            cancer_type_stats[collapsed_type]['original_types'].add(cancer_type)

    # Convert sets to counts for final output
    final_stats = {}
    for ctype, stats in cancer_type_stats.items():
        final_stats[ctype] = {
            'num_unique_patients': len(stats['unique_patients']),
            'original_types': list(stats['original_types'])
        }

    return final_stats


def count_positive_cases_by_cancer_type(ctype_dict, data_module, option):
    """
    Count the number of positive cases for each cancer type in the dataset.

    Args:
        ctype_dict: Dictionary mapping DMP_ID to dates to cancer types
        data_module: FlexibleFeatureDataModule with test dataset
        option: Task option (e.g., 'neutropenia_risk', 'AKI_risk')

    Returns:
        dict: Mapping from collapsed cancer type to number of positive cases
    """
    ctype_to_positive_count = {}

    # Get DMP IDs and labels from the dataset
    dmp_ids = data_module.test_dataset.dataset_DMP_IDs
    labels = data_module.test_dataset.ys

    for i, dmp_id in enumerate(dmp_ids):
        # Extract label (handle both side-effect and binary tasks)
        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
            label = labels[i][0]  # Side-effect tasks: (event, time_to_event)
        else:
            label = labels[i]  # Binary tasks: single value

        # Skip negative cases
        if label != 1:
            continue

        # Get all cancer types for this patient
        cancer_types = get_patient_cancer_type(dmp_id, ctype_dict)
        if cancer_types is None:
            continue

        # Count positive case for each cancer type
        for cancer_type in cancer_types:
            collapsed_type = collapse_cancer_types(cancer_type)
            if collapsed_type is None:
                continue

            if collapsed_type not in ctype_to_positive_count:
                ctype_to_positive_count[collapsed_type] = 0
            ctype_to_positive_count[collapsed_type] += 1

    return ctype_to_positive_count


def print_cancer_type_analysis(ctype_stats, ctype_stats_all=None):
    """
    Print a formatted analysis of cancer type distribution

    Args:
        ctype_stats: Statistics for test set
        ctype_stats_all: Statistics for entire dataset (optional)
    """
    print("=" * 80)
    print("CANCER TYPE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Sort cancer types by number of patients (descending)
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

            # Calculate percentage
            pct_patients = (stats['num_unique_patients'] / stats_all['num_unique_patients'] * 100) if stats_all['num_unique_patients'] > 0 else 0
            print(f"  Test coverage: {pct_patients:.1f}% patients")
        else:
            print(f"  Unique patients: {stats['num_unique_patients']}")

        # Show original types if multiple were collapsed
        if len(stats['original_types']) > 1:
            print(f"  Collapsed from: {', '.join(stats['original_types'])}")

    if total_ctypes > 30:
        print(f"\n... and {total_ctypes - 30} more cancer types")

    print("\n" + "=" * 80)


# COMMAND LINE ARGUMENTS

import argparse
parser = argparse.ArgumentParser(
            prog='eval_across_cancer_types_comparison',
            description='calculate the model performance comparison across cancer types',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-m", "--modules", type=str, default = 'practical', help='module category, either "all", "allnonICD" or connect categories by dashes such as')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset used in training, options: MSK, MSK_CAIA, MSK_CAIA_timeshift')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-rt", "--retrain", type=str, default = 'False', help='whether to retrain the model or not, options: True, False; only legit for AKI_2wks for the MIMICIV_hosp dataset')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-ca", "--ctype_analysis_only", type=str, default = "True", help="whether to only run cancer type analysis without model evaluation")
parser.add_argument("-nc", "--N_ctype", type=int, default = 50, help="minimum number of patients required for a cancer type to be included")
parser.add_argument("-np", "--N_pos", type=int, default = 10, help="minimum number of positive cases required for a cancer type to be included")
parser.add_argument("-en", "--ensemble", type=str, default='True', help='whether to compute ensemble predictions by averaging over 10 folds for each model-cancer-type combination, options: True, False')

args = parser.parse_args()
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

# Default settings for testing
rseed = 1024
option = 'neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution


# Default settings for testing
rseed = 1024
option = 'febrile_neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution

# Default settings for testing
rseed = 1024
option = 'anemia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution

# Default settings for testing
rseed = 1024
option = 'Liver_toxicity'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB"
add_num_infusion = True
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution

# Default settings for testing
rseed = 1024
option = 'AKI'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = True
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution
N_ctype = 50  # Minimum number of patients required
N_pos = 10  # Minimum number of positive cases required


# Default settings for testing
rseed = 1024
option = 'binary_icu_pre'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = False
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution
N_ctype = 50  # Minimum number of patients required
N_pos = 10  # Minimum number of positive cases required

# Default settings for testing
rseed = 1024
option = 'binary'
days = 180
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = False
ctype_analysis_only = False  # Set to True for first run to see cancer type distribution
N_ctype = 50  # Minimum number of patients required
N_pos = 10  # Minimum number of positive cases required

# nohup python3 eval_6_across_cancer_types_comparison.py  > output_binary_ctypes.log 2>&1 &          


# bsub -q medium -W 24:00 -o anemia_run.log \                                                 
#     "cd /data/morrisq/fuc/SPARC/results_vis && \                            
#      python eval_2_across_medications_comparison.py"      

# if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
#     raise ValueError(f"Invalid option: {option}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")

if dataset not in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
    raise ValueError(f"Cancer type analysis only supported for MSK_CAIA datasets, got {dataset}")

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# Handle medication combinations (e.g., 'CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE')
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

## HYPERPARAMETERS ##
batch_size = 512
competing_risk_death_check = False
include_gcsf = False

# Determine module configuration
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

# Load data using the centralized dataset_loading module
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=0,  # Will iterate over folds later
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

# LOAD CANCER TYPE DATA

print("\nCheckpoint 2: Loading cancer type data")

ctype_dict_path = f'{data_source_path}/MSK_CAIA/DMP_ID_to_DATE_to_CANCER_TYPES_dict_delta_MSK_CAIA_till2021_050726.pickle'
with open(ctype_dict_path, 'rb') as f:
    ctype_dict = pickle.load(f)

print(f"Loaded cancer type data for {len(ctype_dict)} patients")

# CANCER TYPE ANALYSIS - Print distribution for manual selection

print("\n" + "="*80)
print("STEP 1: CANCER TYPE DISTRIBUTION ANALYSIS")
print("="*80)
print("Analyzing cancer type distribution for manual selection...")
print("")

# Perform cancer type analysis on test set patients only
test_patient_ids = set(split_dict['test'])

ctype_stats_test = analyze_cancer_type_distribution(ctype_dict, test_patient_ids)
ctype_stats_all = analyze_cancer_type_distribution(ctype_dict)

print_cancer_type_analysis(ctype_stats_test, ctype_stats_all)

# Print top 30 most common cancer types in a clean format for easy copying
print("\n" + "="*80)
print("TOP 30 MOST COMMON CANCER TYPES (for manual selection)")
print("="*80)
print("\nCopy the cancer types you want to analyze and add them to cancer_type_list below:")
print("")

# Sort by patient count
sorted_ctypes = sorted(ctype_stats_test.items(),
                      key=lambda x: x[1]['num_unique_patients'],
                      reverse=True)

top_ctype_dict = {}
for i, (ctype, stats) in enumerate(sorted_ctypes[:30], 1):
    top_ctype_dict[ctype] = {
        'num_unique_patients': stats['num_unique_patients']
    }
    print(f"{i:2d}. '{ctype}'")

    # Show both test set and entire dataset statistics
    if ctype in ctype_stats_all:
        stats_all = ctype_stats_all[ctype]
        print(f"    Test: {stats['num_unique_patients']} patients | "
              f"Entire: {stats_all['num_unique_patients']} patients")

        # Calculate percentage
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

# Exit here if only running cancer type analysis
if ctype_analysis_only:
    print("\nCancer type analysis complete. Exiting without model evaluation.")
    print("Edit the script to add your selected cancer types and set ctype_analysis_only=False")
    sys.exit(0)

# DETERMINE TASK TYPE (needs to happen before data module creation)

# Determine model_set_option and whether to append _risk suffix
# This needs to happen BEFORE creating data modules
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
    model_set_option = 'side_effects_risk'
    option = f'{option}_risk'
elif option in ['binary', 'binary_icu', 'binary_icu_pre']:
    # All binary task variants use model_set_option='binary'
    model_set_option = 'binary'
    # Don't append _risk suffix for these tasks - keep original option for data loading
else:
    model_set_option = option
    # Keep option as is for other tasks

# AUTOMATIC CANCER TYPE FILTERING BASED ON THRESHOLDS

print("\n" + "="*80)
print("STEP 1.5: COUNTING POSITIVE CASES BY CANCER TYPE")
print("="*80)
print(f"Creating data module to count positive cases...")

# Create a data module with all test patients to count positive cases
advanced_model = 'default'
all_test_split_dict = {
    'train': [],
    'val': [],
    'test': list(test_patient_ids)
}

# Create data module for counting (with corrected option value)
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

# Filter cancer types based on both thresholds
cancer_type_list = ['all']  # Always include 'all' category
filtered_ctypes = []

for ctype, stats in sorted_ctypes:
    num_patients = stats['num_unique_patients']
    num_positives = ctype_to_positive_count.get(ctype, 0)

    # Check if cancer type meets both thresholds
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

# MODEL PATHS AND SETUP

all_model_path = sparc_paths.SPARC_MODEL_ROOT

# Set up model folder paths based on dataset and task type
# Determine folder prefix based on task: 'hybrid' for FN/N/A, 'risk' for AKI/LT
# Note: option may have _risk suffix at this point (added at line 577)
base_option = option.replace('_risk', '')
if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
    folder_prefix = 'hybrid'
elif base_option in ['AKI', 'Liver_toxicity']:
    folder_prefix = 'risk'
elif option in ['binary', 'binary_icu', 'binary_icu_pre']:
    folder_prefix = ''  # No prefix for binary tasks
else:
    folder_prefix = ''  # Default no prefix

if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
    if model_set_option == 'side_effects_risk':
        # Side-effect tasks use hybrid or risk prefix based on task type
        model_folder_dict = {
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{folder_prefix}_{base_option}_20tp_topchemos_att_modular'
        }
    else:
        # Binary tasks use standard naming
        model_folder_dict = {
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{option}_20tp_topchemos_att_modular'
        }
else:
    raise ValueError(f"Dataset {dataset} not yet supported for 060726 models")

_ensemble_suffix = '_ensemble' if ensemble else ''
save_name = f'{sparc_paths.SPARC_MODEL_ROOT}/results_vis_060726/eval_across_ctypes/model_performance_dict_across_ctypes_{modules}_{option}_{dataset}_060726{_ensemble_suffix}.csv'

# Generate the split dict (already done by load_dataset for CAIA)
print(f"Using split_dict from load_dataset for {dataset}")
print(f"  Test set size: {len(split_dict['test'])}")

model_performance_dict = {}

# Check if the save csv file exists, if it does, load it
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

        # Load both side-effect and binary metrics (whichever are present)
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

        # Add metrics to the appropriate model and validation fold
        if model_name not in model_performance_dict[ctype_str]:
            model_performance_dict[ctype_str][model_name] = {}

        model_performance_dict[ctype_str][model_name][val_fold] = metrics
        model_performance_dict[ctype_str]['SPARC-P-all'][val_fold] = metrics

    print("Loaded existing model performance data from csv")

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Include columns for both side-effect and binary tasks
all_columns = ['cancer_type', 'model', 'val_fold',
               'c_index_overall', 'c_index_weighted', 'auroc_nextinf_overall', 'auroc_nextinf_weighted',
               'auroc', 'auprc',
               'unique_num_patients', 'unique_positives']

# MODEL EVALUATION LOOP

for ctype in cancer_type_list:
    print(f"\n{'='*80}")
    print(f"Processing cancer type: {ctype}")
    print(f"{'='*80}")

    ctype_str = ctype

    # Check if dictionary exists and has the required structure before checking folds
    if ctype_str in model_performance_dict and 'SPARC-P-all' in model_performance_dict[ctype_str]:
        all_folds_exist = all(fold in model_performance_dict[ctype_str]['SPARC-P-all'] for fold in all_folds)
        if all_folds_exist:
            print(f"Data module for cancer type {ctype_str} already exists for all folds, skipping")
            continue

    pl.seed_everything(rseed)

    advanced_model = 'default'

    # Filter patients by cancer type
    if ctype == 'all':
        # Use ALL test patients (including those without cancer type data)
        # This ensures consistency with eval_across_meds and eval_across_models
        filtered_patient_ids = list(test_patient_ids)
    else:
        # Filter to patients with this specific cancer type (supports double-counting)
        # NOTE: get_patient_cancer_type now returns a LIST of cancer types
        filtered_patient_ids = []
        for pid in test_patient_ids:
            patient_ctypes = get_patient_cancer_type(pid, ctype_dict)
            if patient_ctypes is None:
                continue

            # Check if ANY of the patient's cancer types match the target
            for patient_ctype in patient_ctypes:
                collapsed_ctype = collapse_cancer_types(patient_ctype)
                if collapsed_ctype == ctype:
                    filtered_patient_ids.append(pid)
                    break  # Only add patient once even if multiple types match

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

    # Create modified split_dict with only filtered patients
    filtered_split_dict = {
        'train': [],
        'val': [],
        'test': filtered_patient_ids
    }

    # Create data module with filtered patients
    data_module = FlexibleFeatureDataModule(data_dict_lab_dict, filtered_split_dict, dataset = dataset, component = component,
            batch_size = 512, max_time_len = 20, num_workers=1,
            task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
            advanced_model = advanced_model, control_lack_confounds = False, side_effect_censoring = False,
            include_infusion_dates = include_infusion_dates, premed_only = False, competing_risk_death_check = False,
            anc=('CAIA' in dataset))

    print("data module initiated")

    # Guard: test_dataloader() crashes on empty datasets; check length first
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
    # unique_positives = unique patients who ever had the event.
    pos_patient_ids = set()
    for i in range(len(data_module.test_dataset.dataset_DMP_IDs)):
        # Extract label (handle both side-effect and binary tasks)
        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
            label = data_module.test_dataset.ys[i][0]  # Side-effect tasks: (event, time_to_event)
        else:
            label = data_module.test_dataset.ys[i]  # Binary tasks: single value
        if label == 1:
            pos_patient_ids.add(data_module.test_dataset.dataset_DMP_IDs[i])
    unique_positives = len(pos_patient_ids)
    print("unique_num_patients", unique_num_patients, "unique_positives", unique_positives)

    # Initialize the dictionary structure properly
    if ctype_str not in model_performance_dict:
        model_performance_dict[ctype_str] = {}

    # Check performance - only evaluating SPARC-P on each cancer type
    model_option = 'SPARC-P'

    # Initialize model_option in dict if not exists
    if model_option not in model_performance_dict[ctype_str]:
        model_performance_dict[ctype_str][model_option] = {}

    for val_fold in all_folds:
        pl.seed_everything(rseed)

        # Check if performance already exists BEFORE initializing
        if val_fold in model_performance_dict[ctype_str][model_option] and \
           len(model_performance_dict[ctype_str][model_option][val_fold]) > 0:
            print(f"Performance for cancer type {ctype_str} and model {model_option} fold {val_fold} already exists, skipping")
            continue

        # Initialize the val_fold dictionary
        model_performance_dict[ctype_str][model_option][val_fold] = {}

        print(f"Processing {model_option} for fold {val_fold}")

        # Get model path
        model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        print(model_folder)
        # if not exists, skip
        if not os.path.exists(model_folder):
            print(f"Model folder {model_folder} does not exist, skipping")
            continue
        existing_files = os.listdir(model_folder)

        # Find the checkpoint that has the highest val_loss (i.e. the one that is not last.ckpt)
        ## check if 'epoch' in the filename
        if not any(['epoch' in el for el in existing_files]):
            print(f"No epoch checkpoint found in {model_folder}, skipping")
            continue
        optimal_model_path = [el for el in existing_files if 'epoch' in el][0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(model_option, "optimal_model_path", optimal_model_path)

        # Build module index configuration for 060726 models
        module_index_cfg = build_module_index_config(
            modular_featuredim_dict, advanced_model='default'
        )
        n_static_features = module_index_cfg['n_static']

        # Determine loss option based on task type
        if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
            loss_option = 'hybrid'
        elif base_option in ['AKI', 'Liver_toxicity']:
            loss_option = 'default'
        else:
            loss_option = 'default'

        # Loading the 060726 model with updated parameters
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
            # 060726-specific parameters
            module_index_cfg=module_index_cfg,
            use_module_tokenizer=True,
            use_continuous_time=True,
            n_static_features=n_static_features,
            post_mlp_dropout=0.2,
            loss_option=loss_option
        )

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model {model_option} on device: {device}")

        # Get test set predictions - collect all batches first, then compute metrics
        pl.seed_everything(rseed)

        # Collect all predictions and ground truths from the entire test dataset
        all_y_preds = []
        all_y_events = []
        all_time_to_events = []
        all_num_infusions = []

        batch_count = 0
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # SPARC models with data_prep
                if include_gcsf:
                    xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                else:
                    xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                    gcsf_groundtruth = None

                # Move prepared data to GPU
                xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                if add_num_infusion and num_infusion is not None:
                    num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                # Get predictions
                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Extract ground truth - ensure proper tensor dimensions
                if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                    # Side-effect tasks: y is a tuple (event, time_to_event)
                    fail_indicators = torch.as_tensor(y[0], device=device)
                    time_to_event = torch.as_tensor(y[1], device=device)
                else:
                    # Binary tasks: y is a single tensor, set time_to_event to all 1s
                    fail_indicators = torch.as_tensor(y, device=device)
                    time_to_event = torch.ones_like(fail_indicators, device=device).float()

                # Ensure tensors are at least 1D
                if fail_indicators.dim() == 0:
                    fail_indicators = fail_indicators.unsqueeze(0)
                if time_to_event.dim() == 0:
                    time_to_event = time_to_event.unsqueeze(0)

                # Store batch results (move to CPU for concatenation to save GPU memory)
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

        # Concatenate all batches
        all_y_preds = torch.cat(all_y_preds, dim=0)
        all_y_events = torch.cat(all_y_events, dim=0)
        all_time_to_events = torch.cat(all_time_to_events, dim=0)
        all_num_infusions = torch.cat(all_num_infusions, dim=0)

        print(f"After concatenation - Total samples: {len(all_y_events)}, Positive: {(all_y_events == 1).sum().item()}")

        # Calculate metrics once on the entire dataset
        if len(all_y_events) > 0 and (all_y_events == 1).sum().item() > 0:
            # Determine task type and calculate appropriate metrics
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                # Side-effect tasks: use C-index and next-infusion AUROC
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
                # Binary tasks: directly compute AUROC/AUPRC on entire cohort
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
                auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()

                model_performance_dict[ctype_str][model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[ctype_str][model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")
        else:
            # No positive cases - set default values based on task type
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

        # Store predictions for ensemble if enabled
        if ensemble:
            model_performance_dict[ctype_str][model_option][val_fold]['y_preds'] = all_y_preds.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['y_events'] = all_y_events.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['time_to_events'] = all_time_to_events.cpu().numpy()
            model_performance_dict[ctype_str][model_option][val_fold]['num_infusions'] = all_num_infusions.cpu().numpy()

    # Compute ensemble predictions if enabled
    if ensemble:
        for ctype_str in model_performance_dict.keys():
            for model_option in [k for k in model_performance_dict[ctype_str].keys() if not k.endswith('_ensemble')]:
                # Check if all 10 folds have predictions
                all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                all_folds_complete = all(fold in model_performance_dict[ctype_str][model_option] and
                                        'y_preds' in model_performance_dict[ctype_str][model_option][fold]
                                        for fold in all_folds)

                if all_folds_complete:
                    print(f"\n{'='*80}")
                    print(f"Computing ensemble for {model_option} on {ctype_str}")
                    print(f"{'='*80}")

                    # Collect predictions from all folds
                    fold_preds = [model_performance_dict[ctype_str][model_option][fold]['y_preds'] for fold in all_folds]
                    fold_events = [model_performance_dict[ctype_str][model_option][fold]['y_events'] for fold in all_folds]
                    fold_times = [model_performance_dict[ctype_str][model_option][fold]['time_to_events'] for fold in all_folds]
                    fold_infusions = [model_performance_dict[ctype_str][model_option][fold]['num_infusions'] for fold in all_folds]

                    # Check that all folds have the same number of samples
                    assert len(set(len(p) for p in fold_preds)) == 1, "All folds should have same number of test samples"

                    # Compute mean ensemble predictions
                    ensemble_preds = np.mean(fold_preds, axis=0)
                    ensemble_events = fold_events[0]  # Same across folds
                    ensemble_times = fold_times[0]  # Same across folds
                    ensemble_infusions = fold_infusions[0]  # Same across folds

                    # Convert to tensors
                    import torch
                    ensemble_preds_tensor = torch.tensor(ensemble_preds, dtype=torch.float32)
                    ensemble_events_tensor = torch.tensor(ensemble_events, dtype=torch.float32)
                    ensemble_times_tensor = torch.tensor(ensemble_times, dtype=torch.float32)
                    ensemble_infusions_tensor = torch.tensor(ensemble_infusions, dtype=torch.float32)

                    # Evaluate ensemble
                    if len(ensemble_events_tensor) > 0 and (ensemble_events_tensor == 1).sum().item() > 0:
                        # Determine task type
                        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                            # Side-effect tasks
                            c_idx_dict = calculate_weighted_c_index_by_infusion(
                                ensemble_preds_tensor, ensemble_events_tensor, ensemble_times_tensor, ensemble_infusions_tensor,
                                mode='val', option=option
                            )

                            # Count unique patients and positives
                            unique_patients = len(set([fold_infusions[0][i] for i in range(len(fold_infusions[0]))]))
                            unique_positives = (ensemble_events_tensor == 1).sum().item()

                            # Store ensemble results
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

                            print(f"Ensemble C-index overall: {c_idx_dict.get('c_index_overall', 0.5):.4f}, "
                                  f"C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5):.4f}")
                            print(f"Ensemble AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
                        else:
                            # Binary tasks
                            from torchmetrics import AUROC, AveragePrecision
                            auroc_fn = AUROC(task='binary')
                            auprc_fn = AveragePrecision(task='binary')

                            auroc_score = auroc_fn(ensemble_preds_tensor, ensemble_events_tensor.int()).item()
                            auprc_score = auprc_fn(ensemble_preds_tensor, ensemble_events_tensor.int()).item()

                            # Store ensemble results
                            ensemble_name = f'{model_option}_ensemble'
                            if ensemble_name not in model_performance_dict[ctype_str]:
                                model_performance_dict[ctype_str][ensemble_name] = {}

                            model_performance_dict[ctype_str][ensemble_name]['ensemble'] = {
                                'auroc': auroc_score,
                                'auprc': auprc_score,
                                'unique_num_patients': len(set([fold_infusions[0][i] for i in range(len(fold_infusions[0]))])),
                                'unique_positives': (ensemble_events_tensor == 1).sum().item(),
                            }

                            print(f"Ensemble AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")
                    print(f"{'='*80}\n")

    # Save to csv after each cancer type
    model_performance_df = pd.DataFrame(columns=all_columns)
    # Populate the dataframe from the dictionary
    for ctype_str_key in model_performance_dict.keys():
        for model_option_key in model_performance_dict[ctype_str_key].keys():
            for val_fold_key in model_performance_dict[ctype_str_key][model_option_key].keys():
                row = {
                    'cancer_type': ctype_str_key,
                    'model': model_option_key,
                    'val_fold': val_fold_key
                }
                # Only include metrics, not prediction arrays
                fold_data = {k: v for k, v in model_performance_dict[ctype_str_key][model_option_key][val_fold_key].items()
                            if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
                row.update(fold_data)
                row_values = [row.get(col, None) for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)
    model_performance_df = model_performance_df.drop_duplicates(subset=['cancer_type', 'model', 'val_fold'])
    model_performance_df.to_csv(save_name, index=False)

# Final save to csv
model_performance_df = pd.DataFrame(columns=all_columns)
for ctype_str in model_performance_dict.keys():
    for model_option in model_performance_dict[ctype_str].keys():
        for val_fold in model_performance_dict[ctype_str][model_option].keys():
            row = {
                'cancer_type': ctype_str,
                'model': model_option,
                'val_fold': val_fold
            }
            # Only include metrics, not prediction arrays
            fold_data = {k: v for k, v in model_performance_dict[ctype_str][model_option][val_fold].items()
                        if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
            row.update(fold_data)
            row_values = [row.get(col, None) for col in all_columns]
            model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)

# Remove duplicates and save
model_performance_df = model_performance_df.drop_duplicates(subset=['cancer_type', 'model', 'val_fold'])
model_performance_df.to_csv(save_name, index=False)

print("\nModel performance dict saved to csv")
print(f"Results saved to: {save_name}")
