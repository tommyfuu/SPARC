"""Model performance broken down by medication regimen, for one task.

With --ensemble (-en), the 10 folds' predictions are averaged and written as
'{model}_ensemble' with val_fold='ensemble'. The 'all' row uses every test
patient, matching the baseline in eval_1 and eval_3.

Usage:
    python eval_2_across_medications_comparison_ensemble.py -o AKI -en True
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

# REGIMEN ANALYSIS FUNCTIONS (adapted from across_se_meds_performance_regimen.py)

def check_if_fits_criteria(criteria_medication_name, current_medications, exclusion_meds=None):
    """
    Check if the current medications meet the criteria and do not contain exclusions.

    Uses case-insensitive substring matching:
    - A criteria med 'CISPLATIN' matches 'CISPLATIN', 'cisplatin 50 MG', etc.
    - Checks if criteria med (uppercased) is contained in current med (uppercased)

    Args:
        criteria_medication_name: List of medication criteria (list of lists)
        current_medications: Set of medication names
        exclusion_meds: List of medications that must NOT be present (optional)

    Returns:
        tuple: (fits_criteria, fits_meds)
    """
    fits_criteria = False
    fits_meds = []

    # Convert current medications to uppercase for case-insensitive matching
    current_meds_upper = [med.upper() for med in current_medications]

    # First check exclusions - if any exclusion med is present, reject immediately
    if exclusion_meds:
        for exclusion_med in exclusion_meds:
            exclusion_med_upper = exclusion_med.upper()
            # Check if exclusion med is contained in any current medication
            if any(exclusion_med_upper in current_med_upper for current_med_upper in current_meds_upper):
                return False, []

    # Then check if any of the criteria combinations are met
    for med_sublist in criteria_medication_name:
        # Check if all medications in this sublist are present (as substrings)
        all_present = True
        for criteria_med in med_sublist:
            criteria_med_upper = criteria_med.upper()
            # Check if this criteria med is contained in any current medication
            if not any(criteria_med_upper in current_med_upper for current_med_upper in current_meds_upper):
                all_present = False
                break

        if all_present:
            fits_criteria = True
            fits_meds.extend(med_sublist)

    return fits_criteria, list(set(fits_meds))


def analyze_medication_regimens(postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_deidentified,
                               criteria_medication_name,
                               include_infusion_dates=True):
    """
    Analyze medication regimens taken by patients on each infusion day.

    Args:
        postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_deidentified: Dict mapping DMP_ID to dates to medications
        criteria_medication_name: List of medication criteria (list of lists)
        include_infusion_dates: Whether to include infusion dates in analysis

    Returns:
        dict: Dictionary with regimen statistics
    """

    # Dictionary to store regimen information
    regimen_stats = {}
    patient_regimen_pairs = []  # Store (patient_id, date, regimen) tuples

    for dmp_id, date_to_medications in postbc_DMP_ID_to_DATES_to_nonoral_drug_cats_deidentified.items():
        for infusion_date, medications in date_to_medications.items():
            # Get all medication names for this date
            current_medications = set([medication[0] for medication in medications])

            # Check if this date meets the criteria (has required medications)
            fits_criteria, fits_meds = check_if_fits_criteria(criteria_medication_name, current_medications)

            if fits_criteria:
                # Create regimen as sorted tuple of all medications taken on this date
                regimen = tuple(sorted(current_medications))

                # Store patient-infusion pair
                patient_regimen_pairs.append((dmp_id, infusion_date, regimen))

                # Update regimen statistics
                if regimen not in regimen_stats:
                    regimen_stats[regimen] = {
                        'unique_patients': set(),
                        'patient_infusion_pairs': set()
                    }

                regimen_stats[regimen]['unique_patients'].add(dmp_id)
                regimen_stats[regimen]['patient_infusion_pairs'].add((dmp_id, infusion_date))

    # Convert sets to counts for final output
    final_regimen_stats = {}
    for regimen, stats in regimen_stats.items():
        final_regimen_stats[regimen] = {
            'num_unique_patients': len(stats['unique_patients']),
            'num_patient_infusion_pairs': len(stats['patient_infusion_pairs']),
            'medications': list(regimen)
        }

    return final_regimen_stats, patient_regimen_pairs


def print_regimen_analysis(regimen_stats, regimen_stats_all=None):
    """
    Print a formatted analysis of medication regimens

    Args:
        regimen_stats: Statistics for test set
        regimen_stats_all: Statistics for entire dataset (optional)
    """
    print("=" * 80)
    print("MEDICATION REGIMEN ANALYSIS")
    print("=" * 80)

    # Sort regimens by number of patient-infusion pairs (descending)
    sorted_regimens = sorted(regimen_stats.items(),
                           key=lambda x: x[1]['num_patient_infusion_pairs'],
                           reverse=True)

    total_regimens = len(sorted_regimens)
    total_patient_infusion_pairs = sum([stats['num_patient_infusion_pairs']
                                      for _, stats in sorted_regimens])

    print(f"Total number of unique regimens (Test Set): {total_regimens}")
    print(f"Total patient-infusion pairs (Test Set): {total_patient_infusion_pairs}")

    if regimen_stats_all:
        total_regimens_all = len(regimen_stats_all)
        total_patient_infusion_pairs_all = sum([stats['num_patient_infusion_pairs']
                                               for _, stats in regimen_stats_all.items()])
        print(f"Total number of unique regimens (Entire Dataset): {total_regimens_all}")
        print(f"Total patient-infusion pairs (Entire Dataset): {total_patient_infusion_pairs_all}")

    print("\n" + "=" * 80)

    for i, (regimen, stats) in enumerate(sorted_regimens[:20], 1):  # Show top 20
        print(f"\nRegimen #{i}:")
        print(f"  Medications: {', '.join(stats['medications'])}")

        if regimen_stats_all and regimen in regimen_stats_all:
            stats_all = regimen_stats_all[regimen]
            print(f"  Test Set: {stats['num_unique_patients']} patients, {stats['num_patient_infusion_pairs']} pairs")
            print(f"  Entire Dataset: {stats_all['num_unique_patients']} patients, {stats_all['num_patient_infusion_pairs']} pairs")

            # Calculate percentages
            pct_patients = (stats['num_unique_patients'] / stats_all['num_unique_patients'] * 100) if stats_all['num_unique_patients'] > 0 else 0
            pct_pairs = (stats['num_patient_infusion_pairs'] / stats_all['num_patient_infusion_pairs'] * 100) if stats_all['num_patient_infusion_pairs'] > 0 else 0
            print(f"  Test coverage: {pct_patients:.1f}% patients, {pct_pairs:.1f}% pairs")
        else:
            print(f"  Unique patients: {stats['num_unique_patients']}")
            print(f"  Patient-infusion pairs: {stats['num_patient_infusion_pairs']}")
            print(f"  Avg infusions per patient: {stats['num_patient_infusion_pairs'] / stats['num_unique_patients']:.2f}")

    if total_regimens > 20:
        print(f"\n... and {total_regimens - 20} more regimens")

    print("\n" + "=" * 80)
    print("REGIMEN SIZE DISTRIBUTION (Test Set)")
    print("=" * 80)

    # Analyze regimen sizes
    size_distribution = {}
    for regimen, stats in sorted_regimens:
        size = len(stats['medications'])
        if size not in size_distribution:
            size_distribution[size] = {'count': 0, 'patients': 0, 'pairs': 0}
        size_distribution[size]['count'] += 1
        size_distribution[size]['patients'] += stats['num_unique_patients']
        size_distribution[size]['pairs'] += stats['num_patient_infusion_pairs']

    for size in sorted(size_distribution.keys()):
        stats = size_distribution[size]
        print(f"Regimens with {size} medication(s): {stats['count']} regimens, "
              f"{stats['patients']} patients, {stats['pairs']} patient-infusion pairs")

    return sorted_regimens


# COMMAND LINE ARGUMENTS

import argparse
parser = argparse.ArgumentParser(
            prog='eval_across_medications_comparison',
            description='calculate the model performance comparison across medications',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-m", "--modules", type=str, default = 'practical', help='module category, either "all", "allnonICD" or connect categories by dashes such as')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset used in training, options: MSK, MSK_timeshift, stanfordEHRSHOT')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-rt", "--retrain", type=str, default = 'False', help='whether to retrain the model or not, options: True, False; only legit for AKI_2wks for the MIMICIV_hosp dataset')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-ra", "--regimen_analysis_only", type=str, default = "True", help="whether to only run regimen analysis without model evaluation")
parser.add_argument("-en", "--ensemble", type=str, default='True', help='whether to compute ensemble predictions by averaging over 10 folds for each model-medication combination, options: True, False')

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
regimen_analysis_only = args.regimen_analysis_only.lower() == 'true'
ensemble = args.ensemble.lower() == 'true'

# Default settings for testing
rseed = 1024
option = 'AKI'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID"
add_num_infusion = True
regimen_analysis_only = False  # Set to True for first run to see common regimens

# Default settings for testing
rseed = 1024
option = 'febrile_neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
regimen_analysis_only = False  # Set to True for first run to see common regimens

# Default settings for testing
rseed = 1024
option = 'anemia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
regimen_analysis_only = False  # Set to True for first run to see common regimens

# Default settings for testing
rseed = 1024
option = 'neutropenia'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
add_num_infusion = True
regimen_analysis_only = False  # Set to True for first run to see common regimens


# Default settings for testing
rseed = 1024
option = 'Liver_toxicity'
days = 30
modules = 'parsimonious'
dataset = 'MSK_CAIA'
med_of_choice = "IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB"
add_num_infusion = True
regimen_analysis_only = False  # Set to True for first run to see common regimens

if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
    raise ValueError(f"Invalid option: {option}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")


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
# med_regimen_eval=True will load the unfiltered meds dict (all meds taken on same day)
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
    med_regimen_eval=True,  # Use unfiltered meds (all meds on same day) for regimen analysis
)

print("modular_featuredim_dict", modular_featuredim_dict)
print("Data loading complete")

# LOAD UNFILTERED MEDICATION DATA FOR REGIMEN ANALYSIS

print("\nCheckpoint 2: Using unfiltered medication data for regimen analysis")

# The unfiltered meds dict is now loaded directly by dataset_loading.py when med_regimen_eval=True
# It contains ALL medications taken on the same day, not just the high-risk ones
DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_unfiltered = data_dict_lab_dict['DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT']
print(f"Loaded medication data for {len(DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_unfiltered)} patients")

# REGIMEN ANALYSIS - Print common regimens for manual selection

print("\n" + "="*80)
print("STEP 1: REGIMEN ANALYSIS")
print("="*80)
print("Analyzing common medication regimens for manual selection...")
print("")

# Perform regimen analysis on test set patients only
test_patient_ids = set(split_dict['test'])
DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_test = {
    pid: dates for pid, dates in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_unfiltered.items()
    if pid in test_patient_ids
}

regimen_stats, patient_regimen_pairs = analyze_medication_regimens(
    DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_test,
    criteria_medication_name=med_of_choice
)
regimen_stats_all, patient_regimen_pairs_all = analyze_medication_regimens(
    DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_unfiltered,
    criteria_medication_name=med_of_choice
)

sorted_regimens = print_regimen_analysis(regimen_stats, regimen_stats_all)

# Print top 20 most common regimens in a clean format for easy copying
print("\n" + "="*80)
print("TOP 20 MOST COMMON MEDICATION REGIMENS (for manual selection)")
print("="*80)
print("\nCopy the regimens you want to analyze and add them to med_sublist_list below:")
print("")

top_regimen_dict = {}
for i, (regimen, stats) in enumerate(sorted_regimens[:50], 1):
    top_regimen_dict[regimen] = {
        'num_unique_patients': stats['num_unique_patients'],
        'num_patient_infusion_pairs': stats['num_patient_infusion_pairs']
    }
    print(f"{i:2d}. [{', '.join(repr(med) for med in stats['medications'])}]")

    # Show both test set and entire dataset statistics
    if regimen in regimen_stats_all:
        stats_all = regimen_stats_all[regimen]
        print(f"    Test: {stats['num_unique_patients']} patients, {stats['num_patient_infusion_pairs']} pairs | "
              f"Entire: {stats_all['num_unique_patients']} patients, {stats_all['num_patient_infusion_pairs']} pairs")

        # Calculate percentages
        pct_patients = (stats['num_unique_patients'] / stats_all['num_unique_patients'] * 100) if stats_all['num_unique_patients'] > 0 else 0
        pct_pairs = (stats['num_patient_infusion_pairs'] / stats_all['num_patient_infusion_pairs'] * 100) if stats_all['num_patient_infusion_pairs'] > 0 else 0
        print(f"    Coverage: {pct_patients:.1f}% patients, {pct_pairs:.1f}% pairs")
    else:
        print(f"    Test: {stats['num_unique_patients']} patients, {stats['num_patient_infusion_pairs']} pairs")

print("\n" + "="*80)
print("INSTRUCTIONS FOR NEXT STEP:")
print("="*80)
print("1. Review the regimens above")
print("2. Manually edit this script to add selected regimens to med_sublist_list")
print("3. Set regimen_analysis_only = False")
print("4. Re-run the script to perform model evaluation on selected regimens")
print("="*80)

# check for all the forms of medications in med_of_choice that can have appeared as substrings of patient medications in the test set
print("\nChecking for presence of medications of choice in patient medication lists...")
for med_criteria in med_of_choice:
    med_criteria_upper = med_criteria[0].upper()
    found_in_patients = False
    included_forms = set()  # To track which forms of the medication are found (e.g., 'CISPLATIN', 'CISPLATIN 50 MG')
    for dmp_id, date_to_meds in DMP_ID_to_DATE_to_NON_ORAL_MEDS_DICT_test.items():
        for date, meds in date_to_meds.items():
            meds_upper = [med[0].upper() for med in meds]  # Assuming meds is a list of (med_name, other_info)
            # check for substrings that match the criteria medication
            if any(med_criteria_upper in med_upper for med_upper in meds_upper):
                included_forms.update([med_upper for med_upper in meds_upper if med_criteria_upper in med_upper])
                found_in_patients = True
    print(f"Medication criteria '{med_criteria}' found in patient medications in forms such as: {', '.join(included_forms)}") if found_in_patients else print(f"Medication criteria '{med_criteria}' NOT found in any patient medications")
    
# Exit here if only running regimen analysis
if regimen_analysis_only:
    print("\nRegimen analysis complete. Exiting without model evaluation.")
    print("Edit the script to add your selected regimens and set regimen_analysis_only=False")
    sys.exit(0)

# MANUAL REGIMEN SELECTION - Edit this section after reviewing regimen analysis

# TODO: After reviewing the regimen analysis output above, manually add the
# regimens you want to analyze here. Example format:

aki_med_group = {

    # --- BISPHOSPHONATES BASED ---
    # Zoledronic acid alone: 593 pairs; pamidronate alone: 75 — merge into one monotherapy group + other interactions
    'bisphosphonate': (
        [['ZOLEDRONIC ACID'], ['PAMIDRONATE']],
        []  
    ),

    # --- CISPLATIN MONOTHERAPY ---
    # entries 7,10,15,29,40,44,48,49 = ~343 pairs
    'monotherapy-cisplatin': (
        [['CISPLATIN']],
        # comprehensive — exclude all known cisplatin combination partners
        ['CARBOPLATIN', 'OXALIPLATIN', 'GEMCITABINE', 'PEMETREXED',
         'FLUOROURACIL', 'CAPECITABINE', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'ETOPOSIDE', 'IRINOTECAN', 'TOPOTECAN',
         'VINCRISTINE', 'VINBLASTINE', 'VINORELBINE', 'CYCLOPHOSPHAMIDE',
         'DOXORUBICIN', 'EPIRUBICIN', 'BLEOMYCIN', 'METHOTREXATE',
         'IFOSFAMIDE', 'TEMOZOLOMIDE', 'MITOMYCIN', 'CETUXIMAB',
         'BEVACIZUMAB', 'RAMUCIRUMAB', 'NECITUMUMAB', 'TRASTUZUMAB',
         'PEMBROLIZUMAB', 'NIVOLUMAB', 'DURVALUMAB', 'ATEZOLIZUMAB',
         'CEMIPLIMAB', 'IPILIMUMAB', 'TREMELIMUMAB']
    ),

    # --- CISPLATIN + X ---
    # entries 3,4,12,16,19,27,36,39 = ~509 pairs
    'cisplatin+gemcitabine': (
        [['CISPLATIN', 'GEMCITABINE']],
        ['CARBOPLATIN', 'OXALIPLATIN']  # only exclude other platinums
    ),

    # --- OTHER PLATINUM + X ---
    # entries 20,30,31,34,35,41,46 = ~197 pairs
    'carboplatin+pemetrexed': (
        [['CARBOPLATIN', 'PEMETREXED']],
        ['CISPLATIN', 'OXALIPLATIN']  # only exclude other platinums
    ),

    # --- OTHER COMBOS ---
    # entries 6,8,14,17,18,24,25,32,33,38 = ~560 pairs
    'cmf': (
        [['CYCLOPHOSPHAMIDE', 'METHOTREXATE', 'FLUOROURACIL']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN', 'DOXORUBICIN', 'EPIRUBICIN',
         'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'GEMCITABINE', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'VINCRISTINE', 'VINBLASTINE', 'VINORELBINE',
         'BEVACIZUMAB', 'TRASTUZUMAB', 'PEMBROLIZUMAB', 'RITUXIMAB']
    ),

    # entries 5,11,13,21,26,37,42,50 = ~470 pairs
    # pemetrexed without any platinum — covers monotherapy, +bevacizumab, +pembrolizumab
    'pemetrexed-based': (
        [['PEMETREXED']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN']  # only exclude platinums
    ),
}

heme_med_group = {

    # --- PLATINUM + GEMCITABINE ---
    # combos 4,5,21,22,30,34,35,40 = ~5435 pairs
    # mirrors heme_side_effect_med_groups 'platinum+gemcitabine' exactly
    'platinum+gemcitabine': (
        [['CISPLATIN', 'GEMCITABINE'], ['CARBOPLATIN', 'GEMCITABINE']],
        ['OXALIPLATIN', 'PACLITAXEL', 'DOCETAXEL', 'PEMETREXED',
         'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE', 'VINBLASTINE',
         'BEVACIZUMAB', 'PEMBROLIZUMAB', 'NIVOLUMAB']
    ),

    # --- CISPLATIN + PEMETREXED ---
    # absent from top 50 but retained for clinical validity; broadened to carboplatin given combo 44
    # mirrors heme_side_effect_med_groups 'cisplatin+pemetrexed', broadened to platinum+pemetrexed
    'platinum+pemetrexed': (
        [['CISPLATIN', 'PEMETREXED'], ['CARBOPLATIN', 'PEMETREXED']],
        ['OXALIPLATIN', 'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL',
         'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'BEVACIZUMAB']
    ),

    # --- 5-FU BASED ---
    # combos 8,10,24,25,31,32,33,42 = ~4729 pairs
    # mirrors heme_side_effect_med_groups '5-FU-based' exactly — must contain all three, bevacizumab allowed
    '5fu-based': (
        [['FLUOROURACIL', 'LEUCOVORIN', 'IRINOTECAN']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'ETOPOSIDE', 'TOPOTECAN', 'GEMCITABINE',
         'PACLITAXEL', 'DOCETAXEL', 'PEMETREXED',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN']
    ),

    # --- MONOTHERAPY PLATINUM ---
    # pure cisplatin or carboplatin absent from top 50 — every platinum is paired with chemo
    # retained for clinical validity with comprehensive exclusions
    # mirrors heme_side_effect_med_groups 'monotherapy platinum'
    'monotherapy-platinum': (
        [['CISPLATIN'], ['CARBOPLATIN'], ['OXALIPLATIN']],
        ['OXALIPLATIN', 'GEMCITABINE', 'PEMETREXED',
         'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'FLUOROURACIL', 'CAPECITABINE', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'EPIRUBICIN',
         'VINCRISTINE', 'VINBLASTINE', 'VINORELBINE',
         'BEVACIZUMAB', 'RAMUCIRUMAB', 'PEMBROLIZUMAB', 'NIVOLUMAB',
         'DURVALUMAB', 'ATEZOLIZUMAB', 'CETUXIMAB', 'TRASTUZUMAB']
    ),

    # --- MONOTHERAPY TOPOISOMERASE ---
    # combos 1,7,11: ETOPOSIDE alone or +pegfilgrastim (~1527 pairs)
    # combos 7,49: TOPOTECAN alone or +bevacizumab (~1426 pairs)
    # mirrors heme_side_effect_med_groups 'monotherapy topoisomerase', drops irinotecan
    'monotherapy-topoisomerase': (
        [['ETOPOSIDE'], ['TOPOTECAN'], ['IRINOTECAN']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'FLUOROURACIL', 'IRINOTECAN', 'GEMCITABINE',
         'PACLITAXEL', 'DOCETAXEL', 'PEMETREXED',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE']
    ),

}

lt_med_groups = {

    # --- MONOTHERAPY PD-1 ---
    # combos 1,2,6,22,33: NIVOLUMAB or PEMBROLIZUMAB alone (+saline/supportive only)
    # dominant signal — ~47197 pairs combined
    'monotherapy-pd1': (
        [['NIVOLUMAB'], ['PEMBROLIZUMAB']],
        ['IPILIMUMAB', 'CTLA4',  # no dual checkpoint
         'ATEZOLIZUMAB', 'DURVALUMAB', 'AVELUMAB',  # no other checkpoints
         'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'TOPOTECAN', 'CYCLOPHOSPHAMIDE',
         'DOXORUBICIN', 'BEVACIZUMAB', 'RAMUCIRUMAB',
         'CETUXIMAB', 'TRASTUZUMAB', 'ZOLEDRONIC ACID']
    ),

    # --- MONOTHERAPY PD-L1 ---
    # combos 3,5,25,34: ATEZOLIZUMAB, CEMIPLIMAB, AVELUMAB alone
    # ~5441 pairs combined
    'monotherapy-pdl1': (
        [['ATEZOLIZUMAB'], ['CEMIPLIMAB'], ['AVELUMAB'], ['DURVALUMAB']],
        ['NIVOLUMAB', 'PEMBROLIZUMAB', 'IPILIMUMAB',  # no other checkpoints
         'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'BEVACIZUMAB', 'CARBOPLATIN']
    ),

    # --- MONOTHERAPY CTLA-4 ---
    # combos 4,12: IPILIMUMAB or TREMELIMUMAB alone (IPILIMUMAB has 296 pairs)
    'monotherapy-ctla4': (
        [['IPILIMUMAB'], ['TREMELIMUMAB']],
        ['NIVOLUMAB', 'PEMBROLIZUMAB', 'ATEZOLIZUMAB',
        'DURVALUMAB', 'AVELUMAB', 'CEMIPLIMAB',  # no other checkpoints
        'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
        'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
        'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE',
        'BEVACIZUMAB', 'CETUXIMAB', 'TRASTUZUMAB']
    ),


    # --- DUAL CHECKPOINT ---
    # combo 4,12: IPILIMUMAB + NIVOLUMAB (~2721 pairs); IPILIMUMAB alone (~296 pairs)
    # ipilimumab monotherapy retained here as CTLA-4 mechanism distinct from PD-1
    'dual-checkpoint': (
        [['IPILIMUMAB', 'NIVOLUMAB'], ['IPILIMUMAB', 'PEMBROLIZUMAB'],
        ['TREMELIMUMAB', 'DURVALUMAB']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
        'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
        'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE',
        'BEVACIZUMAB', 'CETUXIMAB', 'TRASTUZUMAB']
    ),

    # --- CHECKPOINT + PEMETREXED ---
    # combos 7,10,11,14,16,26,29,35,50: pembrolizumab + pemetrexed +/- carboplatin
    # ~3363 pairs; vitamin B12/dexamethasone allowed as pemetrexed supportive care
    'checkpoint+pemetrexed': (
        [['PEMBROLIZUMAB', 'PEMETREXED'], ['NIVOLUMAB', 'PEMETREXED']],
        ['CISPLATIN', 'OXALIPLATIN',  # only exclude non-carboplatin platinums
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL',
         'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'BEVACIZUMAB']
    ),

    # --- CHECKPOINT + PLATINUM + ETOPOSIDE ---
    # combos 15,17: atezolizumab + carboplatin + etoposide (~506 pairs)
    # SCLC regimen — clinically distinct enough to isolate
    'checkpoint+platinum+etoposide': (
        [['ATEZOLIZUMAB', 'CARBOPLATIN', 'ETOPOSIDE'],
         ['DURVALUMAB', 'CARBOPLATIN', 'ETOPOSIDE'],
         ['DURVALUMAB', 'CISPLATIN', 'ETOPOSIDE']],
        ['GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PEMETREXED',
         'FLUOROURACIL', 'IRINOTECAN', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'BEVACIZUMAB']
    ),

    # --- CHECKPOINT + BEVACIZUMAB ---
    # combos 13,21,43,47: atezolizumab or pembrolizumab + bevacizumab/biosimilar (~584 pairs)
    'checkpoint+bevacizumab': (
        [['ATEZOLIZUMAB', 'BEVACIZUMAB'], ['PEMBROLIZUMAB', 'BEVACIZUMAB'],
         ['NIVOLUMAB', 'BEVACIZUMAB'], ['ATEZOLIZUMAB', 'BEVACIZUMAB-AWWB'],
         ['PEMBROLIZUMAB', 'BEVACIZUMAB-AWWB']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PEMETREXED',
         'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN']
    ),
}

lt_med_groups = {

    # --- MONOTHERAPY PD-1 ---
    # combos 1,2,6,22,33: NIVOLUMAB or PEMBROLIZUMAB alone (+saline/supportive only)
    # dominant signal — ~47197 pairs combined
    # DOSTARLIMAB and CEMIPLIMAB added to align with original lt_side_effect_med_groups
    'monotherapy-pd1': (
        [['NIVOLUMAB'], ['PEMBROLIZUMAB'], ['DOSTARLIMAB'], ['CEMIPLIMAB']],
        ['IPILIMUMAB', 'TREMELIMUMAB', 'RELATLIMAB',  # no CTLA-4 or LAG-3
         'ATEZOLIZUMAB', 'DURVALUMAB', 'AVELUMAB',  # no PD-L1
         'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'TOPOTECAN', 'CYCLOPHOSPHAMIDE',
         'DOXORUBICIN', 'BEVACIZUMAB', 'RAMUCIRUMAB',
         'CETUXIMAB', 'TRASTUZUMAB', 'ZOLEDRONIC ACID']
    ),

    # --- MONOTHERAPY PD-L1 ---
    # combos 3,5,25,34: ATEZOLIZUMAB, AVELUMAB, DURVALUMAB alone
    # ~5441 pairs combined
    'monotherapy-pdl1': (
        [['ATEZOLIZUMAB'], ['AVELUMAB'], ['DURVALUMAB']],
        ['NIVOLUMAB', 'PEMBROLIZUMAB', 'DOSTARLIMAB', 'CEMIPLIMAB',  # no PD-1
         'IPILIMUMAB', 'TREMELIMUMAB', 'RELATLIMAB',  # no CTLA-4 or LAG-3
         'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'TOPOTECAN', 'BEVACIZUMAB', 'RAMUCIRUMAB',
         'CETUXIMAB', 'TRASTUZUMAB']
    ),

    # --- MONOTHERAPY CTLA-4 ---
    # combo 12: IPILIMUMAB alone (~296 pairs); TREMELIMUMAB retained for completeness
    'monotherapy-ctla4': (
        [['IPILIMUMAB'], ['TREMELIMUMAB']],
        ['NIVOLUMAB', 'PEMBROLIZUMAB', 'DOSTARLIMAB', 'CEMIPLIMAB',
         'ATEZOLIZUMAB', 'DURVALUMAB', 'AVELUMAB', 'RELATLIMAB',  # no other checkpoints
         'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'TOPOTECAN', 'BEVACIZUMAB', 'RAMUCIRUMAB',
         'CETUXIMAB', 'TRASTUZUMAB']
    ),

    # --- DUAL CHECKPOINT ---
    # combo 4: IPILIMUMAB + NIVOLUMAB (~2721 pairs)
    # RELATLIMAB + NIVOLUMAB added to align with original dual-ici group (Opdualag, melanoma)
    # TREMELIMUMAB + DURVALUMAB added for completeness
    'dual-checkpoint': (
        [['IPILIMUMAB', 'NIVOLUMAB'], ['IPILIMUMAB', 'PEMBROLIZUMAB'],
         ['TREMELIMUMAB', 'DURVALUMAB'],
         ['NIVOLUMAB', 'RELATLIMAB']],  # LAG-3 + PD-1
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'PACLITAXEL', 'DOCETAXEL',
         'PACLITAXEL PROTEIN-BOUND', 'FLUOROURACIL', 'IRINOTECAN',
         'ETOPOSIDE', 'TOPOTECAN', 'BEVACIZUMAB', 'RAMUCIRUMAB',
         'CETUXIMAB', 'TRASTUZUMAB']
    ),

    # --- CHECKPOINT + PEMETREXED ---
    # combos 7,10,11,14,16,26,29,35,50: pembrolizumab + pemetrexed +/- carboplatin
    # ~3363 pairs; vitamin B12/dexamethasone allowed as pemetrexed supportive care
    'checkpoint+pemetrexed': (
        [['PEMBROLIZUMAB', 'PEMETREXED'], ['NIVOLUMAB', 'PEMETREXED']],
        ['CISPLATIN', 'OXALIPLATIN',  # carboplatin intentionally allowed
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'BEVACIZUMAB', 'RAMUCIRUMAB']
    ),

    # --- CHECKPOINT + PLATINUM + ETOPOSIDE ---
    # combos 15,17: atezolizumab + carboplatin + etoposide (~506 pairs)
    # SCLC regimen — clinically distinct enough to isolate
    'checkpoint+platinum+etoposide': (
        [['ATEZOLIZUMAB', 'CARBOPLATIN', 'ETOPOSIDE'],
         ['DURVALUMAB', 'CARBOPLATIN', 'ETOPOSIDE'],
         ['DURVALUMAB', 'CISPLATIN', 'ETOPOSIDE']],
        ['GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'PEMETREXED', 'FLUOROURACIL', 'IRINOTECAN', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'BEVACIZUMAB', 'RAMUCIRUMAB']
    ),

    # --- CHECKPOINT + BEVACIZUMAB ---
    # combos 13,21,43,47: atezolizumab or pembrolizumab + bevacizumab/biosimilar (~584 pairs)
    'checkpoint+bevacizumab': (
        [['ATEZOLIZUMAB', 'BEVACIZUMAB'], ['PEMBROLIZUMAB', 'BEVACIZUMAB'],
         ['NIVOLUMAB', 'BEVACIZUMAB'], ['ATEZOLIZUMAB', 'BEVACIZUMAB-AWWB'],
         ['PEMBROLIZUMAB', 'BEVACIZUMAB-AWWB']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'PEMETREXED', 'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'RAMUCIRUMAB', 'CETUXIMAB']
    ),

    # --- CHECKPOINT + TAXANE ---
    # combos 9,24,41: atezolizumab+nab-paclitaxel (TNBC), pembrolizumab+carboplatin+paclitaxel
    # taxanes themselves cause transaminase elevations — distinct LT profile
    # aligned with original ici+taxane, expanded to include nivolumab
    'checkpoint+taxane': (
        [['ATEZOLIZUMAB', 'PACLITAXEL'], ['ATEZOLIZUMAB', 'PACLITAXEL PROTEIN-BOUND'],
         ['ATEZOLIZUMAB', 'DOCETAXEL'],
         ['AVELUMAB', 'PACLITAXEL'], ['AVELUMAB', 'PACLITAXEL PROTEIN-BOUND'],
         ['AVELUMAB', 'DOCETAXEL'],
         ['PEMBROLIZUMAB', 'PACLITAXEL'], ['PEMBROLIZUMAB', 'PACLITAXEL PROTEIN-BOUND'],
         ['PEMBROLIZUMAB', 'DOCETAXEL'],
         ['NIVOLUMAB', 'PACLITAXEL'], ['NIVOLUMAB', 'PACLITAXEL PROTEIN-BOUND'],
         ['NIVOLUMAB', 'DOCETAXEL']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PEMETREXED', 'FLUOROURACIL',
         'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'BEVACIZUMAB', 'RAMUCIRUMAB', 'CETUXIMAB']
    ),

    # --- CHECKPOINT + FLUOROURACIL ---
    # combo 46: nivolumab + oxaliplatin + 5-FU (gastric/GEJ)
    # restored closer to original nivo+5-FU — requires FLUOROURACIL+LEUCOVORIN as backbone
    'checkpoint+fluorouracil': (
        [['NIVOLUMAB', 'FLUOROURACIL', 'LEUCOVORIN'],
         ['PEMBROLIZUMAB', 'FLUOROURACIL', 'LEUCOVORIN']],
        ['CISPLATIN', 'CARBOPLATIN',
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN', 'PEMETREXED',
         'BEVACIZUMAB', 'RAMUCIRUMAB', 'CETUXIMAB']
    ),

    # --- CHECKPOINT + TRASTUZUMAB ---
    # combo 48: pembrolizumab + trastuzumab-anns (Kanjinti) — HER2+ gastric
    'checkpoint+trastuzumab': (
        [['PEMBROLIZUMAB', 'TRASTUZUMAB'], ['NIVOLUMAB', 'TRASTUZUMAB'],
         ['PEMBROLIZUMAB', 'TRASTUZUMAB-ANNS'],
         ['PEMBROLIZUMAB', 'TRASTUZUMAB-DKST'], ['NIVOLUMAB', 'TRASTUZUMAB-DKST']],
        ['CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN',
         'GEMCITABINE', 'PACLITAXEL', 'DOCETAXEL', 'PACLITAXEL PROTEIN-BOUND',
         'PEMETREXED', 'FLUOROURACIL', 'IRINOTECAN', 'ETOPOSIDE', 'TOPOTECAN',
         'BEVACIZUMAB', 'RAMUCIRUMAB', 'CETUXIMAB']
    ),
}

side_effect_to_med_groups = {
    'anemia': heme_med_group,
    'neutropenia': heme_med_group,
    'febrile_neutropenia': heme_med_group,
    'AKI': aki_med_group,
    'Liver_toxicity': lt_med_groups
}

# For now, just analyze all medications combined
med_sublist_list = [('all', med_of_choice, [])]
for med_group_name, (med_combos, exclusion_meds) in side_effect_to_med_groups[option].items():
    med_sublist_list.append((med_group_name, med_combos, exclusion_meds))

print("\n" + "="*80)
print("STEP 2: MODEL EVALUATION ON SELECTED REGIMENS")
print("="*80)

# MODEL PATHS AND SETUP

all_model_path = sparc_paths.SPARC_MODEL_ROOT

# Set up model folder paths based on dataset and task type
# Determine folder prefix based on task: 'hybrid' for FN/N/A, 'risk' for AKI/LT
# Note: At this point, option doesn't have _risk suffix yet (added at line 964)
if option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
    # Hybrid tasks use 'hybrid_' prefix
    folder_prefix = 'hybrid'
    folder_option = option
elif option in ['AKI', 'Liver_toxicity']:
    # Pure risk tasks use 'risk_' prefix
    folder_prefix = 'risk'
    folder_option = option
else:
    raise ValueError(f"Unknown task option: {option}. Must be one of: febrile_neutropenia, neutropenia, anemia, AKI, Liver_toxicity")

if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift']:
    model_folder_dict = {
        'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{folder_prefix}_{folder_option}_20tp_topchemos_att_modular'
    }
else:
    raise ValueError(f"Dataset {dataset} not yet supported for 060726 models")

_ensemble_suffix = '_ensemble' if ensemble else ''
save_name = f'{sparc_paths.SPARC_MODEL_ROOT}/results_vis_060726/eval_across_meds/model_performance_dict_across_meds_{modules}_{option}_{dataset}_regimen_060726{_ensemble_suffix}.csv'


# Generate the split dict from fold dict (or use the one from load_dataset for CAIA)
if 'CAIA' in dataset:
    # For CAIA datasets, split_dict is already properly built by load_dataset()
    print(f"Using split_dict from load_dataset for {dataset}")
    print(f"  Test set size: {len(split_dict['test'])}")
elif dataset == 'MSK':
    with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as handle:
        fold_dict = pickle.load(handle)
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'MSK_timeshift':
    # MSK_timeshift uses all patients from age dict, excluding MSK patients
    fold_dict = {10: data_dict_lab_dict['age'].keys()}

    # Get all DMP_ID in the MSK dataset and exclude from the fold_dict
    data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
    with open(data_path, 'rb') as handle:
        og_set = pickle.load(handle)
    fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'stanfordEHRSHOT':
    # Just use everyone
    with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_basic_lab_values_list_dict_080525.pickle', 'rb') as handle:
        DMP_ID_to_DATES_to_TESTSVALUES_DICT = pickle.load(handle)
    fold_dict = {10: DMP_ID_to_DATES_to_TESTSVALUES_DICT.keys()}
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'MIMICIV_hosp':
    with open(f'{data_source_path}/{dataset}/split_dict_1v1v1_112425.pickle', 'rb') as handle:
        fold_dict_raw = pickle.load(handle)
    fold_dict = {}
    fold_dict[10] = fold_dict_raw['test']
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
else:
    raise ValueError(f"Unknown dataset: {dataset}")

model_performance_dict = {}

# Check if the save csv file exists, if it does, load it
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)

    for index, row in saved_df.iterrows():
        model_name = row['model']
        val_fold = row['val_fold']
        med_sublist_str = row['med_sublist']
        unique_num_patients = row['unique_num_patients']
        unique_positives = row['unique_positives']

        if med_sublist_str not in model_performance_dict:
            model_performance_dict[med_sublist_str] = {
                'SPARC-P-all': {},
                model_name: {}
            }

        metrics = {
            'c_index_overall': row.get('c_index_overall'),
            'c_index_weighted': row.get('c_index_weighted'),
            'auroc_nextinf_overall': row.get('auroc_nextinf_overall'),
            'auroc_nextinf_weighted': row.get('auroc_nextinf_weighted'),
            'unique_num_patients': unique_num_patients,
            'unique_positives': unique_positives,
        }

        # Add metrics to the appropriate model and validation fold
        if model_name not in model_performance_dict[med_sublist_str]:
            model_performance_dict[med_sublist_str][model_name] = {}

        model_performance_dict[med_sublist_str][model_name][val_fold] = metrics
        model_performance_dict[med_sublist_str]['SPARC-P-all'][val_fold] = metrics

print(model_performance_dict)


print("Loaded existing model performance data from csv")

model_set_option = 'side_effects_risk'
option = f'{option}_risk' if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)) else option
all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

all_columns = ['med_sublist', 'model', 'val_fold', 'c_index_overall', 'c_index_weighted', 'auroc_nextinf_overall', 'auroc_nextinf_weighted', 'unique_num_patients', 'unique_positives']


# MODEL EVALUATION LOOP

for idx, med_tuple in enumerate(med_sublist_list):
    # Unpack the tuple: (med_group_name, med_combos, exclusion_meds)
    if isinstance(med_tuple, tuple) and len(med_tuple) == 3:
        med_group_name, med_combos, exclusion_meds = med_tuple
    else:
        # Fallback for old format (shouldn't happen with new code)
        med_group_name = f"group_{idx}"
        med_combos = med_tuple
        exclusion_meds = []

    print(f"med_group_name: {med_group_name}, med_combos: {med_combos}, exclusion_meds: {exclusion_meds}")

    # Use med_group_name as the identifier in the results
    med_sublist_str = med_group_name

    # Check if dictionary exists and has the required structure before checking folds
    if med_sublist_str in model_performance_dict and 'SPARC-P-all' in model_performance_dict[med_sublist_str]:
        all_folds_exist = all(fold in model_performance_dict[med_sublist_str]['SPARC-P-all'] for fold in all_folds)
        if all_folds_exist:
            print(f"Data module for med_group {med_sublist_str} already exists for all folds, skipping")
            continue

    pl.seed_everything(rseed)
    print("current")

    advanced_model = 'default'

    # Determine if we should use regimen_analysis mode
    # For "all" baseline: use exact matching (regimen_analysis=False) to match eval_1
    # For specific regimens: use substring matching (regimen_analysis=True) for CAIA
    use_regimen_analysis = (med_group_name != 'all')
    print(f"med_group_name: {med_group_name}, use_regimen_analysis: {use_regimen_analysis}")

    # Create data module with med_combos
    # Note: exclusion_meds filtering happens at the regimen level via check_if_fits_criteria
    # The data loader includes patients who have ANY of the med_combos
    # Exclusions are handled by filtering out patients/windows that also have exclusion meds
    data_module = FlexibleFeatureDataModule(data_dict_lab_dict, split_dict, dataset = dataset, component = component,
            batch_size = 512, max_time_len = 20, num_workers=1,
            task = option, binary_days = days, med = med_combos, med_as_feature = False, med_l = med_combos,
            advanced_model = advanced_model, control_lack_confounds = False, side_effect_censoring = False,
            include_infusion_dates = include_infusion_dates, premed_only = False, competing_risk_death_check = False,
            anc=('CAIA' in dataset), exclusion_med_l=exclusion_meds, regimen_analysis=use_regimen_analysis)
    
    print("data module initiated")

    # Guard: test_dataloader() crashes on empty datasets; check length first
    if len(data_module.test_dataset) == 0:
        print(f"No eligible patients for med_group {med_sublist_str}, recording null entries and skipping")
        if med_sublist_str not in model_performance_dict:
            model_performance_dict[med_sublist_str] = {}
        null_entry = {
            'c_index_overall': None, 'c_index_weighted': None,
            'auroc_nextinf_overall': None, 'auroc_nextinf_weighted': None,
            'unique_num_patients': 0, 'unique_positives': 0,
        }
        for _model_key in ['SPARC-P', 'SPARC-P-all']:
            if _model_key not in model_performance_dict[med_sublist_str]:
                model_performance_dict[med_sublist_str][_model_key] = {}
            for val_fold in all_folds:
                model_performance_dict[med_sublist_str][_model_key][val_fold] = null_entry.copy()
        continue

    unique_num_patients = len(set(data_module.test_dataloader().dataset.dataset_DMP_IDs))
    # unique_positives = unique patients who ever had the event.
    # For risk tasks ys[i] = (entire_y, num_durations), so entire_y==1 for every window of
    # a patient who eventually had the event — summing would overcount. Count unique IDs instead.
    pos_patient_ids = set()
    for i in range(len(data_module.test_dataset.dataset_DMP_IDs)):
        label = data_module.test_dataset.ys[i][0] if 'risk' in option else data_module.test_dataset.ys[i]
        if label == 1:
            pos_patient_ids.add(data_module.test_dataset.dataset_DMP_IDs[i])
    unique_positives = len(pos_patient_ids)
    print("unique_num_patients", unique_num_patients, "unique_positives", unique_positives)

    # Fix: Initialize the dictionary structure properly
    if med_sublist_str not in model_performance_dict:
        model_performance_dict[med_sublist_str] = {}

    # Check performance - only evaluating SPARC-P on each medication sublist
    model_option = 'SPARC-P'

    # Initialize model_option in dict if not exists
    if model_option not in model_performance_dict[med_sublist_str]:
        model_performance_dict[med_sublist_str][model_option] = {}

    for val_fold in all_folds:
        pl.seed_everything(rseed)

        # Fix: Check if performance already exists BEFORE initializing
        if val_fold in model_performance_dict[med_sublist_str][model_option] and \
           len(model_performance_dict[med_sublist_str][model_option][val_fold]) > 0:
            print(f"Performance for med_sublist {med_sublist_str} and model {model_option} fold {val_fold} already exists, skipping")
            continue

        # Initialize the val_fold dictionary
        model_performance_dict[med_sublist_str][model_option][val_fold] = {}

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
        # Check the base option name (may or may not have _risk suffix by this point)
        base_option = option.replace('_risk', '')
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
            meds_l=med_combos,
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
                # Side-effect tasks: y is a tuple (event, time_to_event)
                fail_indicators = torch.as_tensor(y[0], device=device)
                time_to_event = torch.as_tensor(y[1], device=device)

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
            c_idx_dict = calculate_weighted_c_index_by_infusion(
                all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                mode='val', option=option
            )
            model_performance_dict[med_sublist_str][model_option][val_fold]['c_index_overall'] = c_idx_dict.get('c_index_overall', 0.5)
            model_performance_dict[med_sublist_str][model_option][val_fold]['c_index_weighted'] = c_idx_dict.get('c_index_weighted', 0.5)
            model_performance_dict[med_sublist_str][model_option][val_fold]['auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
            model_performance_dict[med_sublist_str][model_option][val_fold]['auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
            print(f"C-index overall: {c_idx_dict.get('c_index_overall', 0.5)}, C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5)}")
            print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
        else:
            model_performance_dict[med_sublist_str][model_option][val_fold]['c_index_overall'] = 0.5
            model_performance_dict[med_sublist_str][model_option][val_fold]['c_index_weighted'] = 0.5
            model_performance_dict[med_sublist_str][model_option][val_fold]['auroc_nextinf_overall'] = None
            model_performance_dict[med_sublist_str][model_option][val_fold]['auroc_nextinf_weighted'] = None
            print("No positive cases found in this group")

        model_performance_dict[med_sublist_str][model_option][val_fold]['unique_num_patients'] = unique_num_patients
        model_performance_dict[med_sublist_str][model_option][val_fold]['unique_positives'] = unique_positives

        # Store predictions for ensemble if enabled
        if ensemble:
            model_performance_dict[med_sublist_str][model_option][val_fold]['y_preds'] = all_y_preds.cpu().numpy()
            model_performance_dict[med_sublist_str][model_option][val_fold]['y_events'] = all_y_events.cpu().numpy()
            model_performance_dict[med_sublist_str][model_option][val_fold]['time_to_events'] = all_time_to_events.cpu().numpy()
            model_performance_dict[med_sublist_str][model_option][val_fold]['num_infusions'] = all_num_infusions.cpu().numpy()

    # Compute ensemble predictions if enabled
    if ensemble:
        for med_sublist_str in model_performance_dict.keys():
            for model_option in [k for k in model_performance_dict[med_sublist_str].keys() if not k.endswith('_ensemble')]:
                # Check if all 10 folds have predictions
                all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                all_folds_complete = all(fold in model_performance_dict[med_sublist_str][model_option] and
                                        'y_preds' in model_performance_dict[med_sublist_str][model_option][fold]
                                        for fold in all_folds)

                if all_folds_complete:
                    print(f"\n{'='*80}")
                    print(f"Computing ensemble for {model_option} on {med_sublist_str}")
                    print(f"{'='*80}")

                    # Collect predictions from all folds
                    fold_preds = [model_performance_dict[med_sublist_str][model_option][fold]['y_preds'] for fold in all_folds]
                    fold_events = [model_performance_dict[med_sublist_str][model_option][fold]['y_events'] for fold in all_folds]
                    fold_times = [model_performance_dict[med_sublist_str][model_option][fold]['time_to_events'] for fold in all_folds]
                    fold_infusions = [model_performance_dict[med_sublist_str][model_option][fold]['num_infusions'] for fold in all_folds]

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
                        c_idx_dict = calculate_weighted_c_index_by_infusion(
                            ensemble_preds_tensor, ensemble_events_tensor, ensemble_times_tensor, ensemble_infusions_tensor,
                            mode='val', option=option
                        )

                        # Count unique patients and positives
                        unique_patients = len(set([fold_infusions[0][i] for i in range(len(fold_infusions[0]))]))
                        unique_positives = (ensemble_events_tensor == 1).sum().item()

                        # Store ensemble results
                        ensemble_name = f'{model_option}_ensemble'
                        if ensemble_name not in model_performance_dict[med_sublist_str]:
                            model_performance_dict[med_sublist_str][ensemble_name] = {}

                        model_performance_dict[med_sublist_str][ensemble_name]['ensemble'] = {
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
                    print(f"{'='*80}\n")

    # Save to csv - moved outside the loop
    model_performance_df = pd.DataFrame(columns=all_columns)
    # Populate the dataframe from the dictionary
    for med_sublist_str in model_performance_dict.keys():
        for model_option in model_performance_dict[med_sublist_str].keys():
            for val_fold in model_performance_dict[med_sublist_str][model_option].keys():
                row = {
                    'med_sublist': med_sublist_str,
                    'model': model_option,
                    'val_fold': val_fold
                }
                # Only include metrics, not prediction arrays
                fold_data = {k: v for k, v in model_performance_dict[med_sublist_str][model_option][val_fold].items()
                            if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
                row.update(fold_data)
                row_values = [row.get(col, None) for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)
    model_performance_df = model_performance_df.drop_duplicates(subset=['med_sublist', 'model', 'val_fold'])
    model_performance_df.to_csv(save_name, index=False)

# Populate the dataframe from the dictionary (final save)
model_performance_df = pd.DataFrame(columns=all_columns)
for med_sublist_str in model_performance_dict.keys():
    for model_option in model_performance_dict[med_sublist_str].keys():
        for val_fold in model_performance_dict[med_sublist_str][model_option].keys():
            row = {
                'med_sublist': med_sublist_str,
                'model': model_option,
                'val_fold': val_fold
            }
            # Only include metrics, not prediction arrays
            fold_data = {k: v for k, v in model_performance_dict[med_sublist_str][model_option][val_fold].items()
                        if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
            row.update(fold_data)
            row_values = [row.get(col, None) for col in all_columns]
            model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row_values], columns=all_columns)], ignore_index=True)

# Remove duplicates and save
model_performance_df = model_performance_df.drop_duplicates(subset=['med_sublist', 'model', 'val_fold'])
model_performance_df.to_csv(save_name, index=False)

print("Model performance dict saved to csv")
