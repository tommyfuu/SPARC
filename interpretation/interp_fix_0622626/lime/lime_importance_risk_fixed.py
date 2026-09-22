"""Calculate and visualize LIME feature importance for SPARC side-effect risk models.

Loads data through dataset_loading.load_dataset(), computes LIME explanations
across all 10 CV folds and averages them. Writes a per-fold lime_df pickle in
the format lime_importance_risk_vis.py expects, plus two heatmap PDFs, one
including the timestamp feature and one without.

-qt runs fewer folds on a data subset for an end-to-end sanity check.

Output:
    <out_dir>/lime_df_val_fold_{fold}_{option}_{dataset}.pickle
    <out_dir>/lime_values_{option}_{dataset}[_remove_time].pdf
  defaulting to <all_model_path>/results_vis_060726/lime_fixed/
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pickle
import sys
import inspect
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from MSK_featurename_list import (
    BASIC_TESTS, BASIC_VITALS, EXTRA_TESTS, EXTRA_VITALS, CBC_DIFFERENTIAL_TESTS,
    RBC_CBC_TESTS, BMP_TESTS, LFT_TESTS, build_module_index_config
)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import sparc_paths

DEFAULT_DATA_SOURCE_PATH = sparc_paths.DATA_ROOT
DEFAULT_MODEL_PATH = sparc_paths.SPARC_MODEL_ROOT

# COMMAND LINE ARGUMENTS

import argparse

parser = argparse.ArgumentParser(
    prog='lime_importance_risk_fixed',
    description='Calculate and visualise LIME importance for SPARC risk models',
    epilog='Text at the bottom of help')
parser.add_argument('-r',  '--rseed',            type=int,   default=1024,      help='random seed')
parser.add_argument('-o',  '--option',           type=str,   default='AKI',     help='task: AKI, anemia, neutropenia, febrile_neutropenia, Liver_toxicity')
parser.add_argument('-d',  '--days',             type=int,   default=30,        help='number of days for binary classification tasks')
parser.add_argument('-m',  '--modules',          type=str,   default='parsimonious', help='module category: practical, all')
parser.add_argument('-ds', '--dataset',          type=str,   default='MSK_CAIA', help='dataset: MSK_CAIA, MSK')
parser.add_argument('-me', '--med_of_choice',    type=str,   default='CISPLATIN', help='comma-separated medication(s) of choice')
parser.add_argument('-an', '--add_num_infusion', type=str,   default='False',   help='whether to add number of infusion as a feature')
parser.add_argument('-qt', '--quick_test',       action='store_true',           help='quick-test mode: fewer folds and small data subset')
parser.add_argument('-ns', '--num_samples',      type=int,   default=2000,      help='LIME perturbation samples per instance (default 2000; original LIME default is 5000)')
parser.add_argument('-nj', '--n_jobs',           type=int,   default=4,         help='parallel LIME workers via joblib threading (default 4)')
parser.add_argument('-fs', '--fold_start',       type=int,   default=0,         help='first fold to process (inclusive); allows splitting folds across machines')
parser.add_argument('-fe', '--fold_end',         type=int,   default=-1,        help='last fold to process (exclusive); -1 means process all folds')
parser.add_argument('-dsp','--data_source_path', type=str, default=DEFAULT_DATA_SOURCE_PATH, help='root holding the per-dataset data volumes')
parser.add_argument('-mp', '--all_model_path',   type=str, default=DEFAULT_MODEL_PATH,       help='root holding the 060726 SPARC checkpoints')
parser.add_argument('-op', '--out_dir',          type=str, default=None,                      help='output dir (default: <all_model_path>/results_vis_060726/lime_fixed)')
parser.add_argument('-po', '--plot_only',        action='store_true',           help='skip LIME computation; load all existing per-fold pickles and re-plot')

# python3 lime_importance_risk_fixed.py -o febrile_neutropenia -ds MSK_CAIA \
#   -me "DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL" \
#   -an True --fold_start  9 --fold_end 10

args = parser.parse_args()
data_source_path = args.data_source_path
all_model_path   = args.all_model_path
out_dir          = args.out_dir or f'{all_model_path}/results_vis_060726/lime_fixed'
os.makedirs(out_dir, exist_ok=True)
rseed            = args.rseed
option           = args.option
days             = args.days
modules          = args.modules
dataset          = args.dataset
add_num_infusion = args.add_num_infusion.lower() == 'true'
QUICK_TEST       = args.quick_test
num_samples      = args.num_samples
n_jobs           = args.n_jobs
fold_start       = args.fold_start
fold_end         = args.fold_end
PLOT_ONLY        = args.plot_only

# only import if not plot-only, since dataset_loading can be slow and plot-only just needs the pickles
if not PLOT_ONLY:
    # Unified data loading + data module (FlexibleFeatureDataModule lives here)
    from dataset_loading import load_dataset
    from data_loaders_inf_risk_fixed import *
    from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS

    # Model (060726 version with modular tokenization)
    from SPARC_inf_risk_modular import SPARC

    # LIME
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from joblib import Parallel, delayed

    import pytorch_lightning as pl
    import torch


# Number of folds to process
NUM_FOLDS_QUICK_TEST = 2
num_folds = NUM_FOLDS_QUICK_TEST if QUICK_TEST else 10

# Fold range (allows splitting work across machines)
# --plot_only always aggregates all folds regardless of --fold_start/--fold_end
if PLOT_ONLY:
    fold_range = range(0, num_folds)
    print("PLOT_ONLY mode: will load all fold pickles and re-plot.")
else:
    _fold_end = num_folds if fold_end == -1 else fold_end
    fold_range = range(fold_start, _fold_end)
print(f"Processing folds: {list(fold_range)}")

# Parse med_of_choice into list-of-lists format
med_of_choice_raw = args.med_of_choice.split(',')
med_of_choice_raw = [med.strip() for med in med_of_choice_raw]
med_of_choice = []
for med_str in med_of_choice_raw:
    if '+' in med_str:
        med_of_choice.append([m.strip() for m in med_str.split('+')])
    else:
        med_of_choice.append([med_str])
print("med_of_choice:", med_of_choice)

# QUICK TEST BANNER

if QUICK_TEST:
    print("\n" + "=" * 70)
    print("QUICK TEST MODE ENABLED")
    print(f"  - Using small data subset (quick_test=True in load_dataset)")
    print(f"  - Processing only first {num_folds} model fold(s)")
    print("=" * 70 + "\n")

# VALIDATE OPTION

if not bool(re.match(
        r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$'
        r'|^(binary|binary_icu_pre)$',
        option)):
    raise ValueError(
        f"Invalid option '{option}'. Must be one of: "
        "anemia, neutropenia, febrile_neutropenia, AKI, Liver_toxicity "
        "(optionally with _risk or _next_risk suffix), binary, binary_icu_pre.")

# Set model_set_option and normalize option name to match model/data paths.
if option in ['binary', 'binary_icu_pre']:
    model_set_option = 'binary'
else:
    # Normalize bare side-effect option names (e.g. 'AKI' → 'AKI_risk'),
    # mirroring eval_2_across_medications_comparison.py (line 957-958).
    model_set_option = 'side_effects_risk'
    if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
        option = f'{option}_risk'
print(f"option (normalized): {option}, model_set_option: {model_set_option}")

# DATA LOADING + FEATURE DIMS

T_MAX = 20   # max_time_len (constant across all configs)

if not PLOT_ONLY:
    print("Loading dataset via dataset_loading.load_dataset()...")
    data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
        dataset=dataset,
        data_source_path=data_source_path,
        option=option,
        modules=modules,
        valfold=0,       # will iterate over folds below
        mode='train',    # need train split to build LIME background distribution
        rseed=rseed,
        competing_risk_death_check=False,
        include_gcsf=False,
        data_fraction_option='default',
        quick_test=QUICK_TEST,
    )
    print("modular_featuredim_dict:", modular_featuredim_dict)
    print("Data loading complete.")

    # Feature dim layout in SPARC_inf_risk.py:
    #   F_TOTAL = 2 * (base_features + 1) + 1
    #   [0 : base_features]           actual feature values
    #   [base_features]               time stamp
    #   [base_features+1 : F_TOTAL-1] missingness indicators
    #   [F_TOTAL-1]                   padding indicator
    base_features = sum(modular_featuredim_dict.values())   # e.g. 32 for practical
    F_TOTAL       = 2 * (base_features + 1) + 1             # e.g. 67 for practical
    n_actual      = base_features + 1                       # features + time stamp
    print(f"base_features={base_features}, F_TOTAL={F_TOTAL}, n_actual={n_actual}")
else:
    # Infer dims from the first existing pickle so we don't need to load the full dataset.
    # Pickle columns: T_MAX*F_TOTAL columns named t{t}_f{f}, then num_infusion_* columns.
    _first_pickle = f'{out_dir}/lime_df_val_fold_0_{option}_{dataset}.pickle'
    if not os.path.exists(_first_pickle):
        raise FileNotFoundError(
            f"--plot_only requires existing fold pickles; not found: {_first_pickle}")
    with open(_first_pickle, 'rb') as _fh:
        _tmp_df = pickle.load(_fh)
    _xs_col_count = sum(1 for c in _tmp_df.columns if re.match(r'^t\d+_f\d+$', c))
    F_TOTAL       = _xs_col_count // T_MAX
    base_features = (F_TOTAL - 3) // 2
    n_actual      = base_features + 1
    print(f"[plot_only] Inferred from pickle: base_features={base_features}, "
          f"F_TOTAL={F_TOTAL}, n_actual={n_actual}")

    # Reconstruct modular_featuredim_dict from modules parameter for plot_only mode
    # This is needed to build feature names correctly
    from MSK_featurename_list import MSK_numfeature_dict
    modular_featuredim_dict = {}
    modular_featuredim_dict['gender'] = MSK_numfeature_dict['gender']
    modular_featuredim_dict['age'] = MSK_numfeature_dict['age']

    if modules == 'practical':
        modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']
        modular_featuredim_dict['basic_vitals'] = MSK_numfeature_dict['basic_vitals']
    elif modules == 'parsimonious':
        modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']
        modular_featuredim_dict['cbc_differential'] = MSK_numfeature_dict['cbc_differential']
        modular_featuredim_dict['basic_vitals'] = MSK_numfeature_dict['basic_vitals']
    elif modules == 'all':
        modular_featuredim_dict['basic_lab'] = MSK_numfeature_dict['basic_lab']
        modular_featuredim_dict['extra_lab'] = MSK_numfeature_dict['extra_lab']
        modular_featuredim_dict['basic_vitals'] = MSK_numfeature_dict['basic_vitals']
        modular_featuredim_dict['extra_vitals'] = MSK_numfeature_dict['extra_vitals']

    print(f"[plot_only] Reconstructed modular_featuredim_dict: {modular_featuredim_dict}")

# FEATURE NAMES  (n_actual names: actual features + time stamp)

# Dynamically build feature names based on modular_featuredim_dict
# The order follows the hardcoded loop in data_loaders_inf_risk_fixed.py:
# ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']

feature_names_with_time = ['gender', 'age at seq']

# Add time-series features in the canonical order
if 'basic_lab' in modular_featuredim_dict:
    feature_names_with_time.extend(BASIC_TESTS)
if 'extra_lab' in modular_featuredim_dict:
    feature_names_with_time.extend(EXTRA_TESTS)
if 'cbc_differential' in modular_featuredim_dict:
    feature_names_with_time.extend(CBC_DIFFERENTIAL_TESTS)
if 'basic_vitals' in modular_featuredim_dict:
    feature_names_with_time.extend(BASIC_VITALS)
if 'extra_vitals' in modular_featuredim_dict:
    feature_names_with_time.extend(EXTRA_VITALS)

# Add timestamp
feature_names_with_time.append('time stamp')

feature_names_without_time = feature_names_with_time[:-1]

assert len(feature_names_with_time) == n_actual, (
    f"feature_names length {len(feature_names_with_time)} != n_actual {n_actual}"
)

print(f"\nFeature names constructed for modules='{modules}':")
print(f"  Static features (2): {feature_names_with_time[:2]}")
time_series_start = 2
for mod in ['basic_lab', 'extra_lab', 'cbc_differential', 'basic_vitals', 'extra_vitals']:
    if mod in modular_featuredim_dict:
        dim = modular_featuredim_dict[mod]
        print(f"  {mod} ({dim}): {feature_names_with_time[time_series_start:time_series_start+dim]}")
        time_series_start += dim
print(f"  Time stamp: {feature_names_with_time[-1]}")

# PLOTTING-SPECIFIC REORDERING (clinical grouping)
# For plotting, reorder features by clinical category:
#   1. Static: gender, age
#   2. CBC: RBC_CBC_TESTS
#   3. CBC Differential: CBC_DIFFERENTIAL_TESTS (if present)
#   4. BMP: BMP_TESTS
#   5. LFT: LFT_TESTS
#   6. Vitals: BASIC_VITALS
#   7. Timestamp

def create_plotting_order(feature_names_with_time):
    """Create indices to reorder features for clinical grouping in plots."""
    plotting_order_names = ['gender', 'age at seq']

    # Add CBC tests (RBC panel + WBC + Platelets)
    for test in RBC_CBC_TESTS:
        if test in feature_names_with_time:
            plotting_order_names.append(test)

    # Add CBC Differential if present
    if 'cbc_differential' in modular_featuredim_dict:
        for test in CBC_DIFFERENTIAL_TESTS:
            if test in feature_names_with_time:
                plotting_order_names.append(test)

    # Add BMP tests
    for test in BMP_TESTS:
        if test in feature_names_with_time:
            plotting_order_names.append(test)

    # Add LFT tests
    for test in LFT_TESTS:
        if test in feature_names_with_time:
            plotting_order_names.append(test)

    # Add vitals
    for test in BASIC_VITALS:
        if test in feature_names_with_time:
            plotting_order_names.append(test)

    # Add extra vitals if present
    if 'extra_vitals' in modular_featuredim_dict:
        for test in EXTRA_VITALS:
            if test in feature_names_with_time:
                plotting_order_names.append(test)

    # Add extra labs if present (after vitals)
    if 'extra_lab' in modular_featuredim_dict:
        for test in EXTRA_TESTS:
            if test in feature_names_with_time:
                plotting_order_names.append(test)

    # Add timestamp
    plotting_order_names.append('time stamp')

    # Create reordering indices
    plotting_indices = [feature_names_with_time.index(name) for name in plotting_order_names]

    return plotting_order_names, plotting_indices

feature_names_plotting_order, plotting_reorder_indices = create_plotting_order(feature_names_with_time)

# RENAME FEATURES FOR PLOTTING (more readable names)
FEATURE_DISPLAY_NAMES = {
    'gender': 'Gender',
    'age at seq': 'Age',
    # CBC
    'RBC': 'RBC',
    'MCV': 'MCV',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'RDW': 'RDW',
    'HGB': 'HGB',
    'HCT': 'HCT',
    'WBC': 'WBC',
    'Platelets': 'Platelets',
    # CBC Differential
    'Neut': 'Neutrophil',
    'Mono': 'Monocytes',
    'Lymph': 'Lymphocytes',
    'Baso': 'Basophils',
    'Eos': 'Eosinophils',
    # BMP
    'Glucose': 'Glucose',
    'Calcium': 'Calcium',
    'Sodium': 'Sodium',
    'Potassium': 'Potassium',
    'CO2': 'CO2',
    'Creatinine': 'Creatinine',
    'Chloride': 'Chloride',
    'BUN': 'BUN',
    # LFT
    'Albumin': 'Albumin',
    'Protein, Total': 'Protein, Total',
    'ALP': 'ALP',
    'ALT': 'ALT',
    'AST': 'AST',
    'Bilirubin, Total': 'Bilirubin, Total',
    # Vitals
    'BMI': 'BMI',
    'Blood Pressure BP Systolic': 'Systolic BP',
    'Blood Pressure Diastolic': 'Diastolic BP',
    'O2': 'O2',
    'Pulse Heart Rate': 'Pulse Hrt Rate',
    'Resp Rate': 'Resp Rate',
    'Temperature C': 'Temperature',
    # Timestamp
    'time stamp': 'Time Stamp',
}

# Apply display names
feature_names_plotting_display = [FEATURE_DISPLAY_NAMES.get(name, name) for name in feature_names_plotting_order]

# COLOR CODING FOR CLINICAL CATEGORIES
def get_feature_colors(feature_names, modular_featuredim_dict):
    """Assign colors to features based on clinical category.

    Color scheme:
    - Static (Gender, Age): gray
    - CBC (basic): red
    - CBC Differential: pink
    - BMP: green
    - LFT: orange
    - Vitals: blue
    - Timestamp: black
    """
    colors = []

    for name in feature_names:
        original_name = None
        # Reverse lookup in display names
        for orig, disp in FEATURE_DISPLAY_NAMES.items():
            if disp == name:
                original_name = orig
                break
        if original_name is None:
            original_name = name

        # Assign color based on category
        if original_name in ['gender', 'age at seq', 'Gender', 'Age']:
            colors.append('#808080')  # gray
        elif original_name in RBC_CBC_TESTS:
            colors.append('#d62728')  # red
        elif original_name in CBC_DIFFERENTIAL_TESTS or name in ['Neutrophil', 'Monocytes', 'Lymphocytes', 'Basophils', 'Eosinophils']:
            colors.append('#ff9896')  # pink
        elif original_name in BMP_TESTS:
            colors.append('#2ca02c')  # green
        elif original_name in LFT_TESTS:
            colors.append('#ff7f0e')  # orange
        elif original_name in BASIC_VITALS or name in ['Systolic BP', 'Diastolic BP', 'Pulse Hrt Rate']:
            colors.append('#1f77b4')  # blue
        elif original_name == 'time stamp' or name == 'Time Stamp':
            colors.append('#000000')  # black
        else:
            colors.append('#7f7f7f')  # default gray

    return colors

feature_colors = get_feature_colors(feature_names_plotting_display, modular_featuredim_dict)

print(f"\nPlotting order (clinical grouping):")
print(f"  Static (2): {feature_names_plotting_display[:2]}")
idx = 2
print(f"  CBC ({len(RBC_CBC_TESTS)}): {feature_names_plotting_display[idx:idx+len(RBC_CBC_TESTS)]}")
idx += len(RBC_CBC_TESTS)
if 'cbc_differential' in modular_featuredim_dict:
    print(f"  CBC Differential ({len(CBC_DIFFERENTIAL_TESTS)}): {feature_names_plotting_display[idx:idx+len(CBC_DIFFERENTIAL_TESTS)]}")
    idx += len(CBC_DIFFERENTIAL_TESTS)
print(f"  BMP ({len(BMP_TESTS)}): {feature_names_plotting_display[idx:idx+len(BMP_TESTS)]}")
idx += len(BMP_TESTS)
print(f"  LFT ({len(LFT_TESTS)}): {feature_names_plotting_display[idx:idx+len(LFT_TESTS)]}")
idx += len(LFT_TESTS)
print(f"  Vitals ({len(BASIC_VITALS)}): {feature_names_plotting_display[idx:idx+len(BASIC_VITALS)]}")
idx += len(BASIC_VITALS)
print(f"  Timestamp: {feature_names_plotting_display[-1]}")

# MODEL PATH HELPER

def get_model_folder(dataset, modules, option, val_fold):
    """Return the path to the SPARC-P model checkpoint folder for a given fold (060726 version)."""
    base_option = re.sub(r'_risk$|_next_risk$', '', option)

    # Determine task prefix (hybrid_ for FN/N/A, risk_ for AKI/LT, plain for binary)
    if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
        task_prefix = 'hybrid_'
    elif base_option in ['AKI', 'Liver_toxicity']:
        task_prefix = 'risk_'
    elif base_option in ['binary', 'binary_icu_pre']:
        task_prefix = ''
    else:
        raise ValueError(f"Unknown task: {base_option}")

    if 'CAIA' in dataset:
        folder = (
            f'{all_model_path}/{modules}_MSK_CAIA_060726/'
            f'SPARC_finetuned_models/'
            f'{modules}_finetuned_models_{task_prefix}{base_option}_20tp_topchemos_att_modular'
            f'/fold{val_fold}'
        )
    else:
        # MSK / MSK_timeshift
        folder = (
            f'{all_model_path}/{modules}_MSK_060726/'
            f'SPARC_finetuned_models/'
            f'{modules}_finetuned_models_{task_prefix}{base_option}_20tp_topchemos_att_modular'
            f'/fold{val_fold}'
        )
    return folder

# LIME INSTANCE HELPER  (defined once; reused inside fold loop)

def _explain_one(instance_vec, explainer, lime_wrapper, flat_feature_names, num_samples):
    """Explain a single flattened test instance; returns a list of LIME weights."""
    exp = explainer.explain_instance(
        instance_vec,
        lime_wrapper.predict,
        num_features=len(flat_feature_names),
        num_samples=num_samples,
    )
    contrib_dict = dict(exp.as_list())
    return [contrib_dict.get(name, 0.0) for name in flat_feature_names]

# LIME COMPUTATION — iterate over folds

across_folds_lime_values    = []   # [num_folds, T, F-1]   (mask col removed)
num_infusion_values         = []   # [num_folds, num_infusion_dim]

for val_fold in fold_range:
    print(f"\n{'='*60}")
    print(f"FOLD {val_fold}/{num_folds - 1}")
    print(f"{'='*60}")

    # Check if pickle already exists for this fold
    pickle_path = f'{out_dir}/lime_df_val_fold_{val_fold}_{option}_{dataset}.pickle'
    if os.path.exists(pickle_path):
        print(f"  Loading cached results from {pickle_path}")
        with open(pickle_path, 'rb') as fh:
            lime_df = pickle.load(fh)
    elif PLOT_ONLY:
        continue
        # raise FileNotFoundError(
        #     f"--plot_only requires all fold pickles to exist; missing: {pickle_path}")
    else:
        # Build data module for this fold
        pl.seed_everything(rseed)
        advanced_model = 'default'

        # include_infusion_dates must come from load_dataset (True for all side-effect
        # tasks — dataset_loading.py line 294: include_infusion_dates = se_task).
        # add_num_infusion only controls whether num_infusion appears in the LIME output.
        data_module = FlexibleFeatureDataModule(
            data_dict_lab_dict, split_dict,
            dataset=dataset, component=component,
            batch_size=512, max_time_len=20, num_workers=1,
            task=option, binary_days=days,
            med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
            advanced_model=advanced_model,
            control_lack_confounds=False,
            side_effect_censoring=False,
            include_infusion_dates=include_infusion_dates,
            premed_only=False,
            competing_risk_death_check=False,
            anc=('CAIA' in dataset),
        )
        print("  Data module initialised.")

        # Load model
        model_folder = get_model_folder(dataset, modules, option, val_fold)
        print(f"  Model folder: {model_folder}")
        existing_files = os.listdir(model_folder)
        ckpt_name = [f for f in existing_files if 'epoch' in f][0]
        optimal_model_path = os.path.join(model_folder, ckpt_name)
        print(f"  Checkpoint: {optimal_model_path}")

        # 060726 Model Configuration
        use_module_tokenizer = True   # Per-clinical-module tokenization
        use_continuous_time = True    # Learned continuous-time embedding
        post_mlp_dropout = 0.2        # Post-transformer MLP dropout

        # Build per-module tokenizer config
        module_index_cfg = build_module_index_config(
            modular_featuredim_dict, advanced_model='default'
        )
        n_static_features = module_index_cfg['n_static']

        # Determine loss_option based on task type
        base_option = re.sub(r'_risk$|_next_risk$', '', option)
        if base_option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
            loss_option = 'hybrid'  # Cox + BCE
        else:
            loss_option = 'default'  # Pure Cox/risk

        # Load model with 060726-specific parameters
        model = SPARC.load_from_checkpoint(
            optimal_model_path,
            modular_featuredim_dict=modular_featuredim_dict,
            option=model_set_option,
            advanced_model='default',
            max_time_len=20,
            med_as_feature=False,
            meds_l=med_of_choice,
            loss_option=loss_option,
            side_effects_num_infusion=include_infusion_dates,
            temporal_aggregation='attention',
            missing_mask=True,
            # 060726-specific parameters
            module_index_cfg=module_index_cfg,
            use_module_tokenizer=use_module_tokenizer,
            use_continuous_time=use_continuous_time,
            n_static_features=n_static_features,
            post_mlp_dropout=post_mlp_dropout,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"  Model on device: {device}")

        # Build LIME background distribution from one training batch
        train_batch = next(iter(data_module.train_dataloader()))
        train_xs, train_y, train_mask, _, num_infusion, _DMP_IDs = model.data_prep(
            train_batch, time_calibration=model.time_calibration
        )
        train_xs = train_xs.to(device)

        T, F = train_xs.shape[1], train_xs.shape[2]
        assert T == T_MAX and F == F_TOTAL, (
            f"Unexpected xs shape: got T={T}, F={F}; expected T_MAX={T_MAX}, F_TOTAL={F_TOTAL}")

        train_data_xs = train_xs.cpu().numpy().reshape(train_xs.shape[0], -1)  # [N, T*F]
        if num_infusion is not None:
            train_data_ni = num_infusion.cpu().numpy()
            if train_data_ni.ndim == 1:
                train_data_ni = train_data_ni.reshape(-1, 1)
        else:
            train_data_ni = np.zeros((train_data_xs.shape[0], 1))
        num_infusion_dim = train_data_ni.shape[1]

        train_data_np = np.concatenate([train_data_xs, train_data_ni], axis=1)

        # LIME model wrapper
        class LimeModelWrapper:
            def __init__(self, model, T, F, num_infusion_dim, device):
                self.model = model
                self.T = T
                self.F = F
                self.num_infusion_dim = num_infusion_dim
                self.device = device

            def predict(self, flat_inputs):
                """
                flat_inputs: np.ndarray [batch_size, T*F + num_infusion_dim]
                Returns class probabilities: [batch_size, 2]
                """
                xs_flat      = flat_inputs[:, :self.T * self.F]
                ni_flat      = flat_inputs[:, self.T * self.F:]

                xs = torch.tensor(xs_flat, dtype=torch.float32).reshape(-1, self.T, self.F).to(self.device)
                ni = torch.tensor(ni_flat.squeeze(), dtype=torch.float32).to(self.device)

                mask = torch.zeros(xs.size(0), self.T).bool().to(self.device).T  # [T, B]

                with torch.no_grad():
                    logits = self.model(xs, src_key_padding_mask=mask, num_infusion=ni)
                    probs  = torch.sigmoid(logits).cpu().numpy().flatten()
                    probs  = np.stack([1 - probs, probs], axis=1)
                return probs

        lime_wrapper = LimeModelWrapper(model, T, F, num_infusion_dim, device)

        # LIME feature names
        flat_feature_names = [f"t{t}_f{f}" for t in range(T) for f in range(F)]
        flat_feature_names += [f"num_infusion_{i}" for i in range(num_infusion_dim)]

        # LIME explainer
        explainer = LimeTabularExplainer(
            training_data=train_data_np,
            mode='classification',
            feature_names=flat_feature_names,
            discretize_continuous=False,
        )

        # Collect all test instances
        all_test_instances = []
        for idx, batch in enumerate(data_module.test_dataloader()):
            print(f"    Preparing test batch {idx + 1}/{len(data_module.test_dataloader())}", end='\r')
            test_xs, _y, _mask, _, ni_test, _DMP = model.data_prep(
                batch, time_calibration=model.time_calibration
            )
            test_xs = test_xs.to(device)
            xs_np = test_xs.cpu().numpy().reshape(test_xs.shape[0], -1)
            if ni_test is not None:
                ni_np = ni_test.cpu().numpy()
                if ni_np.ndim == 1:
                    ni_np = ni_np.reshape(-1, 1)
            else:
                ni_np = np.zeros((xs_np.shape[0], 1))
            all_test_instances.append(np.concatenate([xs_np, ni_np], axis=1))

        all_test_instances = np.vstack(all_test_instances)

        #     # In quick-test mode, explain only one batch worth of instances

        print(f"\n  Total test instances: {all_test_instances.shape[0]}")

        # Run LIME explanations (parallel via joblib threading)
        # threading backend is safe with CUDA: PyTorch releases the GIL
        # during GPU ops, so n_jobs workers can overlap on the GPU.
        #
        # Checkpointing: results are flushed to a .partial pickle every
        # CHECKPOINT_EVERY instances so a wall-time kill doesn't lose
        # everything.  On the next run the partial file is resumed from.
        n_instances = all_test_instances.shape[0]
        CHECKPOINT_EVERY = 500
        checkpoint_path = pickle_path + '.partial'

        # Resume from checkpoint if one exists
        start_idx = 0
        results = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as fh:
                results = pickle.load(fh)
            start_idx = len(results)
            print(f"  Resuming from checkpoint: {start_idx}/{n_instances} already done.")

        print(f"  Running LIME on {n_instances - start_idx} remaining instances "
              f"(n_jobs={n_jobs}, num_samples={num_samples})...")

        for chunk_start in range(start_idx, n_instances, CHECKPOINT_EVERY):
            chunk_end = min(chunk_start + CHECKPOINT_EVERY, n_instances)
            chunk = Parallel(n_jobs=n_jobs, backend='threading', verbose=5)(
                delayed(_explain_one)(
                    all_test_instances[i], explainer, lime_wrapper,
                    flat_feature_names, num_samples,
                )
                for i in range(chunk_start, chunk_end)
            )
            results.extend(chunk)
            with open(checkpoint_path, 'wb') as fh:
                pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Checkpoint saved: {len(results)}/{n_instances} done.")

        lime_df = pd.DataFrame(results, columns=flat_feature_names)
        print(f"  lime_df shape: {lime_df.shape}")

        # Save final pickle and remove checkpoint
        with open(pickle_path, 'wb') as fh:
            pickle.dump(lime_df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print(f"  Saved: {pickle_path}")

    print(f"  Test instances (fold {val_fold}): {lime_df.shape[0]} total")

    # Aggregate per-fold values
    # lime_df columns: T_MAX * F_TOTAL actual-feature columns, then num_infusion_dim columns
    lime_xs_vals = lime_df.iloc[:, :T_MAX * F_TOTAL].values.reshape(-1, T_MAX, F_TOTAL)  # [N, T, F_TOTAL]
    lime_ni_vals = lime_df.iloc[:, T_MAX * F_TOTAL:].values                               # [N, num_infusion_dim]

    # Keep only actual features + time stamp; discard missingness indicators and padding.
    # Layout: [0:n_actual] = real values+time, [n_actual:F_TOTAL] = missingness+padding
    lime_xs_actual = lime_xs_vals[:, :, :n_actual]                                        # [N, T, n_actual]

    lime_signed_xs = lime_xs_actual.mean(axis=0)                                          # [T, n_actual]
    lime_avg_ni    = lime_ni_vals.mean(axis=0)                                            # [num_infusion_dim]

    across_folds_lime_values.append(lime_signed_xs)
    num_infusion_values.append(lime_avg_ni)

# AGGREGATE ACROSS FOLDS

across_folds_lime_values = np.array(across_folds_lime_values)  # [num_folds, T, n_actual]

# With time stamp (last column of n_actual is the time stamp)
lime_values_avg      = across_folds_lime_values.mean(axis=0)                          # [T, n_actual]
lime_values_avg_T    = np.abs(across_folds_lime_values).mean(axis=(0, 2))             # [T]
# Weighted average over timepoints using timepoint importance as weights
_abs_avg_folds       = np.abs(across_folds_lime_values).mean(axis=0)                  # [T, n_actual]
lime_values_avg_F    = np.average(_abs_avg_folds, axis=0, weights=lime_values_avg_T)  # [n_actual]

# Without time stamp (drop the last column of n_actual = base_features)
lime_values_wotime   = across_folds_lime_values[:, :, :-1]                            # [num_folds, T, base_features]
lime_values_avg_wt   = lime_values_wotime.mean(axis=0)                                # [T, base_features]
lime_values_avg_T_wt = np.abs(lime_values_wotime).mean(axis=(0, 2))                   # [T]
# Weighted average over timepoints using timepoint importance as weights
_abs_avg_folds_wt    = np.abs(lime_values_wotime).mean(axis=0)                        # [T, base_features]
lime_values_avg_F_wt = np.average(_abs_avg_folds_wt, axis=0, weights=lime_values_avg_T_wt)  # [base_features]

avg_num_infusion = np.mean(num_infusion_values)

# VISUALIZATION

def plot_lime_values(
    lime_values_avg,
    lime_values_avg_F,
    lime_values_avg_T,
    feat_names,
    num_infusion_val,
    save_path,
    n_folds_used,
    title_suffix='',
    num_infusion_included=False,
    feat_colors=None,
):
    """
    Plot LIME values as a heatmap (time × feature) with:
      - Left panel: time importance (greyscale)
      - Bottom panel: feature importance (greyscale)
      - Main panel: signed LIME values (coolwarm)
      - Right colorbar
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    if num_infusion_included:
        T_dim, F_dim = lime_values_avg.shape
        lime_ext = np.full((T_dim, F_dim + 1), np.nan)
        lime_ext[:, :-1] = lime_values_avg
        lime_ext[0, -1] = 0.0   # neutral colour for the num_infusion cell
        feat_names_ext = list(feat_names) + ['num_infusion']
        lime_F_ext = np.append(lime_values_avg_F, np.nan)
        lime_plot  = lime_ext
        feat_plot  = feat_names_ext
        lime_F_plot = lime_F_ext
    else:
        lime_plot  = lime_values_avg
        feat_plot  = feat_names
        lime_F_plot = lime_values_avg_F

    abs_max = np.nanmax(np.abs(lime_plot))
    vmin, vmax = -abs_max, abs_max

    n_t = lime_plot.shape[0]
    n_f = lime_plot.shape[1]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[0.5, 20, 1],
        height_ratios=[20, 1],
        wspace=0.05,
        hspace=0.05,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_bot  = fig.add_subplot(gs[1, 1])
    cax     = fig.add_subplot(gs[0, 2])

    # Time importance (left)
    ax_left.imshow(np.expand_dims(lime_values_avg_T, axis=1),
                   aspect='auto', cmap='Greys', interpolation='nearest')
    ax_left.set_yticks(range(len(lime_values_avg_T)))
    ax_left.set_yticklabels(range(len(lime_values_avg_T)))
    ax_left.set_xticks([])
    ax_left.set_title('Time\nImportance', fontsize=10)
    ax_left.tick_params(left=False)

    # Main LIME heatmap
    im0 = ax_main.imshow(
        lime_plot, aspect='auto', cmap='coolwarm',
        interpolation='nearest', vmin=vmin, vmax=vmax,
        extent=[0, n_f, n_t, 0],
    )
    title = f'LIME values (avg across {n_folds_used} folds, all patients){title_suffix}'
    ax_main.set_title(title, fontsize=11)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlim(0, n_f)

    if num_infusion_included:
        F_orig = lime_values_avg.shape[1]
        rect = mpatches.Rectangle(
            (F_orig, 0), 1, 1,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax_main.add_patch(rect)

    # Feature importance (bottom)
    ax_bot.imshow(
        np.expand_dims(lime_F_plot, axis=0), aspect='auto',
        cmap='Greys', interpolation='nearest',
        extent=[0, n_f, 1, 0],
    )
    ax_bot.set_yticks([0.5])
    ax_bot.set_yticklabels(['Feature\nImportance'])
    ax_bot.set_xticks(np.arange(len(feat_plot)) + 0.5)
    ax_bot.set_xticklabels(feat_plot, rotation=45, ha='right', fontsize=8)

    # Apply color coding to x-axis labels if provided
    if feat_colors is not None:
        for tick_label, color in zip(ax_bot.get_xticklabels(), feat_colors):
            tick_label.set_color(color)
            tick_label.set_weight('bold')

    ax_bot.set_xlim(0, n_f)
    ax_bot.tick_params(axis='y', which='both', length=0)

    plt.colorbar(im0, cax=cax)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {save_path}")


n_folds_used = len(across_folds_lime_values)

# Apply clinical grouping reordering for plots
print("\nApplying clinical grouping reordering for plots...")

# Reorder with timestamp
lime_values_avg_reordered = lime_values_avg[:, plotting_reorder_indices]
lime_values_avg_F_reordered = lime_values_avg_F[plotting_reorder_indices]

# Reorder without timestamp (exclude last feature which is timestamp)
plotting_reorder_indices_wt = [i for i in plotting_reorder_indices if i < len(feature_names_without_time)]
lime_values_avg_wt_reordered = lime_values_avg_wt[:, plotting_reorder_indices_wt]
lime_values_avg_F_wt_reordered = lime_values_avg_F_wt[plotting_reorder_indices_wt]
feature_names_plotting_display_wt = [feature_names_plotting_display[i] for i in range(len(feature_names_plotting_display)) if feature_names_plotting_display[i] != 'Time Stamp']
feature_colors_wt = [feature_colors[i] for i in range(len(feature_colors)) if feature_names_plotting_display[i] != 'Time Stamp']

# Plot with time stamp
plot_lime_values(
    lime_values_avg_reordered, lime_values_avg_F_reordered, lime_values_avg_T,
    feat_names=feature_names_plotting_display,
    num_infusion_val=avg_num_infusion,
    save_path=f'{out_dir}/lime_values_{option}_{dataset}.pdf',
    n_folds_used=n_folds_used,
    title_suffix='',
    num_infusion_included=add_num_infusion,
    feat_colors=feature_colors,
)

# Plot without time stamp
plot_lime_values(
    lime_values_avg_wt_reordered, lime_values_avg_F_wt_reordered, lime_values_avg_T_wt,
    feat_names=feature_names_plotting_display_wt,
    num_infusion_val=avg_num_infusion,
    save_path=f'{out_dir}/lime_values_{option}_{dataset}_remove_time.pdf',
    n_folds_used=n_folds_used,
    title_suffix=' [no timestamp]',
    num_infusion_included=add_num_infusion,
    feat_colors=feature_colors_wt,
)

# Plot time series features only (always remove age at seq and gender, remove num_infusion if present)
# Use reordered version but exclude gender and age (first 2 features)
ts_indices = [i for i in range(len(feature_names_plotting_display)) if feature_names_plotting_display[i] not in ['Gender', 'Age', 'Time Stamp']]
lime_values_avg_ts_reordered = lime_values_avg_reordered[:, ts_indices]
lime_values_avg_F_ts_reordered = lime_values_avg_F_reordered[ts_indices]
feature_names_ts_display = [feature_names_plotting_display[i] for i in ts_indices]
feature_colors_ts = [feature_colors[i] for i in ts_indices]

plot_lime_values(
    lime_values_avg_ts_reordered, lime_values_avg_F_ts_reordered, lime_values_avg_T,
    feat_names=feature_names_ts_display,
    num_infusion_val=avg_num_infusion,
    save_path=f'{out_dir}/lime_values_{option}_{dataset}_time_series_only.pdf',
    n_folds_used=n_folds_used,
    title_suffix=' [time series features only]',
    num_infusion_included=False,
    feat_colors=feature_colors_ts,
)
print("\nDone.")
