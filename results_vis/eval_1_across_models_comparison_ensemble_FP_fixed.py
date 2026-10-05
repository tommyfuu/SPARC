"""Evaluate every model on one task, per fold and as a 10-fold mean ensemble.

Binary tasks: AUROC/AUPRC. Side-effect tasks: weighted C-index and next-infusion AUROC/AUPRC.
-en True adds '{model}_ensemble' rows (bootstrap CIs) and saves the ensemble predictions.

Usage: python eval_1_across_models_comparison_ensemble_FP_fixed.py -o AKI -en True --ci_method bootstrap
"""

import os
import pickle
import sys, inspect
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import re
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# put the repo root, data/ and models/ on sys.path
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
from SPARC_inf_risk_modular import SPARC

from MSK_featurename_list import build_module_index_config

for model_option in ['lasttp_MLP', 'raindrop', 'rnn', 'duett']:
    sparc_paths.add_benchmark_to_path(model_option)
import json


from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, AveragePrecision, Accuracy, Precision, Recall
c_index = ConcordanceIndex()
loss_fn = nn.BCELoss(reduction='none')
auroc = AUROC(task='binary')
accuracy = Accuracy(task='binary')
balanced_accuracy = Accuracy(task='binary', average='macro')
precision = Precision(task='binary')
recall = Recall(task='binary')
auprc = AveragePrecision(task='binary')


from eval_utils import calculate_weighted_c_index_by_infusion

import argparse
parser = argparse.ArgumentParser(
            prog='across_models_comparison',
            description='calculate the model performance comparison across models',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-m", "--modules", type=str, default = 'practical', help='module category, either "all", "allnonICD" or connect categories by dashes such as')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK_CAIA', help='dataset used in training, options: MSK, MSK_timeshift, stanfordEHRSHOT')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-mm", "--missing_mask", type=str, default = "True", help="True | False — concatenate missingness indicators as extra features (default True); use False for no-mask (nm) models")
parser.add_argument("-cp", "--caia_portion", type=str, default = "prior", help="prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021")
parser.add_argument("-z", "--zeroshot", type=str, default = 'True', help='whether to use zeroshot (MSK-trained) models for MIMICIV tasks; when False, finetuned MIMICIV models are used and _finetuned is appended to save names, options: True, False')
parser.add_argument("-federated", "--federated", action='store_true', help='whether to only evaluate federated models (SPARC-P) or to evaluate all models including benchmarks')
parser.add_argument("-dsp", "--data_source_path", type=str, default=sparc_paths.DATA_ROOT + '/', help='path to data directory')
parser.add_argument("-mp", "--model_path", type=str, default=sparc_paths.SPARC_MODEL_ROOT, help='path to 060726 model checkpoints directory')
parser.add_argument("-rp", "--results_path", type=str, default=None, help='path to save results (default: sparc_paths.SPARC_RESULTS_VIS_ROOT)')
parser.add_argument("-smp", "--sparc_model_path", type=str, default=None, help='root holding ONLY the SPARC / SPARC-P checkpoints; defaults to --model_path. Set this to evaluate batch_first=True SPARC against the published benchmarks, which keep resolving under --model_path.')
parser.add_argument("-bf", "--batch_first", type=str, default='True', help='True | False -- build SPARC with batch_first=True so the encoder attends over time. MUST match how the checkpoints were trained: batch_first changes no tensor shapes, so a mismatch loads silently and evaluates the wrong attention axis.')
parser.add_argument("-en", "--ensemble", type=str, default='True', help='whether to compute ensemble predictions by averaging over 10 folds for each model, options: True, False')
parser.add_argument("--ci_method", type=str, default='bootstrap', help='confidence interval method for the ensemble AUROC/AUPRC: none or bootstrap (default: bootstrap)')
parser.add_argument("--n_bootstrap", type=int, default=1000, help='number of bootstrap samples for CI calculation (default: 1000)')

args = parser.parse_args()
rseed = args.rseed
option = args.option
med_of_choice = args.med_of_choice
days = args.days
modules = args.modules
dataset = args.dataset
add_num_infusion = args.add_num_infusion.lower() == 'true'
missing_mask = args.missing_mask.lower() == 'true'
caia_portion = args.caia_portion.lower()
zeroshot = args.zeroshot.lower() == 'true'
federated = args.federated
data_source_path = args.data_source_path
model_path = args.model_path
results_path = args.results_path if args.results_path else sparc_paths.SPARC_RESULTS_VIS_ROOT
ensemble = args.ensemble.lower() == 'true'
ci_method = args.ci_method.lower()
if ci_method not in ('none', 'bootstrap'):
    raise SystemExit(f"--ci_method must be 'none' or 'bootstrap', got {ci_method!r}")
n_bootstrap = args.n_bootstrap


if not federated:
    all_models = ['SPARC', 'SPARC-P', 'lasttp_xgboost', 'temporal_xgboost',
                  'temporal_tabpfn', 'lasttp_MLP', 'rnn', 'duett', 'raindrop']
else:
    all_models = ['SPARC-P']


if not (bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)) or option in ['binary', 'binary_icu_pre']):
    raise ValueError(f"Invalid option: {option}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# combinations like 'CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE' become one inner list
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


# hyperparameters
batch_size = 512
competing_risk_death_check = False
include_gcsf = False

# SPARC architecture; must match training (-umt True -uct True -pmd 0.2 -tp attention)
use_module_tokenizer = True
use_continuous_time = True
post_mlp_dropout = 0.2
use_sigmoid = True
schedule_option = 'default'
learning_rate = 1e-4

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
    valfold=0,
    mode='test_all' if 'timeshift' in dataset else 'test',
    rseed=rseed,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=include_gcsf,
    data_fraction_option='default',
    quick_test=False,
    caia_portion=caia_portion,
)

print("modular_featuredim_dict", modular_featuredim_dict)
print("Data loading complete")

# per-module tokenizer config
module_index_cfg = None
n_static_features = 0
if use_module_tokenizer:
    module_index_cfg = build_module_index_config(
        modular_featuredim_dict, advanced_model='default',
    )
    n_static_features = module_index_cfg['n_static']
    print("=" * 70)
    print("PER-MODULE TOKENIZER CONFIG")
    print(f"  n_static={module_index_cfg['n_static']}, "
          f"n_dynamic={module_index_cfg['n_dynamic']}, "
          f"n_base={module_index_cfg['n_base']}")
    print("=" * 70)

# SPARC / SPARC-P can live in a different root than the benchmarks
all_model_path = args.sparc_model_path if args.sparc_model_path else model_path
batch_first = args.batch_first == 'True'
print(f"SPARC/SPARC-P root : {all_model_path}")
print(f"batch_first        : {batch_first}")

# benchmarks live in the published (non-_060726) tree
benchmark_model_path = (model_path.replace('_060726', '') if '_060726' in model_path
                        else sparc_paths.BENCHMARK_MODEL_ROOT)

if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift', 'stanfordEHRSHOT'] or (dataset == 'MIMICIV_hosp' and zeroshot):
    if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
        # binary tasks
        model_folder_dict = {
            'SPARC':   f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_direct_models/{modules}_direct_models_{option}_20tp_topchemos_att_modular',
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{option}_20tp_topchemos_att_modular'
        }
    else:
        # side-effect tasks; folder tag is 'hybrid' for the hematologic ones, 'risk' for AKI / liver
        base_option = option.replace('_risk', '').replace('_next_risk', '')
        if base_option in ['anemia', 'neutropenia', 'febrile_neutropenia']:
            model_type_suffix = 'hybrid'
        else:
            model_type_suffix = 'risk'

        model_folder_dict = {
            'SPARC': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_direct_models/{modules}_direct_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular',
            'SPARC-P': f'{all_model_path}/{modules}_MSK_CAIA_060726/SPARC_finetuned_models/{modules}_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular'
            }
        
else:  # MIMICIV_hosp finetuned models (none trained with batch_first=True)
    mimiciv_sparc_path = sparc_paths.SPARC_MODEL_ROOT + '/parsimonious_MIMICIV_hosp_060726'
    if batch_first:
        raise SystemExit("No batch_first=True MIMICIV_hosp-finetuned SPARC exists; use -z True (zero-shot from the CAIA models).")

    if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
        model_folder_dict = {
            'SPARC': f'{mimiciv_sparc_path}/SPARC_direct_models/parsimonious_direct_models_{option}_20tp_topchemos_att_modular',
            'SPARC-P': f'{mimiciv_sparc_path}/SPARC_finetuned_models/parsimonious_finetuned_models_{option}_20tp_topchemos_att_modular',
        }
    else:
        base_option = option.replace('_risk', '').replace('_next_risk', '')
        if base_option in ['anemia', 'neutropenia', 'febrile_neutropenia']:
            model_type_suffix = 'hybrid'
        else:
            model_type_suffix = 'risk'

        model_folder_dict = {
            'SPARC': f'{mimiciv_sparc_path}/SPARC_direct_models/parsimonious_direct_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular',
            'SPARC-P': f'{mimiciv_sparc_path}/SPARC_finetuned_models/parsimonious_finetuned_models_{model_type_suffix}_{base_option}_20tp_topchemos_att_modular',
        }

    
benchmark_models = ['lasttp_MLP', 'rnn', 'lasttp_xgboost', 'temporal_xgboost',
                    'temporal_tabpfn', 'duett', 'raindrop']

# import only the benchmark classes that are needed
if 'lasttp_MLP' in all_models:
    from lasttp_MLP import *
if 'raindrop' in all_models:
    from raindrop import *
if 'rnn' in all_models:
    from rnn import *
if 'duett' in all_models:
    from duett import DuETT
# survival XGBoost variants for the side-effect tasks
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
    benchmark_models = benchmark_models + ['lasttp_xgboost_risk', 'temporal_xgboost_risk']

for model_option in benchmark_models:
    # benchmark folders use the option without _risk
    option_for_path = option.replace('_risk', '') if '_risk' in option else option
    if not ('MIMICIV' in dataset and not zeroshot):
        if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)) and model_option == 'rnn_deephit':
            model_folder_dict[model_option] =  f'{benchmark_model_path}/{modules}_MSK_CAIA/benchmarks/{model_option}/{modules}_{option_for_path}_risk'
        elif model_option in ['lasttp_xgboost_risk', 'temporal_xgboost_risk']:
            # survival XGBoost: base-model folder, _risk sub-folder
            base_model = model_option.replace('_risk', '')
            model_folder_dict[model_option] = f'{benchmark_model_path}/{modules}_MSK_CAIA/benchmarks/{base_model}/{modules}_{option_for_path}_risk'
        else:
            model_folder_dict[model_option] =  f'{benchmark_model_path}/{modules}_MSK_CAIA/benchmarks/{model_option}/{modules}_{option_for_path}'
        if dataset in ['MSK_CAIA', 'MSK_CAIA_timeshift', 'stanfordEHRSHOT'] or (dataset == 'MIMICIV_hosp' and zeroshot):
            model_folder_dict[model_option] = model_folder_dict[model_option] + '_MSK_CAIA'
    else:
        print("adding MIMICIV_hosp-specific paths for benchmarks since dataset is MIMICIV_hosp and zeroshot is False")

        # XGBoost models live under SL_TESTSL_ROOT, DL benchmarks under the published root
        if model_option in ['lasttp_xgboost', 'temporal_xgboost', 'lasttp_xgboost_risk', 'temporal_xgboost_risk']:
            sl_model_path = sparc_paths.SL_TESTSL_ROOT
            if model_option in ['lasttp_xgboost_risk', 'temporal_xgboost_risk']:
                base_model = model_option.replace('_risk', '')
                model_folder_dict[model_option] = f'{sl_model_path}/{modules}_MIMICIV_hosp/benchmarks/{base_model}/{modules}_{option_for_path}_risk_MIMICIV_hosp'
            else:
                model_folder_dict[model_option] = f'{sl_model_path}/{modules}_MIMICIV_hosp/benchmarks/{model_option}/{modules}_{option_for_path}_MIMICIV_hosp'
        else:
            model_folder_dict[model_option] = f'{benchmark_model_path}/{modules}_MIMICIV_hosp/benchmarks/{model_option}/{modules}_{option_for_path}_MIMICIV_hosp'

model_performance_dict = {}

print("model_folder_dict", model_folder_dict)
_nm_suffix = '' if missing_mask else '_nm'
_finetuned_suffix = '_finetuned' if 'MIMICIV' in dataset and not zeroshot else ''
_ensemble_suffix = '_ensemble' if ensemble else ''
# suffix kept only so output paths match the archived results
_valthresh_suffix = '_valthresh_disabled'
save_name = f'{results_path}/eval_across_models{_nm_suffix}{_valthresh_suffix}/model_performance_dict_across_models_{modules}_{option}_{dataset}{_finetuned_suffix}{_ensemble_suffix}.csv'
print(f"Results will be saved to: {save_name}")
print(f"  - CI method: {ci_method}")


# test split; for CAIA it comes from load_dataset
if 'CAIA' in dataset:
    print(f"Using split_dict from load_dataset for {dataset}")
    print(f"  Test set size: {len(split_dict['test'])}")
elif dataset == 'MSK':
    with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as handle:
        fold_dict = pickle.load(handle)
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'MSK_timeshift':
    # all patients except those in the MSK cohort
    fold_dict = {10: data_dict_lab_dict['age'].keys()}
    data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
    with open(data_path, 'rb') as handle:
        og_set = pickle.load(handle)
    fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'stanfordEHRSHOT':
    # use every patient
    with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_all_included_lab_values_list_dict.pickle', 'rb') as handle:                                                                                                    
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
# raw ensemble predictions per model (the CSV only holds scalar metrics)
ensemble_predictions_store = {}
# Fold rows already in the CSV are reused, but their predictions are not stored, so the
# ensemble for those models cannot be rebuilt. Move the CSV aside to rerun from scratch.
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    performance_df_existing = saved_df[~saved_df['model'].str.contains('averaged')]

    is_side_effect_task = bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    # drop rows with none of the task's metrics
    if is_side_effect_task:
        relevant_cols = ['test_cindex_overall_valid', 'test_cindex_weighted',
                        'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted']
        performance_df_existing = performance_df_existing.dropna(subset=relevant_cols, how='all')
    else:
        relevant_cols = ['auroc', 'auprc']
        performance_df_existing = performance_df_existing.dropna(subset=relevant_cols, how='all')

    for index, row in performance_df_existing.iterrows():
        model_name = row['model']
        if row['val_fold'] == 'ensemble':
            continue
        val_fold = int(row['val_fold'])
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        if is_side_effect_task:
            if 'risk' not in option:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': row.get('test_cindex_overall_valid', row.get('c_index_overall')),
                    'test_cindex_weighted': row.get('test_cindex_weighted', row.get('c_index_weighted')),
                    'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
                    'test_auroc_nextinf_weighted': row.get('test_auroc_nextinf_weighted'),
                }
            else:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': row.get('test_cindex_overall_valid', row.get('c_index_overall')),
                    'test_cindex_weighted': row.get('test_cindex_weighted', row.get('c_index_weighted')),
                    'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
                    'test_auroc_nextinf_weighted': row.get('test_auroc_nextinf_weighted'),
                }
        else:
            model_performance_dict[model_name][val_fold] = {
                'auroc': row.get('auroc'),
                'auprc': row.get('auprc'),
            }
    print("Loaded existing model performance data from csv")

dl_models = ['lasttp_MLP', 'raindrop', 'duett', 'rnn', 'rnn_deephit']

if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    model_set_option = 'side_effects_risk'
    option = f'{option}_risk'
else:
    model_set_option = 'binary'

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for model_option in all_models:
    if model_option in ['duett', 'raindrop']:
        advanced_model = model_option
    else:
        advanced_model = 'default'
    pl.seed_everything(rseed)
    # skip models whose fold rows are all cached
    if model_option in model_performance_dict and False not in [fold in model_performance_dict[model_option] for fold in all_folds]:
        print(f"data module for model {model_option} already exists for all folds, skipping")
        continue

    # only used by SPARC
    temporal_aggregation = 'attention'

    # SPARC and the tree/TabPFN models share the 20-timepoint data module
    if model_option in ['SPARC', 'SPARC-P', 'SPARC-P-Progressive', 'lasttp_xgboost', 'temporal_xgboost', 'lasttp_xgboost_risk', 'temporal_xgboost_risk', 'temporal_tabpfn']:
        print("model_option", model_option, "option", option, "model_set_option", model_set_option, "advanced_model", advanced_model, "temporal_aggregation", temporal_aggregation)
        # CAIA has absolute neutrophil counts; other datasets have percentages
        anc_mode = True if 'CAIA' in dataset else False
        data_module = FlexibleFeatureDataModule(data_dict_lab_dict, split_dict, dataset = dataset, component = component,
            batch_size = 512, max_time_len = 20, num_workers=1,
            task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
            advanced_model = advanced_model, control_lack_confounds = False, side_effect_censoring = False,
            include_infusion_dates = add_num_infusion, premed_only = False, competing_risk_death_check = False,
            anc=anc_mode)
        print("data module initiated")
    elif model_option in dl_models:
        config_path = f'{sparc_paths.BENCHMARK_CONFIG_DIR}/{model_option}_best_hyperparams.json'
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
        if model_option not in  ['rnn_deephit']:
            option_base = option.replace("_risk", "")
            config = all_configs[option_base]['config']
        else:
            config = all_configs[option]['config']
        current_data_dict_lab_dict = data_dict_lab_dict.copy()

        print("model_option", model_option, "option", option, "model_set_option", model_set_option, "advanced_model", advanced_model, "temporal_aggregation", temporal_aggregation)
        anc_mode = True if 'CAIA' in dataset else False
        data_module = FlexibleFeatureDataModule(data_dict_lab_dict, split_dict, dataset = dataset, component = component,
            batch_size = 512, max_time_len = config['max_time_len'], num_workers=1,
            task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
            advanced_model = advanced_model, include_infusion_dates = add_num_infusion, anc=anc_mode)
        print("data module initiated")

    for val_fold in all_folds:
        print("model_option", model_option, "val_fold", val_fold)
        pl.seed_everything(rseed)
        if model_option not in model_performance_dict:
            model_performance_dict[model_option] = {}
        if val_fold in model_performance_dict[model_option]:
            print(f"performance for model {model_option} fold {val_fold} already exists, skipping")
            continue
        else:
            model_performance_dict[model_option][val_fold] = {}
        print(f"Processing {model_option} for fold {val_fold}")

        if model_option in ['SPARC', 'SPARC-P', 'SPARC-P-Progressive']:
            model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        else:
            model_folder = f'{model_folder_dict[model_option]}/fold_{val_fold}'
        deep_models = set(['SPARC', 'SPARC-P', 'SPARC-P-Progressive'] + dl_models)
        if model_option in deep_models:
            print(model_folder)
            # a missing checkpoint skips the fold instead of killing the run
            existing_files = os.listdir(model_folder) if os.path.isdir(model_folder) else []
            # best checkpoint (not last.ckpt)
            epoch_files = [el for el in existing_files if 'epoch' in el]
            if not epoch_files:
                print(f"WARNING: no checkpoint for {model_option} fold {val_fold} "
                      f"in {model_folder} -- skipping this fold")
                model_performance_dict[model_option].pop(val_fold, None)
                continue
            optimal_model_path = os.path.join(model_folder, epoch_files[0])
            print(model_option, "optimal_model_path", optimal_model_path)

        if model_option == 'raindrop':
            from MSK_featurename_list import MSK_numfeature_dict
            total_static_len = sum([v for k,v in MSK_numfeature_dict.items() if k in ['gender', 'age', 'primary_ctype'] and k in data_modules_to_include])
            total_dynamic_len = sum([v for k,v in MSK_numfeature_dict.items() if k in ['basic_lab', 'extra_lab', 'basic_vitals'] and k in data_modules_to_include])
            print("total_static_len", total_static_len, "total_dynamic_len", total_dynamic_len)

        if model_option in ['SPARC', 'SPARC-P', 'SPARC-P-Progressive']:
            # architecture args must match training
            model = SPARC.load_from_checkpoint(
                optimal_model_path,
                modular_featuredim_dict=modular_featuredim_dict,
                option=model_set_option,
                advanced_model="default",
                max_time_len=20,
                med_as_feature=False,
                meds_l=med_of_choice,
                loss_option='default',
                schedule=schedule_option,
                side_effects_num_infusion=add_num_infusion,
                include_gcsf=include_gcsf,
                use_gcsf=include_gcsf,
                use_sigmoid=use_sigmoid,
                temporal_aggregation=temporal_aggregation,
                missing_mask=missing_mask,
                module_index_cfg=module_index_cfg,
                use_module_tokenizer=use_module_tokenizer,
                use_continuous_time=use_continuous_time,
                n_static_features=n_static_features,
                post_mlp_dropout=post_mlp_dropout,
                batch_first=batch_first,
                lr=learning_rate,
            )
            print(f"Loaded {model_option} with architecture params: use_module_tokenizer={use_module_tokenizer}, "
                  f"use_continuous_time={use_continuous_time}, post_mlp_dropout={post_mlp_dropout}, "
                  f"batch_first={batch_first}")
        elif model_option == 'lasttp_MLP':
            model = lasttp_MLP.load_from_checkpoint(optimal_model_path, strict=False,
                modular_featuredim_dict = modular_featuredim_dict, option = model_set_option, weight_decay=config["weight_decay"], lr = config["lr"],
                num_layers=config["num_layers"], side_effects_num_infusion = add_num_infusion)
        if model_option == 'duett':
            # DuETT checkpoints store no hyperparameters; strict load so a mismatch fails
            model = DuETT.load_from_checkpoint(optimal_model_path,
                modular_featuredim_dict = modular_featuredim_dict, option = model_set_option,
                max_time_len=config["max_time_len"], weight_decay=config["weight_decay"], lr = config["lr"],
                num_encoder_layers=config["num_encoder_layers"], dropout=config["dropout"],
                side_effects_num_infusion = add_num_infusion)
        if model_option == 'raindrop':
            model = Raindrop.load_from_checkpoint(optimal_model_path, strict=False,
                d_static=total_static_len, nlayers=config["nlayers"], max_len=config["max_time_len"],
                lr=config["lr"], weight_decay=config["weight_decay"], option = model_set_option, side_effects_num_infusion = add_num_infusion)
        if model_option in ['rnn', 'rnn_deephit']:
            print("model_set_option", model_set_option)
            if model_option == 'rnn_deephit':
                advanced_model = 'rnn_deephit'
            model = RNN.load_from_checkpoint(optimal_model_path, strict=False, advanced_model = advanced_model,
                modular_featuredim_dict=modular_featuredim_dict, num_encoder_layers=config["num_encoder_layers"], max_time_len=config["max_time_len"],
                lr=config["lr"], weight_decay=config["weight_decay"], option = model_set_option, side_effects_num_infusion = add_num_infusion)

        if model_option in ['lasttp_xgboost', 'temporal_xgboost', 'lasttp_xgboost_risk', 'temporal_xgboost_risk']:
            # _risk variants are pickled under the base model name
            base_model = model_option.replace('_risk', '')
            with open(f'{model_folder}/{base_model}_model.pickle', 'rb') as handle:
                model = pickle.load(handle)

        if model_option in ['temporal_tabpfn']:
            with open(f'{model_folder}/{model_option}_model.pickle', 'rb') as handle:
                model = pickle.load(handle)

        if model_option in ['SPARC', 'SPARC-P', 'SPARC-P-Progressive', 'lasttp_MLP', 'raindrop', 'rnn', 'rnn_deephit', 'duett']:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            print(f"Model {model_option} on device: {device}")


        y_true = []
        y_pred = []
        pl.seed_everything(rseed)
        if not 'rf' in model_option and not 'xgboost' in model_option and not 'tabpfn' in model_option:
            # DL models: run the test loader
            all_y_preds = []
            all_y_events = []
            all_time_to_events = []
            all_num_infusions = []

            batch_count = 0
            with torch.no_grad():
                for batch in data_module.test_dataloader():
                    if model_option == 'raindrop':
                        (xs, static, seq_len, times, num_infusion), y = batch
                        xs = model.normalization(xs)

                        xs = xs.to(device)
                        times = times.to(device)

                        # static is a list; stack as in training_step
                        static = torch.stack(static, dim=1).float().to(device)

                        seq_len = seq_len.to(device) if torch.is_tensor(seq_len) else torch.tensor(seq_len, device=device)

                        if add_num_infusion and num_infusion is not None:
                            num_infusion = num_infusion.to(device)

                        y_pred_logits = model(xs, static, times, seq_len, num_infusion).squeeze().float()

                        mask = None
                        gcsf_groundtruth = None
                    else:
                        # data_prep return signatures differ per model
                        if model_option == 'duett':
                            # no time_calibration argument
                            xs_combined, y, mask, num_infusion, _ = model.data_prep(batch)
                            gcsf_groundtruth = None
                        elif model_option in ['lasttp_MLP']:
                            xs_combined, y, mask, num_infusion, _ = model.data_prep(
                                batch, time_calibration=model.time_calibration,
                            )
                            gcsf_groundtruth = None
                        elif model_option in ['rnn', 'rnn_deephit']:
                            xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                                batch, time_calibration=model.time_calibration,
                            )
                            gcsf_groundtruth = None
                        elif include_gcsf:
                            xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                                batch, time_calibration=model.time_calibration,
                            )
                        else:
                            xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                                batch, time_calibration=model.time_calibration,
                            )
                            gcsf_groundtruth = None

                        # binary tasks: dummy infusion number 0
                        if num_infusion is None:
                            batch_size = xs_combined.shape[0]
                            num_infusion = torch.zeros(batch_size, dtype=torch.float32)
                            if batch_count == 0:
                                print(f"DEBUG: Created dummy num_infusion for binary task, shape: {num_infusion.shape}")

                        xs_combined = xs_combined.to(device)
                        mask = mask.to(device)
                        if add_num_infusion and num_infusion is not None:
                            num_infusion = num_infusion.to(device)
                        if gcsf_groundtruth is not None:
                            gcsf_groundtruth = gcsf_groundtruth.to(device)

                        if model_option in ['rnn', 'rnn_deephit', 'lasttp_MLP', 'duett']:
                            y_pred_logits = model(xs_combined, src_key_padding_mask=mask, num_infusion=num_infusion)
                        else:
                            y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                                  num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                    # side-effect tasks: y = (event, time_to_event); binary: time_to_event = 1
                    if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                        fail_indicators = torch.as_tensor(y[0], device=device)
                        time_to_event = torch.as_tensor(y[1], device=device)
                    else:
                        fail_indicators = torch.as_tensor(y, device=device)
                        time_to_event = torch.ones_like(fail_indicators, device=device).float()

                    if batch_count == 0:
                        print("\n=== DEBUG: First Batch Info ===")
                        print(f"Model: {model_option}, Fold: {val_fold}")
                        if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                            print(f"y[0] type: {type(y[0])}, shape/value: {y[0] if not torch.is_tensor(y[0]) else y[0].shape}")
                            print(f"y[1] type: {type(y[1])}, shape/value: {y[1] if not torch.is_tensor(y[1]) else y[1].shape}")
                        else:
                            print(f"y type: {type(y)}, shape/value: {y if not torch.is_tensor(y) else y.shape}")
                        print(f"fail_indicators shape: {fail_indicators.shape}")
                        print(f"time_to_event shape: {time_to_event.shape}")
                        print(f"y_pred_logits shape: {y_pred_logits.shape}")
                        print(f"num_infusion shape: {num_infusion.shape if num_infusion is not None else None}")
                        print("="*50 + "\n")

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

            print(f"\n=== DEBUG: Dataset Summary for {model_option} fold {val_fold} ===")
            print(f"Total batches: {batch_count}")
            print(f"Total samples: {sum(t.numel() for t in all_y_events)}")
            print(f"Positive events: {sum((t == 1).sum().item() for t in all_y_events)}")

            all_y_preds = torch.cat(all_y_preds, dim=0)
            all_y_events = torch.cat(all_y_events, dim=0)
            all_time_to_events = torch.cat(all_time_to_events, dim=0)
            all_num_infusions = torch.cat(all_num_infusions, dim=0)

            print(f"After concatenation - Total samples: {len(all_y_events)}, Positive: {(all_y_events == 1).sum().item()}")
            print("="*50 + "\n")

            print("PRED:", all_y_preds.shape, all_y_preds[:5].tolist(),
                "sum=", all_y_preds.sum().item())
            print("EVT :", all_y_events[:5].tolist(), "sum=", all_y_events.sum().item())
            print("TTE :", all_time_to_events[:5].tolist())
            print("INF :", all_num_infusions[:5].tolist(), "uniq=", len(torch.unique(all_num_infusions)))

            # n and n_pos should be identical across models (same test set)
            n_total = len(all_y_events)
            n_pos_overall = (all_y_events == 1).sum().item()
            print(f"[TEST SET CHECK] model={model_option} fold={val_fold}: "
                  f"n={n_total}, n_pos_overall={n_pos_overall} ({n_pos_overall / n_total:.2%})")
            model_performance_dict[model_option][val_fold]['n_test'] = n_total
            model_performance_dict[model_option][val_fold]['n_pos_overall'] = n_pos_overall
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                n_pos_next_infusion = ((all_y_events == 1) & (all_time_to_events == 1)).sum().item()
                print(f"[TEST SET CHECK] model={model_option} fold={val_fold}: "
                      f"n_pos_next_infusion={n_pos_next_infusion} ({n_pos_next_infusion / n_total:.2%})")
                model_performance_dict[model_option][val_fold]['n_pos_next_infusion'] = n_pos_next_infusion


            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                c_idx_dict = calculate_weighted_c_index_by_infusion(
                    all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                    mode='val', option=option, extended_metrics=True
                )
                model_performance_dict[model_option][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', 0.5)
                model_performance_dict[model_option][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', 0.5)
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
                model_performance_dict[model_option][val_fold]['test_auprc_nextinf_overall'] = c_idx_dict.get('auprc_next_infusion_overall', None)
                model_performance_dict[model_option][val_fold]['test_auprc_nextinf_weighted'] = c_idx_dict.get('auprc_next_infusion_weighted', None)
                print(f"C-index overall: {c_idx_dict.get('c_index_overall', 0.5)}, C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5)}")
                print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
                print(f"AUPRC next-infusion weighted: {c_idx_dict.get('auprc_next_infusion_weighted')}")


                if ensemble:
                    model_performance_dict[model_option][val_fold]['y_preds'] = all_y_preds.cpu().numpy()
                    model_performance_dict[model_option][val_fold]['y_events'] = all_y_events.cpu().numpy()
                    model_performance_dict[model_option][val_fold]['time_to_events'] = all_time_to_events.cpu().numpy()
                    model_performance_dict[model_option][val_fold]['num_infusions'] = all_num_infusions.cpu().numpy()
            else:
                # binary tasks: one AUROC/AUPRC over the whole test set
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
                auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()

                model_performance_dict[model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")


                if ensemble:
                    model_performance_dict[model_option][val_fold]['y_preds'] = all_y_preds.cpu().numpy()
                    model_performance_dict[model_option][val_fold]['y_events'] = all_y_events.cpu().numpy()
        else:
            test_X_l = []
            test_y_l = []
            test_num_infusions = []

            # tree / TabPFN models: build flat feature vectors sample by sample
            first_sample = data_module.test_dataset.__getitem__(0)
            print(f"DEBUG RF: First sample structure: {[type(x) for x in first_sample]}")
            print(f"DEBUG RF: First sample[0] structure: {type(first_sample[0])}, len={len(first_sample[0]) if hasattr(first_sample[0], '__len__') else 'N/A'}")
            if len(first_sample[0]) >= 3:
                print("DEBUG RF: Has num_infusion at index 2")

            for i in range(len(data_module.test_dataset.dataset_DMP_IDs)):
                if model_option in ['lasttp_xgboost', 'lasttp_xgboost_risk']:
                    xs_tp, time_tp = data_module.test_dataset.__getitem__(i)[0][0][0].detach().clone(), data_module.test_dataset.__getitem__(i)[0][1][0].detach().clone()
                else:
                    xs_tp, time_tp = data_module.test_dataset.__getitem__(i)[0][0].detach().clone(), data_module.test_dataset.__getitem__(i)[0][1].detach().clone()
                    time_nan_mask = torch.isnan(time_tp).int().unsqueeze(-1).detach().clone()
                # zero-fill NaNs and append the missingness mask and times
                mask = torch.isnan(xs_tp).detach().clone()
                xs_tp[mask] = 0
                if model_option in ['lasttp_xgboost', 'lasttp_xgboost_risk']:
                    xs_tp = torch.cat((xs_tp, mask, time_tp.unsqueeze(0)), dim=0)
                else:
                    time_tp_combine = time_tp.unsqueeze(-1)
                    time_tp_combine[time_nan_mask.bool()] = 0.0
                    xs_tp = torch.cat((xs_tp, mask, time_tp_combine, time_nan_mask), dim=1).view(-1)
                if add_num_infusion:
                    xs_tp_infusions = data_module.test_dataset.__getitem__(i)[0][2]
                    xs_tp_infusions = torch.tensor(xs_tp_infusions, dtype=torch.float32).unsqueeze(-1)
                    xs_tp = torch.cat((xs_tp, xs_tp_infusions))
                    test_num_infusions.append(xs_tp_infusions.item())
                else:
                    test_num_infusions.append(0)
                test_X_l.append(xs_tp)
                test_y_l.append(data_module.test_dataset.ys[i])

            print(f"DEBUG RF: Collected {len(test_num_infusions)} num_infusions for {len(test_X_l)} samples")

            test_X = torch.stack(test_X_l)
            test_X = test_X.numpy()

            # TabPFN was trained on the first 1000 features
            if model_option in ['temporal_tabpfn']:
                MAX_FEATURES = 1000
                if test_X.shape[1] > MAX_FEATURES:
                    print(f"WARNING: Test data has {test_X.shape[1]} features, truncating to {MAX_FEATURES} to match training")
                    test_X = test_X[:, :MAX_FEATURES]

            # survival XGBoost returns risk scores from predict(); the rest use P(positive)
            if model_option in ['lasttp_xgboost_risk', 'temporal_xgboost_risk']:
                y_pred = model.predict(test_X)
            elif model_option in ['temporal_tabpfn']:
                y_pred_proba = model.predict_proba(test_X)
                y_pred = y_pred_proba[:, 1]
            else:
                y_pred_proba = model.predict_proba(test_X)
                y_pred = y_pred_proba[:, 1]

            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                y_event = torch.tensor([test_y_l[i][0] for i in range(len(test_y_l))])
                time_to_event = torch.tensor([test_y_l[i][1] for i in range(len(test_y_l))]).float()
            else:
                y_event = torch.tensor(test_y_l)
                time_to_event = torch.ones(len(test_y_l)).float()

            test_num_infusions = torch.tensor(test_num_infusions).int()
            y_pred = torch.tensor(y_pred)

            n_total = len(y_event)
            n_pos_overall = (y_event == 1).sum().item()
            print(f"[TEST SET CHECK] model={model_option} fold={val_fold}: "
                  f"n={n_total}, n_pos_overall={n_pos_overall} ({n_pos_overall / n_total:.2%})")
            model_performance_dict[model_option][val_fold]['n_test'] = n_total
            model_performance_dict[model_option][val_fold]['n_pos_overall'] = n_pos_overall
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                n_pos_next_infusion = ((y_event == 1) & (time_to_event == 1)).sum().item()
                print(f"[TEST SET CHECK] model={model_option} fold={val_fold}: "
                      f"n_pos_next_infusion={n_pos_next_infusion} ({n_pos_next_infusion / n_total:.2%})")
                model_performance_dict[model_option][val_fold]['n_pos_next_infusion'] = n_pos_next_infusion

            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                c_idx_dict = calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, test_num_infusions, mode='val', option=option, extended_metrics=True)
                c_index = c_idx_dict.get('c_index_raw', 0.5)
                c_index_weighted = c_idx_dict.get('c_index_weighted', 0.5)
                overall_c_index = c_idx_dict.get('c_index_overall', 0.5)
                weighted_auroc = c_idx_dict.get('auroc_next_infusion_weighted', None)
                overall_auroc = c_idx_dict.get('auroc_next_infusion_overall', None)
                weighted_auprc = c_idx_dict.get('auprc_next_infusion_weighted', None)
                overall_auprc = c_idx_dict.get('auprc_next_infusion_overall', None)

                model_performance_dict[model_option][val_fold]['test_cindex_overall_valid'] = overall_c_index
                model_performance_dict[model_option][val_fold]['test_cindex_weighted'] = c_index_weighted
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_overall'] = overall_auroc
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_weighted'] = weighted_auroc
                model_performance_dict[model_option][val_fold]['test_auprc_nextinf_overall'] = overall_auprc
                model_performance_dict[model_option][val_fold]['test_auprc_nextinf_weighted'] = weighted_auprc

                print(f"C-index: {c_index}, Weighted C-index: {c_index_weighted}, Overall C-index: {overall_c_index}")
                print(f"Weighted AUROC (next infusion): {weighted_auroc}, Overall AUROC (next infusion): {overall_auroc}")
                print(f"Weighted AUPRC (next infusion): {weighted_auprc}, Overall AUPRC (next infusion): {overall_auprc}")


                if ensemble:
                    model_performance_dict[model_option][val_fold]['y_preds'] = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
                    model_performance_dict[model_option][val_fold]['y_events'] = y_event.cpu().numpy() if torch.is_tensor(y_event) else np.array(y_event)
                    model_performance_dict[model_option][val_fold]['time_to_events'] = time_to_event.cpu().numpy() if torch.is_tensor(time_to_event) else np.array(time_to_event)
                    model_performance_dict[model_option][val_fold]['num_infusions'] = np.array(test_num_infusions)
            else:
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(y_pred, y_event.int()).item()
                auprc_score = auprc_fn(y_pred, y_event.int()).item()

                model_performance_dict[model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")


                if ensemble:
                    model_performance_dict[model_option][val_fold]['y_preds'] = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
                    model_performance_dict[model_option][val_fold]['y_events'] = y_event.cpu().numpy() if torch.is_tensor(y_event) else np.array(y_event)

            y_true = test_y_l
            y_pred = y_pred

        print("_______")

    # Ensemble: mean prediction over folds; needs predictions for every fold in this run
    if ensemble and model_option in model_performance_dict:
        all_folds_complete = all(fold in model_performance_dict[model_option] and 'y_preds' in model_performance_dict[model_option][fold]
                                  for fold in all_folds)

        if all_folds_complete:
            print(f"\n{'='*80}")
            print(f"Computing ensemble for {model_option}")
            print(f"{'='*80}")

            fold_preds = [model_performance_dict[model_option][fold]['y_preds'] for fold in all_folds]
            fold_events = [model_performance_dict[model_option][fold]['y_events'] for fold in all_folds]

            assert len(set(len(p) for p in fold_preds)) == 1, "All folds should have same number of test samples"

            ensemble_preds = np.mean(fold_preds, axis=0)
            ensemble_events = fold_events[0]

            # binary tasks: one infusion group and time_to_event=1, so weighted == overall
            ensemble_predictions_store[model_option] = {
                'y_preds': np.asarray(ensemble_preds),
                'y_events': np.asarray(ensemble_events),
                'time_to_events': np.ones(len(ensemble_events), dtype=np.float32),
                'num_infusions': np.zeros(len(ensemble_events), dtype=np.float32),
            }

            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                fold_times = [model_performance_dict[model_option][fold]['time_to_events'] for fold in all_folds]
                fold_infusions = [model_performance_dict[model_option][fold]['num_infusions'] for fold in all_folds]
                ensemble_times = fold_times[0]
                ensemble_infusions = fold_infusions[0]
                ensemble_predictions_store[model_option]['time_to_events'] = np.asarray(ensemble_times)
                ensemble_predictions_store[model_option]['num_infusions'] = np.asarray(ensemble_infusions)

                n_total_ens = len(ensemble_events)
                n_pos_overall_ens = int((np.asarray(ensemble_events) == 1).sum())
                n_pos_next_infusion_ens = int(
                    ((np.asarray(ensemble_events) == 1) & (np.asarray(ensemble_times) == 1)).sum()
                )
                print(f"[TEST SET CHECK] model={model_option} ensemble: "
                      f"n={n_total_ens}, n_pos_overall={n_pos_overall_ens} "
                      f"({n_pos_overall_ens / n_total_ens:.2%}), "
                      f"n_pos_next_infusion={n_pos_next_infusion_ens} "
                      f"({n_pos_next_infusion_ens / n_total_ens:.2%})")

                ensemble_preds_tensor = torch.tensor(ensemble_preds, dtype=torch.float32)
                ensemble_events_tensor = torch.tensor(ensemble_events, dtype=torch.float32)
                ensemble_times_tensor = torch.tensor(ensemble_times, dtype=torch.float32)
                ensemble_infusions_tensor = torch.tensor(ensemble_infusions, dtype=torch.float32)

                c_idx_dict = calculate_weighted_c_index_by_infusion(
                    ensemble_preds_tensor, ensemble_events_tensor, ensemble_times_tensor, ensemble_infusions_tensor,
                    mode='val', option=option, extended_metrics=True
                )

                auroc_auprc_ci = {}
                if ci_method != 'none':
                    print(f"\n--- Computing confidence intervals for AUROC/AUPRC (method: {ci_method}) ---")
                    next_infusion_event = ((ensemble_events_tensor == 1) & (ensemble_times_tensor == 1)).float()
                    next_infusion_event_np = next_infusion_event.cpu().numpy()
                    ensemble_preds_np = np.array(ensemble_preds)

                    # bootstrap only the infusion groups with both classes, as the point estimate does
                    infusions_np = ensemble_infusions_tensor.cpu().numpy()
                    ci_mask = np.zeros(len(next_infusion_event_np), dtype=bool)
                    for _g in np.unique(infusions_np):
                        _m = infusions_np == _g
                        _npos = next_infusion_event_np[_m].sum()
                        if 0 < _npos < _m.sum():
                            ci_mask |= _m
                    if ci_mask.any():
                        next_infusion_event_np = next_infusion_event_np[ci_mask]
                        ensemble_preds_np = ensemble_preds_np[ci_mask]
                        infusions_np = infusions_np[ci_mask]
                    print(f"CI computed on {ci_mask.sum()}/{len(ci_mask)} samples "
                          f"(same infusion-group filter as the point estimate)")

                    if ci_method == 'bootstrap':
                        from sklearn.metrics import roc_auc_score, average_precision_score

                        n_bootstrap = n_bootstrap
                        auroc_scores = []
                        auprc_scores = []
                        # weighted AUROC CI from the same resamples
                        from eval_utils import nextinf_auroc_weighted_overall
                        auroc_weighted_scores = []

                        for i in range(n_bootstrap):
                            indices = np.random.choice(len(next_infusion_event_np),
                                                      size=len(next_infusion_event_np),
                                                      replace=True)
                            y_boot = next_infusion_event_np[indices]
                            p_boot = ensemble_preds_np[indices]
                            inf_boot = infusions_np[indices]

                            if len(np.unique(y_boot)) < 2:
                                continue

                            auroc_scores.append(roc_auc_score(y_boot, p_boot))
                            auprc_scores.append(average_precision_score(y_boot, p_boot))
                            _w, _ = nextinf_auroc_weighted_overall(p_boot, y_boot, inf_boot)
                            if not np.isnan(_w):
                                auroc_weighted_scores.append(_w)

                        if len(auroc_weighted_scores) > 0:
                            auroc_auprc_ci['auroc_weighted_ci_lower'] = np.percentile(auroc_weighted_scores, 2.5)
                            auroc_auprc_ci['auroc_weighted_ci_upper'] = np.percentile(auroc_weighted_scores, 97.5)
                            print(f"Weighted AUROC 95% CI: [{auroc_auprc_ci['auroc_weighted_ci_lower']:.4f}, "
                                  f"{auroc_auprc_ci['auroc_weighted_ci_upper']:.4f}]")

                        if len(auroc_scores) > 0:
                            auroc_auprc_ci['auroc_ci_lower'] = np.percentile(auroc_scores, 2.5)
                            auroc_auprc_ci['auroc_ci_upper'] = np.percentile(auroc_scores, 97.5)
                            auroc_auprc_ci['auprc_ci_lower'] = np.percentile(auprc_scores, 2.5)
                            auroc_auprc_ci['auprc_ci_upper'] = np.percentile(auprc_scores, 97.5)
                            print(f"AUROC 95% CI: [{auroc_auprc_ci['auroc_ci_lower']:.4f}, {auroc_auprc_ci['auroc_ci_upper']:.4f}]")
                            print(f"AUPRC 95% CI: [{auroc_auprc_ci['auprc_ci_lower']:.4f}, {auroc_auprc_ci['auprc_ci_upper']:.4f}]")

                ensemble_name = f'{model_option}_ensemble'
                ensemble_results = {
                    'n_test': n_total_ens,
                    'n_pos_overall': n_pos_overall_ens,
                    'n_pos_next_infusion': n_pos_next_infusion_ens,
                    'test_cindex_overall_valid': c_idx_dict.get('c_index_overall', 0.5),
                    'test_cindex_weighted': c_idx_dict.get('c_index_weighted', 0.5),
                    'test_auroc_nextinf_overall': c_idx_dict.get('auroc_next_infusion_overall', None),
                    'test_auroc_nextinf_weighted': c_idx_dict.get('auroc_next_infusion_weighted', None),
                    'test_auprc_nextinf_overall': c_idx_dict.get('auprc_next_infusion_overall', None),
                    'test_auprc_nextinf_weighted': c_idx_dict.get('auprc_next_infusion_weighted', None),
                    'test_precision_nextinf_weighted': c_idx_dict.get('precision_next_infusion_weighted', None),
                    'test_recall_nextinf_weighted': c_idx_dict.get('recall_next_infusion_weighted', None),
                }
                if auroc_auprc_ci:
                    for key, value in auroc_auprc_ci.items():
                        ensemble_results[key] = value


                model_performance_dict[ensemble_name] = {'ensemble': ensemble_results}

                print(f"Ensemble C-index overall: {c_idx_dict.get('c_index_overall', 0.5):.4f}, "
                      f"C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5):.4f}")
                print(f"Ensemble AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, "
                      f"AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
            else:
                # binary tasks
                n_total_ens = len(ensemble_events)
                n_pos_overall_ens = int((np.asarray(ensemble_events) == 1).sum())
                print(f"[TEST SET CHECK] model={model_option} ensemble: "
                      f"n={n_total_ens}, n_pos_overall={n_pos_overall_ens} "
                      f"({n_pos_overall_ens / n_total_ens:.2%})")

                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')
                ensemble_preds_tensor = torch.tensor(ensemble_preds, dtype=torch.float32)
                ensemble_events_tensor = torch.tensor(ensemble_events, dtype=torch.int32)

                metrics_with_ci = {
                    'n_test': n_total_ens,
                    'n_pos_overall': n_pos_overall_ens,
                    'auroc': auroc_fn(ensemble_preds_tensor, ensemble_events_tensor).item(),
                    'auprc': auprc_fn(ensemble_preds_tensor, ensemble_events_tensor).item(),
                }
                print(f"Ensemble AUROC: {metrics_with_ci['auroc']:.4f}, AUPRC: {metrics_with_ci['auprc']:.4f}")

                if ci_method != 'none':
                    print(f"\n--- Computing confidence intervals for AUROC/AUPRC (method: {ci_method}) ---")
                    ensemble_events_np = np.array(ensemble_events)
                    ensemble_preds_np = np.array(ensemble_preds)

                    if ci_method == 'bootstrap':
                        from sklearn.metrics import roc_auc_score, average_precision_score

                        auroc_scores = []
                        auprc_scores = []

                        for i in range(n_bootstrap):
                            indices = np.random.choice(len(ensemble_events_np),
                                                      size=len(ensemble_events_np),
                                                      replace=True)
                            y_boot = ensemble_events_np[indices]
                            p_boot = ensemble_preds_np[indices]

                            if len(np.unique(y_boot)) < 2:
                                continue

                            auroc_scores.append(roc_auc_score(y_boot, p_boot))
                            auprc_scores.append(average_precision_score(y_boot, p_boot))

                        if len(auroc_scores) > 0:
                            metrics_with_ci['auroc_ci_lower'] = np.percentile(auroc_scores, 2.5)
                            metrics_with_ci['auroc_ci_upper'] = np.percentile(auroc_scores, 97.5)
                            metrics_with_ci['auprc_ci_lower'] = np.percentile(auprc_scores, 2.5)
                            metrics_with_ci['auprc_ci_upper'] = np.percentile(auprc_scores, 97.5)
                            print(f"AUROC 95% CI: [{metrics_with_ci['auroc_ci_lower']:.4f}, {metrics_with_ci['auroc_ci_upper']:.4f}]")
                            print(f"AUPRC 95% CI: [{metrics_with_ci['auprc_ci_lower']:.4f}, {metrics_with_ci['auprc_ci_upper']:.4f}]")

                ensemble_name = f'{model_option}_ensemble'
                model_performance_dict[ensemble_name] = {
                    'ensemble': metrics_with_ci
                }

            # save the ensemble predictions now so they survive an interrupted run
            predictions_dir = f'{results_path}/eval_across_models{_nm_suffix}{_valthresh_suffix}/ensemble_predictions'
            os.makedirs(predictions_dir, exist_ok=True)
            predictions_save_path = os.path.join(
                predictions_dir, f'{modules}_{option}_{dataset}{_finetuned_suffix}_{model_option}.pickle'
            )
            with open(predictions_save_path, 'wb') as f:
                pickle.dump(ensemble_predictions_store[model_option], f)
            print(f"Saved raw ensemble predictions for {model_option} to {predictions_save_path}")

            print(f"{'='*80}\n")

    # rewrite the CSV after each model
    if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
        all_columns = ['model', 'val_fold', 'n_test', 'n_pos_overall', 'n_pos_next_infusion',
                    'test_cindex_overall_valid', 'test_cindex_weighted',
                    'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                    'test_auprc_nextinf_overall', 'test_auprc_nextinf_weighted']

        # auroc_ci_* is the CI of the overall next-infusion AUROC, auroc_weighted_ci_* of the weighted one
        if ci_method == 'bootstrap':
            all_columns.extend(['auroc_ci_lower', 'auroc_ci_upper', 'auprc_ci_lower', 'auprc_ci_upper',
                                'auroc_weighted_ci_lower', 'auroc_weighted_ci_upper'])


    else:
        # binary tasks have a single AUROC
        all_columns = ['model', 'val_fold', 'n_test', 'n_pos_overall', 'auroc', 'auprc']

        if ci_method == 'bootstrap':
            all_columns.extend(['auroc_ci_lower', 'auroc_ci_upper', 'auprc_ci_lower', 'auprc_ci_upper'])

    model_performance_df = pd.DataFrame(columns=all_columns)
    for model_name in model_performance_dict.keys():
        for fold in model_performance_dict[model_name].keys():
            row = {'model': model_name, 'val_fold': fold}
            fold_data = {k: v for k, v in model_performance_dict[model_name][fold].items()
                        if k not in ['y_preds', 'y_events', 'time_to_events', 'num_infusions']}
            row.update(fold_data)
            row = [row.get(col, None) for col in all_columns]
            model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    model_performance_df.to_csv(save_name, index=False)

# ensemble_predictions/*.pickle hold y_preds, y_events, time_to_events, num_infusions
predictions_dir_summary = f'{results_path}/eval_across_models{_nm_suffix}{_valthresh_suffix}/ensemble_predictions'
print(f"\nRaw ensemble predictions for all completed models saved under: {predictions_dir_summary}")
