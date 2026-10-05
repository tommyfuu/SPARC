"""DC6: how elapsed time enters the model (Fourier embedding vs raw dt vs none), batch_first=True.

Usage: python3 design_choice_comparison_6_time_representation.py -o AKI -ds MSK -m parsimonious
"""
import os
import pickle
import sys, inspect
import pandas as pd
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
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset
from SPARC_inf_risk_modular import SPARC as SPARC_modular

# Per-module tokenizer config (use_module_tokenizer=True)
from MSK_featurename_list import build_module_index_config

import torch
import torch.nn as nn
import pytorch_lightning as pl
import re


from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, F1Score, AveragePrecision, Accuracy, Precision, Recall

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

import argparse
parser = argparse.ArgumentParser(
            prog='across_models_comparison',
            description='calculate the model performance comparison across models',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-fl", "--folds", type=str, default='all', help="Comma-separated fold indices to evaluate (e.g. '0,1'), or 'all' for 0-9. Useful for a quick single-fold check before a full run.")
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-m", "--modules", type=str, default = 'practical', help='module category, either "all", "allnonICD" or connect categories by dashes such as')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset used in training, options: MSK, MSK_timeshift, stanfordEHRSHOT')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-sp", "--save_preds", type=str, default = 'True', help='whether to save the predictions or not, options: True, False')
parser.add_argument("-rt", "--retrain", type=str, default = 'False', help='whether to retrain the model or not, options: True, False; only legit for AKI_2wks for the MIMICIV_hosp dataset')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-cp", "--caia_portion", type=str, default = "prior", help="prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021")
parser.add_argument("-z", "--zeroshot", type=str, default = 'True', help='whether to use zeroshot (MSK-trained) models for MIMICIV tasks; when False, finetuned MIMICIV models are used and _finetuned is appended to save names, options: True, False')
parser.add_argument("-federated", "--federated", action='store_true', help='whether to only evaluate federated models (SPARC-P) or to evaluate all models including benchmarks')
parser.add_argument("-dsp", "--data_source_path", type=str, default=sparc_paths.DATA_ROOT + '/', help='path to data directory')

parser.add_argument("-ar", "--ablation_root", type=str,
                    default='/data1/morrisq/fuc/SPARC_ablation_direct_092326',
                    help='root holding the ablation arms, one subdirectory per arm')
parser.add_argument("-rr", "--results_root", type=str, default=None,
                    help='where to write the CSV (default: <ablation_root>/results)')
args = parser.parse_args()
rseed = args.rseed
option = args.option
med_of_choice = args.med_of_choice
days = args.days
modules = args.modules
dataset = args.dataset
save_preds = args.save_preds.lower() == 'true'
retrain = args.retrain.lower() == 'true'
add_num_infusion = args.add_num_infusion.lower() == 'true'
mode = args.mode.lower()
caia_portion = args.caia_portion.lower()
zeroshot = args.zeroshot.lower() == 'true'
federated = args.federated
data_source_path = args.data_source_path

# Arms ('fourier' is the control):
#   fourier  -uct True  -traw False : learned 256 sin/cos + Linear(512,512)   (A9)
#   rawtime  -uct False -traw True  : scalar dt through nn.Linear(1,512)      (A16)
#   notime   -uct False -traw False : no time path                            (A15)
all_models = ['fourier', 'rawtime', 'notime']
# SPARC_DC10_BF=False swaps rawtime for the batch_first=False run (A17).
_BF = os.environ.get('SPARC_DC10_BF', 'True') == 'True'
_RAW_ROOT = 'A16_rawtime_bf' if _BF else 'A17_rawtime_nobf'

ARM_ROOT                 = {'fourier': 'A9_batchfirst', 'rawtime': _RAW_ROOT, 'notime': 'A15_notime_bf'}
ARM_USE_MODULE_TOKENIZER = {'fourier': True,        'rawtime': True,        'notime': True}
ARM_USE_CONTINUOUS_TIME  = {'fourier': True,        'rawtime': False,       'notime': False}
ARM_TIME_AS_RAW_FEATURE  = {'fourier': False,       'rawtime': True,        'notime': False}
ARM_BATCH_FIRST          = {'fourier': True,        'rawtime': _BF,         'notime': True}
ARM_MISSING_MASK         = {'fourier': True,        'rawtime': True,        'notime': True}
ARM_TEMPORAL_AGGREGATION = {'fourier': 'attention', 'rawtime': 'attention', 'notime': 'attention'}


if not (bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)) or option in ['binary', 'binary_icu_pre']):
    raise ValueError(f"Invalid option: {option}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# Medication combinations, e.g. 'CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE'
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

# Must match training: -umt True -uct True -pmd 0.2 -tp attention
use_module_tokenizer = True
use_continuous_time = True
post_mlp_dropout = 0.2
use_sigmoid = True            # eval mode, not recalibration
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

# Per-module tokenizer config
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

ablation_root = args.ablation_root
results_root = args.results_root or f'{ablation_root}/results'
# Model folder names by task type
if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    new_model_name = f'{modules}_direct_models_{option}_20tp_topchemos_att_modular'
else:
    base_option = option.replace('_risk', '').replace('_next_risk', '')

    # anemia/neutropenia/febrile_neutropenia were trained with hybrid, AKI/Liver_toxicity with risk
    new_model_type_suffix = 'hybrid' if base_option in ['anemia', 'neutropenia', 'febrile_neutropenia'] else 'risk'
    new_model_name = f'{modules}_direct_models_{new_model_type_suffix}_{base_option}_20tp_topchemos_att_modular'

# Model folders from the arm table
model_folder_dict = {}
for _arm in all_models:
    _sub = ARM_ROOT[_arm]
    _root = f'{ablation_root}/{_sub}'
    model_folder_dict[_arm] = f'{_root}/{modules}_{dataset}_060726/SPARC_direct_models/{new_model_name}'

print("=" * 80)
print("MODEL COMPARISON: Continuous-time Representation")
print("=" * 80)
for _arm in all_models:
    print(f"{_arm:<11} (-umt {ARM_USE_MODULE_TOKENIZER[_arm]} -uct {ARM_USE_CONTINUOUS_TIME[_arm]} "
          f"-traw {ARM_TIME_AS_RAW_FEATURE[_arm]} -bf {ARM_BATCH_FIRST[_arm]} "
          f"-mm {ARM_MISSING_MASK[_arm]} -tp {ARM_TEMPORAL_AGGREGATION[_arm]}): {model_folder_dict[_arm]}")
print("=" * 80)

model_performance_dict = {}

save_name = (f'{results_root}/design_choices_10_rawtime' + ('_bf' if _BF else '_nobf')
             + f'/rawtime_{modules}_{option}_{dataset}.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)


# Test split; CAIA uses the split_dict built by load_dataset
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
    # All patients not in the original MSK cohort
    fold_dict = {10: data_dict_lab_dict['age'].keys()}

    data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
    with open(data_path, 'rb') as handle:
        og_set = pickle.load(handle)
    fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}
elif dataset == 'stanfordEHRSHOT':
    # Use everyone
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
# Reuse cached results if the CSV exists
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    performance_df_existing = saved_df[~saved_df['model'].str.contains('averaged')]

    is_side_effect_task = bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    # Drop rows only when all of the task's metric columns are NaN
    if is_side_effect_task:
        relevant_cols = ['test_cindex_overall_valid', 'test_cindex_weighted',
                        'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted']
        performance_df_existing = performance_df_existing.dropna(subset=relevant_cols, how='all')
    else:
        relevant_cols = ['auroc', 'auprc']
        performance_df_existing = performance_df_existing.dropna(subset=relevant_cols, how='all')

    for index, row in performance_df_existing.iterrows():
        model_name = row['model']
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

# Side-effect tasks use the _risk option; binary tasks use model_set_option='binary'
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    model_set_option = 'side_effects_risk'
    option = f'{option}_risk'
else:
    model_set_option = 'binary'

all_folds = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if args.folds.lower() == 'all'
             else [int(x) for x in args.folds.split(',') if x.strip() != ''])

# Shared data module; the missingness mask is applied model-side, so it is arm-independent.
advanced_model = 'default'
anc_mode = True if 'CAIA' in dataset else False

print("Creating data module...")
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model=advanced_model, control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False,
    anc=anc_mode
)
print("Data module initiated\n")

for model_option in all_models:
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_option}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    # Same model class for every arm
    ModelClass = SPARC_modular
    arm_use_module_tokenizer = ARM_USE_MODULE_TOKENIZER[model_option]
    arm_use_continuous_time = ARM_USE_CONTINUOUS_TIME[model_option]
    arm_time_as_raw_feature = ARM_TIME_AS_RAW_FEATURE[model_option]
    arm_missing_mask = ARM_MISSING_MASK[model_option]
    arm_temporal_aggregation = ARM_TEMPORAL_AGGREGATION[model_option]

    # Skip if already computed for all folds
    if model_option in model_performance_dict and False not in [fold in model_performance_dict[model_option] for fold in all_folds]:
        print(f"Model {model_option} already exists for all folds, skipping")
        continue

    for val_fold in all_folds:
        print(f"\n--- Fold {val_fold} ---")
        pl.seed_everything(rseed)

        if model_option not in model_performance_dict:
            model_performance_dict[model_option] = {}

        if val_fold in model_performance_dict[model_option]:
            print(f"Performance for {model_option} fold {val_fold} already exists, skipping")
            continue
        else:
            model_performance_dict[model_option][val_fold] = {}

        model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        print(f"Model folder: {model_folder}")

        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist, skipping fold {val_fold}")
            continue

        existing_files = os.listdir(model_folder)
        # Use the saved epoch checkpoint, not last.ckpt
        epoch_ckpts = [el for el in existing_files if 'epoch' in el]
        if not epoch_ckpts:
            print(f"WARNING: No epoch checkpoint found, skipping fold {val_fold}")
            continue

        optimal_model_path = epoch_ckpts[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading: {optimal_model_path}")

        # Architecture args must match training
        model = ModelClass.load_from_checkpoint(
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
            temporal_aggregation=arm_temporal_aggregation,
            missing_mask=arm_missing_mask,
            module_index_cfg=module_index_cfg if arm_use_module_tokenizer else None,
            use_module_tokenizer=arm_use_module_tokenizer,
            use_continuous_time=arm_use_continuous_time,
            time_as_raw_feature=arm_time_as_raw_feature,
            batch_first=ARM_BATCH_FIRST[model_option],
            n_static_features=n_static_features,
            post_mlp_dropout=post_mlp_dropout,
            lr=learning_rate,
        )
        print(f"Loaded {model_option} with: use_module_tokenizer={arm_use_module_tokenizer}, "
              f"missing_mask={arm_missing_mask}, temporal_aggregation={arm_temporal_aggregation}, "
              f"post_mlp_dropout={post_mlp_dropout}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model on device: {device}")

        pl.seed_everything(rseed)

        # Collect predictions over the full test set
        all_y_preds = []
        all_y_events = []
        all_time_to_events = []
        all_num_infusions = []

        batch_count = 0
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # data_prep works on CPU tensors
                if include_gcsf:
                    xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                else:
                    xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                        perturbation_index=model.perturbation_index,
                        primary_ctype_returned=model.primary_ctype_returned,
                    )
                    gcsf_groundtruth = None

                # Binary tasks have no num_infusion; use zeros
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

                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Side-effect tasks: y = (event, time_to_event); binary: y is a tensor, time set to 1
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
                    # Binary tasks: zero infusion numbers
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

            # Sanity check
            if val_fold == 0:
                print(f"[{model_option}] fold0 PRED sum: {all_y_preds.sum().item():.6f}, "
                    f"EVT sum: {all_y_events.sum().item()}, "
                    f"TTE sum: {all_time_to_events.float().sum().item():.1f}, "
                    f"N: {len(all_y_preds)}")

            print(f"After concatenation - Total samples: {len(all_y_events)}, Positive: {(all_y_events == 1).sum().item()}")
            print("="*50 + "\n")

            print("PRED:", all_y_preds.shape, all_y_preds[:5].tolist(),
                "sum=", all_y_preds.sum().item())
            print("EVT :", all_y_events[:5].tolist(), "sum=", all_y_events.sum().item())
            print("TTE :", all_time_to_events[:5].tolist())
            print("INF :", all_num_infusions[:5].tolist(), "uniq=", len(torch.unique(all_num_infusions)))


            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                # Side-effect tasks: C-index and next-infusion AUROC
                c_idx_dict = calculate_weighted_c_index_by_infusion(
                    all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                    mode='val', option=option
                )
                model_performance_dict[model_option][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', 0.5)
                model_performance_dict[model_option][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', 0.5)
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
                model_performance_dict[model_option][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
                print(f"C-index overall: {c_idx_dict.get('c_index_overall', 0.5)}, C-index weighted: {c_idx_dict.get('c_index_weighted', 0.5)}")
                print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")
            else:
                # Binary tasks: AUROC/AUPRC over the whole cohort
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
                auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()

                model_performance_dict[model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")

        print("_______")

    # Rebuild the CSV after each model to avoid duplicate rows
    if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                    'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted', 'auroc', 'auprc']
    else:
        all_columns = ['model', 'val_fold', 'auroc', 'auprc']
    model_performance_df = pd.DataFrame(columns=all_columns)
    for model_name in model_performance_dict.keys():
        for fold in model_performance_dict[model_name].keys():
            row = {'model': model_name, 'val_fold': fold}
            row.update(model_performance_dict[model_name][fold])
            row = [row.get(col, None) for col in all_columns]
            model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    model_performance_df.to_csv(save_name, index=False)

