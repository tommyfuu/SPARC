import os
import pickle
import sys, inspect
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
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
# Use the fixed versions of data loaders and model
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset
from MSK_featurename_list import build_module_index_config
from SPARC_inf_risk_modular import *
import json
import ast
data_source_path=sparc_paths.DATA_ROOT + '/'

from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, F1Score, AveragePrecision, Accuracy, Precision, Recall

from eval_utils import calculate_weighted_c_index_by_infusion


import argparse
parser = argparse.ArgumentParser(
            prog='design_choice_comparison_3_medsasfeatures',
            description='Compare SPARC models: medications as features vs. not as features',
            epilog='Comparison script for med_as_feature False vs True')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'AKI', help='side effect option: anemia, neutropenia, febrile_neutropenia, AKI, Liver_toxicity')
parser.add_argument("-m", "--modules", type=str, default = 'parsimonious', help='module category')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset: MSK, MSK_timeshift')

args = parser.parse_args()
rseed = args.rseed
option_base = args.option
modules = args.modules
dataset = args.dataset


if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option_base)):
    raise ValueError(f"Invalid option: {option_base}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")

# Define medication configurations for each side effect
med_configs = {
    'AKI': ["CISPLATIN", "IFOSFAMIDE", "PEMETREXED", "METHOTREXATE", "PAMIDRONATE", "ZOLEDRONIC ACID"],
    'Liver_toxicity': ["IPILIMUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "PEMBROLIZUMAB", "AVELUMAB", "CEMIPLIMAB", "DOSTARLIMAB"],
    'febrile_neutropenia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                            ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
    'anemia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
               ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
    'neutropenia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                    ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
}

# Get medication configuration for current task
med_of_choice = med_configs[option_base]

# Convert to list format expected by the data loader
def process_med_list(med_list):
    """Convert medication list to the format expected by data loader."""
    result = []
    for med in med_list:
        if isinstance(med, list):
            result.append(med)
        else:
            result.append([med])
    return result

med_of_choice_processed = process_med_list(med_of_choice)
print("Medication configuration:", med_of_choice_processed)

## HYPERPARAMETERS ##
batch_size = 512
competing_risk_death_check = False
include_gcsf = False
days = 30
add_num_infusion = True

# Architecture parameters for 060726 models
use_module_tokenizer = True   # -umt True
use_continuous_time = True    # -uct True
post_mlp_dropout = 0.2        # -pmd 0.2
use_sigmoid = True
schedule_option = 'default'
learning_rate = 1e-4
missing_mask = True
temporal_aggregation = 'attention'

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
    option=f'{option_base}_risk',  # Use risk version for data loading
    modules=modules,
    valfold=0,
    mode='test',
    rseed=rseed,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=include_gcsf,
    data_fraction_option='default',
    quick_test=False,
)

print("modular_featuredim_dict", modular_featuredim_dict)
print("Data loading complete")

# Define model paths
all_model_path = sparc_paths.SPARC_MODEL_ROOT

# Determine loss function for med_as_feature models based on task
# AKI and Liver_toxicity use RISK, others use HYBRID
if option_base in ['AKI', 'Liver_toxicity']:
    medasfeature_loss = 'risk'
else:
    medasfeature_loss = 'hybrid'

# Define model configurations
model_configs = {
    'med_as_feature_False': {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{all_model_path}/{modules}_MSK_060726/SPARC_direct_models/parsimonious_direct_models_risk_{option_base}_20tp_topchemos_att_modular',
        'description': 'Medications NOT as features (default)',
        'med_as_feature': False
    },
    'med_as_feature_True': {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{all_model_path}/{modules}_MSK_060726/med_as_feature_exp/parsimonious_direct_models_{medasfeature_loss}_{option_base}_20tp_topchemos_att_medasfeature_modular',
        'description': 'Medications as features',
        'med_as_feature': True
    }
}

# # if AKI add _redo to the paths
# if option_base == 'AKI':
#     model_configs['med_as_feature_False']['path'] += '_redo'
#     model_configs['med_as_feature_True']['path'] += '_redo'

# Generate the split dict from fold dict
if dataset == 'MSK':
    with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as handle:
        fold_dict = pickle.load(handle)
elif dataset == 'MSK_timeshift':
    fold_dict = {10: data_dict_lab_dict['age'].keys()}
    # Get all DMP_ID in the MSK dataset and exclude from the fold_dict
    data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
    with open(data_path, 'rb') as handle:
        og_set = pickle.load(handle)
    fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]

testDMP_IDs = fold_dict[10]
testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}

save_name = f'{all_model_path}/results_vis_060726/design_choices_3_medsasfeatures/SPARC_medsasfeatures_{modules}_{option_base}_{dataset}.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)

model_performance_dict = {}

# Check if the save csv file exists, if it does, load it
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    performance_df_existing = saved_df.dropna()
    for index, row in performance_df_existing.iterrows():
        model_name = row['model']
        val_fold = int(row['val_fold'])
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        model_performance_dict[model_name][val_fold] = {
            'test_cindex_overall_valid': row.get('test_cindex_overall_valid'),
            'test_cindex_weighted': row.get('test_cindex_weighted'),
            'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
            'test_auroc_nextinf_weighted': row.get('test_auroc_nextinf_weighted'),
        }
    print("Loaded existing model performance data from csv")

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
model_set_option = 'side_effects_risk'

# Build module index config for 060726 models
module_index_cfg = build_module_index_config(modular_featuredim_dict)

# Calculate n_static_features
static_module_names = ['gender', 'age']
n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)

for model_name, config in model_configs.items():
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"Option: {config['option']}, Temporal Aggregation: {config['temporal_aggregation']}")
    print(f"Med as feature: {config['med_as_feature']}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    # Only load dataloader if NOT (already exist for all 10 folds for this model)
    if model_name in model_performance_dict and False not in [fold in model_performance_dict[model_name] for fold in all_folds]:
        print(f"Data module for model {model_name} already exists for all folds, skipping")
        continue

    # Create data module with appropriate med_as_feature setting
    data_module = FlexibleFeatureDataModule(
        data_dict_lab_dict, split_dict, dataset=dataset, component=component,
        batch_size=512, max_time_len=20, num_workers=1,
        task=f'{option_base}_risk', binary_days=days, med=med_of_choice_processed,
        med_as_feature=config['med_as_feature'], med_l=med_of_choice_processed,
        advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
        include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False
    )
    print(f"Data module initiated for task {option_base}_risk with med_as_feature={config['med_as_feature']}")

    # Check performance for each fold
    for val_fold in all_folds:
        print(f"\nModel: {model_name}, Fold: {val_fold}")
        pl.seed_everything(rseed)

        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        if val_fold in model_performance_dict[model_name]:
            print(f"Performance for model {model_name} fold {val_fold} already exists, skipping")
            continue
        else:
            model_performance_dict[model_name][val_fold] = {}

        # Get model path
        model_folder = f'{config["path"]}/fold{val_fold}'

        print(f"Looking for model in: {model_folder}")
        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist: {model_folder}")
            print(f"Skipping fold {val_fold}")
            continue

        existing_files = os.listdir(model_folder)
        # Find the checkpoint that has the highest val_loss (i.e. the one that is not last.ckpt)
        epoch_files = [el for el in existing_files if 'epoch' in el]
        if len(epoch_files) == 0:
            print(f"WARNING: No epoch checkpoint found in {model_folder}")
            print(f"Skipping fold {val_fold}")
            continue

        optimal_model_path = epoch_files[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading model from: {optimal_model_path}")

        # Load the model with appropriate configuration
        try:
            model = SPARC.load_from_checkpoint(
                optimal_model_path,
                modular_featuredim_dict=modular_featuredim_dict,
                option=model_set_option,
                advanced_model="default",
                max_time_len=20,
                med_as_feature=config['med_as_feature'],
                meds_l=med_of_choice_processed,
                loss_option=config['loss_option'],
                schedule=schedule_option,
                side_effects_num_infusion=add_num_infusion,
                include_gcsf=include_gcsf,
                use_gcsf=include_gcsf,
                use_sigmoid=use_sigmoid,
                temporal_aggregation=config['temporal_aggregation'],
                missing_mask=missing_mask,
                module_index_cfg=module_index_cfg,
                use_module_tokenizer=use_module_tokenizer,
                use_continuous_time=use_continuous_time,
                n_static_features=n_static_features,
                post_mlp_dropout=post_mlp_dropout,
                lr=learning_rate,
            )
            print(f"Model loaded successfully with med_as_feature={config['med_as_feature']}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Skipping fold {val_fold}")
            continue

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model on device: {device}")

        # Collect all predictions and ground truths from the entire test dataset
        all_y_preds = []
        all_y_events = []
        all_time_to_events = []
        all_num_infusions = []

        batch_count = 0
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # Prepare batch data (data_prep works with CPU data)
                if include_gcsf:
                    xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                        primary_ctype_returned=model.primary_ctype_returned,
                    )
                else:
                    xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                        perturbation_index=model.perturbation_index,
                        primary_ctype_returned=model.primary_ctype_returned,
                    )
                    gcsf_groundtruth = None

                # Move prepared data to GPU
                # When med_as_feature=True, xs_combined is a tuple (xs, meds_data)
                if isinstance(xs_combined, tuple):
                    xs_combined = tuple(x.to(device) for x in xs_combined)
                else:
                    xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                # Get predictions
                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Extract ground truth - ensure proper tensor dimensions
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
                all_num_infusions.append(num_infusion.detach().cpu())

                batch_count += 1

        # Concatenate all batches
        all_y_preds = torch.cat(all_y_preds, dim=0)
        all_y_events = torch.cat(all_y_events, dim=0)
        all_time_to_events = torch.cat(all_time_to_events, dim=0)
        all_num_infusions = torch.cat(all_num_infusions, dim=0)

        # Evaluate on the entire test dataset
        c_idx_dict = calculate_weighted_c_index_by_infusion(
            all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
            mode='val', option=config['option']
        )

        # Extract metrics
        model_performance_dict[model_name][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', None)
        model_performance_dict[model_name][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)

        print(f"C-index overall: {c_idx_dict.get('c_index_overall')}, C-index weighted: {c_idx_dict.get('c_index_weighted')}")
        print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")

        # Save after each fold
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted', 'description']
        model_performance_df = pd.DataFrame(columns=all_columns)
        for mn in model_performance_dict.keys():
            for vf in model_performance_dict[mn].keys():
                row = {'model': mn, 'val_fold': vf, 'description': model_configs[mn]['description']}
                row.update(model_performance_dict[mn][vf])
                row = [row[col] if col in row else None for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
        model_performance_df.to_csv(save_name, index=False)
        print(f"Saved results to {save_name}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

# Calculate and print summary statistics
print("\nSummary Statistics (mean ± std across folds):")
print("-" * 80)
summary_df = pd.read_csv(save_name)
for model_name in model_configs.keys():
    if model_name in summary_df['model'].values:
        model_data = summary_df[summary_df['model'] == model_name]
        print(f"\n{model_name} ({model_configs[model_name]['description']}):")
        for metric in ['test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted']:
            if metric in model_data.columns:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

print(f"\nResults saved to: {save_name}")
print("Done!")
