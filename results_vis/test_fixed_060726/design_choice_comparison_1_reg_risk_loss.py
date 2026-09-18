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
from SPARC_inf_risk_fixed_060726 import *  # Use the 060726 version

# Import module tokenizer config builder for use_module_tokenizer=True
from MSK_featurename_list_060726 import build_module_index_config

import json
import ast
data_source_path=sparc_paths.DATA_ROOT + '/'

# python3 across_SPARC_att_risk_comparison.py -o AKI -ds MSK -m practical  -me "CISPLATIN"  -an True   -r 1024

from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, F1Score, AveragePrecision, Accuracy, Precision, Recall

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_weighted_c_index_by_infusion(y_pred, y_event, time_to_event, num_infusion, mode='train', option='AKI_risk'):
    """
    Calculate weighted C-index stratified by infusion number.
    Only includes infusions where C-index can be computed.

    Returns:
        dict with keys: c_index_weighted, c_index_overall, num_groups, num_samples,
                    auroc metrics, and optionally binary_loss (as tensor for backprop)
    """
    auroc_fn = AUROC(task='binary')
    accuracy_fn = Accuracy(task='binary')
    precision_fn = Precision(task='binary')
    recall_fn = Recall(task='binary')
    auprc_fn = AveragePrecision(task='binary')
    c_index = ConcordanceIndex()
    output_dict = {}

    # Ensure device consistency
    device = y_pred.device if torch.is_tensor(y_pred) else 'cpu'
    num_infusion_tensor = torch.as_tensor(num_infusion, device=device)
    y_event_tensor = torch.as_tensor(y_event, device=device)
    time_to_event_tensor = torch.as_tensor(time_to_event, device=device).float()

    unique_infusions = torch.unique(num_infusion_tensor)
    risk_scores = y_pred.squeeze()

    # ========================================================================
    # Part 1: Risk prediction metrics (C-index)
    # ========================================================================
    if 'risk' in option:
        # First pass: identify valid infusions (has both events and censoring)
        valid_infusions = []
        for infusion in unique_infusions:
            mask = num_infusion_tensor == infusion
            current_events = y_event_tensor[mask]

            if current_events.sum() > 0 and current_events.sum() < len(current_events):
                valid_infusions.append(infusion)

        # Calculate raw C-index (no stratification)
        try:
            c_idx_raw = c_index(
                risk_scores.unsqueeze(1),
                y_event_tensor.bool(),
                time_to_event_tensor
            ).item()
        except Exception:
            c_idx_raw = 0.5

        # Initialize with defaults
        c_index_weighted = 0.5
        overall_c_index = 0.5
        num_groups = 0
        total_samples = 0
        computable_results = []

        if len(valid_infusions) > 0:
            # Second pass: compute C-index for valid infusions
            for infusion in valid_infusions:
                mask = num_infusion_tensor == infusion
                current_risk = risk_scores[mask]
                current_events = y_event_tensor[mask]
                current_times = time_to_event_tensor[mask].float()

                try:
                    c_idx_val = c_index(
                        current_risk.unsqueeze(1),
                        current_events.bool(),
                        current_times
                    )
                    c_idx_value = c_idx_val.item()
                    if not torch.isnan(c_idx_val):
                        computable_results.append({
                            'infusion': infusion.item(),
                            'c_index': c_idx_value,
                            'num_samples': mask.sum().item(),
                            'mask': mask
                        })
                except Exception as e:
                    continue

        if len(computable_results) > 0:
            # Calculate weighted C-index
            total_samples = sum(r['num_samples'] for r in computable_results)
            c_index_weighted = sum(
                r['c_index'] * r['num_samples'] / total_samples
                for r in computable_results
            )

            # Calculate overall C-index
            all_mask = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
            for r in computable_results:
                all_mask |= r['mask']

            try:
                overall_c_index = c_index(
                    risk_scores[all_mask].unsqueeze(1),
                    y_event_tensor[all_mask].bool(),
                    time_to_event_tensor[all_mask]
                ).item()
            except Exception:
                overall_c_index = 0.5

            num_groups = len(computable_results)

        output_dict.update({
            'c_index_raw': c_idx_raw,
            'c_index_weighted': c_index_weighted,
            'c_index_overall': overall_c_index,
            'num_groups': num_groups,
            'num_samples': total_samples,
            'per_infusion': computable_results,
        })

    # ========================================================================
    # Part 2: Next-infusion side effect prediction (AUROC)
    # ========================================================================

    # Create binary labels: 1 if side effect occurs in next infusion, 0 otherwise
    time_to_event_tensor = time_to_event_tensor.flatten()
    next_infusion_event = ((y_event_tensor == 1) & (time_to_event_tensor == 1)).float()

    valid_infusions_auroc = []
    for infusion in unique_infusions:
        mask = num_infusion_tensor == infusion
        current_next_events = next_infusion_event[mask]

        # Valid if has both positives and negatives
        has_positives = current_next_events.sum() > 0
        has_negatives = current_next_events.sum() < len(current_next_events)

        if has_positives and has_negatives:
            valid_infusions_auroc.append(infusion)

    # Initialize defaults
    auroc_next_weighted = 0.5
    auroc_next_overall = 0.5
    accuracy_next_weighted = 0.5
    precision_next_weighted = 0.5
    recall_next_weighted = 0.5
    auprc_next_weighted = 0.5
    num_groups_auroc = 0
    num_samples_auroc = 0

    if len(valid_infusions_auroc) > 0:
        # Second pass: compute AUROC for valid infusions
        computable_auroc_results = []

        for infusion in valid_infusions_auroc:
            mask = num_infusion_tensor == infusion
            current_risk = risk_scores[mask]
            current_next_events = next_infusion_event[mask]
            try:
                current_res = {'num_samples': mask.sum().item()}
                if mode != 'train':
                    # Calculate metrics
                    current_risk = current_risk.float()
                    auroc = auroc_fn(current_risk, current_next_events.int())
                    accuracy = accuracy_fn(current_risk, current_next_events.int())
                    precision = precision_fn(current_risk, current_next_events.int())
                    recall = recall_fn(current_risk, current_next_events.int())
                    auprc = auprc_fn(current_risk, current_next_events.int())

                    current_res = {
                        'infusion': infusion.item(),
                        'auroc': auroc.item(),
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'auprc': auprc,
                        'num_samples': mask.sum().item(),
                        'num_positives': current_next_events.sum().item(),
                        'mask': mask
                    }

                computable_auroc_results.append(current_res)

            except Exception as e:
                print(f"Skipping AUROC for infusion {infusion.item()}: {e}")
                continue

        if len(computable_auroc_results) > 0:
            # Calculate weighted metrics
            total_samples_auroc = sum(r['num_samples'] for r in computable_auroc_results)
            if mode != 'train':
                auroc_next_weighted = sum(
                    r['auroc'] * r['num_samples'] / total_samples_auroc
                    for r in computable_auroc_results
                )
                accuracy_next_weighted = sum(
                    r['accuracy'] * r['num_samples'] / total_samples_auroc
                    for r in computable_auroc_results
                )
                precision_next_weighted = sum(
                    r['precision'] * r['num_samples'] / total_samples_auroc
                    for r in computable_auroc_results
                )
                recall_next_weighted = sum(
                    r['recall'] * r['num_samples'] / total_samples_auroc
                    for r in computable_auroc_results
                )
                auprc_next_weighted = sum(
                    r['auprc'] * r['num_samples'] / total_samples_auroc
                    for r in computable_auroc_results
                )
                output_dict.update({
                    # Next-infusion prediction metrics
                    'auroc_next_infusion_weighted': auroc_next_weighted,
                    'accuracy_next_infusion_weighted': accuracy_next_weighted,
                    'precision_next_infusion_weighted': precision_next_weighted,
                    'recall_next_infusion_weighted': recall_next_weighted,
                    'auprc_next_infusion_weighted': auprc_next_weighted,
                    'num_groups_next_infusion': num_groups_auroc,
                    'per_infusion_auroc': computable_auroc_results
                })

            if mode != 'train':
                # Calculate overall AUROC across all valid samples
                all_mask_auroc = torch.zeros_like(num_infusion_tensor, dtype=torch.bool)
                for r in computable_auroc_results:
                    all_mask_auroc |= r['mask']

                try:
                    auroc_next_overall = auroc_fn(
                        risk_scores[all_mask_auroc],
                        next_infusion_event[all_mask_auroc].int()
                    ).item()
                except Exception:
                    auroc_next_overall = 0.5
                output_dict['auroc_next_infusion_overall'] = auroc_next_overall

                num_groups_auroc = len(computable_auroc_results)
                num_samples_auroc = total_samples_auroc
    else:
        if mode in ['test', 'val']:
            output_dict.update({
                'auroc_next_infusion_weighted': auroc_next_weighted,
                'auroc_next_infusion_overall': auroc_next_overall,
                'accuracy_next_infusion_weighted': accuracy_next_weighted,
                'precision_next_infusion_weighted': precision_next_weighted,
                'recall_next_infusion_weighted': recall_next_weighted,
                'auprc_next_infusion_weighted': auprc_next_weighted,
                'num_groups_next_infusion': num_groups_auroc,
                'num_samples_next_infusion': num_samples_auroc,
            })

    return output_dict


import argparse
parser = argparse.ArgumentParser(
            prog='across_SPARC_att_risk_comparison',
            description='Compare SPARC models: vanilla vs Cox loss vs attention mechanism',
            epilog='Comparison script for different SPARC variants')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'AKI', help='side effect option: anemia, neutropenia, febrile_neutropenia, AKI, Liver_toxicity')
parser.add_argument("-m", "--modules", type=str, default = 'parsimonious', help='module category')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset: MSK, MSK_timeshift')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "True", help="whether to add number of infusion as a feature")

args = parser.parse_args()
rseed = args.rseed
option_base = args.option
modules = args.modules
dataset = args.dataset
med_of_choice = args.med_of_choice
add_num_infusion = args.add_num_infusion.lower() == 'true'


# # Hardcoded values for testing
# rseed = 1024
# option_base = 'febrile_neutropenia'
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True

# # Hardcoded values for testing
# rseed = 1024
# option_base = 'Liver_toxicity'
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = "IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB"
# add_num_infusion = True

# # Hardcoded values for testing - early
# rseed = 1024
# option_base = 'anemia'
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True

# # Hardcoded values for testing - early
# rseed = 1024
# option_base = 'neutropenia'
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True


if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option_base)):
    raise ValueError(f"Invalid option: {option_base}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', or 'Liver_toxicity'.")

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# Handle medication combinations
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
days = 30

# Model architecture parameters (MUST match training configuration)
# These come from the training command: -umt True -uct True -pmd 0.2 -tp attention
use_module_tokenizer = True   # -umt True: per-clinical-module tokenization
use_continuous_time = True    # -uct True: learned continuous-time embedding
post_mlp_dropout = 0.2        # -pmd 0.2: dropout for post-transformer MLP
missing_mask = True           # Default: concatenate missingness indicators
use_sigmoid = True            # True for test/eval mode (not recalibrate)
schedule_option = 'default'   # Default AdamW scheduler
learning_rate = 1e-4          # Default learning rate

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

# Build per-module tokenizer config (required when use_module_tokenizer=True)
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

# Define model configurations
all_model_path = sparc_paths.SPARC_MODEL_ROOT

model_configs = {
    'SPARC_vanilla': {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{all_model_path}/{modules}_MSK_060726/SPARC_direct_models/{modules}_direct_models_{option_base}_20tp_topchemos_att_modular',
        'description': 'Vanilla SPARC (BCE loss, attention)'
    },
    'SPARC_cox': {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{all_model_path}/{modules}_MSK_060726/SPARC_direct_models/{modules}_direct_models_risk_{option_base}_20tp_topchemos_att_modular',
        'description': 'SPARC with Cox loss (attention)'
    },
    'SPARC_hybrid': {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'hybrid',
        'path': f'{all_model_path}/{modules}_MSK_060726/SPARC_direct_models/{modules}_direct_models_hybrid_{option_base}_20tp_topchemos_att_modular',
        'description': 'SPARC with hybrid loss (attention)'
    }
}


# # add redo to path if it's AKI
# if option_base == 'AKI':
#     for model_key in model_configs.keys():
#         model_configs[model_key]['path'] = model_configs[model_key]['path'] + '_redo'

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

save_name = f'{all_model_path}/results_vis_060726/design_choices_1_regvscox/SPARC_reg_risk_comparison_{modules}_{option_base}_{dataset}.csv'
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

# Create data module
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1,
    task=f'{option_base}_risk', binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
    include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False
)
print(f"Data module initiated for task {option_base}_risk with")

for model_name, config in model_configs.items():
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"Option: {config['option']}, Temporal Aggregation: {config['temporal_aggregation']}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    # Only load dataloader if NOT (already exist for all 10 folds for this model)
    if model_name in model_performance_dict and False not in [fold in model_performance_dict[model_name] for fold in all_folds]:
        print(f"Data module for model {model_name} already exists for all folds, skipping")
        continue

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

        # Check if folder exists, skip if not
        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist, skipping: {model_folder}")
            continue

        existing_files = os.listdir(model_folder)
        # Find the checkpoint that has the highest val_loss (i.e. the one that is not last.ckpt)
        checkpoint_files = [el for el in existing_files if 'epoch' in el and el.endswith('.ckpt')]

        if len(checkpoint_files) == 0:
            print(f"WARNING: No checkpoint files found in {model_folder}, skipping")
            continue

        optimal_model_path = checkpoint_files[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading model from: {optimal_model_path}")

        # Load the model with appropriate configuration
        # IMPORTANT: Must include ALL parameters from training to match architecture
        model = SPARC.load_from_checkpoint(
            optimal_model_path,
            modular_featuredim_dict=modular_featuredim_dict,
            option=model_set_option,
            advanced_model="default",
            max_time_len=20,
            med_as_feature=False,
            meds_l=med_of_choice,
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
        print("model_set_option:", model_set_option)
        print("Architecture params: use_module_tokenizer={}, use_continuous_time={}, post_mlp_dropout={}".format(
            use_module_tokenizer, use_continuous_time, post_mlp_dropout))

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

                # # Debug: Print first batch info
                # if batch_count == 0:
                #     print(f"\n=== DEBUG: First Batch Info ===")
                #     # print(y.shape, y)
                #     print(f"y[0] type: {type(y[0])}, shape/value: {y[0] if not torch.is_tensor(y[0]) else y[0].shape}")
                #     print(f"y[1] type: {type(y[1])}, shape/value: {y[1] if not torch.is_tensor(y[1]) else y[1].shape}")
                #     print(f"fail_indicators shape: {fail_indicators.shape}")
                #     print(f"time_to_event shape: {time_to_event.shape}")
                #     print(f"y_pred_logits shape: {y_pred_logits.shape}")
                #     print(f"num_infusion shape: {num_infusion.shape}")
                #     print(f"First 3 fail_indicators: {fail_indicators[:3] if fail_indicators.numel() > 0 else fail_indicators}")
                #     print(f"First 3 time_to_event: {time_to_event[:3] if time_to_event.numel() > 0 else time_to_event}")
                #     print(f"First 3 num_infusion: {num_infusion[:3] if num_infusion.numel() > 0 else num_infusion}")
                #     print("="*50 + "\n")

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

        # print(f"\n=== DEBUG: Pre-concatenation Info ===")
        # print(f"Number of batches: {batch_count}")
        # print(f"Sample shapes from all_y_events: {[t.shape for t in all_y_events[:3]]}")
        # print(f"Sample shapes from all_time_to_events: {[t.shape for t in all_time_to_events[:3]]}")
        # print(f"Sample shapes from all_y_preds: {[t.shape for t in all_y_preds[:3]]}")
        # print("="*50 + "\n")

        # Concatenate all batches
        all_y_preds = torch.cat(all_y_preds, dim=0)
        all_y_events = torch.cat(all_y_events, dim=0)
        all_time_to_events = torch.cat(all_time_to_events, dim=0)
        all_num_infusions = torch.cat(all_num_infusions, dim=0)

        ### DEBUGGING ##
        if val_fold == 0:
            print(f"[{model_name}] fold0 PRED sum: {all_y_preds.sum().item():.6f}, "
                f"EVT sum: {all_y_events.sum().item()}, "
                f"TTE sum: {all_time_to_events.float().sum().item():.1f}, "
                f"N: {len(all_y_preds)}")
        ### DEBUGGING ##

        # # Print summary statistics
        # print(f"\n=== DEBUG: Dataset Summary Statistics ===")
        # print(f"Total samples: {len(all_y_events)}")
        # print(f"Positive events (fail_indicators=1): {(all_y_events == 1).sum().item()} ({100 * (all_y_events == 1).float().mean().item():.2f}%)")
        # print(f"Negative events (fail_indicators=0): {(all_y_events == 0).sum().item()} ({100 * (all_y_events == 0).float().mean().item():.2f}%)")
        # print(f"\nTime to event statistics:")
        # print(f"  Min: {all_time_to_events.min().item():.2f}")
        # print(f"  Max: {all_time_to_events.max().item():.2f}")
        # print(f"  Mean: {all_time_to_events.float().mean().item():.2f}")
        # print(f"  Median: {all_time_to_events.float().median().item():.2f}")
        # print(f"\nNumber of infusions statistics:")
        # print(f"  Min: {all_num_infusions.min().item()}")
        # print(f"  Max: {all_num_infusions.max().item()}")
        # print(f"  Mean: {all_num_infusions.float().mean().item():.2f}")
        # print(f"  Median: {all_num_infusions.float().median().item():.2f}")
        # print(f"\nPrediction statistics:")
        # print(f"  Min: {all_y_preds.min().item():.4f}")
        # print(f"  Max: {all_y_preds.max().item():.4f}")
        # print(f"  Mean: {all_y_preds.mean().item():.4f}")
        # print(f"  Median: {all_y_preds.median().item():.4f}")
        # print("="*50 + "\n")

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
