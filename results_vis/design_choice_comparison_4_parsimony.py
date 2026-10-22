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
            prog='design_choice_comparison_4_parsimony',
            description='Compare SPARC models: parsimonious vs. all modules',
            epilog='Comparison script for module selection: practical vs all')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'AKI', help='side effect option: anemia, neutropenia, febrile_neutropenia, AKI, Liver_toxicity')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset: MSK, MSK_timeshift')

args = parser.parse_args()
rseed = args.rseed
option_base = args.option
dataset = args.dataset



if not (bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option_base)) or option_base in ['binary', 'binary_icu_pre']):
    raise ValueError(f"Invalid option: {option_base}. Must be one of 'anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', 'Liver_toxicity', 'binary', or 'binary_icu_pre'.")

# Medication sets per side effect
med_configs = {
    'AKI': ["CISPLATIN", "IFOSFAMIDE", "PEMETREXED", "METHOTREXATE", "PAMIDRONATE", "ZOLEDRONIC ACID"],
    'Liver_toxicity': ["IPILIMUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "PEMBROLIZUMAB", "AVELUMAB", "CEMIPLIMAB", "DOSTARLIMAB"],
    'febrile_neutropenia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                            ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
    'anemia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
               ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
    'neutropenia': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                    ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
    'binary': ["CISPLATIN"],
    'binary_icu_pre': ["CISPLATIN"],
}

# Get medication configuration for current task
med_of_choice = med_configs[option_base]

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

is_binary_task = option_base in ['binary', 'binary_icu_pre']
task_option = option_base if is_binary_task else f'{option_base}_risk'

# hyperparameters
batch_size = 512
competing_risk_death_check = False
include_gcsf = False
days = 180 if option_base == 'binary' else 30
add_num_infusion = not is_binary_task

# Architecture params for the 060726 models
use_module_tokenizer = True   # -umt True
use_continuous_time = True    # -uct True
post_mlp_dropout = 0.2        # -pmd 0.2
use_sigmoid = True
schedule_option = 'default'
learning_rate = 1e-4
missing_mask = True
temporal_aggregation = 'attention'

# Model paths
# batch_first=True arms: 'all' from A11_dc4all, 'parsimonious' reference from A9.
ABLATION_ROOT = os.environ.get('SPARC_ABLATION_ROOT',
                               '/path/to/SPARC_ablation_models')
BATCH_FIRST = True

# Loss: risk for AKI/Liver_toxicity, hybrid otherwise
if is_binary_task:
    practical_model_name = f'parsimonious_direct_models_{option_base}_20tp_topchemos_att_modular'
    all_model_name = f'all_direct_models_{option_base}_20tp_topchemos_att_modular'
    # A11 rerun folders have no '_nopmdmed3' suffix
else:
    if option_base in ['AKI', 'Liver_toxicity']:
        loss_suffix = 'risk'
    else:
        loss_suffix = 'hybrid'
    practical_model_name = f'parsimonious_direct_models_{loss_suffix}_{option_base}_20tp_topchemos_att_modular'
    all_model_name = f'all_direct_models_{loss_suffix}_{option_base}_20tp_topchemos_att_modular'

model_configs = {
    'all': {
        'modules': 'all',
        'data_modules': ['gender', 'age', 'basic_lab', 'basic_vitals', 'extra_lab', 'extra_vitals'],
        'component': 'all',
        'option': task_option,
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{ABLATION_ROOT}/A11_dc4all/parsimonious_MSK_060726/SPARC_parsimony_design/{all_model_name}',
        'description': 'All modules',
        'med_as_feature': False
    },
    'parsimonious': {
        'modules': 'parsimonious',
        'data_modules': ['gender', 'age', 'cbc_differential', 'basic_lab', 'basic_vitals'],
        'component': 'parsimonious',
        'option': task_option,
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{ABLATION_ROOT}/A9_batchfirst/parsimonious_MSK_060726/SPARC_direct_models/{practical_model_name}',
        'description': 'Parsimonious modules',
        'med_as_feature': False
    }
}

print("Checkpoint 1: Setting up configurations")
print(f"Comparing: parsimonious ({model_configs['parsimonious']['data_modules']}) vs all ({model_configs['all']['data_modules']})")

save_name = os.environ.get('SPARC_DC4_RESULTS', f'{ABLATION_ROOT}/results') + f'/design_choices_4_parsimony/SPARC_parsimony_{option_base}_{dataset}.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)

model_performance_dict = {}

# Reuse cached results if the CSV exists
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    key_cols = ['auroc', 'auprc'] if is_binary_task else ['test_cindex_overall_valid', 'test_cindex_weighted']
    key_cols = [c for c in key_cols if c in saved_df.columns]
    performance_df_existing = saved_df.dropna(subset=key_cols) if key_cols else saved_df.dropna()
    for index, row in performance_df_existing.iterrows():
        model_name = row['model']
        val_fold = int(row['val_fold'])
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        if is_binary_task:
            model_performance_dict[model_name][val_fold] = {
                'auroc': row.get('auroc'),
                'auprc': row.get('auprc'),
            }
        else:
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': row.get('test_cindex_overall_valid'),
                'test_cindex_weighted': row.get('test_cindex_weighted'),
                'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
                'test_auroc_nextinf_weighted': row.get('test_auroc_nextinf_weighted'),
            }
    print("Loaded existing model performance data from csv")

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
model_set_option = 'binary' if is_binary_task else 'side_effects_risk'

for model_name, config in model_configs.items():
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"Modules: {config['modules']}, Data modules: {config['data_modules']}")
    print(f"Option: {config['option']}, Temporal Aggregation: {config['temporal_aggregation']}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    # Skip models that already have all folds
    if model_name in model_performance_dict and False not in [fold in model_performance_dict[model_name] for fold in all_folds]:
        print(f"Data module for model {model_name} already exists for all folds, skipping")
        continue

    print(f"Loading dataset for {model_name} with modules={config['modules']}")
    data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
        dataset=dataset,
        data_source_path=data_source_path,
        option=task_option,
        modules=config['modules'],
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

    # Module index config for the 060726 models
    module_index_cfg = build_module_index_config(modular_featuredim_dict)

    # Static features: gender + age for parsimonious; for 'all', every static key
    # in build_module_index_config's static_keys_order
    if config['modules'] == 'parsimonious':
        static_module_names = ['gender', 'age']
    else:
        static_module_names = ['gender', 'age', 'primary_ctype', 'genetic_ancestry', 'met_status', 'clinical', 'mutation']
    n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)
    print(f"Calculated n_static_features for {config['modules']} with keys {static_module_names}: {n_static_features}")

    # Test split from the fold dict
    if dataset == 'MSK':
        with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as handle:
            fold_dict = pickle.load(handle)
    elif dataset == 'MSK_CAIA':
        with open(f'{data_source_path}/{dataset}/fold_dict_11.pickle', 'rb') as handle:
            fold_dict = pickle.load(handle)
    elif dataset == 'MSK_timeshift':
        fold_dict = {10: data_dict_lab_dict['age'].keys()}
        # Timeshift test set: patients not in the original MSK cohort
        data_path = f'{data_source_path}/MSK/postbc_DMP_ID_to_DATES_to_TESTSVALUES_DICT_deidentified_MSK.pickle'
        with open(data_path, 'rb') as handle:
            og_set = pickle.load(handle)
        fold_dict[10] = [DMP_ID for DMP_ID in fold_dict[10] if DMP_ID not in og_set.keys()]

    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [DMP_ID for DMP_ID in testDMP_IDs if DMP_ID in data_dict_lab_dict.get('basic_lab', data_dict_lab_dict.get('age', {})).keys()]
    split_dict = {'train': [], 'val': [], 'test': testDMP_IDs}

    data_module = FlexibleFeatureDataModule(
        data_dict_lab_dict, split_dict, dataset=dataset, component=config['component'],
        batch_size=512, max_time_len=20, num_workers=1,
        task=task_option, binary_days=days, med=med_of_choice_processed,
        med_as_feature=config['med_as_feature'], med_l=med_of_choice_processed,
        advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
        include_infusion_dates=add_num_infusion if not is_binary_task else False, premed_only=False, competing_risk_death_check=False
    )
    print(f"Data module initiated for task {task_option} with modules={config['modules']}")

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

        model_folder = f'{config["path"]}/fold{val_fold}'

        print(f"Looking for model in: {model_folder}")
        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist: {model_folder}")
            print(f"Skipping fold {val_fold}")
            continue

        existing_files = os.listdir(model_folder)
        # Use the saved epoch checkpoint, not last.ckpt
        epoch_files = [el for el in existing_files if 'epoch' in el]
        if len(epoch_files) == 0:
            print(f"WARNING: No epoch checkpoint found in {model_folder}")
            print(f"Skipping fold {val_fold}")
            continue

        optimal_model_path = epoch_files[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading model from: {optimal_model_path}")

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
                batch_first=BATCH_FIRST,
                lr=learning_rate,
            )
            print(f"Model loaded successfully with modules={config['modules']}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Skipping fold {val_fold}")
            continue

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model on device: {device}")

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
                        primary_ctype_returned=model.primary_ctype_returned,
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
                    num_infusion = torch.zeros(xs_combined.shape[0], dtype=torch.float32)

                # With med_as_feature=True, xs_combined is (xs, meds_data)
                if isinstance(xs_combined, tuple):
                    xs_combined = tuple(x.to(device) for x in xs_combined)
                else:
                    xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Binary tasks return y as a plain tensor, not a tuple
                if is_binary_task:
                    fail_indicators = torch.as_tensor(y, device=device)
                    time_to_event = torch.ones_like(fail_indicators, device=device).float()
                else:
                    fail_indicators = torch.as_tensor(y[0], device=device)
                    time_to_event = torch.as_tensor(y[1], device=device)

                if fail_indicators.dim() == 0:
                    fail_indicators = fail_indicators.unsqueeze(0)
                if time_to_event.dim() == 0:
                    time_to_event = time_to_event.unsqueeze(0)

                all_y_preds.append(y_pred_logits.detach().cpu())
                all_y_events.append(fail_indicators.detach().cpu())
                all_time_to_events.append(time_to_event.detach().cpu())
                all_num_infusions.append(num_infusion.detach().cpu())

                batch_count += 1

        all_y_preds = torch.cat(all_y_preds, dim=0)
        all_y_events = torch.cat(all_y_events, dim=0)
        all_time_to_events = torch.cat(all_time_to_events, dim=0)
        all_num_infusions = torch.cat(all_num_infusions, dim=0)

        if is_binary_task:
            auroc_fn = AUROC(task='binary')
            auprc_fn = AveragePrecision(task='binary')
            auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
            auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()
            model_performance_dict[model_name][val_fold]['auroc'] = auroc_score
            model_performance_dict[model_name][val_fold]['auprc'] = auprc_score
            print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")
        else:
            c_idx_dict = calculate_weighted_c_index_by_infusion(
                all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                mode='val', option=config['option']
            )
            model_performance_dict[model_name][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', None)
            model_performance_dict[model_name][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', None)
            model_performance_dict[model_name][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
            model_performance_dict[model_name][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
            print(f"C-index overall: {c_idx_dict.get('c_index_overall')}, C-index weighted: {c_idx_dict.get('c_index_weighted')}")
            print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")

        # Save after each fold
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted', 'auroc', 'auprc', 'description']
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
        metrics_to_print = ['auroc', 'auprc'] if is_binary_task else [
            'test_cindex_overall_valid', 'test_cindex_weighted',
            'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted']
        for metric in metrics_to_print:
            if metric in model_data.columns:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

print(f"\nResults saved to: {save_name}")
print("Done!")
