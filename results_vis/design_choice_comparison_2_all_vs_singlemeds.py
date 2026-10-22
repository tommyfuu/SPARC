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
import json
import ast
data_source_path=sparc_paths.DATA_ROOT + '/'

from torchsurv.metrics.cindex import ConcordanceIndex
from torchmetrics import AUROC, F1Score, AveragePrecision, Accuracy, Precision, Recall

from eval_utils import calculate_weighted_c_index_by_infusion


import argparse
parser = argparse.ArgumentParser(
            prog='design_choice_comparison_2_all_vs_singlemeds',
            description='Compare SPARC models: all-meds vs single-med training',
            epilog='Comparison script for all-meds vs single-med models')
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

# hyperparameters
batch_size = 512
competing_risk_death_check = False
include_gcsf = False
days = 30
add_num_infusion = True

# Architecture params for the 060726 models
use_module_tokenizer = True   # -umt True
use_continuous_time = True    # -uct True
post_mlp_dropout = 0.2        # -pmd 0.2
use_sigmoid = True
schedule_option = 'default'
learning_rate = 1e-4
missing_mask = True
temporal_aggregation = 'attention'

# Medication sets per side effect
med_configs = {
    'AKI': {
        'all_meds': ["CISPLATIN", "IFOSFAMIDE", "PEMETREXED", "METHOTREXATE", "PAMIDRONATE", "ZOLEDRONIC ACID"],
        'single_meds': {
            'CISPLATIN': ["CISPLATIN"],
            'IFOSFAMIDE': ["IFOSFAMIDE"],
            'PEMETREXED': ["PEMETREXED"],
            'METHOTREXATE': ["METHOTREXATE"],
            'PAMIDRONATE': ["PAMIDRONATE"],
            'ZOLEDRONIC_ACID': ["ZOLEDRONIC ACID"],
        }
    },
    'Liver_toxicity': {
        'all_meds': ["IPILIMUMAB", "ATEZOLIZUMAB", "NIVOLUMAB", "PEMBROLIZUMAB", "AVELUMAB", "CEMIPLIMAB", "DOSTARLIMAB"],
        'single_meds': {
            'IPILIMUMAB': ["IPILIMUMAB"],
            'ATEZOLIZUMAB': ["ATEZOLIZUMAB"],
            'NIVOLUMAB': ["NIVOLUMAB"],
            'PEMBROLIZUMAB': ["PEMBROLIZUMAB"],
            'AVELUMAB': ["AVELUMAB"],
            'CEMIPLIMAB': ["CEMIPLIMAB"],
            'DOSTARLIMAB': ["DOSTARLIMAB"],
        }
    },
    'febrile_neutropenia': {
        'all_meds': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                     ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
        'single_meds': {
            'DOCETAXEL': ["DOCETAXEL"],
            'SACITUZUMAB_GOVITECAN': ["SACITUZUMAB GOVITECAN"],
            'CISPLATIN': ["CISPLATIN"],
            'TOPOTECAN': ["TOPOTECAN"],
            'IRINOTECAN': ["IRINOTECAN", "IRINOTECAN LIPOSOMAL"],
            'CYCLOPHOSPHAMIDE_DOXORUBICIN_VINCRISTINE': [["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"]],
            'BENDAMUSTINE': ["BENDAMUSTINE"],
            'CARBOPLATIN': ["CARBOPLATIN"],
            'ETOPOSIDE': ["ETOPOSIDE"],
            'CABAZITAXEL': ["CABAZITAXEL"],
        }
    },
    'anemia': {
        'all_meds': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                     ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
        'single_meds': {
            'DOCETAXEL': ["DOCETAXEL"],
            'SACITUZUMAB_GOVITECAN': ["SACITUZUMAB GOVITECAN"],
            'CISPLATIN': ["CISPLATIN"],
            'TOPOTECAN': ["TOPOTECAN"],
            'IRINOTECAN': ["IRINOTECAN", "IRINOTECAN LIPOSOMAL"],
            'CYCLOPHOSPHAMIDE_DOXORUBICIN_VINCRISTINE': [["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"]],
            'BENDAMUSTINE': ["BENDAMUSTINE"],
            'CARBOPLATIN': ["CARBOPLATIN"],
            'ETOPOSIDE': ["ETOPOSIDE"],
            'CABAZITAXEL': ["CABAZITAXEL"],
        }
    },
    'neutropenia': {
        'all_meds': ["DOCETAXEL", "SACITUZUMAB GOVITECAN", "CISPLATIN", "TOPOTECAN", "IRINOTECAN", "IRINOTECAN LIPOSOMAL",
                     ["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"], "BENDAMUSTINE", "CARBOPLATIN", "ETOPOSIDE", "CABAZITAXEL"],
        'single_meds': {
            'DOCETAXEL': ["DOCETAXEL"],
            'SACITUZUMAB_GOVITECAN': ["SACITUZUMAB GOVITECAN"],
            'CISPLATIN': ["CISPLATIN"],
            'TOPOTECAN': ["TOPOTECAN"],
            'IRINOTECAN': ["IRINOTECAN", "IRINOTECAN LIPOSOMAL"],
            'CYCLOPHOSPHAMIDE_DOXORUBICIN_VINCRISTINE': [["CYCLOPHOSPHAMIDE", "DOXORUBICIN", "VINCRISTINE"]],
            'BENDAMUSTINE': ["BENDAMUSTINE"],
            'CARBOPLATIN': ["CARBOPLATIN"],
            'ETOPOSIDE': ["ETOPOSIDE"],
            'CABAZITAXEL': ["CABAZITAXEL"],
        }
    }
}

# Get medication configuration for current task
med_of_choice = med_configs[option_base]['all_meds']
single_med_configs = med_configs[option_base]['single_meds']

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
print("All-meds configuration:", med_of_choice_processed)


print("Checkpoint 1: Loading dataset using dataset_loading.py")

data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=f'{option_base}_risk',
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

# batch_first=True arms: single-med from A13_singlemed; all-meds from A9_batchfirst
# (AKI, Liver_toxicity) or A10_dc1loss (anemia, febrile_neutropenia, neutropenia).
ABLATION_ROOT = os.environ.get('SPARC_ABLATION_ROOT',
                               '/path/to/SPARC_ablation_models')
BATCH_FIRST = True

model_configs = {}

# both arms use the risk loss so the comparison is loss-matched
single_med_loss = 'risk'
if option_base in ['AKI', 'Liver_toxicity']:
    ALLMEDS_ROOT = f'{ABLATION_ROOT}/A9_batchfirst'
else:
    ALLMEDS_ROOT = f'{ABLATION_ROOT}/A10_dc1loss'

# All-meds model (risk loss)
model_configs['all_meds'] = {
    'option': f'{option_base}_risk',
    'temporal_aggregation': 'attention',
    'loss_option': 'default',
    'path': f'{ALLMEDS_ROOT}/{modules}_MSK_060726/SPARC_direct_models/parsimonious_direct_models_risk_{option_base}_20tp_topchemos_att_modular',
    'description': f'All meds ({len(med_of_choice)} medications)',
    'meds': med_of_choice_processed
}

# Single-med models (risk loss)
for med_name, med_list in single_med_configs.items():
    processed_med = process_med_list(med_list)
    model_configs[f'single_med_{med_name}'] = {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{ABLATION_ROOT}/A13_singlemed/{modules}_MSK_060726/sg_med_exp/parsimonious_direct_models_{single_med_loss}_{option_base}_20tp_single_med_{med_name}_att_modular',
        'description': f'Single med: {med_name}',
        'meds': processed_med
    }

# Test split from the fold dict
if dataset == 'MSK':
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

save_name = (os.environ.get('SPARC_DC2_RESULTS', f'{ABLATION_ROOT}/results')
             + f'/design_choices_2_all_vs_singlemeds/SPARC_allmeds_vs_singlemeds_{modules}_{option_base}_{dataset}.csv')
os.makedirs(os.path.dirname(save_name), exist_ok=True)

model_performance_dict = {}

# Reuse cached results if the CSV exists
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    performance_df_existing = saved_df.dropna(subset=['val_fold'])  # only require val_fold
    for index, row in performance_df_existing.iterrows():
        model_name = row['model']
        description = row.get('description', '')

        # Skip models marked as non-existent
        if 'NON-EXISTENT' in str(description):
            print(f"Skipping non-existent model from saved results: {model_name}")
            if model_name not in model_performance_dict:
                model_performance_dict[model_name] = {}
            val_fold = int(row['val_fold'])
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': None,
                'test_cindex_weighted': None,
                'test_auroc_nextinf_overall': None,
                'test_auroc_nextinf_weighted': None,
                'num_test_samples': row.get('num_test_samples'),
                'num_positive_samples': row.get('num_positive_samples'),
            }
            continue

        val_fold = int(row['val_fold'])
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        model_performance_dict[model_name][val_fold] = {
            'test_cindex_overall_valid': row.get('test_cindex_overall_valid'),
            'test_cindex_weighted': row.get('test_cindex_weighted'),
            'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
            'test_auroc_nextinf_weighted': row.get('test_auroc_nextinf_weighted'),
            'num_test_samples': row.get('num_test_samples'),
            'num_positive_samples': row.get('num_positive_samples'),
        }
    print("Loaded existing model performance data from csv")

all_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
model_set_option = 'side_effects_risk'

# Module index config for the 060726 models
module_index_cfg = build_module_index_config(modular_featuredim_dict)

static_module_names = ['gender', 'age']
n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)

# Step 1: Test all_meds model on single medication datasets
print("\n" + "="*80)
print("STEP 1: Testing all_meds model on individual medication datasets")
print("="*80 + "\n")

all_meds_config = model_configs['all_meds']
for med_name, med_list in single_med_configs.items():
    model_name = f'all_meds_on_{med_name}'
    processed_med = process_med_list(med_list)

    print(f"\n{'='*80}")
    print(f"Testing all_meds model on: {med_name}")
    print(f"Medications: {processed_med}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    if not os.path.exists(all_meds_config['path']):
        print(f"WARNING: All-meds model path does not exist: {all_meds_config['path']}")
        print(f"Creating NON-EXISTENT placeholder rows for {med_name}")

        # Mark all folds as non-existent
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}

        model_configs[model_name] = {
            'option': f'{option_base}_risk',
            'temporal_aggregation': 'attention',
            'loss_option': 'default',
            'path': all_meds_config['path'],
            'description': f'NON-EXISTENT: All-meds model tested on {med_name}',
            'meds': processed_med
        }

        for val_fold in all_folds:
            if val_fold not in model_performance_dict[model_name]:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': None,
                    'test_cindex_weighted': None,
                    'test_auroc_nextinf_overall': None,
                    'test_auroc_nextinf_weighted': None,
                    'num_test_samples': None,
                    'num_positive_samples': None,
                }

        # Save after adding placeholders
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
        model_performance_df = pd.DataFrame(columns=all_columns)
        for mn in model_performance_dict.keys():
            for vf in model_performance_dict[mn].keys():
                row = {'model': mn, 'val_fold': vf, 'description': model_configs[mn]['description']}
                row.update(model_performance_dict[mn][vf])
                row = [row[col] if col in row else None for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
        model_performance_df.to_csv(save_name, index=False)
        print(f"Saved NON-EXISTENT placeholder results to {save_name}")

        continue

    # Skip models that already have all folds
    if model_name in model_performance_dict and False not in [fold in model_performance_dict[model_name] for fold in all_folds]:
        if model_name in model_configs and 'NON-EXISTENT' in model_configs[model_name].get('description', ''):
            print(f"Model {model_name} already marked as NON-EXISTENT for all folds, skipping")
        else:
            print(f"Data module for model {model_name} already exists for all folds, skipping")
        continue

    # Create data module with the single medication data
    data_module = FlexibleFeatureDataModule(
        data_dict_lab_dict, split_dict, dataset=dataset, component=component,
        batch_size=512, max_time_len=20, num_workers=1,
        task=f'{option_base}_risk', binary_days=days, med=processed_med, med_as_feature=False, med_l=processed_med,
        advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
        include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False
    )
    print(f"Data module initiated for single medication: {med_name}")

    # Count test samples and positive samples
    test_sample_count = 0
    test_positive_count = 0

    if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None:
        try:
            test_sample_count = len(data_module.test_dataset)
            if test_sample_count > 0:
                for i in range(test_sample_count):
                    try:
                        sample = data_module.test_dataset[i]
                        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            y = sample[1]
                            # y is (event_indicator, time_to_event)
                            if isinstance(y, (list, tuple)) and len(y) >= 1:
                                if y[0] == 1:
                                    test_positive_count += 1
                            elif torch.is_tensor(y):
                                if y.item() == 1:
                                    test_positive_count += 1
                    except:
                        continue
        except:
            test_sample_count = 0
            test_positive_count = 0

    print(f"Test samples for {med_name}: {test_sample_count} (positives: {test_positive_count})")

    # Skip if the test set is too small
    if test_sample_count < 50:
        print(f"WARNING: Insufficient test samples ({test_sample_count} < 50) for {med_name}")
        print(f"Creating placeholder rows for this medication")

        # Placeholder rows with sample counts but no metrics
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}

        model_configs[model_name] = {
            'option': f'{option_base}_risk',
            'temporal_aggregation': 'attention',
            'loss_option': 'default',
            'path': all_meds_config['path'],
            'description': f'All-meds model tested on {med_name}',
            'meds': processed_med
        }

        for val_fold in all_folds:
            if val_fold not in model_performance_dict[model_name]:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': None,
                    'test_cindex_weighted': None,
                    'test_auroc_nextinf_overall': None,
                    'test_auroc_nextinf_weighted': None,
                    'num_test_samples': test_sample_count,
                    'num_positive_samples': test_positive_count,
                }

        # Save after adding placeholders
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
        model_performance_df = pd.DataFrame(columns=all_columns)
        for mn in model_performance_dict.keys():
            for vf in model_performance_dict[mn].keys():
                row = {'model': mn, 'val_fold': vf, 'description': model_configs[mn]['description']}
                row.update(model_performance_dict[mn][vf])
                row = [row[col] if col in row else None for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
        model_performance_df.to_csv(save_name, index=False)
        print(f"Saved placeholder results to {save_name}")

        continue

    model_configs[model_name] = {
        'option': f'{option_base}_risk',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': all_meds_config['path'],
        'description': f'All-meds model tested on {med_name}',
        'meds': processed_med
    }

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

        # all-meds checkpoint
        model_folder = f'{all_meds_config["path"]}/fold{val_fold}'

        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist: {model_folder}")
            print(f"Marking fold {val_fold} as NON-EXISTENT")
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': None,
                'test_cindex_weighted': None,
                'test_auroc_nextinf_overall': None,
                'test_auroc_nextinf_weighted': None,
                'num_test_samples': None,
                'num_positive_samples': None,
            }
            # Flag the missing fold in the description
            if 'NON-EXISTENT FOLD' not in model_configs[model_name]['description']:
                model_configs[model_name]['description'] += ' (NON-EXISTENT FOLD)'
            continue

        print(f"Looking for model in: {model_folder}")
        existing_files = os.listdir(model_folder)
        # Use the saved epoch checkpoint, not last.ckpt
        epoch_files = [el for el in existing_files if 'epoch' in el]
        if len(epoch_files) == 0:
            print(f"WARNING: No epoch checkpoint found in {model_folder}")
            print(f"Marking fold {val_fold} as NON-EXISTENT")
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': None,
                'test_cindex_weighted': None,
                'test_auroc_nextinf_overall': None,
                'test_auroc_nextinf_weighted': None,
                'num_test_samples': None,
                'num_positive_samples': None,
            }
            # Flag the missing fold in the description
            if 'NON-EXISTENT FOLD' not in model_configs[model_name]['description']:
                model_configs[model_name]['description'] += ' (NON-EXISTENT FOLD)'
            continue

        optimal_model_path = epoch_files[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading model from: {optimal_model_path}")

        # Load the all_meds model with all medications in the config
        try:
            model = SPARC.load_from_checkpoint(
                optimal_model_path,
                modular_featuredim_dict=modular_featuredim_dict,
                option=model_set_option,
                advanced_model="default",
                max_time_len=20,
                med_as_feature=False,
                meds_l=all_meds_config['meds'],
                loss_option=all_meds_config['loss_option'],
                schedule=schedule_option,
                side_effects_num_infusion=add_num_infusion,
                include_gcsf=include_gcsf,
                use_gcsf=include_gcsf,
                use_sigmoid=use_sigmoid,
                temporal_aggregation=all_meds_config['temporal_aggregation'],
                missing_mask=missing_mask,
                module_index_cfg=module_index_cfg,
                use_module_tokenizer=use_module_tokenizer,
                use_continuous_time=use_continuous_time,
                n_static_features=n_static_features,
                post_mlp_dropout=post_mlp_dropout,
                batch_first=BATCH_FIRST,
                lr=learning_rate,
            )
            print("Model loaded successfully")
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

                xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Ground truth
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

        c_idx_dict = calculate_weighted_c_index_by_infusion(
            all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
            mode='val', option=all_meds_config['option']
        )

        model_performance_dict[model_name][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', None)
        model_performance_dict[model_name][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
        model_performance_dict[model_name][val_fold]['num_test_samples'] = test_sample_count
        model_performance_dict[model_name][val_fold]['num_positive_samples'] = test_positive_count

        print(f"C-index overall: {c_idx_dict.get('c_index_overall')}, C-index weighted: {c_idx_dict.get('c_index_weighted')}")
        print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")

        # Save after each fold
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
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
print("STEP 2: Testing models trained on individual medications")
print("="*80 + "\n")

for model_name, config in model_configs.items():
    # all_meds_on_* entries were handled in step 1
    if model_name.startswith('all_meds_on_'):
        continue

    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"Option: {config['option']}, Temporal Aggregation: {config['temporal_aggregation']}")
    print(f"Medications: {config['meds']}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    if not os.path.exists(config['path']):
        print(f"WARNING: Model path does not exist: {config['path']}")
        print(f"Creating NON-EXISTENT placeholder rows for {model_name}")

        # Mark all folds as non-existent
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}

        model_configs[model_name]['description'] = f'NON-EXISTENT: {config["description"]}'

        for val_fold in all_folds:
            if val_fold not in model_performance_dict[model_name]:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': None,
                    'test_cindex_weighted': None,
                    'test_auroc_nextinf_overall': None,
                    'test_auroc_nextinf_weighted': None,
                    'num_test_samples': None,
                    'num_positive_samples': None,
                }

        # Save after adding placeholders
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
        model_performance_df = pd.DataFrame(columns=all_columns)
        for mn in model_performance_dict.keys():
            for vf in model_performance_dict[mn].keys():
                row = {'model': mn, 'val_fold': vf, 'description': model_configs[mn]['description']}
                row.update(model_performance_dict[mn][vf])
                row = [row[col] if col in row else None for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
        model_performance_df.to_csv(save_name, index=False)
        print(f"Saved NON-EXISTENT placeholder results to {save_name}")

        continue

    # Skip models that already have all folds
    if model_name in model_performance_dict and False not in [fold in model_performance_dict[model_name] for fold in all_folds]:
        if model_name in model_configs and 'NON-EXISTENT' in model_configs[model_name].get('description', ''):
            print(f"Model {model_name} already marked as NON-EXISTENT for all folds, skipping")
        else:
            print(f"Data module for model {model_name} already exists for all folds, skipping")
        continue

    # Create data module with the appropriate medications
    data_module = FlexibleFeatureDataModule(
        data_dict_lab_dict, split_dict, dataset=dataset, component=component,
        batch_size=512, max_time_len=20, num_workers=1,
        task=f'{option_base}_risk', binary_days=days, med=config['meds'], med_as_feature=False, med_l=config['meds'],
        advanced_model='default', control_lack_confounds=False, side_effect_censoring=False,
        include_infusion_dates=add_num_infusion, premed_only=False, competing_risk_death_check=False
    )
    print(f"Data module initiated for task {option_base}_risk with {len(config['meds'])} medication(s)")

    # Count test samples and positive samples
    test_sample_count = 0
    test_positive_count = 0

    if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None:
        try:
            test_sample_count = len(data_module.test_dataset)
            if test_sample_count > 0:
                for i in range(test_sample_count):
                    try:
                        sample = data_module.test_dataset[i]
                        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            y = sample[1]
                            # y is (event_indicator, time_to_event)
                            if isinstance(y, (list, tuple)) and len(y) >= 1:
                                if y[0] == 1:
                                    test_positive_count += 1
                            elif torch.is_tensor(y):
                                if y.item() == 1:
                                    test_positive_count += 1
                    except:
                        continue
        except:
            test_sample_count = 0
            test_positive_count = 0

    print(f"Test samples for {model_name}: {test_sample_count} (positives: {test_positive_count})")

    # Single-med models: skip if the test set is too small
    if model_name.startswith('single_med_') and test_sample_count < 50:
        print(f"WARNING: Insufficient test samples ({test_sample_count} < 50) for {model_name}")
        print(f"Creating placeholder rows for this medication")

        # Placeholder rows with sample counts but no metrics
        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}

        for val_fold in all_folds:
            if val_fold not in model_performance_dict[model_name]:
                model_performance_dict[model_name][val_fold] = {
                    'test_cindex_overall_valid': None,
                    'test_cindex_weighted': None,
                    'test_auroc_nextinf_overall': None,
                    'test_auroc_nextinf_weighted': None,
                    'num_test_samples': test_sample_count,
                    'num_positive_samples': test_positive_count,
                }

        # Save after adding placeholders
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
        model_performance_df = pd.DataFrame(columns=all_columns)
        for mn in model_performance_dict.keys():
            for vf in model_performance_dict[mn].keys():
                row = {'model': mn, 'val_fold': vf, 'description': model_configs[mn]['description']}
                row.update(model_performance_dict[mn][vf])
                row = [row[col] if col in row else None for col in all_columns]
                model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
        model_performance_df.to_csv(save_name, index=False)
        print(f"Saved placeholder results to {save_name}")

        continue

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

        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist: {model_folder}")
            print(f"Marking fold {val_fold} as NON-EXISTENT")
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': None,
                'test_cindex_weighted': None,
                'test_auroc_nextinf_overall': None,
                'test_auroc_nextinf_weighted': None,
                'num_test_samples': None,
                'num_positive_samples': None,
            }
            # Flag the missing fold in the description
            if 'NON-EXISTENT FOLD' not in model_configs[model_name]['description']:
                model_configs[model_name]['description'] += ' (NON-EXISTENT FOLD)'
            continue

        print(f"Looking for model in: {model_folder}")
        existing_files = os.listdir(model_folder)
        # Use the saved epoch checkpoint, not last.ckpt
        epoch_files = [el for el in existing_files if 'epoch' in el]
        if len(epoch_files) == 0:
            print(f"WARNING: No epoch checkpoint found in {model_folder}")
            print(f"Marking fold {val_fold} as NON-EXISTENT")
            model_performance_dict[model_name][val_fold] = {
                'test_cindex_overall_valid': None,
                'test_cindex_weighted': None,
                'test_auroc_nextinf_overall': None,
                'test_auroc_nextinf_weighted': None,
                'num_test_samples': None,
                'num_positive_samples': None,
            }
            # Flag the missing fold in the description
            if 'NON-EXISTENT FOLD' not in model_configs[model_name]['description']:
                model_configs[model_name]['description'] += ' (NON-EXISTENT FOLD)'
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
                med_as_feature=False,
                meds_l=config['meds'],
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
            print("Model loaded successfully")
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

                xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                y_pred_logits = model(xs_combined, src_key_padding_mask=mask,
                                      num_infusion=num_infusion, gcsf_groundtruth=gcsf_groundtruth)

                # Ground truth
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

        c_idx_dict = calculate_weighted_c_index_by_infusion(
            all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
            mode='val', option=config['option']
        )

        model_performance_dict[model_name][val_fold]['test_cindex_overall_valid'] = c_idx_dict.get('c_index_overall', None)
        model_performance_dict[model_name][val_fold]['test_cindex_weighted'] = c_idx_dict.get('c_index_weighted', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_overall'] = c_idx_dict.get('auroc_next_infusion_overall', None)
        model_performance_dict[model_name][val_fold]['test_auroc_nextinf_weighted'] = c_idx_dict.get('auroc_next_infusion_weighted', None)
        model_performance_dict[model_name][val_fold]['num_test_samples'] = test_sample_count
        model_performance_dict[model_name][val_fold]['num_positive_samples'] = test_positive_count

        print(f"C-index overall: {c_idx_dict.get('c_index_overall')}, C-index weighted: {c_idx_dict.get('c_index_weighted')}")
        print(f"AUROC next-infusion overall: {c_idx_dict.get('auroc_next_infusion_overall')}, AUROC next-infusion weighted: {c_idx_dict.get('auroc_next_infusion_weighted')}")

        # Save after each fold
        all_columns = ['model', 'val_fold', 'test_cindex_overall_valid', 'test_cindex_weighted',
                       'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
                       'num_test_samples', 'num_positive_samples', 'description']
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
        description = model_configs[model_name]['description']

        # Skip statistics for non-existent models
        if 'NON-EXISTENT' in description:
            print(f"\n{model_name}: {description}")
            # Show sample counts if available
            if 'num_test_samples' in model_data.columns:
                sample_counts = model_data['num_test_samples'].dropna()
                if len(sample_counts) > 0:
                    print(f"  [Model or checkpoints not found]")
            continue

        print(f"\n{model_name} ({description}):")

        # Show sample counts
        if 'num_test_samples' in model_data.columns:
            sample_counts = model_data['num_test_samples'].dropna()
            positive_counts = model_data['num_positive_samples'].dropna()
            if len(sample_counts) > 0:
                mean_samples = sample_counts.mean()
                mean_positives = positive_counts.mean()
                print(f"  Test samples: {mean_samples:.0f} (positives: {mean_positives:.0f})")

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
