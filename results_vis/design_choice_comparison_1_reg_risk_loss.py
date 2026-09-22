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
# Use the fixed versions of data loaders and model
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset
from SPARC_inf_risk_modular import *  # Use the 060726 version

# Import module tokenizer config builder for use_module_tokenizer=True
from MSK_featurename_list import build_module_index_config

data_source_path=sparc_paths.DATA_ROOT + '/'

# python3 across_SPARC_att_risk_comparison.py -o AKI -ds MSK -m practical  -me "CISPLATIN"  -an True   -r 1024


torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from eval_utils import calculate_weighted_c_index_by_infusion


import argparse
parser = argparse.ArgumentParser(
            prog='across_SPARC_att_risk_comparison',
            description='Compare SPARC side-effect models by loss: unweighted BCE (vanilla) vs Cox vs hybrid',
            epilog='Comparison script for different SPARC variants')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-o", "--option", type=str, default = 'AKI', help='side effect option: anemia, neutropenia, febrile_neutropenia, AKI, Liver_toxicity')
parser.add_argument("-m", "--modules", type=str, default = 'parsimonious', help='module category')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset: MSK, MSK_timeshift')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "True", help="whether to add number of infusion as a feature")
parser.add_argument("-vp", "--vanilla_model_path", type=str, default=None,
                    help="Root holding the unweighted-BCE ('vanilla') side-effect models, retrained "
                         "with -wbce False. Defaults to sparc_paths.SPARC_MODEL_ROOT_VANILLA. The "
                         "vanilla arm uses the non-risk framing (-o AKI, not AKI_risk), so its loss "
                         "is plain BCE rather than the weighted BCE the 060726 models used.")
parser.add_argument("-vs", "--vanilla_stem", type=str, default='MSK_092126',
                    help="Dataset stem inside the vanilla root, e.g. MSK_092126 for "
                         "<root>/<modules>_MSK_092126/SPARC_direct_models/")
parser.add_argument("-rp", "--results_path", type=str, default=None,
                    help="Where to write the comparison CSV (default: <SPARC_MODEL_ROOT>/results_vis_060726)")

args = parser.parse_args()
rseed = args.rseed
option_base = args.option
modules = args.modules
dataset = args.dataset
med_of_choice = args.med_of_choice
add_num_infusion = args.add_num_infusion.lower() == 'true'


# # Hardcoded values for testing
# # Hardcoded values for testing
# # Hardcoded values for testing - early
# # Hardcoded values for testing - early
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

# The vanilla arm comes from a separate root: those models were retrained with
# -wbce False, so the BCE is unweighted. The cox and hybrid arms stay on 060726.
vanilla_model_path = (args.vanilla_model_path
                      or getattr(sparc_paths, 'SPARC_MODEL_ROOT_VANILLA', None)
                      or all_model_path)
vanilla_stem = args.vanilla_stem

model_configs = {
    'SPARC_vanilla': {
        # non-risk framing -> option='side_effects' -> plain BCE, not cox
        'option': f'{option_base}',
        'temporal_aggregation': 'attention',
        'loss_option': 'default',
        'path': f'{vanilla_model_path}/{modules}_{vanilla_stem}/SPARC_direct_models/{modules}_direct_models_{option_base}_vanilla_20tp_topchemos_att_modular',
        'description': 'Vanilla SPARC (unweighted BCE loss, attention)'
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

results_path = args.results_path or f'{all_model_path}/results_vis_060726'
save_name = f'{results_path}/design_choices_1_regvscox/SPARC_reg_risk_comparison_{modules}_{option_base}_{dataset}.csv'
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

        ### DEBUGGING ##
        if val_fold == 0:
            print(f"[{model_name}] fold0 PRED sum: {all_y_preds.sum().item():.6f}, "
                f"EVT sum: {all_y_events.sum().item()}, "
                f"TTE sum: {all_time_to_events.float().sum().item():.1f}, "
                f"N: {len(all_y_preds)}")
        ### DEBUGGING ##


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
