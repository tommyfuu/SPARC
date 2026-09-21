"""
design_choice_comparison_5_modular_new_fixed.py

Single-variable comparison of the per-module tokenizer.

    full   lilac-trained direct models   -umt True  -uct True -wbce True
    nomod  ablation arm A2_nomod         -umt False -uct True -wbce True

Both arms use the same model class (SPARC_inf_risk_modular) and the same
continuous-time setting, so the tokenizer is the only thing that varies.

design_choice_comparison_5's "without_modular" arm instead points at the
pre-060726 models, which use a different model class and lack the
continuous-time encoding and weighted BCE as well, so it confounds three
changes at once -- which its own figure title concedes ("Modular Tokenization
+ Continuous Time Encoding").

    python3 design_choice_comparison_5_modular_new_fixed.py -o AKI_risk -ds MSK -m parsimonious
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
# Use the fixed versions of data loaders and model
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset
from SPARC_inf_risk_modular import SPARC as SPARC_modular  # New version with modular tokenization

# Import module tokenizer config builder for use_module_tokenizer=True
from MSK_featurename_list import build_module_index_config

import torch
import torch.nn as nn
import pytorch_lightning as pl
import re
# data_source_path will be set via command-line argument (--data_source_path)
# Default: '/data/morrisq/fuc/SPARC_data/' if not specified


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
                except Exception:
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
parser.add_argument("-mm", "--missing_mask", type=str, default = "True", help="True | False — concatenate missingness indicators as extra features (default True); use False for no-mask (nm) models")
parser.add_argument("-cp", "--caia_portion", type=str, default = "prior", help="prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021")
parser.add_argument("-z", "--zeroshot", type=str, default = 'True', help='whether to use zeroshot (MSK-trained) models for MIMICIV tasks; when False, finetuned MIMICIV models are used and _finetuned is appended to save names, options: True, False')
parser.add_argument("-federated", "--federated", action='store_true', help='whether to only evaluate federated models (SPARC-P) or to evaluate all models including benchmarks')
parser.add_argument("-dsp", "--data_source_path", type=str, default=sparc_paths.DATA_ROOT + '/', help='path to data directory')
parser.add_argument("-mp", "--model_path", type=str, default=sparc_paths.SPARC_MODEL_ROOT, help='path to 060726 model checkpoints directory')
parser.add_argument("-rp", "--results_path", type=str, default=None, help='path to save results (defaults to model_path/results_vis_060726 if not specified)')

parser.add_argument("-ar", "--ablation_root", type=str,
                    default='/data1/morrisq/fuc/SPARC_ablation_direct_091826',
                    help='root holding the ablation arms; expects <root>/A2_nomod/...')
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
missing_mask = args.missing_mask.lower() == 'true'
caia_portion = args.caia_portion.lower()
zeroshot = args.zeroshot.lower() == 'true'
federated = args.federated
data_source_path = args.data_source_path
model_path = args.model_path
results_path = args.results_path if args.results_path else f'{args.model_path}/results_vis_060726'


# Example hardcoded values for testing (comment out when using command line args)
caia_portion = 'prior'


# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'febrile_neutropenia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True
# caia_portion = 'prior'

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'anemia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True
# caia_portion = 'prior'

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'anemia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True
# caia_portion = 'prior'
# zeroshot = True

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'neutropenia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True
# caia_portion = 'prior'

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'febrile_neutropenia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK_CAIA'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True
# caia_portion = 'prior'
# zeroshot = True

# Example hardcoded values for testing (comment out when using command line args)
rseed = 1024
option = 'Liver_toxicity'
days = 30
modules = 'parsimonious'
dataset = 'MSK'
med_of_choice = "IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB"
add_num_infusion = True

# Example hardcoded values for testing (comment out when using command line args)

# Example hardcoded values for testing (comment out when using command line args)

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'anemia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK_CAIA'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True

# # Example hardcoded values for testing (comment out when using command line args)
# rseed = 1024
# option = 'neutropenia'
# days = 30
# modules = 'parsimonious'
# dataset = 'MSK_CAIA'
# med_of_choice = 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL'
# add_num_infusion = True

# Model selection depends on task type:
#   - Side-effect tasks (AKI, anemia, neutropenia, febrile_neutropenia, Liver_toxicity):
#     'with_modular' is identical to SPARC_cox in design_choice_comparison_1, so it is
#     redundant here and skipped -- only 'without_modular' is evaluated/saved.
#   - Binary tasks (binary, binary_icu_pre): design_choice_comparison_1 does NOT produce
#     these results, so 'with_modular' is the only source and MUST be evaluated here.
# Both arms are always evaluated. Unlike design_choice_comparison_5 the control
# is not a different model class, so neither arm duplicates
# design_choice_comparison_1 and neither can be skipped.
all_models = ['full', 'nomod']


if not (bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)) or option in ['binary', 'binary_icu_pre']):
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

# Model architecture parameters (MUST match training configuration for 060726 models)
# These come from the training command: -umt True -uct True -pmd 0.2 -tp attention
use_module_tokenizer = True   # -umt True: per-clinical-module tokenization
use_continuous_time = True    # -uct True: learned continuous-time embedding
post_mlp_dropout = 0.2        # -pmd 0.2: dropout for post-transformer MLP
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
    option=option,
    modules=modules,
    valfold=0,  # Will iterate over folds later
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

# Model paths for comparison
ablation_root = args.ablation_root
results_root = args.results_root or f'{ablation_root}/results'
new_model_path = sparc_paths.SPARC_MODEL_ROOT  # lilac full-feature models
old_model_path = sparc_paths.RESULTS_ROOT  # Without modular tokenization

# Determine model names based on task type
if not bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    # Binary tasks (e.g., binary, binary_icu_pre)
    new_model_name = f'{modules}_direct_models_{option}_20tp_topchemos_att_modular'
    old_model_name = f'{modules}_direct_models_{option}_20tp_topchemos_att'
else:
    # Side-effect tasks (e.g., AKI_risk, febrile_neutropenia_risk, Liver_toxicity_risk)
    base_option = option.replace('_risk', '').replace('_next_risk', '')

    # For anemia, neutropenia, febrile_neutropenia: both old and new models use _hybrid_
    # For AKI, Liver_toxicity: new models use _risk_, old models use _hybrid_
    if base_option in ['anemia', 'neutropenia', 'febrile_neutropenia']:
        new_model_type_suffix = 'hybrid'
        old_model_type_suffix = 'hybrid'
    else:  # AKI, Liver_toxicity
        new_model_type_suffix = 'risk'
        old_model_type_suffix = 'hybrid'

    new_model_name = f'{modules}_direct_models_{new_model_type_suffix}_{base_option}_20tp_topchemos_att_modular'
    old_model_name = f'{modules}_direct_models_{old_model_type_suffix}_{base_option}_20tp_topchemos_att'

# Set model folder paths
model_folder_dict = {
    # lilac-trained direct models: -umt True -uct True -wbce True
    'full':  f'{new_model_path}/{modules}_{dataset}_060726/SPARC_direct_models/{new_model_name}',
    # ablation arm with ONLY the module tokenizer removed: -umt False -uct True -wbce True
    'nomod': f'{ablation_root}/A2_nomod/{modules}_{dataset}_060726/SPARC_direct_models/{new_model_name}_new',
}

print("=" * 80)
print("MODEL COMPARISON: Modular Tokenization + Continuous Time Encoding")
print("=" * 80)
print(f"full  (-umt True  -uct True -wbce True): {model_folder_dict['full']}")
print(f"nomod (-umt False -uct True -wbce True): {model_folder_dict['nomod']}")
print("=" * 80)

model_performance_dict = {}

# Save results to design_choices_5_modular_timeencode directory
save_name = f'{results_root}/design_choices_5_modular_new_fixed/modular_new_fixed_{modules}_{option}_{dataset}.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)


# Generate the split dict from fold dict (or use the one from load_dataset for CAIA)
if 'CAIA' in dataset:
    # For CAIA datasets, split_dict is already properly built by load_dataset()
    # It includes proper fold assignment from the JSON bundle
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
    # # Just use everyone
    # with open(f'{data_source_path}/{dataset}/stanford_ehrshots_mapped_cancer_patient_id_to_date_to_basic_lab_values_list_dict_080525.pickle', 'rb') as handle:
    #     DMP_ID_to_DATES_to_TESTSVALUES_DICT = pickle.load(handle)
    # Just use everyone                                                                                                          
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
# Check if the save csv file exists, if it does, load it
if os.path.exists(save_name):
    print(f"Loading existing model performance dict from {save_name}")
    saved_df = pd.read_csv(save_name)
    # Remove the rows where model contains 'averaged'
    performance_df_existing = saved_df[~saved_df['model'].str.contains('averaged')]

    # Determine task type to know which columns to check for NaN
    is_side_effect_task = bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option))

    # Only drop rows where relevant columns are NaN (not all columns)
    if is_side_effect_task:
        # For side-effect tasks, keep rows where at least one side-effect metric exists
        relevant_cols = ['test_cindex_overall_valid', 'test_cindex_weighted',
                        'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted']
        performance_df_existing = performance_df_existing.dropna(subset=relevant_cols, how='all')
    else:
        # For binary tasks, keep rows where at least one binary metric exists
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

# Determine model_set_option and modify option based on task type
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    # Side-effect tasks: append _risk and set model_set_option accordingly
    model_set_option = 'side_effects_risk'
    option = f'{option}_risk'
else:
    # Binary tasks: keep option as-is and set model_set_option to 'binary'
    model_set_option = 'binary'

all_folds = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if args.folds.lower() == 'all'
             else [int(x) for x in args.folds.split(',') if x.strip() != ''])

# Create shared data module (used for both models)
temporal_aggregation = 'attention'
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

# Iterate through both model configurations
for model_option in all_models:
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_option}")
    print(f"{'='*80}\n")

    pl.seed_everything(rseed)

    # Determine which model class to use
    # Same model class and same continuous-time setting for BOTH arms; the only
    # thing that varies is the per-module tokenizer. design_choice_comparison_5
    # instead swapped the model class and disabled continuous time too, so its
    # two arms differed in three ways at once.
    ModelClass = SPARC_modular
    use_continuous_time_flag = True
    use_module_tokenizer_flag = (model_option == 'full')

    # Skip if already computed for all folds
    if model_option in model_performance_dict and False not in [fold in model_performance_dict[model_option] for fold in all_folds]:
        print(f"Model {model_option} already exists for all folds, skipping")
        continue

    # Check performance for each fold
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

        # Get model path
        model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        print(f"Model folder: {model_folder}")

        if not os.path.exists(model_folder):
            print(f"WARNING: Model folder does not exist, skipping fold {val_fold}")
            continue

        existing_files = os.listdir(model_folder)
        # Find the checkpoint that has the highest val_loss (i.e. the one that is not last.ckpt)
        epoch_ckpts = [el for el in existing_files if 'epoch' in el]
        if not epoch_ckpts:
            print(f"WARNING: No epoch checkpoint found, skipping fold {val_fold}")
            continue

        optimal_model_path = epoch_ckpts[0]
        optimal_model_path = os.path.join(model_folder, optimal_model_path)
        print(f"Loading: {optimal_model_path}")

        # Load model with appropriate architecture parameters
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
            temporal_aggregation=temporal_aggregation,
            missing_mask=missing_mask,
            module_index_cfg=module_index_cfg if use_module_tokenizer_flag else None,
            use_module_tokenizer=use_module_tokenizer_flag,
            use_continuous_time=use_continuous_time_flag,
            n_static_features=n_static_features,
            post_mlp_dropout=post_mlp_dropout,
            lr=learning_rate,
        )
        print(f"Loaded {model_option} with: use_module_tokenizer={use_module_tokenizer_flag}, "
              f"use_continuous_time={use_continuous_time_flag}, post_mlp_dropout={post_mlp_dropout}")

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model on device: {device}")

        # Get test set predictions
        pl.seed_everything(rseed)

        # Collect all predictions and ground truths from the entire test dataset
        all_y_preds = []
        all_y_events = []
        all_time_to_events = []
        all_num_infusions = []

        batch_count = 0
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # SPARC models use data_prep method
                # Prepare batch data (data_prep works with CPU data)
                if include_gcsf:
                    # SPARC with GCSF
                    xs_combined, y, mask, _, num_infusion, _, gcsf_groundtruth = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                    )
                else:
                    # SPARC without GCSF (standard case)
                    xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                        batch, time_calibration=model.time_calibration,
                        perturbation_index=model.perturbation_index,
                        primary_ctype_returned=model.primary_ctype_returned,
                    )
                    gcsf_groundtruth = None

                # For binary tasks, create dummy infusion numbers if None
                if num_infusion is None:
                    # Create dummy zeros with same batch size as xs_combined
                    batch_size = xs_combined.shape[0]
                    num_infusion = torch.zeros(batch_size, dtype=torch.float32)
                    if batch_count == 0:
                        print(f"DEBUG: Created dummy num_infusion for binary task, shape: {num_infusion.shape}")

                # Move prepared data to GPU
                xs_combined = xs_combined.to(device)
                mask = mask.to(device)
                if add_num_infusion and num_infusion is not None:
                    num_infusion = num_infusion.to(device)
                if gcsf_groundtruth is not None:
                    gcsf_groundtruth = gcsf_groundtruth.to(device)

                # Get predictions - both SPARC models support gcsf_groundtruth parameter
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

                # Debug: Print first batch info
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
                    # For binary tasks without infusion numbers, create dummy infusion numbers (all zeros)
                    all_num_infusions.append(torch.zeros(fail_indicators.shape[0], dtype=torch.long))

                batch_count += 1

            print(f"\n=== DEBUG: Dataset Summary for {model_option} fold {val_fold} ===")
            print(f"Total batches: {batch_count}")
            print(f"Total samples: {sum(t.numel() for t in all_y_events)}")
            print(f"Positive events: {sum((t == 1).sum().item() for t in all_y_events)}")

            # Concatenate all batches
            all_y_preds = torch.cat(all_y_preds, dim=0)
            all_y_events = torch.cat(all_y_events, dim=0)
            all_time_to_events = torch.cat(all_time_to_events, dim=0)
            # Always concatenate infusion numbers (either real or dummy zeros for binary tasks)
            all_num_infusions = torch.cat(all_num_infusions, dim=0)

            ### DEBUGGING ##
            if val_fold == 0:
                print(f"[{model_option}] fold0 PRED sum: {all_y_preds.sum().item():.6f}, "
                    f"EVT sum: {all_y_events.sum().item()}, "
                    f"TTE sum: {all_time_to_events.float().sum().item():.1f}, "
                    f"N: {len(all_y_preds)}")
            ### DEBUGGING ##

            print(f"After concatenation - Total samples: {len(all_y_events)}, Positive: {(all_y_events == 1).sum().item()}")
            print("="*50 + "\n")

            print("PRED:", all_y_preds.shape, all_y_preds[:5].tolist(),
                "sum=", all_y_preds.sum().item())
            print("EVT :", all_y_events[:5].tolist(), "sum=", all_y_events.sum().item())
            print("TTE :", all_time_to_events[:5].tolist())
            print("INF :", all_num_infusions[:5].tolist(), "uniq=", len(torch.unique(all_num_infusions)))


            # Extract metrics based on task type
            if 'risk' in option or bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
                # Side-effect tasks: use C-index and next-infusion AUROC
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
                # Binary tasks (binary, binary_icu_pre): directly compute AUROC/AUPRC on entire cohort
                # No need for C-index since there's no time-to-event component
                auroc_fn = AUROC(task='binary')
                auprc_fn = AveragePrecision(task='binary')

                auroc_score = auroc_fn(all_y_preds, all_y_events.int()).item()
                auprc_score = auprc_fn(all_y_preds, all_y_events.int()).item()

                model_performance_dict[model_option][val_fold]['auroc'] = auroc_score
                model_performance_dict[model_option][val_fold]['auprc'] = auprc_score
                print(f"AUROC: {auroc_score:.4f}, AUPRC: {auprc_score:.4f}")

        print("_______")

    # Save after each model completes - rebuild dataframe from scratch to avoid duplicates
    # Include columns for both binary and side-effect tasks
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

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    model_performance_df.to_csv(save_name, index=False)

# # For each DL model, calculate the averaged y_pred over folds (late fusion) with test set y_true to calculate metrics
# for model_option in ['SPARC', 'SPARC-P', 'SPARC-P-Progressive'] + dl_models:
#     if model_set_option == 'binary':
#         y_preds_averaged = np.mean(np.array([model_performance_dict[model_option][val_fold]['y_preds'] for val_fold in model_performance_dict[model_option].keys()]), axis=0).tolist()
#         y_trues = model_performance_dict[model_option][0]['y_trues']  # y_trues is the same for all folds
#         model_performance_dict[f'{model_option}_averaged'] = {}
#         if model_set_option == 'binary':
#             auroc = AUROC(task='binary')
#             f1 = F1Score(task='binary')
#             auprc = AveragePrecision(task='binary')
#             auroc_score = auroc(torch.tensor(y_preds_averaged), torch.tensor(y_trues)).item()
#             f1_score = f1(torch.tensor(y_preds_averaged), torch.tensor(y_trues)).item()
#             auprc_score = auprc(torch.tensor(y_preds_averaged).float(), torch.tensor(y_trues)).item()
#             print(f'model {model_option} AUROC: {auroc_score}, F1: {f1_score}, AUPRC: {auprc_score}')
#             model_performance_dict[f'{model_option}_averaged'] = {'auroc': auroc_score,  'auprc': auprc_score, 'y_preds': y_preds_averaged, 'y_trues': y_trues}

# for model_option in ['lasttp_xgboost', 'temporal_xgboost']:
#     if model_set_option == 'binary':
#         y_preds_averaged = np.mean(np.array([model_performance_dict[model_option][val_fold]['y_preds'] for val_fold in model_performance_dict[model_option].keys()]), axis=0).tolist()
#         y_trues = model_performance_dict[model_option][0]['y_trues']  # y_trues is the same for all folds
#         model_performance_dict[f'{model_option}_averaged'] = {}
#         auroc = AUROC(task='binary')
#         f1 = F1Score(task='binary')
#         auprc = AveragePrecision(task='binary')
#         auroc_score = auroc(torch.tensor(y_preds_averaged), torch.tensor(y_trues)).item()
#         f1_score = f1(torch.tensor(y_preds_averaged), torch.tensor(y_trues)).item()
#         auprc_score = auprc(torch.tensor(y_preds_averaged).float(), torch.tensor(y_trues)).item()
#         print(f'model {model_option} AUROC: {auroc_score}, F1: {f1_score}, AUPRC: {auprc_score}')
#         model_performance_dict[f'{model_option}_averaged'] = {'auroc': auroc_score, 'auprc': auprc_score, 'y_preds': y_preds_averaged, 'y_trues': y_trues}

# # Final save - rebuild entire dataframe including averaged models
# model_performance_df = pd.DataFrame(columns=all_columns)
# for model_option in model_performance_dict.keys():
#     if 'average' not in model_option:
#         for val_fold in model_performance_dict[model_option].keys():
#             row = {'model': model_option, 'val_fold': val_fold}
#             row.update(model_performance_dict[model_option][val_fold])
#             row = [row[col] if col in row else None for col in all_columns]
#             model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
#     else:
#         row = {'model': model_option, 'val_fold': 'averaged'}
#         row.update(model_performance_dict[model_option])
#         row = [row[col] if col in row else None for col in all_columns]
#         model_performance_df = pd.concat([model_performance_df, pd.DataFrame([row], columns=all_columns)], ignore_index=True)
# # Save once at the end after all averaging
# model_performance_df.to_csv(save_name, index=False)

# print("done")