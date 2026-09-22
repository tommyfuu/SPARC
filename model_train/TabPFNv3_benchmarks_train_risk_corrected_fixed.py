import random
from pytorch_lightning import Trainer
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import copy
import pickle
import os
import sys
import json
# Put the repo root, data/ and models/ on sys.path, from any working directory.
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models'),
           os.path.join(_R, 'results_vis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset   # <-- all data loading in one place

# TabPFN imports
try:
    from tabpfn import TabPFNClassifier
except ImportError:
    raise ImportError("TabPFN not installed. Please install with: pip install tabpfn")

data_source_path=sparc_paths.DATA_ROOT + '/'

valfold = 0
advanced_model = 'default'
data_fraction_option = 'default'

import argparse
parser = argparse.ArgumentParser(
            prog='TabPFN v3 benchmark training for SPARC',
            description='Train the TabPFN v3 model on flattened temporal features',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-mo", "--model", type=str, default = 'temporal_tabpfn', help='model option: temporal_tabpfn')
parser.add_argument("-m", "--modules", type=str, default = 'parsimonious', help='modules to include')
parser.add_argument("-f", "--valfold", type=int, default = 1, help='fold number for validation, options: 1-10, fold number = i-1')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset to use, options: MSK, MSK_timeshift, MIMICIV_hosp, stanfordEHRSHOT')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-qt", "--quick_test", type=str, default = "False", help="True | False — if True, runs a quick test with 1 epoch and a small subset of the data (for testing the code pipeline)")
parser.add_argument("-cp", "--caia_portion", type=str, default = "prior", help="prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021")

args = parser.parse_args()
rseed = args.rseed
modules = args.modules
model_option = args.model
valfold = args.valfold - 1
option = args.option
days = args.days
med_of_choice = args.med_of_choice
dataset = args.dataset
add_num_infusion = args.add_num_infusion.lower() == 'true'
mode = args.mode.lower()
quick_test = args.quick_test.lower() == 'true'
caia_portion = args.caia_portion.lower()

# Hard-coded to 1 for fair comparison with other models (no ensemble voting)
n_ensemble = 1

# Validate valfold
if valfold not in range(10):
    raise ValueError(f'valfold must be between 0 and 9 (pass 1-10 to -f argument, got {args.valfold})')

# Validate model option
if model_option not in ['temporal_tabpfn']:
    raise ValueError("model_option must be 'temporal_tabpfn'")


# Shared with eval_1/eval_2/eval_3 and the design-choice scripts; one copy
# in results_vis/eval_utils.py so the metric cannot drift between them.
from eval_utils import calculate_weighted_c_index_by_infusion


competing_risk_death_check = False
med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]

# Handle combination medications (e.g., CARBOPLATIN+PACLITAXEL)
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

max_time_len = 20

# TabPFN only supports binary classification
if 'risk' in option:
    raise ValueError("Risk prediction framings are not supported with TabPFN. Use binary classification tasks only.")

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Initializing TabPFN v3 with n_ensemble={n_ensemble}, device={device}")
model = TabPFNClassifier(
    n_estimators=n_ensemble,  # Number of ensemble members
    device=device,  # Use GPU if available
    random_state=rseed
)

# Load dataset (all loading/normalization/splitting is in dataset_loading.py)
print("Loading dataset...")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=valfold,
    mode=mode,
    rseed=rseed,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=False,  # SL benchmarks don't use GCSF
    data_fraction_option=data_fraction_option,
    quick_test=quick_test,
    caia_portion=caia_portion,
)
print("modular_featuredim_dict", modular_featuredim_dict)
print("current validation fold:", valfold)

# Set anc flag for CAIA datasets
anc = 'CAIA' in dataset

# Random seed
pl.seed_everything(rseed)

if dataset == 'MSK':
    output_model_path = f'{sparc_paths.RESULTS_ROOT}/parsimonious/benchmarks/{model_option}/{modules}_{option}/fold_{valfold}/'
else:
    output_model_path = f'{sparc_paths.RESULTS_ROOT}/{modules}_{dataset}/benchmarks/{model_option}/{modules}_{option}_{dataset}/fold_{valfold}/'
print("output model path", output_model_path)

if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    include_infusion_dates = True
else:
    include_infusion_dates = False

# Override add_num_infusion for binary tasks (infusion numbers only relevant for side-effect tasks)
if option in ['binary', 'binary_icu_pre', 'binary_risk'] and add_num_infusion:
    print(f"WARNING: add_num_infusion=True is not applicable for task '{option}'. Setting to False.")
    add_num_infusion = False

data_module = FlexibleFeatureDataModule(data_dict_lab_dict, split_dict, dataset = dataset, component = component,
    batch_size = 512, max_time_len = max_time_len, num_workers=1,
    task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
    advanced_model = advanced_model, include_infusion_dates = include_infusion_dates, anc=anc)
print("data module initiated")

# Set random seed
np.random.seed(rseed)

# Prepare training data
print("Preparing training data...")
train_dataset = data_module.train_dataset
train_X_l = []
train_y_l = []
for i in range(len(data_module.train_dataset.dataset_DMP_IDs)):
    # temporal_tabpfn uses all time points
    xs_tp, time_tp = train_dataset.__getitem__(i)[0][0].detach().clone(), train_dataset.__getitem__(i)[0][1].detach().clone()
    time_nan_mask = torch.isnan(time_tp).int().unsqueeze(-1).detach().clone()

    # Find mask
    mask = torch.isnan(xs_tp).detach().clone()
    # Fill missing values with 0
    xs_tp[mask] = 0

    # Concatenate and flatten (temporal approach)
    time_tp_combine = time_tp.unsqueeze(-1)
    time_tp_combine[time_nan_mask.bool()] = 0.0
    xs_tp = torch.cat((xs_tp, mask, time_tp_combine, time_nan_mask), dim=1).view(-1)

    # Add num_infusions if needed
    if include_infusion_dates:
        xs_tp_infusions = train_dataset.__getitem__(i)[0][2]
        xs_tp_infusions = torch.tensor(xs_tp_infusions, dtype=torch.float32).unsqueeze(-1)
        xs_tp = torch.cat((xs_tp, xs_tp_infusions))

    train_X_l.append(xs_tp)
    train_y_l.append(train_dataset.ys[i])

train_X = torch.stack(train_X_l)
train_X = train_X.numpy()
train_y = np.array(train_y_l)

print(f"Training data shape: {train_X.shape}, labels shape: {train_y.shape}")

# TabPFN has dataset size limitations (typically ~10,000 samples and ~1,000 features)
# Check if we need to subsample
MAX_SAMPLES = 10000
MAX_FEATURES = 1000

if train_X.shape[0] > MAX_SAMPLES:
    print(f"WARNING: Training set has {train_X.shape[0]} samples, which exceeds TabPFN's limit of {MAX_SAMPLES}.")
    print(f"Randomly subsampling to {MAX_SAMPLES} samples...")
    subsample_idx = np.random.choice(train_X.shape[0], MAX_SAMPLES, replace=False)
    train_X = train_X[subsample_idx]
    train_y = train_y[subsample_idx]

if train_X.shape[1] > MAX_FEATURES:
    print(f"WARNING: Feature dimension is {train_X.shape[1]}, which exceeds TabPFN's limit of {MAX_FEATURES}.")
    print(f"You may need to use feature selection or dimensionality reduction.")
    # For now, we'll just truncate to the first MAX_FEATURES
    print(f"Truncating to first {MAX_FEATURES} features...")
    train_X = train_X[:, :MAX_FEATURES]

# Train TabPFN
print("Training TabPFN...")
model.fit(train_X, train_y)
print("Training complete!")

# Prepare test data
print("Preparing test data...")
test_dataset = data_module.test_dataset
test_X_l = []
test_y_l = []
test_num_infusions = []
for i in range(len(data_module.test_dataset.dataset_DMP_IDs)):
    # temporal_tabpfn uses all time points
    xs_tp, time_tp = test_dataset.__getitem__(i)[0][0].detach().clone(), test_dataset.__getitem__(i)[0][1].detach().clone()
    time_nan_mask = torch.isnan(time_tp).int().unsqueeze(-1).detach().clone()

    # Find mask
    mask = torch.isnan(xs_tp).detach().clone()
    # Fill missing values with 0
    xs_tp[mask] = 0

    # Concatenate and flatten (temporal approach)
    time_tp_combine = time_tp.unsqueeze(-1)
    time_tp_combine[time_nan_mask.bool()] = 0.0
    xs_tp = torch.cat((xs_tp, mask, time_tp.unsqueeze(-1), time_nan_mask), dim=1).view(-1)

    # Add num_infusions if needed
    if include_infusion_dates:
        xs_tp_infusions = test_dataset.__getitem__(i)[0][2]
        xs_tp_infusions = torch.tensor(xs_tp_infusions, dtype=torch.float32).unsqueeze(-1)
        xs_tp = torch.cat((xs_tp, xs_tp_infusions))
        test_num_infusions.append(xs_tp_infusions.item())

    test_X_l.append(xs_tp)
    test_y_l.append(test_dataset.ys[i])

test_X = torch.stack(test_X_l)
test_X = test_X.numpy()

# Apply same feature truncation as training if needed
if test_X.shape[1] > MAX_FEATURES:
    test_X = test_X[:, :MAX_FEATURES]

print(f"Test data shape: {test_X.shape}")

# Evaluate TabPFN
print("Evaluating TabPFN...")
# Use predict_proba for classification (returns probabilities)
y_pred_proba = model.predict_proba(test_X)
y_pred = y_pred_proba[:, 1]  # Probability of positive class

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

# Evaluate based on task type
if option in ['binary', 'binary_icu_pre']:
    auroc_score = auroc(torch.tensor(y_pred), torch.tensor(test_y_l))
    f1_score = f1(torch.tensor(y_pred), torch.tensor(test_y_l))
    auprc_score = auprc(torch.tensor(y_pred).float(), torch.tensor(test_y_l))
    print(f'AUROC: {auroc_score}, F1: {f1_score}, AUPRC: {auprc_score}')
elif option in ['anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', 'Liver_toxicity']:
    # Binary prediction task for next infusion side effect
    y_event = torch.tensor(test_y_l)
    test_num_infusions = torch.tensor(test_num_infusions)
    y_pred = torch.tensor(y_pred).float()
    c_idx_dict = calculate_weighted_c_index_by_infusion(y_pred, y_event, torch.ones_like(y_event), test_num_infusions, mode='val', option=option)
    if c_idx_dict is not None:
        print(f"Weighted AUROC (next infusion): {c_idx_dict['auroc_next_infusion_weighted']}, Overall AUROC (next infusion): {c_idx_dict['auroc_next_infusion_overall']}")
    else:
        print("Could not compute metrics (insufficient valid infusions)")

# Save the model
if not os.path.exists(output_model_path):
    os.makedirs(output_model_path, exist_ok=True)

with open(f'{output_model_path}/{model_option}_model.pickle', 'wb') as handle:
    pickle.dump(model, handle)

print(f"Model saved to {output_model_path}/{model_option}_model.pickle")
