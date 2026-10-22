"""XGBoost shallow-learning benchmarks: lasttp_xgboost (last timepoint) or temporal_xgboost (20 timepoints flattened).

Grid-searches an XGBClassifier on one fold and writes it to <root>/<model_option>/<dataset>/<modules>_<option>/fold_<n>/.
Risk framings (*_risk) are rejected since XGBClassifier cannot fit (event, duration) targets.
Usage: python3 SL_benchmarks_train_risk_corrected_fixed.py -ds MSK_CAIA -cp prior -mo temporal_xgboost -f 1 -o AKI -an True -d 180 -me "CISPLATIN,IFOSFAMIDE,PEMETREXED"
"""

import argparse
import json
import os
import pickle
import re
import sys

# Importable from any working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (os.path.join(REPO_ROOT, 'data'), os.path.join(REPO_ROOT, 'models'),
           os.path.join(REPO_ROOT, 'results_vis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from torchmetrics import AUROC, F1Score, AveragePrecision

from data_loaders_inf_risk_fixed import *   # noqa: F401,F403  (FlexibleFeatureDataModule et al.)
from dataset_loading import load_dataset


DEFAULT_DATA_SOURCE_PATH = os.environ.get(
    'SPARC_DATA_PATH', '/path/to/SPARC_data')
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.environ.get('SPARC_MODEL_PATH', '/path/to/SPARC_models'),
    'SL_benchmarks')

# Side-effect tasks (with or without a risk framing suffix).
SE_TASK_PATTERN = r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$'

# lasttp uses only the most recent timepoint; temporal flattens the whole window.
MODEL_REGISTRY = {
    'lasttp_xgboost':   {'representation': 'lasttp'},
    'temporal_xgboost': {'representation': 'temporal'},
}

PARAM_GRID = {'n_estimators': [2, 10, 100], 'max_depth': [2, 5]}


def build_estimator(rseed):
    return XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                         random_state=rseed)


def build_output_model_path(output_root, model_option, dataset, caia_portion,
                            modules, option, valfold):
    """<root>/<model_option>/<dataset>_<caia_portion>/<modules>_<option>/fold_<n>/ (prior and post kept apart)."""
    dataset_dir = f'{dataset}_{caia_portion}' if 'CAIA' in dataset else dataset
    return os.path.join(output_root, model_option, dataset_dir,
                        f'{modules}_{option}', f'fold_{valfold}')


loss_fn = nn.BCELoss(reduction='none')


# shared with eval_1/2/3 via results_vis/eval_utils.py
from eval_utils import calculate_weighted_c_index_by_infusion


def build_feature_matrix(ds, representation, include_infusion_dates, add_num_infusion):
    """Flatten a FlexibleFeatureDataset into a dense (n_samples, n_features) matrix.

    lasttp:   [last-timepoint values | mask | time]; NaN time is kept (XGBoost handles it).
    temporal: [values | mask | time | time-missingness] per timepoint, flattened.
    Must match the feature build in eval_1_across_models_comparison_ensemble_final_FP_fixed.py.
    """
    X_rows = []
    y_rows = []
    num_infusions = []

    for i in range(len(ds.dataset_DMP_IDs)):
        item = ds[i]
        xs_all, time_all = item[0][0], item[0][1]

        if representation == 'lasttp':
            xs_tp = xs_all[0].detach().clone()
            time_tp = time_all[0].detach().clone()
            mask = torch.isnan(xs_tp)
            xs_tp = torch.nan_to_num(xs_tp, nan=0.0)
            # NaN time kept on purpose
            time_col = time_tp.reshape(1).float()
            feats = torch.cat((xs_tp, mask.float(), time_col), dim=0)
        else:
            xs_tp = xs_all.detach().clone()
            time_tp = time_all.detach().clone()
            time_nan_mask = torch.isnan(time_tp).float().unsqueeze(-1)
            mask = torch.isnan(xs_tp)
            xs_tp = torch.nan_to_num(xs_tp, nan=0.0)
            time_col = torch.nan_to_num(time_tp.float().unsqueeze(-1), nan=0.0)
            feats = torch.cat((xs_tp, mask.float(), time_col, time_nan_mask), dim=1)

        feats = feats.reshape(-1)

        if include_infusion_dates:
            inf_tensor = torch.as_tensor(item[0][2], dtype=torch.float32).reshape(-1)
            num_infusions.append(float(inf_tensor[0]))
            if add_num_infusion:
                feats = torch.cat((feats, inf_tensor))

        X_rows.append(feats)
        y_rows.append(ds.ys[i])

    if len(X_rows) == 0:
        raise ValueError('Dataset produced 0 samples — check the split, dataset and task options.')

    X = torch.stack(X_rows).numpy().astype(np.float32)
    # NaN is expected in the lasttp time column; inf would mean a normalization bug
    if np.isinf(X).any():
        raise ValueError('Feature matrix contains infinite values.')
    return X, y_rows, num_infusions


def _to_jsonable(obj):
    """Drop tensors/masks and coerce numpy/torch scalars so json.dump succeeds."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()
                if k not in ('mask', 'per_infusion', 'per_infusion_auroc', 'binary_loss')}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Arguments
parser = argparse.ArgumentParser(
    prog='SPARC shallow-learning benchmarks',
    description='Train a tree-ensemble benchmark on the SPARC feature matrix',
)
parser.add_argument("-r", "--rseed", type=int, default=1024, help='random seed')
parser.add_argument("-mo", "--model", type=str, default='lasttp_xgboost',
                    help=f'model to train, options: {", ".join(sorted(MODEL_REGISTRY))}')
parser.add_argument("-m", "--modules", type=str, default='parsimonious', help='modules to include')
parser.add_argument("-f", "--valfold", type=int, default=1,
                    help='fold number for validation, options: 1-10, fold number = i-1')
parser.add_argument("-o", "--option", type=str, default='binary',
                    help='options: binary, anemia, binary_icu, binary_icu_pre, binary_risk, '
                         'and the <side_effect>[_risk|_next_risk] tasks')
parser.add_argument("-d", "--days", type=int, default=180, help='number of days for binary classification tasks')
parser.add_argument("-me", "--med_of_choice", type=str, default='CISPLATIN',
                    help='medication of choice to predict post prediction risk of X')
parser.add_argument("-ds", "--dataset", type=str, default='MSK',
                    help='dataset to use, options: MSK, MSK_timeshift, MIMICIV_hosp, stanfordEHRSHOT, MSK_CAIA')
parser.add_argument("-an", "--add_num_infusion", type=str, default="False",
                    help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default="train", help="train or test")
parser.add_argument("-qt", "--quick_test", type=str, default="False",
                    help="True | False — if True, runs on a small subset of the data (pipeline smoke test)")
parser.add_argument("-cp", "--caia_portion", type=str, default="prior",
                    help="prior | post | entire - which portion of MSK_CAIA to use (wrt 2021)")
parser.add_argument("-vp", "--volume_path", type=str, default=DEFAULT_DATA_SOURCE_PATH,
                    help=f'base path to all datasets (default: {DEFAULT_DATA_SOURCE_PATH}, '
                         f'override with $SPARC_DATA_PATH)')
parser.add_argument("-op", "--output_model_path", type=str, default=DEFAULT_OUTPUT_ROOT,
                    help=f'output ROOT for models; the per-model subtree is appended automatically '
                         f'(default: {DEFAULT_OUTPUT_ROOT}, override with $SPARC_MODEL_PATH)')

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
data_source_path = args.volume_path
output_root = args.output_model_path

advanced_model = 'default'
data_fraction_option = 'default'
competing_risk_death_check = False
max_time_len = 20

      
if valfold not in range(10):
    raise ValueError(f'valfold must be between 0 and 9 (pass 1-10 to -f argument, got {args.valfold})')

if model_option not in MODEL_REGISTRY:
    raise ValueError(f'model_option must be one of {sorted(MODEL_REGISTRY)} (got {model_option!r})')

representation = MODEL_REGISTRY[model_option]['representation']

if not os.path.isdir(data_source_path):
    raise FileNotFoundError(
        f'data_source_path does not exist: {data_source_path}\n'
        f'Pass -vp/--volume_path or set $SPARC_DATA_PATH to the SPARC_data directory.')

# Task / model compatibility
is_se_task = bool(re.match(SE_TASK_PATTERN, option))
is_risk_task = 'risk' in option

if is_risk_task:
    # targets are (event, duration) tuples, which XGBClassifier cannot fit
    raise ValueError(
        f"Task {option!r} uses a survival (time-to-event) framing, which "
        f"{model_option!r} cannot fit. Risk framings are not supported by the "
        f"XGBoost benchmarks; use a non-risk task such as "
        f"{option.replace('_next_risk', '').replace('_risk', '')!r}.")

# med_of_choice: 'A,B+C' -> [['A'], ['B', 'C']]
med_of_choice = [med.strip() for med in med_of_choice.split(',')]
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        result_meds.append([med.strip() for med in med_str.split('+')])
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print("med_of_choice is a list of medications:", med_of_choice)

# output dir
output_model_path = build_output_model_path(
    output_root, model_option, dataset, caia_portion, modules, option, valfold)
os.makedirs(output_model_path, exist_ok=True)
print("output model path", output_model_path)

# load dataset
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
    include_gcsf=False,
    data_fraction_option=data_fraction_option,
    quick_test=quick_test,
    caia_portion=caia_portion,
)
print("modular_featuredim_dict", modular_featuredim_dict)
print("current validation fold:", valfold)

# Infusion numbers are only meaningful for the side-effect tasks.
include_infusion_dates = is_se_task

if option in ['binary', 'binary_icu_pre', 'binary_risk'] and add_num_infusion:
    print(f"WARNING: add_num_infusion=True is not applicable for task '{option}'. Setting to False.")
    add_num_infusion = False

anc = 'CAIA' in dataset

pl.seed_everything(rseed)
np.random.seed(rseed)

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    data_source_path=data_source_path,
    batch_size=512, max_time_len=max_time_len, num_workers=1,
    task=option, binary_days=days, med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model=advanced_model, include_infusion_dates=include_infusion_dates, anc=anc)
print("data module initiated")

# Feature matrices
print("Building train feature matrix...")
train_X, train_y_l, _ = build_feature_matrix(
    data_module.train_dataset, representation, include_infusion_dates, add_num_infusion)
print("Building test feature matrix...")
test_X, test_y_l, test_num_infusions = build_feature_matrix(
    data_module.test_dataset, representation, include_infusion_dates, add_num_infusion)
print("train_X", train_X.shape, "test_X", test_X.shape)

if train_X.shape[1] != test_X.shape[1]:
    raise ValueError(f'Train/test feature dimension mismatch: {train_X.shape[1]} vs {test_X.shape[1]}')

# Grid search
print("param_grid", PARAM_GRID)

grid_search = GridSearchCV(
    estimator=build_estimator(rseed),
    param_grid=PARAM_GRID, cv=3, n_jobs=-1, verbose=2,
)

grid_search.fit(train_X, np.asarray(train_y_l))
print("best params", grid_search.best_params_, "best CV score", grid_search.best_score_)

# Evaluate on the held-out split
metrics = {}
y_pred = torch.tensor(grid_search.predict_proba(test_X)[:, 1]).float()

if option in ['binary', 'binary_icu_pre']:
    test_y = torch.tensor(test_y_l)
    metrics['auroc'] = AUROC(task='binary')(y_pred, test_y).item()
    metrics['f1'] = F1Score(task='binary')(y_pred, test_y).item()
    metrics['auprc'] = AveragePrecision(task='binary')(y_pred, test_y).item()
    print(f"AUROC: {metrics['auroc']}, F1: {metrics['f1']}, AUPRC: {metrics['auprc']}")
elif is_se_task:
    y_event = torch.tensor(test_y_l)
    c_idx_dict = calculate_weighted_c_index_by_infusion(
        y_pred, y_event, torch.ones_like(y_event), torch.tensor(test_num_infusions),
        mode='val', option=option)
    if c_idx_dict is not None:
        metrics.update(_to_jsonable(c_idx_dict))
        print(f"Weighted AUROC (next infusion): {c_idx_dict['auroc_next_infusion_weighted']}, "
              f"Overall AUROC (next infusion): {c_idx_dict['auroc_next_infusion_overall']}")
    else:
        print("Could not compute metrics (insufficient valid infusions)")
else:
    print(f"No evaluation defined for task '{option}'; skipping metrics.")

# Save the best model plus provenance, under the per-model-type directory
best_estimator = grid_search.best_estimator_

model_file = os.path.join(output_model_path, f'{model_option}_model.pickle')
with open(model_file, 'wb') as handle:
    pickle.dump(best_estimator, handle)

with open(os.path.join(output_model_path, 'best_params.json'), 'w') as handle:
    json.dump(_to_jsonable({'best_params': grid_search.best_params_,
                            'best_cv_score': grid_search.best_score_,
                            'param_grid': PARAM_GRID}), handle, indent=2)

with open(os.path.join(output_model_path, 'metrics.json'), 'w') as handle:
    json.dump(_to_jsonable(metrics), handle, indent=2)

with open(os.path.join(output_model_path, 'run_config.json'), 'w') as handle:
    json.dump(_to_jsonable({
        'model_option': model_option,
        'representation': representation,
        'dataset': dataset,
        'caia_portion': caia_portion if 'CAIA' in dataset else None,
        'modules': modules,
        'option': option,
        'days': days,
        'valfold': valfold,
        'rseed': rseed,
        'mode': mode,
        'add_num_infusion': add_num_infusion,
        'include_infusion_dates': include_infusion_dates,
        'quick_test': quick_test,
        'max_time_len': max_time_len,
        'med_of_choice': med_of_choice,
        'data_source_path': data_source_path,
        'output_root': output_root,
        'n_train': int(train_X.shape[0]),
        'n_test': int(test_X.shape[0]),
        'n_features': int(train_X.shape[1]),
    }), handle, indent=2)

print(f"Model saved to {model_file}")
