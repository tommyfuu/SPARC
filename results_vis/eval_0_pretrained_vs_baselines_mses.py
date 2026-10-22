"""Per-feature test MSE of the SPARC-P pretrained models (time-series features only) vs. baselines: training-set mean
(Random_Chance), previous timepoint, OLS on the feature's own previous value, and ridge (alpha 1) on all features at the
previous timepoint or at every earlier visit, plus missingness indicators. Regressions are fit on training folds 1-9.
With mode='test' the train split is empty, so the "training-set mean" used by Random_Chance and the fallbacks is 0.
Usage: python eval_0_pretrained_vs_baselines_mses.py --dataset MSK_CAIA --modules parsimonious --caia_portion prior
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, inspect
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
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
import torch
import numpy as np
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser(description='Calculate MSE per feature for SPARC-P pretrained models')
parser.add_argument('--dataset', type=str, default='MSK_CAIA',
                    help='Dataset name (MSK_CAIA, MSK_CAIA_timeshift, etc.)')
parser.add_argument('--modules', type=str, default='parsimonious',
                    choices=['practical', 'parsimonious'],
                    help='Module configuration')
parser.add_argument('--caia_portion', type=str, default='prior',
                    choices=['prior', 'post', 'entire'],
                    help='CAIA dataset portion (only for MSK_CAIA datasets)')
parser.add_argument('--quick_test', type=str, default='False',
                    help='Quick test mode (True/False)')
parser.add_argument('--num_folds_quick_test', type=int, default=2,
                    help='Number of folds in quick test mode')
parser.add_argument('--data_source_path', type=str, default=sparc_paths.DATA_ROOT + '/',
                    help='Path to data directory')
parser.add_argument('--model_path', type=str, default=sparc_paths.SPARC_MODEL_ROOT,
                    help='Path to model checkpoints')
parser.add_argument('--results_path', type=str, default=sparc_paths.SPARC_RESULTS_VIS_ROOT,
                    help='Path to save results')

args = parser.parse_args()

# configuration
option = 'pretrain'
modules = args.modules
dataset = args.dataset
caia_portion = args.caia_portion

# hyperparameters
batch_size = 512
data_source_path = args.data_source_path
model_path = args.model_path
results_path = args.results_path

# architecture flags; must match the pretrained checkpoints
use_module_tokenizer = True
use_continuous_time = True
post_mlp_dropout = 0.2
missing_mask = True
temporal_aggregation = 'attention'

QUICK_TEST = args.quick_test.lower() == 'true'
NUM_FOLDS_QUICK_TEST = args.num_folds_quick_test

print(f"\n{'='*70}")
print(f"CONFIGURATION")
print(f"{'='*70}")
print(f"Dataset: {dataset}")
print(f"Modules: {modules}")
print(f"CAIA portion: {caia_portion}")
print(f"Quick test: {QUICK_TEST}")
if QUICK_TEST:
    print(f"Num folds (quick test): {NUM_FOLDS_QUICK_TEST}")
print(f"Data source path: {data_source_path}")
print(f"Model path: {model_path}")
print(f"Results path: {results_path}")
print(f"{'='*70}\n")

# data loading

if QUICK_TEST:
    print("\n" + "="*70)
    print("QUICK TEST MODE ENABLED")
    print(f"- Using small data subset")
    print(f"- Loading only first {NUM_FOLDS_QUICK_TEST} model folds")
    print("="*70 + "\n")

print("Loading dataset using unified data loading function...")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=0,
    mode='test' if 'timeshift' not in dataset else 'test_all',
    rseed=42,
    competing_risk_death_check=False,
    include_gcsf=False,
    data_fraction_option='default',
    quick_test=QUICK_TEST,
    caia_portion=caia_portion,
)

print("modular_featuredim_dict", modular_featuredim_dict)

module_index_cfg = build_module_index_config(modular_featuredim_dict)

if modules == 'parsimonious':
    static_module_names = ['gender', 'age']
else:  # 'all'
    static_module_names = ['gender', 'age', 'primary_ctype', 'genetic_ancestry', 'met_status', 'clinical', 'mutation']
n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)
print(f"Calculated n_static_features for {modules} with keys {static_module_names}: {n_static_features}")

print(f"Test set size: {len(split_dict['test'])}")

# data module
pl.seed_everything(42)
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default'
)

advanced_model = 'default'

from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS, CBC_DIFFERENTIAL_TESTS

if modules == 'parsimonious':
    included_features = ['gender', 'age'] + BASIC_TESTS + CBC_DIFFERENTIAL_TESTS + BASIC_VITALS
    time_series_start_idx = 2  # skip gender (0) and age (1)
    time_series_features = BASIC_TESTS + CBC_DIFFERENTIAL_TESTS + BASIC_VITALS
    print(f"\nParsimonious module: Total features: {len(included_features)}")
    print(f"  - Static features (not predicted): gender, age (indices 0-1)")
    print(f"  - Time series features (predicted): {len(time_series_features)} features (indices {time_series_start_idx}+)")
else:
    included_features = ['gender', 'age'] + BASIC_TESTS + BASIC_VITALS
    time_series_start_idx = 2
    time_series_features = BASIC_TESTS + BASIC_VITALS
    print(f"\nPractical module: Total features: {len(included_features)}")
    print(f"  - Static features (not predicted): gender, age (indices 0-1)")
    print(f"  - Time series features (predicted): {len(time_series_features)} features (indices {time_series_start_idx}+)")

# training-set mean per feature (chance baseline)
print("\nCalculating training set feature means for chance baseline...")

train_features = []
train_DMP_IDs = split_dict['train']
print(f"Number of training patients: {len(train_DMP_IDs)}")

train_data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default'
)

train_data_module.setup(stage='fit')

for batch, DMP_IDs in train_data_module.train_dataloader():
    (xs, times), y = batch
    # y is the last-timepoint features to predict
    train_features.append(y.cpu())

# train split is empty with mode='test', so these means are 0 (data are z-scored)
if len(train_features) == 0:
    print("WARNING: No training data collected. Using zeros as baseline.")
    n_features_estimate = len(included_features)
    train_feature_means = np.zeros(n_features_estimate)
else:
    train_features = torch.cat(train_features, dim=0)
    print(f"Collected {len(train_features)} training samples")

    # per-feature mean, ignoring NaN
    train_feature_means = []
    for feature_idx in range(train_features.shape[1]):
        feature_values = train_features[:, feature_idx]
        valid_mask = ~torch.isnan(feature_values)
        if valid_mask.sum() > 0:
            feature_mean = feature_values[valid_mask].mean().item()
        else:
            feature_mean = 0.0
        train_feature_means.append(feature_mean)

    train_feature_means = np.array(train_feature_means)
    print(f"Calculated means for {len(train_feature_means)} features")

print("\n" + "="*70)
print("COLLECTING TEST SET PREDICTIONS FROM ALL FOLDS")
print("="*70)

num_folds = NUM_FOLDS_QUICK_TEST if QUICK_TEST else 10
print(f"\nLoading {num_folds} model fold(s)...")

all_fold_predictions = []  # per fold: [n_samples, n_features]
all_fold_ground_truth = []
prev_tp_baseline = []      # [n_samples, n_features], filled on fold 0
test_hist, test_times = [], []   # [B, T, F], [B, T]
all_DMP_IDs = None

for fold in range(num_folds):
    print(f"\n--- Processing Fold {fold} ---")
    pl.seed_everything(42)

    if modules == 'practical':
        model_folder = f'{model_path}/practical_MSK_CAIA_060726/practical_pretrain_20tp_augmented_att_modular/fold{fold}/'
    elif modules == 'parsimonious':
        model_folder = f'{model_path}/parsimonious_MSK_CAIA_060726/parsimonious_pretrain_20tp_augmented_att_modular/fold{fold}/'

    existing_files = os.listdir(model_folder)
    model_checkpoint = [el for el in existing_files if 'epoch' in el][0]
    optimal_model_path = f'{model_folder}/{model_checkpoint}'
    print(f"Loading model from: {optimal_model_path}")

    model = SPARC.load_from_checkpoint(
        optimal_model_path,
        modular_featuredim_dict=modular_featuredim_dict,
        option='pretrain',
        advanced_model=advanced_model,
        max_time_len=20,
        temporal_aggregation=temporal_aggregation,
        missing_mask=missing_mask,
        module_index_cfg=module_index_cfg,
        use_module_tokenizer=use_module_tokenizer,
        use_continuous_time=use_continuous_time,
        n_static_features=n_static_features,
        post_mlp_dropout=post_mlp_dropout
    )
    model.eval()

    fold_predictions = []
    fold_ground_truth = []
    fold_DMP_IDs = []
    seen_patients = set()

    for batch, DMP_IDs in data_module.test_dataloader():
        torch.cuda.empty_cache()

        # first window per patient only
        unique_indices = []
        batch_DMP_IDs = []
        for idx, dmp_id in enumerate(DMP_IDs):
            if dmp_id not in seen_patients:
                unique_indices.append(idx)
                seen_patients.add(dmp_id)
                batch_DMP_IDs.append(dmp_id)

        if len(unique_indices) == 0:
            continue

        (xs, times), y = batch
        batch_subset = (((xs[unique_indices], times[unique_indices]), y[unique_indices]), batch_DMP_IDs)

        xs_combined, y_targets, src_key_padding_mask_combined, last_tp_xs_to_predict = model.data_prep(batch_subset)
        last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]

        if fold == 0:
            # timepoint 0 = target visit, 1 = previous visit
            hist = xs[unique_indices].detach().cpu().numpy().astype(np.float64)       # [B, T, F], NaN = missing
            assert np.allclose(hist[:, 0, :], last_tp_xs_to_predict.detach().cpu().numpy(), equal_nan=True), \
                "raw timepoint 0 differs from the prediction target"
            prev_tp_baseline.append(hist[:, 1, :])
            test_hist.append(hist); test_times.append(times[unique_indices].detach().cpu().numpy().astype(np.float64))

        xs_combined = xs_combined.to(model.device)
        src_key_padding_mask_combined = src_key_padding_mask_combined.to(model.device)
        last_tp_times = last_tp_times.to(model.device)

        z, z1 = model(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

        # z1: reconstructed features, [batch_size, n_features]
        predictions = z1.detach().cpu()

        fold_predictions.append(predictions)
        fold_ground_truth.append(last_tp_xs_to_predict.detach().cpu())
        fold_DMP_IDs.extend(batch_DMP_IDs)

        del xs_combined, src_key_padding_mask_combined, last_tp_times, z, z1, predictions

    fold_predictions_concat = torch.cat(fold_predictions, dim=0)
    fold_ground_truth_concat = torch.cat(fold_ground_truth, dim=0)

    all_fold_predictions.append(fold_predictions_concat.numpy())
    all_fold_ground_truth.append(fold_ground_truth_concat.numpy())

    if all_DMP_IDs is None:
        all_DMP_IDs = fold_DMP_IDs

    print(f"Fold {fold}: Processed {len(fold_DMP_IDs)} unique patients")
    print(f"  Predictions shape: {fold_predictions_concat.shape}")
    print(f"  Ground truth shape: {fold_ground_truth_concat.shape}")

prev_tp_baseline = np.concatenate(prev_tp_baseline, axis=0)

n_samples = len(all_fold_predictions[0])
n_features = all_fold_predictions[0].shape[1]
print(f"\n" + "="*70)
print(f"Collected predictions from {num_folds} folds")
print(f"Total test samples: {n_samples}")
print(f"Total features: {n_features}")
assert all(len(p) == n_samples for p in all_fold_predictions), "All folds should have same number of test samples"
assert all(p.shape[1] == n_features for p in all_fold_predictions), "All folds should have same number of features"
print("ok All folds have consistent dimensions")
print("="*70)

# ground truth is the same across folds
ground_truth = all_fold_ground_truth[0]

if len(train_feature_means) != n_features:
    print(f"\nWARNING: train_feature_means length ({len(train_feature_means)}) != n_features ({n_features})")
    if len(train_feature_means) < n_features:
        print(f"  Padding train_feature_means with zeros for missing features")
        train_feature_means = np.pad(train_feature_means, (0, n_features - len(train_feature_means)), 'constant', constant_values=0.0)
    else:
        print(f"  Truncating train_feature_means to match n_features")
        train_feature_means = train_feature_means[:n_features]
    print(f"  Adjusted train_feature_means length: {len(train_feature_means)}")

# regression baselines
from sklearn.linear_model import LinearRegression, Ridge
# fit on training folds 1-9
_dd_train, _, split_dict_regtrain, _component_train, _ = load_dataset(
    dataset=dataset, data_source_path=data_source_path, option=option, modules=modules, valfold=0,
    mode='train', rseed=42, competing_risk_death_check=False, include_gcsf=False,
    data_fraction_option='default', quick_test=QUICK_TEST, caia_portion=caia_portion)
assert not set(split_dict_regtrain['train']) & set(split_dict['test']), "training and test patients overlap"
pl.seed_everything(42)
_reg_dm = FlexibleFeatureDataModule(
    _dd_train, split_dict_regtrain,
    dataset=dataset, component=_component_train,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default')
_reg_dm.setup(stage='fit')
train_target, train_prev = [], []   # timepoints 0 and 1
train_hist, train_times = [], []
for batch, _ids in _reg_dm.train_dataloader():
    (xs, times), y = batch
    _x = xs.detach().cpu().numpy().astype(np.float64)
    train_target.append(_x[:, 0, :]); train_prev.append(_x[:, 1, :])
    train_hist.append(_x); train_times.append(times.detach().cpu().numpy().astype(np.float64))
del _dd_train
train_target = np.concatenate(train_target, axis=0)[:, :n_features]
train_prev = np.concatenate(train_prev, axis=0)[:, :n_features]
print(f"\nFitting regression baselines on {len(train_target)} training windows...")
prev_fill = np.array([np.nanmean(c) if np.isfinite(c).any() else 0.0 for c in train_prev.T])
def _design(prev):
    """previous visit values (NaN -> train mean) + missingness flags"""
    miss = np.isnan(prev)
    return np.hstack([np.where(miss, prev_fill, prev), miss.astype(np.float64)])
X_train_ridge, X_test_ridge = _design(train_prev), _design(prev_tp_baseline[:, :n_features])
linear_prev_pred = np.full((n_samples, n_features), np.nan)
ridge_pred = np.full((n_samples, n_features), np.nan)
ridge_all_pred = np.full((n_samples, n_features), np.nan)

def _design_all(hist, times):
    """all earlier visits: values (NaN -> train mean), missingness flags, time to target visit, visit-exists flag"""
    h = hist[:, 1:, :n_features]
    miss = np.isnan(h)
    vals = np.where(miss, prev_fill[None, None, :], h)
    gap = times[:, :1] - times[:, 1:]
    exists = ~(miss.all(axis=2) & np.isnan(times[:, 1:]))
    gap_ok = np.isfinite(gap) & exists
    return np.hstack([vals.reshape(len(h), -1), miss.reshape(len(h), -1).astype(np.float64),
                      np.where(gap_ok, gap, 0.0), exists.astype(np.float64)])

_th = np.concatenate(train_hist, axis=0); _tt = np.concatenate(train_times, axis=0)
X_train_all = _design_all(_th, _tt); del _th, _tt, train_hist, train_times
X_test_all = _design_all(np.concatenate(test_hist, axis=0), np.concatenate(test_times, axis=0))
# standardize (time gaps are on another scale)
_mu, _sd = X_train_all.mean(0), X_train_all.std(0)
_sd[_sd == 0] = 1.0
X_train_all = (X_train_all - _mu) / _sd; X_test_all = (X_test_all - _mu) / _sd
print(f"All-timepoint ridge design: {X_train_all.shape[1]} columns ({X_train_all.shape[0]} training windows)")
for feature_idx in range(time_series_start_idx, n_features):
    yt = train_target[:, feature_idx]
    obs = ~np.isnan(yt)
    # OLS on the feature's own previous value
    both = obs & ~np.isnan(train_prev[:, feature_idx])
    prev_test = prev_tp_baseline[:, feature_idx]
    if both.sum() >= 2:
        lr = LinearRegression().fit(train_prev[both, feature_idx:feature_idx + 1], yt[both])
        linear_prev_pred[:, feature_idx] = np.where(np.isnan(prev_test), train_feature_means[feature_idx],
                                                    lr.predict(np.nan_to_num(prev_test)[:, None]))
    else:
        linear_prev_pred[:, feature_idx] = train_feature_means[feature_idx]
    # ridge on all features
    ridge_pred[:, feature_idx] = Ridge(alpha=1.0).fit(X_train_ridge[obs], yt[obs]).predict(X_test_ridge) \
        if obs.sum() >= 2 else train_feature_means[feature_idx]
    ridge_all_pred[:, feature_idx] = Ridge(alpha=1.0).fit(X_train_all[obs], yt[obs]).predict(X_test_all) \
        if obs.sum() >= 2 else train_feature_means[feature_idx]

# per-feature MSE (time-series features only)
print("\n" + "="*70)
print("CALCULATING MSE PER FEATURE (TIME SERIES FEATURES ONLY)")
print("="*70)
print(f"Skipping static features: gender (idx 0), age (idx 1)")
print(f"Computing MSE for time series features: indices {time_series_start_idx} to {n_features-1}")

results = []

# feature_idx -> per-fold MSE, for the across-fold std
fold_mse_per_feature = {}

# 1. per fold
for fold_idx, fold_preds in enumerate(all_fold_predictions):
    print(f"\nCalculating MSE for Fold {fold_idx}...")

    for feature_idx in range(time_series_start_idx, n_features):
        feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

        pred = fold_preds[:, feature_idx]
        truth = ground_truth[:, feature_idx]

        valid_mask = ~np.isnan(truth)
        if valid_mask.sum() > 0:
            squared_errors = (pred[valid_mask] - truth[valid_mask]) ** 2
            mse = np.mean(squared_errors)
            mse_std_within_fold = np.std(squared_errors)  # across patients
            n_valid = valid_mask.sum()

            if feature_idx not in fold_mse_per_feature:
                fold_mse_per_feature[feature_idx] = []
            fold_mse_per_feature[feature_idx].append(mse)
        else:
            mse = np.nan
            mse_std_within_fold = np.nan
            n_valid = 0

        results.append({
            'dataset': dataset,
            'modules': modules,
            'model': 'SPARC-P',
            'fold': fold_idx,
            'feature_idx': feature_idx,
            'feature_name': feature_name,
            'mse': mse,
            'mse_std_within_fold': mse_std_within_fold,
            'n_valid_samples': n_valid,
            'n_total_samples': n_samples
        })

    print(f"  Completed {n_features - time_series_start_idx} time series features")

# 2. aggregate across folds per feature
print(f"\nCalculating per-feature statistics across folds...")
for feature_idx in range(time_series_start_idx, n_features):
    feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

    if feature_idx in fold_mse_per_feature and len(fold_mse_per_feature[feature_idx]) > 0:
        mse_values = np.array(fold_mse_per_feature[feature_idx])
        mse_values_valid = mse_values[~np.isnan(mse_values)]

        if len(mse_values_valid) > 0:
            mean_mse_across_folds = np.mean(mse_values_valid)
            std_mse_across_folds = np.std(mse_values_valid) if len(mse_values_valid) > 1 else 0.0
        else:
            mean_mse_across_folds = np.nan
            std_mse_across_folds = np.nan
    else:
        mean_mse_across_folds = np.nan
        std_mse_across_folds = np.nan

    results.append({
        'dataset': dataset,
        'modules': modules,
        'model': 'SPARC-P',
        'fold': 'mean_across_folds',
        'feature_idx': feature_idx,
        'feature_name': feature_name,
        'mse': mean_mse_across_folds,
        'mse_std_across_folds': std_mse_across_folds,
        'mse_std_within_fold': np.nan,
        'n_valid_samples': n_samples,
        'n_total_samples': n_samples
    })

print(f"  Completed statistics for {n_features - time_series_start_idx} features")

# 3. ensemble: average predictions across folds, then score
print(f"\nCalculating MSE for Ensemble (averaged across {num_folds} folds)...")
ensemble_preds = np.mean(all_fold_predictions, axis=0)

for feature_idx in range(time_series_start_idx, n_features):
    feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

    pred = ensemble_preds[:, feature_idx]
    truth = ground_truth[:, feature_idx]

    valid_mask = ~np.isnan(truth)
    if valid_mask.sum() > 0:
        squared_errors = (pred[valid_mask] - truth[valid_mask]) ** 2
        mse = np.mean(squared_errors)
        mse_std_within_fold = np.std(squared_errors)
        n_valid = valid_mask.sum()
    else:
        mse = np.nan
        mse_std_within_fold = np.nan
        n_valid = 0

    results.append({
        'dataset': dataset,
        'modules': modules,
        'model': 'SPARC-P',
        'fold': 'ensemble',
        'feature_idx': feature_idx,
        'feature_name': feature_name,
        'mse': mse,
        'mse_std_within_fold': mse_std_within_fold,
        'mse_std_across_folds': np.nan,
        'n_valid_samples': n_valid,
        'n_total_samples': n_samples
    })

print(f"  Completed {n_features - time_series_start_idx} features")

# 4. chance baseline: predict the training-set mean
print(f"\nCalculating MSE for Random Chance Baseline (training set mean)...")

for feature_idx in range(time_series_start_idx, n_features):
    feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

    pred = np.full(n_samples, train_feature_means[feature_idx])
    truth = ground_truth[:, feature_idx]

    valid_mask = ~np.isnan(truth)
    if valid_mask.sum() > 0:
        squared_errors = (pred[valid_mask] - truth[valid_mask]) ** 2
        mse = np.mean(squared_errors)
        mse_std_within_fold = np.std(squared_errors)
        n_valid = valid_mask.sum()
    else:
        mse = np.nan
        mse_std_within_fold = np.nan
        n_valid = 0

    results.append({
        'dataset': dataset,
        'modules': modules,
        'model': 'Random_Chance',
        'fold': 'baseline',
        'feature_idx': feature_idx,
        'feature_name': feature_name,
        'mse': mse,
        'mse_std_within_fold': mse_std_within_fold,
        'mse_std_across_folds': np.nan,
        'n_valid_samples': n_valid,
        'n_total_samples': n_samples
    })

print(f"  Completed {n_features - time_series_start_idx} features")

# 4b. previous-timepoint and regression baselines (NaN -> train mean)
for baseline_name, baseline_values in [('Previous_Timepoint', prev_tp_baseline),
                                       ('Linear_Previous_Timepoint', linear_prev_pred),
                                       ('Ridge_Previous_Timepoint_Missingness', ridge_pred),
                                       ('Ridge_All_Timepoints_Missingness', ridge_all_pred)]:
    print(f"\nCalculating MSE for {baseline_name} baseline...")
    for feature_idx in range(time_series_start_idx, n_features):
        feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

        carried = baseline_values[:, feature_idx]
        no_history = np.isnan(prev_tp_baseline[:, feature_idx])   # previous timepoint not measured
        pred = np.where(np.isnan(carried), train_feature_means[feature_idx], carried)
        truth = ground_truth[:, feature_idx]

        valid_mask = ~np.isnan(truth)
        if valid_mask.sum() > 0:
            squared_errors = (pred[valid_mask] - truth[valid_mask]) ** 2
            mse = np.mean(squared_errors)
            mse_std_within_fold = np.std(squared_errors)
            n_valid = valid_mask.sum()
            # % rows with no previous value (n/a for ridge)
            pct_fallback = np.nan if baseline_name.startswith('Ridge') else 100 * np.mean(no_history[valid_mask])
        else:
            mse = np.nan
            mse_std_within_fold = np.nan
            n_valid = 0
            pct_fallback = np.nan

        results.append({
            'dataset': dataset,
            'modules': modules,
            'model': baseline_name,
            'fold': 'baseline',
            'feature_idx': feature_idx,
            'feature_name': feature_name,
            'mse': mse,
            'mse_std_within_fold': mse_std_within_fold,
            'mse_std_across_folds': np.nan,
            'n_valid_samples': n_valid,
            'n_total_samples': n_samples,
            'pct_fallback_to_train_mean': pct_fallback,
        })

    print(f"  Completed {n_features - time_series_start_idx} features")

# 5. aggregate over all time-series features
print(f"\nCalculating aggregate statistics across ALL time series features...")

# SPARC-P mean across folds
sparc_mean_fold_mses = [row['mse'] for row in results
                         if row['model'] == 'SPARC-P' and row['fold'] == 'mean_across_folds']
sparc_mean_fold_mses_valid = [x for x in sparc_mean_fold_mses if not np.isnan(x)]
if len(sparc_mean_fold_mses_valid) > 0:
    overall_mean_mse = np.mean(sparc_mean_fold_mses_valid)
    overall_std_mse = np.std(sparc_mean_fold_mses_valid)
else:
    overall_mean_mse = np.nan
    overall_std_mse = np.nan

results.append({
    'dataset': dataset,
    'modules': modules,
    'model': 'SPARC-P',
    'fold': 'overall_across_all_features',
    'feature_idx': -1,
    'feature_name': 'ALL_TIME_SERIES_FEATURES',
    'mse': overall_mean_mse,
    'mse_std_within_fold': np.nan,
    'mse_std_across_folds': overall_std_mse,
    'n_valid_samples': n_samples,
    'n_total_samples': n_samples
})

# SPARC-P ensemble
sparc_ensemble_mses = [row['mse'] for row in results
                        if row['model'] == 'SPARC-P' and row['fold'] == 'ensemble']
sparc_ensemble_mses_valid = [x for x in sparc_ensemble_mses if not np.isnan(x)]
if len(sparc_ensemble_mses_valid) > 0:
    overall_ensemble_mse = np.mean(sparc_ensemble_mses_valid)
    overall_ensemble_std = np.std(sparc_ensemble_mses_valid)
else:
    overall_ensemble_mse = np.nan
    overall_ensemble_std = np.nan

results.append({
    'dataset': dataset,
    'modules': modules,
    'model': 'SPARC-P',
    'fold': 'ensemble_across_all_features',
    'feature_idx': -1,
    'feature_name': 'ALL_TIME_SERIES_FEATURES',
    'mse': overall_ensemble_mse,
    'mse_std_within_fold': np.nan,
    'mse_std_across_folds': overall_ensemble_std,
    'n_valid_samples': n_samples,
    'n_total_samples': n_samples
})

# Random chance baseline
baseline_mses = [row['mse'] for row in results
                 if row['model'] == 'Random_Chance' and row['fold'] == 'baseline']
baseline_mses_valid = [x for x in baseline_mses if not np.isnan(x)]
if len(baseline_mses_valid) > 0:
    overall_baseline_mse = np.mean(baseline_mses_valid)
    overall_baseline_std = np.std(baseline_mses_valid)
else:
    overall_baseline_mse = np.nan
    overall_baseline_std = np.nan

results.append({
    'dataset': dataset,
    'modules': modules,
    'model': 'Random_Chance',
    'fold': 'baseline_across_all_features',
    'feature_idx': -1,
    'feature_name': 'ALL_TIME_SERIES_FEATURES',
    'mse': overall_baseline_mse,
    'mse_std_within_fold': np.nan,
    'mse_std_across_folds': overall_baseline_std,
    'n_valid_samples': n_samples,
    'n_total_samples': n_samples
})

for baseline_name in ['Previous_Timepoint', 'Linear_Previous_Timepoint', 'Ridge_Previous_Timepoint_Missingness',
                      'Ridge_All_Timepoints_Missingness']:
    _m = [row['mse'] for row in results
          if row['model'] == baseline_name and row['fold'] == 'baseline' and not np.isnan(row['mse'])]
    results.append({
        'dataset': dataset,
        'modules': modules,
        'model': baseline_name,
        'fold': 'baseline_across_all_features',
        'feature_idx': -1,
        'feature_name': 'ALL_TIME_SERIES_FEATURES',
        'mse': np.mean(_m) if _m else np.nan,
        'mse_std_within_fold': np.nan,
        'mse_std_across_folds': np.std(_m) if _m else np.nan,
        'n_valid_samples': n_samples,
        'n_total_samples': n_samples
    })
    print(f"  Overall mean MSE ({baseline_name}): {np.mean(_m) if _m else np.nan:.6f}")

print(f"  Overall mean MSE (SPARC-P mean across folds): {overall_mean_mse:.6f}")
print(f"  Overall mean MSE (SPARC-P ensemble): {overall_ensemble_mse:.6f}")
print(f"  Overall mean MSE (Random Chance): {overall_baseline_mse:.6f}")

# save results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)

output_dir = f'{results_path}/eval_0_mse'
os.makedirs(output_dir, exist_ok=True)

quick_test_suffix = '_quicktest' if QUICK_TEST else ''
output_file = f'{output_dir}/mse_results_{dataset}_{modules}{quick_test_suffix}.csv'
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print(f"Total rows: {len(results_df)}")

print(f"\n{'='*70}")
print(f"OVERALL RESULTS (Across All Time Series Features)")
print(f"{'='*70}")
overall_df = results_df[results_df['feature_name'] == 'ALL_TIME_SERIES_FEATURES']
print(overall_df[['model', 'fold', 'mse', 'mse_std_across_folds']].to_string(index=False))

print(f"\n{'='*70}")
print(f"PER-FEATURE SUMMARY STATISTICS")
print(f"{'='*70}")
mean_across_folds_df = results_df[results_df['fold'] == 'mean_across_folds']
print(f"\nTop 10 features with lowest MSE (SPARC-P mean across folds):")
print(mean_across_folds_df.nsmallest(10, 'mse')[['feature_name', 'mse', 'mse_std_across_folds']].to_string(index=False))

print(f"\nTop 10 features with highest MSE (SPARC-P mean across folds):")
print(mean_across_folds_df.nlargest(10, 'mse')[['feature_name', 'mse', 'mse_std_across_folds']].to_string(index=False))

print(f"\n{'='*70}")
print(f"ENSEMBLE VS BASELINE COMPARISON (Sample Features)")
print(f"{'='*70}")
ensemble_df = results_df[results_df['fold'] == 'ensemble']
baseline_df = results_df[(results_df['fold'] == 'baseline') & (results_df['model'] == 'Random_Chance')]
comparison_df = pd.merge(
    ensemble_df[['feature_name', 'mse']],
    baseline_df[['feature_name', 'mse']],
    on='feature_name',
    suffixes=('_ensemble', '_baseline')
)
comparison_df['improvement'] = (comparison_df['mse_baseline'] - comparison_df['mse_ensemble']) / comparison_df['mse_baseline'] * 100
print(f"\nSample features (showing MSE improvement %):")
print(comparison_df.head(10)[['feature_name', 'mse_ensemble', 'mse_baseline', 'improvement']].to_string(index=False))

print("\n" + "="*70)
print("MSE CALCULATION COMPLETE")
print("="*70)
