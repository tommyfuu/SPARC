"""Per-feature MSE for the SPARC-P pretrained models on the held-out test set,
against a random-chance baseline (the training-set mean).

Only time-series features are scored; gender and age are not predicted.

Rows are written per feature for each fold 0-9, for 'mean_across_folds', for
'ensemble' (predictions averaged before scoring), and for 'baseline'. A final
row per model uses feature_name='ALL_TIME_SERIES_FEATURES' to aggregate across
features.

Output: {results_path}/eval_0_mse/mse_results_{dataset}_{modules}.csv

Usage:
    python eval_0_pretrained_mse_vs_chance.py --dataset MSK_CAIA --modules parsimonious --caia_portion prior
    python eval_0_pretrained_mse_vs_chance.py --dataset MSK_CAIA --quick_test True --num_folds_quick_test 2
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
import torch
import numpy as np
import pytorch_lightning as pl
import argparse

# COMMAND-LINE ARGUMENTS
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
parser.add_argument('--results_path', type=str, default=sparc_paths.SPARC_MODEL_ROOT + '/results_vis_060726',
                    help='Path to save results')

args = parser.parse_args()

# CONFIGURATION
option = 'pretrain'
modules = args.modules
dataset = args.dataset
caia_portion = args.caia_portion

## HYPERPARAMETERS ##
batch_size = 512
data_source_path = args.data_source_path
model_path = args.model_path
results_path = args.results_path

# 060726 Architecture parameters
use_module_tokenizer = True
use_continuous_time = True
post_mlp_dropout = 0.2
missing_mask = True
temporal_aggregation = 'attention'

# Quick test mode (for rapid testing)
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

# DATA LOADING

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

# Build module index config for 060726 models
module_index_cfg = build_module_index_config(modular_featuredim_dict)

# Calculate n_static_features based on module configuration
if modules == 'parsimonious':
    static_module_names = ['gender', 'age']
else:  # 'all'
    static_module_names = ['gender', 'age', 'primary_ctype', 'genetic_ancestry', 'met_status', 'clinical', 'mutation']
n_static_features = sum(modular_featuredim_dict.get(mod, 0) for mod in static_module_names)
print(f"Calculated n_static_features for {modules} with keys {static_module_names}: {n_static_features}")

print(f"Test set size: {len(split_dict['test'])}")

# SETUP DATA MODULE
pl.seed_everything(42)
data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default'
)

advanced_model = 'default'

# Import feature lists
from MSK_featurename_list import BASIC_TESTS, BASIC_VITALS, CBC_DIFFERENTIAL_TESTS

# Build included_features list based on module
if modules == 'parsimonious':
    included_features = ['gender', 'age'] + BASIC_TESTS + CBC_DIFFERENTIAL_TESTS + BASIC_VITALS
    # Only time series features are predicted (not gender/age)
    time_series_start_idx = 2  # Skip gender (idx 0) and age (idx 1)
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

# CALCULATE TRAIN SET MEAN (for chance baseline)
print("\nCalculating training set feature means for chance baseline...")

# Collect all training features from the train set
# Use the train split from split_dict
train_features = []
train_DMP_IDs = split_dict['train']
print(f"Number of training patients: {len(train_DMP_IDs)}")

# Create a simple data module to access train data
train_data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1, pretrain_idx=True,
    task='pretrain', binary_days=180, med='CISPLATIN', advanced_model='default'
)

# Setup the data module to initialize dataloaders
train_data_module.setup(stage='fit')

# Collect all training features
for batch, DMP_IDs in train_data_module.train_dataloader():
    (xs, times), y = batch
    # y contains the last timepoint features to predict
    train_features.append(y.cpu())

if len(train_features) == 0:
    print("WARNING: No training data collected. Using zeros as baseline.")
    # Use the number of features from test set
    n_features_estimate = len(included_features)
    train_feature_means = np.zeros(n_features_estimate)
else:
    train_features = torch.cat(train_features, dim=0)
    print(f"Collected {len(train_features)} training samples")

    # Calculate mean per feature (ignoring NaN values)
    train_feature_means = []
    for feature_idx in range(train_features.shape[1]):
        feature_values = train_features[:, feature_idx]
        valid_mask = ~torch.isnan(feature_values)
        if valid_mask.sum() > 0:
            feature_mean = feature_values[valid_mask].mean().item()
        else:
            feature_mean = 0.0  # Default if all NaN
        train_feature_means.append(feature_mean)

    train_feature_means = np.array(train_feature_means)
    print(f"Calculated means for {len(train_feature_means)} features")

# COLLECT TEST SET PREDICTIONS FROM ALL FOLDS
print("\n" + "="*70)
print("COLLECTING TEST SET PREDICTIONS FROM ALL FOLDS")
print("="*70)

num_folds = NUM_FOLDS_QUICK_TEST if QUICK_TEST else 10
print(f"\nLoading {num_folds} model fold(s)...")

all_fold_predictions = []  # List of [n_samples, n_features] arrays per fold
all_fold_ground_truth = []  # List of [n_samples, n_features] arrays per fold
all_DMP_IDs = None  # Should be same across all folds

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

    # Collect predictions and ground truth for this fold
    fold_predictions = []
    fold_ground_truth = []
    fold_DMP_IDs = []
    seen_patients = set()

    for batch, DMP_IDs in data_module.test_dataloader():
        torch.cuda.empty_cache()

        # Filter to keep only the first window for each patient
        unique_indices = []
        batch_DMP_IDs = []
        for idx, dmp_id in enumerate(DMP_IDs):
            if dmp_id not in seen_patients:
                unique_indices.append(idx)
                seen_patients.add(dmp_id)
                batch_DMP_IDs.append(dmp_id)

        if len(unique_indices) == 0:
            continue

        # Subset batch to unique patients only
        (xs, times), y = batch
        batch_subset = (((xs[unique_indices], times[unique_indices]), y[unique_indices]), batch_DMP_IDs)

        xs_combined, y_targets, src_key_padding_mask_combined, last_tp_xs_to_predict = model.data_prep(batch_subset)
        last_tp_xs_to_predict, last_tp_times = last_tp_xs_to_predict[:, :-1], last_tp_xs_to_predict[:, -1]

        # Move to device
        xs_combined = xs_combined.to(model.device)
        src_key_padding_mask_combined = src_key_padding_mask_combined.to(model.device)
        last_tp_times = last_tp_times.to(model.device)

        # Get predictions from model
        z, z1 = model(xs_combined, src_key_padding_mask=src_key_padding_mask_combined, last_tp_times=last_tp_times)

        # Get feature predictions (z1 is the reconstructed features)
        # For SPARC pretrain models, z1 should be [batch_size, n_features]
        predictions = z1.detach().cpu()

        fold_predictions.append(predictions)
        fold_ground_truth.append(last_tp_xs_to_predict.detach().cpu())
        fold_DMP_IDs.extend(batch_DMP_IDs)

        # Clean up GPU memory
        del xs_combined, src_key_padding_mask_combined, last_tp_times, z, z1, predictions

    # Concatenate all batches for this fold
    fold_predictions_concat = torch.cat(fold_predictions, dim=0)
    fold_ground_truth_concat = torch.cat(fold_ground_truth, dim=0)

    all_fold_predictions.append(fold_predictions_concat.numpy())
    all_fold_ground_truth.append(fold_ground_truth_concat.numpy())

    if all_DMP_IDs is None:
        all_DMP_IDs = fold_DMP_IDs

    print(f"Fold {fold}: Processed {len(fold_DMP_IDs)} unique patients")
    print(f"  Predictions shape: {fold_predictions_concat.shape}")
    print(f"  Ground truth shape: {fold_ground_truth_concat.shape}")

# Verify all folds have same samples and features
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

# Use ground truth from fold 0 (should be same across all folds)
ground_truth = all_fold_ground_truth[0]

# Verify train_feature_means has the correct length
if len(train_feature_means) != n_features:
    print(f"\nWARNING: train_feature_means length ({len(train_feature_means)}) != n_features ({n_features})")
    if len(train_feature_means) < n_features:
        print(f"  Padding train_feature_means with zeros for missing features")
        train_feature_means = np.pad(train_feature_means, (0, n_features - len(train_feature_means)), 'constant', constant_values=0.0)
    else:
        print(f"  Truncating train_feature_means to match n_features")
        train_feature_means = train_feature_means[:n_features]
    print(f"  Adjusted train_feature_means length: {len(train_feature_means)}")

# CALCULATE MSE PER FEATURE (TIME SERIES FEATURES ONLY)
print("\n" + "="*70)
print("CALCULATING MSE PER FEATURE (TIME SERIES FEATURES ONLY)")
print("="*70)
print(f"Skipping static features: gender (idx 0), age (idx 1)")
print(f"Computing MSE for time series features: indices {time_series_start_idx} to {n_features-1}")

results = []

# Store MSE values per feature across folds for variance calculation
fold_mse_per_feature = {}  # feature_idx -> list of MSE values across folds

# 1. Calculate MSE for each individual fold
for fold_idx, fold_preds in enumerate(all_fold_predictions):
    print(f"\nCalculating MSE for Fold {fold_idx}...")

    # Only calculate for time series features (skip gender and age)
    for feature_idx in range(time_series_start_idx, n_features):
        feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

        # Get predictions and ground truth for this feature
        pred = fold_preds[:, feature_idx]
        truth = ground_truth[:, feature_idx]

        # Calculate MSE (excluding NaN values)
        valid_mask = ~np.isnan(truth)
        if valid_mask.sum() > 0:
            squared_errors = (pred[valid_mask] - truth[valid_mask]) ** 2
            mse = np.mean(squared_errors)
            mse_std_within_fold = np.std(squared_errors)  # Variance within fold (across patients)
            n_valid = valid_mask.sum()

            # Store for cross-fold variance calculation
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
            'mse_std_within_fold': mse_std_within_fold,  # Std of squared errors within this fold
            'n_valid_samples': n_valid,
            'n_total_samples': n_samples
        })

    print(f"  Completed {n_features - time_series_start_idx} time series features")

# 2. Calculate aggregate statistics across folds per feature
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
        'mse_std_across_folds': std_mse_across_folds,  # Std of MSE across folds
        'mse_std_within_fold': np.nan,  # Not applicable for aggregate
        'n_valid_samples': n_samples,  # Same for all folds
        'n_total_samples': n_samples
    })

print(f"  Completed statistics for {n_features - time_series_start_idx} features")

# 3. Calculate MSE for ensemble (average predictions across folds first)
print(f"\nCalculating MSE for Ensemble (averaged across {num_folds} folds)...")
ensemble_preds = np.mean(all_fold_predictions, axis=0)

for feature_idx in range(time_series_start_idx, n_features):
    feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

    # Get predictions and ground truth for this feature
    pred = ensemble_preds[:, feature_idx]
    truth = ground_truth[:, feature_idx]

    # Calculate MSE (excluding NaN values)
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
        'mse_std_across_folds': np.nan,  # Not applicable for ensemble
        'n_valid_samples': n_valid,
        'n_total_samples': n_samples
    })

print(f"  Completed {n_features - time_series_start_idx} features")

# 4. Calculate MSE for random chance baseline (predict training set mean)
print(f"\nCalculating MSE for Random Chance Baseline (training set mean)...")

for feature_idx in range(time_series_start_idx, n_features):
    feature_name = included_features[feature_idx] if feature_idx < len(included_features) else f"feature_{feature_idx}"

    # Predict training set mean for all samples
    pred = np.full(n_samples, train_feature_means[feature_idx])
    truth = ground_truth[:, feature_idx]

    # Calculate MSE (excluding NaN values)
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
        'mse_std_across_folds': np.nan,  # Not applicable for baseline
        'n_valid_samples': n_valid,
        'n_total_samples': n_samples
    })

print(f"  Completed {n_features - time_series_start_idx} features")

# 5. Calculate aggregate statistics across ALL time series features
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

print(f"  Overall mean MSE (SPARC-P mean across folds): {overall_mean_mse:.6f}")
print(f"  Overall mean MSE (SPARC-P ensemble): {overall_ensemble_mse:.6f}")
print(f"  Overall mean MSE (Random Chance): {overall_baseline_mse:.6f}")

# SAVE RESULTS TO CSV
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)

# Create output directory
output_dir = f'{results_path}/eval_0_mse'
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
quick_test_suffix = '_quicktest' if QUICK_TEST else ''
output_file = f'{output_dir}/mse_results_{dataset}_{modules}{quick_test_suffix}.csv'
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print(f"Total rows: {len(results_df)}")

# Print overall results
print(f"\n{'='*70}")
print(f"OVERALL RESULTS (Across All Time Series Features)")
print(f"{'='*70}")
overall_df = results_df[results_df['feature_name'] == 'ALL_TIME_SERIES_FEATURES']
print(overall_df[['model', 'fold', 'mse', 'mse_std_across_folds']].to_string(index=False))

# Print per-feature summary statistics
print(f"\n{'='*70}")
print(f"PER-FEATURE SUMMARY STATISTICS")
print(f"{'='*70}")
mean_across_folds_df = results_df[results_df['fold'] == 'mean_across_folds']
print(f"\nTop 10 features with lowest MSE (SPARC-P mean across folds):")
print(mean_across_folds_df.nsmallest(10, 'mse')[['feature_name', 'mse', 'mse_std_across_folds']].to_string(index=False))

print(f"\nTop 10 features with highest MSE (SPARC-P mean across folds):")
print(mean_across_folds_df.nlargest(10, 'mse')[['feature_name', 'mse', 'mse_std_across_folds']].to_string(index=False))

# Print ensemble vs baseline comparison
print(f"\n{'='*70}")
print(f"ENSEMBLE VS BASELINE COMPARISON (Sample Features)")
print(f"{'='*70}")
ensemble_df = results_df[results_df['fold'] == 'ensemble']
baseline_df = results_df[results_df['fold'] == 'baseline']
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
