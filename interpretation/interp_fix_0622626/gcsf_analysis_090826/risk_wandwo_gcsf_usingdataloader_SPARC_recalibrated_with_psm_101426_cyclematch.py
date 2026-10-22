#!/usr/bin/env python3
"""G-CSF effect on neutropenia by SPARC predicted risk tertile, propensity-score matched instead of IPTW (Supp Fig 8a).

Usage: python3 risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_101426_cyclematch.py -bf True -mp <model dir> [-mr 1 2 10]
-gl: G-CSF must start at most this many days before the prediction date. MSK_CAIA (-cp prior) only.

"""

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
import os
import re
import pickle
import sys
import inspect
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repodir = os.path.abspath(os.path.join(currentdir, '..', '..', '..'))

sys.path.insert(0, os.path.join(repodir, 'data'))
sys.path.insert(0, os.path.join(repodir, 'models'))

from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset

from SPARC_inf_risk_modular import SPARC
from MSK_featurename_list import build_module_index_config


import argparse

parser = argparse.ArgumentParser(
    prog='risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm',
    description='Compute SPARC risk-stratified G-CSF impact analysis with propensity score matching.',
    epilog='Text at the bottom of help')
parser.add_argument("-r",   "--rseed",             type=int,   default=1024,
                    help='random seed')
parser.add_argument("-o",   "--option",            type=str,   default='neutropenia',
                    help='base task option: neutropenia or febrile_neutropenia')
parser.add_argument("-d",   "--days",              type=int,   default=180,
                    help='number of days for binary classification tasks')
parser.add_argument("-m",   "--modules",           type=str,   default='parsimonious',
                    help='module category: all | allnonICD | practical')
parser.add_argument("-ds",  "--dataset",           type=str,   default='MSK_CAIA',
                    help='dataset: MSK | MSK_timeshift | MSK_CAIA | MSK_CAIA_timeshift | ...')
parser.add_argument("-me",  "--med_of_choice",     type=str,
                    default='DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                            'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                            'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL',
                    help='comma-separated medications (use + to join combination regimens)')
parser.add_argument("-an",  "--add_num_infusion",  type=str,   default='True',
                    help='True | False — add infusion number as feature')
parser.add_argument("-gl",  "--gcsf_lim",          type=str,   default='2',
                    help='G-CSF start-date window relative to prediction date: '
                         'None (no limit) | int (days before prediction date)')
parser.add_argument("-vp",  "--volume_path",       type=str,   default=sparc_paths.DATA_ROOT + '/',
                    help='base path to all datasets')
parser.add_argument("-mp",  "--model_base_path",   type=str,   default='',
                    help='override the model checkpoint directory. '
                         'If empty, uses the default path for the dataset type.')
parser.add_argument("-cp",  "--caia_portion",      type=str,   default='prior',
                    help='prior | post | entire — which portion of CAIA dataset to use (CAIA only)')
parser.add_argument("-od",  "--output_dir",        type=str,   default='',
                    help='directory to save output files. Defaults to ./out/')
parser.add_argument("-mr",  "--match_ratios",      type=int,   nargs='+', default=[1, 10],
                    help="k values (controls per G-CSF infusion) for the PSM")
parser.add_argument("-cals", "--caliper_sd",       type=float, default=0.2,
                    help="caliper on |logit PS difference| in units of its pooled SD (Austin 2011: 0.2); 0 disables")
parser.add_argument("-bf",  "--batch_first",       type=str,   default='True',
                    help="True | False -- build SPARC with batch_first=True so the encoder attends "
                         "over time. MUST match how the checkpoints were trained: batch_first changes "
                         "no tensor shapes, so a mismatch loads silently and costs 4-8 AUROC points.")
parser.add_argument("-ns",  "--num_strata",        type=int,   default=3,
                    help='number of equal-sized risk strata for the stratified '
                         'analysis: 5 = quintiles (default), 3 = tertiles, '
                         '4 = quartiles, 2 = median split. Cutoffs and output '
                         'filenames are keyed by this value, so tertile and '
                         'quintile runs do not overwrite each other.')

args = parser.parse_args()

rseed             = args.rseed
option            = args.option
days              = args.days
modules           = args.modules
dataset           = args.dataset
add_num_infusion  = args.add_num_infusion.lower() == 'true'
med_of_choice     = args.med_of_choice
data_source_path  = args.volume_path
caia_portion      = args.caia_portion.lower()
num_strata        = args.num_strata
if num_strata < 2:
    raise SystemExit(
        f"ERROR: -ns/--num_strata must be at least 2 (got {num_strata}); "
        "the stratified analysis needs at least two risk groups to compare."
    )

gcsf_lim = args.gcsf_lim
if gcsf_lim.lower() == 'none':
    gcsf_lim = None
else:
    gcsf_lim = int(gcsf_lim)

output_dir = args.output_dir if args.output_dir else os.path.join(currentdir, 'out')

calibration_suffix = 'raw'
if not args.output_dir:
    if dataset == 'MSK_CAIA':
        output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}')
    else:
        output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}_{dataset}')
    if args.batch_first == 'True':
        output_dir += '_bfTrue'
    output_dir += '_cyclematch'
os.makedirs(output_dir, exist_ok=True)
print(f"[output] writing to: {output_dir}")

competing_risk_death_check = False

# option must be the base name, without the _risk suffix
base_option = re.sub(r'(_next_risk|_risk)$', '', option)
if base_option not in ('neutropenia', 'febrile_neutropenia'):
    raise ValueError(
        f"Invalid option '{option}'. This script only supports "
        "'neutropenia' or 'febrile_neutropenia' (without _risk suffix)."
    )
option = base_option

is_caia = 'CAIA' in dataset

task_option         = f'{option}_risk'
model_set_option    = 'side_effects_risk'
temporal_aggregation = 'attention'

default_model_root  = sparc_paths.SPARC_MODEL_ROOT
model_subfolder     = (f'{modules}_MSK_CAIA_060726/SPARC_finetuned_models/'
                        f'{modules}_finetuned_models_hybrid_{option}_20tp_topchemos_att_modular')

model_root      = args.model_base_path if args.model_base_path else default_model_root
model_base_path = os.path.join(model_root, model_subfolder)

# comma separates regimens, + joins drugs in a combination
med_of_choice = med_of_choice.split(',')
med_of_choice = [m.strip() for m in med_of_choice]
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        result_meds.append([m.strip() for m in med_str.split('+')])
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print("med_of_choice:", med_of_choice)

# Hyperparameters
batch_size        = 512
max_time_len_data = 20
advanced_model    = 'default'
valfold           = 0   # placeholder; all 10 folds are used below

# train mode, to get the train/val/test splits
print("Loading dataset in train mode to get train/val/test splits...")
data_dict_lab_dict_train, modular_featuredim_dict, split_dict_full, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=task_option,
    modules=modules,
    valfold=valfold,
    mode='train',
    rseed=rseed,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=True,
    data_fraction_option='default',
    caia_portion=caia_portion,
)
print("modular_featuredim_dict:", modular_featuredim_dict)

# build gcsf_dict from the supportive-meds pickle if load_dataset did not
if 'gcsf_dict' not in data_dict_lab_dict_train:
    if is_caia:
        sparc_data_root = data_source_path
        dataset_folder  = os.path.join(sparc_data_root, dataset)
        supportive_meds_path = None
        if os.path.isdir(dataset_folder):
            candidates = [
                f for f in os.listdir(dataset_folder)
                if 'SUPPORTIVE_MEDS' in f and f.endswith('.pickle')
            ]
            if candidates:
                supportive_meds_path = os.path.join(dataset_folder, sorted(candidates)[-1])

        if supportive_meds_path and os.path.exists(supportive_meds_path):
            print(f"Building gcsf_dict from supportive-meds file: {supportive_meds_path}")
            with open(supportive_meds_path, 'rb') as f:
                supportive_meds = pickle.load(f)
            gcsf_dict = {}
            for dmp_id, date_to_meds in supportive_meds.items():
                gcsf_dates = []
                for date, meds_list in date_to_meds.items():
                    for med_entry in meds_list:
                        if med_entry[0] == 'GCSF':
                            gcsf_dates.append(date)
                            break
                if gcsf_dates:
                    gcsf_dict[dmp_id] = sorted(gcsf_dates)
            data_dict_lab_dict_train['gcsf_dict'] = gcsf_dict
            print(f"  Built gcsf_dict: {len(gcsf_dict)} patients with G-CSF dates")
        else:
            # an empty gcsf_dict would make everyone unexposed, so fail instead
            raise SystemExit(
                f"\nERROR: no supportive-meds pickle (*SUPPORTIVE_MEDS*.pickle) "
                f"found for {dataset} in {dataset_folder}.\n"
                f"Without it no patient has any G-CSF exposure and this "
                f"analysis is meaningless.\n"
                f"Check -vp/--volume_path points at the SPARC_data root that "
                f"contains {dataset}/."
            )
    else:
        print("WARNING: gcsf_dict missing after load_dataset for MSK. Using empty dict.")
        data_dict_lab_dict_train['gcsf_dict'] = {}

print(f"Train set size: {len(split_dict_full['train'])}")
print(f"Val set size: {len(split_dict_full['val'])}")
print(f"Test set size: {len(split_dict_full['test'])}")

pl.seed_everything(rseed)

# train split: fits the propensity model
data_module_train = FlexibleFeatureDataModule(
    data_dict_lab_dict_train, split_dict_full,
    dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=max_time_len_data, num_workers=1,
    primary_ctype_returned=False,
    task=task_option, binary_days=days,
    med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model=advanced_model, control_lack_confounds=False,
    side_effect_censoring=False,
    include_infusion_dates=add_num_infusion,
    premed_only=False,
    competing_risk_death_check=False,
    include_gcsf=True,
    gcsf_lim=gcsf_lim,
    anc=is_caia,
)

# test split: SPARC predictions
split_dict_test = {'train': [], 'val': [], 'test': split_dict_full['test']}
data_module_test = FlexibleFeatureDataModule(
    data_dict_lab_dict_train, split_dict_test,
    dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=max_time_len_data, num_workers=1,
    primary_ctype_returned=False,
    task=task_option, binary_days=days,
    med=med_of_choice, med_as_feature=False, med_l=med_of_choice,
    advanced_model=advanced_model, control_lack_confounds=False,
    side_effect_censoring=False,
    include_infusion_dates=add_num_infusion,
    premed_only=False,
    competing_risk_death_check=False,
    include_gcsf=True,
    gcsf_lim=gcsf_lim,
    anc=is_caia,
)
print("Data modules initiated")

train_dataset = data_module_train.train_dataset
test_dataset = data_module_test.test_dataset

train_DMP_IDs = list(train_dataset.dataset_DMP_IDs)
train_cutoffdates = list(train_dataset.cutoffdates)
test_DMP_IDs = list(test_dataset.dataset_DMP_IDs)
test_cutoffdates = list(test_dataset.cutoffdates)

print(f"Train samples: {len(train_DMP_IDs)}")
print(f"Test samples: {len(test_DMP_IDs)}")
print(f"Model path: {model_base_path}")
print(f"task_option={task_option}, model_set_option={model_set_option}, temporal_aggregation={temporal_aggregation}")

# key cutoffs/outputs by -ns; quintiles keep the unsuffixed names
strata_tag = '' if num_strata == 5 else f'_{num_strata}strata'

# PSM only on MSK_CAIA prior; the unmatched and IPTW analyses (all datasets) are in ..._101326_iptw.py
if not (dataset == 'MSK_CAIA' and caia_portion == 'prior'):
    raise SystemExit(f"PSM is run only on MSK_CAIA with -cp prior (got {dataset}, {caia_portion})")

print("\n" + "="*80)
print("SPARC RISK-STRATIFIED ANALYSIS OF G-CSF IMPACT")
print("="*80 + "\n")
print("Step 1: Loading SPARC model predictions across all folds...")

patient_predictions = {}

for val_fold in range(10):
    pl.seed_everything(rseed)

    model_folder = f'{model_base_path}/fold{val_fold}'
    print(f"\nProcessing fold {val_fold}: {model_folder}")
    epoch_ckpts = [el for el in os.listdir(model_folder) if 'epoch' in el]
    optimal_model_path = os.path.join(model_folder, epoch_ckpts[0])

    module_index_cfg = build_module_index_config(
        modular_featuredim_dict, advanced_model='default'
    )
    n_static_features = module_index_cfg['n_static']

    # FN/N/A models were trained with the hybrid (Cox + BCE) loss
    if option in ['febrile_neutropenia', 'neutropenia', 'anemia']:
        loss_option = 'hybrid'
    else:
        loss_option = 'default'

    load_kwargs = dict(
        modular_featuredim_dict=modular_featuredim_dict,
        option=model_set_option,
        advanced_model='default',
        max_time_len=max_time_len_data,
        med_as_feature=False,
        meds_l=med_of_choice,
        loss_option=loss_option,
        side_effects_num_infusion=add_num_infusion,
        include_gcsf=True,
        use_gcsf=False,
        primary_ctype_returned=False,
        use_sigmoid=False,
        temporal_aggregation=temporal_aggregation,
        module_index_cfg=module_index_cfg,
        use_module_tokenizer=True,
        use_continuous_time=True,
        n_static_features=n_static_features,
        post_mlp_dropout=0.2,
        batch_first=(args.batch_first == 'True'),
    )
    model = SPARC.load_from_checkpoint(optimal_model_path, **load_kwargs)

    pl.seed_everything(rseed)
    basic_feature_trainer = pl.Trainer(max_epochs=1, deterministic=True)
    # unshuffled, so prediction row idx is test_dataset[idx] (test_dataloader() shuffles by default)
    data_module_test.test_shuffle = False
    y_pred_true_info = basic_feature_trainer.predict(model, dataloaders=data_module_test.test_dataloader())

    y_pred_logit  = np.concatenate([el['y_pred'] for el in y_pred_true_info], axis=0).squeeze()
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred_logit))

    p_final = y_pred_sigmoid

    _yt = [el['y_true'] for el in y_pred_true_info]
    if isinstance(_yt[0], (tuple, list)) and not isinstance(_yt[0], np.ndarray):
        y_true = np.concatenate([
            np.array(el[0].cpu() if hasattr(el[0], 'cpu') else el[0])
            for el in _yt
        ], axis=0)
    else:
        y_true = np.concatenate(_yt, axis=0)

    gcsf_flag = np.concatenate([
        np.array(el['gcsf_groundtruth'].cpu() if hasattr(el['gcsf_groundtruth'], 'cpu') else el['gcsf_groundtruth'])
        for el in y_pred_true_info
    ], axis=0)

    # prediction rows must follow test_dataset order
    assert len(gcsf_flag) == len(test_dataset), \
        f"{len(gcsf_flag)} predictions for {len(test_dataset)} test samples"
    assert np.array_equal(np.asarray(gcsf_flag).astype(bool).ravel(),
                          np.asarray(test_dataset.gcsf_groundtruth).astype(bool).ravel()), \
        "prediction rows are not in test_dataset order (G-CSF flags differ)"

    print(f"fold {val_fold}: p_final min/max {p_final.min():.4f}/{p_final.max():.4f}, "
          f"raw sigmoid min/max {y_pred_sigmoid.min():.4f}/{y_pred_sigmoid.max():.4f}")

    for idx in range(len(test_dataset)):
        if idx not in patient_predictions:
            patient_predictions[idx] = {
                'raw_predictions': [],
                'predictions':     [],
                'cutoffdate':      [],
                'outcomes':        [],
                'gcsf_flags':      [],
                'DMP_ID':          test_DMP_IDs[idx],
            }
        patient_predictions[idx]['raw_predictions'].append(float(y_pred_sigmoid[idx]))
        patient_predictions[idx]['predictions'].append(float(p_final[idx]))
        patient_predictions[idx]['cutoffdate'].append(test_cutoffdates[idx])
        patient_predictions[idx]['outcomes'].append(y_true[idx])
        patient_predictions[idx]['gcsf_flags'].append(gcsf_flag[idx])

print(f"\nTotal unique patient-prediction pairs collected: {len(patient_predictions)}")

# Step 2: average fold predictions per sample
print("\nStep 2: Aggregating SPARC predictions by patient...")

patient_data = []
for i, pred_dict in patient_predictions.items():
    patient_data.append({
        'sample_idx':                 i,   # test_dataset index, used to attach the propensity score
        'infusion_cycle':             int(test_dataset.num_infusions[i]),   # infusion index within the regimen
        'DMP_ID':                     pred_dict['DMP_ID'],
        'cutoffdate':                 pred_dict['cutoffdate'][0],
        'raw_sparc_risk_score':       np.mean(pred_dict['raw_predictions']),
        'sparc_risk_score':           np.mean(pred_dict['predictions']),
        'sparc_risk_score_per_fold':  pred_dict['predictions'],
        'gcsf':                       pred_dict['gcsf_flags'][0],
        'outcome':                    pred_dict['outcomes'][0],
    })

df_patients_test = pd.DataFrame(patient_data)
df_patients_test = df_patients_test.sort_values(by=['DMP_ID', 'cutoffdate']).reset_index(drop=True)

# Step 3: propensity model on raw input features (train set)
print("\n" + "="*80)
print("STEP 3: EXTRACTING RAW FEATURES AND BUILDING PROPENSITY SCORE MODEL ON TRAIN SET")
print("="*80 + "\n")

print("Extracting raw input features from train and test datasets...")

def extract_features_from_dataset(dataset):
    """Flattened per-sample features for the propensity model, as in the temporal_xgboost benchmark.

    Returns features (n_samples, n_features), dmp_ids, cutoffdates, gcsf_flags, outcomes.
    """
    features_list = []
    dmp_ids = []
    cutoffdates = []
    gcsf_flags = []
    outcomes = []

    for idx in range(len(dataset)):
        # x = (X_ts, X_dates, [num_infusion], [DMP_ID], [gcsf_groundtruth])
        data_tuple = dataset[idx]
        x_tuple = data_tuple[0]

        xs_tp = x_tuple[0].detach().clone()  # (time_steps, features)
        time_tp = x_tuple[1].detach().clone()  # (time_steps,)

        mask = torch.isnan(xs_tp).detach().clone()

        xs_tp[mask] = 0

        time_nan_mask = torch.isnan(time_tp).int().unsqueeze(-1).detach().clone()
        time_tp_combine = time_tp.unsqueeze(-1)
        time_tp_combine[time_nan_mask.bool()] = 0.0

        # flatten [values, nan mask, time, time nan mask]
        xs_tp = torch.cat((xs_tp, mask, time_tp_combine, time_nan_mask), dim=1).view(-1)

        if add_num_infusion and len(x_tuple) >= 3:
            xs_tp_infusions = x_tuple[2]
            if not isinstance(xs_tp_infusions, torch.Tensor):
                xs_tp_infusions = torch.tensor(xs_tp_infusions, dtype=torch.float32)
            xs_tp_infusions = xs_tp_infusions.unsqueeze(-1)
            xs_tp = torch.cat((xs_tp, xs_tp_infusions))

        feature_vec = xs_tp.numpy()

        features_list.append(feature_vec)
        dmp_ids.append(dataset.dataset_DMP_IDs[idx])
        cutoffdates.append(dataset.cutoffdates[idx])
        gcsf_flags.append(dataset.gcsf_groundtruth[idx])

        y = data_tuple[1]
        outcomes.append(y)

    features = np.array(features_list)
    gcsf_flags = np.array(gcsf_flags)

    # risk tasks give (event, time) tuples
    if isinstance(outcomes[0], (tuple, list)):
        outcomes = np.array([o[0] for o in outcomes])
    else:
        outcomes = np.array(outcomes)

    return features, dmp_ids, cutoffdates, gcsf_flags, outcomes

print("Extracting features from train set...")
X_train_raw, train_dmp_ids_raw, train_cutoffdates_raw, y_train_gcsf, y_train_outcome = extract_features_from_dataset(train_dataset)

print("Extracting features from test set...")
X_test_raw, test_dmp_ids_raw, test_cutoffdates_raw, y_test_gcsf, y_test_outcome = extract_features_from_dataset(test_dataset)

print(f"\nTrain set: {len(X_train_raw)} samples, {X_train_raw.shape[1]} features")
print(f"Test set: {len(X_test_raw)} samples, {X_test_raw.shape[1]} features")
print(f"Train set G-CSF: {y_train_gcsf.sum()} / {len(y_train_gcsf)} ({y_train_gcsf.mean()*100:.1f}%)")
print(f"Test set G-CSF: {y_test_gcsf.sum()} / {len(y_test_gcsf)} ({y_test_gcsf.mean()*100:.1f}%)")

# standardize: features are on very different scales and lbfgs does not converge otherwise
print("\nBuilding logistic regression model to predict G-CSF usage from raw patient features...")
ps_scaler = StandardScaler()
X_train_scaled = ps_scaler.fit_transform(X_train_raw)
X_test_scaled = ps_scaler.transform(X_test_raw)

ps_model = LogisticRegression(random_state=rseed, max_iter=1000, solver='lbfgs')
ps_model.fit(X_train_scaled, y_train_gcsf)

print(f"Propensity score model trained with {X_train_raw.shape[1]} features")
print(f"Number of non-zero coefficients: {np.sum(ps_model.coef_ != 0)}")

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
y_test_gcsf_pred_proba = ps_model.predict_proba(X_test_scaled)[:, 1]
y_train_gcsf_pred_proba = ps_model.predict_proba(X_train_scaled)[:, 1]

train_ps_auroc = roc_auc_score(y_train_gcsf, y_train_gcsf_pred_proba)
train_ps_auprc = average_precision_score(y_train_gcsf, y_train_gcsf_pred_proba)
test_ps_auroc = roc_auc_score(y_test_gcsf, y_test_gcsf_pred_proba)
test_ps_auprc = average_precision_score(y_test_gcsf, y_test_gcsf_pred_proba)

print(f"\nPropensity Score Model Performance:")
print(f"  Train set - AUROC: {train_ps_auroc:.4f}, AUPRC: {train_ps_auprc:.4f}")
print(f"  Test set  - AUROC: {test_ps_auroc:.4f}, AUPRC: {test_ps_auprc:.4f}")

# attach by dataset index: the table is sorted by DMP_ID, cutoffdate, the scores are in dataset order
_sidx = df_patients_test['sample_idx'].values
assert len(y_test_gcsf_pred_proba) == len(test_dataset) and len(np.unique(_sidx)) == len(_sidx)
assert all(str(a) == str(b) for a, b in zip(df_patients_test['DMP_ID'].values,
                                            np.asarray(test_dmp_ids_raw, dtype=object)[_sidx])), \
    "DMP_IDs differ between the prediction table and the propensity features"
assert np.array_equal(df_patients_test['gcsf'].values.astype(bool),
                      np.asarray(y_test_gcsf).astype(bool).ravel()[_sidx]), \
    "G-CSF flags differ between the prediction table and the propensity features"
df_patients_test['propensity_score'] = y_test_gcsf_pred_proba[_sidx]
_stored_auroc = roc_auc_score(df_patients_test['gcsf'].astype(int), df_patients_test['propensity_score'])
print(f"  Test AUROC of the stored propensity_score column: {_stored_auroc:.4f} (must equal the test AUROC above)")
assert abs(_stored_auroc - test_ps_auroc) < 1e-9, "stored propensity scores are misaligned"

print(f"\nPropensity score range in test set: [{df_patients_test['propensity_score'].min():.4f}, {df_patients_test['propensity_score'].max():.4f}]")
print(f"Mean propensity score: {df_patients_test['propensity_score'].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df_patients_test[df_patients_test['gcsf']==0]['propensity_score'],
         bins=30, alpha=0.6, label='No G-CSF', color='#e74c3c', edgecolor='black')
ax1.hist(df_patients_test[df_patients_test['gcsf']==1]['propensity_score'],
         bins=30, alpha=0.6, label='G-CSF', color='#2ecc71', edgecolor='black')
ax1.set_xlabel('Propensity Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Propensity Score Distribution by G-CSF Status (Test Set)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_test_gcsf, y_test_gcsf_pred_proba)
ax2.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'Test AUROC = {test_ps_auroc:.3f}')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('Propensity Score Model ROC Curve', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag
plt.savefig(os.path.join(output_dir, f'propensity_score_model_performance_{dataset}{tag}.pdf'), dpi=300)
print(f"Saved propensity score model performance figure to {output_dir}")


# Step 4: exact on SPARC tertile x infusion cycle, 1:k with caliper, patient-level bootstrap
print("\n" + "="*80)
print("STEP 4: PSM EXACT ON SPARC TERTILE AND INFUSION CYCLE (1:k, caliper, patient-level bootstrap)")
print("="*80)
from psm_cycle_tertile_matched import run_cycle_tertile_psm
if num_strata == 3:
    run_cycle_tertile_psm(df_patients_test, output_dir, dataset=dataset, ks=args.match_ratios,
                          caliper_sd=args.caliper_sd)
else:
    print("skipped: run_cycle_tertile_psm is written for tertiles (-ns 3)")

print("\nDone.")