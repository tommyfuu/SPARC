#!/usr/bin/env python3
"""G-CSF effect on neutropenia / febrile neutropenia by SPARC predicted-risk stratum: unmatched and IPTW.

IPTW per stratum: overlap (ATO) weights on propensity scores recalibrated on the test set.
CIs/p from a patient-level cluster bootstrap. Predictions use an unshuffled loader and PS is attached by dataset index.
Usage: python3 risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_101326_iptw.py -o neutropenia -ds MSK_CAIA -ns 3 -gl 2
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
import matplotlib.ticker as mticker
import pytorch_lightning as pl
from scipy.stats import fisher_exact
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
parser.add_argument("-ie",  "--iptw_estimands",    type=str,   default='ATO,ATE',
                    help="Comma-separated IPTW estimands for Step 5: ATO (overlap weights), ATE (stabilized), "
                         "ATT. Empty string skips Step 5. Default: ATO,ATE")
parser.add_argument("-it",  "--iptw_trim",         type=float, default=0.01,
                    help="Truncate ATE/ATT weights at the [trim, 1-trim] quantiles (0 = none; not applied to ATO). "
                         "Default 0.01")
parser.add_argument("-ib",  "--iptw_boot",         type=int,   default=2000,
                    help="Patient-level bootstrap resamples for the IPTW CIs and p-values. Default 2000")
parser.add_argument("-ir",  "--iptw_recalibrate",  type=str,   default='True',
                    help="True | False -- recalibrate the propensity scores on the test set (logistic regression of "
                         "G-CSF on logit PS) before weighting. Matching (Step 4) always uses the original scores.")
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
# default output dir only when -od is not given
if not args.output_dir:
    if dataset == 'MSK_CAIA':
        output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}')
    else:
        output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}_{dataset}')
    if args.batch_first == 'True':
        output_dir += '_bfTrue'
    output_dir += '_iptw'
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

batch_size        = 512
max_time_len_data = 20
advanced_model    = 'default'
valfold           = 0   # placeholder; all 10 folds are used below

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

# risk cutoffs are set on MSK_CAIA prior and reused for the other datasets
cutoffs_reference_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}_iptw')
os.makedirs(cutoffs_reference_dir, exist_ok=True)

# key cutoffs/outputs by -ns; quintiles keep the unsuffixed names
strata_tag = '' if num_strata == 5 else f'_{num_strata}strata'

cutoffs_filepath = os.path.join(
    cutoffs_reference_dir, f'SPARC_risk_score_cutoffs_{option}{strata_tag}.pickle'
)

should_save_cutoffs = (dataset == 'MSK_CAIA' and caia_portion == 'prior')
should_load_cutoffs = (dataset != 'MSK_CAIA')

reference_cutoffs = None
if should_load_cutoffs:
    if os.path.exists(cutoffs_filepath):
        with open(cutoffs_filepath, 'rb') as f:
            reference_cutoffs = pickle.load(f)
        # cutoffs = num_strata-1 percentile edges plus the max
        if len(reference_cutoffs) != num_strata:
            raise SystemExit(
                f"\nERROR: reference cutoffs at {cutoffs_filepath} contain "
                f"{len(reference_cutoffs)} entries, which describes "
                f"{len(reference_cutoffs)} strata, but -ns/--num_strata is "
                f"{num_strata}.\nRegenerate the reference for this stratum "
                f"count by running the MSK_CAIA prior-portion analysis with "
                f"-ns {num_strata} first."
            )
        print(f"\nLoaded reference risk score cutoffs from {cutoffs_filepath}")
        print(f"Reference cutoffs: {reference_cutoffs}")
    else:
        print(f"\nWARNING: Could not find reference cutoffs at {cutoffs_filepath}")
        print(f"Will compute cutoffs from current dataset instead.")

# propensity model and IPTW only on MSK_CAIA prior; temporal validation sets get the unmatched analysis
run_psm = (dataset == 'MSK_CAIA' and caia_portion == 'prior')

if not run_psm:
    print("\n" + "="*80)
    print("PROPENSITY / IPTW ANALYSIS WILL BE SKIPPED")
    if dataset != 'MSK_CAIA':
        print(f"Reason: {dataset} is a temporal validation dataset")
        print("IPTW is only performed on the main MSK_CAIA dataset")
    else:
        print(f"Reason: caia_portion='{caia_portion}' (IPTW only for 'prior' portion)")
    print("="*80 + "\n")
else:
    print("\n" + "="*80)
    print("PROPENSITY / IPTW ANALYSIS WILL BE PERFORMED")
    print("="*80 + "\n")

def _bootstrap_arr_nnt_ci(gcsf_outcomes, no_gcsf_outcomes, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap CI for ARR (no-G-CSF rate minus G-CSF rate) and NNT.

    Groups are resampled independently (not a paired bootstrap). NNT CI is NaN
    unless the ARR CI lies above zero.
    """
    rng = np.random.default_rng(seed)
    gcsf_outcomes = np.asarray(gcsf_outcomes)
    no_gcsf_outcomes = np.asarray(no_gcsf_outcomes)
    n_g, n_ng = len(gcsf_outcomes), len(no_gcsf_outcomes)

    arr_boot = np.empty(n_boot)
    for b in range(n_boot):
        gr_b  = rng.choice(gcsf_outcomes,    size=n_g,  replace=True).mean()
        ngr_b = rng.choice(no_gcsf_outcomes, size=n_ng, replace=True).mean()
        arr_boot[b] = ngr_b - gr_b

    arr_lo, arr_hi = np.percentile(arr_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if arr_lo > 0:
        nnt_boot = np.where(arr_boot > 0, 1.0 / arr_boot, np.nan)
        nnt_lo, nnt_hi = np.nanpercentile(nnt_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        nnt_lo = nnt_hi = np.nan
    return arr_lo, arr_hi, nnt_lo, nnt_hi


def run_gcsf_analysis(df_patients, output_suffix, output_dir, option, dataset, score_label, gcsf_lim,
                      reference_cutoffs=None, save_cutoffs=False, cutoffs_filepath=None, num_strata=5):
    """Stratified G-CSF analysis and figures for one patient table; returns per-stratum results.

    df_patients needs DMP_ID, cutoffdate, sparc_risk_score, gcsf, outcome.
    reference_cutoffs: MSK_CAIA cutoffs to reuse; if None they are computed (and saved if save_cutoffs).
    """
    tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag

    from sklearn.metrics import roc_auc_score
    overall_auroc = roc_auc_score(df_patients['outcome'], df_patients['sparc_risk_score'])
    print(f"\nOverall AUROC for SPARC risk score predicting outcome: {overall_auroc:.4f}")
    print(f"Total patients: {len(df_patients)}")
    print(f"Patients with G-CSF: {df_patients['gcsf'].sum()}")
    print(f"Patients with outcome: {df_patients['outcome'].sum()}")
    print(f"SPARC risk score range: [{df_patients['sparc_risk_score'].min():.4f}, {df_patients['sparc_risk_score'].max():.4f}]")

    print(f"\nSPARC risk score percentiles{output_suffix}:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(df_patients['sparc_risk_score'], p):.4f}")

    print("\n" + "="*80)
    print(f"SPARC RISK-STRATIFIED G-CSF IMPACT ANALYSIS ({num_strata} STRATA){output_suffix}")
    print("="*80 + "\n")

    if reference_cutoffs is not None:
        SPARC_risk_score_cut_offs = reference_cutoffs
        print(f"Using reference SPARC risk score cutoffs from MSK_CAIA: {SPARC_risk_score_cut_offs}")
    else:
        percentiles = [100 * (i + 1) / num_strata for i in range(num_strata - 1)]
        SPARC_risk_score_cut_offs = [
            np.percentile(df_patients['sparc_risk_score'], p) for p in percentiles
        ] + [df_patients['sparc_risk_score'].max()]
        print(f"Using SPARC risk score cutoffs computed from current dataset: {SPARC_risk_score_cut_offs}")

        if save_cutoffs and cutoffs_filepath is not None:
            os.makedirs(os.path.dirname(cutoffs_filepath), exist_ok=True)
            with open(cutoffs_filepath, 'wb') as f:
                pickle.dump(SPARC_risk_score_cut_offs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved SPARC risk score cutoffs to {cutoffs_filepath}")

    def assign_risk_stratum(score, cutoffs, num_strata):
        for i in range(len(cutoffs) - 1):
            if score < cutoffs[i]:
                if i == 0:
                    return f'Stratum 1 (<{cutoffs[i]*100:.1f}%)'
                else:
                    return f'Stratum {i+1} ({cutoffs[i-1]*100:.1f}-{cutoffs[i]*100:.1f}%)'
        return f'Stratum {num_strata} (≥{cutoffs[-2]*100:.1f}%)'

    df_patients['risk_stratum'] = df_patients['sparc_risk_score'].apply(
        lambda x: assign_risk_stratum(x, SPARC_risk_score_cut_offs, num_strata)
    )

    stratum_order = []
    stratum_order.append(f'Stratum 1 (<{SPARC_risk_score_cut_offs[0]*100:.1f}%)')
    for i in range(1, num_strata - 1):
        stratum_order.append(
            f'Stratum {i+1} ({SPARC_risk_score_cut_offs[i-1]*100:.1f}-{SPARC_risk_score_cut_offs[i]*100:.1f}%)'
        )
    stratum_order.append(f'Stratum {num_strata} (≥{SPARC_risk_score_cut_offs[-2]*100:.1f}%)')

    print("\nPatient distribution by SPARC risk stratum:")
    print("-"*80)
    for stratum in stratum_order:
        stratum_df = df_patients[df_patients['risk_stratum'] == stratum]
        n_total   = len(stratum_df)
        n_g       = stratum_df['gcsf'].sum()
        rate      = stratum_df['outcome'].sum() / n_total if n_total > 0 else 0
        avg_risk  = stratum_df['sparc_risk_score'].mean()
        print(f"\n{stratum}:")
        print(f"  Total: n={n_total} (Observed outcome rate: {rate:.3f})")
        print(f"  G-CSF: n={n_g}, No G-CSF: n={n_total - n_g}")
        print(f"  Avg SPARC risk score: {avg_risk:.4f}")
        print(f"  SPARC score range: [{stratum_df['sparc_risk_score'].min():.4f}, {stratum_df['sparc_risk_score'].max():.4f}]")

    print("\n" + "="*80)
    print("G-CSF IMPACT ANALYSIS BY RISK STRATUM")
    print("="*80 + "\n")

    stratum_results = []
    for stratum in stratum_order:
        stratum_df    = df_patients[df_patients['risk_stratum'] == stratum]
        if len(stratum_df) == 0:
            print(f"\n{stratum}: No patients in this stratum"); continue

        gcsf_group    = stratum_df[stratum_df['gcsf'] == 1]
        no_gcsf_group = stratum_df[stratum_df['gcsf'] == 0]
        if len(gcsf_group) == 0 and len(no_gcsf_group) == 0:
            print(f"\n{stratum}: No patients in this stratum"); continue

        n_g  = len(gcsf_group); n_ng = len(no_gcsf_group)

        gr  = gcsf_group['outcome'].mean()    if n_g  > 0 else np.nan
        ngr = no_gcsf_group['outcome'].mean() if n_ng > 0 else np.nan

        gcsf_with    = int((gcsf_group['outcome']    == 1).sum()) if n_g  > 0 else 0
        no_gcsf_with = int((no_gcsf_group['outcome'] == 1).sum()) if n_ng > 0 else 0

        if n_g > 0 and n_ng > 0:
            arr        = ngr - gr
            rrr        = arr / ngr if ngr > 0 else 0
            rr         = (gr / ngr) if ngr > 0 else np.nan
            odds_ratio, p_val = fisher_exact([[gcsf_with,    n_g  - gcsf_with],
                                              [no_gcsf_with, n_ng - no_gcsf_with]])
            nnt = 1 / arr if arr > 0 else np.inf

            arr_lower, arr_upper, nnt_ci_lower, nnt_ci_upper = _bootstrap_arr_nnt_ci(
                gcsf_group['outcome'].to_numpy(), no_gcsf_group['outcome'].to_numpy()
            )
        else:
            arr = rrr = rr = odds_ratio = p_val = nnt = np.nan
            arr_lower = arr_upper = nnt_ci_lower = nnt_ci_upper = np.nan

        stratum_results.append({
            'stratum':                  stratum,
            'n_gcsf':                   n_g,
            'n_no_gcsf':                n_ng,
            'gcsf_outcome_rate':        gr,
            'no_gcsf_outcome_rate':     ngr,
            'absolute_risk_reduction':  arr,
            'arr_ci_lower':             arr_lower,
            'arr_ci_upper':             arr_upper,
            'relative_risk_reduction':  rrr,
            'relative_risk':            rr,
            'odds_ratio':               odds_ratio,
            'p_value':                  p_val,
            'nnt':                      nnt,
            'nnt_ci_lower':             nnt_ci_lower,
            'nnt_ci_upper':             nnt_ci_upper,
            'avg_sparc_risk_gcsf':      gcsf_group['sparc_risk_score'].mean() if n_g  > 0 else np.nan,
            'avg_sparc_risk_no_gcsf':   no_gcsf_group['sparc_risk_score'].mean() if n_ng > 0 else np.nan,
        })

        print(f"\n{'='*60}\n{stratum}\n{'='*60}")
        if n_g > 0:
            print(f"G-CSF:    n={n_g}, outcome rate={gr:.3f}, "
                  f"avg SPARC risk={gcsf_group['sparc_risk_score'].mean():.4f}")
        else:
            print(f"G-CSF:    n=0 (no data)")
        if n_ng > 0:
            print(f"No G-CSF: n={n_ng}, outcome rate={ngr:.3f}, "
                  f"avg SPARC risk={no_gcsf_group['sparc_risk_score'].mean():.4f}")
        else:
            print(f"No G-CSF: n=0 (no data)")
        if n_g > 0 and n_ng > 0:
            print(f"ARR={arr:.3f} ({arr*100:.1f}%) [95% bootstrap CI: {arr_lower:.3f} to {arr_upper:.3f}], "
                  f"RRR={rrr:.3f}, RR={rr:.3f}, OR={odds_ratio:.3f}, p={p_val:.4f}")
            if nnt != np.inf and nnt > 0:
                if not (np.isnan(nnt_ci_lower) or np.isnan(nnt_ci_upper)):
                    print(f"NNT: {nnt:.1f} (95% bootstrap CI: {nnt_ci_lower:.1f} to {nnt_ci_upper:.1f})")
                else:
                    print(f"NNT: {nnt:.1f} (95% CI not estimable — ARR CI crosses 0)")
        else:
            print("(Cross-group statistics unavailable — only one group present)")

    # Does NNT vary with risk stratum? Tested via a gcsf x stratum_rank logistic
    # interaction, Breslow-Day OR homogeneity, and Spearman of rank vs NNT.
    print("\n" + "="*80)
    print(f"ASSOCIATION: SPARC RISK STRATUM vs NNT WITH G-CSF{output_suffix}")
    print("="*80 + "\n")

    # ordinal rank, 0 = lowest risk
    stratum_rank_map = {s: i for i, s in enumerate(stratum_order)}
    df_assoc = df_patients[df_patients['risk_stratum'].isin(stratum_rank_map)].copy()
    df_assoc['stratum_rank'] = df_assoc['risk_stratum'].map(stratum_rank_map).astype(float)

    interaction_p = np.nan
    interaction_coef = np.nan
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        # NaNs here make statsmodels return NaN coefficients silently
        df_fit = df_assoc[['outcome', 'gcsf', 'stratum_rank']].copy()
        df_fit['outcome'] = pd.to_numeric(df_fit['outcome'], errors='coerce')
        df_fit['gcsf'] = pd.to_numeric(df_fit['gcsf'], errors='coerce')
        df_fit['stratum_rank'] = pd.to_numeric(df_fit['stratum_rank'], errors='coerce')
        n_before = len(df_fit)
        df_fit = df_fit.dropna()
        if len(df_fit) < n_before:
            print(f"  Dropped {n_before - len(df_fit)} rows with missing "
                  f"outcome/gcsf/stratum_rank before fitting.")
        # centered so the gcsf main effect is at the middle stratum
        df_fit['stratum_rank_c'] = df_fit['stratum_rank'] - df_fit['stratum_rank'].mean()

        if df_fit['gcsf'].nunique() < 2 or df_fit['outcome'].nunique() < 2:
            print("  Interaction model skipped: gcsf or outcome has no variation.")
        else:
            model_int = smf.logit(
                'outcome ~ gcsf * stratum_rank_c', data=df_fit
            ).fit(disp=False)
            interaction_coef = model_int.params.get('gcsf:stratum_rank_c', np.nan)
            interaction_p = model_int.pvalues.get('gcsf:stratum_rank_c', np.nan)
            if np.isnan(interaction_coef) or np.isnan(interaction_p):
                print("  Interaction model returned NaN (likely separation or "
                      "collinearity); see Breslow-Day / Spearman results below.")
            print("Logistic model: outcome ~ gcsf * stratum_rank (rank centered)")
            print(f"  G-CSF x stratum-rank interaction coefficient: {interaction_coef:.4f}")
            print(f"  Interaction p-value (trend in G-CSF effect across strata): {interaction_p:.4f}")
            if not np.isnan(interaction_p):
                if interaction_p < 0.05:
                    direction = ("larger G-CSF benefit / smaller NNT at higher risk"
                                 if interaction_coef < 0 else
                                 "smaller G-CSF benefit / larger NNT at higher risk")
                    print(f"  -> Significant: risk stratum IS associated with NNT ({direction}).")
                else:
                    print("  -> Not significant: no evidence NNT varies linearly across risk strata.")
    except Exception as e:
        print(f"  Could not fit interaction model ({e}); skipping interaction test.")

    # Breslow-Day: OR homogeneity across strata, no linear-trend assumption
    bd_stat = bd_p = np.nan
    try:
        import statsmodels.api as sm
        tables = []
        for stratum in stratum_order:
            sdf = df_patients[df_patients['risk_stratum'] == stratum]
            g = sdf[sdf['gcsf'] == 1]; ng = sdf[sdf['gcsf'] == 0]
            if len(g) == 0 or len(ng) == 0:
                continue
            # 2x2: rows = gcsf/no-gcsf, cols = event/no-event
            tbl = np.array([
                [int((g['outcome'] == 1).sum()),  int((g['outcome'] == 0).sum())],
                [int((ng['outcome'] == 1).sum()), int((ng['outcome'] == 0).sum())],
            ])
            if (tbl.sum(axis=1) > 0).all() and (tbl.sum(axis=0) > 0).all():
                tables.append(tbl)
        if len(tables) >= 2:
            st = sm.stats.StratifiedTable(tables)
            bd_res = st.test_equal_odds()
            bd_stat = float(bd_res.statistic)
            bd_p = float(bd_res.pvalue)
            print(f"\nBreslow-Day homogeneity of OR across strata: "
                  f"chi2={bd_stat:.3f}, p={bd_p:.4f} (n={len(tables)} strata)")
            if bd_p < 0.05:
                print("  -> OR (and thus NNT) differs across risk strata.")
            else:
                print("  -> No significant heterogeneity of the G-CSF effect across strata.")
    except Exception as e:
        print(f"  Breslow-Day test unavailable ({e}).")

    from scipy.stats import spearmanr
    results_df_tmp = pd.DataFrame(stratum_results)
    results_df_tmp['stratum_rank'] = results_df_tmp['stratum'].map(stratum_rank_map)
    nnt_finite = results_df_tmp[
        np.isfinite(results_df_tmp['nnt']) & results_df_tmp['nnt'].notna()
    ]
    spearman_rho = spearman_p = np.nan
    if len(nnt_finite) >= 3:
        spearman_rho, spearman_p = spearmanr(
            nnt_finite['stratum_rank'], nnt_finite['nnt']
        )
        print(f"\nSpearman correlation (stratum rank vs per-stratum NNT): "
              f"rho={spearman_rho:.3f}, p={spearman_p:.4f} (n={len(nnt_finite)} strata)")
    else:
        print("\nSpearman correlation: too few strata with a finite NNT to compute.")

    nnt_table = results_df_tmp[[
        'stratum', 'n_gcsf', 'n_no_gcsf',
        'gcsf_outcome_rate', 'no_gcsf_outcome_rate',
        'absolute_risk_reduction', 'arr_ci_lower', 'arr_ci_upper',
        'nnt', 'nnt_ci_lower', 'nnt_ci_upper', 'p_value',
    ]].copy()
    nnt_csv = f'sparc_nnt_by_risk_stratum_{score_label}_{option}_{dataset}{tag}{output_suffix}.csv'
    nnt_table.to_csv(os.path.join(output_dir, nnt_csv), index=False)
    print(f"\nSaved NNT-by-stratum table to {os.path.join(output_dir, nnt_csv)}")

    if len(stratum_results) > 0:
        results_df = pd.DataFrame(stratum_results)

        # styled to match vis_eval_5_clinical_benchmark_forest_plot_MSK.py
        n_str = len(results_df)
        y_pos = np.arange(n_str, dtype=float)
        # bar thickness, and centre-to-centre spacing of the two arms
        BAR_H, ROW_GAP = 0.32, 0.40
        fig, ax1 = plt.subplots(figsize=(9, max(5.0, 1.15 * n_str + 2.4)))

        strat_labels = [s.replace(' (', '\n(') for s in results_df['stratum']]

        g_rate  = results_df['gcsf_outcome_rate'].fillna(0).to_numpy()
        ng_rate = results_df['no_gcsf_outcome_rate'].fillna(0).to_numpy()

        ax1.barh(y_pos - ROW_GAP/2, g_rate,  height=BAR_H, color='#3498db',
                 edgecolor='black', linewidth=0.9, label='G-CSF', zorder=2)
        ax1.barh(y_pos + ROW_GAP/2, ng_rate, height=BAR_H, color='#e74c3c',
                 edgecolor='black', linewidth=0.9, label='No G-CSF', zorder=2)

        outcome_label = 'Febrile neutropenia' if 'febrile' in option else 'Neutropenia'
        ax1.set_xlabel(f'{outcome_label} rate (%)', fontsize=12)
        ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax1.set_title(f'{outcome_label} rate by SPARC risk stratum',
                      fontsize=13, pad=12, loc='left')
        ax1.set_yticks(y_pos); ax1.set_yticklabels(strat_labels, fontsize=11)
        ax1.tick_params(axis='y', length=0)
        ax1.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        ax1.invert_yaxis()   # stratum 1 (lowest risk) at the top
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2,
                   fontsize=10, frameon=False, handlelength=1.4, columnspacing=2.0)

        rate_max = max(g_rate.max(), ng_rate.max())
        # headroom for the n/p annotation column
        x_hi  = min(max(rate_max * 1.55, rate_max + 0.12), 1.0)
        sig_x = x_hi - 0.01 * (x_hi - 0)
        ax1.set_xlim(0, x_hi)

        for i, (n_g, n_ng, pv) in enumerate(zip(
                results_df['n_gcsf'], results_df['n_no_gcsf'], results_df['p_value'])):
            if pd.isna(pv):
                p_text = 'n/a'
            else:
                p_text = ('p<0.001***' if pv < 0.001 else f'p={pv:.3f}**' if pv < 0.01
                          else f'p={pv:.3f}*' if pv < 0.05 else f'p={pv:.2f}')
            ax1.text(sig_x, y_pos[i], f'n={int(n_g)}/{int(n_ng)}, {p_text}',
                     ha='right', va='center', fontsize=8, fontweight='bold',
                     color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
            f'sparc_gcsf_effect_by_risk_stratum_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'),
            bbox_inches='tight', dpi=300)
        print("Saved outcome rate figure.")

    return stratum_results


def plot_ps_overlap(df_patients, cutoffs, output_dir, dataset, gcsf_lim, num_strata, raw_ps=None):
    """Positivity check: PS overlap, G-CSF vs no G-CSF, overall and per risk stratum; writes PDF and CSV summary.

    propensity_score holds the weighting scores; raw_ps (pre-recalibration), if given, is also summarized in the CSV.
    """
    tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag
    t_all = df_patients['gcsf'].astype(bool).values
    s_all = np.full(len(df_patients), num_strata - 1)
    for i in range(len(cutoffs) - 2, -1, -1):
        s_all[df_patients['sparc_risk_score'].values < cutoffs[i]] = i
    labels = ['All infusions'] + [f'Stratum 1 (<{cutoffs[0]*100:.1f}%)'] + \
             [f'Stratum {i+1} ({cutoffs[i-1]*100:.1f}-{cutoffs[i]*100:.1f}%)' for i in range(1, num_strata - 1)] + \
             [f'Stratum {num_strata} (≥{cutoffs[-2]*100:.1f}%)']
    masks = [np.ones(len(df_patients), bool)] + [s_all == k for k in range(num_strata)]

    def summarize(e, score):
        rows = []
        for lab, m in zip(labels, masks):
            et, ec = e[m & t_all], e[m & ~t_all]
            lo, hi = max(et.min(), ec.min()), min(et.max(), ec.max())   # common support
            rows.append({'scores': score, 'group': lab, 'n_gcsf': len(et), 'n_no_gcsf': len(ec),
                         'gcsf_min': et.min(), 'gcsf_p01': np.quantile(et, 0.01), 'gcsf_p99': np.quantile(et, 0.99),
                         'gcsf_max': et.max(), 'no_gcsf_min': ec.min(), 'no_gcsf_p01': np.quantile(ec, 0.01),
                         'no_gcsf_p99': np.quantile(ec, 0.99), 'no_gcsf_max': ec.max(),
                         'pct_gcsf_outside_common_support': 100 * np.mean((et < lo) | (et > hi)),
                         'pct_no_gcsf_outside_common_support': 100 * np.mean((ec < lo) | (ec > hi)),
                         'pct_ps_below_0.05_or_above_0.95': 100 * np.mean((e[m] < 0.05) | (e[m] > 0.95))})
        return rows

    e = df_patients['propensity_score'].values.astype(float)
    rows = summarize(e, 'used for weighting')
    if raw_ps is not None:
        rows = summarize(np.asarray(raw_ps, dtype=float), 'raw (train model)') + rows
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, f'propensity_overlap_{dataset}{tag}.csv'), index=False)

    fig, axes = plt.subplots(1, num_strata + 1, figsize=(4.2 * (num_strata + 1), 3.8), sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, lab, m, r in zip(axes, labels, masks, [x for x in rows if x['scores'] == 'used for weighting']):
        ax.hist(e[m & ~t_all], bins=bins, density=True, alpha=0.55, color='#e74c3c', edgecolor='black',
                linewidth=0.4, label=f'No G-CSF (n={r["n_no_gcsf"]:,})')
        ax.hist(e[m & t_all], bins=bins, density=True, alpha=0.55, color='#3498db', edgecolor='black',
                linewidth=0.4, label=f'G-CSF (n={r["n_gcsf"]:,})')
        ax.set_title(lab, fontsize=11, loc='left')
        ax.set_xlabel('Propensity score' + (' (recalibrated)' if raw_ps is not None else ''), fontsize=10)
        ax.text(0.98, 0.70, f'outside common support:\nG-CSF {r["pct_gcsf_outside_common_support"]:.1f}%, '
                f'no G-CSF {r["pct_no_gcsf_outside_common_support"]:.1f}%\n'
                f'PS <0.05 or >0.95: {r["pct_ps_below_0.05_or_above_0.95"]:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#333333')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Density', fontsize=10)
    fig.suptitle('Propensity-score overlap by G-CSF status (positivity check)', fontsize=13, x=0.01, ha='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'propensity_overlap_{dataset}{tag}.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved propensity-score overlap figure and table to {os.path.join(output_dir, f'propensity_overlap_{dataset}{tag}')}.{{pdf,csv}}")


def run_iptw_analysis(df_patients, cutoffs, output_dir, option, dataset, score_label, gcsf_lim, num_strata,
                      estimand='ATE', trim=0.0, n_boot=2000, seed=1024, output_suffix='_iptw'):
    """IPTW G-CSF effect within each SPARC risk stratum (unmatched-analysis cutoffs).

    Weights (treated, control): ATE stabilized p/e, (1-p)/(1-e); ATT 1, e/(1-e); ATO overlap 1-e, e
    (Li, Morgan & Zaslavsky 2018; bounded in [0, 1]). trim > 0 truncates ATE/ATT weights at [trim, 1-trim] quantiles.
    Rates are Hajek means. CIs and p from a patient-level (DMP_ID) cluster bootstrap; weights are not refit
    per resample since the PS model was fit on train.
    """
    tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag
    d = df_patients[['DMP_ID', 'sparc_risk_score', 'gcsf', 'outcome', 'propensity_score']].copy()
    d['gcsf'] = d['gcsf'].astype(bool)
    d['outcome'] = d['outcome'].astype(float)
    e = np.clip(d['propensity_score'].values.astype(float), 1e-6, 1 - 1e-6)
    t = d['gcsf'].values
    p_t = t.mean()
    if estimand == 'ATT':
        w = np.where(t, 1.0, e / (1 - e))
    elif estimand == 'ATO':
        w = np.where(t, 1 - e, e)
    else:
        w = np.where(t, p_t / e, (1 - p_t) / (1 - e))
    if trim > 0 and estimand != 'ATO':
        lo, hi = np.quantile(w, [trim, 1 - trim])
        w = np.clip(w, lo, hi)
    d['w'] = w

    # strata as in the unmatched analysis
    d['s'] = int(num_strata - 1)
    for i in range(len(cutoffs) - 2, -1, -1):
        d.loc[d['sparc_risk_score'] < cutoffs[i], 's'] = i
    labels = [f'Stratum 1 (<{cutoffs[0]*100:.1f}%)'] + \
             [f'Stratum {i+1} ({cutoffs[i-1]*100:.1f}-{cutoffs[i]*100:.1f}%)' for i in range(1, num_strata - 1)] + \
             [f'Stratum {num_strata} (≥{cutoffs[-2]*100:.1f}%)']

    S_, T_, W_, Y_ = d['s'].values, t, w, d['outcome'].values
    def rates(rows=None):
        """per stratum: (weighted rate G-CSF, weighted rate no G-CSF), Hajek means"""
        s_, t_, w_, y_ = (S_, T_, W_, Y_) if rows is None else (S_[rows], T_[rows], W_[rows], Y_[rows])
        cell = s_ * 2 + (~t_).astype(int)          # 2k = G-CSF, 2k+1 = no G-CSF
        ws = np.bincount(cell, weights=w_, minlength=2 * num_strata)
        wy = np.bincount(cell, weights=w_ * y_, minlength=2 * num_strata)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (wy / ws).reshape(num_strata, 2)

    def wsmd(x, tt, ww):
        """weighted standardized mean difference"""
        m1, m0 = np.average(x[tt], weights=ww[tt]), np.average(x[~tt], weights=ww[~tt])
        v1 = np.average((x[tt] - m1) ** 2, weights=ww[tt]); v0 = np.average((x[~tt] - m0) ** 2, weights=ww[~tt])
        return (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else np.nan

    est = rates()
    arr = est[:, 1] - est[:, 0]

    # patient-level cluster bootstrap
    rng = np.random.default_rng(seed)
    groups = d.groupby('DMP_ID').indices
    pids = np.array(list(groups.keys()), dtype=object)
    idx_lists = [groups[p] for p in pids]
    boot = np.full((n_boot, num_strata), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(pids), len(pids))
        rows = np.concatenate([idx_lists[i] for i in pick])
        r = rates(rows)
        boot[b] = r[:, 1] - r[:, 0]

    print("\n" + "=" * 80)
    print(f"IPTW ANALYSIS ({estimand}{' overlap' if estimand == 'ATO' else ', stabilized'} weights{f', trimmed at {trim:g}/{1-trim:g} quantiles' if trim > 0 and estimand != 'ATO' else ''}; "
          f"patient-level bootstrap, B={n_boot})")
    print("=" * 80)
    print(f"Weights: min {w.min():.3f}, median {np.median(w):.3f}, max {w.max():.3f}; "
          f"{d['DMP_ID'].nunique()} patients, {len(d)} infusions")

    rows_out = []
    lps = np.log(e / (1 - e))
    for k in range(num_strata):
        m = d['s'].values == k
        tt, ww = t[m], w[m]
        ess = [ww[tt].sum() ** 2 / (ww[tt] ** 2).sum(), ww[~tt].sum() ** 2 / (ww[~tt] ** 2).sum()]
        bk = boot[:, k][np.isfinite(boot[:, k])]
        lo, hi = np.percentile(bk, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(bk <= 0), np.mean(bk >= 0)))
        nnt = 1 / arr[k] if arr[k] > 0 else np.nan
        rows_out.append({
            'stratum': labels[k], 'n_gcsf': int(tt.sum()), 'n_no_gcsf': int((~tt).sum()),
            'ess_gcsf': ess[0], 'ess_no_gcsf': ess[1],
            'gcsf_outcome_rate': est[k, 0], 'no_gcsf_outcome_rate': est[k, 1],
            'absolute_risk_reduction': arr[k], 'arr_ci_lower': lo, 'arr_ci_upper': hi,
            'nnt': nnt, 'nnt_ci_lower': 1 / hi if hi > 0 else np.nan, 'nnt_ci_upper': 1 / lo if lo > 0 else np.nan,
            'p_value': p, 'smd_logit_ps_unweighted': wsmd(lps[m], tt, np.ones(m.sum())),
            'smd_logit_ps_weighted': wsmd(lps[m], tt, ww),
            'smd_sparc_weighted': wsmd(d['sparc_risk_score'].values[m], tt, ww),
        })
        r = rows_out[-1]
        print(f"{labels[k]}: G-CSF {r['gcsf_outcome_rate']:.3f} vs no G-CSF {r['no_gcsf_outcome_rate']:.3f}; "
              f"ARR={arr[k]*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}], NNT={nnt:.1f}, p={p:.3g}; "
              f"ESS {ess[0]:.0f}/{ess[1]:.0f}; SMD logit-PS {r['smd_logit_ps_unweighted']:.2f} -> "
              f"{r['smd_logit_ps_weighted']:.2f}")
    res = pd.DataFrame(rows_out)

    # does the effect differ between the highest and lowest stratum?
    dk = boot[:, -1] - boot[:, 0]
    dk = dk[np.isfinite(dk)]
    p_het = min(1.0, 2 * min(np.mean(dk <= 0), np.mean(dk >= 0)))
    print(f"ARR(highest) - ARR(lowest) = {(arr[-1]-arr[0])*100:.1f}% "
          f"[{np.percentile(dk, 2.5)*100:.1f}, {np.percentile(dk, 97.5)*100:.1f}], bootstrap p={p_het:.3g}")
    res['arr_diff_highest_minus_lowest'] = arr[-1] - arr[0]
    res['arr_diff_p'] = p_het

    stem = f'sparc_iptw_{estimand.lower()}_by_risk_stratum_{score_label}_{option}_{dataset}{tag}' + \
           (f'_trim{trim:g}' if trim > 0 and estimand != 'ATO' else '')
    res.to_csv(os.path.join(output_dir, f'{stem}.csv'), index=False)

    # same style as the outcome-rate-by-stratum figure
    y_pos = np.arange(num_strata, dtype=float)
    BAR_H, ROW_GAP = 0.32, 0.40
    fig, ax1 = plt.subplots(figsize=(9, max(5.0, 1.15 * num_strata + 2.4)))
    g_rate, ng_rate = res['gcsf_outcome_rate'].to_numpy(), res['no_gcsf_outcome_rate'].to_numpy()
    ax1.barh(y_pos - ROW_GAP/2, g_rate, height=BAR_H, color='#3498db', edgecolor='black', linewidth=0.9,
             label='G-CSF', zorder=2)
    ax1.barh(y_pos + ROW_GAP/2, ng_rate, height=BAR_H, color='#e74c3c', edgecolor='black', linewidth=0.9,
             label='No G-CSF', zorder=2)
    outcome_label = 'Febrile neutropenia' if 'febrile' in option else 'Neutropenia'
    ax1.set_xlabel(f'{outcome_label} rate, IPTW-weighted (%)', fontsize=12)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    _wlabel = {'ATO': 'overlap-weighted', 'ATE': 'IPTW, ATE weights', 'ATT': 'IPTW, ATT weights'}.get(estimand, estimand)
    if trim > 0 and estimand != 'ATO':
        _wlabel += f', truncated {trim:g}/{1-trim:g}'
    ax1.set_title(f'{outcome_label} rate by SPARC risk stratum ({_wlabel})', fontsize=13, pad=12, loc='left')
    # group sizes at the end of each bar: infusions, and effective sample size after weighting
    for i, r in res.iterrows():
        for yy, rate, n, ess in [(y_pos[i] - ROW_GAP/2, r['gcsf_outcome_rate'], r['n_gcsf'], r['ess_gcsf']),
                                 (y_pos[i] + ROW_GAP/2, r['no_gcsf_outcome_rate'], r['n_no_gcsf'], r['ess_no_gcsf'])]:
            ax1.text(rate + 0.004, yy, f'n={int(n):,} (eff. {ess:,.0f})', ha='left', va='center', fontsize=7.5,
                     color='#333333')
    ax1.set_yticks(y_pos); ax1.set_yticklabels([s.replace(' (', '\n(') for s in res['stratum']], fontsize=11)
    ax1.tick_params(axis='y', length=0)
    ax1.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.invert_yaxis()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=10, frameon=False)
    rate_max = np.nanmax([g_rate.max(), ng_rate.max()])
    x_hi = min(max(rate_max * 1.55, rate_max + 0.12), 1.0)
    ax1.set_xlim(0, x_hi)
    for i, r in res.iterrows():
        pv = r['p_value']
        # bootstrap p is on a 1/B grid; 0 means no resample crossed zero
        p_text = f'p<{2/n_boot:.3g}' if pv < 2 / n_boot else (f'p={pv:.3f}' if pv < 0.1 else f'p={pv:.2f}')
        # NNT = 1/ARR; its CI inverts the ARR CI, open-ended (∞) when the ARR CI includes 0
        _nnt = lambda x: f'{1 / x:.0f}' if x > 0 else '∞'
        nnt_text = (f'NNT {_nnt(r["absolute_risk_reduction"])} [{_nnt(r["arr_ci_upper"])}, {_nnt(r["arr_ci_lower"])}]'
                    if r['absolute_risk_reduction'] > 0 else 'NNT n/a (no benefit)')
        ax1.text(x_hi * 0.99, y_pos[i], f'{nnt_text}, {p_text}',
                 ha='right', va='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stem}.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved IPTW table and figure to {os.path.join(output_dir, stem)}.{{csv,pdf}}")
    return res


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

print("\nStep 2: Aggregating SPARC predictions by patient...")

patient_data = []
for i, pred_dict in patient_predictions.items():
    patient_data.append({
        'sample_idx':                 i,   # test_dataset index, used to attach the propensity score
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

score_label = 'sigmoid'

if run_psm:
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
    

print("\n" + "="*80)
print("STEP 4: RUNNING STANDARD ANALYSIS")
print("="*80)
stratum_results_original = run_gcsf_analysis(
    df_patients_test, '', output_dir, option, dataset, score_label, gcsf_lim,
    reference_cutoffs=reference_cutoffs,
    save_cutoffs=should_save_cutoffs,
    cutoffs_filepath=cutoffs_filepath,
    num_strata=num_strata
)

_estimands = [x.strip().upper() for x in args.iptw_estimands.split(',') if x.strip()]
if run_psm and _estimands:
    print("\n" + "="*80)
    print("STEP 5: RUNNING IPTW ANALYSIS")
    print("="*80)
    df_iptw = df_patients_test.copy()
    if args.iptw_recalibrate == 'True':
        _e = np.clip(df_iptw['propensity_score'].values.astype(float), 1e-6, 1 - 1e-6)
        _l = np.log(_e / (1 - _e)).reshape(-1, 1)
        _cal = LogisticRegression(C=1e6, max_iter=1000).fit(_l, df_iptw['gcsf'].astype(int))
        df_iptw['propensity_score'] = _cal.predict_proba(_l)[:, 1]
        print(f"Recalibrated propensity scores on the test set: intercept {_cal.intercept_[0]:.3f}, "
              f"slope {_cal.coef_[0][0]:.3f} (slope < 1 = overconfident); "
              f"range {df_iptw['propensity_score'].min():.4f}-{df_iptw['propensity_score'].max():.4f}")
    _iptw_cuts = reference_cutoffs if reference_cutoffs is not None else [
        np.percentile(df_patients_test['sparc_risk_score'], 100 * (i + 1) / num_strata)
        for i in range(num_strata - 1)] + [df_patients_test['sparc_risk_score'].max()]
    plot_ps_overlap(df_iptw, _iptw_cuts, output_dir, dataset, gcsf_lim, num_strata,
                    raw_ps=df_patients_test['propensity_score'].values if args.iptw_recalibrate == 'True' else None)
    for _est in _estimands:
        run_iptw_analysis(df_iptw, _iptw_cuts, output_dir, option, dataset, score_label, gcsf_lim, num_strata,
                          estimand=_est, trim=args.iptw_trim, n_boot=args.iptw_boot, seed=rseed)

print("\nDone.")