#!/usr/bin/env python3
"""SPARC G-CSF analysis: does G-CSF reduce neutropenia / febrile neutropenia,
stratified by predicted risk, with propensity-score matching.

Uses the 060726 architecture (modular tokenization, continuous-time embeddings,
post-MLP dropout), loading models from
{model_path}/parsimonious_MSK_CAIA_060726/.

Scores are the raw sigmoid; output goes to out_{option}_raw/.

-gl sets the G-CSF window: -gl 2 means G-CSF must start no more than 2 days
before the prediction date. -ns sets the number of risk strata.

Usage (neutropenia or febrile_neutropenia, -rc True or False):
    python3 risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_062526.py \
        -o neutropenia -ds MSK_CAIA -ns 5 -gl 2 -an True -me "<myelosuppressive list>"

Writes per run:
    sparc_{option}_risk_{score_label}_{dataset}_gcsflim{gl}.csv
    sparc_calibration_plot_{score_label}_{option}_{dataset}_gcsflim{gl}.pdf
    sparc_risk_score_distributions_{score_label}_{option}_{dataset}_gcsflim{gl}.pdf
    sparc_gcsf_effect_by_risk_stratum_{score_label}_{option}_{dataset}_gcsflim{gl}.pdf
    propensity_score_model_performance_{dataset}_gcsflim{gl}.pdf
    SPARC_risk_score_cutoffs_{option}.pickle
"""

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
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.proportion import proportion_confint

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# repo root: interpretation/interp_fix_0622626/gcsf_analysis -> up 3 levels
repodir = os.path.abspath(os.path.join(currentdir, '..', '..', '..'))

sys.path.insert(0, os.path.join(repodir, 'data'))
sys.path.insert(0, os.path.join(repodir, 'models'))

from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset

# Import 060726 SPARC model and feature configuration
from SPARC_inf_risk_modular import SPARC
from MSK_featurename_list import build_module_index_config


# Argument parsing
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

# Hard-coded overrides (COMMENT OUT TO USE COMMAND-LINE ARGUMENTS)
# Uncomment below lines only for debugging/testing with fixed parameters
# dataset          = 'MSK_CAIA'
# option           = 'neutropenia'
# days             = 180
# modules          = 'parsimonious'
# med_of_choice    = ('DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
#                     'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
#                     'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL')
# add_num_infusion = True
# caia_portion     = 'prior'
# gcsf_lim         = 2
# num_strata       = 5

calibration_suffix = 'raw'
if dataset == 'MSK_CAIA':
    output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}')
else:
    output_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}_{dataset}')
os.makedirs(output_dir, exist_ok=True)

competing_risk_death_check = False

# Validate option: must be the base form (without _risk suffix)
base_option = re.sub(r'(_next_risk|_risk)$', '', option)
if base_option not in ('neutropenia', 'febrile_neutropenia'):
    raise ValueError(
        f"Invalid option '{option}'. This script only supports "
        "'neutropenia' or 'febrile_neutropenia' (without _risk suffix)."
    )
option = base_option   # normalise — always work with base option

# Dataset-specific settings
is_caia = 'CAIA' in dataset

task_option         = f'{option}_risk'
model_set_option    = 'side_effects_risk'
temporal_aggregation = 'attention'

# 060726 Model paths
default_model_root  = sparc_paths.SPARC_MODEL_ROOT
model_subfolder     = (f'{modules}_MSK_CAIA_060726/SPARC_finetuned_models/'
                        f'{modules}_finetuned_models_hybrid_{option}_20tp_topchemos_att_modular')

model_root      = args.model_base_path if args.model_base_path else default_model_root
model_base_path = os.path.join(model_root, model_subfolder)

# Parse medication list (supports comma and + separators)
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
valfold           = 0   # placeholder for load_dataset; we iterate over all 10 folds below

# Load dataset via the unified loader IN TRAIN MODE to get train/val/test splits
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

# Load GCSF dict if not already present
if 'gcsf_dict' not in data_dict_lab_dict_train:
    if is_caia:
        # Must follow -vp/--volume_path. This was previously hardcoded to
        # /data/morrisq/fuc/SPARC_data, so on any host with a different mount
        # the supportive-meds pickle was silently not found, every patient got
        # gcsf=0, and the G-CSF comparison below ran on an all-unexposed
        # cohort without failing.
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
            # Do not fall back to an empty gcsf_dict. This script's entire
            # output is a comparison of outcomes with vs. without G-CSF; with
            # no exposure data every patient is un-exposed, and the analysis
            # would complete and write plausible-looking files describing
            # nothing. Fail here instead.
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

# Create data modules for train and test
pl.seed_everything(rseed)

# Train data module (for propensity score model training)
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

# Test data module (for getting predictions)
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

# Pre-extract per-sample DMP_IDs and cutoffdates
train_DMP_IDs = list(train_dataset.dataset_DMP_IDs)
train_cutoffdates = list(train_dataset.cutoffdates)
test_DMP_IDs = list(test_dataset.dataset_DMP_IDs)
test_cutoffdates = list(test_dataset.cutoffdates)

print(f"Train samples: {len(train_DMP_IDs)}")
print(f"Test samples: {len(test_DMP_IDs)}")
print(f"Model path: {model_base_path}")
print(f"task_option={task_option}, model_set_option={model_set_option}, temporal_aggregation={temporal_aggregation}")

# Determine whether to save or load risk score cutoffs
# Path to save/load cutoffs (always relative to the MSK_CAIA reference directory)
cutoffs_reference_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}')

# Cutoffs and outputs are keyed by the number of strata. Without this a
# tertile run would load (or overwrite) the quintile reference cutoffs, and
# its figures would overwrite the quintile ones. Quintiles keep the original
# unsuffixed names so existing 5-strata outputs and references are unchanged.
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
        # A cutoffs list holds num_strata-1 percentile edges plus the max, so
        # its length must equal num_strata. A mismatch means the file was
        # written by a run with a different -ns; using it anyway would stratify
        # into the wrong number of bins and silently drop patients whose
        # stratum label is absent from stratum_order.
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

# Determine whether to run PSM analysis
# PSM is only appropriate for the main MSK_CAIA dataset (prior portion)
# For temporal validation datasets, we skip PSM and only run standard analysis
run_psm = (dataset == 'MSK_CAIA' and caia_portion == 'prior')

if not run_psm:
    print("\n" + "="*80)
    print("PSM ANALYSIS WILL BE SKIPPED")
    if dataset != 'MSK_CAIA':
        print(f"Reason: {dataset} is a temporal validation dataset")
        print("PSM is only performed on the main MSK_CAIA dataset")
    else:
        print(f"Reason: caia_portion='{caia_portion}' (PSM only for 'prior' portion)")
    print("="*80 + "\n")
else:
    print("\n" + "="*80)
    print("PSM ANALYSIS WILL BE PERFORMED")
    print("="*80 + "\n")

# Helper function to run analysis and produce figures
def run_gcsf_analysis(df_patients, output_suffix, output_dir, option, dataset, score_label, gcsf_lim,
                      reference_cutoffs=None, save_cutoffs=False, cutoffs_filepath=None, num_strata=5):
    """
    Run the complete G-CSF analysis on a given patient dataframe.

    Parameters
    ----------
    df_patients : pd.DataFrame
        Patient data with columns: DMP_ID, cutoffdate, raw_sparc_risk_score, sparc_risk_score, gcsf, outcome
    output_suffix : str
        Suffix to append to output filenames (e.g., '_post_psm')
    output_dir : str
        Directory to save output files
    option : str
        Task option (e.g., 'neutropenia')
    dataset : str
        Dataset name (e.g., 'MSK_CAIA')
    score_label : str
        Score label (e.g. 'sigmoid')
    gcsf_lim : int or None
        G-CSF time limit
    reference_cutoffs : list of float or None
        Pre-defined risk score cutoffs from MSK_CAIA. If provided, uses these instead of computing from data.
    save_cutoffs : bool
        Whether to save the computed cutoffs to file (only applies if reference_cutoffs is None)
    cutoffs_filepath : str or None
        Path to save cutoffs file (required if save_cutoffs=True)
    num_strata : int
        Number of risk strata to use for stratified analysis (default: 5)

    Returns
    -------
    stratum_results : list of dict
        Results by risk stratum
    """
    # strata_tag ('' for quintiles, '_3strata' etc. otherwise) keeps runs
    # with different -ns from overwriting each other's outputs.
    tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag

    # Save predictions
    csv_name = f'sparc_{option}_risk_{score_label}_{dataset}{tag}{output_suffix}.csv'
    df_patients.to_csv(os.path.join(output_dir, csv_name))
    print(f"Saved predictions to {os.path.join(output_dir, csv_name)}")

    from sklearn.metrics import roc_auc_score
    overall_auroc = roc_auc_score(df_patients['outcome'], df_patients['sparc_risk_score'])
    print(f"\nOverall AUROC for SPARC risk score predicting outcome: {overall_auroc:.4f}")
    print(f"Total patients: {len(df_patients)}")
    print(f"Patients with G-CSF: {df_patients['gcsf'].sum()}")
    print(f"Patients with outcome: {df_patients['outcome'].sum()}")
    print(f"SPARC risk score range: [{df_patients['sparc_risk_score'].min():.4f}, {df_patients['sparc_risk_score'].max():.4f}]")

    # Risk score distributions
    print("\n" + "="*80)
    print(f"VISUALIZING SPARC RISK SCORE DISTRIBUTIONS{output_suffix}")
    print("="*80 + "\n")

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(df_patients['sparc_risk_score'], p):.4f}")

    df_gcsf    = df_patients[df_patients['gcsf'] == 1]
    df_no_gcsf = df_patients[df_patients['gcsf'] == 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.hist(df_patients['sparc_risk_score'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(df_patients['sparc_risk_score'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df_patients["sparc_risk_score"].median():.4f}')
    ax1.axvline(df_patients['sparc_risk_score'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {df_patients["sparc_risk_score"].mean():.4f}')
    ax1.set_xlabel('SPARC Risk Score', fontsize=12); ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall SPARC Risk Score Distribution', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(df_no_gcsf['sparc_risk_score'], bins=50, alpha=0.6, color='#e74c3c',
             label=f'No G-CSF (n={len(df_no_gcsf)})', edgecolor='black')
    ax2.hist(df_gcsf['sparc_risk_score'],    bins=50, alpha=0.6, color='#2ecc71',
             label=f'G-CSF (n={len(df_gcsf)})', edgecolor='black')
    ax2.set_xlabel('SPARC Risk Score', fontsize=12); ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('SPARC Risk Score Distribution by G-CSF Status', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    ax3 = axes[1, 0]
    if len(df_no_gcsf) > 1:
        df_no_gcsf['sparc_risk_score'].plot(kind='kde', ax=ax3, color='#e74c3c', linewidth=2,
                                             label=f'No G-CSF (n={len(df_no_gcsf)})')
    if len(df_gcsf) > 1:
        df_gcsf['sparc_risk_score'].plot(kind='kde', ax=ax3, color='#2ecc71', linewidth=2,
                                          label=f'G-CSF (n={len(df_gcsf)})')
    ax3.set_xlabel('SPARC Risk Score', fontsize=12); ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('SPARC Risk Score Density by G-CSF Status', fontsize=13, fontweight='bold')
    ax3.legend(); ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    bp = ax4.boxplot([df_no_gcsf['sparc_risk_score'], df_gcsf['sparc_risk_score']],
                     labels=['No G-CSF', 'G-CSF'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#e74c3c'); bp['boxes'][1].set_facecolor('#2ecc71')
    ax4.set_ylabel('SPARC Risk Score', fontsize=12)
    ax4.set_title('SPARC Risk Score Comparison by G-CSF Status', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    if len(df_no_gcsf) > 0 and len(df_gcsf) > 0:
        _, p_val_mw = mannwhitneyu(df_no_gcsf['sparc_risk_score'], df_gcsf['sparc_risk_score'])
        ax4.text(0.5, 0.95, f'Mann-Whitney U test: p={p_val_mw:.4f}',
                 transform=ax4.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
        f'sparc_risk_score_distributions_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'),
        bbox_inches='tight', dpi=300)
    print("Saved distribution figure.")

    # Risk-stratified G-CSF impact analysis
    print("\n" + "="*80)
    print(f"SPARC RISK-STRATIFIED G-CSF IMPACT ANALYSIS ({num_strata} STRATA){output_suffix}")
    print("="*80 + "\n")

    # Use reference cutoffs if provided, otherwise compute from current data
    if reference_cutoffs is not None:
        SPARC_risk_score_cut_offs = reference_cutoffs
        print(f"Using reference SPARC risk score cutoffs from MSK_CAIA: {SPARC_risk_score_cut_offs}")
    else:
        # Compute percentile cutoffs based on num_strata
        percentiles = [100 * (i + 1) / num_strata for i in range(num_strata - 1)]
        SPARC_risk_score_cut_offs = [
            np.percentile(df_patients['sparc_risk_score'], p) for p in percentiles
        ] + [df_patients['sparc_risk_score'].max()]
        print(f"Using SPARC risk score cutoffs computed from current dataset: {SPARC_risk_score_cut_offs}")

        # Save cutoffs if requested (e.g., for MSK_CAIA prior portion)
        if save_cutoffs and cutoffs_filepath is not None:
            os.makedirs(os.path.dirname(cutoffs_filepath), exist_ok=True)
            with open(cutoffs_filepath, 'wb') as f:
                pickle.dump(SPARC_risk_score_cut_offs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved SPARC risk score cutoffs to {cutoffs_filepath}")

    # Generate stratum labels dynamically based on num_strata
    def assign_risk_stratum(score, cutoffs, num_strata):
        for i in range(len(cutoffs) - 1):
            if score < cutoffs[i]:
                if i == 0:
                    return f'Stratum 1 (<{cutoffs[i]*100:.1f}%)'
                else:
                    return f'Stratum {i+1} ({cutoffs[i-1]*100:.1f}-{cutoffs[i]*100:.1f}%)'
        # Last stratum
        return f'Stratum {num_strata} (≥{cutoffs[-2]*100:.1f}%)'

    df_patients['risk_stratum'] = df_patients['sparc_risk_score'].apply(
        lambda x: assign_risk_stratum(x, SPARC_risk_score_cut_offs, num_strata)
    )

    # Build stratum order list
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

        gcsf_ci    = proportion_confint(gcsf_with,    n_g,  method='wilson') if n_g  > 0 else (np.nan, np.nan)
        no_gcsf_ci = proportion_confint(no_gcsf_with, n_ng, method='wilson') if n_ng > 0 else (np.nan, np.nan)

        if n_g > 0 and n_ng > 0:
            arr        = ngr - gr
            rrr        = arr / ngr if ngr > 0 else 0
            rr         = (gr / ngr) if ngr > 0 else np.nan
            odds_ratio, p_val = fisher_exact([[gcsf_with,    n_g  - gcsf_with],
                                              [no_gcsf_with, n_ng - no_gcsf_with]])
            nnt = 1 / arr if arr > 0 else np.inf

            # 95% CI for the ARR via the Newcombe method (difference of two
            # independent Wilson intervals), then invert the ARR CI bounds to
            # obtain the NNT CI. When the ARR CI crosses zero the NNT CI is
            # undefined in the usual sense (it spans NNT-to-benefit through
            # infinity into NNT-to-harm), so we flag those bounds as NaN.
            arr_lower = no_gcsf_ci[0] - gcsf_ci[1]
            arr_upper = no_gcsf_ci[1] - gcsf_ci[0]
            if arr_lower > 0:
                nnt_ci_lower = 1 / arr_upper
                nnt_ci_upper = 1 / arr_lower
            else:
                nnt_ci_lower = np.nan
                nnt_ci_upper = np.nan
        else:
            arr = rrr = rr = odds_ratio = p_val = nnt = np.nan
            arr_lower = arr_upper = nnt_ci_lower = nnt_ci_upper = np.nan

        stratum_results.append({
            'stratum':                  stratum,
            'n_gcsf':                   n_g,
            'n_no_gcsf':                n_ng,
            'gcsf_outcome_rate':        gr,
            'gcsf_ci_lower':            gcsf_ci[0],
            'gcsf_ci_upper':            gcsf_ci[1],
            'no_gcsf_outcome_rate':     ngr,
            'no_gcsf_ci_lower':         no_gcsf_ci[0],
            'no_gcsf_ci_upper':         no_gcsf_ci[1],
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
            print(f"G-CSF:    n={n_g}, outcome rate={gr:.3f} (95% CI: {gcsf_ci[0]:.3f}-{gcsf_ci[1]:.3f}), "
                  f"avg SPARC risk={gcsf_group['sparc_risk_score'].mean():.4f}")
        else:
            print(f"G-CSF:    n=0 (no data)")
        if n_ng > 0:
            print(f"No G-CSF: n={n_ng}, outcome rate={ngr:.3f} (95% CI: {no_gcsf_ci[0]:.3f}-{no_gcsf_ci[1]:.3f}), "
                  f"avg SPARC risk={no_gcsf_group['sparc_risk_score'].mean():.4f}")
        else:
            print(f"No G-CSF: n=0 (no data)")
        if n_g > 0 and n_ng > 0:
            print(f"ARR={arr:.3f} ({arr*100:.1f}%) [95% CI: {arr_lower:.3f} to {arr_upper:.3f}], "
                  f"RRR={rrr:.3f}, RR={rr:.3f}, OR={odds_ratio:.3f}, p={p_val:.4f}")
            if nnt != np.inf and nnt > 0:
                if not (np.isnan(nnt_ci_lower) or np.isnan(nnt_ci_upper)):
                    print(f"NNT: {nnt:.1f} (95% CI: {nnt_ci_lower:.1f} to {nnt_ci_upper:.1f})")
                else:
                    print(f"NNT: {nnt:.1f} (95% CI not estimable — ARR CI crosses 0)")
        else:
            print("(Cross-group statistics unavailable — only one group present)")

    # Association between SPARC risk stratum and NNT with G-CSF
    # Question: are SPARC's predicted risk strata associated with the NNT with
    # G-CSF to avoid a neutropenia event? NNT = 1/ARR, so a monotone association
    # between risk stratum and NNT is equivalent to a trend in the G-CSF
    # treatment effect (ARR / relative effect) across ordered strata. We test
    # this two ways:
    #   (1) A logistic-regression test for a stratum x G-CSF interaction trend:
    #       outcome ~ gcsf + stratum_rank + gcsf:stratum_rank. A significant
    #       interaction term means the G-CSF effect (hence NNT) changes across
    #       ordered risk strata.
    #   (2) Spearman rank correlation between stratum rank and the per-stratum
    #       NNT, as a simple monotonicity summary.
    print("\n" + "="*80)
    print(f"ASSOCIATION: SPARC RISK STRATUM vs NNT WITH G-CSF{output_suffix}")
    print("="*80 + "\n")

    # Map each stratum label to an ordinal rank (1 = lowest risk)
    stratum_rank_map = {s: i for i, s in enumerate(stratum_order)}
    df_assoc = df_patients[df_patients['risk_stratum'].isin(stratum_rank_map)].copy()
    df_assoc['stratum_rank'] = df_assoc['risk_stratum'].map(stratum_rank_map).astype(float)

    interaction_p = np.nan
    interaction_coef = np.nan
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        # Coerce to clean numeric 0/1 and drop any rows with missing values,
        # otherwise the design matrix can become rank-deficient and statsmodels
        # returns NaN coefficients silently.
        df_fit = df_assoc[['outcome', 'gcsf', 'stratum_rank']].copy()
        df_fit['outcome'] = pd.to_numeric(df_fit['outcome'], errors='coerce')
        df_fit['gcsf'] = pd.to_numeric(df_fit['gcsf'], errors='coerce')
        df_fit['stratum_rank'] = pd.to_numeric(df_fit['stratum_rank'], errors='coerce')
        n_before = len(df_fit)
        df_fit = df_fit.dropna()
        if len(df_fit) < n_before:
            print(f"  Dropped {n_before - len(df_fit)} rows with missing "
                  f"outcome/gcsf/stratum_rank before fitting.")
        # Center rank so the main G-CSF effect is interpretable at the middle stratum
        df_fit['stratum_rank_c'] = df_fit['stratum_rank'] - df_fit['stratum_rank'].mean()

        # Guard against degenerate design (need variation in both gcsf and outcome)
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

    # Breslow-Day test for homogeneity of the odds ratio across strata.
    # This is a model-free check of whether the G-CSF effect (and hence NNT)
    # is constant across risk strata, and does not assume a linear trend.
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
            bd_res = st.test_equal_odds()   # Breslow-Day
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

    # Spearman correlation between stratum rank and per-stratum NNT
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

    # Save a focused NNT-by-stratum table
    nnt_table = results_df_tmp[[
        'stratum', 'n_gcsf', 'n_no_gcsf',
        'gcsf_outcome_rate', 'no_gcsf_outcome_rate',
        'absolute_risk_reduction', 'arr_ci_lower', 'arr_ci_upper',
        'nnt', 'nnt_ci_lower', 'nnt_ci_upper', 'p_value',
    ]].copy()
    nnt_csv = f'sparc_nnt_by_risk_stratum_{score_label}_{option}_{dataset}{tag}{output_suffix}.csv'
    nnt_table.to_csv(os.path.join(output_dir, nnt_csv), index=False)
    print(f"\nSaved NNT-by-stratum table to {os.path.join(output_dir, nnt_csv)}")

    # Figure: NNT (with CI) by risk stratum
    plot_df = results_df_tmp[
        np.isfinite(results_df_tmp['nnt']) & results_df_tmp['nnt'].notna()
    ].copy()
    if len(plot_df) > 0:
        fig, axn = plt.subplots(figsize=(10, 6))
        xn = np.arange(len(plot_df))
        nnt_vals = plot_df['nnt'].values
        lo = plot_df['nnt_ci_lower'].values
        hi = plot_df['nnt_ci_upper'].values
        yerr_lo = np.where(np.isnan(lo), 0, np.maximum(0, nnt_vals - lo))
        yerr_hi = np.where(np.isnan(hi), 0, np.maximum(0, hi - nnt_vals))
        axn.bar(xn, nnt_vals, yerr=[yerr_lo, yerr_hi], capsize=6,
                color='#8e44ad', alpha=0.85, edgecolor='black')
        for i, (v, l, h) in enumerate(zip(nnt_vals, lo, hi)):
            if np.isnan(l) or np.isnan(h):
                lab = f'NNT={v:.0f}\n(CI n/e)'
            else:
                lab = f'NNT={v:.0f}\n({l:.0f}-{h:.0f})'
            axn.text(i, v + max(nnt_vals) * 0.02, lab, ha='center', va='bottom', fontsize=8)
        axn.set_xticks(xn)
        axn.set_xticklabels([s.replace(' ', '\n') for s in plot_df['stratum']], fontsize=9)
        axn.set_xlabel('SPARC Risk Stratum', fontsize=11)
        axn.set_ylabel('Number Needed to Treat (G-CSF)', fontsize=11)
        outcome_label_n = 'Febrile Neutropenia' if 'febrile' in option else 'Neutropenia'
        title_bits = f'NNT with G-CSF to Avoid {outcome_label_n} by SPARC Risk Stratum'
        if not np.isnan(interaction_p):
            title_bits += f'\ninteraction p={interaction_p:.3f}'
        if not np.isnan(spearman_rho):
            title_bits += f'   |   Spearman rho={spearman_rho:.2f}, p={spearman_p:.3f}'
        axn.set_title(title_bits, fontsize=12, fontweight='bold')
        axn.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        nnt_fig = f'sparc_nnt_by_risk_stratum_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'
        plt.savefig(os.path.join(output_dir, nnt_fig), bbox_inches='tight', dpi=300)
        print(f"Saved NNT-by-stratum figure to {os.path.join(output_dir, nnt_fig)}")
        plt.close()

    # Visualisation: outcome rates & ARR by stratum
    if len(stratum_results) > 0:
        results_df = pd.DataFrame(stratum_results)

        # Horizontal grouped bars, styled to match the clinical-benchmark
        # figure (vis_eval_7_clinical_benchmark_forest_plot_MSK.py): black-edged solid
        # bars, separate black error bars, dashed value-axis grid, no top/right
        # spines, legend below the axes. Both panels share the stratum axis, so
        # they are flipped together.
        n_str = len(results_df)
        y_pos = np.arange(n_str, dtype=float)
        # BAR_H is bar thickness, ROW_GAP the centre-to-centre spacing of the
        # two arms, so ROW_GAP - BAR_H is the visible gap between them (as in
        # the clinical-benchmark figure, which leaves 0.10 between bars).
        BAR_H, ROW_GAP = 0.32, 0.40
        SIG_COL = '#555555'
        fig, axes = plt.subplots(1, 2, figsize=(15, max(5.0, 1.15 * n_str + 2.4)))

        # Two lines: "Stratum k" over its score range, which is too long inline.
        strat_labels = [s.replace(' (', '\n(') for s in results_df['stratum']]

        ax1 = axes[0]
        ge_lo = np.nan_to_num(np.maximum(0, results_df['gcsf_outcome_rate']    - results_df['gcsf_ci_lower']))
        ge_hi = np.nan_to_num(np.maximum(0, results_df['gcsf_ci_upper']        - results_df['gcsf_outcome_rate']))
        ne_lo = np.nan_to_num(np.maximum(0, results_df['no_gcsf_outcome_rate'] - results_df['no_gcsf_ci_lower']))
        ne_hi = np.nan_to_num(np.maximum(0, results_df['no_gcsf_ci_upper']     - results_df['no_gcsf_outcome_rate']))
        g_rate  = results_df['gcsf_outcome_rate'].fillna(0).to_numpy()
        ng_rate = results_df['no_gcsf_outcome_rate'].fillna(0).to_numpy()

        ax1.barh(y_pos - ROW_GAP/2, g_rate,  height=BAR_H, color='#3498db',
                 edgecolor='black', linewidth=0.9, label='G-CSF', zorder=2)
        ax1.barh(y_pos + ROW_GAP/2, ng_rate, height=BAR_H, color='#e74c3c',
                 edgecolor='black', linewidth=0.9, label='No G-CSF', zorder=2)
        ax1.errorbar(g_rate, y_pos - ROW_GAP/2, xerr=[ge_lo, ge_hi], fmt='none',
                     ecolor='black', elinewidth=1.1, capsize=3, capthick=1.1, zorder=3)
        ax1.errorbar(ng_rate, y_pos + ROW_GAP/2, xerr=[ne_lo, ne_hi], fmt='none',
                     ecolor='black', elinewidth=1.1, capsize=3, capthick=1.1, zorder=3)

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

        ci_upper_vals = pd.concat([results_df['gcsf_ci_upper'], results_df['no_gcsf_ci_upper']]).dropna()
        y_max = ci_upper_vals.max() if len(ci_upper_vals) > 0 else 1.0
        # Headroom for the n/p annotation column at the right edge. Needs to
        # clear the longest label ("n=NNNN/NNNN, p<0.001***") without it
        # overlapping the widest CI cap.
        x_hi  = min(max(y_max * 1.55, y_max + 0.12), 1.0)
        sig_x = x_hi - 0.01 * (x_hi - 0)
        ax1.set_xlim(0, x_hi)

        for i, (n_g, n_ng, pv) in enumerate(zip(
                results_df['n_gcsf'], results_df['n_no_gcsf'], results_df['p_value'])):
            if pd.isna(pv):
                p_text = 'n/a'
            else:
                p_text = ('p<0.001***' if pv < 0.001 else f'p={pv:.3f}**' if pv < 0.01
                          else f'p={pv:.3f}*' if pv < 0.05 else f'p={pv:.2f}')
            # Fixed column at the right edge rather than trailing each CI, so
            # the annotations line up (matches the clinical-benchmark figure).
            ax1.text(sig_x, y_pos[i], f'n={int(n_g)}/{int(n_ng)}, {p_text}',
                     ha='right', va='center', fontsize=8, fontweight='bold',
                     color='black')

        ax2 = axes[1]
        arr_vals = results_df['absolute_risk_reduction'].fillna(0).to_numpy()
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in arr_vals]
        ax2.barh(y_pos, arr_vals, height=0.52, color=colors,
                 edgecolor='black', linewidth=0.9, zorder=2)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
        ax2.set_xlabel('Absolute risk reduction', fontsize=12)
        ax2.set_title('G-CSF treatment effect (ARR)', fontsize=13, pad=12, loc='left')
        ax2.set_yticks(y_pos); ax2.set_yticklabels(strat_labels, fontsize=11)
        ax2.tick_params(axis='y', length=0)
        ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.invert_yaxis()

        v_abs_min = max(np.abs(arr_vals).min(), 1e-6)
        for i, (v, pv) in enumerate(zip(arr_vals, results_df['p_value'])):
            if pd.isna(pv):
                label = f'{v:.3f} (n/a)'
            else:
                sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
                label = f'{v:.3f}{sig}'
            ax2.text(v + (v_abs_min * 0.2 if v >= 0 else -v_abs_min * 0.2), y_pos[i],
                     label, ha='left' if v >= 0 else 'right', va='center',
                     fontsize=9, color=SIG_COL)
        ax2.margins(x=0.22)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
            f'sparc_gcsf_effect_by_risk_stratum_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'),
            bbox_inches='tight', dpi=300)
        print("Saved treatment effect figure.")

    # Calibration plot (risk score vs observed outcome rate) with per-fold CIs
    print("\n" + "="*80)
    print("CREATING CALIBRATION PLOT WITH PER-FOLD CONFIDENCE INTERVALS")
    print("="*80 + "\n")

    # Check if we have per-fold predictions
    if 'sparc_risk_score_per_fold' in df_patients.columns:
        print("Computing calibration curves across all 10 folds...")

        # Use fixed bins based on mean scores to ensure consistency across folds
        n_bins = 10  # Reduce bins for stability across folds
        outcomes = df_patients['outcome'].values
        mean_scores = df_patients['sparc_risk_score'].values

        # Compute quantile-based bin edges from mean scores
        bin_edges = np.percentile(mean_scores, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] = bin_edges[-1] + 1e-8  # Ensure last bin includes max value

        # Collect calibration curves for each fold
        fold_calibrations = []
        fold_bin_centers = []

        for fold_idx in range(10):
            # Extract fold-specific predictions
            fold_preds = np.array([row[fold_idx] for row in df_patients['sparc_risk_score_per_fold']])

            # Assign to bins and compute observed rates
            bin_indices = np.digitize(fold_preds, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            prob_true_fold = []
            prob_pred_fold = []

            for bin_i in range(n_bins):
                mask = (bin_indices == bin_i)
                if mask.sum() > 0:
                    prob_true_fold.append(outcomes[mask].mean())
                    prob_pred_fold.append(fold_preds[mask].mean())
                else:
                    prob_true_fold.append(np.nan)
                    prob_pred_fold.append(np.nan)

            fold_calibrations.append(prob_true_fold)
            fold_bin_centers.append(prob_pred_fold)

        # Compute mean and percentile-based CIs across folds
        fold_calibrations = np.array(fold_calibrations)  # shape: (10, n_bins)
        fold_bin_centers = np.array(fold_bin_centers)    # shape: (10, n_bins)

        # For each bin, compute mean and CI (ignoring NaN bins)
        prob_true_mean = np.nanmean(fold_calibrations, axis=0)
        prob_true_lower = np.nanpercentile(fold_calibrations, 2.5, axis=0)
        prob_true_upper = np.nanpercentile(fold_calibrations, 97.5, axis=0)
        prob_pred_mean = np.nanmean(fold_bin_centers, axis=0)

        # Remove bins where all folds are NaN
        valid_bins = ~np.isnan(prob_true_mean)
        prob_true_mean = prob_true_mean[valid_bins]
        prob_true_lower = prob_true_lower[valid_bins]
        prob_true_upper = prob_true_upper[valid_bins]
        prob_pred_mean = prob_pred_mean[valid_bins]

        print(f"Computed calibration across {n_bins} bins with {valid_bins.sum()} non-empty bins")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot individual fold calibration curves (thin lines)
        for fold_idx in range(10):
            fold_prob_true = fold_calibrations[fold_idx, valid_bins]
            fold_prob_pred = fold_bin_centers[fold_idx, valid_bins]
            ax.plot(fold_prob_pred, fold_prob_true, 'o-', alpha=0.2, color='#3498db', markersize=3, linewidth=1)

        # Plot mean with CI band
        ax.plot(prob_pred_mean, prob_true_mean, 'o-', label='Mean SPARC (across folds)',
                color='#3498db', linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(prob_pred_mean, prob_true_lower, prob_true_upper,
                        alpha=0.3, color='#3498db', label='95% CI (across folds)')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Perfect Calibration')

        ax.set_xlabel('Mean Predicted Risk', fontsize=12)
        ax.set_ylabel('Observed Outcome Rate', fontsize=12)
        ax.set_title('Calibration Plot with Per-Fold Confidence Intervals', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
            f'sparc_calibration_plot_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'),
            bbox_inches='tight', dpi=300)
        print("Saved calibration plot with per-fold confidence intervals.")
        plt.close()

    else:
        # Fallback to original single calibration curve if per-fold data not available
        print("WARNING: Per-fold predictions not available, using mean predictions only")
        prob_true, prob_pred = calibration_curve(
            df_patients['outcome'], df_patients['sparc_risk_score'], n_bins=20, strategy='quantile')
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='SPARC Risk Score', color='#3498db')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Risk', fontsize=12)
        plt.ylabel('Observed Outcome Rate', fontsize=12)
        plt.title('Calibration Plot for SPARC Risk Score', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir,
            f'sparc_calibration_plot_{score_label}_{option}_{dataset}{tag}{output_suffix}.pdf'),
            bbox_inches='tight', dpi=300)
        print("Saved calibration plot.")
        plt.close()

    return stratum_results


# Step 1: collect predictions across all 10 folds (same as original)
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

    # Build 060726 module tokenizer configuration
    module_index_cfg = build_module_index_config(
        modular_featuredim_dict, advanced_model='default'
    )
    n_static_features = module_index_cfg['n_static']

    # Determine loss_option based on task type
    # Hybrid tasks (FN, N, A) use 'hybrid' (Cox + BCE), others use 'default'
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
        # 060726-specific parameters
        module_index_cfg=module_index_cfg,
        use_module_tokenizer=True,
        use_continuous_time=True,
        n_static_features=n_static_features,
        post_mlp_dropout=0.2,
    )
    model = SPARC.load_from_checkpoint(optimal_model_path, **load_kwargs)

    pl.seed_everything(rseed)
    basic_feature_trainer = pl.Trainer(max_epochs=1, deterministic=True)
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

# Step 2: Aggregate predictions by patient
print("\nStep 2: Aggregating SPARC predictions by patient...")

patient_data = []
for i, pred_dict in patient_predictions.items():
    patient_data.append({
        'DMP_ID':                     pred_dict['DMP_ID'],
        'cutoffdate':                 pred_dict['cutoffdate'][0],
        'raw_sparc_risk_score':       np.mean(pred_dict['raw_predictions']),
        'sparc_risk_score':           np.mean(pred_dict['predictions']),
        'sparc_risk_score_per_fold':  pred_dict['predictions'],  # Keep per-fold predictions
        'gcsf':                       pred_dict['gcsf_flags'][0],
        'outcome':                    pred_dict['outcomes'][0],
    })

df_patients_test = pd.DataFrame(patient_data)
df_patients_test = df_patients_test.sort_values(by=['DMP_ID', 'cutoffdate']).reset_index(drop=True)

score_label = 'sigmoid'

if run_psm:
    # Step 3: Extract raw input features and build propensity score model on train set
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING RAW FEATURES AND BUILDING PROPENSITY SCORE MODEL ON TRAIN SET")
    print("="*80 + "\n")

    # Extract raw features from train and test datasets
    print("Extracting raw input features from train and test datasets...")
    
    def extract_features_from_dataset(dataset):
        """
        Extract flattened features from dataset for propensity score modeling.
        Follows the temporal_xgboost approach from SL_benchmarks_train_risk_corrected_fixed.py
    
        Returns:
            features: np.array of shape (n_samples, n_features)
            dmp_ids: list of DMP_IDs
            cutoffdates: list of cutoffdates
            gcsf_flags: np.array of G-CSF flags
            outcomes: np.array of outcomes
        """
        features_list = []
        dmp_ids = []
        cutoffdates = []
        gcsf_flags = []
        outcomes = []
    
        for idx in range(len(dataset)):
            # Get the raw data tuple
            # dataset[idx] returns (x, y) where x is (X_ts, X_dates, [num_infusion], [DMP_ID], [gcsf_groundtruth])
            data_tuple = dataset[idx]
            x_tuple = data_tuple[0]  # The features tuple
    
            # Extract time series features (X_ts) and timestamps (X_dates)
            xs_tp = x_tuple[0].detach().clone()  # Shape: (time_steps, features)
            time_tp = x_tuple[1].detach().clone()  # Shape: (time_steps,)
    
            # Create mask for missing values (NaN)
            mask = torch.isnan(xs_tp).detach().clone()
    
            # Fill missing values with 0
            xs_tp[mask] = 0
    
            # For temporal models: concatenate features, mask, time, and time_nan_mask
            # Then flatten everything
            time_nan_mask = torch.isnan(time_tp).int().unsqueeze(-1).detach().clone()
            time_tp_combine = time_tp.unsqueeze(-1)
            time_tp_combine[time_nan_mask.bool()] = 0.0
    
            # Concatenate: [xs_tp (cleaned), mask, time, time_nan_mask]
            xs_tp = torch.cat((xs_tp, mask, time_tp_combine, time_nan_mask), dim=1).view(-1)
    
            # Add infusion number if available
            if add_num_infusion and len(x_tuple) >= 3:
                xs_tp_infusions = x_tuple[2]
                if not isinstance(xs_tp_infusions, torch.Tensor):
                    xs_tp_infusions = torch.tensor(xs_tp_infusions, dtype=torch.float32)
                xs_tp_infusions = xs_tp_infusions.unsqueeze(-1)
                xs_tp = torch.cat((xs_tp, xs_tp_infusions))
    
            # Convert to numpy
            feature_vec = xs_tp.numpy()
    
            features_list.append(feature_vec)
            dmp_ids.append(dataset.dataset_DMP_IDs[idx])
            cutoffdates.append(dataset.cutoffdates[idx])
            gcsf_flags.append(dataset.gcsf_groundtruth[idx])
    
            # Get outcome from y (second element of the tuple)
            y = data_tuple[1]
            outcomes.append(y)
    
        features = np.array(features_list)
        gcsf_flags = np.array(gcsf_flags)
    
        # Handle outcomes (might be tuple for risk tasks)
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
    
    # Build logistic regression model to predict G-CSF usage from raw features
    print("\nBuilding logistic regression model to predict G-CSF usage from raw patient features...")
    ps_model = LogisticRegression(random_state=rseed, max_iter=1000, solver='lbfgs')
    ps_model.fit(X_train_raw, y_train_gcsf)
    
    print(f"Propensity score model trained with {X_train_raw.shape[1]} features")
    print(f"Number of non-zero coefficients: {np.sum(ps_model.coef_ != 0)}")
    
    # Evaluate propensity score model on test set
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    y_test_gcsf_pred_proba = ps_model.predict_proba(X_test_raw)[:, 1]
    y_train_gcsf_pred_proba = ps_model.predict_proba(X_train_raw)[:, 1]
    
    train_ps_auroc = roc_auc_score(y_train_gcsf, y_train_gcsf_pred_proba)
    train_ps_auprc = average_precision_score(y_train_gcsf, y_train_gcsf_pred_proba)
    test_ps_auroc = roc_auc_score(y_test_gcsf, y_test_gcsf_pred_proba)
    test_ps_auprc = average_precision_score(y_test_gcsf, y_test_gcsf_pred_proba)
    
    print(f"\nPropensity Score Model Performance:")
    print(f"  Train set - AUROC: {train_ps_auroc:.4f}, AUPRC: {train_ps_auprc:.4f}")
    print(f"  Test set  - AUROC: {test_ps_auroc:.4f}, AUPRC: {test_ps_auprc:.4f}")
    
    # Add propensity scores to patient dataframes
    df_patients_test['propensity_score'] = y_test_gcsf_pred_proba
    
    print(f"\nPropensity score range in test set: [{df_patients_test['propensity_score'].min():.4f}, {df_patients_test['propensity_score'].max():.4f}]")
    print(f"Mean propensity score: {df_patients_test['propensity_score'].mean():.4f}")
    
    # Plot propensity score distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Propensity score distributions by G-CSF status
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
    
    # Plot 2: ROC curve
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
    # strata_tag ('' for quintiles, '_3strata' etc. otherwise) keeps runs
    # with different -ns from overwriting each other's outputs.
    tag = (f'_gcsflim{gcsf_lim}' if gcsf_lim is not None else '') + strata_tag
    plt.savefig(os.path.join(output_dir, f'propensity_score_model_performance_{dataset}{tag}.pdf'), dpi=300)
    print(f"Saved propensity score model performance figure to {output_dir}")
    
    # Step 4: Perform 1:1 nearest neighbor propensity score matching
    print("\n" + "="*80)
    print("STEP 4: PERFORMING 1:1 NEAREST NEIGHBOR PROPENSITY SCORE MATCHING")
    print("="*80 + "\n")
    
    # Separate G-CSF and no G-CSF groups
    gcsf_indices = df_patients_test[df_patients_test['gcsf'] == 1].index.values
    no_gcsf_indices = df_patients_test[df_patients_test['gcsf'] == 0].index.values
    
    print(f"G-CSF group: {len(gcsf_indices)} patients")
    print(f"No G-CSF group: {len(no_gcsf_indices)} patients")
    
    # We'll match each G-CSF patient to the nearest no-G-CSF patient (1:1 matching without replacement)
    # Use the smaller group as the treatment group
    if len(gcsf_indices) <= len(no_gcsf_indices):
        treatment_indices = gcsf_indices
        control_indices = no_gcsf_indices
        treatment_label = "G-CSF"
    else:
        treatment_indices = no_gcsf_indices
        control_indices = gcsf_indices
        treatment_label = "No G-CSF"
    
    print(f"\nMatching {len(treatment_indices)} {treatment_label} patients to control patients...")
    
    # Get propensity scores
    treatment_ps = df_patients_test.loc[treatment_indices, 'propensity_score'].values.reshape(-1, 1)
    control_ps = df_patients_test.loc[control_indices, 'propensity_score'].values.reshape(-1, 1)
    
    # Perform nearest neighbor matching
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_ps)
    distances, matched_control_idx = nn.kneighbors(treatment_ps)
    
    # Get the actual indices from the dataframe
    matched_control_indices = control_indices[matched_control_idx.flatten()]
    
    # Create matched dataset
    matched_indices = np.concatenate([treatment_indices, matched_control_indices])
    df_patients_matched = df_patients_test.loc[matched_indices].copy()
    
    print(f"\nMatched dataset size: {len(df_patients_matched)} patients")
    print(f"Matched G-CSF: {df_patients_matched['gcsf'].sum()} patients")
    print(f"Matched No G-CSF: {(df_patients_matched['gcsf'] == 0).sum()} patients")
    print(f"Mean propensity score difference: {distances.mean():.4f}")
    print(f"Max propensity score difference: {distances.max():.4f}")
    
    # Check balance
    print("\n" + "="*80)
    print("CHECKING COVARIATE BALANCE AFTER MATCHING")
    print("="*80 + "\n")
    
    print("Before matching:")
    print(f"  G-CSF mean SPARC score: {df_patients_test[df_patients_test['gcsf']==1]['sparc_risk_score'].mean():.4f}")
    print(f"  No G-CSF mean SPARC score: {df_patients_test[df_patients_test['gcsf']==0]['sparc_risk_score'].mean():.4f}")
    _, p_before = mannwhitneyu(
        df_patients_test[df_patients_test['gcsf']==1]['sparc_risk_score'],
        df_patients_test[df_patients_test['gcsf']==0]['sparc_risk_score']
    )
    print(f"  Mann-Whitney U p-value: {p_before:.4f}")
    
    print("\nAfter matching:")
    print(f"  G-CSF mean SPARC score: {df_patients_matched[df_patients_matched['gcsf']==1]['sparc_risk_score'].mean():.4f}")
    print(f"  No G-CSF mean SPARC score: {df_patients_matched[df_patients_matched['gcsf']==0]['sparc_risk_score'].mean():.4f}")
    _, p_after = mannwhitneyu(
        df_patients_matched[df_patients_matched['gcsf']==1]['sparc_risk_score'],
        df_patients_matched[df_patients_matched['gcsf']==0]['sparc_risk_score']
    )
    print(f"  Mann-Whitney U p-value: {p_after:.4f}")

# Step 5: Run standard analysis (always performed)
print("\n" + "="*80)
print("STEP 5: RUNNING STANDARD ANALYSIS")
print("="*80)
stratum_results_original = run_gcsf_analysis(
    df_patients_test, '', output_dir, option, dataset, score_label, gcsf_lim,
    reference_cutoffs=reference_cutoffs,
    save_cutoffs=should_save_cutoffs,
    cutoffs_filepath=cutoffs_filepath,
    num_strata=num_strata
)

if run_psm:
    # Step 6: Run PSM analysis
    print("\n" + "="*80)
    print("STEP 6: RUNNING PSM ANALYSIS")
    print("="*80)
    stratum_results_psm = run_gcsf_analysis(
        df_patients_matched, '_post_psm', output_dir, option, dataset, score_label, gcsf_lim,
        reference_cutoffs=reference_cutoffs,
        save_cutoffs=False,  # Don't save cutoffs twice
        cutoffs_filepath=None,
        num_strata=num_strata
    )
else:
    print("\nSkipping PSM analysis (Step 6)")

print("\nDone.")