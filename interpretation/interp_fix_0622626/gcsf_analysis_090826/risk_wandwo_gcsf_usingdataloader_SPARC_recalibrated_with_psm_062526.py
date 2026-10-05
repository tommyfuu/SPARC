#!/usr/bin/env python3
"""G-CSF effect on neutropenia / febrile neutropenia, stratified by SPARC predicted risk, with PSM.

-gl: G-CSF must start at most this many days before the prediction date. -ns: number of risk strata.
Usage: python3 risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_062526.py -o neutropenia -ds MSK_CAIA -ns 5 -gl 2
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
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repodir = os.path.abspath(os.path.join(currentdir, '..', '..', '..'))

sys.path.insert(0, os.path.join(repodir, 'data'))
sys.path.insert(0, os.path.join(repodir, 'models'))

from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset

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
parser.add_argument("-k",   "--match_k",           type=int,   default=1,
                    help="k:1 propensity matching -- keep k controls per treated patient. "
                         "k=1 reproduces the original behaviour; larger k retains more of the "
                         "control pool and lowers variance at some cost in match quality.")
parser.add_argument("-ws",  "--within_stratum",    type=str,   default='False',
                    help="True | False -- match INSIDE each risk stratum rather than globally. "
                         "Estimates are computed per stratum, so balance should be enforced there.")
parser.add_argument("-cal", "--caliper",           type=float, default=0.0,
                    help="Propensity-score caliper; 0 disables. Matches beyond this distance are dropped.")
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

# model paths
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

# risk cutoffs are set on MSK_CAIA prior and reused for the other datasets
cutoffs_reference_dir = os.path.join(currentdir, f'out_{option}_{calibration_suffix}')

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

# PSM only on MSK_CAIA prior; temporal validation sets get the unmatched analysis
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

    # NNT by stratum figure
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

    # Visualisation: outcome rates by stratum
    if len(stratum_results) > 0:
        results_df = pd.DataFrame(stratum_results)

        # styled to match vis_eval_7_clinical_benchmark_forest_plot_MSK.py
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

    # Calibration plot (risk score vs observed outcome rate) with per-fold CIs
    print("\n" + "="*80)
    print("CREATING CALIBRATION PLOT WITH PER-FOLD CONFIDENCE INTERVALS")
    print("="*80 + "\n")

    if 'sparc_risk_score_per_fold' in df_patients.columns:
        print("Computing calibration curves across all 10 folds...")

        # bins fixed on the fold-mean score so all folds share them
        n_bins = 10
        outcomes = df_patients['outcome'].values
        mean_scores = df_patients['sparc_risk_score'].values

        bin_edges = np.percentile(mean_scores, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] = bin_edges[-1] + 1e-8  # include the max

        fold_calibrations = []
        fold_bin_centers = []

        for fold_idx in range(10):
            fold_preds = np.array([row[fold_idx] for row in df_patients['sparc_risk_score_per_fold']])

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

        fold_calibrations = np.array(fold_calibrations)  # shape: (10, n_bins)
        fold_bin_centers = np.array(fold_bin_centers)    # shape: (10, n_bins)

        # per-bin mean and 95% range across folds
        prob_true_mean = np.nanmean(fold_calibrations, axis=0)
        prob_true_lower = np.nanpercentile(fold_calibrations, 2.5, axis=0)
        prob_true_upper = np.nanpercentile(fold_calibrations, 97.5, axis=0)
        prob_pred_mean = np.nanmean(fold_bin_centers, axis=0)

        valid_bins = ~np.isnan(prob_true_mean)
        prob_true_mean = prob_true_mean[valid_bins]
        prob_true_lower = prob_true_lower[valid_bins]
        prob_true_upper = prob_true_upper[valid_bins]
        prob_pred_mean = prob_pred_mean[valid_bins]

        print(f"Computed calibration across {n_bins} bins with {valid_bins.sum()} non-empty bins")

        fig, ax = plt.subplots(figsize=(8, 6))

        # per-fold curves
        for fold_idx in range(10):
            fold_prob_true = fold_calibrations[fold_idx, valid_bins]
            fold_prob_pred = fold_bin_centers[fold_idx, valid_bins]
            ax.plot(fold_prob_pred, fold_prob_true, 'o-', alpha=0.2, color='#3498db', markersize=3, linewidth=1)

        ax.plot(prob_pred_mean, prob_true_mean, 'o-', label='Mean SPARC (across folds)',
                color='#3498db', linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(prob_pred_mean, prob_true_lower, prob_true_upper,
                        alpha=0.3, color='#3498db', label='95% CI (across folds)')

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


# Step 1: predictions from all 10 fold models
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
        'sparc_risk_score_per_fold':  pred_dict['predictions'],
        'gcsf':                       pred_dict['gcsf_flags'][0],
        'outcome':                    pred_dict['outcomes'][0],
    })

df_patients_test = pd.DataFrame(patient_data)
df_patients_test = df_patients_test.sort_values(by=['DMP_ID', 'cutoffdate']).reset_index(drop=True)

score_label = 'sigmoid'

if run_psm:
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
    
    df_patients_test['propensity_score'] = y_test_gcsf_pred_proba
    
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
    
    # Step 4: nearest-neighbour propensity matching
    print("\n" + "="*80)
    print("STEP 4: PERFORMING 1:1 NEAREST NEIGHBOR PROPENSITY SCORE MATCHING")
    print("="*80 + "\n")
    
    gcsf_indices = df_patients_test[df_patients_test['gcsf'] == 1].index.values
    no_gcsf_indices = df_patients_test[df_patients_test['gcsf'] == 0].index.values
    
    print(f"G-CSF group: {len(gcsf_indices)} patients")
    print(f"No G-CSF group: {len(no_gcsf_indices)} patients")
    
    # the smaller arm is treated as "treatment"
    if len(gcsf_indices) <= len(no_gcsf_indices):
        treatment_indices = gcsf_indices
        control_indices = no_gcsf_indices
        treatment_label = "G-CSF"
    else:
        treatment_indices = no_gcsf_indices
        control_indices = gcsf_indices
        treatment_label = "No G-CSF"
    
    print(f"\nMatching {len(treatment_indices)} {treatment_label} patients to control patients...")
    
    treatment_ps = df_patients_test.loc[treatment_indices, 'propensity_score'].values.reshape(-1, 1)
    control_ps = df_patients_test.loc[control_indices, 'propensity_score'].values.reshape(-1, 1)
    
    # k:1 matching is with replacement (controls can be reused), so the bootstrap
    # CI is somewhat optimistic. k>1 keeps more of the control pool.
    _k = max(1, int(args.match_k))
    _cal = float(args.caliper)

    def _match_block(_treat_idx, _ctrl_idx):
        """k:1 match one block; returns (treated_kept, control_matched)."""
        if len(_treat_idx) == 0 or len(_ctrl_idx) == 0:
            return np.array([], dtype=_treat_idx.dtype), np.array([], dtype=_ctrl_idx.dtype)
        _tp = df_patients_test.loc[_treat_idx, 'propensity_score'].values.reshape(-1, 1)
        _cp = df_patients_test.loc[_ctrl_idx, 'propensity_score'].values.reshape(-1, 1)
        _kk = min(_k, len(_ctrl_idx))
        _nn = NearestNeighbors(n_neighbors=_kk, algorithm='ball_tree')
        _nn.fit(_cp)
        _d, _mi = _nn.kneighbors(_tp)
        if _cal > 0:
            _keep_t, _keep_c = [], []
            for _r in range(_mi.shape[0]):
                _ok = _mi[_r][_d[_r] <= _cal]
                if len(_ok):
                    _keep_t.append(_treat_idx[_r]); _keep_c.extend(_ctrl_idx[_ok])
            return np.array(_keep_t), np.array(_keep_c)
        return _treat_idx, _ctrl_idx[_mi.flatten()]

    if args.within_stratum == 'True':
        # strata from the unmatched test set, same cutoffs as the analysis
        _cuts = reference_cutoffs if reference_cutoffs is not None else [
            np.percentile(df_patients_test['sparc_risk_score'], 100.0 * (i + 1) / num_strata)
            for i in range(num_strata)]
        def _strat(_x):
            for _i in range(len(_cuts) - 1):
                if _x < _cuts[_i]:
                    return _i
            return num_strata - 1
        _sv = df_patients_test['sparc_risk_score'].apply(_strat)
        _t_all, _c_all = [], []
        for _sname in sorted(_sv.unique()):
            _m = (_sv == _sname)
            _ti = np.intersect1d(treatment_indices, df_patients_test.index[_m])
            _ci = np.intersect1d(control_indices,   df_patients_test.index[_m])
            _t, _c = _match_block(_ti, _ci)
            print(f"  stratum {_sname}: matched {len(_t)} treated to {len(_c)} controls "
                  f"(pool {len(_ci)})")
            _t_all.append(_t); _c_all.append(_c)
        treatment_kept = np.concatenate([a for a in _t_all if len(a)]) if any(len(a) for a in _t_all) else np.array([])
        matched_control_indices = np.concatenate([a for a in _c_all if len(a)]) if any(len(a) for a in _c_all) else np.array([])
    else:
        treatment_kept, matched_control_indices = _match_block(treatment_indices, control_indices)

    print(f"Matching mode: {'within-stratum' if args.within_stratum=='True' else 'global'}, "
          f"k={_k}, caliper={_cal if _cal>0 else 'none'}")

    matched_indices = np.concatenate([treatment_kept, matched_control_indices])
    df_patients_matched = df_patients_test.loc[matched_indices].copy()
    
    print(f"\nMatched dataset size: {len(df_patients_matched)} patients")
    print(f"Matched G-CSF: {df_patients_matched['gcsf'].sum()} patients")
    print(f"Matched No G-CSF: {(df_patients_matched['gcsf'] == 0).sum()} patients")
    # PS distance of the realised matches
    _ps_t = df_patients_test.loc[treatment_kept, 'propensity_score'].values
    _ps_c = df_patients_test.loc[matched_control_indices, 'propensity_score'].values
    if len(_ps_t) and len(_ps_c):
        _reps = max(1, len(_ps_c) // max(1, len(_ps_t)))
        _dist = np.abs(np.repeat(_ps_t, _reps)[:len(_ps_c)] - _ps_c[:len(_ps_t) * _reps])
        print(f"Mean propensity score difference: {_dist.mean():.4f}")
        print(f"Max propensity score difference: {_dist.max():.4f}")
    else:
        print("Mean propensity score difference: n/a (no matches)")
    
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

# Step 5: unmatched analysis
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
    # Step 6: matched analysis
    print("\n" + "="*80)
    print("STEP 6: RUNNING PSM ANALYSIS")
    print("="*80)
    stratum_results_psm = run_gcsf_analysis(
        df_patients_matched, '_post_psm', output_dir, option, dataset, score_label, gcsf_lim,
        reference_cutoffs=reference_cutoffs,
        save_cutoffs=False,
        cutoffs_filepath=None,
        num_strata=num_strata
    )
else:
    print("\nSkipping PSM analysis (Step 6)")

print("\nDone.")