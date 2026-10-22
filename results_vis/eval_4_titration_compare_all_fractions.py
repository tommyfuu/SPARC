"""Directly-trained vs pretrained-then-finetuned SPARC across training-set sizes.

Fractions 1/1 to 1/256 (1/32 only for binary_icu_pre), all scored on the full test set;
missing fractions are skipped.

Usage: python3 eval_4_titration_compare_all_fractions.py -o AKI_risk
"""

import os
import sys
import pickle
import inspect

import torch
import pandas as pd
import pytorch_lightning as pl

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
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

from data_loaders_inf_risk_fixed import FlexibleFeatureDataModule
from dataset_loading import load_dataset
from SPARC_inf_risk_modular import *
from MSK_featurename_list import build_module_index_config


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

from eval_utils import calculate_weighted_c_index_by_infusion
from torchmetrics import AUROC, AveragePrecision

# Arguments
import argparse

parser = argparse.ArgumentParser(
    prog='eval_4_titration_compare_all_fractions',
    description='Compare direct vs. finetuned SPARC across ALL training-data fractions')
parser.add_argument("-o",  "--option",  type=str,  default='AKI_risk',
                    help='task: AKI_risk, febrile_neutropenia_risk, neutropenia_risk, '
                         'anemia_risk, Liver_toxicity_risk, binary, binary_icu_pre')
parser.add_argument("-ds", "--dataset", type=str,  default='MSK_CAIA')
parser.add_argument("-m",  "--modules", type=str,  default='parsimonious')
parser.add_argument("-r",  "--rseed",   type=int,  default=1024)
parser.add_argument("-mp", "--model_base_path", type=str, default=None,
                    help='root holding the FULL-DATA (1/1) models, i.e. the dir containing '
                         'SPARC_direct_models/ and SPARC_finetuned_models/. '
                         'Default: <SPARC_MODEL_ROOT>/parsimonious_MSK_CAIA_060726')
parser.add_argument("-tr", "--trainsize_base_path", type=str,
                    default='/path/to/SPARC_titration_models',
                    help='root holding the TRAINSIZE models, i.e. the dir containing '
                         'SPARC_direct_models_trainsize/. Defaults to the bf=True titration tree. '
                         'The titration run lives in a separate root, so these differ.')
parser.add_argument("-rp", "--results_path", type=str, default=None,
                    help='where to write the CSV. Default: sparc_paths.SPARC_TITRATION_RESULTS_DIR')
parser.add_argument("-bf", "--batch_first", type=str, default='True',
                    help='True | False -- build SPARC with batch_first=True. MUST match how the '
                         'checkpoints were trained; a mismatch loads silently and costs 4-8 AUROC points.')
parser.add_argument("-cp", "--caia_portion", type=str, default='prior',
                    help="prior | post | entire — which slice of MSK_CAIA to load "
                         "(wrt 2021). MUST match eval_1 (default 'prior'); omitting it "
                         "lets load_dataset fall back to its own default and silently "
                         "changes the test cohort.")
args = parser.parse_args()

option  = args.option
dataset = args.dataset
modules = args.modules
rseed   = args.rseed
caia_portion = args.caia_portion.lower()

# Training-set fractions
STANDARD_FRACTIONS = ['1/1', '1/4', '1/16', '1/64', '1/128', '1/256']

# 1/32 only exists for binary_icu_pre
if option == 'binary_icu_pre':
    ALL_FRACTIONS = ['1/1', '1/4', '1/16', '1/32', '1/64', '1/128', '1/256']
else:
    ALL_FRACTIONS = STANDARD_FRACTIONS

print(f"\nEvaluating task '{option}' across fractions: {ALL_FRACTIONS}")

# Task config
TASK_MED_MAP = {
    'AKI_risk':                   ("CISPLATIN,IFOSFAMIDE,PEMETREXED,METHOTREXATE,PAMIDRONATE,ZOLEDRONIC ACID", True),
    'febrile_neutropenia_risk':   ('DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                                   'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                                   'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL', True),
    'neutropenia_risk':           ('DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                                   'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                                   'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL', True),
    'anemia_risk':                ('DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,'
                                   'IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,'
                                   'BENDAMUSTINE,CARBOPLATIN,ETOPOSIDE,CABAZITAXEL', True),
    'Liver_toxicity_risk':        ("IPILIMUMAB,ATEZOLIZUMAB,NIVOLUMAB,PEMBROLIZUMAB,AVELUMAB,CEMIPLIMAB,DOSTARLIMAB", True),
    'binary':                     ("CISPLATIN", False),
    'binary_icu_pre':             ("CISPLATIN", False),
}

# task -> model subfolder name
TASK_PATH_NAME = {
    'AKI_risk':                 'risk_AKI',
    'febrile_neutropenia_risk': 'hybrid_febrile_neutropenia',
    'neutropenia_risk':         'hybrid_neutropenia',
    'anemia_risk':              'hybrid_anemia',
    'Liver_toxicity_risk':      'risk_Liver_toxicity',
    'binary':                   'binary',
    'binary_icu_pre':           'binary_icu_pre',
}

if option not in TASK_MED_MAP:
    raise ValueError(f"Unknown task: {option}. Choose from: {list(TASK_MED_MAP.keys())}")

med_str, add_num_infusion = TASK_MED_MAP[option]
is_binary_task = option in ('binary', 'binary_icu_pre')

# med string -> list of lists ('+' joins a regimen)
med_of_choice = []
for m in med_str.split(','):
    m = m.strip()
    if '+' in m:
        med_of_choice.append([x.strip() for x in m.split('+')])
    else:
        med_of_choice.append([m])

days = 30 if option == 'binary_icu_pre' else 180

# Paths
data_source_path = sparc_paths.DATA_ROOT + '/'
base_model_path  = args.model_base_path or (sparc_paths.SPARC_MODEL_ROOT + '/parsimonious_MSK_CAIA_060726')
# titration models live in their own root, separate from the 1/1 models
trainsize_base_path = args.trainsize_base_path or base_model_path
print(f"[paths] full-data (1/1) root : {base_model_path}")
print(f"[paths] trainsize root       : {trainsize_base_path}")
task_folder_name = TASK_PATH_NAME[option]

save_dir  = args.results_path or sparc_paths.SPARC_TITRATION_RESULTS_DIR
print(f"[paths] results              : {save_dir}")
os.makedirs(save_dir, exist_ok=True)
save_name = f'{save_dir}/trainsize_comparison_{option}_{dataset}_all_fractions.csv'

# Load dataset (full test set)
print(f"\nLoading dataset: task={option}  dataset={dataset} ...")
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
    dataset=dataset,
    data_source_path=data_source_path,
    option=option,
    modules=modules,
    valfold=0,
    mode='test_all' if 'timeshift' in dataset else 'test',
    rseed=rseed,
    competing_risk_death_check=False,
    include_gcsf=False,
    data_fraction_option='default',
    quick_test=False,
    caia_portion=caia_portion,
)
print("modular_featuredim_dict:", modular_featuredim_dict)

# Model config; must match training and eval_1
use_module_tokenizer = True   # -umt True
use_continuous_time = True    # -uct True
post_mlp_dropout = 0.2        # -pmd 0.2
use_sigmoid = True            # probabilities, not logits
missing_mask = True
schedule_option = 'default'
include_gcsf = False
learning_rate = 1e-4

# Per-module tokenizer config
n_static_features = 0
if use_module_tokenizer:
    module_index_cfg = build_module_index_config(
        modular_featuredim_dict, advanced_model='default'
    )
    n_static_features = module_index_cfg['n_static']
    print(f"Module tokenizer config built: n_static={n_static_features}")
else:
    module_index_cfg = None

# Test split
key_module = 'basic_lab' if 'basic_lab' in data_dict_lab_dict else 'age'

if dataset == 'MSK_CAIA':
    testDMP_IDs = split_dict['test']          # as-is, like eval_1 (no basic_lab filter)

elif dataset == 'MSK':
    with open(f'{data_source_path}/MSK/fold_dict_11.pickle', 'rb') as f:
        fold_dict = pickle.load(f)
    testDMP_IDs = fold_dict[10]
    testDMP_IDs = [pid for pid in testDMP_IDs if pid in data_dict_lab_dict[key_module]]
    split_dict  = {'train': [], 'val': [], 'test': testDMP_IDs}
else:
    raise ValueError(f"Unsupported dataset for fold splitting: {dataset}")

# Data module
model_set_option = 'binary' if is_binary_task else 'side_effects_risk'

# hybrid (Cox + BCE) for FN/neutropenia/anemia, Cox only for AKI/LT
if task_folder_name.startswith('hybrid_'):
    loss_option = 'hybrid'
else:
    loss_option = 'default'
print(f"Task: {option}, loss_option: {loss_option}, model_set_option: {model_set_option}")

# eval_1 loads every checkpoint with loss_option='default' (it only affects the training loss)
load_loss_option = 'default'

# CAIA stores absolute neutrophil counts and needs anc=True (as in eval_1); otherwise the
# neutrophil feature is mis-scaled
anc_mode = True if 'CAIA' in dataset else False

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict, dataset=dataset, component=component,
    batch_size=512, max_time_len=20, num_workers=1,
    task=option, binary_days=days, med=med_of_choice,
    med_as_feature=False, med_l=med_of_choice,
    advanced_model='default', control_lack_confounds=False,
    side_effect_censoring=False,
    include_infusion_dates=add_num_infusion,  # False for binary tasks
    premed_only=False, competing_risk_death_check=False,
    anc=anc_mode,
)
print(f"Data module ready — test patients: {len(testDMP_IDs)}  (anc={anc_mode})")

# hash of the test IDs, to check the cohort matches eval_1
import hashlib
_id_hash = hashlib.md5(
    ','.join(map(str, sorted(map(str, testDMP_IDs)))).encode()
).hexdigest()
print(f"TEST COHORT FINGERPRINT: n={len(testDMP_IDs)}  md5={_id_hash}  "
      f"caia_portion={caia_portion}")

# Resume from existing results
model_performance_dict = {}  # {model_name: {fraction: {fold: metrics}}}
if os.path.exists(save_name):
    print(f"\nResuming from existing results: {save_name}")
    saved_df = pd.read_csv(save_name)
    key_cols = (['auroc', 'auprc'] if is_binary_task
                else ['test_cindex_overall_valid', 'test_cindex_weighted'])
    key_cols = [c for c in key_cols if c in saved_df.columns]
    perf_df_existing = saved_df.dropna(subset=key_cols) if key_cols else saved_df.dropna()

    for _, row in perf_df_existing.iterrows():
        mn = row['model']
        frac = row['data_fraction']
        vf = int(row['val_fold'])

        if mn not in model_performance_dict:
            model_performance_dict[mn] = {}
        if frac not in model_performance_dict[mn]:
            model_performance_dict[mn][frac] = {}

        if is_binary_task:
            model_performance_dict[mn][frac][vf] = {
                'auroc': row.get('auroc'),
                'auprc': row.get('auprc'),
            }
        else:
            model_performance_dict[mn][frac][vf] = {
                'test_cindex_overall_valid':  row.get('test_cindex_overall_valid'),
                'test_cindex_weighted':       row.get('test_cindex_weighted'),
                'test_auroc_nextinf_overall': row.get('test_auroc_nextinf_overall'),
                'test_auroc_nextinf_weighted':row.get('test_auroc_nextinf_weighted'),
            }

# Save results to CSV
ALL_COLUMNS = ['model', 'training_method', 'data_fraction', 'val_fold',
               'test_cindex_overall_valid', 'test_cindex_weighted',
               'test_auroc_nextinf_overall', 'test_auroc_nextinf_weighted',
               'auroc', 'auprc']

def _save_results():
    rows = []
    for mn, fractions in model_performance_dict.items():
        training_method = 'Finetuned (SPARC-P)' if 'finetuned' in mn else 'Direct (SPARC)'
        for frac, folds in fractions.items():
            for vf, metrics in folds.items():
                row = {
                    'model': mn,
                    'training_method': training_method,
                    'data_fraction': frac,
                    'val_fold': vf
                }
                row.update(metrics)
                rows.append({col: row.get(col) for col in ALL_COLUMNS})
    pd.DataFrame(rows, columns=ALL_COLUMNS).to_csv(save_name, index=False)
    print(f"  Saved → {save_name}")

# Evaluate each fraction
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auroc_fn = AUROC(task='binary')
auprc_fn = AveragePrecision(task='binary')

all_folds = list(range(10))

for data_fraction in ALL_FRACTIONS:
    print(f"\n{'#'*80}")
    print(f"# DATA FRACTION: {data_fraction}")
    print(f"{'#'*80}")

    # 1/1 models are in the non-trainsize directories
    if data_fraction == '1/1':
        direct_model_path = (
            f'{base_model_path}/SPARC_direct_models/'
            f'parsimonious_direct_models_{task_folder_name}_20tp_topchemos_att_modular'
        )
        finetuned_model_path = (
            f'{base_model_path}/SPARC_finetuned_models/'
            f'parsimonious_finetuned_models_{task_folder_name}_20tp_topchemos_att_modular'
        )
    else:
        # 1/4 -> 1over4
        frac_denom = data_fraction.split('/')[1]
        frac_tag   = f'1over{frac_denom}'

        direct_model_path = (
            f'{trainsize_base_path}/SPARC_direct_models_trainsize/'
            f'parsimonious_direct_models_{task_folder_name}_20tp_topchemos_att_modular_trainsize_{frac_tag}'
        )
        finetuned_model_path = (
            f'{trainsize_base_path}/SPARC_finetuned_models_trainsize/'
            f'parsimonious_finetuned_models_{task_folder_name}_20tp_topchemos_att_modular_trainsize_{frac_tag}'
        )

    model_configs = {
        'SPARC_direct': {
            'path':        direct_model_path,
            'description': f'Direct (no pretrain), frac={data_fraction}',
        },
        'SPARC_finetuned': {
            'path':        finetuned_model_path,
            'description': f'Finetuned (pretrained init), frac={data_fraction}',
        },
    }

    for model_name, config in model_configs.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}  |  {config['description']}")
        print(f"Path:  {config['path']}")
        print(f"{'='*70}")

        if not os.path.isdir(config['path']):
            print(f"  Directory not found: {config['path']}, skipping this fraction/method.")
            continue

        if model_name not in model_performance_dict:
            model_performance_dict[model_name] = {}
        if data_fraction not in model_performance_dict[model_name]:
            model_performance_dict[model_name][data_fraction] = {}

        if all(vf in model_performance_dict[model_name][data_fraction] for vf in all_folds):
            print("  All folds already evaluated for this fraction, skipping.")
            continue

        for val_fold in all_folds:
            if val_fold in model_performance_dict[model_name].get(data_fraction, {}):
                print(f"  Fold {val_fold}: already done, skipping.")
                continue

            model_folder = f'{config["path"]}/fold{val_fold}'
            if not os.path.isdir(model_folder):
                print(f"  Fold {val_fold}: directory not found ({model_folder}), skipping.")
                continue

            # same checkpoint pick as eval_1: first listdir entry containing 'epoch'
            # (with several checkpoints, a different rule would pick a different epoch)
            ckpts = [f for f in os.listdir(model_folder) if 'epoch' in f]

            # fallback: version directories from restarts
            if not ckpts:
                log_dir = f'{model_folder}/logs_csv/lightning_logs_csv'
                if os.path.exists(log_dir):
                    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
                    for version_dir in sorted(version_dirs, reverse=True):
                        version_path = f'{log_dir}/{version_dir}'
                        if os.path.isdir(version_path):
                            ckpts = [f for f in os.listdir(version_path) if 'epoch' in f]
                            if ckpts:
                                model_folder = version_path
                                break

            if not ckpts:
                print(f"  Fold {val_fold}: no checkpoint found, skipping.")
                continue

            if len(ckpts) > 1:
                print(f"  Fold {val_fold}: WARNING {len(ckpts)} epoch checkpoints present "
                      f"{ckpts}; using [0]={ckpts[0]} (matches eval_1).")

            ckpt_path = os.path.join(model_folder, ckpts[0])
            print(f"  Fold {val_fold}: loading {ckpt_path}")

            pl.seed_everything(rseed)

            try:
                model = SPARC.load_from_checkpoint(
                    ckpt_path,
                    modular_featuredim_dict=modular_featuredim_dict,
                    option=model_set_option,
                    advanced_model='default',
                    max_time_len=20,
                    med_as_feature=False,
                    meds_l=med_of_choice,
                    side_effects_num_infusion=add_num_infusion,
                    temporal_aggregation='attention',
                    loss_option=load_loss_option,  # as in eval_1
                    schedule=schedule_option,
                    include_gcsf=include_gcsf,
                    use_gcsf=include_gcsf,
                    use_sigmoid=use_sigmoid,
                    missing_mask=missing_mask,
                    lr=learning_rate,
                    module_index_cfg=module_index_cfg,
                    use_module_tokenizer=use_module_tokenizer,
                    use_continuous_time=use_continuous_time,
                    n_static_features=n_static_features,
                    post_mlp_dropout=post_mlp_dropout,
                    batch_first=(args.batch_first == 'True'),
                )
            except Exception as e:
                print(f"  Fold {val_fold}: failed to load checkpoint — {e}")
                print(f"    This may be expected for extreme fractions where training failed.")
                continue

            model = model.to(device)
            model.eval()

            all_y_preds, all_y_events, all_time_to_events, all_num_infusions = [], [], [], []
            pl.seed_everything(rseed)
            try:
                with torch.no_grad():
                    for batch in data_module.test_dataloader():
                        xs_combined, y, mask, _, num_infusion, _ = model.data_prep(
                            batch, time_calibration=model.time_calibration,
                        )
                        if num_infusion is None:
                            num_infusion = torch.zeros(xs_combined.shape[0], dtype=torch.float32)

                        xs_combined  = xs_combined.to(device)
                        mask         = mask.to(device)
                        num_infusion = num_infusion.to(device)

                        y_pred = model(xs_combined, src_key_padding_mask=mask,
                                       num_infusion=num_infusion, gcsf_groundtruth=None)

                        if is_binary_task:
                            fail_indicators = torch.as_tensor(y, device=device)
                            time_to_event   = torch.ones_like(fail_indicators).float()
                        else:
                            fail_indicators = torch.as_tensor(y[0], device=device)
                            time_to_event   = torch.as_tensor(y[1], device=device)

                        if fail_indicators.dim() == 0:
                            fail_indicators = fail_indicators.unsqueeze(0)
                        if time_to_event.dim() == 0:
                            time_to_event = time_to_event.unsqueeze(0)

                        all_y_preds.append(y_pred.detach().cpu())
                        all_y_events.append(fail_indicators.detach().cpu())
                        all_time_to_events.append(time_to_event.detach().cpu())
                        all_num_infusions.append(num_infusion.detach().cpu())

                all_y_preds        = torch.cat(all_y_preds,        dim=0)
                all_y_events       = torch.cat(all_y_events,       dim=0)
                all_time_to_events = torch.cat(all_time_to_events, dim=0)
                all_num_infusions  = torch.cat(all_num_infusions,  dim=0)

                print("PRED:", all_y_preds.shape, all_y_preds[:5].tolist(),
                    "sum=", all_y_preds.sum().item())
                print("EVT :", all_y_events[:5].tolist(), "sum=", all_y_events.sum().item())
                print("TTE :", all_time_to_events[:5].tolist())
                print("INF :", all_num_infusions[:5].tolist(), "uniq=", len(torch.unique(all_num_infusions)))

                if is_binary_task:
                    auroc = auroc_fn(all_y_preds, all_y_events.int()).item()
                    auprc = auprc_fn(all_y_preds, all_y_events.int()).item()
                    model_performance_dict[model_name][data_fraction][val_fold] = {
                        'auroc': auroc,
                        'auprc': auprc
                    }
                    print(f"    AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
                else:
                    c_idx_dict = calculate_weighted_c_index_by_infusion(
                        all_y_preds, all_y_events, all_time_to_events, all_num_infusions,
                        mode='val', option=option,
                    )
                    model_performance_dict[model_name][data_fraction][val_fold] = {
                        'test_cindex_overall_valid':  c_idx_dict.get('c_index_overall'),
                        'test_cindex_weighted':       c_idx_dict.get('c_index_weighted'),
                        'test_auroc_nextinf_overall': c_idx_dict.get('auroc_next_infusion_overall'),
                        'test_auroc_nextinf_weighted':c_idx_dict.get('auroc_next_infusion_weighted'),
                    }
                    print(f"    C-index overall={c_idx_dict.get('c_index_overall'):.4f}  "
                          f"weighted={c_idx_dict.get('c_index_weighted'):.4f}")
                    print(f"    AUROC-next overall={c_idx_dict.get('auroc_next_infusion_overall'):.4f}  "
                          f"weighted={c_idx_dict.get('auroc_next_infusion_weighted'):.4f}")

                _save_results()

            except Exception as e:
                print(f"  Fold {val_fold}: evaluation failed — {e}")
                print(f"    This may be expected for extreme fractions.")
                continue

# Summary across fractions
import numpy as np

print("\n" + "="*80)
print(f"SUMMARY ACROSS ALL FRACTIONS  |  task={option}")
print("="*80)

primary_metrics = (['auroc', 'auprc'] if is_binary_task else
                   ['test_auroc_nextinf_weighted', 'test_cindex_weighted'])

for data_fraction in ALL_FRACTIONS:
    print(f"\n{'-'*80}")
    print(f"Data Fraction: {data_fraction}")
    print(f"{'-'*80}")

    for model_name in ['SPARC_direct', 'SPARC_finetuned']:
        if model_name not in model_performance_dict:
            continue
        if data_fraction not in model_performance_dict[model_name]:
            continue

        folds_data = model_performance_dict[model_name][data_fraction]
        method_label = 'Direct (SPARC)' if 'direct' in model_name else 'Finetuned (SPARC-P)'
        print(f"\n  {method_label}")

        for metric in primary_metrics:
            vals = [v[metric] for v in folds_data.values()
                    if isinstance(v, dict) and v.get(metric) is not None]
            if vals:
                print(f"    {metric:40s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")
            else:
                print(f"    {metric:40s}  — (no data)")

    # finetuned - direct
    direct_folds   = model_performance_dict.get('SPARC_direct', {}).get(data_fraction, {})
    finetune_folds = model_performance_dict.get('SPARC_finetuned', {}).get(data_fraction, {})
    shared_folds   = sorted(set(direct_folds) & set(finetune_folds))

    if shared_folds:
        print(f"\n  Delta (Finetuned − Direct):")
        for metric in primary_metrics:
            deltas = [
                finetune_folds[vf].get(metric, None) - direct_folds[vf].get(metric, None)
                for vf in shared_folds
                if finetune_folds[vf].get(metric) is not None
                and direct_folds[vf].get(metric) is not None
            ]
            if deltas:
                print(f"    {metric:40s}  Δ={np.mean(deltas):+.4f} ± {np.std(deltas):.4f}  (n={len(deltas)})")
            else:
                print(f"    {metric:40s}  — (no paired data)")

print("\n" + "="*80)
print("Done. All fractions evaluated.")
print(f"Results saved to: {save_name}")
print("="*80)