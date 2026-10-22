#!/usr/bin/env python3
"""Evaluate MSK-trained (zero-shot) or INSIGHT-fine-tuned (--custom_model_path) SPARC / SPARC-P, 10 folds and their
mean ensemble, on INSIGHT. -ps True evaluates the pooled dataset and writes one result per site.
Without split files every patient is scored.
Usage: python eval_insight.py -o neutropenia -me "DOCETAXEL,..." -an True -dsp <site or root> -rp <out> [-ps True]
"""
# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models'), os.path.join(_R, 'results_vis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torchmetrics import AUROC, AveragePrecision

from data_loaders_inf_risk_fixed import FlexibleFeatureDataModule
from dataset_loading import load_dataset
from SPARC_inf_risk_modular import SPARC
from MSK_featurename_list import build_module_index_config
from eval_utils import calculate_weighted_c_index_by_infusion

# task -> model folder suffix
OPTION_TO_TASK_DIR = {
    'binary': 'binary', 'binary_icu_pre': 'binary_icu_pre', 'AKI': 'risk_AKI',
    'Liver_toxicity': 'risk_Liver_toxicity', 'anemia': 'hybrid_anemia',
    'febrile_neutropenia': 'hybrid_febrile_neutropenia', 'neutropenia': 'hybrid_neutropenia'}
SIDE_EFFECT_OPTIONS = ['AKI', 'Liver_toxicity', 'anemia', 'febrile_neutropenia', 'neutropenia']
SITES = ['A', 'B', 'C', 'D', 'E', 'F']
SE_METRIC_COLS = ['test_cindex_overall_valid', 'test_cindex_weighted', 'test_auroc_nextinf_overall',
                  'test_auroc_nextinf_weighted', 'test_auprc_nextinf_overall', 'test_auprc_nextinf_weighted']
CI_COLS_SE = ['auroc_ci_lower', 'auroc_ci_upper', 'auprc_ci_lower', 'auprc_ci_upper',
              'auroc_weighted_ci_lower', 'auroc_weighted_ci_upper']
CI_COLS_BIN = ['auroc_ci_lower', 'auroc_ci_upper', 'auprc_ci_lower', 'auprc_ci_upper']

parser = argparse.ArgumentParser(description='Evaluate SPARC / SPARC-P on INSIGHT')
parser.add_argument('-r', '--rseed', type=int, default=1024)
parser.add_argument('-o', '--option', type=str, required=True, choices=list(OPTION_TO_TASK_DIR))
parser.add_argument('-d', '--days', type=int, default=180, help='days for the mortality tasks')
parser.add_argument('-me', '--med_of_choice', type=str, default='CISPLATIN',
                    help='comma-separated regimens, + joins drugs in a combination')
parser.add_argument('-an', '--add_num_infusion', type=str, default='False')
parser.add_argument('-mm', '--missing_mask', type=str, default='True')
parser.add_argument('-dsp', '--data_source_path', type=str, default=sparc_paths.INSIGHT_DATA_DIR,
                    help='one site folder (-ps False) or the root holding A-F and split/ (-ps True)')
parser.add_argument('-mp', '--model_path', type=str,
                    default=os.path.join(sparc_paths.SPARC_MODEL_ROOT, 'parsimonious_MSK_CAIA_060726'),
                    help='root holding SPARC_finetuned_models/ and SPARC_direct_models/ (MSK-trained models)')
parser.add_argument('-cmp', '--custom_model_path', type=str, default='',
                    help='fine-tuned models: folder with fold0..fold9 (overrides -mp for SPARC-P)')
parser.add_argument('-rp', '--results_path', type=str,
                    default=os.path.join(sparc_paths.INSIGHT_RESULTS_DIR, 'eval_insight'))
parser.add_argument('--models', nargs='+', default=['SPARC-P'], choices=['SPARC-P', 'SPARC'])
parser.add_argument('-en', '--ensemble', type=str, default='True', help='also score the mean of the fold predictions')
parser.add_argument('-bf', '--batch_first', type=str, default='True',
                    help='must match training (all published INSIGHT runs use batch_first=True models)')
parser.add_argument('-ps', '--per_site', type=str, default='False')
parser.add_argument('-nb', '--n_bootstrap', type=int, default=1000, help='bootstrap resamples for ensemble CIs; 0 = none')
parser.add_argument('--n_folds', type=int, default=10, help='evaluate and ensemble starting models 0..n_folds-1')
args = parser.parse_args()

rseed = args.rseed
batch_first = args.batch_first.lower() == 'true'
option_orig = option = args.option
add_num_infusion = args.add_num_infusion.lower() == 'true'
missing_mask = args.missing_mask.lower() == 'true'
data_source_path, results_path = args.data_source_path, args.results_path
ensemble = args.ensemble.lower() == 'true'
per_site = args.per_site.lower() == 'true'
med_of_choice = [[x.strip() for x in m.split('+')] for m in (s.strip() for s in args.med_of_choice.split(','))]

# architecture, as in training
modules, batch_size, include_gcsf = 'parsimonious', 512, False
use_module_tokenizer, use_continuous_time, post_mlp_dropout = True, True, 0.2
use_sigmoid, schedule_option, learning_rate = True, 'default', 1e-4

dataset_type = 'insight_pooled' if per_site else 'insight'
data_dict_lab_dict, modular_featuredim_dict, split_dict, component, _ = load_dataset(
    dataset=dataset_type, data_source_path=data_source_path, option=option, modules=modules, valfold=0,
    mode='test', rseed=rseed, competing_risk_death_check=False, include_gcsf=False,
    data_fraction_option='default', quick_test=False)
print(f"split sizes: train {len(split_dict['train'])}, val {len(split_dict['val'])}, test {len(split_dict['test'])}")

patient_to_site = {}
if per_site:
    for site in SITES:
        p = os.path.join(data_source_path, 'split', 'test', f'{site}.json')
        if os.path.exists(p):
            with open(p) as f:
                patient_to_site.update({str(pid): site for pid in json.load(f)})
    print(f'per-site mode: {len(patient_to_site)} test patients mapped to sites')

module_index_cfg = build_module_index_config(modular_featuredim_dict, advanced_model='default')
n_static_features = module_index_cfg['n_static']

is_side_effect = option in SIDE_EFFECT_OPTIONS
if is_side_effect:
    model_set_option, option = 'side_effects_risk', f'{option}_risk'
else:
    model_set_option = 'binary'

task_dir = OPTION_TO_TASK_DIR[option_orig]
model_folder_dict = {
    'SPARC-P': args.custom_model_path or
               f'{args.model_path}/SPARC_finetuned_models/parsimonious_finetuned_models_{task_dir}_20tp_topchemos_att_modular',
    'SPARC': f'{args.model_path}/SPARC_direct_models/parsimonious_direct_models_{task_dir}_20tp_topchemos_att_modular'}
print('model folders:', {m: model_folder_dict[m] for m in args.models})

model_tag = args.models[0] if len(args.models) == 1 else 'multi'
save_name = f'{results_path}/eval_across_models_insight/model_performance_dict_parsimonious_{option_orig}_{model_tag}_insight.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)


def cindex_metrics(preds, events, times, num_inf):
    d = calculate_weighted_c_index_by_infusion(preds, events, times, num_inf, mode='val', option=option,
                                               extended_metrics=True)
    return {'test_cindex_overall_valid': d.get('c_index_overall', 0.5),
            'test_cindex_weighted': d.get('c_index_weighted', 0.5),
            'test_auroc_nextinf_overall': d.get('auroc_next_infusion_overall'),
            'test_auroc_nextinf_weighted': d.get('auroc_next_infusion_weighted'),
            'test_auprc_nextinf_overall': d.get('auprc_next_infusion_overall'),
            'test_auprc_nextinf_weighted': d.get('auprc_next_infusion_weighted')}


def binary_metrics(preds, events):
    return {'auroc': AUROC(task='binary')(preds, events.int()).item(),
            'auprc': AveragePrecision(task='binary')(preds, events.int()).item()}


def bootstrap_ci(preds, events, times, num_inf, side_effect, n_boot, seed=1024):
    """95% percentile-bootstrap CIs for an ensemble row, resampling test rows.
    Side-effect tasks use only infusion groups with both classes, re-checked in each resample."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    if n_boot <= 0:
        return {}
    to_np = lambda x: x.detach().cpu().numpy().ravel() if hasattr(x, 'detach') else np.asarray(x).ravel()
    p = to_np(preds).astype(float)
    if side_effect:
        y = ((to_np(events) == 1) & (to_np(times) == 1)).astype(int)
        g = to_np(num_inf)
        keep = np.zeros(len(y), bool)
        for u in np.unique(g):
            m = g == u
            if 0 < y[m].sum() < m.sum():
                keep |= m
        if not keep.any():
            return {}
        p, y, g = p[keep], y[keep], g[keep]
    else:
        y, g = to_np(events).astype(int), None
    rng = np.random.default_rng(seed)
    au, ap, aw = [], [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        yb, pb = y[ix], p[ix]
        if len(np.unique(yb)) < 2:
            continue
        au.append(roc_auc_score(yb, pb)); ap.append(average_precision_score(yb, pb))
        if g is not None:
            gb = g[ix]; num = den = 0.0
            for u in np.unique(gb):
                m = gb == u
                if 0 < yb[m].sum() < m.sum():
                    num += roc_auc_score(yb[m], pb[m]) * m.sum(); den += m.sum()
            if den > 0:
                aw.append(num / den)
    out = {}
    if au:
        out.update(auroc_ci_lower=np.percentile(au, 2.5), auroc_ci_upper=np.percentile(au, 97.5),
                   auprc_ci_lower=np.percentile(ap, 2.5), auprc_ci_upper=np.percentile(ap, 97.5))
    if aw:
        out.update(auroc_weighted_ci_lower=np.percentile(aw, 2.5), auroc_weighted_ci_upper=np.percentile(aw, 97.5))
    return out


def test_loader(dm):
    """Test loader in dataset order, so rows line up with test_dataset.dataset_DMP_IDs."""
    dl = dm.test_dataloader()
    if not batch_first:
        return dl
    return torch.utils.data.DataLoader(dl.dataset, batch_size=dl.batch_size, shuffle=False, num_workers=0,
                                       collate_fn=dl.collate_fn)


def upsert_row(path, row, cols):
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
    df = df[~((df['model'] == row['model']) & (df['val_fold'].astype(str) == str(row['val_fold'])))]
    pd.concat([df, pd.DataFrame([[row.get(c) for c in cols]], columns=cols)], ignore_index=True).to_csv(path, index=False)


def site_csv(site):
    p = os.path.join(results_path, site, 'eval_across_models_insight',
                     f'model_performance_dict_parsimonious_{option_orig}_{model_tag}_insight.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


all_columns = (['model', 'val_fold'] + SE_METRIC_COLS + ['auroc', 'auprc'] + CI_COLS_SE if is_side_effect
               else ['model', 'val_fold', 'auroc', 'auprc'] + CI_COLS_BIN)
fold_columns = (['model', 'val_fold'] + SE_METRIC_COLS + ['auroc', 'auprc'] if is_side_effect
                else ['model', 'val_fold', 'auroc', 'auprc'])

# resume from an existing CSV (fold rows only; the ensemble is rebuilt from the fold predictions)
model_performance_dict, failed_folds, fold_preds_store = {}, [], {}
if os.path.exists(save_name):
    saved = pd.read_csv(save_name)
    saved = saved[~saved['model'].str.contains('averaged|_ensemble', na=False, regex=True)
                  & (saved['val_fold'].astype(str) != 'ensemble')]
    for _, row in saved.iterrows():
        vf = int(row['val_fold']) if str(row['val_fold']).lstrip('-').isdigit() else str(row['val_fold'])
        keys = SE_METRIC_COLS if is_side_effect else ['auroc', 'auprc']
        model_performance_dict.setdefault(row['model'], {})[vf] = {k: row.get(k) for k in keys}
    print(f'resuming from {save_name}: {len(saved)} fold rows')

all_folds = list(range(args.n_folds))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_option in args.models:
    done = model_option in model_performance_dict and all(f in model_performance_dict[model_option] for f in all_folds)
    if done and ((not ensemble) or f'{model_option}_ensemble' in model_performance_dict):
        print(f'{model_option}: all folds and the ensemble already done')
        continue
    pl.seed_everything(rseed)
    data_module = FlexibleFeatureDataModule(
        data_dict_lab_dict, split_dict, dataset=dataset_type, component=component,
        batch_size=batch_size, max_time_len=20, num_workers=1, task=option, binary_days=args.days,
        med=med_of_choice, med_as_feature=False, med_l=med_of_choice, advanced_model='default',
        control_lack_confounds=False, side_effect_censoring=False, include_infusion_dates=add_num_infusion,
        premed_only=False, competing_risk_death_check=False, anc=True,   # INSIGHT labs hold absolute counts
        trim_test_last_batch=False)   # score every test window, as in the published INSIGHT results
    test_ids = [str(p) for p in data_module.test_dataset.dataset_DMP_IDs]
    site_masks = ({s: torch.tensor([patient_to_site.get(p) == s for p in test_ids], dtype=torch.bool) for s in SITES}
                  if per_site and patient_to_site else {})
    print(f'\n{"=" * 60}\nModel: {model_option}\n{"=" * 60}')

    ensemble_already_done = f'{model_option}_ensemble' in model_performance_dict
    for val_fold in all_folds:
        pl.seed_everything(rseed)
        perf = model_performance_dict.setdefault(model_option, {})
        if val_fold in perf:
            if not ensemble or ensemble_already_done:
                continue
            print(f'  fold {val_fold}: done, re-running inference for the ensemble')
        perf[val_fold] = {}
        model_folder = f'{model_folder_dict[model_option]}/fold{val_fold}'
        try:
            ckpts = [f for f in os.listdir(model_folder) if 'epoch' in f]
            if not ckpts:
                print(f'  no checkpoint in {model_folder}, skipped')
                del perf[val_fold]
                continue
            model = SPARC.load_from_checkpoint(
                os.path.join(model_folder, ckpts[0]), modular_featuredim_dict=modular_featuredim_dict,
                option=model_set_option, advanced_model='default', max_time_len=20, med_as_feature=False,
                meds_l=med_of_choice, loss_option='default', schedule=schedule_option,
                side_effects_num_infusion=add_num_infusion, include_gcsf=include_gcsf, use_gcsf=include_gcsf,
                use_sigmoid=use_sigmoid, temporal_aggregation='attention', missing_mask=missing_mask,
                module_index_cfg=module_index_cfg, use_module_tokenizer=use_module_tokenizer,
                use_continuous_time=use_continuous_time, n_static_features=n_static_features,
                post_mlp_dropout=post_mlp_dropout, batch_first=batch_first, lr=learning_rate)
        except Exception as e:
            print(f'  ERROR loading {model_option} fold {val_fold}: {e}')
            failed_folds.append(f'{model_option} fold {val_fold}')
            del perf[val_fold]
            continue
        model = model.to(device).eval()
        print(f'  fold {val_fold}: {model_folder}')

        preds, events, times, ninf = [], [], [], []
        with torch.no_grad():
            for batch in test_loader(data_module):
                xs, y, mask, _, num_infusion, _ = model.data_prep(batch, time_calibration=model.time_calibration)
                xs, mask = xs.to(device), mask.to(device)
                if add_num_infusion and num_infusion is not None:
                    num_infusion = num_infusion.to(device)
                p = model(xs, src_key_padding_mask=mask, num_infusion=num_infusion, gcsf_groundtruth=None)
                if num_infusion is None:
                    num_infusion = torch.zeros(p.shape[0], dtype=torch.float32, device=device)
                ev = torch.as_tensor(y[0] if is_side_effect else y, device=device)
                tt = torch.as_tensor(y[1], device=device) if is_side_effect else torch.ones_like(ev).float()
                ev, tt = (ev.unsqueeze(0) if ev.dim() == 0 else ev), (tt.unsqueeze(0) if tt.dim() == 0 else tt)
                preds.append(p.detach().cpu()); events.append(ev.detach().cpu())
                times.append(tt.detach().cpu()); ninf.append(num_infusion.detach().cpu())
        preds, events, times, ninf = (torch.cat(x, dim=0) for x in (preds, events, times, ninf))

        for site, m in site_masks.items():
            if int(m.sum()) == 0:
                continue
            row = {'model': model_option, 'val_fold': val_fold}
            row.update(cindex_metrics(preds[m], events[m], times[m], ninf[m]) if is_side_effect
                       else binary_metrics(preds[m], events[m]))
            upsert_row(site_csv(site), row, fold_columns)
        if ensemble:
            fold_preds_store.setdefault(model_option, {})[val_fold] = {
                'preds': preds.clone(), 'events': events.clone(), 'times': times.clone(), 'num_inf': ninf.clone()}
        perf[val_fold].update(cindex_metrics(preds, events, times, ninf) if is_side_effect
                              else binary_metrics(preds, events))
        print(f'  samples {len(events)}, positives {(events == 1).sum().item()}, {perf[val_fold]}')

        # ensemble = mean of the fold predictions so far (rewritten after every fold)
        if ensemble and fold_preds_store.get(model_option):
            stored = fold_preds_store[model_option]
            ens_preds = torch.stack([stored[f]['preds'].float() for f in sorted(stored)]).mean(dim=0)
            ref = stored[sorted(stored)[0]]
            ens_events, ens_times, ens_ninf = ref['events'], ref['times'], ref['num_inf']
            ens_key = f'{model_option}_ensemble'
            ens = (cindex_metrics(ens_preds, ens_events, ens_times, ens_ninf) if is_side_effect
                   else binary_metrics(ens_preds, ens_events))
            ens.update(bootstrap_ci(ens_preds, ens_events, ens_times, ens_ninf, is_side_effect, args.n_bootstrap))
            model_performance_dict.setdefault(ens_key, {})['ensemble'] = ens

            def save_ens_preds(root, m=None):
                sel = (lambda t: t[m]) if m is not None else (lambda t: t)
                n = int(len(sel(ens_preds)))
                d = {'y_preds': sel(ens_preds).cpu().numpy().ravel(),
                     'y_events': sel(ens_events).cpu().numpy().ravel(),
                     'time_to_events': sel(ens_times).cpu().numpy().ravel() if is_side_effect else np.ones(n),
                     'num_infusions': sel(ens_ninf).cpu().numpy().ravel() if is_side_effect else np.zeros(n),
                     'patient_ids': np.array(test_ids)[m.numpy()] if m is not None else np.array(test_ids),
                     'n_folds': len(stored)}
                assert len(d['patient_ids']) == len(d['y_preds']), 'patient IDs not row-aligned'
                out = os.path.join(root, 'ensemble_predictions', f'{option_orig}_{model_tag}_ensemble_preds.pickle')
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, 'wb') as f:
                    pickle.dump(d, f)

            if not per_site:
                save_ens_preds(results_path)
            for site, m in site_masks.items():
                if int(m.sum()) == 0:
                    continue
                save_ens_preds(os.path.join(results_path, site), m)
                row = {'model': ens_key, 'val_fold': 'ensemble'}
                if is_side_effect:
                    row.update(cindex_metrics(ens_preds[m], ens_events[m], ens_times[m], ens_ninf[m]))
                else:
                    row.update(binary_metrics(ens_preds[m], ens_events[m]))
                row.update(bootstrap_ci(ens_preds[m], ens_events[m], ens_times[m], ens_ninf[m], is_side_effect,
                                        args.n_bootstrap))
                upsert_row(site_csv(site), row, all_columns)

        # save after every fold
        rows = [[{'model': mn, 'val_fold': f, **fd}.get(c) for c in all_columns]
                for mn, folds in model_performance_dict.items() for f, fd in folds.items()]
        pd.DataFrame(rows, columns=all_columns).to_csv(save_name, index=False)
        print(f'  saved {save_name}')

print(f'\nDone: {save_name}')
if failed_folds:
    print(f'FAILED folds ({len(failed_folds)}): {failed_folds}; rerun (the script resumes)')
    sys.exit(1)
