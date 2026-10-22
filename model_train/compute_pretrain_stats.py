#!/usr/bin/env python3
"""Pretraining stats per fold (1-10, as the training script's -f), averaged across folds: training windows, mean real
timepoints per window, mean non-NaN dynamic measurements per window. Loads data as SPARC_train_inf_risk_fixed.py does.

Usage: python3 compute_pretrain_stats.py -ds MSK_CAIA -n 20 -m parsimonious --caia_portion <as in training> -umt True -uct True
"""

import os
import sys
import argparse

import numpy as np
import torch
import pytorch_lightning as pl

# put repo root, data/ and models/ on sys.path
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loaders_inf_risk_fixed import FlexibleFeatureDataModule
from dataset_loading import load_dataset


# Per-window stat extraction
def _real_dates_mask(x_dates):
    """Mask of real (non-padding) dates; _finalize_output pads with NaN."""
    if isinstance(x_dates, torch.Tensor):
        d = x_dates.detach().cpu().numpy()
    else:
        d = np.asarray(x_dates, dtype=float)
    return ~np.isnan(d)


def _timepoints_and_measurements_for_window(x_ts, x_dates, n_static):
    """(num_timepoints, num_measurements) for one window: real (non-padded) dates and non-NaN
    dynamic values, excluding the first n_static columns of x_ts [T, F]."""
    if isinstance(x_ts, (tuple, list)):
        # raindrop packs (ts_tensor, static_out, seq_len); take the tensor
        x_ts = x_ts[0]
    ts = x_ts.detach().cpu().numpy() if isinstance(x_ts, torch.Tensor) else np.asarray(x_ts, dtype=float)

    real = _real_dates_mask(x_dates)
    num_timepoints = int(real.sum())

    ts_real = ts[real]
    if n_static > 0:
        dynamic = ts_real[:, n_static:]
    else:
        dynamic = ts_real
    num_measurements = int(np.count_nonzero(~np.isnan(dynamic)))

    return num_timepoints, num_measurements


def stats_for_dataset(dataset, n_static):
    """Return (n_windows, tp_per_window, meas_per_window)."""
    x_ts_list = dataset.X_ts
    x_dates_list = dataset.X_dates

    n_windows = len(x_ts_list)
    tp_per_window = []
    meas_per_window = []
    for x_ts, x_dates in zip(x_ts_list, x_dates_list):
        tp, meas = _timepoints_and_measurements_for_window(x_ts, x_dates, n_static)
        tp_per_window.append(tp)
        meas_per_window.append(meas)

    return n_windows, tp_per_window, meas_per_window


# data-relevant args of the training script
def build_parser():
    p = argparse.ArgumentParser(description='Compute per-fold pretraining window stats for SPARC.')
    p.add_argument("-ds", "--dataset", type=str, default='MSK_CAIA')
    p.add_argument("--data_source_path", type=str, default='')
    p.add_argument("-m", "--modules_cat", type=str, default='parsimonious',
                   help='all | all_labs | mutation_clinical | practical | parsimonious')
    p.add_argument("-n", "--num_tp", type=int, default=20, help='max sequence length (max_time_len)')
    p.add_argument("-r", "--rseed", type=int, default=1024)
    p.add_argument("--folds", type=str, default='1-10',
                   help="folds to average over, e.g. '1-10' or '1,2,3'")
    # flags that can influence loading / feature dims
    p.add_argument("-crd", "--competing_risk_death_check", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--include_gcsf", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("-df", "--data_fraction_option", type=str, default='default',
                   help='default | 1/5 | 1/25 | 1/125')
    p.add_argument("--quick_test", type=lambda x: str(x).lower() == 'true', default=False)
    p.add_argument("--caia_portion", type=str, default=None)
    p.add_argument("-umt", "--use_module_tokenizer", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("-uct", "--use_continuous_time", type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument("--advanced_model", type=str, default='default')
    p.add_argument("--batch_size", type=int, default=512)
    return p


def parse_folds(spec):
    spec = spec.strip()
    if '-' in spec and ',' not in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',') if x.strip()]


def main():
    args = build_parser().parse_args()
    folds = parse_folds(args.folds)
    option = 'pretrain'

    # one value per fold
    fold_n_windows = []
    fold_avg_tp = []
    fold_avg_meas = []

    for valfold in folds:
        print()
        print(f"FOLD {valfold}")
        print()

        pl.seed_everything(args.rseed)

        data_dict_lab_dict, modular_featuredim_dict, split_dict, component, include_infusion_dates = load_dataset(
            dataset=args.dataset,
            data_source_path=args.data_source_path,
            option=option,
            modules=args.modules_cat,
            valfold=valfold - 1,   # folds are 1-10 as in the training script (-f), load_dataset takes 0-9
            mode='train',
            rseed=args.rseed,
            competing_risk_death_check=args.competing_risk_death_check,
            include_gcsf=args.include_gcsf,
            data_fraction_option=args.data_fraction_option,
            quick_test=args.quick_test,
            caia_portion=args.caia_portion,
        )

        # leading static columns (prepended for default/duett models, see _finalize_output) are excluded
        n_static = 0
        if args.use_module_tokenizer:
            try:
                from MSK_featurename_list import build_module_index_config
                module_index_cfg = build_module_index_config(
                    modular_featuredim_dict, advanced_model=args.advanced_model,
                )
                n_static = module_index_cfg['n_static']
            except Exception as e:
                print(f"  (could not build module_index_cfg: {e}; falling back)")
        if n_static == 0:
            for k in ('gender', 'age', 'mutation', 'clinical'):
                if k in modular_featuredim_dict:
                    n_static += modular_featuredim_dict[k]
        print(f"  n_static feature columns (excluded from measurements): {n_static}")

        dm = FlexibleFeatureDataModule(
            data_dict_lab_dict, split_dict,
            dataset=args.dataset, component=component,
            batch_size=args.batch_size, max_time_len=args.num_tp, num_workers=4,
            task=option,
            advanced_model=args.advanced_model,
            include_infusion_dates=include_infusion_dates,
            competing_risk_death_check=args.competing_risk_death_check,
            include_gcsf=args.include_gcsf,
            anc=('CAIA' in args.dataset or args.dataset == 'insight'),
        )

        train_ds = dm.train_dataset

        n_windows, tp_list, meas_list = stats_for_dataset(train_ds, n_static)

        avg_tp = float(np.mean(tp_list)) if tp_list else float('nan')
        avg_meas = float(np.mean(meas_list)) if meas_list else float('nan')

        print(f"  windows                : {n_windows}")
        print(f"  avg time points/window : {avg_tp:.4f}")
        print(f"  avg measurements/window: {avg_meas:.4f}")

        fold_n_windows.append(n_windows)
        fold_avg_tp.append(avg_tp)
        fold_avg_meas.append(avg_meas)

    mean_windows = float(np.mean(fold_n_windows))
    mean_tp = float(np.mean(fold_avg_tp))
    mean_meas = float(np.mean(fold_avg_meas))

    print()
    print("PER-FOLD SUMMARY")
    print()
    print(f"{'fold':>6} | {'windows':>10} | {'avg_timepoints':>15} | {'avg_measurements':>17}")
    for i, f in enumerate(folds):
        print(f"{f:>6} | {fold_n_windows[i]:>10} | {fold_avg_tp[i]:>15.4f} | {fold_avg_meas[i]:>17.4f}")

    print()
    print("AVERAGED ACROSS FOLDS")
    print()
    print(f"  windows                : {mean_windows:,.1f}  (~{round(mean_windows):,})")
    print(f"  avg time points/window : {mean_tp:.2f}")
    print(f"  avg measurements/window: {mean_meas:.2f}")
    print()
    print("Sentence:")
    print(f'  "After pretraining on {round(mean_windows):,} windows containing an average of '
          f'{mean_tp:.1f} time points and {mean_meas:.1f} measurements per window, '
          f'SPARC demonstrated meaningful improvement over random change."')


if __name__ == '__main__':
    main()