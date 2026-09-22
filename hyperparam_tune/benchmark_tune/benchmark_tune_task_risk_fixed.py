"""
benchmark_tune_task_risk_fixed.py

Hyperparameter tuning script for benchmark DL models on SPARC tasks.
"""
import os
import sys
import pytorch_lightning as pl

# Set CUDA environment for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Put the repo root, data/ and models/ on sys.path, from any working directory.
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths

from benchmark_search_spaces import (SEARCH_SPACES, ADVANCED_MODEL,
                                     n_combinations, config_at, tune_run_dirname)
from data_loaders_inf_risk_fixed import FlexibleFeatureDataModule
from dataset_loading import load_dataset

import argparse
parser = argparse.ArgumentParser(
    prog='hyperparameter tuning for benchmark models',
    description='Train benchmark models with hyperparameter search',
    epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default=1024, help='random seed')
parser.add_argument("-mo", "--model", type=str, default='lasttp_MLP', help='model: lasttp_MLP | duett | raindrop | rnn | rnn_deephit')
parser.add_argument("-m", "--modules", type=str, default='parsimonious', help='modules: all | mutation_clinical | practical')
parser.add_argument("-i", "--hyper_tune_iter", type=int, default=1, help='hyperparameter tuning iteration (1-indexed)')
parser.add_argument("-o", "--option", type=str, default='binary', help='task: binary | anemia_risk | neutropenia_risk | AKI_risk | etc.')
parser.add_argument("-d", "--days", type=int, default=180, help='horizon days for binary tasks')
parser.add_argument("-me", "--med_of_choice", type=str, default='CISPLATIN', help='medication(s) for side effect tasks')
parser.add_argument("-an", "--add_num_infusion", type=str, default="False", help="add infusion number as feature")
parser.add_argument("-ds", "--dataset", type=str, default='MSK', help='dataset: MSK | MSK_timeshift | MIMICIV_hosp')
parser.add_argument("-vp", "--volume_path", type=str, default=sparc_paths.DATA_ROOT + '/', help='base path to all datasets')
parser.add_argument("-op", "--output_model_path", type=str, default=sparc_paths.RESULTS_ROOT + '/parsimonious_MSK/benchmark_tunes/', help='output directory')
parser.add_argument("-esp", "--early_stop_patience", type=int, default=0,
                    help="Epochs of no val_loss improvement before stopping. "
                         "0 (default) disables it and every configuration runs "
                         "the full 200 epochs.")

args = parser.parse_args()
rseed = args.rseed
modules = args.modules
model_option = args.model
hyper_tune_iter = args.hyper_tune_iter - 1  # Convert to 0-indexed
option = args.option
days = args.days
med_of_choice = args.med_of_choice
add_num_infusion = args.add_num_infusion.lower() == 'true'
dataset = args.dataset
early_stop_patience = args.early_stop_patience
data_source_path = args.volume_path
output_base_path = args.output_model_path

competing_risk_death_check = True
med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]

result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        split_meds = med_str.split('+')
        split_meds = [med.strip() for med in split_meds]
        result_meds.append(split_meds)
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
print(f"Medications: {med_of_choice}")

advanced_model = 'default'
if model_option in ['duett', 'raindrop']:
    advanced_model = model_option

if model_option in ['lasttp_MLP', 'raindrop', 'rnn', 'rnn_deephit']:
    sparc_paths.add_benchmark_to_path(model_option)
    if model_option == 'lasttp_MLP':
        from lasttp_MLP import lasttp_MLP

batch_size = 512
valfold = 0  # Use fold 0 for validation
mode = 'train'
data_fraction_option = 'default'
include_gcsf = False
quick_test = False

pl.seed_everything(rseed)

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
    include_gcsf=include_gcsf,
    data_fraction_option=data_fraction_option,
    quick_test=quick_test,
    caia_portion='prior',
)

# Override split_dict to use validation set instead of test set
split_dict['test'] = []

print(f"Dataset loaded. Train: {len(split_dict['train'])}, Val: {len(split_dict['val'])}, Test: {len(split_dict['test'])}")
print(f"Modular feature dimensions: {modular_featuredim_dict}")

# For Raindrop, calculate static and dynamic feature dimensions
if advanced_model in ['raindrop']:
    total_static_len = sum([v for k, v in modular_featuredim_dict.items()
                           if k in ['gender', 'age', 'primary_ctype']])
    total_dynamic_len = sum([v for k, v in modular_featuredim_dict.items()
                            if k in ['basic_lab', 'extra_lab', 'basic_vitals']])
    print(f"Static features: {total_static_len}, Dynamic features: {total_dynamic_len}")

# The grid lives in benchmark_search_spaces.py
search_space = SEARCH_SPACES[model_option]
advanced_model = ADVANCED_MODEL[model_option]

total_combinations = n_combinations(model_option)
print(f"Total hyperparameter combinations: {total_combinations}")

config = config_at(model_option, hyper_tune_iter)
print(f"Hyperparameter configuration #{hyper_tune_iter + 1}: {config}")

output_model_path = os.path.join(
    output_base_path, model_option, option,
    tune_run_dirname(model_option, option, config))

os.makedirs(output_model_path, exist_ok=True)

import re
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    include_infusion_dates = True
    model_set_option = 'side_effects'
    if 'next_risk' in option:
        model_set_option = 'side_effects_next_risk'
    elif 'risk' in option:
        model_set_option = 'side_effects_risk'
    print(f"Side effects model: {model_set_option}")
else:
    model_set_option = 'binary'

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=batch_size,
    max_time_len=config['max_time_len'],
    num_workers=1,
    task=option,
    binary_days=days,
    med=med_of_choice,
    med_as_feature=False,
    med_l=med_of_choice,
    advanced_model=advanced_model,
    include_infusion_dates=include_infusion_dates,
    control_lack_confounds=False,
    side_effect_censoring=False,
    premed_only=False,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=include_gcsf,
    anc=('CAIA' in dataset),
)
print("Data module initiated")

last_ckpt_path = f'{output_model_path}/last.ckpt'

if not os.path.exists(last_ckpt_path):
    print("Training from scratch")
    if model_option == 'lasttp_MLP':
        model = lasttp_MLP(
            modular_featuredim_dict, option=model_set_option,
            weight_decay=config["weight_decay"], lr=config["lr"],
            num_layers=config["num_layers"],
            side_effects_num_infusion=add_num_infusion
        )
    elif model_option == 'duett':
        sparc_paths.add_benchmark_to_path('duett')
        from duett import DuETT
        model = DuETT(
            modular_featuredim_dict, option=model_set_option,
            max_time_len=config["max_time_len"],
            weight_decay=config["weight_decay"], lr=config["lr"],
            num_encoder_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            side_effects_num_infusion=add_num_infusion,
        )
    resume = False
else:
    print(f"Loading checkpoint from {last_ckpt_path}")
    if model_option == 'lasttp_MLP':
        model = lasttp_MLP.load_from_checkpoint(
            last_ckpt_path, strict=False,
            modular_featuredim_dict=modular_featuredim_dict,
            option=model_set_option,
            weight_decay=config["weight_decay"], lr=config["lr"],
            num_layers=config["num_layers"],
            side_effects_num_infusion=add_num_infusion
        )
    elif model_option == 'duett':
        # Same standalone class the fresh branch builds; resuming through SPARC
        # would load these weights into a different architecture.
        sparc_paths.add_benchmark_to_path('duett')
        from duett import DuETT
        model = DuETT.load_from_checkpoint(
            last_ckpt_path, strict=False,
            modular_featuredim_dict=modular_featuredim_dict,
            option=model_set_option,
            max_time_len=config["max_time_len"],
            weight_decay=config["weight_decay"], lr=config["lr"],
            num_encoder_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            side_effects_num_infusion=add_num_infusion,
        )
    resume = True

checkpoint = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    mode='min',
    monitor='val_loss',
    dirpath=output_model_path,
    filename='epoch={epoch}-step={step}',
    enable_version_counter=False,
)

logger_csv = pl.loggers.CSVLogger(f"{output_model_path}/logs_csv", name="lightning_logs_csv")
logger_tb = pl.loggers.TensorBoardLogger(f"{output_model_path}/logs_tb", name="lightning_logs_tb")

max_epochs = 200

callbacks = [checkpoint]
if early_stop_patience > 0:
    callbacks.append(pl.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=early_stop_patience,
        check_finite=True))
    print(f"EarlyStopping enabled: monitoring val_loss, patience={early_stop_patience}")
else:
    print("EarlyStopping disabled - running the full max_epochs")

trainer = pl.Trainer(
    max_epochs=max_epochs,
    logger=[logger_csv, logger_tb],
    callbacks=callbacks,
    deterministic=True,
    accumulate_grad_batches=10,
    gradient_clip_val=0.5,
    gradient_clip_algorithm='norm',
    accelerator="gpu",
    devices="auto"
)

if not resume:
    print("Starting training from scratch")
    trainer.fit(model, data_module)
else:
    print(f"Resuming training from {last_ckpt_path}")
    trainer.fit(model, data_module, ckpt_path=last_ckpt_path)

print("Training complete!")
print(f"Best checkpoint: {checkpoint.best_model_path}")
