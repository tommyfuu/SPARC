import torch
import pytorch_lightning as pl
import os
import sys
import json
# Put the repo root, data/ and models/ on sys.path, from any working directory.
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sparc_paths
from data_loaders_inf_risk_fixed import *
from dataset_loading import load_dataset   # <-- all data loading in one place
data_source_path=sparc_paths.DATA_ROOT + '/'

valfold = 0
advanced_model = 'default'
# dataset = 'MSK'
# dataset = 'MIMICIV_hosp'
data_fraction_option = 'default'

import argparse
parser = argparse.ArgumentParser(
            prog='hyperparameter tuning on pretraining for SPARC',
            description='Train the basic/extra feature model',
            epilog='Text at the bottom of help')
parser.add_argument("-r", "--rseed", type=int, default = 1024, help='random seed')
parser.add_argument("-mo", "--model", type=str, default = 'lasttp_MLP', help='model for which to tune the hyperparam')
parser.add_argument("-m", "--modules", type=str, default = 'parsimonious', help='modules to include')
parser.add_argument("-f", "--valfold", type=int, default = 1, help='fold number for validation, options: 1-10, fold number = i-1. If all, then iterative explore all options')
parser.add_argument("-o", "--option", type=str, default = 'binary', help='options: binary, anemia, binary_icu, binary_icu_pre, pretrain, binary_risk. It has to be pretrain when scheme is pretrain')
parser.add_argument("-d", "--days", type=int, default = 180, help='number of days for binary classification tasks')
parser.add_argument("-me", "--med_of_choice", type=str, default = 'CISPLATIN', help='medication of choice to predict post prediction risk of X')
parser.add_argument("-ds", "--dataset", type=str, default = 'MSK', help='dataset to use, options: MSK, MSK_timeshift, MIMICIV_hosp, stanfordEHRSHOT')
parser.add_argument("-an", "--add_num_infusion", type=str, default = "False", help="whether to add number of infusion as a feature")
parser.add_argument("-md", "--mode", type=str, default = "train", help="train or test")
parser.add_argument("-qt", "--quick_test", type=str, default = "False", help="True | False — if True, runs a quick test with 1 epoch and a small subset of the data (for testing the code pipeline)")
parser.add_argument("-cp", "--caia_portion", type=str, default = "prior", help="prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021")
parser.add_argument("-esp", "--early_stop_patience", type=int, default = 0, help="Early stopping patience in epochs (default 0 = disabled). Set to e.g. 30 to enable early stopping.")

args = parser.parse_args()
rseed = args.rseed
modules = args.modules
model_option = args.model
valfold = args.valfold - 1
option = args.option
days = args.days
med_of_choice = args.med_of_choice
dataset = args.dataset
add_num_infusion = args.add_num_infusion.lower() == 'true'
mode = args.mode.lower()
quick_test = args.quick_test.lower() == 'true'
caia_portion = args.caia_portion.lower()
early_stop_patience = args.early_stop_patience

# Validate valfold
if valfold not in range(10):
    raise ValueError(f'valfold must be between 0 and 9 (pass 1-10 to -f argument, got {args.valfold})')

med_of_choice = med_of_choice.split(',')
med_of_choice = [med.strip() for med in med_of_choice]
# JUST ADDED 100825: tackle the case where it's e.g. 'DOCETAXEL,SACITUZUMAB GOVITECAN,CISPLATIN,TOPOTECAN,IRINOTECAN,IRINOTECAN LIPOSOMAL,CYCLOPHOSPHAMIDE+DOXORUBICIN+VINCRISTINE,BENDAMUSTINE,CARBOPLATIN+PACLITAXEL,CARBOPLATIN+ETOPOSIDE,CABAZITAXEL'
result_meds = []
for med_str in med_of_choice:
    if '+' in med_str:
        split_meds = med_str.split('+')
        split_meds = [med.strip() for med in split_meds]
        result_meds.append(split_meds)
    else:
        result_meds.append([med_str])
med_of_choice = result_meds
#     assert med_str in ALL_MEDICATIONS, f"med_of_choice {med_str} not in the list of medications"
print("med_of_choice is a list of medications:", med_of_choice)

if model_option == 'lasttp_MLP':
    advanced_model = 'default'
elif model_option == 'duett':
    advanced_model = 'duett'

# # load config from json
config_path = f'{sparc_paths.BENCHMARK_CONFIG_DIR}/{model_option}_best_hyperparams.json'
with open(config_path, 'r') as f:
    all_configs = json.load(f)

if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)$', option)):
    config = all_configs[f'{option}']['config']
    print("getting the risk config for", option)
elif bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)$', option)):
    base_option = option.replace('_next_risk', '').replace('_risk', '')
    config = all_configs[f'{base_option}']['config']
else:
    config = all_configs[option]['config']


if model_option in ['duett', 'raindrop']:
    advanced_model = model_option   
if model_option in ['lasttp_MLP', 'raindrop', 'rnn', 'rnn_deephit', 'duett']:
    sparc_paths.add_benchmark_to_path(model_option)
    if model_option == 'duett':
        # Standalone DuETT (models/benchmarks/duett/): its own architecture, no
        # dependency on the SPARC classes. advanced_model='duett' above is still
        # required -- the loader branches on it for the value/count layout.
        from duett import DuETT
    elif model_option == 'lasttp_MLP':
        from lasttp_MLP import *
    elif model_option == 'raindrop':
        from raindrop import *
    elif model_option in ['rnn', 'rnn_deephit']:
        sparc_paths.add_benchmark_to_path('rnn')
        from rnn import *



## HYPERPARAMETERS ## 
batch_size = 512
competing_risk_death_check = False
    

#  Load dataset (all loading/normalization/splitting is in dataset_loading.py)
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
    include_gcsf=False,  # DL benchmarks don't use GCSF
    data_fraction_option=data_fraction_option,
    quick_test=quick_test,
    caia_portion=caia_portion,
)
print("modular_featuredim_dict", modular_featuredim_dict)
print("current validation fold:", valfold)

pl.seed_everything(rseed)

# Data loader
print("med of choice", med_of_choice)


if dataset == 'MSK':
    output_model_path = f'{sparc_paths.RESULTS_ROOT}/practical/benchmarks/{model_option}/{modules}_{option}/fold_{valfold}/'
else:
    output_model_path = f'{sparc_paths.RESULTS_ROOT}/{modules}_{dataset}/benchmarks/{model_option}/{modules}_{option}_{dataset}/fold_{valfold}/'
print("output model path", output_model_path)

if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    model_set_option = 'side_effects'
    print("side effects model")
    if 'risk' in option:
        if 'next_risk' in option:
            model_set_option = 'side_effects_next_risk'
        else:
            model_set_option = 'side_effects_risk'

    # Risk models are stored under the base task name (parsimonious_AKI_MSK_CAIA,
    # not parsimonious_AKI_risk_MSK_CAIA). eval_1 strips '_risk' the same way when
    # it builds benchmark paths, so the two must stay in step.
    if 'risk' in option:
        output_model_path = output_model_path.replace(f'{option}', option.replace('_risk', ''))
else:
    model_set_option = 'binary'

print("FINAL output model path", output_model_path)


if advanced_model in ['raindrop']:
    total_static_len = sum([v for k,v in modular_featuredim_dict.items() if k in ['gender', 'age', 'primary_ctype']])
    total_dynamic_len = sum([v for k,v in modular_featuredim_dict.items() if k in ['basic_lab', 'extra_lab', 'basic_vitals']])
    print("total_static_len", total_static_len, "total_dynamic_len", total_dynamic_len)

print("config", config)

# Override add_num_infusion for binary tasks (infusion numbers only relevant for side-effect tasks)
if option in ['binary', 'binary_icu_pre', 'binary_risk'] and add_num_infusion:
    print(f"WARNING: add_num_infusion=True is not applicable for task '{option}'. Setting to False.")
    add_num_infusion = False

data_module = FlexibleFeatureDataModule(data_dict_lab_dict, split_dict, dataset = dataset, component = component,
    batch_size = 512, max_time_len = config['max_time_len'], num_workers=1,
    task = option, binary_days = days, med = med_of_choice, med_as_feature = False, med_l = med_of_choice,
    advanced_model = advanced_model, include_infusion_dates = include_infusion_dates, anc=('CAIA' in dataset))
print("data module initiated")


print("last model path", f'{output_model_path}/last.ckpt')
if not os.path.exists(f'{output_model_path}/last.ckpt'):
    print("training from scratch")
    if model_option == 'lasttp_MLP':
        model = lasttp_MLP(modular_featuredim_dict, option = model_set_option, weight_decay=config["weight_decay"], lr = config["lr"],
                    num_layers=config["num_layers"], side_effects_num_infusion = add_num_infusion)
    elif model_option == 'duett':
        model = DuETT(modular_featuredim_dict, option = model_set_option,
                max_time_len=config["max_time_len"], weight_decay=config["weight_decay"], lr = config["lr"],
                num_encoder_layers=config["num_encoder_layers"],  dropout=config["dropout"], side_effects_num_infusion = add_num_infusion)
    elif model_option == 'raindrop':
        model = Raindrop(d_static=total_static_len, nlayers=config["nlayers"], max_len=config["max_time_len"],
                lr=config["lr"], weight_decay=config["weight_decay"], option = model_set_option, side_effects_num_infusion = add_num_infusion)
    elif model_option in ['rnn', 'rnn_deephit']:
        model = RNN(modular_featuredim_dict=modular_featuredim_dict, num_encoder_layers=config["num_encoder_layers"],
                max_time_len=config["max_time_len"], lr=config["lr"], weight_decay=config["weight_decay"],
                option = model_set_option, side_effects_num_infusion = add_num_infusion, advanced_model = advanced_model)
    resume = False
else:
    if mode == 'test':
        # find best checkpoint
        print("loading best checkpoint for testing")
        ## check the files in output_model_path ending with .ckpt and not last.ckpt in output_model_path
        files = os.listdir(output_model_path)
        ckpt_files = [f for f in files if f.endswith('.ckpt') and f != 'last.ckpt']
        best_ckpt_file = ckpt_files[0]
        checkpoint_path_load = f'{output_model_path}/{best_ckpt_file}'
    else:
        checkpoint_path_load = f'{output_model_path}/last.ckpt'

    print("loading checkpoint from", checkpoint_path_load)
    
    if model_option == 'lasttp_MLP':
        model = lasttp_MLP.load_from_checkpoint(checkpoint_path_load, strict=False,
            modular_featuredim_dict = modular_featuredim_dict, option = model_set_option, weight_decay=config["weight_decay"], lr = config["lr"],
            num_layers=config["num_layers"], side_effects_num_infusion = add_num_infusion)
    if model_option == 'duett':
        model = DuETT.load_from_checkpoint(checkpoint_path_load, strict=False,
            modular_featuredim_dict = modular_featuredim_dict, option = model_set_option,
            max_time_len=config["max_time_len"], weight_decay=config["weight_decay"], lr = config["lr"],
            num_encoder_layers=config["num_encoder_layers"],  dropout=config["dropout"], side_effects_num_infusion = add_num_infusion)
    if model_option == 'raindrop':
        model = Raindrop.load_from_checkpoint(checkpoint_path_load, strict=False,
            d_static=total_static_len, nlayers=config["nlayers"], max_len=config["max_time_len"],
            lr=config["lr"], weight_decay=config["weight_decay"], option = model_set_option, side_effects_num_infusion = add_num_infusion)
    if model_option in ['rnn', 'rnn_deephit']:
        print("model_set_option", model_set_option)
        model = RNN.load_from_checkpoint(checkpoint_path_load, strict=False,
            modular_featuredim_dict=modular_featuredim_dict, num_encoder_layers=config["num_encoder_layers"], max_time_len=config["max_time_len"],
            lr=config["lr"], weight_decay=config["weight_decay"], option = model_set_option, side_effects_num_infusion = add_num_infusion, advanced_model = advanced_model)
    resume = True

checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=1, save_last=True, mode='min', monitor='val_loss', dirpath=output_model_path)
logger_csv = pl.loggers.CSVLogger(f"{output_model_path}/logs_csv", name="lightning_logs_csv")
logger_tb = pl.loggers.TensorBoardLogger(f"{output_model_path}/logs_tb", name="lightning_logs_tb")

max_epochs = 200

# Build callbacks list
callbacks_list = [checkpoint]

# Add early stopping callback if patience > 0
if early_stop_patience > 0:
    print(f"Adding EarlyStopping with patience={early_stop_patience}, monitoring val_loss (min)")
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        mode='min',
        verbose=True,
        min_delta=0.0001  # Minimum change to qualify as improvement
    )
    callbacks_list.append(early_stop_callback)
else:
    print("Early stopping disabled (patience=0)")

trainer = pl.Trainer(max_epochs=max_epochs, logger=[logger_csv, logger_tb], callbacks=callbacks_list, 
    deterministic=True, accumulate_grad_batches=10, gradient_clip_val=0.5, gradient_clip_algorithm='norm',
    accelerator="gpu", devices="auto")
trainer.test(model, data_module.test_dataloader())

if mode != 'test':
    print("training...")
    if not resume:
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, data_module, ckpt_path=f'{output_model_path}/last.ckpt')
        
    trainer.test(model, data_module.test_dataloader())
