#%%
"""Train / evaluate SPARC infusion-risk models. Data loading lives in data/dataset_loading.py."""
import os
import sys
import copy

# Lightning rejects a multi-task SLURM allocation without --ntasks-per-node, such as an
# interactive OnDemand session; this job name makes it run as a plain local process there.
if int(os.environ.get('SLURM_NTASKS', '1')) > 1 and 'SLURM_NTASKS_PER_NODE' not in os.environ:
    os.environ['SLURM_JOB_NAME'] = 'interactive'

import torch
import numpy as np
import pytorch_lightning as pl
import statsmodels.api as sm


def is_checkpoint_corrupted(ckpt_path):
    """Check if a checkpoint file is corrupted by attempting to load it."""
    if not os.path.exists(ckpt_path):
        return False  # missing, not corrupted

    try:
        torch.load(ckpt_path, map_location='cpu', weights_only=False)
        return False
    except RuntimeError as e:
        if 'PytorchStreamReader failed reading zip archive' in str(e) or 'failed finding central directory' in str(e):
            return True
        raise
    except Exception:
        return True


def write_test_predictions(model, dataset, path, batch_size, seed):
    """Write one row per test sample: patient id, prediction day, infusion number, label(s) and score.

    With -bf False the encoder attends across the batch, so samples are scored in the shuffled
    batches that trainer.test and the paper's evaluation use (seed_everything(seed), then a
    shuffled loader), and written back in dataset order.
    """
    import pandas as pd
    from pytorch_lightning.utilities import move_data_to_device
    from torch.utils.data import DataLoader, Subset
    def as_columns(y):   # *_risk labels arrive as [event, time_to_event] tensors
        if isinstance(y, (list, tuple)):
            return torch.stack([torch.as_tensor(v).reshape(-1).float() for v in y], dim=1)
        return torch.as_tensor(y).reshape(-1, 1).float()

    pl.seed_everything(seed)
    order = torch.cat(list(DataLoader(range(len(dataset)), batch_size=batch_size, shuffle=True)))
    back = torch.argsort(order)
    model.eval()
    with torch.no_grad():
        outputs = [model.predict_step(move_data_to_device(batch, model.device), i)
                   for i, batch in enumerate(DataLoader(Subset(dataset, order.tolist()), batch_size=batch_size))]
    scores = torch.cat([o['y_pred'].reshape(-1).float().cpu() for o in outputs])[back].numpy()
    labels = torch.cat([as_columns(o['y_true']).cpu() for o in outputs])[back].numpy()
    df = pd.DataFrame({'patient_id': dataset.dataset_DMP_IDs})
    if len(getattr(dataset, 'cutoffdates', [])) == len(df):
        df['prediction_day'] = dataset.cutoffdates
    if outputs[0].get('num_infusion') is not None:
        infusions = torch.cat([torch.as_tensor(o['num_infusion']).reshape(-1) for o in outputs])
        df['infusion_num'] = infusions[back].numpy()
    if labels.shape[1] == 2:
        df['event'], df['time_to_event'] = labels[:, 0], labels[:, 1]
    else:
        df['label'] = labels[:, 0]
    df['score'] = scores
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} test predictions to {path}")

# put the repo root, data/ and models/ on sys.path
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

from SPARC_inf_risk_modular import *


# Arguments
import argparse
parser = argparse.ArgumentParser(
    prog='SPARC_train',
    description='Pretrain, finetune or evaluate (-mo test) SPARC on one fold.')
parser.add_argument("-r",   "--rseed",                   type=int,                              default=1024,        help='random seed')
parser.add_argument("-f",   "--valfold",                 type=int,                              default=1,           help='fold number for validation (1-10)')
parser.add_argument("-s",   "--scheme",                  type=str,                              default='pretrain',  help='pretrain | finetune')
parser.add_argument("-o",   "--option",                  type=str,                              default='binary',    help='binary | pretrain | neutropenia_risk | AKI_risk | ...')
parser.add_argument("-d",   "--days",                    type=int,                              default=180,         help='horizon days for binary tasks')
parser.add_argument("-pp",  "--pretrained_model_path",   type=str,                              default='',          help='path to a pretrained checkpoint to start from')
parser.add_argument("-op",  "--output_model_path",       type=str,                              default=sparc_paths.CTFM_DIR + "_models_modular" + '/basic_lab_direct_binary', help='output directory for checkpoints / logs')
parser.add_argument("-e",   "--epochs",                  type=int,                              default=200,         help='number of training epochs')
parser.add_argument("-m",   "--modules_cat",             type=str,                              default='all',       help='all | all_labs | mutation_clinical | practical | parsimonious')
parser.add_argument("-n",   "--num_tp",                  type=int,                              default=20,          help='max sequence length')
parser.add_argument("-me",  "--med_of_choice",           type=str,                              default='CISPLATIN', help='comma-separated medication names')
parser.add_argument("-mo",  "--mode",                    type=str,                              default='train',     help='train | test')
parser.add_argument("-ds",  "--dataset",                 type=str,                              default='MSK',       help='MSK | MSK_timeshift | stanfordEHRSHOT | MIMICIV_hosp | *CAIA*')
parser.add_argument("-mop", "--model_option",            type=str,                              default='default',   help='default | duett')
parser.add_argument("-df",  "--data_fraction_option",    type=str,                              default='default',   help='default | 1/5 | 1/25 | 1/125')
parser.add_argument("-mf",  "--med_as_feature",          type=str,                              default='False',     help='True | False')
parser.add_argument("-vp",  "--volume_path",             type=str,                              default=sparc_paths.DATA_ROOT + '/', help='base path to all datasets')
parser.add_argument("-an",  "--add_num_infusion",        type=str,                              default='False',     help='True | False — add infusion number as feature')
parser.add_argument("-lo",  "--loss_option",             type=str,                              default='default',   help='default (bce/cox depending on binary/side effect or side_effect_risk) | hybrid (cox+bce weighted by data)')
parser.add_argument("-sd",  "--schedule_option",         type=str,                              default='default',   help='default (adamw constant) | reduce_on_plateau')
parser.add_argument("-crd", "--competing_risk_death_check", type=str,                           default='False',     help='True | False')
parser.add_argument("-gcsf","--include_gcsf",            type=str,                              default='False',     help='True | False — include GCSF administration as a feature')
parser.add_argument("-tp",  "--time_processing",         type=str,                              default='attention', choices=['attention'], help="attention (only supported value; kept so existing launchers' -tp attention still parses)")
parser.add_argument("-qt",  "--quick_test",              type=str,                              default='False',     help='True | False — if True, runs a quick test with 1 epoch and a small subset of the data (for testing the code pipeline)')
parser.add_argument("-cp",  "--caia_portion",            type=str,                              default='prior',     help='prior | post | entire - which portion of MSK_CAIA dataset to use (only relevant if dataset is MSK_CAIA) wrt 2021')
parser.add_argument("-cdw", "--change_detection_weight", type=float,                            default=0.1,         help='Weight for auxiliary change detection loss in pretraining (default 0.1)')
parser.add_argument("-esp", "--early_stop_patience",     type=int,                              default=0,          help='Early stopping patience in epochs (default 50 for pretraining, 30 recommended for finetuning). Set to 0 to disable.')
parser.add_argument("-lr",  "--learning_rate",           type=float,                            default=1e-4,        help='Learning rate (default 1e-4; for linear probe consider 1e-3 or 1e-2)')
parser.add_argument("-mm",  "--missing_mask",             type=str,                              default='True',      choices=['True'],      help="True (only supported value; the -10 sentinel no-mask path was removed)")
parser.add_argument("-umt", "--use_module_tokenizer",    type=str,                              default='False',     help='True | False — use per-clinical-module input tokenization (erythroid/myeloid/renal/hepatic) with missingness gating')
parser.add_argument("-uct", "--use_continuous_time",     type=str,                              default='True',      help='True | False — add learned continuous-time embedding from actual time values')
parser.add_argument("-pmd", "--post_mlp_dropout",        type=float,                            default=-1.0,        help='Dropout for post-transformer representation MLP (default: same as model dropout, i.e. 0.1). Set >=0 to override.')
parser.add_argument("-traw","--time_as_raw_feature",     type=str,                              default='False',     help='True | False — feed elapsed time as a raw scalar through a linear map. Use with -uct False to test time-as-feature WITHOUT the Fourier embedding.')
parser.add_argument("-bf",  "--batch_first",             type=str,                              default='True',     help='True | False — build the transformer with batch_first=True so it attends over time (all reported models).')
parser.add_argument("-wbce","--weighted_bce",            type=str,                              default='True',      help='True | False — use WeightedBCELoss (pos_weight=n_neg/n_pos) to handle class imbalance; False uses standard BCE')
parser.add_argument("-bs",  "--batch_size",              type=int,                              default=512,         help='batch size (default 512)')
parser.add_argument("-acc", "--accelerator",             type=str,                              default='auto',      help='auto | gpu | cpu — Lightning accelerator (auto uses a GPU when one is available)')

#%%
args = parser.parse_args()
print(args)

rseed               = args.rseed
valfold             = args.valfold - 1
option              = args.option
days                = args.days
scheme              = args.scheme
pretrained_model_path = args.pretrained_model_path
output_model_path   = args.output_model_path
epochs              = args.epochs
modules             = args.modules_cat
max_time_len_data   = args.num_tp
med_of_choice       = args.med_of_choice
mode                = args.mode
dataset             = args.dataset
advanced_model      = args.model_option
data_fraction_option = args.data_fraction_option
med_as_feature      = args.med_as_feature.lower() == 'true'
data_source_path    = args.volume_path
add_num_infusion    = args.add_num_infusion.lower() == 'true'
loss_option         = args.loss_option
schedule_option     = args.schedule_option
competing_risk_death_check = args.competing_risk_death_check.lower() == 'true'
include_gcsf        = args.include_gcsf.lower() == 'true'
time_processing     = args.time_processing   # 'attention' (enforced by argparse choices)
quick_test          = args.quick_test.lower() == 'true'
caia_portion       = args.caia_portion.lower()
change_detection_weight = args.change_detection_weight
early_stop_patience = args.early_stop_patience
learning_rate       = args.learning_rate
missing_mask        = args.missing_mask.lower() == 'true'   # always True (enforced by argparse choices)
use_module_tokenizer = args.use_module_tokenizer.lower() == 'true'
batch_first          = args.batch_first.lower() == 'true'   # default True
time_as_raw_feature  = args.time_as_raw_feature.lower() == 'true'
use_continuous_time  = args.use_continuous_time.lower() == 'true'
post_mlp_dropout     = None if args.post_mlp_dropout < 0 else args.post_mlp_dropout
weighted_bce         = args.weighted_bce.lower() == 'true'

# Fixed settings; no launcher varies these.
control_confounds     = False   # -> DataModule control_lack_confounds
side_effect_censoring = False   # stored by the loader, never read
premed_only           = False   # -> DataModule premed_only
resume                = False   # real value is derived from last.ckpt below

# medication list, comma or + separated
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
batch_size = args.batch_size

# Override option when scheme == 'pretrain'
if scheme == 'pretrain' and option != 'pretrain' and mode != 'test':
    option = 'pretrain'
    print("Warning: scheme is pretrain; option overridden to 'pretrain'")


if valfold not in range(10):
    raise ValueError('valfold must be between 1 and 10 (11th fold is the held-out test set)')

if mode == 'test':
    # test only a model trained for this fold, never a pretrained or untrained one
    fold_dir = f'{output_model_path}/fold{valfold}'
    if not any(x.startswith('epoch=') and x.endswith('.ckpt')
               for x in (os.listdir(fold_dir) if os.path.isdir(fold_dir) else [])):
        raise FileNotFoundError(f"No trained checkpoint (epoch=*.ckpt) in {fold_dir}; train this fold first")

# Load data
print("Loading dataset …")
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
    caia_portion=caia_portion,
)
print("modular_featuredim_dict", modular_featuredim_dict)
print("current validation fold:", valfold)

# Per-module tokenizer config (use_module_tokenizer=True only)
module_index_cfg = None
n_static_features = 0
if use_module_tokenizer:
    from MSK_featurename_list import build_module_index_config
    module_index_cfg = build_module_index_config(
        modular_featuredim_dict, advanced_model=advanced_model,
    )
    n_static_features = module_index_cfg['n_static']
    print("=" * 70)
    print("PER-MODULE TOKENIZER CONFIG")
    print(f"  n_static={module_index_cfg['n_static']}, "
          f"n_dynamic={module_index_cfg['n_dynamic']}, "
          f"n_base={module_index_cfg['n_base']}")
    for mod, idxs in module_index_cfg['modules']:
        print(f"  {mod:18s} ({len(idxs):2d}): {idxs}")
    print("=" * 70)

# Data module
pl.seed_everything(rseed)
print("med of choice", med_of_choice)

# infusion number is only meaningful for side-effect tasks
if option in ['binary', 'binary_icu_pre', 'binary_risk'] and add_num_infusion:
    print(f"WARNING: add_num_infusion=True is not applicable for task '{option}'. Setting to False.")
    add_num_infusion = False

print("all arguments for data module:")
print(f"dataset: {dataset}, component: {component}, batch_size: {batch_size}, max_time_len: {max_time_len_data}, task: {option}, binary_days: {days}, med_as_feature: {med_as_feature}, advanced_model: {advanced_model}, control_lack_confounds: {control_confounds}, side_effect_censoring: {side_effect_censoring}, include_infusion_dates: {include_infusion_dates}, premed_only: {premed_only}, competing_risk_death_check: {competing_risk_death_check}, include_gcsf: {include_gcsf}")

data_module = FlexibleFeatureDataModule(
    data_dict_lab_dict, split_dict,
    dataset=dataset, component=component,
    batch_size=batch_size, max_time_len=max_time_len_data, num_workers=4,
    task=option, binary_days=days,
    med=med_of_choice, med_as_feature=med_as_feature, med_l=med_of_choice,
    advanced_model=advanced_model, control_lack_confounds=control_confounds,
    side_effect_censoring=side_effect_censoring,
    include_infusion_dates=include_infusion_dates,
    premed_only=premed_only,
    competing_risk_death_check=competing_risk_death_check,
    include_gcsf=include_gcsf,
    anc=('CAIA' in dataset or dataset == 'insight'),
)
print("data module initiated")
print(f"DEBUG: Train size={len(split_dict['train'])}, Val size={len(split_dict['val'])}, Test size={len(split_dict['test'])}")
print(f"DEBUG: First 5 val IDs: {list(split_dict['val'])[:5]}")
if len(data_module.test_dataset) == 0:
    raise ValueError(f"No test samples for {option}: fold 10 has no patient with a usable sample, "
                     f"or fewer than -bs/5 = {batch_size // 5}, which are dropped")

# pos_weight for binary tasks
binary_pos_weight = None
side_effects_pos_weight = None
_SE_OPTIONS = ['anemia', 'neutropenia', 'febrile_neutropenia', 'AKI', 'Liver_toxicity']
if weighted_bce:
    if option in ['binary', 'binary_icu_pre']:
        try:
            train_ys = data_module.train_dataset.ys
            n_pos = sum(1 for y in train_ys if y == 1)
            n_neg = len(train_ys) - n_pos
            binary_pos_weight = n_neg / max(n_pos, 1)
            print(f"Binary pos_weight: {binary_pos_weight:.2f} ({n_pos} pos, {n_neg} neg, ratio {n_neg/max(n_pos,1):.1f}:1)")
        except Exception as e:
            print(f"Warning: could not compute pos_weight: {e}")
    elif option in _SE_OPTIONS:
        try:
            train_ys = data_module.train_dataset.ys
            n_pos = sum(1 for y in train_ys if y == 1)
            n_neg = len(train_ys) - n_pos
            side_effects_pos_weight = n_neg / max(n_pos, 1)
            print(f"Side-effects pos_weight: {side_effects_pos_weight:.2f} ({n_pos} pos, {n_neg} neg, ratio {n_neg/max(n_pos,1):.1f}:1)")
        except Exception as e:
            print(f"Warning: could not compute side_effects pos_weight: {e}")
else:
    print("WeightedBCE disabled, using standard BCE (pos_weight=None)")


# Determine model_option from task option
use_sigmoid = True

if option == 'pretrain':
    model_option = 'pretrain'
elif bool(__import__('re').match(
        r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    if 'next_risk' in option:
        model_option = 'side_effects_next_risk'
    elif 'risk' in option:
        model_option = 'side_effects_risk'
    else:
        model_option = 'side_effects'
    print("side effects model →", model_option)
else:
    model_option = 'binary'

# Build or load model
if modules in ['all', 'practical', 'parsimonious']:
    no_existing_checkpoint = (
        pretrained_model_path == ''
        and not os.path.exists(f'{output_model_path}/fold{valfold}/last.ckpt')
        and mode != 'test'
    )
    if no_existing_checkpoint:
        print("initiating new model")
        model = SPARC(
            modular_featuredim_dict, option=model_option,
            advanced_model=advanced_model, max_time_len=max_time_len_data,
            med_as_feature=med_as_feature, meds_l=med_of_choice,
            loss_option=loss_option,
            schedule = schedule_option,
            side_effects_num_infusion=add_num_infusion,
            include_gcsf=include_gcsf, use_gcsf=include_gcsf,
            use_sigmoid=use_sigmoid,
            temporal_aggregation=time_processing,
            missing_mask=missing_mask,
            module_index_cfg=module_index_cfg,
            use_module_tokenizer=use_module_tokenizer,
            use_continuous_time=use_continuous_time,
            n_static_features=n_static_features,
            post_mlp_dropout=post_mlp_dropout,
            batch_first=batch_first,
            time_as_raw_feature=time_as_raw_feature,
            lr=learning_rate,
            binary_pos_weight=binary_pos_weight,
            side_effects_pos_weight=side_effects_pos_weight,
        )
        resume = None
    else:
        fold_dir = f'{output_model_path}/fold{valfold}'
        if os.path.exists(f'{fold_dir}/last.ckpt') and mode != 'test':
            last_ckpt_path = f'{fold_dir}/last.ckpt'

            if is_checkpoint_corrupted(last_ckpt_path):
                print(f"WARNING: {last_ckpt_path} is corrupted (likely due to job interruption during save).")
                print("Deleting corrupted checkpoint and falling back to best epoch checkpoint...")
                os.remove(last_ckpt_path)

                # best epoch checkpoint
                ckpts = [x for x in os.listdir(fold_dir)
                         if x != 'last.ckpt' and not os.path.isdir(f'{fold_dir}/{x}') and '.ckpt' in x]
                if ckpts:
                    optimal_model_path = f'{fold_dir}/{ckpts[0]}'
                    print(f"Using best checkpoint instead: {optimal_model_path}")
                    resuming_finetuning_checkpoint = True
                else:
                    print("No valid epoch checkpoint found. Starting from pretrained model.")
                    # remove the folder so a corrupted checkpoint is not picked up later
                    os.rmdir(fold_dir)
                    os.mkdir(fold_dir)
                    if pretrained_model_path == '':
                        raise ValueError("No valid checkpoint found and no pretrained model path provided.")
                    pretrain_fold_dir = f'{pretrained_model_path}/fold{valfold}'
                    ckpts = [x for x in os.listdir(pretrain_fold_dir)
                             if x != 'last.ckpt' and not os.path.isdir(f'{pretrain_fold_dir}/{x}') and 'ckpt' in x]
                    optimal_model_path = f'{pretrain_fold_dir}/{ckpts[0]}'
                    resuming_finetuning_checkpoint = False
            else:
                print("RESUME CHECKPOINT 1")
                optimal_model_path = last_ckpt_path
                resuming_finetuning_checkpoint = True
        elif mode == 'test' and os.path.exists(fold_dir):
            ckpts = [x for x in os.listdir(fold_dir)
                     if x != 'last.ckpt' and not os.path.isdir(f'{fold_dir}/{x}') and '.ckpt' in x]
            print("test mode, checkpoints:", ckpts)
            optimal_model_path = f'{fold_dir}/{[x for x in ckpts if "last" not in x][0]}'
            print("optimal model path for testing:", optimal_model_path)
            resuming_finetuning_checkpoint = True
        else:
            print("RESUME CHECKPOINT 2")
            pretrain_fold_dir = f'{pretrained_model_path}/fold{valfold}'
            ckpts = [x for x in os.listdir(pretrain_fold_dir)
                     if x != 'last.ckpt' and not os.path.isdir(f'{pretrain_fold_dir}/{x}') and 'ckpt' in x]
            optimal_model_path = f'{pretrain_fold_dir}/{[x for x in ckpts if "last" not in x][0]}'
            resuming_finetuning_checkpoint = False

        print("loading model from:", optimal_model_path)

        if is_checkpoint_corrupted(optimal_model_path):
            raise RuntimeError(f"Checkpoint {optimal_model_path} is corrupted. Please check checkpoint files.")

        if advanced_model in ['default', 'duett']:
            if modules == 'all' and advanced_model == 'duett':
                raise NotImplementedError("DuETT not implemented for all modules")
            if not resuming_finetuning_checkpoint and med_as_feature:
                print("custom load: med_as_feature")
                # load_state_dict is overridden with strict=False
                model = SPARC.custom_load_from_checkpoint(
                    optimal_model_path, load_w_meds=False,
                    modular_featuredim_dict=modular_featuredim_dict, option=model_option,
                    advanced_model=advanced_model, max_time_len=max_time_len_data,
                    med_as_feature=med_as_feature, meds_l=med_of_choice,
                    loss_option=loss_option,
                    schedule = schedule_option,
                    use_sigmoid=use_sigmoid,
                    temporal_aggregation=time_processing,
                    missing_mask=missing_mask,
                    module_index_cfg=module_index_cfg,
                    use_module_tokenizer=use_module_tokenizer,
                    use_continuous_time=use_continuous_time,
                    n_static_features=n_static_features,
                    post_mlp_dropout=post_mlp_dropout,
                    batch_first=batch_first,
                    time_as_raw_feature=time_as_raw_feature,
                    lr=learning_rate,
                    binary_pos_weight=binary_pos_weight,
                    side_effects_pos_weight=side_effects_pos_weight,
                )
            elif (not resuming_finetuning_checkpoint
                  and model_option in ['side_effects', 'side_effects_risk', 'side_effects_next_risk']
                  and add_num_infusion):
                print("custom load: side_effects_num_infusion")
                # load_state_dict is overridden with strict=False
                model = SPARC.custom_load_from_checkpoint_sideeffect(
                    optimal_model_path,
                    modular_featuredim_dict=modular_featuredim_dict, option=model_option,
                    advanced_model=advanced_model, max_time_len=max_time_len_data,
                    med_as_feature=med_as_feature, meds_l=med_of_choice,
                    side_effects_num_infusion=True,
                    loss_option=loss_option,
                    schedule = schedule_option,
                    include_gcsf=include_gcsf, use_gcsf=include_gcsf, use_sigmoid=use_sigmoid,
                    temporal_aggregation=time_processing,
                    missing_mask=missing_mask,
                    module_index_cfg=module_index_cfg,
                    use_module_tokenizer=use_module_tokenizer,
                    use_continuous_time=use_continuous_time,
                    n_static_features=n_static_features,
                    post_mlp_dropout=post_mlp_dropout,
                    batch_first=batch_first,
                    time_as_raw_feature=time_as_raw_feature,
                    lr=learning_rate,
                    binary_pos_weight=binary_pos_weight,
                    side_effects_pos_weight=side_effects_pos_weight,
                )
            else:
                print("standard load:", model_option, "add_num_infusion", add_num_infusion, "med_as_feature", med_as_feature)
                # load_state_dict uses strict=False: Gemma weights are not saved in checkpoints
                model = SPARC.load_from_checkpoint(
                    optimal_model_path,
                    modular_featuredim_dict=modular_featuredim_dict, option=model_option,
                    advanced_model=advanced_model, max_time_len=max_time_len_data,
                    med_as_feature=med_as_feature, meds_l=med_of_choice,
                    loss_option=loss_option,
                    schedule = schedule_option,
                    side_effects_num_infusion=add_num_infusion,
                    include_gcsf=include_gcsf, use_gcsf=include_gcsf, use_sigmoid=use_sigmoid,
                    temporal_aggregation=time_processing,
                    missing_mask=missing_mask,
                    module_index_cfg=module_index_cfg,
                    use_module_tokenizer=use_module_tokenizer,
                    use_continuous_time=use_continuous_time,
                    n_static_features=n_static_features,
                    post_mlp_dropout=post_mlp_dropout,
                    batch_first=batch_first,
                    time_as_raw_feature=time_as_raw_feature,
                    lr=learning_rate,
                    binary_pos_weight=binary_pos_weight,
                    side_effects_pos_weight=side_effects_pos_weight,
                )


resume_ckpt_path = f'{output_model_path}/fold{valfold}/last.ckpt'
resume = os.path.exists(resume_ckpt_path) and not is_checkpoint_corrupted(resume_ckpt_path)

# If last.ckpt is corrupted, fall back to best epoch checkpoint
if os.path.exists(resume_ckpt_path) and is_checkpoint_corrupted(resume_ckpt_path):
    print(f"WARNING: Resume checkpoint {resume_ckpt_path} is corrupted. Cannot resume from last checkpoint.")
    os.remove(resume_ckpt_path)

    # best epoch checkpoint to resume from, if any
    fold_dir = f'{output_model_path}/fold{valfold}'
    if os.path.exists(fold_dir):
        ckpts = [x for x in os.listdir(fold_dir)
                 if x != 'last.ckpt' and not os.path.isdir(f'{fold_dir}/{x}') and '.ckpt' in x]
        if ckpts:
            resume_ckpt_path = f'{fold_dir}/{ckpts[0]}'
            print(f"Resuming from best epoch checkpoint instead: {resume_ckpt_path}")
            resume = True
        else:
            print("No valid epoch checkpoint found.")
            resume = False
    else:
        print("Fold directory does not exist yet.")
        resume = False

# Output directory + trainer setup
output_model_path = f'{output_model_path}/fold{valfold}'
os.makedirs(output_model_path, exist_ok=True)

logger_csv = pl.loggers.CSVLogger(f"{output_model_path}/logs_csv", name="lightning_logs_csv")
logger_tb  = pl.loggers.TensorBoardLogger(f"{output_model_path}/logs_tb",  name="lightning_logs_tb")

# side-effect tasks: checkpoint on max val_auroc_nextinf_weighted; others on val_loss
if bool(re.match(r'^(anemia|neutropenia|febrile_neutropenia|AKI|Liver_toxicity)(_next_risk|_risk)?$', option)):
    monitor_metric = 'val_auroc_nextinf_weighted'
    mode_metric = 'max'
else:
    monitor_metric = 'val_loss'
    mode_metric = 'min' 
    print("Using val_loss as checkpoint metric for non-side-effect task")

checkpoint = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    mode=mode_metric,
    monitor=monitor_metric,
    dirpath=output_model_path,
    filename='epoch={epoch}-step={step}',
    enable_version_counter=False,  # no -v1, -v2 suffixes
)

callbacks_list = [checkpoint]

if early_stop_patience > 0:
    print(f"Adding EarlyStopping with patience={early_stop_patience}, monitoring {monitor_metric} ({mode_metric})")
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=early_stop_patience,
        mode=mode_metric,
        verbose=True,
        min_delta=0.0001
    )
    callbacks_list.append(early_stop_callback)
else:
    print("Early stopping disabled (patience=0)")

print("initiating trainer")
print("resume:", resume)

if mode != 'train':
    existing = os.listdir(f'{output_model_path}/')
    epoch_ckpts = [el for el in existing if 'epoch' in el]
    if epoch_ckpts:
        optimal_model_path = os.path.join(output_model_path, epoch_ckpts[0])
        print("optimal_model_path", optimal_model_path)

trainer = pl.Trainer(
    max_epochs=epochs,
    logger=[logger_csv, logger_tb],
    callbacks=callbacks_list,
    deterministic=True,
    accumulate_grad_batches=10,
    gradient_clip_val=0.5,
    gradient_clip_algorithm='norm',
    accelerator=args.accelerator,
    devices=1,   # the model does not support DDP (unused parameters)
)
pl.seed_everything(rseed)

# test pass only when not training
if mode != 'train':
    trainer.test(model, data_module.test_dataloader())

# Train
if mode == 'train':
    if resume:
        # re-check before resuming
        if is_checkpoint_corrupted(resume_ckpt_path):
            print(f"ERROR: Checkpoint {resume_ckpt_path} is corrupted. Cannot resume training.")
            raise RuntimeError(f"Corrupted checkpoint: {resume_ckpt_path}")
        print(f"Resuming training from: {resume_ckpt_path}")
        trainer.fit(model, data_module, ckpt_path=resume_ckpt_path)
    else:
        trainer.fit(model, data_module)

    # keep only the best checkpoint across all runs
    print("Cleaning up checkpoints - keeping only the overall best...")
    epoch_ckpts = [f for f in os.listdir(output_model_path)
                   if f.startswith('epoch=') and f.endswith('.ckpt')]

    if len(epoch_ckpts) > 1:
        # find the checkpoint with the lowest validation loss
        best_val_loss = float('inf')
        best_ckpt = None

        for ckpt_file in epoch_ckpts:
            ckpt_path = os.path.join(output_model_path, ckpt_file)
            try:
                ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                # ModelCheckpoint stores the monitored metric in the checkpoint
                if 'callbacks' in ckpt_data:
                    for cb_state in ckpt_data['callbacks'].values():
                        if 'best_model_score' in cb_state:
                            val_loss = cb_state['best_model_score'].item()
                            print(f"  {ckpt_file}: val_loss={val_loss:.6f}")
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_ckpt = ckpt_file
                            break
            except Exception as e:
                print(f"  Warning: Could not load {ckpt_file}: {e}")

        # delete the rest
        if best_ckpt:
            print(f"Keeping best checkpoint: {best_ckpt} (val_loss={best_val_loss:.6f})")
            for ckpt_file in epoch_ckpts:
                if ckpt_file != best_ckpt:
                    ckpt_path = os.path.join(output_model_path, ckpt_file)
                    print(f"  Removing: {ckpt_file}")
                    os.remove(ckpt_path)
            optimal_model_path = os.path.join(output_model_path, best_ckpt)
        else:
            print("Warning: Could not determine best checkpoint, keeping all")
            optimal_model_path = checkpoint.best_model_path
    else:
        optimal_model_path = checkpoint.best_model_path

    trainer.test(model, data_module.test_dataloader())
    print("training finished")


# Final evaluation with the best checkpoint
print("best checkpoint path:", checkpoint.best_model_path, optimal_model_path)
print("loss_option:", loss_option, " | include_gcsf:", include_gcsf)
model = SPARC.load_from_checkpoint(
    optimal_model_path,
    modular_featuredim_dict=modular_featuredim_dict, option=model_option,
    advanced_model=advanced_model, max_time_len=max_time_len_data,
    med_as_feature=med_as_feature, meds_l=med_of_choice,
    loss_option=loss_option,
    schedule = schedule_option,
    side_effects_num_infusion=add_num_infusion,
    include_gcsf=include_gcsf, use_gcsf=include_gcsf, use_sigmoid=use_sigmoid,
    temporal_aggregation=time_processing,
    missing_mask=missing_mask,
    module_index_cfg=module_index_cfg,
    use_module_tokenizer=use_module_tokenizer,
    use_continuous_time=use_continuous_time,
    n_static_features=n_static_features,
    post_mlp_dropout=post_mlp_dropout,
    batch_first=batch_first,
    time_as_raw_feature=time_as_raw_feature,
    lr=learning_rate,
    binary_pos_weight=binary_pos_weight,
    side_effects_pos_weight=side_effects_pos_weight,
)
pl.seed_everything(rseed)
trainer.test(model, data_module.test_dataloader())
if model_option != 'pretrain':
    write_test_predictions(model, data_module.test_dataset,
                           f'{output_model_path}/test_predictions.csv', batch_size, rseed)
