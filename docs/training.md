# Training and inference

`sparc train` and `sparc eval` run `model_train/SPARC_train_inf_risk_fixed.py` with the flags
you give; `sparc train --help` lists them all. Running the script with `python` directly works
the same way. Each run uses one device: a GPU if there is one, else the CPU (`-acc gpu` or
`-acc cpu` forces it).

## Folds

Patients are split into 11 folds: 0–9 for cross-validation and 10 as the held-out test set.
`-f 1` to `-f 10` picks the validation fold (`-f 1` validates on fold 0 and trains on folds
1–9). Each run trains one fold and writes to `<output>/fold<f-1>/` (`-f 1` writes `fold0`), so a
full cross-validation is ten runs, for example a SLURM array.

## Pretrain

```bash
sparc train --config examples/pretrain.yaml -f 1
```

`-s pretrain -o pretrain` trains the model to predict each patient's next visit from the
previous ones, without labels.

## Finetune

```bash
sparc train --config examples/finetune.yaml -f 1
```

`-s finetune -pp <dir>` starts from a pretrained model. `<dir>` is the pretraining run's
output directory (the parent of `fold0` … `fold9`), and each fold starts from the pretrained
model of the same fold. Without `-pp` the model is trained from scratch. The MSK-CAIA pretrained
model is not distributed with the repository; on the MSK cluster it is
`parsimonious_MSK_CAIA_060726/parsimonious_pretrain_20tp_augmented_att_modular` under
`sparc_paths.SPARC_MODEL_ROOT`, and it needs the model settings in `examples/finetune.yaml`.

## Evaluate

```bash
sparc eval --config examples/finetune.yaml -f 1
```

This loads the best checkpoint in the fold's output folder, tests it on fold 10 and logs the
metrics; it stops if the fold has no trained checkpoint. Give it the same model flags as the
training run, because checkpoints do not store them. For supervised tasks, training and
evaluation both write `test_predictions.csv`:

| Column | Meaning |
|---|---|
| `patient_id` | Patient. |
| `prediction_day` | Day the prediction is made (discharge, ICU admission, or infusion). |
| `infusion_num` | Infusion number, for side-effect tasks. |
| `label`, or `event` and `time_to_event` | Outcome; `_risk` tasks use the pair. |
| `score` | Predicted probability (binary tasks) or risk score (`_risk` tasks). |

!!! note
    The test metrics in the log average per-batch values; the paper, like any metric computed
    over the whole CSV, pools all samples. If the sample count leaves fewer than `-bs`/5 in the
    last batch, the last samples are not scored (the log says `Trimming test_dataset`), so lower
    `-bs` for small cohorts.

The paper reports the mean score of the ten fold models:

```python
import glob, pandas as pd
folds = [pd.read_csv(f) for f in sorted(glob.glob('runs/neutropenia/fold*/test_predictions.csv'))]
keys = [c for c in folds[0].columns if c != 'score']
ensemble = pd.concat(folds).groupby(keys, sort=False, as_index=False)['score'].mean()
```

### Another cohort

`sparc eval` builds samples from outcomes, as training does, so it scores only samples whose
outcome can be worked out. To score a new cohort with a trained model:

1. Put it in its own folder, for example `NEW_CAIA`, with `fold` 10 and a `last_contact_day`
   for every patient.
2. Link the trained checkpoints into a new output folder, so the run's own predictions are not
   overwritten: `mkdir -p runs/new/fold0 && ln -s $PWD/runs/neutropenia/fold0/epoch=*.ckpt runs/new/fold0/`
3. Run `sparc eval` with the model's flags and `-ds NEW_CAIA -op runs/new -f 1`.

The cohort is z-scored with its own statistics, as in the paper's temporal validation. To use
the training cohort's instead, copy its `normalization_factor_dict_*.pickle` for the same `-m`
(for MSK_CAIA, `normalization_factor_dict_parsimonious_till2021_031226.pickle`) into the new
folder as `normalization_factor_dict_NEW_CAIA_<-m value>.pickle` before the first run.

Every patient needs a `last_contact_day`; a missing one stops the run. Side-effect tasks skip
patients with fewer than two measurements of the outcome's lab in the 180 days before their
first infusion, with the event already in that window, or with no such lab after it. They also
skip infusions after the first event, and the last infusion if follow-up ends within 30 days of
it (90 for `Liver_toxicity`). Binary tasks skip patients alive with less than `-d` days of
follow-up after the prediction day.

## Config files

A config file is YAML with the flags' long names. Flags on the command line override it.
Lists become comma-separated values and `true`/`false` become `True`/`False`.

```yaml
dataset: MYSITE_CAIA
volume_path: /path/to/data
scheme: finetune
option: neutropenia_risk
med_of_choice: [CISPLATIN, CARBOPLATIN]
add_num_infusion: true
```

`examples/pretrain.yaml` and `examples/finetune.yaml` hold the settings of the models in the
paper.

## Main flags

| Flag | Long name | Default | Meaning |
|---|---|---|---|
| `-ds` | `dataset` | `MSK` | Dataset folder under `-vp`, for example `MYSITE_CAIA`. |
| `-vp` | `volume_path` | `sparc_paths.DATA_ROOT` | Data root. |
| `-s` | `scheme` | `pretrain` | `pretrain` or `finetune`. |
| `-o` | `option` | `binary` | Task; see [what each task needs](data.md#what-each-task-needs). |
| `-f` | `valfold` | `1` | Validation fold, 1–10. |
| `-m` | `modules_cat` | `all` | Feature set. Use `parsimonious` (or `practical`); `all` needs MSK-only features. |
| `-me` | `med_of_choice` | `CISPLATIN` | Drugs for side-effect tasks, comma-separated. `A+B` is a combination regimen. |
| `-an` | `add_num_infusion` | `False` | Give the infusion number to the model. Side-effect tasks need `True`. |
| `-lo` | `loss_option` | `default` | `hybrid` adds a BCE term to the Cox loss of `_risk` tasks. |
| `-d` | `days` | `180` | Horizon of the binary tasks, in days. |
| `-pp` | `pretrained_model_path` | none | Pretraining output to finetune from. |
| `-op` | `output_model_path` | an MSK path | Output directory; always set it. |
| `-e`, `-esp` | `epochs`, `early_stop_patience` | `200`, `0` | Epochs, and patience for early stopping (0 turns it off). |
| `-n` | `num_tp` | `20` | Visits per sample. |
| `-tp` | `time_processing` | `attention` | How visits are pooled: `attention`, `minmaxmean` or `mlp`. |
| `-umt`, `-uct`, `-pmd` | `use_module_tokenizer`, `use_continuous_time`, `post_mlp_dropout` | `False`, `True`, model default | Model settings; the models in the paper use `True`, `True`, `0.2`. |
| `-bf` | `batch_first` | `True` | `True` makes the encoder attend over time, as in all models in the paper. Checkpoints record the value and refuse to load with the other one; older checkpoints without it load either way, so pass the value used in training. |
| `-bs` | `batch_size` | `512` | Batch size. |
| `-acc` | `accelerator` | `auto` | `auto`, `gpu` or `cpu`. |
| `-r` | `rseed` | `1024` | Random seed. |
| `-mo` | `mode` | `train` | `train` or `test`; `sparc eval` sets `test`. |

## Outputs

```
<output>/fold<f-1>/
    epoch=epoch=N-step=step=M.ckpt   best checkpoint
    last.ckpt                        latest checkpoint; rerunning the same command resumes from it
    test_predictions.csv             supervised tasks
    logs_csv/.../metrics.csv         training curves and test metrics
    logs_tb/                         the same, for TensorBoard
```

The best checkpoint has the lowest validation loss, or for side-effect tasks the highest
next-infusion AUROC.

## SLURM

One array task per fold, from the repository root (edit the partition and environment for your
cluster):

```bash
#!/bin/bash
#SBATCH --job-name=sparc_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/sparc_finetune_%A_%a.out
sparc train --config examples/finetune.yaml -f "$SLURM_ARRAY_TASK_ID" -acc gpu
```

Pretraining is the same with `examples/pretrain.yaml`; chain the two with
`sbatch --dependency=afterok:<pretrain job id>`. Finetuning already writes test predictions;
`sparc eval` with the same config re-evaluates a trained fold.
