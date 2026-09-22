# SPARC
Code repository for SPARC: Longitudinal AI Models Predict Cancer Adverse Events

## Layout

```
sparc_paths.py          all filesystem roots, overridable by environment variable
data/                   cohort loading and featurisation, plus MIMIC-IV preprocessing
models/                 the SPARC model and the deep-learning benchmarks
model_train/            training entry points
hyperparam_tune/        benchmark hyperparameter search
results_vis/            evaluations (eval_*), design-choice ablations, and figures (vis_*)
interpretation/         LIME feature importance and treatment-effect analyses
cohort_summaries/       descriptive statistics and Table 1
```

Every path lives in `sparc_paths.py` and can be redirected without editing code:

```bash
export SPARC_DATA_PATH=/path/to/SPARC_data
export SPARC_MODEL_PATH=/path/to/SPARC_fixed_models
export SPARC_RESULTS_PATH=/path/to/results
```

Most scripts also take `-dsp`, `-mp` and `-rp`/`-op` to override the same roots per run.

## data/

| file | what it does |
|---|---|
| `dataset_loading.py` | `load_dataset()` — the single entry point every script uses to load a cohort |
| `data_loaders_inf_risk_fixed.py` | `FlexibleFeatureDataModule` / `FlexibleFeatureDataset`: turns per-patient records into model tensors |
| `data_loader_side_effect_infusions_utils_risk_fixed.py` | side-effect label construction (event, time-to-event, infusion number) |
| `MSK_featurename_list.py` | the canonical feature lists and module definitions |
| `data_split.py` | patient-level train/val/test splits |

`data/MIMICIV_preprocessing/` builds the MIMIC-IV external-validation cohort. Steps run in
order and each reads the previous one's output; see its own `README.md`.

```bash
cd data/MIMICIV_preprocessing && python 1_extract_cancer_patients.py
```

## models/

| file | what it does |
|---|---|
| `SPARC_inf_risk_modular.py` | the SPARC model architecture |
| `benchmarks/duett/duett.py` | DuETT benchmark |
| `benchmarks/duett/duett_model.py` | DuETT encoder: DuETT helper script |
| `benchmarks/lasttp_MLP/lasttp_MLP.py` | last-timepoint MLP benchmark |
| `benchmarks/rnn/rnn.py` | RNN benchmark |
| `benchmarks/raindrop/raindrop.py` | Raindrop benchmark (`ob_propagation.py` is its graph layer) |

## model_train/

```bash
# pretrain, then finetune SPARC on one task and fold
python3 SPARC_train_inf_risk_fixed.py -ds MSK_CAIA -f 1 -s finetune \
    -o febrile_neutropenia_risk -n 20 -m parsimonious -cp prior \
    -pp <pretrained_dir> -op <output_dir> \
    -me "DOCETAXEL,CISPLATIN,..." -an True -tp attention \
    -umt True -uct True -pmd 0.2 -lo hybrid -e 200 -esp 30

# deep-learning benchmarks (duett | lasttp_MLP | rnn | raindrop)
python3 DL_benchmarks_train_risk_corrected_fixed.py -mo duett -ds MSK_CAIA -o AKI_risk -f 1

# shallow benchmarks (XGBoost variants)
bash SL_benchmarks_train_risk_corrected_fixed_091626.sh

# TabPFN benchmark (needs TABPFN_TOKEN, see below)
python3 TabPFNv3_benchmarks_train_risk_corrected_fixed.py -mo temporal_tabpfn -ds MSK_CAIA -o binary -f 1
```

`-s pretrain | finetune`, `-f` is the fold (1-10), `-umt`/`-uct`/`-wbce` switch the
module tokenizer, continuous-time encoding and weighted BCE for the ablation arms.

TabPFN v3 requires a one-time licence acceptance at <https://ux.priorlabs.ai> and
`export TABPFN_TOKEN=...` before it can download model weights.

## hyperparam_tune/benchmark_tune/

Two stages. The tuner trains one grid point per invocation; the collector reads the
sweep and writes the best config per task.

```bash
# one grid point (submit as an array over -i)
python3 benchmark_tune_task_risk_fixed.py -mo duett -o AKI_risk -i 1

# pick the winners -> benchmark_configs_nonfill/duett_best_hyperparams.json
python3 benchmark_identify_bestparams_task_risk_fixed.py -mo duett
```

`benchmark_search_spaces.py` holds the grids and the run-directory naming, shared by
both so they cannot disagree about where a run lives.

## results_vis/

Each `eval_N` writes CSVs; the matching `vis_eval_N` turns them into figures. All the
`vis_` scripts take `--save_dir` so figures can be regenerated without overwriting
published ones.

| script | what it measures |
|---|---|
| `eval_0_pretrained_mse_vs_chance.py` | per-feature MSE of the pretrained model vs a random-chance baseline |
| `eval_0_mse_pairwise_test.py` | paired t-test over those per-feature MSEs |
| `eval_0_pretrained_embedding_umapoption.py` | 2D projection of the pretrained embeddings |
| `eval_1_across_models_...py` | the main benchmark: every model, every task |
| `eval_2_across_medications_...py` | performance by medication regimen |
| `eval_3_across_cancer_types_...py` | performance by cancer type |
| `eval_4_titration_...py` | direct vs pretrained-then-finetuned across training-set sizes |
| `eval_5_relative_risk_...py` | relative risk of the top-k highest-risk patients |
| `eval_6_calibration_...py` | reliability curves, Brier / ECE / MCE, Platt scaling |
| `eval_7_SPARC_P_vs_*.py` | head-to-head against four published clinical scores |
| `design_choice_comparison_1..5*.py` | ablations: loss function, medication scope, meds-as-features, parsimony, module tokenizer |

```bash
# one task, one dataset
python3 eval_1_across_models_comparison_ensemble_FP_fixed.py \
    -ds MSK_CAIA -o AKI -d 180 -m parsimonious -an True -cp prior \
    -me "CISPLATIN,IFOSFAMIDE,..." -rp <results_dir>

# the whole 4 datasets x 7 tasks grid
sbatch submit_eval_1_all_tasks.sh

# figure from those CSVs
python3 vis_eval_1_across_tasks_t1e_controlled_bar_horizontal_succinct_wilcoxon_all.py \
    --dataset MSK_CAIA --stat_test wilcoxon --correction bonferroni --save_dir <dir>
```

## interpretation/

```bash
# LIME feature importance, averaged across folds
python3 interp_fix_0622626/lime/lime_importance_risk_fixed.py \
    -o neutropenia -ds MSK_CAIA -m parsimonious -me "..." -an True -op <dir>

# does G-CSF reduce neutropenia, stratified by predicted risk, with PSM
python3 interp_fix_0622626/gcsf_analysis_090826/risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_062526.py \
    -o neutropenia -ds MSK_CAIA -ns 5 -gl 2 -an True -me "..."
```

## cohort_summaries/

| directory | what it produces |
|---|---|
| `dataset_summaries/` | per-cohort size and outcome-rate summaries and their combined plot |
| `demographic_tables/` | Table 1, built in four numbered steps |
| `feature_correlation_analysis/` | pairwise feature correlation heatmap |
| `cv_analysis/` | coefficient-of-variation analysis, ordered to match the correlation heatmap |
| `cancer_types/`, `insight_cancer_types/` | cancer-type extraction and summaries |

```bash
python3 cohort_summaries/cv_analysis/cv_analysis_fixed.py
python3 cohort_summaries/demographic_tables/dem_table_1_dataset_summaries.py
```
