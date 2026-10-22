# SPARC
Code repository for SPARC: Longitudinal AI Models Predict Cancer Adverse Events

## Install

```bash
pip install -e .            # installs the `sparc` command
```

`sparc train` and `sparc eval` run `model_train/SPARC_train_inf_risk_fixed.py`, so they take
its flags, or a YAML config (`examples/`). Input data can be the original JSON bundle
(`DMP_ID_to_dicts_final.json`), the same bundle pickled, or CSV or Parquet tables (`patients`,
`labs`, `vitals`, `medications`, `admissions`). The
documentation (`pip install -e ".[docs]" && mkdocs serve`) covers the formats, a demo on
synthetic data, training, inference and SLURM.

## Layout

```
sparc_paths.py          all filesystem roots, overridable by environment variable
data/                   cohort loading and featurisation, plus MIMIC-IV and EHRSHOT preprocessing
models/                 the SPARC model and the deep-learning benchmarks
model_train/            training entry points
hyperparam_tune/        benchmark hyperparameter search
results_vis/            evaluations (eval_*), design-choice ablations, and figures (vis_*)
interpretation/         LIME feature importance and treatment-effect analyses
cohort_summaries/       descriptive statistics and Table 1
src/sparc_clinical/     the `sparc` command and Python package, a thin layer over the code above
tests/                  pytest suite (`pytest`), runs on CPU with synthetic data
docs/, mkdocs.yml       documentation site
examples/               YAML configs for pretraining and finetuning
```

Every path lives in `sparc_paths.py` and can be redirected without editing code:

```bash
export SPARC_DATA_PATH=/path/to/SPARC_data
export SPARC_MODEL_PATH=/path/to/SPARC_fixed_models
export SPARC_RESULTS_PATH=/path/to/results
```

Most scripts also take `-dsp`, `-mp` and `-rp`/`-op` to override the same roots per run.
INSIGHT external-validation code and preprocessing: see the
[`external-validation-insight` branch](https://github.com/tommyfuu/SPARC/tree/external-validation-insight).

Defaults that are placeholders (`/path/to/...`) must be set before the script runs. The remaining
variables, each read by the scripts named:

| variable | used by |
|---|---|
| `SPARC_COHORT_SUMMARIES_DIR` | `cohort_summaries/demographic_tables/dem_table_4_build_table.py`, `cohort_summaries/dataset_summaries/dataset_summaries_combined_plot.py` (output) |
| `SPARC_DATASET_SUMMARIES_DIR` | `cohort_summaries/dataset_summaries/dataset_summaries_combined_plot.py` |
| `SPARC_RACE_ETHNICITY_DIR` | `cohort_summaries/demographic_tables/dem_table_3_harmonize_race_ethnicity.py` |
| `SPARC_CANCER_TYPES_DIR`, `SPARC_CANCER_TYPES_OUT` | `cohort_summaries/cancer_types/ctypes_parse.py` |
| `SPARC_MIMIC_RAW`, `SPARC_MIMIC_OUT` | `cohort_summaries/cancer_types/check_cancer_types_mimic.py` |

The two INSIGHT cancer-type scripts take their locations as required flags instead:

```bash
python cohort_summaries/cancer_types/check_cancer_types_insight.py \
    --raw-dir <raw INSIGHT CSVs> --processed-dir <processed per-site pickles>
python cohort_summaries/insight_cancer_types/cancer_type_summary.py --pickle-dir <dir> [-od <reports dir>]
```

## data/

| file | what it does |
|---|---|
| `dataset_loading.py` | `load_dataset()` — the single entry point every script uses to load a cohort |
| `bundle_io.py` | reads and writes the `*_CAIA` patient bundle as JSON, pickle, CSV or Parquet |
| `data_loaders_inf_risk_fixed.py` | `FlexibleFeatureDataModule` / `FlexibleFeatureDataset`: turns per-patient records into model tensors |
| `data_loader_side_effect_infusions_utils_risk_fixed.py` | side-effect label construction (event, time-to-event, infusion number) |
| `MSK_featurename_list.py` | the canonical feature lists and module definitions |
| `data_split.py` | patient-level train/val/test splits |

`data/MIMICIV_preprocessing/` builds the MIMIC-IV external-validation cohort. Steps run in
order and each reads the previous one's output; see its own `README.md`.

```bash
cd data/MIMICIV_preprocessing && python 1_extract_cancer_patients.py
```

`data/EHRSHOT_preprocessing/` builds the Stanford EHRSHOT external-validation cohort the same
way; see its own `README.md`.

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

# shallow benchmarks (temporal_xgboost | lasttp_xgboost)
python3 SL_benchmarks_train_risk_corrected_fixed.py -mo temporal_xgboost -ds MSK_CAIA -o AKI -f 1

# TabPFN benchmark (needs TABPFN_TOKEN, see below)
python3 TabPFNv3_benchmarks_train_risk_corrected_fixed.py -mo temporal_tabpfn -ds MSK_CAIA -o binary -f 1
```

`-s pretrain | finetune`, `-f` is the fold (1-10), `-umt`/`-uct`/`-wbce` switch the
module tokenizer, continuous-time encoding and weighted BCE for the ablation arms. `-bs` sets
the batch size and `-acc` the accelerator. Supervised runs also write `test_predictions.csv`.

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
| `eval_0_pretrained_vs_baselines_mses.py` | per-feature MSE of the pretrained model vs naive and regression baselines |
| `eval_0_mse_pairwise_test.py` | paired Wilcoxon test of per-feature MSE, SPARC-P ensemble vs one baseline |
| `eval_0_pretrained_embedding_umapoption.py` | 2D projection of the pretrained embeddings |
| `eval_1_across_models_...py` | the main benchmark: every model, every task |
| `eval_2_across_medications_...py` | performance by medication regimen |
| `eval_3_across_cancer_types_...py` | performance by cancer type |
| `eval_4_titration_...py` | direct vs pretrained-then-finetuned across training-set sizes |
| `eval_5_SPARC_P_vs_*.py` | head-to-head against four published clinical scores |
| `design_choice_comparison_{1..6}_*.py` | ablations: loss function, medication scope, meds-as-features, parsimony, module tokenizer, time representation (continuous-time encoding vs raw time vs no time) |

```bash
# one task, one dataset
# (SPARC / SPARC-P as 10-fold ensembles, each benchmark as its fold-0 model)
python3 eval_1_across_models_comparison_ensemble_final_FP_fixed.py \
    -ds MSK_CAIA -o AKI -d 180 -m parsimonious -an True -cp prior \
    -me "CISPLATIN,IFOSFAMIDE,..." -rp <results_dir>

# figure from those CSVs: ensemble bars, 95% bootstrap CIs, paired bootstrap or DeLong tests
python3 vis_eval_1_across_tasks_t1e_controlled_bar_horizontal_succinct_wilcoxon_all_final.py \
    --dataset MSK_CAIA --stat_test delong --style simple --save_dir <dir>
```

## interpretation/

```bash
# LIME feature importance, averaged across folds
python3 interp_fix_0622626/lime/lime_importance_risk_fixed.py \
    -o neutropenia -ds MSK_CAIA -m parsimonious -me "..." -an True -op <dir>

# does G-CSF reduce neutropenia, by SPARC risk tertile: IPTW (overlap weights; main analysis)
python3 interp_fix_0622626/gcsf_analysis_090826/risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_101326_iptw.py \
    -o neutropenia -ds MSK_CAIA -ns 3 -gl 2 -bf True -mp <model dir>

# sensitivity analysis: PSM exact on risk tertile and infusion cycle, 1:k with caliper
python3 interp_fix_0622626/gcsf_analysis_090826/risk_wandwo_gcsf_usingdataloader_SPARC_recalibrated_with_psm_101426_cyclematch.py \
    -bf True -mp <model dir> -mr 1 2 10
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

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License](https://creativecommons.org/licenses/by-nc-nd/4.0/) (CC BY-NC-ND 4.0).
See [`LICENSE`](LICENSE).
