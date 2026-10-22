"""Filesystem roots for the repo. The defaults are placeholders: set SPARC_DATA_PATH, SPARC_MODEL_PATH and
SPARC_RESULTS_PATH (and the other SPARC_* variables below) to your own locations.

Scripts put the repo root on sys.path (walking up from __file__ to the dir holding data/ and models/)
before `import sparc_paths`. Run directly to print the resolved paths.
"""

import os

# repo layout, derived from this file
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
BENCHMARKS_DIR = os.path.join(MODELS_DIR, 'benchmarks')
HYPERPARAM_DIR = os.path.join(REPO_ROOT, 'hyperparam_tune')

# data, model and result roots (placeholders; set the environment variables)
DATA_ROOT = os.environ.get(
    'SPARC_DATA_PATH',
    '/path/to/SPARC_data')

MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_PATH',
    '/path/to/SPARC_models')

RESULTS_ROOT = os.environ.get(
    'SPARC_RESULTS_PATH',
    '/path/to/SPARC_results')

BENCHMARK_MODEL_ROOT = os.environ.get('SPARC_BENCHMARK_MODEL_PATH', RESULTS_ROOT)
# batch_first=True MSK-CAIA SPARC / SPARC-P models
SPARC_BF_MODEL_ROOT = os.environ.get(
    'SPARC_BF_MODEL_PATH', '/path/to/SPARC_bf_models')
SPARC_MODEL_ROOT = os.environ.get('SPARC_060726_MODEL_PATH', SPARC_BF_MODEL_ROOT)
BATCH_FIRST_DEFAULT = True
# bf=True evaluation results and figures
SPARC_RESULTS_VIS_ROOT = os.environ.get(
    'SPARC_RESULTS_VIS_PATH', os.path.join(SPARC_BF_MODEL_ROOT, 'results_vis_eval1_bfTrue'))
SPARC_LIME_BF_DIR = os.environ.get(
    'SPARC_LIME_BF_DIR', os.path.join(SPARC_RESULTS_VIS_ROOT, 'lime_fixed_bfTrue'))
SPARC_TITRATION_RESULTS_DIR = os.environ.get(
    'SPARC_TITRATION_RESULTS_DIR',
    os.path.join(SPARC_RESULTS_VIS_ROOT, 'results_trainsize_bfTrue'))


# derived locations, each overridable
COHORT_SUMMARIES_DIR = os.environ.get(
    'SPARC_COHORT_SUMMARIES_DIR',
    os.path.join(RESULTS_ROOT, 'results_vis', 'cohort_summaries'))

DATASET_SUMMARIES_DIR = os.environ.get(
    'SPARC_DATASET_SUMMARIES_DIR',
    os.path.join(COHORT_SUMMARIES_DIR, 'dataset_summaries'))

CANCER_TYPES_DIR = os.environ.get(
    'SPARC_CANCER_TYPES_DIR',
    os.path.join(COHORT_SUMMARIES_DIR, 'cancer_types'))

CANCER_TYPES_OUT_DIR = os.environ.get('SPARC_CANCER_TYPES_OUT', CANCER_TYPES_DIR)

RACE_ETHNICITY_DIR = os.environ.get(
    'SPARC_RACE_ETHNICITY_DIR',
    os.path.join(COHORT_SUMMARIES_DIR, 'race_ethnicity'))

RACE_ETHNICITY_HARMONIZED_DIR = os.environ.get(
    'SPARC_RACE_ETHNICITY_HARMONIZED_DIR',
    os.path.join(RACE_ETHNICITY_DIR, 'harmonized'))

BENCHMARK_CONFIG_DIR = os.environ.get(
    'SPARC_BENCHMARK_CONFIG_DIR',
    os.path.join(HYPERPARAM_DIR, 'benchmark_tune', 'benchmark_configs_nonfill'))

# Genomic feature table, which sits beside the data root rather than inside it.
GENOMIC_FT_CSV = os.environ.get(
    'SPARC_GENOMIC_FT',
    os.path.join(os.path.dirname(os.path.normpath(DATA_ROOT)),
                 'impact_genomic_ft', 'output_050725', 'impact_genomic_ft_341.csv'))

# Raw MIMIC-IV release, used only by data/MIMICIV_preprocessing/.
MIMIC_RAW_DIR = os.environ.get(
    'SPARC_MIMIC_RAW', os.path.join(os.path.dirname(os.path.normpath(DATA_ROOT)),
                                    'mimic_iv_3.1_mult', 'hosp'))

# Raw Stanford EHRSHOT release (EHRSHOT_ASSETS), used only by data/EHRSHOT_preprocessing/.
EHRSHOT_ASSETS_DIR = os.environ.get(
    'SPARC_EHRSHOT_ASSETS', os.path.join(os.path.dirname(os.path.normpath(DATA_ROOT)),
                                         'EHRSHOT_ASSETS'))

# roots for rarely used scripts; kept configurable

# eval_1, MIMICIV_hosp + XGBoost + -z False only (not reached by default)
SL_TESTSL_ROOT = os.environ.get(
    'SPARC_SL_TESTSL_ROOT', '/path/to/SPARC_testsl_models')

# inputs to the one-off data/data_split.py
FROZEN_DATA_DIR = os.environ.get(
    'SPARC_FROZEN_DATA_DIR', '/path/to/frozen_data')
CTFM_DIR = os.environ.get('SPARC_CTFM_DIR', '/path/to/CTFoundationModel')


def benchmark_model_dir(model_option):
    """models/benchmarks/<model_option>/ -- the per-architecture class modules."""
    return os.path.join(BENCHMARKS_DIR, model_option)


def add_benchmark_to_path(model_option):
    """Put a benchmark's module directory on sys.path."""
    import sys
    p = benchmark_model_dir(model_option)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def describe():
    """Print the resolved paths, marking ones missing on this filesystem."""
    rows = [
        ('REPO_ROOT', REPO_ROOT, None),
        ('DATA_ROOT', DATA_ROOT, 'SPARC_DATA_PATH'),
        ('MODEL_ROOT', MODEL_ROOT, 'SPARC_MODEL_PATH'),
        ('RESULTS_ROOT', RESULTS_ROOT, 'SPARC_RESULTS_PATH'),
        ('SPARC_MODEL_ROOT', SPARC_MODEL_ROOT, 'SPARC_060726_MODEL_PATH'),
        ('COHORT_SUMMARIES_DIR', COHORT_SUMMARIES_DIR, 'SPARC_COHORT_SUMMARIES_DIR'),
        ('DATASET_SUMMARIES_DIR', DATASET_SUMMARIES_DIR, 'SPARC_DATASET_SUMMARIES_DIR'),
        ('CANCER_TYPES_DIR', CANCER_TYPES_DIR, 'SPARC_CANCER_TYPES_DIR'),
        ('RACE_ETHNICITY_DIR', RACE_ETHNICITY_DIR, 'SPARC_RACE_ETHNICITY_DIR'),
        ('RACE_ETHNICITY_HARMONIZED_DIR', RACE_ETHNICITY_HARMONIZED_DIR,
         'SPARC_RACE_ETHNICITY_HARMONIZED_DIR'),
        ('BENCHMARK_CONFIG_DIR', BENCHMARK_CONFIG_DIR, 'SPARC_BENCHMARK_CONFIG_DIR'),
        ('GENOMIC_FT_CSV', GENOMIC_FT_CSV, 'SPARC_GENOMIC_FT'),
        ('MIMIC_RAW_DIR', MIMIC_RAW_DIR, 'SPARC_MIMIC_RAW'),
        ('EHRSHOT_ASSETS_DIR', EHRSHOT_ASSETS_DIR, 'SPARC_EHRSHOT_ASSETS'),
        ('SL_TESTSL_ROOT', SL_TESTSL_ROOT, 'SPARC_SL_TESTSL_ROOT'),
        ('FROZEN_DATA_DIR', FROZEN_DATA_DIR, 'SPARC_FROZEN_DATA_DIR'),
        ('CTFM_DIR', CTFM_DIR, 'SPARC_CTFM_DIR'),
    ]
    w = max(len(n) for n, _, _ in rows)
    print('SPARC resolved paths (* = missing on this filesystem)\n')
    for name, value, env in rows:
        mark = ' ' if os.path.exists(value) else '*'
        src = f'  [${env}]' if env and os.environ.get(env) else ''
        print(f'{mark} {name:<{w}}  {value}{src}')


if __name__ == '__main__':
    describe()
