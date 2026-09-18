"""
sparc_paths.py -- one place for every cluster-specific path in this repo.

Porting the repo to a new cluster should require editing this file, or
exporting the environment variables below. No other script should contain an
absolute path that cannot be overridden.

Environment variables
---------------------
    SPARC_DATA_PATH      root holding the per-dataset data volumes
                         (MSK_CAIA/, MIMICIV_hosp/, MSK/, stanfordEHRSHOT/, ...)
    SPARC_MODEL_PATH     root for models this repo trains
    SPARC_RESULTS_PATH   root for evaluation outputs and figures

Anything more specific is derived from those three, so a new cluster normally
only needs:

    export SPARC_DATA_PATH=/new/cluster/SPARC_data
    export SPARC_MODEL_PATH=/new/cluster/SPARC_models
    export SPARC_RESULTS_PATH=/new/cluster/SPARC_fixed_models

Importing
---------
A script cannot `import sparc_paths` until the repo root is on sys.path, so
each script starts with a short bootstrap that walks up from __file__ to the
directory containing data/ and models/. Copy this block verbatim:

    # --- SPARC bootstrap: cwd- and depth-independent imports -------------
    import os, sys
    _R = os.path.dirname(os.path.abspath(__file__))
    while _R != os.path.dirname(_R) and not all(
            os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
        _R = os.path.dirname(_R)
    for _p in (_R, os.path.join(_R, 'data'), os.path.join(_R, 'models')):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    # ---------------------------------------------------------------------

after which `import sparc_paths` and `from data_loaders_inf_risk_fixed import *`
both work from any working directory.

Why a walk rather than a fixed number of dirname() calls: scripts in this repo
sit at three different depths (model_train/, results_vis/test_fixed_060726/,
interpretation/interp_fix_0622626/lime/), and files get moved between them.
"""

import os

# ---------------------------------------------------------------------------
# Repo layout (derived from this file's location -- never configure this)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
BENCHMARKS_DIR = os.path.join(MODELS_DIR, 'benchmarks')
HYPERPARAM_DIR = os.path.join(REPO_ROOT, 'hyperparam_tune')

# ---------------------------------------------------------------------------
# Cluster-specific roots. These are the only values a port needs to change.
#
# DATA_ROOT points at the lilac transfer copy, which is the live one. Note
# /data1/morrisq/fuc/SPARC_data also exists but is a stale partial copy whose
# MSK_CAIA bundle is missing DMP_ID_to_cutoff_day_dict and
# DMP_ID_to_fold_num_dict; several scripts used to default there and failed.
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    'SPARC_DATA_PATH',
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_data')

MODEL_ROOT = os.environ.get(
    'SPARC_MODEL_PATH',
    '/data1/morrisq/fuc/SPARC_models')

RESULTS_ROOT = os.environ.get(
    'SPARC_RESULTS_PATH',
    '/data1/morrisq/fuc/lilac_transfer_091126/SPARC_fixed_models')

# Checkpoints for the 060726 SPARC / SPARC-P runs live in a sibling of
# RESULTS_ROOT. eval_1 derives the benchmark root by stripping '_060726', so
# the two must stay a matched pair.
SPARC_MODEL_ROOT = os.environ.get(
    'SPARC_060726_MODEL_PATH', RESULTS_ROOT + '_060726')

# ---------------------------------------------------------------------------
# Derived locations. Each is independently overridable, because some of them
# were already honoured as environment variables before this module existed
# and scripts in the wild may still set them.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Roots that no longer exist on this filesystem. They are kept configurable so
# the code paths that reference them stay portable, but nothing here resolves
# today -- see code_cleaning_progress.md.
# ---------------------------------------------------------------------------

# Read by eval_1 only for MIMICIV_hosp + XGBoost + `-z False`. `zeroshot`
# defaults to True and no launcher passes -z, so this branch never executes;
# it would fail immediately if it did.
SL_TESTSL_ROOT = os.environ.get(
    'SPARC_SL_TESTSL_ROOT', '/data/morrisq/fuc/SPARC_fixed_models_testsl')

# Inputs to the one-off fold-generation script data/data_split.py, which is
# imported by nothing and runs its work at module import.
FROZEN_DATA_DIR = os.environ.get(
    'SPARC_FROZEN_DATA_DIR', '/data/morrisq/fuc/frozen_step6_dePHI-ed_data_101623')
CTFM_DIR = os.environ.get('SPARC_CTFM_DIR', '/data/morrisq/fuc/CTFoundationModel')


def benchmark_model_dir(model_option):
    """models/benchmarks/<model_option>/ -- the per-architecture class modules."""
    return os.path.join(BENCHMARKS_DIR, model_option)


def add_benchmark_to_path(model_option):
    """Put a benchmark's module directory on sys.path, cwd-independently.

    Replaces `sys.path.append(f'../../models/benchmarks/{model_option}/')`,
    which only worked when the script was run from its own directory.
    """
    import sys
    p = benchmark_model_dir(model_option)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def describe():
    """Print the resolved configuration -- useful when porting."""
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
