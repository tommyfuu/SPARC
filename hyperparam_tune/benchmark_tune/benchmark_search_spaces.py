"""Hyperparameter grids for the benchmark models.
"""
import math


SEARCH_SPACES = {
    'lasttp_MLP': {
        "lr": [1e-3, 1e-6],
        "weight_decay": [1e-5],
        "num_layers": [1, 4],
        "max_time_len": [20],
    },
    'duett': {
        "num_encoder_layers": [4],           # -> 2 layers per axis
        "lr": [1e-3, 3e-4, 1e-4],
        "dropout": [0.1],
        "weight_decay": [1e-6],
        "max_time_len": [20],
    },
    'raindrop': {
        "lr": [1e-3, 1e-6],
        "weight_decay": [1e-5],
        "nlayers": [4, 8],
        "max_time_len": [30],
    },
    'rnn': {
        "lr": [1e-3, 1e-6],
        "weight_decay": [1e-5],
        "num_encoder_layers": [1, 2],
        "max_time_len": [20],
    },
}
SEARCH_SPACES['rnn_deephit'] = SEARCH_SPACES['rnn']

# What the data loader is told to emit for each model.
ADVANCED_MODEL = {
    'lasttp_MLP': 'default',
    'duett': 'duett',
    'raindrop': 'raindrop',
    'rnn': 'default',
    'rnn_deephit': 'rnn_deephit',
}

# per-model run-directory template; must match the names already on disk
_DIRNAME = {
    'lasttp_MLP':  '{m}_{o}_hyperparam_tune_nl_{num_layers}_lr_{lr}_wd_{weight_decay}_max_time_len_{max_time_len}',
    'duett':       '{m}_{o}_hyperparam_tune_nl_{num_encoder_layers}_lr_{lr}_max_time_len_{max_time_len}',
    'raindrop':    '{m}_{o}_hyperparam_tune_nl_{nlayers}_lr_{lr}_max_len_{max_time_len}',
    'rnn':         '{m}_{o}_hyperparam_tune_nl_{num_encoder_layers}_lr_{lr}_max_len_{max_time_len}',
}
_DIRNAME['rnn_deephit'] = _DIRNAME['rnn']


def n_combinations(model_option):
    return math.prod(len(v) for v in SEARCH_SPACES[model_option].values())


def config_at(model_option, hyper_tune_iter):
    """Config at a flat grid index, matching the tuner's -i."""
    space = SEARCH_SPACES[model_option]
    combination, div = {}, 1
    for key, values in reversed(list(space.items())):
        combination[key] = values[(hyper_tune_iter // div) % len(values)]
        div *= len(values)
    return {k: combination[k] for k in space}


def iter_configs(model_option):
    for i in range(n_combinations(model_option)):
        yield i, config_at(model_option, i)


def tune_run_dirname(model_option, option, config):
    return _DIRNAME[model_option].format(m=model_option, o=option, **config)
