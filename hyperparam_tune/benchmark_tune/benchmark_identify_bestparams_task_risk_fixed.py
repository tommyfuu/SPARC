"""Pick the best hyperparameters per task (min val_loss) from a benchmark tuning sweep;
writes {BENCHMARK_CONFIG_DIR}/{model}_best_hyperparams.json.

Usage: python benchmark_identify_bestparams_task_risk_fixed.py -mo duett [-n]   # -n: dry run
"""
import argparse
import json
import os
import sys

import pandas as pd

# Put the repo root on sys.path, from any working directory.
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not all(
        os.path.isdir(os.path.join(_R, _d)) for _d in ('data', 'models')):
    _R = os.path.dirname(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sparc_paths
from benchmark_search_spaces import iter_configs, n_combinations, tune_run_dirname

ALL_OPTIONS = ['binary', 'binary_icu_pre', 'febrile_neutropenia',
               'Liver_toxicity', 'AKI', 'neutropenia', 'anemia']

parser = argparse.ArgumentParser(
    description='Pick the best hyperparameters per task from a tuning sweep.')
parser.add_argument('-mo', '--model', required=True,
                    choices=['lasttp_MLP', 'duett', 'raindrop', 'rnn', 'rnn_deephit'],
                    help='which benchmark to collect')
parser.add_argument('-m', '--modules', default='parsimonious')
parser.add_argument('-ds', '--dataset', default='MSK',
                    help='dataset the sweep was run on (default: %(default)s)')
parser.add_argument('-o', '--options', default=','.join(ALL_OPTIONS),
                    help='comma-separated tasks to collect (default: all seven)')
parser.add_argument('-tp', '--tune_path', default=None,
                    help='root holding {modules}_{dataset}/benchmark_tunes '
                         '(default: sparc_paths.RESULTS_ROOT)')
parser.add_argument('-od', '--out_dir', default=None,
                    help='where to write the json (default: sparc_paths.BENCHMARK_CONFIG_DIR)')
parser.add_argument('-n', '--dry_run', action='store_true',
                    help='report what would be chosen without writing the json')
parser.add_argument('--allow_partial', action='store_true',
                    help='write the json even when some grid points have no '
                         'val_loss. Off by default, so a half-finished sweep '
                         'cannot quietly narrow an existing json.')
args = parser.parse_args()

model_option = args.model
options_l = [o.strip() for o in args.options.split(',') if o.strip()]
tune_root = args.tune_path or sparc_paths.RESULTS_ROOT
out_dir = args.out_dir or sparc_paths.BENCHMARK_CONFIG_DIR
base = os.path.join(tune_root, f'{args.modules}_{args.dataset}', 'benchmark_tunes')

print(f'Collecting {model_option}: {n_combinations(model_option)} grid points '
      f'x {len(options_l)} tasks')
print(f'  sweep root : {base}')
print(f'  output dir : {out_dir}')


def min_val_loss(run_dir):
    """Lowest val_loss across a run's Lightning versions, or None if it never validated."""
    logs = os.path.join(run_dir, 'logs_csv', 'lightning_logs_csv')
    if not os.path.isdir(logs):
        return None
    best = None
    for version in sorted(os.listdir(logs)):
        metrics = os.path.join(logs, version, 'metrics.csv')
        if not os.path.exists(metrics):
            continue
        df = pd.read_csv(metrics)
        if 'val_loss' not in df.columns:
            continue
        v = df['val_loss'].min()
        if pd.isna(v):
            continue
        best = v if best is None else min(best, v)
    return best


best_hyperparams = {}
incomplete = []
for option in options_l:
    scored = []
    for _, config in iter_configs(model_option):
        run_dir = os.path.join(base, model_option, option,
                               tune_run_dirname(model_option, option, config))
        v = min_val_loss(run_dir)
        if v is None:
            incomplete.append((option, os.path.basename(run_dir)))
        else:
            scored.append((v, config))

    print(f'\n{option}: {len(scored)}/{n_combinations(model_option)} grid points with a val_loss')
    for v, config in sorted(scored, key=lambda t: t[0]):
        print(f'    {v:.6f}  {config}')

    if not scored:
        print(f'    NO RUNS FOUND -- {option} cannot be assigned a config')
        continue
    val_loss, config = min(scored, key=lambda t: t[0])
    best_hyperparams[option] = {'val_loss': float(val_loss), 'config': config}

if incomplete:
    print(f'\n{len(incomplete)} grid point(s) had no val_loss:')
    for option, name in incomplete:
        print(f'    {option:22} {name}')
    print('  Either those runs have not finished, or the grid in '
          'benchmark_search_spaces.py no longer matches what was tuned.')
    if not (args.allow_partial or args.dry_run):
        sys.exit('\nERROR: refusing to write a json from an incomplete sweep, '
                 'which would change which config eval_1 and the training '
                 'scripts rebuild. Finish the sweep, or pass --allow_partial.')

missing = [o for o in options_l if o not in best_hyperparams]
if missing:
    sys.exit(f'\nERROR: no runs at all for {missing}. Refusing to write a json '
             f'that would be missing tasks the training scripts look up.')

out_json_path = os.path.join(out_dir, f'{model_option}_best_hyperparams.json')
if args.dry_run:
    print(f'\n[dry run] would write {out_json_path}')
    print(json.dumps(best_hyperparams, indent=2))
else:
    os.makedirs(out_dir, exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(best_hyperparams, f)
    print(f'\nWrote {out_json_path}')
