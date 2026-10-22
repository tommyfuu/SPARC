"""The `sparc` command.

`train` and `eval` run model_train/SPARC_train_inf_risk_fixed.py, so they take exactly its
flags (see `sparc train --help`). A --config YAML file gives the same flags by their long
names; flags on the command line override it.
"""

import argparse
import glob
import os
import runpy
import sys
import warnings
from collections import Counter

from sparc_clinical._repo import TRAIN_SCRIPT


def config_to_args(config):
    """{'dataset': 'X', 'add_num_infusion': True, 'med_of_choice': ['A', 'B']} ->
    ['--dataset', 'X', '--add_num_infusion', 'True', '--med_of_choice', 'A,B']"""
    args = []
    for key, value in (config or {}).items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ','.join(str(v) for v in value)
        args += [f'--{key}', str(value)]
    return args


def load_config(path):
    import yaml
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping of flag names to values")
    return config


def run_training(args):
    """Run the training script with these command-line arguments."""
    saved = sys.argv
    sys.argv = [TRAIN_SCRIPT, *args]
    try:
        runpy.run_path(TRAIN_SCRIPT, run_name='__main__')
    finally:
        sys.argv = saved


def train(config=None, **flags):
    """Python equivalent of `sparc train`: config is a YAML path or dict, flags override it."""
    if isinstance(config, (str, os.PathLike)):
        config = load_config(config)
    run_training(config_to_args(config) + config_to_args(flags))


def evaluate(config=None, **flags):
    """Python equivalent of `sparc eval`."""
    train(config, **{**flags, 'mode': 'test'})


def _split_config(argv):
    """Pull --config FILE out of the pass-through arguments."""
    argv = list(argv)
    for i, a in enumerate(argv):
        if a == '--config' or a.startswith('--config='):
            path = a[len('--config='):] if '=' in a else (argv[i + 1] if i + 1 < len(argv) else '')
            if not path or path.startswith('-'):
                raise SystemExit("sparc: --config needs a YAML file")
            return path, argv[:i] + argv[i + (1 if '=' in a else 2):]
    return None, argv


def _remove_normalization_caches(directory):
    # load_dataset caches normalized data next to the bundle and never checks if it is stale
    for path in glob.glob(os.path.join(directory, 'normaliz*_dict*.pickle')):
        os.remove(path)
        print(f"removed stale cache {path}")


def _write(bundle, dst, fmt):
    from bundle_io import write_bundle
    out = write_bundle(bundle, dst, fmt)
    _remove_normalization_caches(out if os.path.isdir(out) else os.path.dirname(out))
    print(f"wrote {out}")


def _summary(bundle):
    from bundle_io import FOLD, ICU, INPATIENT, LABS, MEDS
    folds = Counter(bundle[FOLD].values())
    meds = {m[0] for days in bundle[MEDS].values() for ms in days.values() for m in ms}
    return '\n'.join([
        f"patients with labs  {len(bundle[LABS])}",
        f"lab days            {sum(len(d) for d in bundle[LABS].values())}",
        f"patients per fold   {dict(sorted(folds.items()))}  (fold 10 = test)",
        f"inpatient / ICU     {len(bundle[INPATIENT])} / {len(bundle[ICU])} patients",
        f"medications         {len(bundle[MEDS])} patients, {len(meds)} drugs: {', '.join(sorted(meds)[:10])}"
        + (' ...' if len(meds) > 10 else ''),
    ])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog='sparc', description=__doc__.split('\n', 2)[2])
    sub = parser.add_subparsers(dest='command', required=True)
    # listed for --help only: their arguments go straight to the training script (below)
    sub.add_parser('train', add_help=False, help='pretrain or finetune one fold')
    sub.add_parser('eval', add_help=False, help='evaluate a trained fold and write test predictions')
    p = sub.add_parser('convert', help='convert a bundle between formats')
    p.add_argument('src', help='bundle file or directory')
    p.add_argument('dst', help='output file or directory')
    p.add_argument('--format', choices=['json', 'pickle', 'csv', 'parquet'],
                   help='output format (default: pickle for a .pkl DST, else json)')
    p = sub.add_parser('validate', help='check a bundle and print a summary')
    p.add_argument('src', help='bundle file or directory')
    p = sub.add_parser('synthetic', help='write a synthetic cohort')
    p.add_argument('dst', help='output directory, e.g. data/DEMO_CAIA')
    p.add_argument('--patients', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--format', choices=['json', 'pickle', 'csv', 'parquet'],
                   help='output format (default: pickle for a .pkl DST, else json)')

    if argv[:1] in (['train'], ['eval']):
        config, rest = _split_config(argv[1:])
        try:
            args = config_to_args(load_config(config) if config else {}) + rest
        except (OSError, ValueError) as e:
            print(f"ERROR: {e}")
            return 1
        if argv[0] == 'eval':
            args += ['--mode', 'test']
        run_training(args)
        return 0

    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1


def _run(args):
    from bundle_io import read_bundle, validate_bundle
    if args.command == 'convert':
        _write(read_bundle(args.src), args.dst, args.format)
    elif args.command == 'validate':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            bundle = read_bundle(args.src)
            problems = validate_bundle(bundle)
        for w in caught:
            print(f"WARNING: {w.message}")
        for problem in problems:
            print(f"ERROR: {problem}")
        if problems:
            return 1
        print(_summary(bundle))
    elif args.command == 'synthetic':
        from sparc_clinical.synthetic import make_synthetic_bundle
        _write(make_synthetic_bundle(args.patients, args.seed), args.dst, args.format)
    return 0


if __name__ == '__main__':
    sys.exit(main())
