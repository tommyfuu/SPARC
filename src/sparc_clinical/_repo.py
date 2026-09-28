"""Find the SPARC repository this package wraps and put its research code on sys.path."""

import os
import sys


def _find_root():
    here = os.path.dirname(os.path.abspath(__file__))
    start = os.environ.get('SPARC_REPO', here)
    root = start
    while root != os.path.dirname(root):
        if all(os.path.isdir(os.path.join(root, d)) for d in ('data', 'models', 'model_train')):
            return root
        root = os.path.dirname(root)
    raise ImportError(
        f"sparc_clinical could not find the SPARC repository above {start}. It runs the code in "
        f"data/, models/ and model_train/, so install it from a clone with `pip install -e .`, "
        f"or set SPARC_REPO to the clone.")


REPO_ROOT = _find_root()
TRAIN_SCRIPT = os.path.join(REPO_ROOT, 'model_train', 'SPARC_train_inf_risk_fixed.py')

for _p in (os.path.join(REPO_ROOT, 'models'), os.path.join(REPO_ROOT, 'data'), REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
