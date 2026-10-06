import os

import pytest

from sparc_clinical import make_synthetic_bundle, write_bundle


@pytest.fixture(scope='session', autouse=True)
def _no_slurm():
    # Lightning reads SLURM_* and refuses to start inside a multi-task allocation.
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith('SLURM_'):
                mp.delenv(key)
        yield


@pytest.fixture(scope='session')
def bundle():
    return make_synthetic_bundle(n_patients=200, seed=0)


@pytest.fixture(scope='session')
def data_root(tmp_path_factory, bundle):
    """A data root holding the synthetic cohort as SYN_CAIA (JSON)."""
    root = tmp_path_factory.mktemp('data')
    write_bundle(bundle, str(root / 'SYN_CAIA'))
    return root
