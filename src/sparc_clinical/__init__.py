"""SPARC: longitudinal models that predict adverse events of cancer treatment.

sparc_clinical is a thin layer over the research code in this repository. The model
(models/), data pipeline (data/) and training script (model_train/) are used as they
are; this package exposes the input formats (data/bundle_io.py), the `sparc` command and
YAML configs.
"""

import importlib

import sparc_clinical._repo  # noqa: F401  (puts data/ and models/ on sys.path)

__version__ = "1.0.0"

_EXPORTS = {
    'read_bundle': 'bundle_io',
    'write_bundle': 'bundle_io',
    'tables_to_bundle': 'bundle_io',
    'bundle_to_tables': 'bundle_io',
    'validate_bundle': 'bundle_io',
    'load_dataset': 'dataset_loading',
    'SPARC': 'SPARC_inf_risk_modular',
    'make_synthetic_bundle': 'sparc_clinical.synthetic',
    'train': 'sparc_clinical.cli',
    'evaluate': 'sparc_clinical.cli',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    # Imported on first use, so reading data does not pull in torch.
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'sparc_clinical' has no attribute {name!r}")
