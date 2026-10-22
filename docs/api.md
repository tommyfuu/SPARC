# Python API

Everything the `sparc` command does is available from Python:

```python
import pandas as pd
import sparc_clinical as sparc

bundle = sparc.tables_to_bundle(
    patients=pd.read_csv("export/patients.csv"),
    labs=pd.read_csv("export/labs.csv"),
    medications=pd.read_csv("export/medications.csv"),
)
assert sparc.validate_bundle(bundle) == []
sparc.write_bundle(bundle, "data/MYSITE_CAIA", "pickle")

sparc.train("examples/finetune.yaml", valfold=1)       # the same as `sparc train`
sparc.evaluate("examples/finetune.yaml", valfold=1)    # the same as `sparc eval`
```

`sparc_clinical` also re-exports the research code it runs: `load_dataset`
(`data/dataset_loading.py`) and the `SPARC` model (`models/SPARC_inf_risk_modular.py`).

## Reading and writing data

::: bundle_io.read_bundle

::: bundle_io.write_bundle

::: bundle_io.tables_to_bundle

::: bundle_io.bundle_to_tables

::: bundle_io.validate_bundle

::: sparc_clinical.synthetic.make_synthetic_bundle

## Training

::: sparc_clinical.cli.train

::: sparc_clinical.cli.evaluate
