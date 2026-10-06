# SPARC

SPARC is a transformer that reads a cancer patient's routine labs and vitals over time and
predicts adverse events of treatment (neutropenia, febrile neutropenia, anemia, acute kidney
injury, liver toxicity) and mortality. It is pretrained on the MSK cohort and can be
finetuned on data from another institution.

This site covers the `sparc` command and the `sparc_clinical` Python package, which wrap
the research code in the [repository](https://github.com/tommyfuu/SPARC):

- [Getting started](getting-started.md): install and run the whole pipeline on synthetic data.
- [Input data](data.md): the formats SPARC reads (JSON, pickle, CSV, Parquet) and their columns.
- [Training and inference](training.md): pretrain, finetune, evaluate, configs and SLURM.
- [Python API](api.md): the same steps from Python.

The evaluation and figure scripts behind the paper (`results_vis/`, `cohort_summaries/`,
`interpretation/`) are described in the repository README.
